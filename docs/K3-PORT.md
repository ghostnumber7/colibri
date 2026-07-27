# Kimi-K3 engine port — extracted math spec and work order

Status: conversion of `/mnt/d/kimi_k3_i4` in progress (streaming `coli convert --repo
moonshotai/Kimi-K3`); converter support landed in 78574be. This document is the
ground-truth spec for the C engine port, extracted 2026-07-27 from the K3 repo's
`modeling_kimi_linear.py` / `configuration_kimi_k3.py` (repo sha 9f62e4e) and
fla-org/flash-linear-attention `main` (`fla/ops/kda/*`, `fla/modules/fused_norm_gate.py`),
cross-checked against the real checkpoint's shard headers. Where the released modeling
code and the checkpoint disagree, the CHECKPOINT is authoritative (see A_log below).

## Architecture at a glance

- model_type `kimi_linear` (nested under a `kimi_k3` multimodal container; the
  converted container config is the flattened text_config).
- hidden 7168, vocab 163840, 93 layers, untied lm_head, final RMSNorm, rms_eps 1e-5.
- Layer 0 (0-indexed) is dense mlp (`first_k_dense_replace: 1`, intermediate 33792);
  layers 1..92 are MoE.
- **Layer-type lists in `linear_attn_config` are 1-INDEXED**: `is_kda_layer(i)` ⇔
  `(i+1) ∈ kda_layers`. 0-indexed full-MLA layers: {3, 7, 11, ..., 87, 91, 92}
  (from `full_attn_layers` [4,8,...,92,93] — note 92 AND 93 ⇒ the last TWO
  0-indexed full layers are 91 and 92). The other 69 layers are KDA.
- No RoPE anywhere: MLA layers are NoPE (`mla_use_nope: true`, `rotary_emb = None`
  in modeling; the rope-dim halves of q/k are used UNROTATED); KDA is position-free
  by construction. No rope/yarn tables needed. max_position_embeddings 1048576.
- No MTP (`num_nextn_predict_layers: 0`).

## situ activation (all MLPs: dense, shared experts, routed experts)

`hidden_act: "situ"`, beta = `activation_situ_beta` = 4.0,
linear_beta = `activation_situ_linear_beta` = 25.0. For gate g and up u (f32):

```
situ(g)  = beta * tanh(g / beta) * sigmoid(g)          # smooth-clipped swish
u'       = linear_beta * tanh(u / linear_beta)         # soft-clip of the up path
out      = situ(g) * u'
```

Replaces silu(g)*u at every gate/up/down MLP site.

## MoE (layers 1..92)

- Router: `gate.weight` [896, 7168] f32 matmul → sigmoid scores; choice scores =
  scores + `e_score_correction_bias` [896]; top-16 by choice score (num_expert_group
  = topk_group = 1 ⇒ no group masking); weights = RAW sigmoid scores gathered at the
  chosen experts, renormalized to sum 1 (`moe_renormalize`), × routed_scaling_factor
  1.0. Same noaux_tc scheme as DeepSeek-V3/K2.6.
- **Latent MoE**: routed experts run in a 3584-dim latent space:
  ```
  z   = routed_expert_down_proj(h)          # [7168] -> [3584], bf16 resident (int8 in container)
  y   = Σ_e w_e * expert_e(z)               # experts: w1 gate [3072,3584], w3 up [3072,3584],
                                            #          w2 down [3584,3072], situ act, mxfp4
  y   = routed_expert_norm(y)               # RMSNorm [3584] (latent_moe_use_norm)
  out = routed_expert_up_proj(y) + shared_experts(h)   # [3584] -> [7168]
  ```
- Shared experts: 2 fused (KimiMLP on full hidden, intermediate 2×3072 = 6144,
  situ act), bf16 residents (int8 in container).
- Expert weights are mxfp4 in the container (see decode contract below).

## mxfp4 expert decode contract (container fmt: u8 .qs)

`<base>.weight` = u8 [O*ceil(I/2)] e2m1 nibble bytes, LOW nibble = even element;
`<base>.weight.qs` = u8 [O*ceil(I/32)] e8m0. Decode:

```
w[o,i] = E2M1[nibble(o,i)] * 2^(qs[o, i/32] - 127)
E2M1   = {0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6}
```

Verified bit-exact vs compressed-tensors 0.17.1 on a real shard tensor (converter
commit 78574be; pinned by test_transcode_mxfp4_is_raw_passthrough).

## MLA layers (24 of 93)

DeepSeek-V3 MLA shapes (q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64,
v_head 128, 96 heads) with two twists:

1. **NoPE**: no rotary applied; q_rot/k_rot halves participate in the dot product
   raw. k_rot (64 dims from kv_a_proj_with_mqa) is shared across heads (MQA-style),
   exactly like the rope half of standard MLA but without rotation.
2. **Output gate** (`mla_use_output_gate`): `attn_out ⊙ sigmoid(g_proj(h))` before
   o_proj, g_proj [12288, 7168].

Scaling = (qk_nope + qk_rope)^-0.5 = 192^-0.5. No softmax-scale mscale correction
(no yarn). KV cache: same MLA cache structure as K2.6/GLM for these 24 layers only.

## KDA layers (69 of 93) — Kimi Delta Attention

Per layer: q_proj/k_proj/v_proj [12288, 7168], depthwise causal short conv
(kernel 4, SILU activation, no bias) per channel on each of q/k/v, f_a_proj
[128, 7168] + f_b_proj [12288, 128] (low-rank per-channel gate), b_proj [96, 7168]
(per-head beta), A_log f32 [128], dt_bias f32 [12288], o_norm [128], g_proj
[12288, 7168] (full-rank output gate, `use_full_rank_gate: true`), o_proj
[7168, 12288]. 96 heads × head_dim 128 (H = HV, no GVA).

Decode step (S=1), per head h (all f32; state S_h [128k × 128v] per head):

```
q̃,k̃,ṽ = short_conv_silu(q_proj h, k_proj h, v_proj h)   # per-channel: y_t = silu(Σ_{j=0..3} w[j]·x_{t-3+j})
q,k    = l2norm_per_head(q̃), l2norm_per_head(k̃)          # use_qk_l2norm_in_kernel
q      = q * K^-0.5                                       # scale = head_dim^-0.5 (fla default)
g_raw  = f_b_proj(f_a_proj(h)) + dt_bias                  # [96, 128] per-channel
g      = lower_bound * sigmoid(exp(A_log[d]) * g_raw)     # lower_bound = -5.0 (safe gate);
                                                          # A_log is PER-DIM, see below
beta   = sigmoid(b_proj(h))                               # [96] per-head
S_h    = S_h ⊙ exp(g_h)[:,None]                           # per-k-dim decay (col-broadcast over v)
S_h   += (beta_h * k_h) ⊗ (v_h - k_hᵀ S_h)                # delta rule
o_h    = q_hᵀ S_h                                         # [128]
o_h    = rmsnorm(o_h) ⊙ o_norm.weight ⊙ sigmoid(g_out_h)  # g_out = g_proj(h), per-channel
out    = o_proj(concat_heads(o))
```

(fla reference: `fla/ops/kda/naive.py::naive_recurrent_kda` + `gate.py::
naive_kda_lowerbound_gate` + `fused_norm_gate.py` sigmoid branch.)

**A_log is per-dim [head_dim=128], NOT per-head [96]**: the checkpoint ships [128]
with 96 heads; the released modeling file allocates [num_heads] (stale — it cannot
load the checkpoint), and predecessor Kimi-Linear-48B shipped [1,1,32,1] per-head.
Shape arithmetic admits no other reading for K3 (f_a_proj [128,7168] and o_norm
[128] fix head_dim=128; b_proj [96,7168] fixes num_heads=96). The engine should
branch on the tensor's length at load: len==head_dim ⇒ per-dim exp(A_log[d]),
len==num_heads ⇒ per-head exp(A_log[h]) — robust to a future fixed checkpoint.

State per KDA layer: recurrent S 96×128×128 f32 = 6.3 MB + conv cache 3×12288×3
f32 ≈ 0.44 MB ⇒ ~434 MB total for all 69 layers, CONSTANT in context length.
Prefill can run the same recurrence token-by-token (state-update cost ~6.3M MAC/
layer/token, ~1% of the layer's projection matmuls).

## Per-layer residual mixers (attn_res_block_size = 12)

DenseFormer-style block-residual mixing, `KimiDecoderLayer._forward_attn_residual`:

- Model-level `block_residual` list B starts empty; every layer with
  `layer_idx % 12 == 0` (0-indexed: 0, 12, 24, 36, 48, 60, 72, 84) SNAPSHOTS the
  incoming prefix_sum into B (before attention) and suppresses its own incoming
  residual (prefix_sum = attn output alone at those layers).
- Mixing operator `_apply_attn_res(x, B, proj, norm)`:
  ```
  V      = [B_0 ... B_{n-1}, x]                    # n snapshots + current
  k_i    = V_i / rms(V_i)                          # rmsnorm WITHOUT weight
  s_i    = k_i · (norm.weight ⊙ proj.weight)       # proj: [1, 7168] res_proj
  p      = softmax(s)                              # over the n+1 candidates
  out    = Σ p_i · V_i                             # f32 mix of UN-normalized V
  ```
- Applied (a) to the incoming hidden before input_layernorm using
  self_attention_res_norm/proj (only when B non-empty), (b) to prefix_sum after
  attention using mlp_res_norm/proj (input to post_attention_layernorm + MoE),
  and (c) once at model end using output_attn_res_norm/proj, before the final norm.
- Layer returns prefix_sum = (mixed-attn path) + moe_out accumulated as in the
  reference; port the exact control flow from `_forward_attn_residual`.

## Sampling / tokenizer

- tiktoken.model + tokenizer_config.json → tokenizer.json via gen_kimi_tokenizer
  (verified: 163600 vocab + 16 added tokens). bos 163584 `[BOS]`, eos 163586
  `[EOS]`, pad 163839. Chat template: in tokenizer_config.json (K3 naming,
  `encoding_k3.py` in repo); needs a kimi_k3 entry in coli/openai_server dispatch
  keyed on model_type `kimi_linear`.
- generation_config.json: `{"temperature": 1.0, "top_p": 1.0}` (trivial).

## Engine work order

1. fmt=7 mxfp4 kernel in quant.h + loader detection (u8 .qs ⇒ mxfp4 g32) +
   bench vs fmt=4 (same 4.25 bpw; LUT decode is ALU-cheap, path stays
   bandwidth-bound).
2. Arch plumbing: model_type kimi_linear, 1-indexed kda/full layer lists,
   first_k_dense_replace=1, expert naming block_sparse_moe.experts.N.w1/w2/w3
   (w1=gate, w3=up, w2=down), latent down/norm/up residents, situ activation,
   sigmoid router + renorm (mostly shared with K2.6's noaux_tc path).
3. KDA cell (recurrent form only, used for both prefill and decode) + per-layer
   state buffers; NoPE MLA variant + output gates.
4. Residual mixers (block snapshots + 3 mixing sites).
5. CLI/server: chat template, stop tokens, detection, doctor.
6. Oracle: tiny random-weight kimi_linear model through HF modeling (patched
   A_log shape) vs engine, TF logit comparison; then SCORE loss vs full config
   on the real container.
