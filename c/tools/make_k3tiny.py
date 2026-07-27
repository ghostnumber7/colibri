"""Builds a tiny synthetic Kimi-K3 (kimi_linear) checkpoint as an ORACLE for the C
engine's K3 forward pass: real architecture (KDA linear attention with per-dim
A_log, NoPE MLA with output gate, latent MoE with mxfp4 routed experts, situ
activation, DenseFormer-style residual mixers), toy dimensions, random weights.

Follows the make_k2tiny.py idiom with its two hard-won corrections:
1. The reference is computed AFTER round-tripping every tensor the converter
   quantises through the converter's OWN math (quant_int8 for residents at the
   mxfp4-pack default --ebits 8, embed/lm_head at --io-bits 8) -- otherwise the
   teacher-forced gate misses positions for reasons unrelated to the engine.
2. The reference forward is c/tools/k3_ref.py -- the SAME implementation that is
   cross-checked against Moonshot's modeling code by tests/test_k3_ref_crosscheck.py
   -- so the oracle chain is: Moonshot modeling == k3_ref == this reference ==
   (to be proven) the C engine.

The routed experts are generated DIRECTLY as mxfp4 bytes (random e2m1 nibbles +
e8m0 group-32 scales) -- the exact on-disk source format -- and the reference uses
their dequantisation via the pinned decode contract (w = E2M1[nib] * 2^(qs-127)),
so the converter's raw passthrough plus the engine's matmul_mxfp4 must reproduce
the reference bit-for-close.

Usage:
  .venv/bin/python c/tools/make_k3tiny.py --outdir c/k3_tiny
  # writes c/k3_tiny/{config.json,model.safetensors} and c/ref_k3.json
  .venv/bin/python c/tools/convert_fp8_to_int4.py --indir c/k3_tiny --outdir c/k3_tiny_i4
  SNAP=c/k3_tiny_i4 REF=c/ref_k3.json TF=1 ./colibri     # expect nfull/nfull
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))
from convert_fp8_to_int4 import classify, quant_int8, _E2M1          # noqa: E402
from k3_ref import K3Ref                                              # noqa: E402

from safetensors.torch import save_file                               # noqa: E402

RESIDENT_BITS = 8          # mxfp4-pack sources default --ebits 8 (lossless-ish residents)
IO_BITS = 8                # cmd_convert passes --io-bits 8
RESIDENT_SKIP_KINDS = {"f32", "skip", "consumed", "x"}

# Toy dims -- the SAME shape set the k3_ref cross-check validated against
# Moonshot's modeling code (tests/test_k3_ref_crosscheck.py TINY).
TEXT_CONFIG = dict(
    model_type="kimi_linear", architectures=["KimiLinearForCausalLM"],
    vocab_size=96, hidden_size=48, intermediate_size=64,
    num_hidden_layers=7, num_attention_heads=2, num_key_value_heads=2,
    hidden_act="situ", activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
    rms_norm_eps=1e-5,
    q_lora_rank=24, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8,
    v_head_dim=8, mla_use_nope=True, mla_use_output_gate=True,
    # latent + moe_inter are multiples of 32: mxfp4 is group-32 along the input
    # dim and the converter refuses ragged groups (transcode_mxfp4 guard).
    moe_intermediate_size=32, num_experts=4, num_experts_per_token=2,
    num_shared_experts=1, routed_scaling_factor=1.0, moe_renormalize=True,
    moe_router_activation_func="sigmoid", num_expert_group=1, topk_group=1,
    first_k_dense_replace=1, moe_layer_freq=1,
    routed_expert_hidden_size=32, latent_moe_use_norm=True,
    attn_res_block_size=3,
    linear_attn_config=dict(kda_layers=[1, 2, 3, 5, 6], full_attn_layers=[4, 7],
                            num_heads=2, head_dim=16, short_conv_kernel_size=4,
                            use_full_rank_gate=True, gate_lower_bound=-5.0),
    num_nextn_predict_layers=0,
    bos_token_id=1, eos_token_id=2, pad_token_id=0,
    tie_word_embeddings=False,
    quantization_config=dict(format="mxfp4-pack-quantized", config_groups={}),
)
NESTED_CONFIG = dict(
    model_type="kimi_k3", architectures=["KimiK3ForConditionalGeneration"],
    text_config=TEXT_CONFIG,
    vision_config=dict(patch_size=14),
)

H = TEXT_CONFIG["hidden_size"]
LA = TEXT_CONFIG["linear_attn_config"]
KH, KD = LA["num_heads"], LA["head_dim"]
KP = KH * KD
CK = LA["short_conv_kernel_size"]
NL = TEXT_CONFIG["num_hidden_layers"]
LAT = TEXT_CONFIG["routed_expert_hidden_size"]
MI = TEXT_CONFIG["moe_intermediate_size"]
NE = TEXT_CONFIG["num_experts"]
NH = TEXT_CONFIG["num_attention_heads"]
QL, KVL = TEXT_CONFIG["q_lora_rank"], TEXT_CONFIG["kv_lora_rank"]
DN, DR, DV = TEXT_CONFIG["qk_nope_head_dim"], TEXT_CONFIG["qk_rope_head_dim"], TEXT_CONFIG["v_head_dim"]
SI = MI * TEXT_CONFIG["num_shared_experts"]
VOCAB = TEXT_CONFIG["vocab_size"]
KDA_SET = {i - 1 for i in LA["kda_layers"]}


def bf16_rt(w):
    """f32 -> bf16 -> f32 round trip: the source stores bf16, the converter reads it."""
    return torch.from_numpy(w).to(torch.bfloat16).to(torch.float32).numpy()


def mxfp4_dequant(packed, scale, gs=32):
    O, Ih = packed.shape
    I = Ih * 2
    nib = np.empty((O, I), np.int64)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = (packed >> 4) & 0x0F
    w4 = np.array(_E2M1, np.float32)[nib]
    sc = np.exp2(scale.astype(np.float32) - 127.0)
    return w4 * np.repeat(sc, gs, axis=1)[:, :I]


def main():
    global KDA_SET, LAT
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(_TOOLS_DIR.parent / "k3_tiny"))
    ap.add_argument("--ref", default=None)
    ap.add_argument("--seed", type=int, default=0)
    # Ablations for bisecting engine-vs-reference mismatches: each drops ONE K3
    # feature from both the checkpoint and the reference.
    ap.add_argument("--no-res", action="store_true", help="drop the residual mixers")
    ap.add_argument("--no-latent", action="store_true",
                    help="experts bracket hidden directly (and ship bf16, int8-quantized "
                         "by the converter, instead of mxfp4)")
    ap.add_argument("--all-mla", action="store_true", help="no KDA layers (drop linear_attn_config)")
    ap.add_argument("--all-kda", action="store_true", help="every layer KDA")
    ap.add_argument("--latent-int8", action="store_true",
                    help="keep the latent bracket but ship bf16 experts (int8 in the "
                         "container) -- splits a latent-wiring bug from an mxfp4 one")
    a = ap.parse_args()
    if a.no_res:
        TEXT_CONFIG.pop("attn_res_block_size", None)
    if a.no_latent:
        TEXT_CONFIG.pop("routed_expert_hidden_size", None)
        TEXT_CONFIG.pop("latent_moe_use_norm", None)
        LAT = None
    if a.all_mla:
        TEXT_CONFIG.pop("linear_attn_config", None)
        KDA_SET = set()
    elif a.all_kda:
        TEXT_CONFIG["linear_attn_config"] = dict(
            kda_layers=list(range(1, NL + 1)), full_attn_layers=[],
            num_heads=KH, head_dim=KD, short_conv_kernel_size=CK,
            use_full_rank_gate=True, gate_lower_bound=-5.0)
        KDA_SET = set(range(NL))
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    def w2(o, i, std=0.05):
        return (rng.standard_normal((o, i)) * std).astype(np.float32)

    def norm1(n):
        return rng.uniform(0.8, 1.2, n).astype(np.float32)

    src = {}          # tensors as stored in the SOURCE checkpoint (torch, real dtypes)
    ref_w = {}        # f32 reference weights (post round-trip), container names

    def put(name, w, dtype=torch.bfloat16):
        """Store `w` under language_model.<name> at `dtype`; reference sees the
        dtype-round-tripped f32 value (further quantised later by kind)."""
        t = torch.from_numpy(w).to(dtype)
        src["language_model." + name] = t
        ref_w[name] = t.to(torch.float32).numpy()

    has_res = "attn_res_block_size" in TEXT_CONFIG
    put("model.embed_tokens.weight", w2(VOCAB, H, 0.3))
    put("lm_head.weight", w2(VOCAB, H, 0.3))
    put("model.norm.weight", norm1(H))
    if has_res:
        put("model.output_attn_res_norm.weight", norm1(H))
        put("model.output_attn_res_proj.weight", w2(1, H, 0.3))

    expert_bytes = {}   # name-prefix -> (packed u8, scale u8) for the source
    for li in range(NL):
        L = f"model.layers.{li}"
        put(f"{L}.input_layernorm.weight", norm1(H))
        put(f"{L}.post_attention_layernorm.weight", norm1(H))
        if has_res:
            put(f"{L}.self_attention_res_norm.weight", norm1(H))
            put(f"{L}.mlp_res_norm.weight", norm1(H))
            put(f"{L}.self_attention_res_proj.weight", w2(1, H, 0.3))
            put(f"{L}.mlp_res_proj.weight", w2(1, H, 0.3))
        if li in KDA_SET:
            put(f"{L}.self_attn.q_proj.weight", w2(KP, H))
            put(f"{L}.self_attn.k_proj.weight", w2(KP, H))
            put(f"{L}.self_attn.v_proj.weight", w2(KP, H))
            for s in ("q", "k", "v"):
                put(f"{L}.self_attn.{s}_conv1d.weight",
                    (rng.standard_normal((KP, 1, CK)) * 0.3).astype(np.float32), torch.float32)
            put(f"{L}.self_attn.f_a_proj.weight", w2(KD, H))
            put(f"{L}.self_attn.f_b_proj.weight", w2(KP, KD))
            put(f"{L}.self_attn.b_proj.weight", w2(KH, H))
            put(f"{L}.self_attn.A_log",
                np.log(rng.uniform(1, 16, KD)).astype(np.float32), torch.float32)  # PER-DIM
            put(f"{L}.self_attn.dt_bias",
                (rng.standard_normal(KP) * 0.5).astype(np.float32), torch.float32)
            put(f"{L}.self_attn.o_norm.weight", norm1(KD).astype(np.float32), torch.float32)
            put(f"{L}.self_attn.g_proj.weight", w2(KP, H))
            put(f"{L}.self_attn.o_proj.weight", w2(H, KP))
        else:
            put(f"{L}.self_attn.q_a_proj.weight", w2(QL, H))
            put(f"{L}.self_attn.q_a_layernorm.weight", norm1(QL))
            put(f"{L}.self_attn.q_b_proj.weight", w2(NH * (DN + DR), QL))
            put(f"{L}.self_attn.kv_a_proj_with_mqa.weight", w2(KVL + DR, H))
            put(f"{L}.self_attn.kv_a_layernorm.weight", norm1(KVL))
            put(f"{L}.self_attn.kv_b_proj.weight", w2(NH * (DN + DV), KVL))
            put(f"{L}.self_attn.g_proj.weight", w2(NH * DV, H))
            put(f"{L}.self_attn.o_proj.weight", w2(H, NH * DV))
        if li < TEXT_CONFIG["first_k_dense_replace"]:
            put(f"{L}.mlp.gate_proj.weight", w2(TEXT_CONFIG["intermediate_size"], H))
            put(f"{L}.mlp.up_proj.weight", w2(TEXT_CONFIG["intermediate_size"], H))
            put(f"{L}.mlp.down_proj.weight", w2(H, TEXT_CONFIG["intermediate_size"]))
        else:
            B = f"{L}.block_sparse_moe"
            put(f"{B}.gate.weight", w2(NE, H, 0.3))
            put(f"{B}.gate.e_score_correction_bias",
                np.linspace(-0.1, 0.1, NE).astype(np.float32), torch.float32)
            put(f"{B}.shared_experts.gate_proj.weight", w2(SI, H))
            put(f"{B}.shared_experts.up_proj.weight", w2(SI, H))
            put(f"{B}.shared_experts.down_proj.weight", w2(H, SI))
            if LAT:
                put(f"{B}.routed_expert_down_proj.weight", w2(LAT, H))
                put(f"{B}.routed_expert_up_proj.weight", w2(H, LAT))
                put(f"{B}.routed_expert_norm.weight", norm1(LAT))
            De = LAT if LAT else H
            for e in range(NE):
                for mat, (O, I) in (("w1", (MI, De)), ("w3", (MI, De)), ("w2", (De, MI))):
                    base = f"{B}.experts.{e}.{mat}"
                    if LAT and not a.latent_int8:   # real K3: mxfp4 bytes verbatim
                        assert I % 32 == 0, "mxfp4 needs the input dim to be a multiple of 32"
                        packed = rng.integers(0, 256, size=(O, I // 2)).astype(np.uint8)
                        # e8m0 116..122 = 2^-11..2^-5: |w| <= 6*2^-5 = 0.19, matching the
                        # 0.05-std residents. The real container sits in the same band
                        # (observed 112..122). Wildly large expert weights only amplify
                        # benign fp ordering noise into argmax flips on a random model.
                        scale = rng.integers(116, 123, size=(O, I // 32)).astype(np.uint8)
                        src["language_model." + base + ".weight_packed"] = torch.from_numpy(packed)
                        src["language_model." + base + ".weight_scale"] = torch.from_numpy(scale)
                        ref_w[base + ".weight"] = mxfp4_dequant(packed, scale)
                    else:        # --no-latent ablation: bf16 experts, int8 in the container
                        put(base + ".weight", w2(O, I))

    # a vision tensor the converter must DROP
    src["vision_tower.encoder.blocks.0.wqkv.weight"] = torch.randn(8, 8).to(torch.bfloat16)

    # Round-trip every 2-D resident the converter quantises through its OWN math
    # (int8: residents AND io at these settings), keyed by classify() so this can
    # never drift from the real conversion.
    for name in list(ref_w):
        w = ref_w[name]
        if w.ndim != 2 or name.endswith("weight_packed"):
            continue
        kind = classify(name, NL)
        if kind in RESIDENT_SKIP_KINDS:
            continue
        bits = IO_BITS if kind == "io" else RESIDENT_BITS
        q, s = quant_int8(w, bits)
        codes = q.view(np.int8).reshape(w.shape).astype(np.float32)
        ref_w[name] = codes * s[:, None]

    # ---- reference forward (k3_ref) on the round-tripped weights ----
    weights = {k: torch.from_numpy(v.astype(np.float32)) for k, v in ref_w.items()}
    ref = K3Ref(TEXT_CONFIG, weights)
    prompt = [3, 17, 42, 9, 88, 5, 61, 23]
    full = list(prompt)
    n_new = 8
    with torch.no_grad():
        for _ in range(n_new):                       # greedy, no cache (recompute)
            logits = ref.forward(torch.tensor(full))
            full.append(int(logits[-1].argmax()))
        tf_logits = ref.forward(torch.tensor(full))
        tf_pred = [int(t) for t in tf_logits.argmax(-1)]

    save_file(src, str(outdir / "model.safetensors"))
    (outdir / "config.json").write_text(json.dumps(NESTED_CONFIG, indent=2))

    ref_path = Path(a.ref) if a.ref else (
        _TOOLS_DIR.parent / ("ref_k3.json" if outdir.name == "k3_tiny"
                              else f"ref_{outdir.name}.json"))
    ref_path.write_text(json.dumps(
        {"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred}))
    print(f"[k3tiny] wrote {outdir}/model.safetensors ({len(src)} tensors), "
          f"{outdir}/config.json, {ref_path}")
    print(f"[k3tiny] full_ids={full}")
    print(f"[k3tiny] next: .venv/bin/python c/tools/convert_fp8_to_int4.py "
          f"--indir {outdir} --outdir {outdir}_i4 && "
          f"SNAP={outdir}_i4 REF={ref_path} TF=1 ./colibri")


if __name__ == "__main__":
    main()
