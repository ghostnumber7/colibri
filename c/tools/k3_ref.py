"""Pure-torch reference forward for the Kimi-K3 text backbone (model_type
`kimi_linear`) -- the oracle the C engine's K3 port is validated against.

Written FROM THE EXTRACTED SPEC in docs/K3-PORT.md (situ activation, latent MoE,
NoPE MLA with output gate, KDA delta-rule recurrence with the -5.0 safe gate,
DenseFormer-style residual mixers), independently of Moonshot's modeling code, so
it can live in this repo without carrying the Kimi K3 License. It is cross-checked
one time against the HF repo's own `modeling_kimi_linear.py` (with fla's MIT
`naive_chunk_kda`/`naive_kda_lowerbound_gate` as kernel stubs) by
`tests/test_k3_ref_crosscheck.py` -- network/deps gated, skipped by default -- so a
misreading of the spec cannot silently become the engine's ground truth.

Conventions:
- weights: a dict name -> f32 torch.Tensor using CONTAINER names (no
  `language_model.` prefix): model.layers.N..., model.embed_tokens.weight,
  lm_head.weight, model.norm.weight, model.output_attn_res_{norm,proj}.weight.
- config: the FLATTENED container config dict (text_config content).
- forward(ids) -> f32 logits [T, vocab]; batch 1, no cache (the KDA recurrence is
  sequential over T, which is exactly the engine's decode-path math; chunked
  prefill kernels must reproduce it bit-for-close).

A_log length semantics (see docs/K3-PORT.md): len == head_dim -> per-dim decay
temperature exp(A_log[d]); len == num_heads -> per-head exp(A_log[h]). The real
K3 checkpoint ships per-dim [128]; the released modeling file's [num_heads]
allocation cannot load it.
"""
import math

import torch
import torch.nn.functional as F

__all__ = ["K3Ref"]


def _rmsnorm(x, weight, eps):
    x32 = x.float()
    return (weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps))).to(x.dtype)


def _situ(gate, up, beta, linear_beta):
    """situ(g)*u' : beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta).
    linear_beta may be None (up passes through), matching the modeling code."""
    g = gate.float()
    u = up.float()
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return a * u


def _l2norm(x, eps=1e-6):
    """Per-vector l2 normalization over the last dim (fla's use_qk_l2norm_in_kernel)."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


class K3Ref:
    def __init__(self, config, weights):
        self.cfg = config
        self.w = weights
        self.eps = config.get("rms_norm_eps", 1e-5)
        self.hidden = config["hidden_size"]
        self.n_layers = config["num_hidden_layers"]
        self.situ_beta = config.get("activation_situ_beta") or 1.0
        self.situ_linear_beta = config.get("activation_situ_linear_beta")
        assert config.get("hidden_act", "situ") == "situ"
        lac = config["linear_attn_config"]
        # 1-INDEXED in the config (configuration_kimi_k3.is_kda_layer): layer i
        # (0-indexed) is KDA iff (i+1) is in kda_layers.
        self.kda_layers = {i - 1 for i in lac["kda_layers"]}
        self.kda_heads = lac["num_heads"]
        self.kda_head_dim = lac["head_dim"]
        self.conv_size = lac["short_conv_kernel_size"]
        self.gate_lower_bound = lac.get("gate_lower_bound")
        assert lac.get("use_full_rank_gate", False), "only full-rank output gate supported"
        # MLA
        self.q_lora = config["q_lora_rank"]
        self.kv_lora = config["kv_lora_rank"]
        self.qk_nope = config["qk_nope_head_dim"]
        self.qk_rope = config["qk_rope_head_dim"]
        self.v_head = config["v_head_dim"]
        self.n_heads = config["num_attention_heads"]
        assert config.get("mla_use_nope", False), "only the NoPE MLA variant is supported"
        # MoE
        self.first_dense = config.get("first_k_dense_replace", 0)
        self.n_experts = config["num_experts"]
        self.top_k = config["num_experts_per_token"]
        self.n_shared = config.get("num_shared_experts") or 0
        self.routed_scale = config.get("routed_scaling_factor", 1.0)
        self.renormalize = config.get("moe_renormalize", True)
        assert config.get("moe_router_activation_func", "sigmoid") == "sigmoid"
        assert config.get("num_expert_group", 1) == 1, "expert groups not supported"
        self.latent = config.get("routed_expert_hidden_size")
        self.latent_norm = config.get("latent_moe_use_norm", False)
        # Residual mixers
        self.res_block = config.get("attn_res_block_size")

    # ---------------- building blocks ----------------

    def _mlp(self, h, prefix):
        gate = h @ self.w[prefix + ".gate_proj.weight"].T
        up = h @ self.w[prefix + ".up_proj.weight"].T
        return _situ(gate, up, self.situ_beta, self.situ_linear_beta) @ self.w[prefix + ".down_proj.weight"].T

    def _expert(self, z, prefix):
        """Routed expert in latent space: w1=gate, w3=up, w2=down (KimiBlockSparseMLP)."""
        gate = z @ self.w[prefix + ".w1.weight"].T
        up = z @ self.w[prefix + ".w3.weight"].T
        return _situ(gate, up, self.situ_beta, self.situ_linear_beta) @ self.w[prefix + ".w2.weight"].T

    def _moe(self, h, li):
        p = f"model.layers.{li}.block_sparse_moe"
        scores = torch.sigmoid(h.float() @ self.w[p + ".gate.weight"].float().T)     # [T, E]
        choice = scores + self.w[p + ".gate.e_score_correction_bias"].float()
        topk_idx = torch.topk(choice, self.top_k, dim=-1, sorted=False).indices      # [T, k]
        topk_w = scores.gather(1, topk_idx)                                          # raw sigmoid scores
        if self.top_k > 1 and self.renormalize:
            topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
        topk_w = topk_w * self.routed_scale

        z = h @ self.w[p + ".routed_expert_down_proj.weight"].T if self.latent else h
        y = torch.zeros_like(z)
        for t in range(z.shape[0]):
            for slot in range(self.top_k):
                e = int(topk_idx[t, slot])
                y[t] += topk_w[t, slot] * self._expert(z[t:t + 1], f"{p}.experts.{e}")[0]
        if self.latent:
            if self.latent_norm:
                y = _rmsnorm(y, self.w[p + ".routed_expert_norm.weight"], self.eps)
            y = y @ self.w[p + ".routed_expert_up_proj.weight"].T
        if self.n_shared:
            y = y + self._mlp(h, p + ".shared_experts")
        return y

    def _mla(self, h, li):
        p = f"model.layers.{li}.self_attn"
        T = h.shape[0]
        H, dn, dr, dv = self.n_heads, self.qk_nope, self.qk_rope, self.v_head
        q = _rmsnorm(h @ self.w[p + ".q_a_proj.weight"].T, self.w[p + ".q_a_layernorm.weight"], self.eps)
        q = (q @ self.w[p + ".q_b_proj.weight"].T).view(T, H, dn + dr)
        ckv = h @ self.w[p + ".kv_a_proj_with_mqa.weight"].T                          # [T, kv_lora+dr]
        k_pass, k_rot = ckv[:, :self.kv_lora], ckv[:, self.kv_lora:]
        kv = _rmsnorm(k_pass, self.w[p + ".kv_a_layernorm.weight"], self.eps)
        kv = (kv @ self.w[p + ".kv_b_proj.weight"].T).view(T, H, dn + dv)
        k_nope, v = kv[:, :, :dn], kv[:, :, dn:]
        # NoPE: the rope-dim halves are used UNROTATED; k_rot is shared across heads.
        k = torch.cat([k_nope, k_rot.unsqueeze(1).expand(T, H, dr)], dim=-1)          # [T, H, dn+dr]
        scale = (dn + dr) ** -0.5
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", att, v.float()).reshape(T, H * dv)
        gate = torch.sigmoid(h @ self.w[p + ".g_proj.weight"].T).float()
        return (out * gate).to(h.dtype) @ self.w[p + ".o_proj.weight"].T

    def _conv_silu(self, x, weight):
        """Depthwise causal conv (kernel K, left zero-pad K-1) + SILU. x [T, D],
        weight [D, 1, K] with the LAST tap on the current token."""
        K = weight.shape[-1]
        xt = x.T.unsqueeze(0)                                                         # [1, D, T]
        y = F.conv1d(F.pad(xt, (K - 1, 0)), weight, groups=x.shape[1])
        return F.silu(y[0].T)

    def _kda(self, h, li):
        p = f"model.layers.{li}.self_attn"
        T = h.shape[0]
        H, D = self.kda_heads, self.kda_head_dim
        q = self._conv_silu(h @ self.w[p + ".q_proj.weight"].T, self.w[p + ".q_conv1d.weight"])
        k = self._conv_silu(h @ self.w[p + ".k_proj.weight"].T, self.w[p + ".k_conv1d.weight"])
        v = self._conv_silu(h @ self.w[p + ".v_proj.weight"].T, self.w[p + ".v_conv1d.weight"])
        q = _l2norm(q.view(T, H, D).float()) * D ** -0.5
        k = _l2norm(k.view(T, H, D).float())
        v = v.view(T, H, D).float()

        g_raw = ((h @ self.w[p + ".f_a_proj.weight"].T) @ self.w[p + ".f_b_proj.weight"].T).float()
        g_raw = g_raw.view(T, H, D) + self.w[p + ".dt_bias"].float().view(H, D)
        a_log = self.w[p + ".A_log"].float()
        if a_log.numel() == D:            # real K3 checkpoint: per-dim temperature
            a = a_log.exp().view(1, 1, D)
        elif a_log.numel() == H:          # released modeling code / 48B convention
            a = a_log.exp().view(1, H, 1)
        else:
            raise ValueError(f"A_log has {a_log.numel()} entries; expected head_dim {D} or num_heads {H}")
        assert self.gate_lower_bound is not None, "only the safe (lower-bound) gate is supported"
        g = self.gate_lower_bound * torch.sigmoid(a * g_raw)                          # [T, H, D], in (lb, 0)
        beta = torch.sigmoid((h @ self.w[p + ".b_proj.weight"].T).float())            # [T, H]

        S = torch.zeros(H, D, D)                                                      # [H, K, V]
        o = torch.zeros(T, H, D)
        for t in range(T):
            S = S * g[t].exp().unsqueeze(-1)                                          # per-k-dim decay
            kt = k[t]                                                                 # [H, K]
            err = v[t] - torch.einsum("hk,hkv->hv", kt, S)                            # delta
            S = S + torch.einsum("hk,hv->hkv", beta[t].unsqueeze(-1) * kt, err)
            o[t] = torch.einsum("hk,hkv->hv", q[t], S)

        # Gated RMSNorm per head: rmsnorm(o) * o_norm.weight * sigmoid(g_proj(h)).
        gate = (h @ self.w[p + ".g_proj.weight"].T).float().view(T, H, D)
        o = _rmsnorm(o, self.w[p + ".o_norm.weight"].float(), self.eps) * torch.sigmoid(gate)
        return o.reshape(T, H * D).to(h.dtype) @ self.w[p + ".o_proj.weight"].T

    # ---------------- residual mixing (attn_res_block_size) ----------------

    def _apply_attn_res(self, x, block_residual, norm_w, proj_w):
        """softmax-weighted mix of [block snapshots..., x]; scores are rmsnormed
        candidates dotted with (norm.weight * proj.weight). Mixes the UN-normalized
        candidates (modeling_kimi_linear._apply_attn_res)."""
        v = torch.cat([block_residual, x.unsqueeze(1)], dim=1).float()                # [T, n+1, hid]
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        score_w = norm_w.float() * proj_w.float().squeeze(0)
        probs = torch.softmax((k * score_w).sum(-1), dim=-1)                          # [T, n+1]
        return torch.einsum("tn,tnh->th", probs, v).to(x.dtype)

    # ---------------- forward ----------------

    def forward(self, ids):
        cfg_dtype = torch.float32
        h = self.w["model.embed_tokens.weight"][ids].to(cfg_dtype)                    # [T, hid]
        T = h.shape[0]
        block_residual = h.new_zeros(T, 0, self.hidden) if self.res_block else None

        for li in range(self.n_layers):
            lp = f"model.layers.{li}"
            if self.res_block:
                prefix_sum = h
                if block_residual.shape[1] > 0:
                    h = self._apply_attn_res(prefix_sum, block_residual,
                                             self.w[lp + ".self_attention_res_norm.weight"],
                                             self.w[lp + ".self_attention_res_proj.weight"])
                if li % self.res_block == 0:
                    block_residual = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)
                    prefix_sum = None
                x = _rmsnorm(h, self.w[lp + ".input_layernorm.weight"], self.eps)
                attn = self._kda(x, li) if li in self.kda_layers else self._mla(x, li)
                prefix_sum = attn if prefix_sum is None else prefix_sum + attn
                h = self._apply_attn_res(prefix_sum, block_residual,
                                         self.w[lp + ".mlp_res_norm.weight"],
                                         self.w[lp + ".mlp_res_proj.weight"])
                x = _rmsnorm(h, self.w[lp + ".post_attention_layernorm.weight"], self.eps)
                m = self._moe(x, li) if li >= self.first_dense else self._mlp(x, lp + ".mlp")
                h = prefix_sum + m
            else:
                x = _rmsnorm(h, self.w[lp + ".input_layernorm.weight"], self.eps)
                h = h + (self._kda(x, li) if li in self.kda_layers else self._mla(x, li))
                x = _rmsnorm(h, self.w[lp + ".post_attention_layernorm.weight"], self.eps)
                h = h + (self._moe(x, li) if li >= self.first_dense else self._mlp(x, lp + ".mlp"))

        if self.res_block:
            h = self._apply_attn_res(h, block_residual,
                                     self.w["model.output_attn_res_norm.weight"],
                                     self.w["model.output_attn_res_proj.weight"])
        h = _rmsnorm(h, self.w["model.norm.weight"], self.eps)
        return h @ self.w["lm_head.weight"].T
