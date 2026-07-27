"""Cross-check tools/k3_ref.py (the in-repo, license-clean K3 oracle reference)
against Moonshot's own modeling_kimi_linear.py, with fla replaced by naive stubs
whose semantics come from fla-org/flash-linear-attention (MIT): ops/kda/naive.py
(delta-rule recurrence), ops/kda/gate.py (lower-bound safe gate), and
modules/fused_norm_gate.py (sigmoid-gated RMSNorm).

Gated on COLI_K3_MODELING_DIR: a directory holding the K3 repo's
`modeling_kimi_linear.py` and `configuration_kimi_k3.py` (they carry the Kimi K3
License and are not vendored here):

    huggingface-cli download moonshotai/Kimi-K3 modeling_kimi_linear.py \
        configuration_kimi_k3.py --local-dir /tmp/k3_modeling
    COLI_K3_MODELING_DIR=/tmp/k3_modeling pytest tests/test_k3_ref_crosscheck.py

What this proves: k3_ref's situ activation, latent MoE + sigmoid router, NoPE MLA
with output gate, KDA layer plumbing, and the DenseFormer-style residual mixers
reproduce Moonshot's layer/model control flow (maxdiff ~3e-5 fp32 on a 7-layer
tiny model). What it can NOT prove (shared assumptions on both sides, see
docs/K3-PORT.md): the per-dim A_log reading (the released modeling file's own
[num_heads] allocation cannot load the real checkpoint and is patched here), the
l2norm epsilon, and the exact conv/norm-gate formulas taken from fla sources.
Final arbiter for those: the real container producing coherent text + SCORE loss.
"""
import os
import sys
import types
import importlib.util

import pytest
import torch
import torch.nn.functional as F

K3_DIR = os.environ.get("COLI_K3_MODELING_DIR", "")
pytestmark = pytest.mark.skipif(
    not (K3_DIR and os.path.exists(os.path.join(K3_DIR, "modeling_kimi_linear.py"))
         and os.path.exists(os.path.join(K3_DIR, "configuration_kimi_k3.py"))),
    reason="needs the K3 repo modeling files (set COLI_K3_MODELING_DIR)")

TOOLS = os.path.join(os.path.dirname(__file__), "..", "tools")

H_KDA, D_KDA = 2, 16


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def _kda_gate(g, A_log, dt_bias, lower_bound):
    """fla ops/kda/gate.py naive_kda_lowerbound_gate, extended: A_log per-head [H]
    (fla main) or per-dim [D] (the real K3 checkpoint's convention)."""
    H, D = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, D)
    if A_log.numel() == H:
        a = A_log.float().exp().view(H, 1)
    elif A_log.numel() == D:
        a = A_log.float().exp().view(1, D)
    else:
        raise ValueError(A_log.shape)
    return lower_bound * torch.sigmoid(a * g)


def _naive_recurrent_kda(q, k, v, g, beta, scale, initial_state):
    """fla ops/kda/naive.py naive_recurrent_kda (H == HV case)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, g, beta = map(lambda x: x.float(), [q, k, v, g, beta])
    q = q * scale
    S = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('bhk,bhv->bhkv', b_i[..., None] * k_i,
                             v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('bhk,bhkv->bhv', q_i, S)
    return o, S


def _kda(q, k, v, g, beta, A_log, dt_bias, initial_state, output_final_state,
         use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
         use_beta_sigmoid_in_kernel=False, safe_gate=False, lower_bound=None,
         transpose_state_layout=False, cu_seqlens=None, **kw):
    assert use_qk_l2norm_in_kernel and use_gate_in_kernel and use_beta_sigmoid_in_kernel
    assert lower_bound is not None
    q = _l2norm(q.float())
    k = _l2norm(k.float())
    g = _kda_gate(g, A_log, dt_bias, lower_bound)
    beta = torch.sigmoid(beta.float())
    o, S = _naive_recurrent_kda(q, k, v, g, beta, q.shape[-1] ** -0.5, initial_state)
    return o.to(v.dtype), S


class _ShortConvolution(torch.nn.Module):
    """fla ShortConvolution: depthwise causal conv1d + silu, no bias."""

    def __init__(self, hidden_size, kernel_size, activation=None, bias=False):
        super().__init__()
        assert activation == 'silu' and not bias
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(torch.randn(hidden_size, 1, kernel_size))

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        B, T, D = x.shape
        xt = x.transpose(1, 2)
        y = F.conv1d(F.pad(xt, (self.kernel_size - 1, 0)), self.weight, groups=D)
        return F.silu(y).transpose(1, 2), None


class _FusedRMSNormGated(torch.nn.Module):
    """fla fused_norm_gate.py, ACTIVATION == 'sigmoid': rmsnorm(x)*w*sigmoid(g)."""

    def __init__(self, hidden_size, eps=1e-6, activation='sigmoid'):
        super().__init__()
        assert activation == 'sigmoid'
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, g):
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (y * torch.sigmoid(g.float())).to(x.dtype)


def _install_fla_stubs():
    fla = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    fla_modules.FusedRMSNormGated = _FusedRMSNormGated
    fla_modules.ShortConvolution = _ShortConvolution
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_kda = types.ModuleType("fla.ops.kda")
    fla_ops_kda.chunk_kda = _kda
    fla_ops_kda.fused_recurrent_kda = _kda
    fla_ops_utils = types.ModuleType("fla.ops.utils")
    fla_ops_utils_index = types.ModuleType("fla.ops.utils.index")
    fla_ops_utils_index.prepare_cu_seqlens_from_mask = lambda m: None
    fla_ops_utils_index.prepare_lens_from_mask = lambda m: m.sum(-1)
    fla_utils = types.ModuleType("fla.utils")
    fla_utils.tensor_cache = lambda f: f
    for name, mod in [("fla", fla), ("fla.modules", fla_modules), ("fla.ops", fla_ops),
                      ("fla.ops.kda", fla_ops_kda), ("fla.ops.utils", fla_ops_utils),
                      ("fla.ops.utils.index", fla_ops_utils_index), ("fla.utils", fla_utils)]:
        sys.modules[name] = mod


def _load_modeling():
    _install_fla_stubs()
    spec = importlib.util.spec_from_file_location(
        "configuration_kimi_k3", os.path.join(K3_DIR, "configuration_kimi_k3.py"))
    cfgmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfgmod)
    sys.modules["configuration_kimi_k3"] = cfgmod

    src = open(os.path.join(K3_DIR, "modeling_kimi_linear.py")).read()
    src = src.replace("from .configuration_kimi_k3 import KimiLinearConfig",
                      "from configuration_kimi_k3 import KimiLinearConfig")
    # transformers 5.x compat: moved/renamed helpers, all irrelevant to the math.
    src = src.replace(
        "from transformers.utils.generic import OutputRecorder, check_model_inputs",
        "OutputRecorder = lambda *a, **k: None\n"
        "def check_model_inputs(f=None, **kw):\n"
        "    if f is None: return lambda g: g\n"
        "    return f")
    src = src.replace(
        "from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging",
        "from transformers.utils import logging\n"
        "class TransformersKwargs(dict): pass\n"
        "auto_docstring = lambda f=None, **k: (f if f is not None else (lambda g: g))\n"
        "can_return_tuple = lambda f: f")
    src = src.replace(
        "from transformers.masking_utils import create_causal_mask",
        "def create_causal_mask(config=None, input_embeds=None, inputs_embeds=None, **kw):\n"
        "    x = input_embeds if input_embeds is not None else inputs_embeds\n"
        "    T = x.shape[1]\n"
        "    m = torch.full((T, T), float('-inf'), dtype=x.dtype)\n"
        "    m = torch.triu(m, diagonal=1)\n"
        "    return m[None, None, :, :].expand(x.shape[0], 1, T, T)")
    modmod = types.ModuleType("modeling_kimi_linear")
    modmod.__file__ = "modeling_kimi_linear.py"
    sys.modules["modeling_kimi_linear"] = modmod
    exec(compile(src, "modeling_kimi_linear.py", "exec"), modmod.__dict__)
    return cfgmod, modmod


LINEAR_ATTN = dict(kda_layers=[1, 2, 3, 5, 6], full_attn_layers=[4, 7],
                   num_heads=H_KDA, head_dim=D_KDA, short_conv_kernel_size=4,
                   use_full_rank_gate=True, gate_lower_bound=-5.0)
TINY = dict(
    hidden_size=48, num_hidden_layers=7, rms_norm_eps=1e-5,
    hidden_act="situ", activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
    q_lora_rank=24, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8,
    v_head_dim=8, num_attention_heads=2, mla_use_nope=True,
    first_k_dense_replace=1, num_experts=4, num_experts_per_token=2,
    moe_intermediate_size=16,
    num_shared_experts=1, routed_scaling_factor=1.0, moe_renormalize=True,
    moe_router_activation_func="sigmoid", num_expert_group=1,
    routed_expert_hidden_size=24, latent_moe_use_norm=True,
    attn_res_block_size=3, linear_attn_config=LINEAR_ATTN,
)


def test_k3_ref_matches_moonshot_modeling():
    cfgmod, modmod = _load_modeling()
    cfg = cfgmod.KimiLinearConfig(
        vocab_size=96, intermediate_size=64, mla_use_output_gate=True,
        use_cache=False, **TINY)

    torch.manual_seed(0)
    model = modmod.KimiLinearForCausalLM(cfg)
    model.eval()
    # The released modeling allocates A_log [num_heads]; the real K3 checkpoint
    # ships per-dim [head_dim] (docs/K3-PORT.md) -- reshape so that convention loads.
    for layer in model.model.layers:
        if getattr(layer, "is_linear_attn", False):
            layer.self_attn.A_log = torch.nn.Parameter(torch.zeros(D_KDA))

    gen = torch.Generator().manual_seed(1)
    new_sd = {}
    for k, v in model.state_dict().items():
        if k.endswith("A_log"):
            t = torch.log(torch.empty(D_KDA, dtype=torch.float32).uniform_(1, 16, generator=gen))
        elif "norm" in k and k.endswith("weight") and v.dim() == 1 and "conv" not in k:
            t = torch.empty_like(v).uniform_(0.8, 1.2, generator=gen)
        else:
            t = torch.empty(v.shape, dtype=torch.float32).normal_(0, 0.05, generator=gen).to(v.dtype)
        new_sd[k] = t
    model.load_state_dict(new_sd)
    model.config._attn_implementation = "eager"   # modeling force-sets flash_attention_2

    ids = torch.randint(0, 96, (1, 12), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref_logits = model(input_ids=ids, use_cache=False).logits[0].float()

    sys.path.insert(0, TOOLS)
    from k3_ref import K3Ref
    weights = {k: v.float() for k, v in model.state_dict().items()}
    with torch.no_grad():
        my_logits = K3Ref(TINY, weights).forward(ids[0])

    d = (ref_logits - my_logits).abs()
    assert d.max().item() < 1e-4, f"maxdiff {d.max().item():.3e}"
    assert (ref_logits.argmax(-1) == my_logits.argmax(-1)).all()
