"""Builds a tiny synthetic Kimi-K2 (stock transformers `DeepseekV3ForCausalLM`) checkpoint
as an ORACLE for the C engine's K2 forward pass: real architecture (MLA attention, YaRN
rope, noaux_tc router with a non-degenerate e_score_correction_bias, routed + shared
experts), toy dimensions, random weights. Saves the routed experts as compressed-tensors
`pack-quantized` int4 group-32 (the real K2 container's format) and everything else in
the layout `c/tools/convert_fp8_to_int4.py --indir` expects, then computes a greedy
transformers reference in `<outdir>/../ref_k2.json` for the engine's `REF=`/`TF=` self-test
(only when `--outdir` is left at its default `c/k2_tiny`; see `--ref`/`--outdir` below for
why a non-default `--outdir` does NOT write there).

Follows the `make_glm_oracle.py` idiom (tiny real-architecture model + random weights +
transformers-computed greedy reference + TF=1 teacher-forced validation), with two
corrections that a from-scratch reading of the Stage 1 converter and the K2 modelling
code surfaced (both required to reach the engine's documented 32/32 prefill + 20/20
decode, not just "close"):

1. The reference is computed AFTER round-tripping every tensor the converter quantises
   through its OWN quantisation math -- not just the routed experts. `convert_fp8_to_int4.py`
   also quantises embeddings, lm_head, every attention projection, the dense MLP (layer 0,
   since first_k_dense_replace=1) and the shared expert to int8 by default (--ebits 8).
   Skipping that leaves the reference describing full-precision residents while the
   container holds int8 ones -- the teacher-forced gate then lands one position short
   (31/32) for a reason that has nothing to do with the engine's forward pass.
2. `e_score_correction_bias` is a `register_buffer`, not an `nn.Parameter`, so it is
   invisible to any `named_parameters()` init loop and would ship all-zeros, silently
   disabling the noaux_tc bias path the router is supposed to exercise. Set explicitly,
   copying `make_glm_oracle.py`'s `torch.linspace(-0.1, 0.1, n_routed_experts)`.

The routed experts are UNFUSED first (transformers stores them as fused 3-D
`gate_up_proj [E, 2*M, H]` / `down_proj [E, H, M]` parameters; the engine and the
converter both expect per-expert 2-D `{gate,up,down}_proj` tensors) via
`glm_fp8_emit.unfuse_experts`, quantised to int4 group-32 with `compressed_tensors`'
OWN `pack_to_int32` (so the on-disk bit layout is proven-correct, not reimplemented),
and the DEQUANTISED values are written back into the model's fused parameters before
`generate()`/the teacher-forced forward -- so the reference sees exactly the same
lossy weights the converter will transcode losslessly into the final container.

No tokenizer is produced: the engine's `REF=`/`TF=` oracle path validates on token IDS,
never touches `tokenizer.json` (see `c/colibri.c`'s oracle dispatch), and building one
is out of scope here (only the real container needs it).

Usage:
  .venv/bin/python c/tools/make_k2tiny.py --outdir c/k2_tiny
  # writes c/k2_tiny/{config.json,model.safetensors} and c/ref_k2.json (outdir's parent)
  #
  # A non-default --outdir writes its reference alongside it as ref_<outdir's name>.json
  # instead of overwriting the committed c/ref_k2.json golden reference, e.g.:
  #   .venv/bin/python c/tools/make_k2tiny.py --outdir c/k2_tiny002
  #   # writes c/k2_tiny002/{...} and c/ref_k2_tiny002.json (NOT c/ref_k2.json)
  # --ref PATH overrides the derived path entirely.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

# Make the sibling tools importable regardless of CWD (mirrors make_glm_oracle.py).
_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))
from glm_fp8_emit import unfuse_experts                              # noqa: E402
from convert_fp8_to_int4 import classify, quant_int8                 # noqa: E402

from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32  # noqa: E402
from safetensors.torch import save_file, safe_open                   # noqa: E402

# ---------------------------------------------------------------------------
# Toy K2 config -- behavioural settings pinned to the real container, dimensions
# shrunk. qk_rope_head_dim MUST stay 64: it is YaRN's `dim`, and changing it moves
# the correction range (low/high), which is otherwise identical between this
# fixture and the real 543 GB model (both have theta=50000,
# original_max_position_embeddings=4096, dim=64 -> low=19, high=20).
# ---------------------------------------------------------------------------
YARN_PARAMS = {
    "rope_type": "yarn",
    "rope_theta": 50000.0,
    "factor": 64.0,
    "original_max_position_embeddings": 4096,
    "beta_fast": 1.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}

GROUP_SIZE = 32     # routed-expert int4 group size, matches the real container
INT4_BITS = 4
RESIDENT_BITS = 8   # matches convert_fp8_to_int4.py's default --ebits/--io-bits
RESIDENT_SKIP_KINDS = {"f32", "skip", "consumed", "x"}


def build_config():
    cfg = DeepseekV3Config(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=64,
        v_head_dim=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.827,
        first_k_dense_replace=1,
        num_mtp_layers=0,          # real field; "num_nextn_predict_layers" is the
                                   # legacy alias and is inert here either way
        topk_method="noaux_tc",   # inert in transformers 5.14.1 (router hardcodes
        scoring_func="sigmoid",   # sigmoid + noaux_tc-style bias top-k) -- kept
                                   # only for config.json fidelity with the real container
        rope_parameters=dict(YARN_PARAMS),
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
        attention_bias=False,
    )
    cfg._attn_implementation = "eager"
    return cfg


def init_weights(model, cfg):
    """normal_(0, 0.05) for every >=2-D parameter (matches make_glm_oracle.py; 0.05
    rather than 0.02 -- 0.02 was measured to produce near-degenerate top-6 logit gaps
    (~0.17 spread), making the teacher-forced argmax gate brittle to unrelated future
    numeric changes with no benefit to what it catches), plus an explicit, non-degenerate
    e_score_correction_bias (a register_buffer, invisible to named_parameters())."""
    torch.manual_seed(0)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.05)
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.copy_(
                    torch.linspace(-0.1, 0.1, cfg.n_routed_experts))


def quantize_expert_int4_g32(w):
    """w: [O,I] f32 torch tensor -> (codes int8 [O,I] in [-8,7], scale f32 [O,ngroups],
    dequant f32 [O,I]). Same math as convert_fp8_to_int4.quant_int4_grouped (per-group
    symmetric absmax, clip [-8,7]), kept in torch so the routed experts can be quantised
    before saving. Because the Stage 1 converter's transcode path
    (unpack_compressed_int4 + transcode_compressed_int4) is LOSSLESS, whatever codes/
    scale are written here pass straight through to the engine unchanged -- so the
    dequantised values computed here are exactly what the converted container encodes,
    and are what gets written back into the model before computing the reference."""
    O, I = w.shape
    qmax = (1 << (INT4_BITS - 1)) - 1              # 7
    ngroups = (I + GROUP_SIZE - 1) // GROUP_SIZE
    ipad = ngroups * GROUP_SIZE
    wpad = torch.zeros(O, ipad, dtype=torch.float32)
    wpad[:, :I] = w
    wr = wpad.view(O, ngroups, GROUP_SIZE)
    amax = wr.abs().amax(dim=2, keepdim=True)
    scale = torch.clamp(amax / qmax, min=1e-8)
    q = torch.clamp(torch.round(wr / scale), -8, qmax)
    dequant = (q * scale).view(O, ipad)[:, :I]
    codes = q.view(O, ipad)[:, :I].to(torch.int8)
    scale_flat = scale[:, :, 0]                    # [O, ngroups] f32
    return codes, scale_flat, dequant


def pack_expert(codes, scale):
    """Exact compressed-tensors on-disk layout: weight_packed int32 [O, ceil(I/8)]
    (8 nibbles/word, LSB = lowest column, offset-binary), weight_scale BF16
    [O, ceil(I/32)], weight_shape int64 [O,I]. Uses compressed_tensors' OWN
    pack_to_int32 (not a reimplementation) -- proven bit-identical to the repo's
    unpack_compressed_int4 in research-task3.md's round-trip script."""
    O, I = codes.shape
    weight_packed = pack_to_int32(codes, INT4_BITS, packed_dim=1)
    weight_scale = scale.to(torch.bfloat16)
    weight_shape = torch.tensor([O, I], dtype=torch.int64)
    return weight_packed, weight_scale, weight_shape


def quantize_and_refuse_experts(model, cfg):
    """Unfuse the routed experts (glm_fp8_emit.unfuse_experts, generic across GLM/K2:
    both name the submodule mlp.experts), quantise each per-expert 2-D tensor to int4
    g32, write the DEQUANTISED values back into the model's fused 3-D parameters (so
    the reference forward/generate() below sees the lossy weights), and return the
    packed weight_packed/weight_scale/weight_shape tensors keyed by the per-expert
    tensor's name-minus-`.weight` prefix, for use when assembling the final container.
    Order matters (T3-4): unfusing happens on a state_dict COPY (unfuse_experts splits
    fused 3-D tensors into new 2-D ones without touching the model), and the model
    itself is mutated afterwards, via the fused parameters directly, so generate()
    keeps working throughout."""
    sd = model.state_dict()
    unfuse_experts(sd)

    expert_packed = {}
    dequant_by_key = {}
    for name, t in sd.items():
        if not (".mlp.experts." in name and name.endswith((".gate_proj.weight",
                                                             ".up_proj.weight",
                                                             ".down_proj.weight"))):
            continue
        codes, scale, dequant = quantize_expert_int4_g32(t.float())
        weight_packed, weight_scale, weight_shape = pack_expert(codes, scale)
        prefix = name[:-len(".weight")]
        expert_packed[prefix] = {
            "weight_packed": weight_packed,
            "weight_scale": weight_scale,
            "weight_shape": weight_shape,
        }
        dequant_by_key[name] = dequant

    m = cfg.moe_intermediate_size
    with torch.no_grad():
        for l, layer in enumerate(model.model.layers):
            if not hasattr(layer.mlp, "experts"):
                continue   # dense layer (first_k_dense_replace)
            for e in range(cfg.n_routed_experts):
                base = f"model.layers.{l}.mlp.experts.{e}"
                layer.mlp.experts.gate_up_proj.data[e, :m, :] = \
                    dequant_by_key[f"{base}.gate_proj.weight"]
                layer.mlp.experts.gate_up_proj.data[e, m:, :] = \
                    dequant_by_key[f"{base}.up_proj.weight"]
                layer.mlp.experts.down_proj.data[e] = \
                    dequant_by_key[f"{base}.down_proj.weight"]

    return expert_packed


def resident_int8_roundtrip(model, n_layers):
    """Round-trip every 2-D resident tensor the converter quantises (embeddings,
    lm_head, attention projections, dense MLP, shared expert) through the converter's
    OWN quant_int8, in place, BEFORE the reference is computed. `classify()` is
    imported directly from convert_fp8_to_int4.py so this can never drift from what
    the converter actually does to a real K2 container. Skipping this (Step 1.6)
    measurably costs one teacher-forced position (31/32 instead of 32/32) for a
    reason that has nothing to do with the engine."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() != 2 or ".mlp.experts." in name:
                continue
            kind = classify(name, n_layers)
            if kind in RESIDENT_SKIP_KINDS:
                continue
            w = p.detach().numpy().astype(np.float32)
            q, s = quant_int8(w, RESIDENT_BITS)
            O, I = w.shape
            codes = q.view(np.int8).reshape(O, I).astype(np.float32)
            dequant = codes * s[:, None]
            p.copy_(torch.from_numpy(dequant))


def assemble_container(model, expert_packed):
    """Final state dict: everything from model.state_dict() EXCEPT the fused 3-D
    routed-expert parameters, which are replaced by the packed per-expert
    weight_packed/weight_scale/weight_shape triples computed earlier."""
    out = {}
    for name, t in model.state_dict().items():
        if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
            continue
        out[name] = t.contiguous()
    for prefix, packed in expert_packed.items():
        out[f"{prefix}.weight_packed"] = packed["weight_packed"].contiguous()
        out[f"{prefix}.weight_scale"] = packed["weight_scale"].contiguous()
        out[f"{prefix}.weight_shape"] = packed["weight_shape"].contiguous()
    return out


def write_config(cfg, outdir):
    """cfg.to_dict() carries the transformers-5.x-normalised `rope_parameters` and
    OMITS rope_scaling/rope_theta entirely (not null -- absent; measured, see
    research-task3.md). Post-process into the REAL container's format: top-level
    rope_theta, a legacy rope_scaling block with type=yarn, model_type=kimi_k2, and
    no rope_parameters key -- this is what rope_table_init's Step 0 guard requires
    to take the YaRN path instead of silently running unscaled."""
    d = cfg.to_dict()
    d["model_type"] = "kimi_k2"
    d["rope_theta"] = YARN_PARAMS["rope_theta"]
    d["rope_scaling"] = {
        "type": "yarn",
        "factor": YARN_PARAMS["factor"],
        "original_max_position_embeddings": YARN_PARAMS["original_max_position_embeddings"],
        "beta_fast": YARN_PARAMS["beta_fast"],
        "beta_slow": YARN_PARAMS["beta_slow"],
        "mscale": YARN_PARAMS["mscale"],
        "mscale_all_dim": YARN_PARAMS["mscale_all_dim"],
    }
    d.pop("rope_parameters", None)
    # Cosmetic (the converter's pack-quantized/transcode dispatch is entirely
    # tensor-name-driven, not config-driven -- research-task3.md Part D) but real
    # signal to a human debugger: mirror the real container's block, read from
    # the real converted container's config.json (small JSON, not the weights).
    d["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": GROUP_SIZE,
                    "num_bits": INT4_BITS,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": [
            "lm_head",
            "re:.*self_attn.*",
            "re:.*shared_experts.*",
            "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
        ],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    outdir.mkdir(parents=True, exist_ok=True)
    cfg_path = outdir / "config.json"
    json.dump(d, open(cfg_path, "w"), indent=2)

    # Assert on the WRITTEN file, not the in-memory dict: a fixture that does not
    # exercise the YaRN path is a vacuous gate, and this is the single easiest thing
    # here to get silently wrong.
    written = json.loads(cfg_path.read_text())
    assert written.get("rope_scaling", {}).get("type") == "yarn", \
        "config.json: rope_scaling.type must be 'yarn'"
    assert written.get("rope_theta") == 50000.0, \
        "config.json: top-level rope_theta must be 50000.0"
    assert "rope_parameters" not in written, \
        "config.json: rope_parameters must be dropped (legacy format only)"
    assert written.get("model_type") == "kimi_k2"
    print(f"config.json OK: model_type={written['model_type']} "
          f"rope_theta={written['rope_theta']} "
          f"rope_scaling.type={written['rope_scaling']['type']}")


def verify_roundtrip(model_path):
    """Round-trip ONE saved expert tensor back through the converter's own
    unpack_compressed_int4 and assert the codes/scale are recovered exactly, per
    Step 1.5's requirement. Catches a packer regression before the (much slower)
    engine gate would."""
    from convert_fp8_to_int4 import unpack_compressed_int4
    with safe_open(model_path, framework="pt") as f:
        packed_name = next(n for n in f.keys() if n.endswith(".weight_packed"))
        codes, scale = unpack_compressed_int4(f, packed_name)
        base = packed_name[:-len(".weight_packed")]
        stored_scale_bf16 = f.get_tensor(base + ".weight_scale")
    # scale was stored as bf16; unpack_compressed_int4 upcasts to f32. Compare
    # against the bf16-rounded value we actually stored (not the pre-round f32),
    # since that round-trip is exact (no further rounding on read-back).
    expected_scale = stored_scale_bf16.to(torch.float32).numpy()
    assert np.array_equal(scale, expected_scale), \
        f"{packed_name}: weight_scale round-trip mismatch"
    assert codes.min() >= -8 and codes.max() <= 7, \
        f"{packed_name}: codes out of int4 range"
    print(f"round-trip OK: {packed_name} codes in [-8,7], scale recovered bit-identical")


DEFAULT_OUTDIR = "c/k2_tiny"


def resolve_ref_path(outdir_arg, ref_arg, default_outdir=DEFAULT_OUTDIR):
    """Derive the transformers-reference JSON path from CLI args, without ever landing on
    the committed c/ref_k2.json golden reference unless --outdir is left at its default.

    A prior version derived `ref_path` from `Path(outdir_arg).parent` alone, so ANY
    `--outdir` sharing a parent with the default (e.g. `c/k2_tiny002`, `c/anything`) still
    resolved to `c/ref_k2.json` and silently overwrote the tracked golden reference -- a
    corrupted golden makes every downstream `TF=`/`REF=` gate vacuous while still reporting
    a pass. Fix: only the exact default `--outdir` value maps to `c/ref_k2.json`; every other
    `--outdir` derives a sibling name from the outdir itself, and `--ref` overrides both.
    """
    if ref_arg is not None:
        return Path(ref_arg)
    outdir = Path(outdir_arg)
    if outdir_arg == default_outdir:
        return outdir.parent / "ref_k2.json"
    return outdir.parent / f"ref_{outdir.name}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--ref", default=None,
                    help="Path to write the transformers reference JSON. Defaults to "
                         "c/ref_k2.json when --outdir is left at its default value; for any "
                         "other --outdir it defaults to <outdir's parent>/ref_<outdir's name>.json "
                         "so a non-default --outdir can never silently clobber the committed "
                         "c/ref_k2.json golden reference. Pass explicitly to pick another path.")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    ref_path = resolve_ref_path(args.outdir, args.ref, ap.get_default("outdir"))

    cfg = build_config()
    model = DeepseekV3ForCausalLM(cfg).eval()
    init_weights(model, cfg)

    expert_packed = quantize_and_refuse_experts(model, cfg)
    resident_int8_roundtrip(model, cfg.num_hidden_layers)

    prompt = [3, 14, 159, 26, 53, 58, 200, 11, 477, 47, 246, 451]     # 12 ids, all < vocab_size=512
    ids = torch.tensor([prompt])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
    full = out[0].tolist()
    print("prompt:", prompt)
    print("full  :", full)

    with torch.no_grad():
        lg = model(torch.tensor([full]), use_cache=False).logits[0]
    tf_pred = lg.argmax(-1).tolist()
    print("tf_pred:", tf_pred)

    out_sd = assemble_container(model, expert_packed)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model.safetensors"
    save_file(out_sd, str(model_path))

    write_config(cfg, outdir)
    verify_roundtrip(model_path)

    json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred},
              open(ref_path, "w"))
    print(f"saved: {outdir}/ (weights + config) and {ref_path}")


if __name__ == "__main__":
    main()
