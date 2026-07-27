import os, sys, json, importlib.util
import numpy as np
import pytest

TOOLS = os.path.join(os.path.dirname(__file__), "..", "tools")
spec = importlib.util.spec_from_file_location("cvt", os.path.join(TOOLS, "convert_fp8_to_int4.py"))
cvt = importlib.util.module_from_spec(spec); spec.loader.exec_module(cvt)


class FakeST:
    """Minimal stand-in for a safetensors handle (framework='pt')."""
    def __init__(self, tensors):
        import torch
        self.d = {k: (v if hasattr(v, "numpy") else torch.from_numpy(v)) for k, v in tensors.items()}
    def keys(self): return list(self.d.keys())
    def get_tensor(self, n): return self.d[n]


def _pack_int4_columns(codes):
    """Reference packer matching compressed-tensors 'pack-quantized': 8 offset-binary
    4-bit nibbles per int32 (nibble = value + 8, value -8..7 -> nibble 0..15), LSB =
    lowest column index. NOT two's complement — verified against the compressed-tensors
    reference decoder on a real K2 shard."""
    O, I = codes.shape
    pack_factor = 8
    Ipad = ((I + pack_factor - 1) // pack_factor) * pack_factor
    u = ((codes.astype(np.int32) + 8) & 0xF)                 # offset-binary nibble
    padded = np.zeros((O, Ipad), np.uint32); padded[:, :I] = u
    packed = np.zeros((O, Ipad // pack_factor), np.uint32)
    for k in range(pack_factor):
        packed |= (padded[:, k::pack_factor] << (4 * k))
    return packed.astype(np.int32)                          # stored as int32 in safetensors


def test_unpack_compressed_int4_recovers_codes_and_scale():
    import torch
    rng = np.random.default_rng(0)
    O, I, gs = 4, 64, 32
    codes = rng.integers(-8, 8, size=(O, I)).astype(np.int8)     # signed [-8,7]
    scale = (rng.random((O, I // gs)).astype(np.float32) + 0.1)
    packed = _pack_int4_columns(codes)
    f = FakeST({
        "x.gate_proj.weight_packed": packed,
        "x.gate_proj.weight_scale": torch.from_numpy(scale),
        "x.gate_proj.weight_shape": torch.tensor([O, I], dtype=torch.int64),
    })
    got_codes, got_scale = cvt.unpack_compressed_int4(f, "x.gate_proj.weight_packed")
    assert got_codes.dtype == np.int8
    assert got_codes.shape == (O, I)
    assert np.array_equal(got_codes, codes)
    assert np.allclose(got_scale, scale)


def test_unpack_compressed_int4_trims_partial_group_and_pack_padding():
    """I not a multiple of 8 (pack_factor) or 32 (group size): exercises both the
    cols[:, :I] pack-padding trim and the ceil(I/32) partial-group scale count."""
    import torch
    rng = np.random.default_rng(2)
    O, I, gs = 3, 60, 32                                          # ceil(60/8)=8 words/row, ceil(60/32)=2 groups
    ngroups = (I + gs - 1) // gs
    assert ngroups == 2
    codes = rng.integers(-8, 8, size=(O, I)).astype(np.int8)
    scale = (rng.random((O, ngroups)).astype(np.float32) + 0.1)
    packed = _pack_int4_columns(codes)
    f = FakeST({
        "x.up_proj.weight_packed": packed,
        "x.up_proj.weight_scale": torch.from_numpy(scale),
        "x.up_proj.weight_shape": torch.tensor([O, I], dtype=torch.int64),
    })
    got_codes, got_scale = cvt.unpack_compressed_int4(f, "x.up_proj.weight_packed")
    assert got_codes.dtype == np.int8
    assert got_codes.shape == (O, I)
    assert np.array_equal(got_codes, codes)
    assert np.allclose(got_scale, scale)


def test_unpack_compressed_int4_rejects_bad_shapes():
    """Each of the three shape guards must reject its own malformed layout instead of
    silently misreading it (e.g. cols[:, :I] would otherwise CLIP in silence if
    packed.shape[1]*8 < I, since numpy slicing past the end is not an error)."""
    import torch
    rng = np.random.default_rng(3)
    O, I, gs = 4, 64, 32
    codes = rng.integers(-8, 8, size=(O, I)).astype(np.int8)
    scale = (rng.random((O, I // gs)).astype(np.float32) + 0.1)
    packed = _pack_int4_columns(codes)                    # valid: packed.shape == (O, ceil(I/8))

    # 1) weight_shape row count != packed.shape[0]
    f_rows = FakeST({
        "x.gate_proj.weight_packed": packed,
        "x.gate_proj.weight_scale": torch.from_numpy(scale),
        "x.gate_proj.weight_shape": torch.tensor([O + 1, I], dtype=torch.int64),
    })
    with pytest.raises(ValueError):
        cvt.unpack_compressed_int4(f_rows, "x.gate_proj.weight_packed")

    # 2) weight_scale column count != ceil(I/32)
    f_scale = FakeST({
        "x.gate_proj.weight_packed": packed,
        "x.gate_proj.weight_scale": torch.from_numpy(scale[:, :-1]),   # drop a group column
        "x.gate_proj.weight_shape": torch.tensor([O, I], dtype=torch.int64),
    })
    with pytest.raises(ValueError):
        cvt.unpack_compressed_int4(f_scale, "x.gate_proj.weight_packed")

    # 3) weight_shape I such that packed.shape[1] != ceil(I/8) (the newly-added guard)
    f_cols = FakeST({
        "x.gate_proj.weight_packed": packed,                          # packed.shape[1] == 8 (ceil(64/8))
        "x.gate_proj.weight_scale": torch.from_numpy(scale),
        "x.gate_proj.weight_shape": torch.tensor([O, 70], dtype=torch.int64),  # ceil(70/8) == 9 != 8
    })
    with pytest.raises(ValueError):
        cvt.unpack_compressed_int4(f_cols, "x.gate_proj.weight_packed")


def _colibri_fmt4_dequant(qbytes, s_flat, O, I, gs=32):
    """Replicate colibri.c matmul_i4_grouped dequant: w = (nibble-8) * scale_group."""
    rb = (I + 1) // 2
    qb = qbytes.reshape(O, rb)
    lo = (qb & 0x0F).astype(np.int32) - 8
    hi = ((qb >> 4) & 0x0F).astype(np.int32) - 8
    vals = np.empty((O, I), np.int32)
    vals[:, 0::2] = lo[:, :len(range(0, I, 2))]
    vals[:, 1::2] = hi[:, :len(range(1, I, 2))]
    ngroups = (I + gs - 1) // gs
    s = s_flat.reshape(O, ngroups)
    scale_full = np.repeat(s, gs, axis=1)[:, :I]
    return vals.astype(np.float32) * scale_full


def test_transcode_is_bit_exact():
    """The scales fed to transcode_compressed_int4 in production ORIGINATE as bf16 in
    the source checkpoint (unpack_compressed_int4 upcasts them to f32 only so the
    caller has ordinary float arithmetic to work with). To exercise the real lossless
    round-trip -- source bf16 -> f32 (unpack) -> bf16 (transcode) -> f32 (engine
    upcast) -- this test builds `scale` FROM bf16 values, not raw f32 randoms: a scale
    that never was bf16 would make bf16-downcast lossy and the exactness assertion
    below would not actually prove anything about the real pipeline."""
    import numpy as np, ml_dtypes
    rng = np.random.default_rng(2)
    O, I, gs = 4, 64, 32
    codes = rng.integers(-8, 8, size=(O, I)).astype(np.int8)
    # Simulate the source: bf16 scale upcast to f32 by unpack_compressed_int4.
    scale_bf16_src = (rng.random((O, I // gs)).astype(np.float32) + 0.1).astype(ml_dtypes.bfloat16)
    scale = scale_bf16_src.astype(np.float32)                # what unpack_compressed_int4 returns
    qbytes, s_flat = cvt.transcode_compressed_int4(codes, scale)
    assert qbytes.dtype == np.uint8
    assert s_flat.dtype == ml_dtypes.bfloat16, (
        "transcode must emit BF16 scales (matches the source's bf16 precision, "
        "halves .qs bytes vs f32) -- unlike quant_int4/quant_int4_grouped/quant_int8, "
        "which must stay F32")
    assert s_flat.shape == (O * (I // gs),)
    # Engine-side upcast: bf16 -> f32 at load time, then the fmt=4 dequant math.
    s_flat_f32 = s_flat.astype(np.float32)
    deq = _colibri_fmt4_dequant(qbytes, s_flat_f32, O, I, gs)
    # Reference uses the ORIGINAL bf16-sourced scale (upcast to f32), not re-derived --
    # this is the value the engine is supposed to reproduce exactly.
    ref = codes.astype(np.float32) * np.repeat(scale, gs, axis=1)[:, :I]
    assert np.array_equal(deq, ref)          # exact, not approximate: bf16 round-trip is lossless here


def test_transcode_scale_saved_as_bf16_in_safetensors(tmp_path):
    """The whole point of this change: the .qs tensor must be tagged BF16 in the
    safetensors file on disk (not merely a numpy dtype in memory) -- an engine reading
    the container's metadata needs the ON-DISK dtype tag to be BF16, or the halved
    byte count silently misleads it into reading garbage."""
    import ml_dtypes
    from safetensors.numpy import save_file, load_file
    from safetensors import safe_open
    rng = np.random.default_rng(4)
    O, I, gs = 3, 96, 32
    codes = rng.integers(-8, 8, size=(O, I)).astype(np.int8)
    scale_bf16_src = (rng.random((O, I // gs)).astype(np.float32) + 0.1).astype(ml_dtypes.bfloat16)
    scale = scale_bf16_src.astype(np.float32)
    qbytes, s_flat = cvt.transcode_compressed_int4(codes, scale)

    p = str(tmp_path / "t.safetensors")
    save_file({"w.weight": qbytes, "w.weight.qs": s_flat}, p)

    with safe_open(p, framework="numpy") as f:
        assert f.get_slice("w.weight.qs").get_dtype() == "BF16", (
            "the .qs tensor must be tagged BF16 in the safetensors file, not F32/U16")

    reloaded = load_file(p)
    got = reloaded["w.weight.qs"]
    assert got.dtype == ml_dtypes.bfloat16
    assert got.shape == (O * (I // gs),)
    np.testing.assert_array_equal(got.astype(np.float32), s_flat.astype(np.float32))
    assert np.array_equal(reloaded["w.weight"], qbytes)


def test_engine_load_simulation_bf16_qs_group_count():
    """Documents the contract with the engine-side change: given the bf16 `.qs` byte
    count `ns` (bytes, as read off disk), the engine must derive the scale count as
    ns/2 (2 bytes per bf16 scale) -- NOT ns/4 (4 bytes per f32 scale), which is what
    the pre-existing f32-scales code path assumed. Mis-deriving this would silently
    read half the intended number of groups (or overrun the buffer) once .qs switches
    to bf16. This simulates the engine's derivation and checks it recovers exactly
    O * ceil(I/32) scales, then confirms dequant still matches."""
    import ml_dtypes
    rng = np.random.default_rng(5)
    O, I, gs = 5, 160, 32
    ngroups_expected = -(-I // gs)                          # ceil(I/32) = 5
    codes = rng.integers(-8, 8, size=(O, I)).astype(np.int8)
    scale_bf16_src = (rng.random((O, ngroups_expected)).astype(np.float32) + 0.1).astype(ml_dtypes.bfloat16)
    scale = scale_bf16_src.astype(np.float32)
    qbytes, s_flat = cvt.transcode_compressed_int4(codes, scale)

    # Simulate reading the .qs tensor's raw byte count off disk (bf16 = 2 bytes/scale).
    ns_bytes = s_flat.nbytes
    assert s_flat.dtype.itemsize == 2, "bf16 must be 2 bytes/scale on disk"
    engine_derived_scale_count = ns_bytes // 2               # dtype-aware: bf16 -> /2, NOT /4
    assert engine_derived_scale_count == O * ngroups_expected

    s_flat_f32 = s_flat.astype(np.float32)
    deq = _colibri_fmt4_dequant(qbytes, s_flat_f32, O, I, gs)
    ref = codes.astype(np.float32) * np.repeat(scale, gs, axis=1)[:, :I]
    assert np.array_equal(deq, ref)


def test_classify_kimi_names():
    n = 61
    assert cvt.classify("model.layers.5.mlp.experts.3.gate_proj.weight_packed", n) == "x"
    assert cvt.classify("model.layers.5.mlp.experts.3.gate_proj.weight_scale", n) == "consumed"
    assert cvt.classify("model.layers.5.mlp.experts.3.gate_proj.weight_shape", n) == "consumed"
    assert cvt.classify("model.layers.5.self_attn.rotary_emb.inv_freq", n) == "skip"
    # GLM/DeepSeek names must classify EXACTLY as before (regression):
    assert cvt.classify("model.layers.5.mlp.experts.3.gate_proj.weight", n) == "x"
    assert cvt.classify("model.layers.5.self_attn.o_proj.weight", n) == "o"
    assert cvt.classify("model.layers.5.self_attn.kv_b_proj.weight", n) == "kvb"
    assert cvt.classify("model.layers.0.mlp.gate_proj.weight", n) == "dmlp"
    assert cvt.classify("model.layers.5.mlp.shared_experts.up_proj.weight", n) == "sh"
    assert cvt.classify("model.layers.5.mlp.gate.weight", n) == "f32"


# ---------- K2.6: prefix strip THROUGH classify(), vision drop, config flatten ----------
# The ordering trap this section guards against: classify()'s layer_idx logic requires
# p[0]=="model" (a `language_model.` prefix makes it return -1 for EVERY tensor), and its
# embed/lm_head check is an EXACT string match (silently reclassifies to the "q" fallback).
# Neither raises. Every assertion below calls cvt.classify() directly -- never a
# name-mapping helper on the side -- so a regression in classify()'s own prefix handling
# (or in whatever strips the prefix before it) cannot pass silently.


def test_classify_strips_language_model_prefix_matches_unprefixed():
    n = 61
    prefixed = "language_model.model.layers.3.mlp.experts.7.gate_proj.weight"
    bare = "model.layers.3.mlp.experts.7.gate_proj.weight"
    assert cvt.classify(prefixed, n) == cvt.classify(bare, n) == "x"
    # layer_idx must resolve correctly on the STRIPPED name, not -1 (the trap this
    # whole section exists to catch: a `language_model.` prefix makes layer_idx's
    # p[0]=="model" check fail for every tensor).
    assert cvt.layer_idx(cvt.strip_lm_prefix(prefixed)) == 3
    assert cvt.layer_idx(prefixed) == -1, (
        "layer_idx itself is NOT prefix-aware (only classify()/the conversion pipeline "
        "strip before calling it) -- this pins that contract so a future caller of "
        "layer_idx() directly doesn't assume otherwise")


def test_classify_prefixed_embed_and_lm_head_stay_io_not_q_fallback():
    n = 61
    assert cvt.classify("language_model.model.embed_tokens.weight", n) == "io"
    assert cvt.classify("language_model.lm_head.weight", n) == "io"
    # Regression: unprefixed must still classify identically (byte-identical path).
    assert cvt.classify("model.embed_tokens.weight", n) == "io"
    assert cvt.classify("lm_head.weight", n) == "io"


def test_vision_tensors_dropped_and_counted(tmp_path):
    """Vision names are dropped before classify() ever sees them, and the drop is
    counted -- via _shard_tensor_groups' vision_counts param, the actual production
    mechanism, not a side helper that could drift from what conversion really does."""
    import torch
    from safetensors.torch import save_file as save_file_pt

    p = str(tmp_path / "shard.safetensors")
    save_file_pt({
        "language_model.model.layers.0.self_attn.o_proj.weight":
            torch.randn(8, 8).to(torch.bfloat16),
        "vision_tower.encoder.blocks.0.wqkv.weight": torch.randn(8, 8).to(torch.bfloat16),
        "vision_tower.patch_embed.proj.weight": torch.randn(8, 8).to(torch.bfloat16),
        "mm_projector.proj.0.weight": torch.randn(8, 8).to(torch.bfloat16),
    }, p)

    vision_counts = {}
    groups = list(cvt._shard_tensor_groups(p, n_layers=61, ebits=8, io_bits=8, xbits=8,
                                            group_size=64, vision_counts=vision_counts))
    emitted_names = {name for group in groups for name, _ in group}
    assert not any(n.startswith(("vision_tower.", "mm_projector.")) for n in emitted_names)
    assert not any("language_model." in n for n in emitted_names)
    assert any(n.endswith("o_proj.weight") for n in emitted_names)
    assert vision_counts == {"vision_tower": 2, "mm_projector": 1}


def test_vision_drop_category_helper():
    assert cvt.vision_drop_category("vision_tower.encoder.blocks.0.wqkv.weight") == "vision_tower"
    assert cvt.vision_drop_category("mm_projector.proj.0.weight") == "mm_projector"
    assert cvt.vision_drop_category("model.layers.0.self_attn.o_proj.weight") is None
    assert cvt.vision_drop_category("language_model.model.layers.0.self_attn.o_proj.weight") is None


K26_TEXT_CONFIG = {
    "model_type": "kimi_k2", "architectures": ["KimiK2ForCausalLM"],
    "num_hidden_layers": 61, "hidden_size": 7168, "vocab_size": 163840,
    "rope_theta": 50000.0,
    "rope_scaling": {"type": "yarn", "factor": 64.0,
                      "original_max_position_embeddings": 4096,
                      "beta_fast": 32.0, "beta_slow": 1.0,
                      "mscale": 1.0, "mscale_all_dim": 1.0},
    "bos_token_id": 163584, "eos_token_id": 163585, "pad_token_id": 163839,
    "quantization_config": {"format": "pack-quantized", "group_size": 32},
}
K26_NESTED_CONFIG = {
    "model_type": "kimi_k25", "architectures": ["KimiK25ForConditionalGeneration"],
    "text_config": K26_TEXT_CONFIG,
    "vision_config": {"model_type": "moonvit", "hidden_size": 1152},
    "torch_dtype": "bfloat16", "transformers_version": "4.57.0",
}


def test_flatten_container_config_emits_text_config_verbatim():
    flat = cvt.flatten_container_config(K26_NESTED_CONFIG)
    assert flat is K26_TEXT_CONFIG           # verbatim: the SAME object, nothing rebuilt/merged
    assert flat["model_type"] == "kimi_k2"
    assert flat["rope_theta"] == 50000.0
    assert flat["rope_scaling"]["type"] == "yarn"
    assert flat["rope_scaling"]["beta_fast"] == 32.0
    assert "text_config" not in flat and "vision_config" not in flat
    # The second trap: nothing from the TOP level (model_type="kimi_k25" etc.) may
    # leak in. A merge like {**text_config, **top} would restore it.
    assert flat.get("model_type") != "kimi_k25"


def test_flatten_container_config_flat_passthrough_unchanged():
    """A GLM-style flat config (no text_config key) must pass through as the exact
    same object -- the byte-identical regression guard for flat checkpoints."""
    flat_cfg = {"model_type": "kimi_k2", "num_hidden_layers": 61, "rope_theta": 50000.0}
    assert cvt.flatten_container_config(flat_cfg) is flat_cfg


def test_write_metadata_flattens_nested_k26_config(tmp_path):
    """A K2.6-style nested config.json is written to outdir/config.json as text_config's
    content VERBATIM -- nothing added, nothing merged from the top level."""
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"; out.mkdir()
    (src / "config.json").write_text(json.dumps(K26_NESTED_CONFIG))
    cvt._write_metadata(str(src), str(out))
    got = json.loads((out / "config.json").read_text())
    assert got["model_type"] == "kimi_k2"
    assert "text_config" not in got and "vision_config" not in got
    assert got == K26_TEXT_CONFIG


def test_write_config_file_flat_config_copied_byte_for_byte(tmp_path):
    """Flat config (no text_config key): the byte-for-byte shutil.copy path, unchanged."""
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"
    raw = json.dumps({"model_type": "kimi_k2", "num_hidden_layers": 61})
    (src / "config.json").write_text(raw)
    cvt._write_config_file(str(src / "config.json"), str(out))
    assert out.read_text() == raw


def test_write_config_file_null_text_config_is_not_flattened(tmp_path):
    """`"text_config": null` is key-PRESENT but not a dict. The old `"text_config" in cfg`
    guard sent it down the nested branch, flatten returned None, and json.dump wrote the
    literal `null` into the container's config.json -- silently destroying the real config
    (recoverable only by re-converting the whole shard set). It must take the byte-for-byte
    copy path instead, so nothing is lost and the engine fails loudly on the unflattened
    config rather than on a `null` one."""
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"
    raw = json.dumps({"model_type": "kimi_k25", "text_config": None, "num_hidden_layers": 61})
    (src / "config.json").write_text(raw)
    cvt._write_config_file(str(src / "config.json"), str(out))
    assert out.read_text() == raw          # byte-identical, not "null"
    assert json.loads(out.read_text()) is not None


def test_flatten_container_config_null_text_config_passes_through(tmp_path):
    """flatten agrees with c/coli:171 and c/openai_server.py:2224, which both already
    used isinstance: a non-dict text_config is not a nested container."""
    cfg = {"model_type": "kimi_k25", "text_config": None}
    assert cvt.flatten_container_config(cfg) is cfg
    for bad in ([], "x", 3):
        assert cvt.flatten_container_config({"text_config": bad}) == {"text_config": bad}


def test_write_metadata_flat_config_byte_identical(tmp_path):
    """Flat-config (GLM) regression: a flat config.json (no text_config) must be copied
    through BYTE-FOR-BYTE (shutil.copy), not round-tripped through json.dump -- which
    could silently reorder keys or change whitespace even when the data is equal."""
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"; out.mkdir()
    raw = '{  "model_type":   "kimi_k2", "num_hidden_layers": 61 }'   # deliberately odd spacing
    (src / "config.json").write_text(raw)
    cvt._write_metadata(str(src), str(out))
    assert (out / "config.json").read_text() == raw


def test_read_n_layers_from_config_nested_k26(tmp_path):
    """read_n_layers_from_config must resolve n_layers from text_config when the flat
    key is absent (K2.6's nested shape) -- otherwise a K2.6 --indir/--repo source keeps
    --n-layers' 78 default instead of the real 61, silently."""
    d = tmp_path / "k26"; d.mkdir()
    (d / "config.json").write_text(json.dumps(K26_NESTED_CONFIG))
    assert cvt.read_n_layers_from_config(str(d)) == 61

    d2 = tmp_path / "flat"; d2.mkdir()
    (d2 / "config.json").write_text(json.dumps({"num_hidden_layers": 61}))
    assert cvt.read_n_layers_from_config(str(d2)) == 61


def test_classify_rejects_stray_weight_packed():
    """Kimi K2 checkpoints only int4-quantize routed experts (config `ignore` keeps
    shared_experts/attention/dense-mlp/lm_head in bf16), so a `.weight_packed`
    tensor should only ever appear under `.mlp.experts.`. A stray one elsewhere
    must fail loud, not silently fall through to f32 and get misread as raw
    float32 by dequant()."""
    with pytest.raises(SystemExit):
        cvt.classify("model.layers.5.self_attn.q_a_proj.weight_packed", 61)


# ---------- arch detection + tokenizer wiring + end-to-end --indir smoke ----------
# These need a real K2 checkpoint on disk; point COLI_K2_SRC at one to enable them.
import subprocess, glob

K2_DIR = os.environ.get("COLI_K2_SRC", "")
K2_SHARD0 = os.path.join(K2_DIR, "model-00001-of-000062.safetensors") if K2_DIR else ""


@pytest.mark.skipif(not (K2_DIR and os.path.exists(os.path.join(K2_DIR, "config.json"))),
                    reason="needs a local K2 checkpoint (set COLI_K2_SRC)")
def test_read_n_layers_from_config_kimi():
    assert cvt.read_n_layers_from_config(K2_DIR) == 61


@pytest.mark.skipif(not (K2_SHARD0 and os.path.exists(K2_SHARD0)),
                    reason="needs local K2 shard 0 (set COLI_K2_SRC)")
def test_indir_convert_shard0_container(tmp_path):
    """Convert the local shard 0 (layer 0, dense, all bf16) end-to-end via --indir.
    No experts here, so this exercises the bf16->int8 resident path + metadata +
    tokenizer generation, not the expert transcode (covered separately).

    Passing the checkpoint directory straight to --indir would convert every shard
    present (hundreds of GB, hours) instead of staying a fast smoke test. So this stages
    a throwaway dir with ONLY shard 0 (+ config.json + whichever tokenizer files happen
    to be present) symlinked in, and points --indir at that. The source directory is
    never written to."""
    stage = tmp_path / "k2_stage"
    stage.mkdir()
    os.symlink(K2_SHARD0, stage / "model-00001-of-000062.safetensors")
    os.symlink(os.path.join(K2_DIR, "config.json"), stage / "config.json")
    have_tiktoken = os.path.exists(os.path.join(K2_DIR, "tiktoken.model"))
    for fn in ("tiktoken.model", "tokenizer_config.json", "generation_config.json"):
        src = os.path.join(K2_DIR, fn)
        if os.path.exists(src):
            os.symlink(src, stage / fn)

    out = str(tmp_path / "k2_i4")
    # --ebits deliberately OMITTED, on real data: a
    # pack-quantized checkpoint with no --ebits must default to int8 residents (the
    # converter's lossless default), not the int4 the launcher used to force via a
    # hardcoded --ebits 4. No --arch flag either: the converter decides int8-vs-int4
    # purely by inspecting the shard's own compressed-tensors config, not an arch name.
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "convert_fp8_to_int4.py"),
                        "--indir", str(stage), "--outdir", out,
                        "--io-bits", "8", "--group-size", "64"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    shards = glob.glob(os.path.join(out, "out-*.safetensors"))
    assert shards, "no container shards emitted"
    # tokenizer.json must be generated IFF the source dir actually has tiktoken.model.
    # K2's local mirror is mid-download and may not have re-fetched it yet -- in that
    # case the converter must warn and skip, not crash (this is the branch the
    # critical notes call out explicitly).
    tok_out = os.path.join(out, "tokenizer.json")
    if have_tiktoken:
        assert os.path.exists(tok_out), "tokenizer.json not generated"
    else:
        assert not os.path.exists(tok_out)
        assert "tiktoken.model" in r.stdout and "WARNING" in r.stdout
    # config.json copied through
    assert os.path.exists(os.path.join(out, "config.json"))
    # a resident int8 tensor is present with its .qs, and inv_freq was skipped
    from safetensors.numpy import load_file
    tensors = {}
    for sp in shards: tensors.update(load_file(sp))
    keys = set(tensors)
    assert any(k.endswith("o_proj.weight") for k in keys)
    assert any(k.endswith("o_proj.weight.qs") for k in keys)
    assert not any("inv_freq" in k for k in keys), "inv_freq should be skipped"
    # o_proj must be packed as INT8 (one byte per weight), not
    # int4 (one byte per TWO weights). Read O,I from the source shard's own o_proj
    # to avoid hardcoding K2's dims, then check the emitted byte length directly --
    # int4 packing (O*ceil(I/2) bytes) is half int8's (O*I bytes), so the two can't
    # be confused for each other.
    from safetensors import safe_open
    o_key = next(k for k in keys if k.endswith("o_proj.weight"))
    with safe_open(K2_SHARD0, framework="pt") as f:
        src_key = next(n for n in f.keys() if n.endswith("o_proj.weight"))
        O, I = f.get_slice(src_key).get_shape()
    qbytes = tensors[o_key]
    assert qbytes.size == O * I, (
        f"o_proj.weight is {qbytes.size} bytes, expected {O*I} = O*I (int8); "
        f"O*ceil(I/2)={O*((I+1)//2)} would mean it was quantized to int4 instead")


def test_write_metadata_warns_on_missing_glm_tokenizer(tmp_path, capsys):
    """Regression: the GLM-style --indir metadata step must still warn when
    tokenizer.json (or any of the 4 standard metadata files) is absent from the source
    dir, exactly like the old inline copy loop did. A prior refactor into
    _write_metadata() only printed the missing-file warning inside an arch-gated
    branch, silently dropping it for other conversions -- the current, file-driven
    version applies the warning uniformly, regardless of arch."""
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"; out.mkdir()
    (src / "config.json").write_text("{}")
    # tokenizer.json, tokenizer_config.json, generation_config.json all absent
    cvt._write_metadata(str(src), str(out))
    captured = capsys.readouterr()
    assert "WARNING: not found in" in captured.out
    assert "tokenizer.json" in captured.out
    assert "chat/serve need tokenizer.json" in captured.out
    # config.json WAS copied, and nothing crashed
    assert os.path.exists(out / "config.json")


def test_write_metadata_kimi_generated_tokenizer_not_reported_missing(tmp_path, capsys):
    """When kimi tokenizer generation succeeds, tokenizer.json must be reported as
    generated, NOT also listed in the missing-files warning -- even though another,
    genuinely-missing metadata file (generation_config.json, left absent here) still
    gets warned about."""
    import base64
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"; out.mkdir()
    (src / "config.json").write_text("{}")
    tok_a = base64.b64encode(b"a").decode(); tok_b = base64.b64encode(b"b").decode()
    (src / "tiktoken.model").write_text(f"{tok_a} 0\n{tok_b} 1\n")
    (src / "tokenizer_config.json").write_text("{}")
    # generation_config.json intentionally left absent -> still expected in the warning
    cvt._write_metadata(str(src), str(out))
    captured = capsys.readouterr()
    assert os.path.exists(out / "tokenizer.json")
    assert "tokenizer.json(generated)" in captured.out
    warning_lines = [l for l in captured.out.splitlines() if "WARNING: not found in" in l]
    assert len(warning_lines) == 1
    assert "generation_config.json" in warning_lines[0]
    assert "tokenizer.json" not in warning_lines[0].split(":", 2)[-1]   # not in the file list itself
    assert "chat/serve need tokenizer.json" not in warning_lines[0]     # only appended when tokenizer.json IS missing


def test_write_metadata_copies_existing_tokenizer_not_regenerated(tmp_path, capsys):
    """FILE-driven generation gate (not arch-driven): when the source already ships
    its own tokenizer.json (GLM-style), it must be copied through verbatim and NEVER
    regenerated from tiktoken -- even when tiktoken.model also happens to be present
    in the source dir. This is the GLM-identical guarantee: GLM ships tokenizer.json,
    so this path is a plain copy, exactly as it always was."""
    import base64, json
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"; out.mkdir()
    (src / "config.json").write_text("{}")
    (src / "tokenizer.json").write_text(json.dumps({"marker": "source-tokenizer"}))
    tok_a = base64.b64encode(b"a").decode(); tok_b = base64.b64encode(b"b").decode()
    (src / "tiktoken.model").write_text(f"{tok_a} 0\n{tok_b} 1\n")
    (src / "tokenizer_config.json").write_text("{}")
    cvt._write_metadata(str(src), str(out))
    captured = capsys.readouterr()
    assert "tokenizer.json(generated)" not in captured.out
    got = json.loads((out / "tokenizer.json").read_text())
    assert got == {"marker": "source-tokenizer"}, "existing tokenizer.json must be copied verbatim, not regenerated"


def test_ebits_default_resolves_from_source(tmp_path):
    """No --ebits: the converter resolves the default from the source config.json --
    4 for a plain/fp8 source (GLM's historical int4), 8 for a pack-quantized one."""
    plain = tmp_path / "plain_src"; plain.mkdir()
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "convert_fp8_to_int4.py"),
                        "--indir", str(plain), "--outdir", str(tmp_path / "plain_out")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "experts 4-bit" in r.stdout, r.stdout

    k26 = tmp_path / "k26_src"; k26.mkdir()
    (k26 / "config.json").write_text(json.dumps(
        {"num_hidden_layers": 2, "quantization_config": {"format": "pack-quantized"}}))
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "convert_fp8_to_int4.py"),
                        "--indir", str(k26), "--outdir", str(tmp_path / "k26_out")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "experts 8-bit" in r.stdout, r.stdout


# ---------- launcher `coli convert` K2 path ----------
import types


def _load_coli():
    import importlib.machinery
    # `coli` has no .py extension: spec_from_file_location can't infer a loader from
    # the suffix and returns None unless one is passed explicitly.
    path = os.path.join(os.path.dirname(__file__), "..", "coli")
    loader = importlib.machinery.SourceFileLoader("coli", path)
    spec = importlib.util.spec_from_file_location("coli", path, loader=loader)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _convert_args(outdir, **over):
    d = dict(repo="zai-org/GLM-5.2-FP8", model=str(outdir),
             ebits=None, io_bits=8, xbits=0, group_size=64, no_mtp=False,
             indir=None, outdir=None)
    d.update(over)
    return types.SimpleNamespace(**d)


def _run_cmd_convert(coli, monkeypatch, a):
    """Run cmd_convert with subprocess.call mocked; return (recorded calls, exit code)."""
    calls = []
    monkeypatch.setattr(coli.subprocess, "call", lambda cmd, **k: (calls.append(cmd), 0)[1])
    with pytest.raises(SystemExit) as e:
        coli.cmd_convert(a)
    return calls, e.value.code


def test_cmd_convert_no_mtp_in_outdir_config_single_pass_no_ebits(monkeypatch, tmp_path):
    """K2.6 flow: pass 1 writes config.json into outdir; num_nextn_predict_layers=0
    (read through flatten_config, nested here) stops after one pass. --ebits is omitted
    entirely when the user didn't set it -- the converter resolves its own default."""
    coli = _load_coli()
    out = tmp_path / "out"; out.mkdir()
    (out / "config.json").write_text(json.dumps(
        {"model_type": "kimi_k25",
         "text_config": {"model_type": "kimi_k2", "num_nextn_predict_layers": 0}}))
    calls, code = _run_cmd_convert(coli, monkeypatch, _convert_args(out))
    assert code == 0
    assert len(calls) == 1, f"no-MTP convert must be a single pass, got {len(calls)}"
    assert "--ebits" not in calls[0], "launcher must not inject --ebits unless the user set it"
    assert "--mtp" not in calls[0]


def test_cmd_convert_mtp_in_outdir_config_runs_two_passes(monkeypatch, tmp_path):
    """GLM flow: outdir config with num_nextn_predict_layers>0 triggers the MTP pass,
    which forces --ebits 8; the main pass still carries no injected --ebits."""
    coli = _load_coli()
    out = tmp_path / "out"; out.mkdir()
    (out / "config.json").write_text(json.dumps(
        {"model_type": "glm_moe_dsa", "num_nextn_predict_layers": 1}))
    calls, code = _run_cmd_convert(coli, monkeypatch, _convert_args(out))
    assert code == 0
    assert len(calls) == 2, f"glm convert must be two passes, got {len(calls)}"
    assert "--ebits" not in calls[0]
    assert "--mtp" not in calls[0]
    joined1 = " ".join(calls[1])
    assert "--mtp" in calls[1]
    assert "--ebits 8" in joined1, "MTP pass must force int8 even with --ebits omitted from base"


def test_cmd_convert_explicit_ebits_passed_through_and_raised_for_mtp(monkeypatch, tmp_path):
    """--ebits set by the user is forwarded verbatim on pass 1 and raised to >=8 on the
    MTP pass."""
    coli = _load_coli()
    out = tmp_path / "out"; out.mkdir()
    (out / "config.json").write_text(json.dumps({"num_nextn_predict_layers": 1}))
    calls, code = _run_cmd_convert(coli, monkeypatch, _convert_args(out, ebits=6))
    assert code == 0
    assert len(calls) == 2
    assert "--ebits 6" in " ".join(calls[0])
    joined1 = " ".join(calls[1])
    assert "--mtp" in calls[1] and "--ebits 8" in joined1


def test_cmd_convert_missing_outdir_config_fails_soft_to_mtp_pass(monkeypatch, tmp_path):
    """Unreadable outdir config.json -> has_mtp=True (GLM-safe): a failed read must
    never silently skip GLM's MTP pass."""
    coli = _load_coli()
    out = tmp_path / "out"; out.mkdir()          # no config.json written
    calls, code = _run_cmd_convert(coli, monkeypatch, _convert_args(out))
    assert code == 0
    assert len(calls) == 2, "fail-soft must keep GLM's two-pass flow"
    assert "--mtp" in calls[1]


def test_cmd_convert_no_mtp_flag_is_single_pass(monkeypatch, tmp_path):
    """--no-mtp exits after pass 1 without even reading the outdir config."""
    coli = _load_coli()
    out = tmp_path / "out"; out.mkdir()          # no config.json needed
    calls, code = _run_cmd_convert(coli, monkeypatch, _convert_args(out, no_mtp=True))
    assert code == 0
    assert len(calls) == 1
    assert "--mtp" not in calls[0]


def test_cmd_convert_feature_detection_uses_no_network(monkeypatch, tmp_path):
    """The MTP decision reads <outdir>/config.json locally; any urllib call during
    cmd_convert is a regression to the removed HF preflight."""
    coli = _load_coli()
    import urllib.request
    def _boom(*a, **k):
        raise AssertionError("cmd_convert must not touch the network")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(urllib.request, "urlretrieve", _boom, raising=False)
    out = tmp_path / "out"; out.mkdir()
    (out / "config.json").write_text(json.dumps({"num_nextn_predict_layers": 0}))
    calls, code = _run_cmd_convert(coli, monkeypatch,
                                   _convert_args(out, repo="moonshotai/Kimi-K2.6"))
    assert code == 0
    assert len(calls) == 1


# ---------- converter-side --ebits default resolution ----------


def test_resolve_default_ebits_plain_source_is_4(tmp_path):
    """fp8/plain source (no pack-quantized quantization_config): GLM's historical 4."""
    d = tmp_path / "src"; d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    a = types.SimpleNamespace(mtp=False, indexer=False, indir=str(d), repo=None, outdir=None)
    assert cvt._resolve_default_ebits(a) == 4


def test_resolve_default_ebits_mtp_or_indexer_is_8():
    for over in ({"mtp": True, "indexer": False}, {"mtp": False, "indexer": True}):
        a = types.SimpleNamespace(indir=None, repo=None, outdir=None, **over)
        assert cvt._resolve_default_ebits(a) == 8


def test_resolve_default_ebits_pack_quantized_source_is_8(tmp_path):
    d = tmp_path / "src"; d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"quantization_config": {"format": "pack-quantized"}}))
    a = types.SimpleNamespace(mtp=False, indexer=False, indir=str(d), repo=None, outdir=None)
    assert cvt._resolve_default_ebits(a) == 8


def test_source_is_pack_quantized_reads_indir_flat_and_nested(tmp_path):
    flat = tmp_path / "flat"; flat.mkdir()
    (flat / "config.json").write_text(json.dumps(
        {"quantization_config": {"format": "pack-quantized"}}))
    assert cvt._source_is_pack_quantized(
        types.SimpleNamespace(indir=str(flat), repo=None, outdir=None)) is True
    nested = tmp_path / "nested"; nested.mkdir()
    (nested / "config.json").write_text(json.dumps(K26_NESTED_CONFIG))
    assert cvt._source_is_pack_quantized(
        types.SimpleNamespace(indir=str(nested), repo=None, outdir=None)) is True


def test_source_is_pack_quantized_missing_or_malformed_is_false_without_network(tmp_path, monkeypatch):
    """Missing, malformed, or quantization-free config -> False, no raise, no network."""
    import urllib.request
    def _boom(*a, **k):
        raise AssertionError("no network allowed")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = _boom
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    missing = tmp_path / "missing"; missing.mkdir()
    assert cvt._source_is_pack_quantized(
        types.SimpleNamespace(indir=str(missing), repo=None, outdir=None)) is False
    bad = tmp_path / "bad"; bad.mkdir()
    (bad / "config.json").write_text("not json")
    assert cvt._source_is_pack_quantized(
        types.SimpleNamespace(indir=str(bad), repo=None, outdir=None)) is False
    fp8 = tmp_path / "fp8"; fp8.mkdir()
    (fp8 / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    assert cvt._source_is_pack_quantized(
        types.SimpleNamespace(indir=str(fp8), repo=None, outdir=None)) is False


def test_source_is_pack_quantized_repo_mode_peeks_local_meta_then_outdir(tmp_path, monkeypatch):
    """--repo mode reads the copy already on disk in <outdir>/_meta or <outdir> before
    ever considering a download (hf_hub_download stubbed to fail loudly here)."""
    def _boom(*a, **k):
        raise AssertionError("must peek the local copy, not download")
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = _boom
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    pq = json.dumps({"quantization_config": {"format": "pack-quantized"}})
    out_meta = tmp_path / "out_meta"; (out_meta / "_meta").mkdir(parents=True)
    (out_meta / "_meta" / "config.json").write_text(pq)
    assert cvt._source_is_pack_quantized(types.SimpleNamespace(
        indir=None, repo="moonshotai/Kimi-K2.6", outdir=str(out_meta))) is True

    out_root = tmp_path / "out_root"; out_root.mkdir()
    (out_root / "config.json").write_text(pq)
    assert cvt._source_is_pack_quantized(types.SimpleNamespace(
        indir=None, repo="moonshotai/Kimi-K2.6", outdir=str(out_root))) is True


# ---------- Parallel + memory-bounded --indir conversion (this task) ----------
# Correctness is the whole point of this feature: parallel+chunked --indir output must
# be BIT-IDENTICAL to the old single-process, whole-shard-in-RAM output. Every test below
# compares the UNION of tensors across whatever out-*.safetensors files a run produced --
# never individual filenames -- because deterministic-by-shard-index chunk naming means
# --jobs 1 and --jobs N legitimately produce different numbers of files for the same
# input (chunk count depends only on --chunk-gb), while the union of (name, dtype, bytes)
# triples must match exactly.
import time as _time


def _run_convert(indir, outdir, extra_args, timeout=180):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "convert_fp8_to_int4.py"),
                        "--indir", str(indir), "--outdir", str(outdir)] + extra_args,
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return r


def _load_union(outdir):
    """Load every out-*.safetensors chunk file in outdir and merge into one dict,
    asserting no tensor NAME appears in more than one chunk file (the engine loads by
    name across the whole glob, so a name colliding across two chunk files would be a
    real bug -- silently last-write-wins would hide it)."""
    import ml_dtypes                       # noqa: F401 -- registers bfloat16 with numpy (transcode's .qs is bf16)
    from safetensors.numpy import load_file
    files = sorted(glob.glob(os.path.join(outdir, "out-*.safetensors")))
    assert files, f"no out-*.safetensors produced in {outdir}"
    merged = {}
    for fp in files:
        for name, arr in load_file(fp).items():
            assert name not in merged, f"tensor {name!r} appears in more than one chunk file in {outdir}"
            merged[name] = arr
    return merged, files


def _assert_unions_identical(a, b, a_label="A", b_label="B"):
    ak, bk = set(a), set(b)
    assert ak == bk, f"tensor name sets differ: only in {a_label}: {ak-bk}; only in {b_label}: {bk-ak}"
    for k in ak:
        va, vb = a[k], b[k]
        assert va.dtype == vb.dtype, f"{k}: dtype differs ({va.dtype} vs {vb.dtype})"
        assert va.shape == vb.shape, f"{k}: shape differs ({va.shape} vs {vb.shape})"
        assert np.array_equal(va, vb), f"{k}: VALUES differ between {a_label} and {b_label}"


def _build_synth_indir(root, n_layers=4, O=768, I=768, seed=0, with_expert=True):
    """Build a throwaway --indir: N safetensors shards (one per layer), each holding a
    resident bf16 o_proj (quantized path, kind='o'), a resident bf16 dense mlp up_proj
    (kind='dmlp'), and an f32-kept input_layernorm (kind='f32'). The LAST layer's shard
    additionally carries one K2-style compressed-tensors pack-quantized routed expert
    tensor (weight_packed + weight_scale + weight_shape, kind='x') when with_expert=True,
    so the lossless-transcode branch (the most novel, easiest-to-get-wrong code path in
    this converter) is exercised under parallel + chunked conversion too, not just the
    plain dequant->quantize path. Returns the directory path (root itself)."""
    import torch
    from safetensors.torch import save_file as save_file_pt
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"num_hidden_layers": n_layers}, f)
    for li in range(n_layers):
        w_o = (rng.standard_normal((O, I)).astype(np.float32) * 0.02)
        w_up = (rng.standard_normal((O, I)).astype(np.float32) * 0.02)
        norm = rng.standard_normal((O,)).astype(np.float32)
        tens = {
            f"model.layers.{li}.self_attn.o_proj.weight": torch.from_numpy(w_o).to(torch.bfloat16),
            f"model.layers.{li}.mlp.up_proj.weight": torch.from_numpy(w_up).to(torch.bfloat16),
            f"model.layers.{li}.input_layernorm.weight": torch.from_numpy(norm).to(torch.bfloat16),
        }
        if with_expert and li == n_layers - 1:
            Oe, Ie, gs = 64, 256, 32
            codes = rng.integers(-8, 8, size=(Oe, Ie)).astype(np.int8)
            scale = (rng.random((Oe, Ie // gs)).astype(np.float32) + 0.1)
            packed = _pack_int4_columns(codes)
            base = f"model.layers.{li}.mlp.experts.0.gate_proj"
            tens[base + ".weight_packed"] = torch.from_numpy(packed)
            tens[base + ".weight_scale"] = torch.from_numpy(scale)
            tens[base + ".weight_shape"] = torch.tensor([Oe, Ie], dtype=torch.int64)
        save_file_pt(tens, os.path.join(root, f"model-{li:05d}-of-{n_layers:05d}.safetensors"))
    return str(root)


def test_parallel_matches_sequential_synthetic(tmp_path):
    """THE core correctness proof for this task: convert the SAME --indir twice --
    once --jobs 1 --chunk-gb 100 (one chunk per shard, today's original --indir
    behavior) and once --jobs 3 --chunk-gb tiny (forces MULTIPLE chunk files per shard,
    processed by 3 spawned worker processes) -- and assert the union of every tensor
    (name, dtype, shape, bytes) is IDENTICAL between the two runs. This is what proves
    parallelism + chunked flushing changes nothing about the actual conversion output."""
    indir = _build_synth_indir(tmp_path / "src", n_layers=4, O=768, I=768)
    out_seq = tmp_path / "out_seq"; out_par = tmp_path / "out_par"

    _run_convert(indir, out_seq,
                 ["--ebits", "4", "--io-bits", "8", "--group-size", "64",
                  "--jobs", "1", "--chunk-gb", "100"])
    r_par = _run_convert(indir, out_par,
                         ["--ebits", "4", "--io-bits", "8", "--group-size", "64",
                          "--jobs", "3", "--chunk-gb", "0.0003"])

    seq_tensors, seq_files = _load_union(out_seq)
    par_tensors, par_files = _load_union(out_par)

    # Sanity on the test SETUP itself, not just the outcome: --jobs 1 --chunk-gb 100
    # must be exactly one chunk per shard (4 files), and the tiny chunk-gb run must have
    # produced MORE files than shards -- otherwise this test would pass trivially
    # without ever exercising multi-chunk output.
    assert len(seq_files) == 4, f"expected 1 file/shard at chunk-gb 100, got {seq_files}"
    assert len(par_files) > 4, f"expected multiple chunks/shard at chunk-gb 0.0003, got {par_files}"
    assert "--jobs 3" in r_par.stdout, r_par.stdout   # confirms the resolved job count, not just the flag we passed

    _assert_unions_identical(seq_tensors, par_tensors, "sequential(--jobs 1)", "parallel(--jobs 3)")
    # The transcoded expert tensor specifically (the fragile lossless path) must be
    # present and identical too -- _assert_unions_identical already covers it since it's
    # part of the union, this just documents that the test setup does include it.
    assert any(k.endswith(".mlp.experts.0.gate_proj.weight") for k in seq_tensors)
    assert any(k.endswith(".mlp.experts.0.gate_proj.weight.qs") for k in seq_tensors)


@pytest.mark.skipif(not (K2_SHARD0 and os.path.exists(K2_SHARD0)),
                    reason="needs local K2 shard 0 (set COLI_K2_SRC)")
def test_parallel_matches_sequential_real_k2_shard(tmp_path):
    """Same proof as test_parallel_matches_sequential_synthetic, but against REAL K2
    shard 0 (read-only symlink; the source checkpoint is never written to) instead of synthetic
    data -- shard 0 is small (~1 GB, dense layer-0 tensors only, no routed experts) so
    this stays fast while still validating against a real checkpoint's actual bf16
    tensors and byte layout, not just synthetic ones."""
    stage = tmp_path / "k2_stage"; stage.mkdir()
    os.symlink(K2_SHARD0, stage / "model-00001-of-000062.safetensors")
    os.symlink(os.path.join(K2_DIR, "config.json"), stage / "config.json")

    out_seq = tmp_path / "k2_out_seq"; out_par = tmp_path / "k2_out_par"
    _run_convert(stage, out_seq, ["--io-bits", "8", "--group-size", "64",
                                  "--jobs", "1", "--chunk-gb", "100"])
    _run_convert(stage, out_par, ["--io-bits", "8", "--group-size", "64",
                                  "--jobs", "2", "--chunk-gb", "0.01"])

    seq_tensors, seq_files = _load_union(out_seq)
    par_tensors, par_files = _load_union(out_par)
    assert len(par_files) > len(seq_files), "small chunk-gb should split real shard 0 into multiple chunk files"
    _assert_unions_identical(seq_tensors, par_tensors, "sequential(real K2)", "parallel(real K2)")


def test_resume_reprocesses_only_deleted_shard(tmp_path):
    """Delete one shard's `.done` marker + its chunk file(s), rerun with the SAME
    --outdir/params, and confirm: (a) the resume skip-count covers every OTHER shard,
    (b) only the deleted shard's input is reprocessed, and (c) the resulting union is
    IDENTICAL to a fully-fresh from-scratch conversion of the same --indir -- i.e. resume
    doesn't just re-run the missing shard, it reproduces the exact same final container."""
    indir = _build_synth_indir(tmp_path / "src", n_layers=3, O=512, I=512, with_expert=False)
    out = tmp_path / "out"
    extra = ["--ebits", "4", "--io-bits", "8", "--group-size", "64",
             "--jobs", "2", "--chunk-gb", "0.0005"]
    _run_convert(indir, out, extra)

    before_tensors, before_files = _load_union(out)
    last_idx = 2   # n_layers=3 -> shard indices 0,1,2
    last_marker = os.path.join(out, f"out-{last_idx:05d}.done")
    last_chunks = glob.glob(os.path.join(out, f"out-{last_idx:05d}-*.safetensors"))
    assert os.path.exists(last_marker) and last_chunks, "setup: last shard should already be done"
    os.remove(last_marker)
    for c in last_chunks: os.remove(c)

    r = _run_convert(indir, out, extra)
    assert "[RESUME] 2 shard(s) already done" in r.stdout, r.stdout
    assert f"shard {last_idx:05d}" in r.stdout, r.stdout   # the reprocessed one, named in the progress line

    after_tensors, after_files = _load_union(out)
    _assert_unions_identical(before_tensors, after_tensors, "before-delete", "after-resume")

    # Independent from-scratch reference: resume's final state must match a clean run.
    out_fresh = tmp_path / "out_fresh"
    _run_convert(indir, out_fresh, extra)
    fresh_tensors, _ = _load_union(out_fresh)
    _assert_unions_identical(after_tensors, fresh_tensors, "after-resume", "fresh")


def test_resume_removes_stale_leftover_chunks_from_incomplete_attempt(tmp_path):
    """A shard reprocessed after an INCOMPLETE prior attempt (marker missing, but some of
    that attempt's chunk files were left behind on disk -- e.g. the process was killed
    mid-shard) must not leave those stale chunk files sitting alongside the fresh ones:
    duplicate copies of the same tensor NAMES across two files for the same shard index
    is exactly the hazard the deterministic-by-shard-index naming scheme exists to avoid.
    This simulates that by hand-crafting an extra, bogus leftover chunk file at the same
    shard index with one MORE chunk than the real run will ever produce, deleting only
    the `.done` marker (not that bogus file), and checking it's gone after the rerun."""
    indir = _build_synth_indir(tmp_path / "src", n_layers=2, O=512, I=512, with_expert=False)
    out = tmp_path / "out"
    extra = ["--ebits", "4", "--io-bits", "8", "--group-size", "64",
             "--jobs", "1", "--chunk-gb", "0.0005"]
    _run_convert(indir, out, extra)

    idx = 0
    real_chunks = sorted(glob.glob(os.path.join(out, f"out-{idx:05d}-*.safetensors")))
    assert real_chunks, "setup: shard 0 should have produced at least one chunk"
    next_chunk_num = len(real_chunks)   # one past the last real chunk index for this shard
    bogus = os.path.join(out, f"out-{idx:05d}-{next_chunk_num:03d}.safetensors")
    from safetensors.numpy import save_file
    save_file({"bogus.leftover.tensor": np.zeros(4, np.float32)}, bogus)
    assert os.path.exists(bogus)

    os.remove(os.path.join(out, f"out-{idx:05d}.done"))   # marker gone -> shard 0 looks incomplete, gets reprocessed
    _run_convert(indir, out, extra)

    assert not os.path.exists(bogus), "stale leftover chunk from an incomplete attempt must be removed on reprocess"
    tensors, _ = _load_union(out)
    assert "bogus.leftover.tensor" not in tensors


def test_chunk_gb_bounds_worker_memory(tmp_path):
    """Memory-bound proof: with a small --chunk-gb, ONE shard's total output (several
    o_proj-sized tensors, summed) must be split into MULTIPLE chunk files, and each
    chunk file's raw tensor payload (sum of ndarray.nbytes, i.e. what a worker actually
    holds in RAM right before flushing -- independent of safetensors' own on-disk
    header/padding overhead) must stay within chunk_bytes plus one tensor GROUP's worth
    of slack (a group = one weight + its .qs scale, flushed together, see
    _shard_tensor_groups) -- never anywhere near the shard's full output size. This is
    what --chunk-gb is FOR: bounding a worker's peak RSS regardless of shard size."""
    n_layers = 8
    indir = _build_synth_indir(tmp_path / "src", n_layers=n_layers, O=512, I=512, with_expert=False)
    # 0.0001 GB (~107 KB) is smaller than a SINGLE o_proj/up_proj group (~144 KB, computed
    # below) -- so the very first group flushes its own chunk, guaranteeing multiple
    # chunk files per shard rather than depending on cumulative totals crossing the
    # threshold only once at the end.
    chunk_gb = 0.0001
    chunk_bytes = chunk_gb * (1 << 30)
    out = tmp_path / "out"
    _run_convert(indir, out, ["--ebits", "4", "--io-bits", "8", "--group-size", "64",
                              "--jobs", "1", "--chunk-gb", str(chunk_gb)])

    import ml_dtypes  # noqa: F401 -- registers bfloat16 with numpy, harmless even though this shard has none
    from safetensors.numpy import load_file
    # Largest single group in this setup: one o_proj (512x512 int4-grouped) + its .qs,
    # or the up_proj equivalent -- compute the real upper bound instead of guessing.
    max_group_bytes = 512 * ((512 + 1) // 2) + 512 * ((512 + 63) // 64) * 4  # qbytes + qs(f32)
    total_all_chunks = 0
    n_multi_chunk_shards = 0
    for li in range(n_layers):
        chunk_files = sorted(glob.glob(os.path.join(out, f"out-{li:05d}-*.safetensors")))
        assert chunk_files, f"shard {li} produced no chunk files"
        if len(chunk_files) > 1: n_multi_chunk_shards += 1
        for fp in chunk_files:
            payload_bytes = sum(a.nbytes for a in load_file(fp).values())
            total_all_chunks += payload_bytes
            assert payload_bytes <= chunk_bytes + max_group_bytes, (
                f"{fp}: chunk payload {payload_bytes} bytes exceeds the "
                f"chunk-gb bound ({chunk_bytes:.0f} + one group {max_group_bytes}) -- "
                "--chunk-gb failed to bound worker memory")
    assert n_multi_chunk_shards > 0, "chunk-gb 0.0004 should have split at least one shard into multiple files"
    assert chunk_bytes < total_all_chunks / n_layers, (
        "chunk-gb is not actually smaller than a shard's total output in this test setup "
        "-- the memory-bound assertion above would be vacuous")


def test_jobs_1_matches_jobs_1_default_chunk(tmp_path):
    """--jobs 1 with the DEFAULT --chunk-gb (2.0, i.e. no explicit override) must still
    convert every shard correctly and match a --chunk-gb 100 (unbounded) run byte-for-
    byte -- proving the default chunk size doesn't change conversion correctness even
    though these particular tiny synthetic shards are far smaller than 2 GB (so the
    default in practice yields one chunk per shard here, same as --chunk-gb 100)."""
    indir = _build_synth_indir(tmp_path / "src", n_layers=2, O=256, I=256, with_expert=False)
    out_default = tmp_path / "out_default"; out_unbounded = tmp_path / "out_unbounded"
    _run_convert(indir, out_default, ["--ebits", "4", "--io-bits", "8", "--group-size", "64", "--jobs", "1"])
    _run_convert(indir, out_unbounded, ["--ebits", "4", "--io-bits", "8", "--group-size", "64",
                                        "--jobs", "1", "--chunk-gb", "100"])
    a, _ = _load_union(out_default); b, _ = _load_union(out_unbounded)
    _assert_unions_identical(a, b, "jobs1-default-chunk", "jobs1-unbounded-chunk")


def test_refuses_to_resume_old_style_progress_json(tmp_path):
    """An outdir carrying a pre-parallel converter's `.out-progress.json` (the OLD
    global-emission-counter resume manifest, incompatible with this version's per-shard
    `.done` markers and shard-index-based chunk naming) must be refused outright, not
    silently reprocessed under the new scheme -- which would leave the old counter-named
    files (e.g. out-00000.safetensors) behind unnoticed while writing fresh
    out-00000-000.safetensors alongside them, duplicating every tensor name across both
    when the engine globs out-*.safetensors."""
    indir = _build_synth_indir(tmp_path / "src", n_layers=1, O=128, I=128, with_expert=False)
    out = tmp_path / "out"; out.mkdir()
    (out / ".out-progress.json").write_text(json.dumps({"params": {}, "shards": {}}))
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "convert_fp8_to_int4.py"),
                        "--indir", str(indir), "--outdir", str(out),
                        "--ebits", "4", "--io-bits", "8", "--group-size", "64", "--jobs", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr   # the guard prints an ERROR and returns cleanly, doesn't crash
    assert "ERROR" in r.stdout and "OLDER converter version" in r.stdout, r.stdout
    assert not glob.glob(str(out / "out-*.safetensors")), "must not have written anything"
