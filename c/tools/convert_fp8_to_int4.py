"""
Convertitore GLM-5.2-FP8 -> nostro container int4 (STADIO B).

Strategia DISK-SAFE (richiesta dell'utente): scarica UNO shard (~5 GB), lo converte in
int4, lo CANCELLA, passa al prossimo. Il disco non si riempie mai: picco = 1 shard + l'output
int4 che cresce fino a ~372 GB. Controllo di spazio che si ferma se manca margine.

Cosa fa per ogni tensore:
  - pesi FP8 (e4m3) con `*.weight_scale_inv`  -> dequant a blocchi 128x128 -> f32
  - pesi BF16 (norme/embed/lm_head/...)        -> f32
  poi:
  - attn/mlp/shared/expert/embed/lm_head -> QUANTIZZATO int4 (o int8) con la STESSA matematica
    del motore C (np.rint = lrintf, stesse soglie, stesso packing dei nibble) -> token identici
  - norme / router (mlp.gate.weight) / bias / e_score_correction_bias -> tenuti F32
  - indexer DSA / layer MTP (78) / shared_head / eh_proj / *norm dell'indexer -> SALTATI

Output: una dir di safetensors leggibile dal motore C (per ogni peso quantizzato: `nome` U8 =
dati impacchettati, `nome.qs` F32 = scale per riga).

USO:
  # test locale (oracolo tiny, niente download): converte una dir gia' presente
  python3 tools/convert_fp8_to_int4.py --indir glm_tiny --outdir glm_tiny_i4 --ebits 4 --io-bits 4
  # selftest del dequant fp8 (richiede torch)
  python3 tools/convert_fp8_to_int4.py --selftest
  # reale: scarica+converte+cancella shard per shard
  python3 tools/convert_fp8_to_int4.py --repo zai-org/GLM-5.2-FP8 --outdir /home/vincenzo/glm52_i4
"""
import os, sys, glob, json, shutil, argparse
import numpy as np

def read_n_layers_from_config(cfg_dir):
    """Read num_hidden_layers from <cfg_dir>/config.json, if present and readable.
    Inspection-driven replacement for the old arch-keyed detect_arch: n_layers is
    read from the checkpoint's OWN config regardless of which architecture it is
    (GLM, Kimi K2, or anything else DeepSeek-V3-shaped) -- no arch name involved.
    Kimi-K2.6 nests the real params under `text_config` (see flatten_container_config);
    check there too when the flat key is absent, so a K2.6 --indir/--repo source
    resolves the real layer count instead of silently keeping --n-layers' default.
    Returns None (caller keeps whatever --n-layers default/override it already has)
    on any read failure or a missing key -- never raises, since a malformed/partial
    config.json (e.g. a mid-download mirror) must not crash the whole conversion."""
    try:
        cfg = json.loads(open(os.path.join(cfg_dir, "config.json")).read())
    except (OSError, ValueError):
        return None
    if "num_hidden_layers" in cfg:
        return cfg.get("num_hidden_layers")
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        return text_cfg.get("num_hidden_layers")
    return None

# ---------- quantizzazione: identica al C (glm.c) ----------
def quant_int8(w, bits):                       # w: [O,I] f32 -> (qbytes U8 [O*I], scale f32 [O])
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -qmax - 1, qmax).astype(np.int8)
    return q.reshape(-1).view(np.uint8).copy(), s[:, 0].astype(np.float32)

def quant_int4(w, bits):                        # -> (qbytes U8 [O*ceil(I/2)], scale f32 [O])
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -8, qmax).astype(np.int32)  # nibble [-8,7]
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)

def quant_int4_grouped(w, bits, gs=128):
    """Group-scaled int4: one scale per group of `gs` elements along the input dim.
    Drastically reduces quantization error vs per-row scaling — matches the FP8
    source's 128x128 block-scale granularity. Output layout:
      qbytes: same packed nibble format as quant_int4
      scales: f32 [O * ngroups] where ngroups = ceil(I/gs), laid out as
              s[o * ngroups + g] = scale for row o, group g.
    The engine detects this format (fmt=4) by checking the .qs array size."""
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    ngroups = (I + gs - 1) // gs
    # pad I to a multiple of gs for clean reshape, then trim
    Ipad = ngroups * gs
    wpad = np.zeros((O, Ipad), np.float32)
    wpad[:, :I] = w
    wr = wpad.reshape(O, ngroups, gs)                     # [O, ngroups, gs]
    amax = np.abs(wr).max(axis=2, keepdims=True)          # [O, ngroups, 1]
    s = np.maximum(amax / qmax, 1e-8)                     # [O, ngroups, 1]
    q = np.clip(np.rint(wr / s), -8, qmax).astype(np.int32)  # [O, ngroups, gs]
    q = q.reshape(O, Ipad)[:, :I]                         # trim padding -> [O, I]
    # pack nibbles (identical to quant_int4)
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    # scales: flatten [O, ngroups] -> [O * ngroups]
    s_flat = s[:, :, 0].astype(np.float32).reshape(-1)
    return out.reshape(-1), s_flat

def quant_int3_g64(w, bits=3, group=64):        # -> (qbytes U8 [O*ceil(I/64)*24], scales f32 [O*ceil(I/64)])
    """int3 with PER-GROUP scales (fmt=5 in colibri.c): per 64-input group, symmetric absmax
    (qmax=3, clamp [-4,3], stored v+4), packed as 16B low plane (2 bits/val, int2 layout)
    + 8B high plane (1 bit/val). Same math as quant_ablation._quant_last_dim(bits=3,
    group=64) (#132), here with real packing. 3.5 bits/weight effective."""
    O, I = w.shape
    ng = (I + group - 1) // group
    pad = ng * group - I
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    g = wp.reshape(O, ng, group)
    amax = np.abs(g).max(axis=2, keepdims=True)
    s = np.maximum(amax / 3.0, 1e-8)
    q = (np.clip(np.rint(g / s), -4, 3).astype(np.int32) + 4).astype(np.uint8)  # 0..7
    if pad: q[:, -1, group - pad:] = 4                                          # pad packs as 0 after -4
    lo = np.zeros((O, ng, 16), np.uint8)
    for k in range(4):
        lo |= ((q[:, :, k::4] & 3) << (k * 2)).astype(np.uint8)
    hi = np.zeros((O, ng, 8), np.uint8)
    for b in range(8):
        hi |= (((q[:, :, b::8] >> 2) & 1) << b).astype(np.uint8)
    out = np.concatenate([lo, hi], axis=2)                                      # [O, ng, 24]
    return out.reshape(-1), s[:, :, 0].astype(np.float32).reshape(-1)

E8 = "e8"                                       # CLI/bits-plumbing marker for fmt=6 (not a bit width)

def quant_e8(w):                                # -> (qbytes U8 [O*ceil(I/256)*98], tag f32 [1])
    """E8/IQ3 lattice (fmt=6 in colibri.c, #452): rotate the rows first (W@Q,
    block-diagonal FWHT with regenerated signs — iq3_pack.rotate_rows mirrors
    quant.h e8_rot_rows), then pack with the iq3 codec: 98B per 256 weights,
    3.0625 bpw, every scale in-block. The .qs companion is a single float,
    the engine's fmt=6 discriminator — not a scale."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import iq3_pack
    O, I = w.shape
    if I % 256:
        raise SystemExit(f"e8: input dim {I} is not a multiple of 256")
    packed = iq3_pack.encode(iq3_pack.rotate_rows(np.asarray(w, dtype=np.float32)))
    return packed.reshape(-1), np.array([6.0], dtype=np.float32)

def quant_int2(w, bits):                        # -> (qbytes U8 [O*ceil(I/4)], scale f32 [O]); 4/byte
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1                 # bits=2 -> qmax=1, valori [-2,1]
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -2, qmax).astype(np.int32)
    rb = (I + 3) // 4
    out = np.zeros((O, rb), np.uint8)
    for k in range(4):                           # impacchetta 4 valori per byte (identico a pack_int2 in C)
        vk = q[:, k::4]
        out[:, :vk.shape[1]] |= ((vk + 2).astype(np.uint8) << (k * 2))
    return out.reshape(-1), s[:, 0].astype(np.float32)

# ---------- NVFP4 (modelopt) : LUT e2m1 ----------
# FP4 e2m1 = 1 sign + 2 exp + 1 mantissa. 16 codici, magnitudini {0,.5,1,1.5,2,3,4,6}.
# Bit 3 = segno. Ordine impacchettato (compressed_tensors/vLLM): nibble BASSO = elemento
# pari, nibble ALTO = elemento dispari. LUT verificata 1:1 con ml_dtypes.float4_e2m1fn.
# EN: FP4 e2m1 = 1 sign + 2 exp + 1 mantissa. 16 codes, magnitudes {0,.5,1,1.5,2,3,4,6}.
# EN: bit 3 = sign. Packed order (compressed_tensors/vLLM): LOW nibble = even element,
# EN: HIGH nibble = odd element. LUT verified 1:1 against ml_dtypes.float4_e2m1fn.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# ---------- Kimi-K2.6 container unwrapping (prefix strip / vision drop / config flatten) ----------
# K2.6 packages the SAME text backbone colibri already runs (byte-for-byte, validated
# 32/32 TF + 20/20 decode) inside a bigger multimodal container: every text tensor name
# gains a `language_model.` prefix, vision-tower + mm-projector tensors are bundled into
# the same shards, and the config nests the real params under `text_config`. All three
# must be undone BEFORE classify() (or anything else) inspects a name: classify()'s
# layer_idx logic requires p[0]=="model" (a `language_model.` prefix makes it return -1
# for EVERY tensor) and its embed/lm_head check is an EXACT string match (silently
# reclassifies to the "q" fallback) -- neither raises, so this must happen at the single
# point tensor names first enter the pipeline (_shard_tensor_groups' `for name in
# f.keys()` loop), not "as names are written" downstream.
_LM_PREFIX = "language_model."
_VISION_PREFIXES = ("vision_tower.", "mm_projector.")

def strip_lm_prefix(name):
    """Strip K2.6's `language_model.` prefix, if present, so classify() sees the exact
    same bare `model.layers...`/`lm_head.weight` names it always has for GLM. A name
    without the prefix passes through unchanged -- the byte-identical regression guard
    for flat checkpoints."""
    return name[len(_LM_PREFIX):] if name.startswith(_LM_PREFIX) else name

def vision_drop_category(name):
    """Returns "vision_tower" or "mm_projector" if `name` is a vision-tower or
    mm-projector tensor (dropped, deliberately and reversibly -- colibri has no image
    support anywhere in c/colibri.c, c/coli, or c/openai_server.py today), else None.
    Checked BEFORE strip_lm_prefix/classify: vision names never carry a
    `language_model.` prefix in K2.6's own naming, but even if they did, classify()
    has no notion of a vision tensor and must never see one."""
    if name.startswith("vision_tower."): return "vision_tower"
    if name.startswith("mm_projector."): return "mm_projector"
    return None

def flatten_container_config(cfg):
    """K2.6 nests the real text-backbone config under `text_config`, alongside a
    `vision_config` and a top-level model_type of "kimi_k25"/architectures naming the
    multimodal class. Emit `text_config` VERBATIM as the container's config.json --
    carry NOTHING over from the top level: text_config already holds bos/eos/pad_token_id
    and its own model_type "kimi_k2", while a merge like {**text_config, **cfg} would
    restore the TOP level's model_type ("kimi_k25") and the chat-template selectors
    compare EXACTLY to "kimi_k2" -- GLM's template through a Kimi tokenizer, fluent
    garbage, no error. A flat config (GLM: no `text_config` key) passes through
    unchanged; callers should keep using the original bytes (shutil.copy) rather than
    round-tripping this dict through json.dump, so formatting/key-order never drifts."""
    if isinstance(cfg, dict) and isinstance(cfg.get("text_config"), dict):
        return cfg["text_config"]
    return cfg

# ---------- classificazione dei tensori ----------
def layer_idx(name):
    p = name.split(".")
    if len(p) > 2 and p[0] == "model" and p[1] == "layers":
        try: return int(p[2])
        except ValueError: return -1
    return -1

def classify(name, n_layers, keep_mtp=False, keep_idx=False):
    # Defense in depth, on top of _shard_tensor_groups stripping the prefix before
    # calling classify(): strip it here too so classify() is correct no matter what a
    # caller passes it -- its layer_idx logic requires p[0]=="model" (a still-prefixed
    # name returns -1 for EVERY layer) and its embed/lm_head check is an exact string
    # match (silently reclassifies to the "q" fallback). idempotent: a name without the
    # prefix (GLM) is returned unchanged.
    name = strip_lm_prefix(name)
    if name.endswith("_scale_inv"): return "consumed"   # FP8 base: gestito col suo peso
    # Sidecar delle scale, consumati insieme al loro peso: .weight_scale/.weight_scale_2/
    # .input_scale (NVFP4 modelopt) e .weight_shape (compressed-tensors pack-quantized).
    if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale", ".weight_shape")): return "consumed"
    li = layer_idx(name)
    if keep_idx:
        # modalita' --indexer: SOLO i pesi del DSA lightning indexer dei layer principali
        if li < 0 or li >= n_layers or "indexer" not in name: return "skip"
        if name.endswith("norm.weight"): return "f32"
        return "q"                                       # int8 consigliato (--ebits 8): pesi di scoring
    if keep_mtp:
        if li != n_layers: return "skip"                 # solo il layer MTP
        if "indexer" in name: return "skip"              # il DSA indexer resta un no-op
    else:
        if li >= n_layers: return "skip"                 # layer MTP (78)
        if any(k in name for k in ["indexer", "indexers_proj", "eh_proj",
                                    "enorm", "hnorm", "shared_head"]): return "skip"
    if name.endswith("rotary_emb.inv_freq"): return "skip"   # K2 per-layer RoPE buffer; engine derives from theta
    if name.endswith("e_score_correction_bias"): return "f32"
    # router (NON gate_proj): GLM/K2 name it mlp.gate, K3 block_sparse_moe.gate
    if name.endswith(("mlp.gate.weight", "block_sparse_moe.gate.weight")): return "f32"
    if name.endswith("norm.weight") or name == "model.norm.weight": return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"): return "io"
    if ".mlp.experts." in name and name.endswith(".weight"): return "x"          # expert ROUTED (streaming)
    if ".mlp.experts." in name and name.endswith(".weight_packed"): return "x"    # K2 compressed-tensors int4 expert
    if ".block_sparse_moe.experts." in name and name.endswith(".weight_packed"): return "x"  # K3 mxfp4 expert
    # Split resident weights by type for mixed-precision control:
    #   "sh" = shared expert (fires on every token, highest sensitivity)
    #   "o"  = o_proj attention (reconstructs output, biggest attn tensor)
    #   "kvb" = kv_b_proj (reconstructs KV cache on every decode step)
    #   "attn" = other attention projections (q_a, q_b, kv_a)
    #   "dmlp" = dense MLP (first 3 layers)
    if "shared_experts" in name: return "sh"
    if name.endswith("o_proj.weight"): return "o"
    if name.endswith("kv_b_proj.weight"): return "kvb"
    if any(name.endswith(k) for k in ("q_a_proj.weight", "q_b_proj.weight",
                                       "kv_a_proj_with_mqa.weight")): return "attn"
    if any(name.endswith(k) for k in ("mlp.gate_proj.weight", "mlp.up_proj.weight",
                                       "mlp.down_proj.weight")): return "dmlp"
    # K3 per-layer residual mixers ([1, hidden]) -- tiny, keep f32 rather than paying a
    # 1-row int8 quantization for no byte savings that matter.
    if name.endswith(("mlp_res_proj.weight", "self_attention_res_proj.weight",
                      "output_attn_res_proj.weight")): return "f32"
    if name.endswith(".weight"): return "q"              # fallback: other resident weights
    if name.endswith(".weight_packed"):
        raise SystemExit(f"unexpected compressed-tensors packed tensor outside routed experts: "
                          f"{name} — this converter only transcodes .mlp.experts.* (K2 int4) and "
                          f".block_sparse_moe.experts.* (K3 mxfp4) .weight_packed tensors")
    return "f32"

# ---------- dequant NVFP4 (modelopt) di UN tensore expert -> f32 [O,I] ----------
def dequant_nvfp4(f, name):
    """NVFP4 di NVIDIA modelopt (quant_algo=NVFP4, quant_method=modelopt).
      - `name`               U8   [O, I/2]  : due nibble e2m1 per byte lungo la dim di
                                              contrazione (input); pari=nibble basso, dispari=alto.
      - `name.weight_scale`  F8_E4M3 [O, I/16] : scala per-BLOCCO di 16 elementi (group_size=16),
                                              lungo la dim di input. Decodifica f8e4m3 -> f32.
      - `name.weight_scale_2` F32 []        : scala GLOBALE per-tensore, ~amax/(6*448) (piccola).
    Dequant (convenzione modelopt = MOLTIPLICA, NON dividere):
        W[o,i] = e2m1_lut[nibble] * f8_block_scale[o, i//16] * weight_scale_2
    FOOTGUN: llm-compressor/compressed-tensors memorizza il RECIPROCO (global grande) e DIVIDE;
    modelopt memorizza il valore piccolo e MOLTIPLICA. Questo checkpoint e' modelopt -> moltiplica.
    EN: NVIDIA modelopt NVFP4. LOW nibble=even elem, HIGH=odd. weight_scale = per-16-block FP8
    EN: (group_size 16) along the input dim; weight_scale_2 = per-tensor global FP32 (~amax/2688,
    EN: small). Dequant MULTIPLIES both scales. FOOTGUN: llm-compressor stores the reciprocal
    EN: (large global) and DIVIDES; modelopt stores the small value and MULTIPLIES."""
    import torch
    GS = 16                                                       # NVFP4: block scale ogni 16 elementi
    packed = f.get_tensor(name)                                    # uint8 [O, I/2]
    bscale = f.get_tensor(name + "_scale").to(torch.float32)        # [O, ceil(I/16)] da f8e4m3
    gscale = f.get_tensor(name + "_scale_2").to(torch.float32)      # scalare per-tensore
    O, Ih = packed.shape; I = Ih * 2
    # Convenzione: modelopt memorizza il global PICCOLO e MOLTIPLICA. Se e' >=1 e'
    # quasi certamente il reciproco di compressed-tensors (che DIVIDE) -> ci fermiamo
    # invece di corrompere silenziosamente ogni tensore. EN: guard modelopt-vs-CT.
    assert float(gscale) < 1.0, (
        f"{name}: weight_scale_2={float(gscale):.4g} >= 1 sembra il reciproco "
        "(compressed-tensors, che DIVIDE); questo path assume modelopt (MOLTIPLICA)")
    # Il layout deve essere lo scale per-blocco piatto di modelopt: una colonna ogni
    # 16 elementi di input (niente swizzle cutlass/TensorRT). Verifichiamo, non deduciamo:
    # dedurre gs = I // ncol misallinea in silenzio su layout paddati/swizzati.
    nb = (I + GS - 1) // GS
    assert bscale.shape[1] == nb, (
        f"{name}: weight_scale ha {bscale.shape[1]} colonne, attese {nb} = ceil({I}/{GS}); "
        "layout scale inatteso (swizzled/paddato?), rifiuto per non corrompere")
    lut = torch.tensor(_E2M1, dtype=torch.float32)
    nib = torch.empty((O, I), dtype=torch.long)
    nib[:, 0::2] = (packed & 0x0F).to(torch.long)                  # elemento pari = nibble basso
    nib[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.long)           # elemento dispari = nibble alto
    w4 = lut[nib]                                                  # [O, I] valori e2m1
    sc = bscale.repeat_interleave(GS, dim=1)[:, :I]               # blocco parziale di coda: slice a I
    return (w4 * sc * gscale).numpy()

# ---------- compressed-tensors 'pack-quantized' int4 (Kimi K2) ----------
def unpack_compressed_int4(f, packed_name):
    """compressed-tensors W4A16 symmetric int4 -> (codes int8 [O,I] in [-8,7],
    scale f32 [O, ngroups]).  Layout — verified against the compressed-tensors
    reference decoder (real K2 shard model-00002-of-000062.safetensors,
    layer-1 routed experts, 6/6 gate/up/down_proj tensors bit-identical, w_maxdiff=0.0):
      <base>.weight_packed  int32 [O, ceil(I/8)] : 8 nibbles per word, LSB = lowest column,
                                                   OFFSET-BINARY: value = nibble - 8 (nibble
                                                   0..15 -> value -8..7). NOT two's complement
                                                   (an earlier version of this function got
                                                   that wrong — caught by the reference cross-check).
                                                   This is also exactly colibri's own fmt=4
                                                   convention (nibble = value + 8).
      <base>.weight_scale   [O, ceil(I/32)]      : per-group scale (bf16/fp16 -> f32)
      <base>.weight_shape   [2]                  : original [O, I]; trims packing padding
    Dequant convention is w = code * scale (symmetric, no zero point)."""
    import torch
    base = packed_name[:-len(".weight_packed")]
    packed = f.get_tensor(packed_name).numpy().view(np.uint32)          # reinterpret int32 bits
    scale = f.get_tensor(base + ".weight_scale").to(torch.float32).numpy()
    shape = [int(v) for v in f.get_tensor(base + ".weight_shape").tolist()]
    O, I = shape[0], shape[1]
    # Loud guard, same class as dequant_nvfp4's scale-layout assert just above: refuse to
    # silently misread a swizzled/padded shard layout (e.g. a future checkpoint revision)
    # instead of corrupting it.
    if packed.shape[0] != O:
        raise ValueError(f"{packed_name}: weight_packed has {packed.shape[0]} rows, "
                          f"expected {O} = weight_shape[0]; layout unexpected, refusing to corrupt")
    ngroups = (I + 31) // 32
    if scale.shape[1] != ngroups:
        raise ValueError(f"{packed_name}: weight_scale has {scale.shape[1]} columns, expected "
                          f"{ngroups} = ceil({I}/32); scale layout unexpected (swizzled/padded?), "
                          "refusing to corrupt")
    # Without this, packed.shape[1]*8 < I would make cols[:, :I] below silently CLIP (numpy
    # slicing past the end is not an error) instead of raising -- wrong-shaped codes returned
    # in silence. O/I come from the independent weight_shape tensor, so this can't be inferred.
    nwords = (I + 7) // 8
    if packed.shape[1] != nwords:
        raise ValueError(f"{packed_name}: weight_packed has {packed.shape[1]} columns, expected "
                          f"{nwords} = ceil({I}/8); layout unexpected (swizzled/padded?), "
                          "refusing to corrupt")
    pack_factor = 8
    cols = np.empty((packed.shape[0], packed.shape[1] * pack_factor), np.int32)
    for k in range(pack_factor):
        cols[:, k::pack_factor] = (packed >> (4 * k)) & 0xF            # unsigned nibble 0..15
    cols = cols[:, :I]
    codes = (cols.astype(np.int16) - 8).astype(np.int8)                # offset-binary -> [-8,7]
    return codes, scale

def transcode_compressed_int4(codes, scale):
    """Lossless int4 -> colibri fmt=4. codes int8 [O,I] in [-8,7], scale f32 [O,ngroups]
    (group 32) -> (qbytes U8 [O*ceil(I/2)], s_flat BF16 [O*ngroups]).
    nibble = code + 8, packed 2/byte along I (identical layout to quant_int4_grouped,
    minus the quantization step: the codes are already the final int4 values).

    Scales come back as ml_dtypes.bfloat16, NOT f32, unlike every other quant_* path
    in this file. This is deliberate and scoped to K2's transcode only: the source
    checkpoint's `*.weight_scale` tensors ARE bf16 (unpack_compressed_int4 upcasts them
    to f32 only so the arithmetic above has a normal float type to work with) -- so
    downcasting back to bf16 here is a lossless round-trip, not a precision loss, and it
    halves the .qs bytes vs. storing them as f32 (container size + expert-streaming
    bandwidth). The engine's fmt=4 loader upcasts bf16 -> f32 at load time. Do NOT copy
    this bf16 downcast into quant_int4/quant_int4_grouped/quant_int8/etc.: those scales
    are computed from scratch (amax/qmax) for GLM and other checkpoints, not sourced
    from an already-bf16 tensor, so f32 there is the correct on-disk precision."""
    import ml_dtypes
    O, I = codes.shape
    if scale.shape[0] != codes.shape[0]:
        raise ValueError(f"transcode: scale has {scale.shape[0]} rows, codes has {codes.shape[0]}")
    q = codes.astype(np.int32)
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), scale.reshape(-1).astype(ml_dtypes.bfloat16)

# ---------- compressed-tensors 'mxfp4-pack-quantized' (Kimi K3) ----------
def transcode_mxfp4(f, packed_name):
    """compressed-tensors mxfp4 (Kimi K3) -> colibri mxfp4 container, RAW PASSTHROUGH.
    Layout -- verified bit-exact against the compressed-tensors 0.17.1 reference
    primitives (unpack_fp4_from_uint8 + decompress_mx_scale) on a real K3 shard
    (model-00002-of-000096, layer-1 expert-0 w1, maxdiff = 0.0):
      <base>.weight_packed  U8 [O, I/2]  : two e2m1 nibbles per byte along the input dim,
                                           LOW nibble = even element (same convention as
                                           NVFP4/_E2M1, per the compressed-tensors/vLLM
                                           fp4 pack).
      <base>.weight_scale   U8 [O, I/32] : e8m0 per-32-group scale, value = 2^(u8 - 127).
    Unlike K2's int4 pack there is NO .weight_shape sidecar: I derives from the packed
    width (x2), cross-checked against the scale group count (16 packed bytes per group
    of 32). mxfp4's e2m1 grid is NOT representable in fmt=4's uniform int4 grid (the
    ratios 0.5..6 would need codes up to +/-12), so unlike K2 there is no transcode INTO
    fmt=4 -- instead the bytes pass through VERBATIM (lossless by construction, zero
    convert-time math) as a new container format the engine recognizes by the .qs dtype:
      <base>.weight     U8 [O*ceil(I/2)] : the nibble bytes, exactly as in the source
      <base>.weight.qs  U8 [O*I/32]      : the e8m0 bytes, exactly as in the source
    Engine decode contract (pinned by test_transcode_mxfp4_is_raw_passthrough):
      w[o,i] = _E2M1[nibble(o,i)] * 2^(qs[o, i//32] - 127)
    Guards mirror unpack_compressed_int4's: refuse any swizzled/padded/other-dtype
    layout loudly instead of silently corrupting a 1.4 TB conversion."""
    GS = 32
    base = packed_name[:-len(".weight_packed")]
    packed = f.get_tensor(packed_name).numpy()
    scale = f.get_tensor(base + ".weight_scale").numpy()
    if packed.dtype != np.uint8:
        raise ValueError(f"{packed_name}: weight_packed dtype {packed.dtype}, expected uint8")
    if scale.dtype != np.uint8:
        raise ValueError(f"{packed_name}: weight_scale dtype {scale.dtype}, expected uint8 "
                          "(e8m0); a non-u8 scale means a different mxfp4 revision, refusing "
                          "to reinterpret its bytes")
    if packed.ndim != 2 or scale.ndim != 2 or scale.shape[0] != packed.shape[0]:
        raise ValueError(f"{packed_name}: weight_scale rows {scale.shape} vs weight_packed "
                          f"{packed.shape}; layout unexpected, refusing to corrupt")
    if packed.shape[1] % (GS // 2):
        raise ValueError(f"{packed_name}: packed width {packed.shape[1]} bytes is not a "
                          f"multiple of {GS//2} (input dim not a multiple of the group size "
                          f"{GS}); layout unexpected, refusing to corrupt")
    ngroups = packed.shape[1] // (GS // 2)               # = ceil(I/32), I = packed_cols*2
    if scale.shape[1] != ngroups:
        raise ValueError(f"{packed_name}: weight_scale has {scale.shape[1]} columns, expected "
                          f"{ngroups} = (packed_cols*2)/{GS}; scale layout unexpected "
                          "(swizzled/padded?), refusing to corrupt")
    return packed.reshape(-1), scale.reshape(-1)

# ---------- dequant di un tensore (nvfp4 / fp8+scale a blocchi / bf16 / f32) ----------
def dequant(f, name, keys):
    import torch
    sl = f.get_slice(name); dt = sl.get_dtype()
    # NVFP4 (modelopt): pesi expert U8 con sidecar `.weight_scale`. In questo checkpoint gli
    # UNICI tensori U8 sono gli expert NVFP4, ma richiediamo comunque il sidecar (keys e'
    # obbligatorio: senza, un qualunque U8 verrebbe decodificato come NVFP4).
    # EN: NVFP4 expert weights are U8 with a `.weight_scale` sidecar; require the sidecar.
    if dt in ("U8", "uint8") and (name + "_scale") in keys:
        return dequant_nvfp4(f, name)
    if dt in ("F8_E4M3", "float8_e4m3fn"):
        w = f.get_tensor(name).to(torch.float32)
        sc = f.get_tensor(name + "_scale_inv").to(torch.float32)   # [ceil(O/128),ceil(I/128)]
        O, I = w.shape
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:O, :I]
        return (w * sc).numpy()
    return f.get_tensor(name).to(torch.float32).numpy()

# Per-projection bit overrides for ROUTED experts (gate_proj/up_proj/down_proj), set from
# --up-bits/--gate-bits/--down-bits in main(). Empty = uniform xbits. Motivated by the
# measured result that up_proj tolerates int3-g64 at ~zero quality cost while int2 craters
# (OLMoE ablation, PR #168 comment): up-only int3 drops ~8% of expert bytes for free.
# NB: the resume manifests (check_or_record_params and the --indir progress file) already
# record dict(PROJ_BITS) — this global is the definition those sites depend on.
PROJ_BITS = {}

def _shard_tensor_groups(path, n_layers, ebits, io_bits, xbits,
                          keep_mtp=False, keep_idx=False, group_size=0, bits_map=None,
                          vision_counts=None):
    """Core per-tensor conversion logic for ONE shard, factored out of convert_shard so
    both the whole-dict path (convert_shard, used unchanged by --repo) and the
    chunked-flush path (convert_shard_to_files, used by --indir's parallel workers) share
    IDENTICAL math -- this is what makes parallel+chunked output provably equal to the
    old sequential output: there is exactly one place the quantization decisions are made.
    This is ALSO the single point tensor names first enter the pipeline, so it is where
    K2.6's `language_model.` prefix is stripped and its vision tensors are dropped --
    BEFORE classify() (or anything else) inspects a name (see the module-level comment
    above strip_lm_prefix/vision_drop_category for why the order matters).
    `raw_name` (exactly as stored in the shard) is used for every safetensors lookup
    (f.get_tensor/get_slice, unpack_compressed_int4, dequant) since that's the name the
    FILE actually has; `name` (prefix-stripped) is used for classify() and as the output
    tensor name, so the emitted container never carries the `language_model.` prefix.
    `vision_counts`, if given a dict, is incremented in place per dropped category
    ("vision_tower"/"mm_projector") -- the caller aggregates and reports the total.
    Yields one GROUP (a list of (name, ndarray) pairs) per SOURCE tensor consumed from
    the shard: 1 entry for f32-kept tensors, 2 entries (weight + weight.qs) for
    quantized ones. Keeping weight+scale together in one group means a chunk-size flush
    can never split a tensor from its own scale across two output files."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for raw_name in f.keys():
            vcat = vision_drop_category(raw_name)
            if vcat is not None:
                if vision_counts is not None:
                    vision_counts[vcat] = vision_counts.get(vcat, 0) + 1
                continue
            name = strip_lm_prefix(raw_name)
            kind = classify(name, n_layers, keep_mtp, keep_idx)
            if kind in ("skip", "consumed"): continue
            if kind == "x" and name.endswith(".weight_packed"):    # K2/K3: lossless transcode
                base = name[:-len(".weight_packed")]
                # Dispatch on the .weight_shape sidecar, present in K2's int4 pack and
                # absent from K3's mxfp4 pack -- file-driven, no config plumbing needed.
                if (raw_name[:-len(".weight_packed")] + ".weight_shape") in keys:
                    codes, scale = unpack_compressed_int4(f, raw_name)   # K2 int4
                    q, s = transcode_compressed_int4(codes, scale)
                else:
                    q, s = transcode_mxfp4(f, raw_name)                  # K3 mxfp4 passthrough
                yield [(base + ".weight", q), (base + ".weight.qs", s)]
                continue
            w = dequant(f, raw_name, keys)
            if kind == "f32":
                yield [(name, w.astype(np.float32))]
                continue
            # Resolve bits for this tensor type: use bits_map override if provided,
            # otherwise fall back to the classic ebits/xbits/io_bits scheme.
            if bits_map and kind in bits_map:
                bits = bits_map[kind]
            else:
                bits = io_bits if kind == "io" else xbits if kind == "x" else ebits
            # Any unknown kind that fell through classify as "q"
            if bits_map and kind not in bits_map and kind not in ("io", "x", "sh", "o", "kvb", "attn", "dmlp"):
                bits = ebits
            # Per-projection override for routed experts, applied on top of the type-level bits.
            if kind == "x" and PROJ_BITS:          # e.g. up_proj -> 3 (int3-g64) while gate/down stay 4
                for proj, pb in PROJ_BITS.items():
                    if f".{proj}.weight" in name: bits = pb; break
            if w.ndim != 2:        # es. bias 1D non previsto come 'q' -> tienilo f32
                yield [(name, w.astype(np.float32))]
                continue
            if bits == E8:
                # fmt=6 E8/IQ3 — routed-expert projections only, enforced in main().
                q, s = quant_e8(w)
            elif bits == 3:
                # int3-g64 (fmt=5): inherently group-64, distinct from grouped-int4.
                q, s = quant_int3_g64(w)
            elif group_size > 0 and bits <= 4:
                q, s = quant_int4_grouped(w, bits, group_size)
            else:
                q, s = (quant_int2(w, bits) if bits <= 2 else
                        quant_int4(w, bits) if bits <= 4 else quant_int8(w, bits))
            yield [(name, q), (name + ".qs", s)]

def convert_shard(path, out_dict, n_layers, ebits, io_bits, xbits,
                  keep_mtp=False, keep_idx=False, group_size=0, bits_map=None,
                  vision_counts=None):
    for group in _shard_tensor_groups(path, n_layers, ebits, io_bits, xbits,
                                       keep_mtp, keep_idx, group_size, bits_map,
                                       vision_counts):
        for name, arr in group:
            out_dict[name] = arr

def convert_shard_to_files(path, outdir, prefix, shard_idx, n_layers, ebits, io_bits, xbits,
                           keep_mtp=False, keep_idx=False, group_size=0, bits_map=None,
                           chunk_bytes=None, vision_counts=None):
    """Convert ONE shard, flushing accumulated output tensors to
    outdir/{prefix}{shard_idx:05d}-{chunk:03d}.safetensors whenever the accumulated raw
    byte size (sum of ndarray.nbytes) reaches chunk_bytes -- this bounds a single
    worker's peak resident memory to ~chunk_bytes plus one in-flight tensor, which is
    what lets --jobs exceed the naive whole-shard-in-RAM parallelism cap.
    chunk_bytes=None means never flush early: everything accumulates into ONE chunk
    (chunk 000), written at the end -- this is --chunk-gb's "huge" setting used to get
    one-file-per-shard sequential-equivalent behavior for the parallel==sequential proof.
    Each chunk file is written to a `.tmp` sibling, fsync'd (file + directory entry),
    then atomically renamed into place, so a crash mid-write never leaves a half-written
    file under the final name for a resumed run to trip over.
    Returns the list of chunk file BASENAMES written, in order (empty if the shard
    produced no output tensors at all -- e.g. an --mtp/--indexer pass over a shard that
    holds none of the wanted tensors)."""
    from safetensors.numpy import save_file
    chunk, nbytes, chunk_idx, written = {}, 0, 0, []
    def flush():
        nonlocal chunk, nbytes, chunk_idx
        if not chunk: return
        name = f"{prefix}{shard_idx:05d}-{chunk_idx:03d}.safetensors"
        dest = os.path.join(outdir, name); tmp = dest + ".tmp"
        save_file(chunk, tmp)
        _fsync_path(tmp)
        os.replace(tmp, dest)
        _fsync_dir(dest)
        written.append(name)
        chunk_idx += 1; chunk = {}; nbytes = 0
    for group in _shard_tensor_groups(path, n_layers, ebits, io_bits, xbits,
                                       keep_mtp, keep_idx, group_size, bits_map,
                                       vision_counts):
        for name, arr in group:
            chunk[name] = arr
            nbytes += arr.nbytes
        if chunk_bytes is not None and nbytes >= chunk_bytes:
            flush()
    flush()
    return written

def _available_ram_gb():
    """Read MemAvailable from /proc/meminfo (Linux) -- the kernel's own estimate of how
    much memory a new process can get without swapping (unlike MemFree, it accounts for
    reclaimable caches/buffers, so it's the right number to size a worker pool against).
    Falls back to a small, conservative value on any read failure (non-Linux, no /proc,
    malformed line, ...) so --jobs auto still picks something rather than crashing --
    a low fallback under-parallelizes instead of risking an OOM-driven false confidence."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    return 4.0

def _auto_jobs(chunk_gb):
    """Default --jobs for --indir: min(cpu_count, RAM-bounded worker count). Each
    worker's peak RSS is bounded to roughly chunk_gb (the chunk-flush threshold) plus
    ~2 GB of interpreter/torch/numpy/safetensors-mmap overhead, so available RAM allows
    floor(available_gb / (chunk_gb + 2)) workers before risking swap. chunk_gb<=0 (the
    unbounded/one-chunk-per-shard setting) is treated as ~10 GB (typical raw shard
    output size) here so auto-jobs still lands on a sane, RAM-safe worker count instead
    of dividing by a non-positive number."""
    cpu = os.cpu_count() or 1
    cg = chunk_gb if chunk_gb > 0 else 10.0
    ram_workers = max(1, int(_available_ram_gb() // (cg + 2)))
    return max(1, min(cpu, ram_workers))

def _fsync_path(path):
    """fsync a file's own contents by path (used right before an atomic rename, so the
    data is durable on disk before the name that points to it becomes visible)."""
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def _fsync_dir(path):
    """fsync the directory CONTAINING path (used right after an atomic rename, so the
    rename itself -- the directory entry -- is durable across a crash, not just the
    file's bytes). Standard POSIX durable-rename idiom: fsync data, rename, fsync dir."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd = os.open(d, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def _pool_worker_init(proj_bits):
    """multiprocessing Pool initializer: runs once per worker process, before it services
    any task. torch is not fork-safe, so --indir's parallel pool uses the 'spawn' start
    method -- each worker is a genuinely fresh Python interpreter. Thread-count env vars
    (OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS) are ALSO set by the parent (main(), right
    before the Pool is created) so they are already present in the worker's OS
    environment when THIS module's top-level `import numpy as np` runs during the
    worker's bootstrap (which happens before this initializer, since spawn must import
    the module to find the task function at all) -- setting them again here is a no-op
    safety net for that path. torch.set_num_threads(1) is the one call that MUST happen
    here rather than via env var: it's torch's own supported runtime knob and re-caps its
    intra-op thread pool even though torch was already imported once during bootstrap.
    Without this, N worker PROCESSES x M BLAS/torch threads each would oversubscribe the
    machine's cores.
    proj_bits restores this worker's own copy of the PROJ_BITS module global (per-
    projection expert bit overrides read by _shard_tensor_groups) -- each worker is a
    separate process, so the parent's CLI-driven PROJ_BITS mutation doesn't cross the
    process boundary on its own; it must be re-applied here from the value captured in
    the parent before the pool was created."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    global PROJ_BITS
    PROJ_BITS = proj_bits

def _convert_one_shard_task(task):
    """Top-level (module-level, hence picklable for multiprocessing 'spawn') unit of
    work for --indir: convert ONE input shard to its chunk file(s), then write its
    `.done` resume marker. Called either directly in the main process (--jobs 1 /
    single-shard runs -- no Pool, no multiprocessing overhead at all) or inside a spawned
    worker process via Pool.imap_unordered (--jobs > 1). Identical code either way, which
    is what makes --jobs 1 and --jobs N provably produce the same per-shard output: there
    is exactly one implementation of 'convert a shard', not a sequential one and a
    separate parallel one.
    The `.done` marker is written ONLY after its shard's chunk file(s) are fsync'd and
    already visible under their final names (convert_shard_to_files fsyncs+renames each
    chunk as it's flushed) -- so a resumed run that sees `{prefix}{shard_idx:05d}.done`
    can trust every one of that shard's chunk files is complete on disk, never a partial
    write from a killed worker.
    Returns (shard_idx, input_basename, chunk_filenames, vision_counts) -- chunk_filenames
    is [] for a shard that produced no output tensors at all (e.g. an --mtp/--indexer pass
    over a shard holding none of the wanted tensors), mirroring the old empty-marker resume
    semantics without needing a shared "" sentinel. vision_counts is this shard's OWN
    {"vision_tower": n, "mm_projector": n} drop tally -- a fresh dict per call, returned
    (not shared) because --jobs > 1 runs this in a separate spawned process, so a mutable
    dict passed in would never be seen by the parent; main() sums these across all shards.
    Before converting, removes any PRE-EXISTING {prefix}{shard_idx:05d}-*.safetensors
    chunk files for this shard index: this shard is only reprocessed when its `.done`
    marker is absent or one of ITS OWN recorded chunk files went missing (see the
    marker_ok check in main()), which means whatever chunk files already sit at this
    shard's index are leftovers from an earlier, incomplete attempt (interrupted mid-run,
    or a chunk file deleted by hand without touching the marker). Without this cleanup, a
    stale leftover chunk (e.g. `-003.safetensors` from a previous run that used a smaller
    --chunk-gb and got killed after writing 4 chunks) would sit alongside this run's fresh
    chunk files and reintroduce the SAME tensor names a second time -- the exact
    duplicate-across-files hazard convert_shard_to_files's naming scheme exists to avoid."""
    (sp, shard_idx, outdir, prefix, n_layers, ebits, io_bits, xbits,
     keep_mtp, keep_idx, group_size, bits_map, chunk_bytes) = task
    for stale in glob.glob(os.path.join(outdir, f"{prefix}{shard_idx:05d}-*.safetensors")):
        os.remove(stale)
    vision_counts = {}
    written = convert_shard_to_files(sp, outdir, prefix, shard_idx, n_layers, ebits, io_bits, xbits,
                                      keep_mtp=keep_mtp, keep_idx=keep_idx,
                                      group_size=group_size, bits_map=bits_map,
                                      chunk_bytes=chunk_bytes, vision_counts=vision_counts)
    marker = os.path.join(outdir, f"{prefix}{shard_idx:05d}.done")
    tmp = marker + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"input": os.path.basename(sp), "chunks": written}, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, marker)
    _fsync_dir(marker)
    return (shard_idx, os.path.basename(sp), written, vision_counts)

def free_gb(p): return shutil.disk_usage(p).free / 1e9

def check_or_record_params(outdir, prefix, params):
    """#383-class guard, mirrored onto the --repo download loops from the --indir
    path's resume manifest (below): a resumed run with DIFFERENT conversion
    parameters (bits, group size, PROJ_BITS, ...) must not silently mix bit-widths
    across shards in the same outdir -- the #355 failure mode (a second pass with
    changed flags overwriting/interleaving with a finished container in silence).
    Unlike the --indir manifest this doesn't need to track per-shard completion:
    the --repo loops already do that via out-NNNNN.safetensors existence, since
    shard index maps directly to output filename there. Only whether the params
    used SO FAR match this run's needs checking. Returns False (caller should
    abort) on a mismatch, True otherwise; records params on first use."""
    path = os.path.join(outdir, f".{prefix}params.json")
    if os.path.exists(path):
        try: prev = json.loads(open(path).read())
        except (OSError, ValueError): prev = None
        if prev is not None and prev != params:
            print(f"ERROR: {path} records a conversion with {prev};\n"
                  f"       this run uses {params}. Refusing to mix conversions in the "
                  f"same outdir — use a fresh --outdir (or delete {path} and the "
                  f"{prefix}*.safetensors shards to redo).")
            return False
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(params, f, indent=1)   # atomic write, same reasoning as the --indir manifest
    os.replace(tmp, path)
    return True

def _bits(v):                                   # "e8" -> fmt=6 marker; anything else an int width
    return E8 if v == E8 else int(v)

def _write_config_file(src_path, dest_path):
    """Copy config.json from src_path to dest_path, flattening a K2.6-style nested
    config (see flatten_container_config) on the way through. Flat configs (GLM: no
    `text_config` key, or an unparseable/malformed file) are copied byte-for-byte via
    shutil.copy; only a genuinely nested config gets rewritten, and only into
    `text_config`'s own bytes reserialized (nothing merged in from the top level)."""
    try:
        cfg = json.loads(open(src_path).read())
    except (OSError, ValueError):
        cfg = None
    # isinstance(...dict), not `"text_config" in cfg`: key-presence alone was true for
    # `"text_config": null`, which sent a non-dict through flatten and wrote the literal
    # `null` into the container's config.json, destroying the output config. A non-dict
    # text_config falls through to the byte-for-byte shutil.copy below instead, and the
    # engine then fails loudly on the unflattened config rather than on a `null` one.
    if isinstance(cfg, dict) and isinstance(cfg.get("text_config"), dict):
        flat = flatten_container_config(cfg)
        tmp = dest_path + ".tmp"
        with open(tmp, "w") as out:
            json.dump(flat, out, indent=2)
        os.replace(tmp, dest_path)
    else:
        shutil.copy(src_path, dest_path)

def _write_metadata(src_dir, outdir):
    """Copy the four metadata files; generate tokenizer.json from tiktoken.model when
    the source has tiktoken but no tokenizer.json of its own (K2-style) -- FILE-driven,
    not arch-gated: GLM ships tokenizer.json directly, so the four-file copy loop below
    already satisfies it and this generation branch never fires (the `not os.path.exists`
    guard is False). tiktoken.model (and tokenizer_config.json, which gen_kimi_tokenizer
    also needs) may be temporarily absent mid-download (e.g. a partial local K2 mirror) --
    warn and skip instead of crashing. The missing-metadata-file warning applies to ALL
    sources (mirrors the pre-existing GLM missing-tokenizer.json warning this replaces);
    the tiktoken-based generation is layered on top and, on success, removes
    tokenizer.json from the missing list so it isn't double-reported.
    gen_kimi_tokenizer.load_ranks() raises FileNotFoundError if called unconditionally,
    so both the tiktoken.model existence check and the try/except below guard it."""
    copied, missing = [], []
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        s = os.path.join(src_dir, fn)
        if not os.path.exists(s):
            missing.append(fn); continue
        if fn == "config.json":
            _write_config_file(s, os.path.join(outdir, fn))
        else:
            shutil.copy(s, outdir)
        copied.append(fn)
    if not os.path.exists(os.path.join(outdir, "tokenizer.json")):
        tiktoken_path = os.path.join(src_dir, "tiktoken.model")
        if not os.path.exists(tiktoken_path):
            print(f"[META] WARNING: {tiktoken_path} not found; skipping tiktoken-based "
                  "tokenizer.json generation — chat/serve need tokenizer.json")
        else:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import gen_kimi_tokenizer
            try:
                gen_kimi_tokenizer.gen_kimi_tokenizer(src_dir, os.path.join(outdir, "tokenizer.json"))
                copied.append("tokenizer.json(generated)")
                if "tokenizer.json" in missing: missing.remove("tokenizer.json")
            except (OSError, ValueError, KeyError) as ex:
                # Broader than just OSError: a malformed/partial tokenizer_config.json or
                # tiktoken.model (mid-download mirror, truncated file, unexpected schema)
                # must warn-and-continue like a missing file does, not crash the WHOLE
                # conversion (hours of shard work) over one metadata sidecar.
                print(f"[META] WARNING: tokenizer.json generation failed ({type(ex).__name__}: {ex}); "
                      "skipping — chat/serve need tokenizer.json")
    print(f"[META] {outdir}: {', '.join(copied) if copied else 'nothing'}")
    if missing:
        print(f"[META] WARNING: not found in {src_dir}: {', '.join(missing)}"
              + (" — chat/serve need tokenizer.json" if "tokenizer.json" in missing else ""))

# Both lossless-transcode source formats: K2/K2.6's int4 pack and K3's mxfp4 pack.
# Anything in this set defaults residents to int8 (bf16 in the source, must not drop
# to int4) and never consults --xbits for its routed experts.
_PACK_QUANT_FORMATS = ("pack-quantized", "mxfp4-pack-quantized")

def _source_is_pack_quantized(a):
    """Best-effort peek at the source config.json. --indir reads it in place; --repo
    reads the copy a previous/resumed run left in <outdir>/_meta or <outdir>, else
    downloads it there -- the same file the conversion's own metadata step needs, so
    nothing is fetched twice (hf_hub_download reuses local_dir). Any failure returns
    False: fp8 sources (GLM) never needed the peek."""
    if a.indir:      dirs = (a.indir,)
    elif a.outdir:   dirs = (os.path.join(a.outdir, "_meta"), a.outdir)
    else:            dirs = ()
    for d in dirs:
        try:
            cfg = flatten_container_config(json.loads(open(os.path.join(d, "config.json")).read()))
            return cfg.get("quantization_config", {}).get("format") in _PACK_QUANT_FORMATS
        except (OSError, ValueError, AttributeError):
            continue
    if a.repo and a.outdir:
        try:
            from huggingface_hub import hf_hub_download
            meta_dir = os.path.join(a.outdir, "_meta"); os.makedirs(meta_dir, exist_ok=True)
            hf_hub_download(a.repo, "config.json", local_dir=meta_dir)
            cfg = flatten_container_config(json.loads(open(os.path.join(meta_dir, "config.json")).read()))
            return cfg.get("quantization_config", {}).get("format") in _PACK_QUANT_FORMATS
        except Exception:
            return False
    return False

def _resolve_default_ebits(a):
    """Resident-tensor bits when --ebits is not given, decided by the checkpoint:
      - pack-quantized sources (Kimi-K2.6 compressed-tensors): 8. The residents ship
        bf16/int8 and must stay lossless -- int4 here is the over-quantization
        regression this replaced (--arch-gated defaults).
      - fp8 sources (GLM): 4 for the main pass, 8 for --mtp/--indexer (int4 drafts =
        ~0% acceptance, issue #8) -- the historical defaults, unchanged."""
    if a.mtp or a.indexer or _source_is_pack_quantized(a):
        return 8
    return 4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=None)   # bit residenti; default risolto dalla
                                                          # sorgente (vedi _resolve_default_ebits)
    ap.add_argument("--io-bits", type=int, default=8)    # bit di embed/lm_head
    ap.add_argument("--xbits", type=_bits, default=None) # bit degli expert ROUTED (streaming), o "e8" (fmt=6); default=ebits
    # Mixed-precision: per-tensor-type bit overrides. Default = ebits (all same).
    # Set these higher to protect sensitive tensors from quantization error.
    ap.add_argument("--shared-bits", type=int, default=None,
        help="bits for shared expert (fires on every token, highest sensitivity). Default=ebits")
    ap.add_argument("--o-bits", type=int, default=None,
        help="bits for o_proj attention (reconstructs output, biggest attn tensor). Default=ebits")
    ap.add_argument("--kvb-bits", type=int, default=None,
        help="bits for kv_b_proj (reconstructs KV cache on every decode). Default=ebits")
    ap.add_argument("--attn-bits", type=int, default=None,
        help="bits for other attention projections (q_a, q_b, kv_a). Default=ebits")
    ap.add_argument("--dmlp-bits", type=int, default=None,
        help="bits for dense MLP (first 3 layers). Default=ebits")
    ap.add_argument("--group-size", type=int, default=64,
        # gs64 is the community-validated default (#225 root cause, #455 5/5-clean
        # verification, ablation #453: per-row int4 costs -9.3pp mean acc_norm vs
        # -2.2..-3.4pp for group-scaled). Per-row remains available as an explicit
        # opt-out; the resume manifest (check_or_record_params) refuses to mix the
        # two in one outdir, so a resumed pre-default conversion aborts loudly
        # instead of interleaving formats (#355-class).
        help="group size for int4 scales: 64=one scale per 64 elements (default, "
             "much better quality), 0=per-row (legacy; costs ~9pp on quality "
             "benchmarks and is the #455 non-termination trigger)")
    # Per-projection bit overrides for routed experts (orthogonal to the type-level flags above).
    ap.add_argument("--up-bits", type=_bits, default=None,
        help="bits for up_proj in routed experts (e.g. 3 = int3-g64). Default=xbits")
    ap.add_argument("--gate-bits", type=_bits, default=None,
        help="bits for gate_proj in routed experts. Default=xbits")
    ap.add_argument("--down-bits", type=_bits, default=None,
        help="bits for down_proj in routed experts. Default=xbits")
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=0,
        help="--indir ONLY: shards converted in parallel worker processes. 0 (default) = "
             "auto, min(cpu_count, available_RAM_GB // (chunk-gb + 2)) -- RAM-bounded so "
             "N workers x chunk-gb peak each don't exceed physical memory. 1 = sequential, "
             "in-process, no multiprocessing (today's original --indir behavior, byte-for-"
             "byte identical conversion math). --repo is unaffected: it stays the single-"
             "process disk-safe download loop regardless of --jobs.")
    ap.add_argument("--chunk-gb", type=float, default=2.0,
        help="--indir ONLY: a worker flushes its shard's accumulated output tensors to a "
             "new out-{shard:05d}-{chunk:03d}.safetensors file once their raw byte size "
             "reaches this many GB, bounding a worker's peak RAM to ~chunk-gb regardless "
             "of the shard's total output size -- this is what lets --jobs exceed "
             "available_RAM_GB // shard_output_gb. <=0 = unbounded (one chunk per shard, "
             "matching pre-parallel behavior); use a small value (e.g. 0.5) to force "
             "several chunks per shard for testing.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-nvfp4", action="store_true",
        help="unit-test del dequant NVFP4 (LUT e2m1 + round-trip), nessun download / no network")
    ap.add_argument("--selftest-compressed-int4", action="store_true",
        help="unit-test the compressed-tensors int4 unpack (synthetic pack->unpack round-trip, no network)")
    ap.add_argument("--mtp", action="store_true",
        help="download and convert ONLY the MTP head (model.layers.<n_layers>.*) -> out-mtp-*.safetensors")
    ap.add_argument("--indexer", action="store_true",
        help="extract ONLY the DSA lightning-indexer weights -> out-idx-*.safetensors. WARNING: "
             "indexer tensors are spread across nearly every shard, so this re-downloads the whole "
             "repository (~756 GB of traffic) to retain only a few GB. Resumable per shard. "
             "Recommended: --ebits 8.")
    a = ap.parse_args()

    if a.selftest_compressed_int4:
        # Runs before any other argparse post-processing (ebits defaulting, PLAN print, ...)
        # so it works standalone with no other flags, like a selftest should — the pre-existing
        # `[PLAN]` print further down assumes --repo/--indir is set and is out of scope to fix.
        import torch
        import numpy as np  # main() has another `import numpy as np` further down (nvfp4
                             # selftest), which makes `np` function-local for all of main()
                             # per Python scoping rules; rebind it here too so this branch
                             # (which runs earlier) doesn't hit UnboundLocalError.
        rng = np.random.default_rng(1); O, I, gs = 4, 64, 32
        codes = rng.integers(-8, 8, size=(O, I)).astype(np.int8)
        scale = rng.random((O, I // gs)).astype(np.float32) + 0.1
        pf = 8; u = ((codes.astype(np.int32) + 8) & 0xF).astype(np.uint32)  # offset-binary
                             # nibble (value -8..7 -> nibble 0..15), cast to uint32: keeps the
                             # shift/OR chain below in the uint32 domain — NEP 50 promotion
                             # turns `uint32 |= int32` into int64 and raises
                             # UFuncOutputCastingError on numpy>=2 otherwise.
        packed = np.zeros((O, I // pf), np.uint32)
        for k in range(pf): packed |= (u[:, k::pf] << (4 * k))
        class _F:
            def __init__(s, d): s.d = d
            def get_tensor(s, n): return s.d[n]
        f = _F({"w.weight_packed": torch.from_numpy(packed.astype(np.int32)),
                "w.weight_scale": torch.from_numpy(scale),
                "w.weight_shape": torch.tensor([O, I], dtype=torch.int64)})
        gc, gs_ = unpack_compressed_int4(f, "w.weight_packed")
        assert np.array_equal(gc, codes) and np.allclose(gs_, scale), "compressed-int4 unpack round-trip FAILED"
        print("[compressed-int4] synthetic pack->unpack round-trip: OK")
        return

    if a.ebits is None:
        a.ebits = _resolve_default_ebits(a)

    # testa MTP a int4 = acceptance ~0-4% (misurato, issue #8): il draft sbaglia sempre
    # e la speculazione non parte mai. A int8: 39-59%, 2.2-2.8 token/forward. This
    # WARNING only fires when a caller EXPLICITLY lowers --ebits below 8 alongside --mtp.
    if a.mtp and a.ebits < 8 and a.group_size <= 0:
        # Non solo lossy: eh_proj ha ~20-30x di asimmetria di scala fra le due meta' di
        # colonna, quindi l'int4 per-riga (UNA scala per riga) arrotonda a ZERO l'intera
        # meta' embedding -> il draft non vede il token -> acceptance ~0% (issue #8).
        # EN: not merely lossy: eh_proj has ~20-30x column-scale asymmetry, so per-row
        # EN: int4 rounds its ENTIRE embedding half to exact zeros -> the draft cannot
        # EN: see the input token -> ~0% acceptance (issue #8). A container converted
        # EN: this way is repairable in place with tools/repair_mtp_int8.py.
        print(f"WARNING: --mtp with --ebits {a.ebits} and per-row scales ZEROES eh_proj's "
              "embedding half -> MTP acceptance ~0% (issue #8). Use the default --ebits 8, "
              "or drop --group-size 0 to get the group-scaled default.")
    if a.xbits is None: a.xbits = a.ebits
    for proj, val in (("gate_proj", a.gate_bits), ("up_proj", a.up_bits), ("down_proj", a.down_bits)):
        if val is not None: PROJ_BITS[proj] = val
    if PROJ_BITS:
        print(f"[per-projection expert bits] {PROJ_BITS} (others -> xbits={a.xbits})")
    # fmt=6 is all-or-nothing across the three expert projections: gate and up
    # share one rotated input row in the engine (the placement rule in quant.h),
    # so a mixed layout would need two gather buffers for zero measured benefit.
    eff = [PROJ_BITS.get(p, a.xbits) for p in ("gate_proj", "up_proj", "down_proj")]
    if any(b == E8 for b in eff) and not all(b == E8 for b in eff):
        raise SystemExit(f"e8 covers all three expert projections or none (got {eff}); "
                         "use --xbits e8, or none of the e8 flags")

    # Build per-type bits map. If a type-specific arg is set, use it; otherwise the
    # converter falls back to ebits for that type.
    bits_map = {}
    if a.shared_bits is not None: bits_map["sh"] = a.shared_bits
    if a.o_bits is not None:      bits_map["o"] = a.o_bits
    if a.kvb_bits is not None:    bits_map["kvb"] = a.kvb_bits
    if a.attn_bits is not None:   bits_map["attn"] = a.attn_bits
    if a.dmlp_bits is not None:   bits_map["dmlp"] = a.dmlp_bits
    if bits_map:
        print(f"[MIXED] precision map: " + ", ".join(f"{k}={v}bit" for k,v in sorted(bits_map.items())))

    # Il PIANO risolto, PRIMA di toccare qualunque cosa (#383): --mtp/--indexer cambiano il
    # default di ebits a 8 (testa int4 = acceptance ~0%, issue #8) e il ramo grouped e'
    # gated su bits<=4 — combinazioni sorprendenti devono mostrarsi al secondo 1 di un job
    # da ore, non nel size-check dopo. EN: print the RESOLVED plan before doing anything.
    mode = "MTP head only" if a.mtp else "DSA indexer only" if a.indexer else "main model"
    grp = f"grouped gs={a.group_size} (fmt=4)" if (a.group_size and a.ebits <= 4) else \
          (f"PER-ROW (grouped branch needs bits<=4; ebits={a.ebits} disables it)" if a.group_size else "per-row")
    # Cosmetic-only: a pack-quantized source (K2-style compressed-tensors) transcodes its
    # routed experts losslessly (unpack_compressed_int4 + transcode_compressed_int4) --
    # --xbits is never consulted for them, so printing "x {xbits}-bit" here would be
    # misleading. Best-effort local config.json peek, never raises: any failure just
    # falls back to the original "x {xbits}-bit" wording, same as before this note existed.
    # K2.6 nests quantization_config under text_config (flatten_container_config), so this
    # checks there too when the flat key is absent.
    x_note = f"{a.xbits}-bit"
    if a.indir:
        try:
            src_cfg = json.loads(open(os.path.join(a.indir, "config.json")).read())
            src_cfg = flatten_container_config(src_cfg)
            src_fmt = src_cfg.get("quantization_config", {}).get("format")
            if src_fmt in _PACK_QUANT_FORMATS:
                x_note = f"transcode (lossless, source is {src_fmt}; --xbits ignored)"
        except (OSError, ValueError, AttributeError):
            pass
    print(f"[PLAN] mode: {mode} | source: {'local ' + a.indir if a.indir else 'download ' + a.repo} | "
          f"experts {a.ebits}-bit, embed/lm_head {a.io_bits}-bit, x {x_note} | {grp}")

    if a.selftest_nvfp4:
        import torch
        # 1) LUT e2m1: i 16 codici devono decodificare esattamente ai valori attesi.
        lut = torch.tensor(_E2M1, dtype=torch.float32)
        expect = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0]
        assert lut.tolist() == expect, "LUT e2m1 errata"
        print("[nvfp4] LUT e2m1: 16/16 codici OK")
        # 2) round-trip: costruisco un tensore ai SOLI valori rappresentabili (scala nota per
        #    blocco+globale), impacchetto come modelopt, poi dequant deve tornare ESATTO.
        import numpy as np, io
        from safetensors.torch import save as st_save
        from safetensors import safe_open
        rng = np.random.default_rng(0); O, I, GS = 8, 64, 16
        codes = rng.integers(0, 16, size=(O, I)).astype(np.uint8)   # nibble e2m1 casuali
        w4 = np.array(_E2M1, np.float32)[codes]                      # [O,I]
        # scale per-blocco (rappresentabili in f8e4m3) + globale piccola (stile modelopt)
        blk = rng.choice([0.5,1.0,2.0,4.0,8.0], size=(O, I//GS)).astype(np.float32)
        gscale = np.float32(3.9e-5)
        W = w4 * np.repeat(blk, GS, axis=1) * gscale                 # riferimento esatto
        # impacchetto: pari->nibble basso, dispari->alto
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
        import ml_dtypes  # solo per il test: encode f8e4m3 delle scale di blocco
        tens = {name: torch.from_numpy(arr) for name, arr in {
            "w.weight": packed,
            "w.weight_scale": blk.astype(ml_dtypes.float8_e4m3fn).view(np.uint8),  # placeholder
        }.items()}
        # torch non ha un costruttore da bytes f8: passo via file safetensors scritto a mano.
        # piu' semplice: uso direttamente dequant_nvfp4 su un finto 'f' in-memory.
        class _F:
            def __init__(s, d): s.d = d
            def get_tensor(s, n): return s.d[n]
            def get_slice(s, n): return None
        blk_f8 = blk.astype(ml_dtypes.float8_e4m3fn)                 # quantizza le scale a f8
        f = _F({"w.weight": torch.from_numpy(packed),
                "w.weight_scale": torch.from_numpy(blk_f8.view(np.uint8)).view(torch.float8_e4m3fn),
                "w.weight_scale_2": torch.tensor(gscale)})
        got = dequant_nvfp4(f, "w.weight")
        # riferimento con scale gia' quantizzate a f8 (per confronto esatto)
        Wq = w4 * np.repeat(blk_f8.astype(np.float32), GS, axis=1) * gscale
        maxerr = float(np.abs(got - Wq).max())
        print(f"[nvfp4] round-trip encode->dequant: max abs err = {maxerr:.3e} "
              f"({'OK' if maxerr < 1e-9 else 'FAIL'})")
        assert maxerr < 1e-9
        # 3) requant colibri int4 su valori dequantati -> errore piccolo atteso
        q, s = quant_int4(got.astype(np.float32), 4)
        rb = (I + 1)//2; qb = q.reshape(O, rb)
        lo = (qb & 0x0F).astype(np.int32) - 8; hi = ((qb >> 4) & 0x0F).astype(np.int32) - 8
        deq = np.empty((O, I), np.float32); deq[:, 0::2] = lo; deq[:, 1::2] = hi[:, :I-I//2]
        deq = deq * s[:, None]
        rel = np.abs(deq - got).mean() / (np.abs(got).mean() + 1e-12)
        # Informativo, NON un test di uguaglianza: requantizzare int4 per-riga dati che
        # spaziano 16x per il block-scale costa ~0.17 di errore relativo di suo. La soglia
        # larga becca solo una corruzione grossolana, non e' un bound di precisione.
        # EN: informational — per-row int4 requant of 16x-block-range data inherently ~0.17.
        print(f"[nvfp4] dequant->colibri int4->dequant: errore rel medio = {rel:.4f} "
              f"(atteso ~0.17; {'OK' if rel < 0.30 else 'ANOMALO'})")
        assert rel < 0.30, f"requant rel err {rel:.3f} troppo alto: dequant probabilmente corrotto"
        print("[nvfp4] SELFTEST OK")
        return

    if a.selftest:
        import torch
        w = (torch.randn(256, 256) * 0.3)
        O, I = w.shape; bs = 128
        sc = torch.zeros(O // bs, I // bs)
        for bi in range(O // bs):
            for bj in range(I // bs):
                blk = w[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs]
                sc[bi, bj] = blk.abs().max() / 448.0
        q = (w / sc.repeat_interleave(bs,0).repeat_interleave(bs,1)).to(torch.float8_e4m3fn)
        deq = (q.to(torch.float32) * sc.repeat_interleave(bs,0).repeat_interleave(bs,1))
        rel = (deq - w).abs().mean() / w.abs().mean()
        print(f"[selftest fp8 block-dequant] mean relative error = {rel:.4f}  "
              f"({'OK' if rel < 0.05 else 'HIGH'})")
        return

    # n_layers from the checkpoint's OWN config.json, always -- inspection-driven,
    # not gated on arch. --repo writes config.json into outdir's _meta only later
    # (the re-read after download, below, redoes this once that lands).
    cfg_src = a.indir or a.outdir           # --repo path writes config.json into outdir early
    if "--n-layers" not in sys.argv:
        n = read_n_layers_from_config(cfg_src)
        if n: a.n_layers = n

    os.makedirs(a.outdir, exist_ok=True)
    if a.indir:    # conversione locale: PARALLELA e memory-bounded (vedi convert_shard_to_files)
        shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
        # #383: se l'indice c'e', i passaggi --mtp/--indexer convertono SOLO gli shard
        # che contengono i tensori richiesti (3 invece di scandire tutti i 141 — ogni
        # scansione a vuoto apre comunque uno shard da 5 GB). Senza indice: scansione
        # completa come prima.
        # EN: #383: when the index is present, the --mtp/--indexer passes convert ONLY
        # the shards that hold the requested tensors (3 instead of scanning all 141 —
        # every empty scan still opens a 5 GB shard). Without the index: full scan as
        # before.
        if a.mtp or a.indexer:
            idxp = os.path.join(a.indir, "model.safetensors.index.json")
            if os.path.exists(idxp):
                wmap = json.load(open(idxp))["weight_map"]
                if a.mtp:
                    want = {v for k, v in wmap.items() if k.startswith(f"model.layers.{a.n_layers}.")}
                else:
                    want = {v for k, v in wmap.items() if "indexer" in k and 0 <= layer_idx(k) < a.n_layers}
                keep = [sp for sp in shards if os.path.basename(sp) in want]
                print(f"[PLAN] index: {len(keep)}/{len(shards)} local shard(s) hold the requested tensors")
                shards = keep
        # BUG #355: questo ramo ignorava --mtp/--indexer. Con --mtp scriveva
        # out-NNNNN (gli STESSI nomi di una conversione normale) in ebits=8 e
        # keep_mtp=False -> il "secondo passaggio MTP" nella stessa outdir
        # SOVRASCRIVEVA il container gia' finito con una riconversione int8
        # completa, in silenzio (137/141 shard distrutti prima di accorgersene).
        # Ora il ramo locale rispecchia il download path: prefisso corretto,
        # flag passate, shard vuoti saltati.
        prefix = "out-mtp-" if a.mtp else "out-idx-" if a.indexer else "out-"

        # RESUME (#383, now race-free under parallelism): output files are named by the
        # INPUT shard's own index (out-{shard_idx:05d}-{chunk:03d}.safetensors), not a
        # global emission counter -- a shared counter would race across worker
        # processes. A per-shard `{prefix}{shard_idx:05d}.done` marker (written only
        # after ALL of that shard's chunk files are fsync'd + already visible under
        # their final names, see _convert_one_shard_task) is the resume checkpoint: its
        # mere existence means the shard is fully done, whatever chunks it produced
        # (possibly zero, for an --mtp/--indexer pass over a shard with none of the
        # wanted tensors). The params-mixing guard (#355: a resumed run with DIFFERENT
        # bits/group-size/etc. must not silently mix conversions in one outdir) is
        # unchanged in spirit, now backed by the same check_or_record_params() helper
        # the --repo download loops already use, instead of a hand-rolled duplicate.
        # Upgrade guard: an outdir with a PRE-parallel in-progress conversion (older
        # converter version) recorded its resume state in `.{prefix}progress.json` using
        # a global emission counter (out-00000.safetensors, no chunk suffix) -- a naming
        # and resume scheme this version's `.done`-marker logic doesn't recognize at all.
        # Silently proceeding would reprocess every shard fresh under the NEW naming
        # scheme while the OLD counter-named files stay behind unnoticed, and the C
        # engine's glob would then load BOTH old and new copies of the same tensor names.
        # Refuse instead of guessing: the fix is a fresh --outdir (or finish/clean up the
        # old-style conversion with the previous converter version first).
        old_progress = os.path.join(a.outdir, f".{prefix}progress.json")
        if os.path.exists(old_progress):
            print(f"ERROR: {old_progress} exists -- {a.outdir} has an in-progress "
                  "conversion from an OLDER converter version (global-counter output "
                  f"naming, e.g. {prefix}00000.safetensors with no chunk suffix), which "
                  "this parallel/chunked version cannot safely resume in place. Use a "
                  "fresh --outdir, or finish that conversion with the previous converter "
                  "version first.")
            return
        params = {"ebits": a.ebits, "io_bits": a.io_bits, "xbits": a.xbits,
                  "group_size": a.group_size, "n_layers": a.n_layers, "bits_map": bits_map,
                  "proj_bits": dict(PROJ_BITS)}
        if not check_or_record_params(a.outdir, prefix, params): return

        chunk_bytes = None if a.chunk_gb <= 0 else int(a.chunk_gb * (1 << 30))
        jobs = a.jobs if a.jobs > 0 else _auto_jobs(a.chunk_gb)
        proj_bits_snapshot = dict(PROJ_BITS)   # each worker process needs its own copy (see _pool_worker_init)

        tasks, skipped, skipped_with_output = [], 0, 0
        for i, sp in enumerate(shards):
            marker = os.path.join(a.outdir, f"{prefix}{i:05d}.done")
            marker_ok, chunks_recorded = False, []
            if os.path.exists(marker):
                try:
                    chunks_recorded = json.loads(open(marker).read()).get("chunks", [])
                    # Defense in depth beyond the design's baseline "marker exists ->
                    # skip": a marker only counts as done if every chunk file it names is
                    # STILL present. Guards against a chunk file being deleted (by hand, by
                    # a partial cleanup, ...) without also deleting the marker, which would
                    # otherwise silently skip a shard whose on-disk output is incomplete.
                    marker_ok = all(os.path.exists(os.path.join(a.outdir, c)) for c in chunks_recorded)
                except (OSError, ValueError):
                    marker_ok = False   # unparseable marker: safe default is NOT-done -> reprocess
            if marker_ok:
                skipped += 1
                if chunks_recorded: skipped_with_output += 1
                continue
            tasks.append((sp, i, a.outdir, prefix, a.n_layers, a.ebits, a.io_bits, a.xbits,
                          a.mtp, a.indexer, a.group_size, bits_map, chunk_bytes))
        if skipped: print(f"[RESUME] {skipped} shard(s) already done in {a.outdir}, skipped")

        print(f"[PARALLEL] {len(tasks)} shard(s) to convert, --jobs {jobs}, chunk-gb="
              f"{a.chunk_gb if a.chunk_gb > 0 else 'unbounded'}", flush=True)

        results = []
        if jobs <= 1 or len(tasks) <= 1:
            # Sequential, IN-PROCESS, no multiprocessing at all: this is --jobs 1's
            # "reproduce today's exact conversion behavior" path. It runs the exact same
            # _convert_one_shard_task() every parallel worker runs below -- there is one
            # implementation of "convert a shard", not a sequential one and a separate
            # parallel one -- which is what makes --jobs 1 and --jobs N provably produce
            # the same per-shard tensor content (proven by
            # test_kimi_convert.py::test_parallel_matches_sequential).
            for t in tasks:
                r = _convert_one_shard_task(t)
                results.append(r)
                print(f"    -> shard {r[0]:05d} ({r[1]}): {len(r[2])} chunk file(s)", flush=True)
        else:
            import multiprocessing as mp
            # Thread-limiting env vars set in the PARENT before spawning: a spawned
            # child inherits os.environ at process-start time, so these are already
            # present in the child's OS environment when THIS module's top-level
            # `import numpy as np` (and any worker-side `import torch`) runs during the
            # child's bootstrap -- which happens before _pool_worker_init even executes
            # (spawn must import the module to find the task function at all).
            # _pool_worker_init sets them again as a no-op safety net and additionally
            # calls torch.set_num_threads(1), torch's own supported runtime knob for
            # capping its intra-op thread pool. Without this, N worker PROCESSES x M
            # BLAS/torch threads each would oversubscribe the machine's cores. 'spawn'
            # (not the platform default, which is 'fork' on Linux) because torch is not
            # fork-safe.
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[var] = "1"
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=jobs, initializer=_pool_worker_init,
                          initargs=(proj_bits_snapshot,)) as pool:
                for r in pool.imap_unordered(_convert_one_shard_task, tasks):
                    results.append(r)
                    print(f"    -> shard {r[0]:05d} ({r[1]}): {len(r[2])} chunk file(s)", flush=True)

        fresh = sum(1 for r in results if r[2])
        n = fresh + skipped_with_output
        # Sum each shard's own vision-drop tally (r[3]) here in the PARENT: --jobs > 1
        # runs _convert_one_shard_task in separate spawned processes, so this is the
        # only place the per-shard counts can be combined into one total.
        vision_totals = {}
        for r in results:
            for cat, cnt in r[3].items():
                vision_totals[cat] = vision_totals.get(cat, 0) + cnt
        total_dropped = sum(vision_totals.values())
        if total_dropped:
            print(f"[VISION] dropped {total_dropped} tensors "
                  f"({vision_totals.get('vision_tower', 0)} vision_tower, "
                  f"{vision_totals.get('mm_projector', 0)} mm_projector) — text-only container")
        # Metadata step runs ONCE, in the parent, after every worker has finished --
        # config/tokenizer copy+generation only needs to happen once per outdir, not
        # once per shard/worker (and gen_kimi_tokenizer isn't safe to fan out anyway).
        if not a.mtp and not a.indexer:
            _write_metadata(a.indir, a.outdir)
        tag = "MTP" if a.mtp else "indexer" if a.indexer else "main"
        print(f"converted {fresh} {tag} shard(s), {n} in container -> {a.outdir} ({prefix}NNNNN-NNN)")
        return

    # reale: scarica shard per shard, converte, cancella
    # EN: real: download shard by shard, convert, delete
    #
    # ROBUSTEZZA RETE: timeout brevi sulle read cosi' un download appeso FALLISCE invece
    # di restare fermo per sempre. 8s, non 30: "timeout" = ZERO byte ricevuti in quella
    # finestra; su un transfer vivo i chunk arrivano di continuo, quindi 8s e' sicuro e
    # uno stallo costa 8s invece di 30.
    # EN: NETWORK ROBUSTNESS: short read timeouts so a hung download FAILS instead of
    # EN: sitting there forever. 8s, not 30: a "timeout" means ZERO bytes received in that
    # EN: window; a live transfer delivers chunks constantly, so 8s is safe and a stall
    # EN: costs 8s instead of 30.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "8")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
    # log con timestamp: i messaggi "Trying to resume" di hf_hub diventano databili.
    # EN: timestamped logs: hf_hub's "Trying to resume" messages become datable.
    import logging
    logging.basicConfig(format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    # hf_xet si blocca quando la rete si riavvia (connessioni zombie senza timeout):
    # forza la via HTTP classica, che curl ha dimostrato funzionare. (misurato 2026-07-02)
    # EN: hf_xet hangs when the network restarts (zombie connections with no timeout):
    # EN: force the classic HTTP path, which curl proved works (measured 2026-07-02).
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # =0 per riabilitare xet / to re-enable xet
    from huggingface_hub import HfApi, hf_hub_download

    # lock anti-doppione: DUE convertitori sulla stessa outdir si corrompono a vicenda.
    # EN: anti-duplicate lock: TWO converters on the same outdir corrupt each other.
    # fcntl is Unix-only; on Windows use msvcrt or skip locking.
    lock = open(os.path.join(a.outdir, ".convert.lock"), "w")
    try:
        import fcntl
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("ERROR: another converter is already using this output directory. Exiting."); return
    except ImportError:
        try:
            import msvcrt
            try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                print("ERROR: another converter is already using this output directory. Exiting."); return
        except ImportError:
            pass  # no locking available — single-user converter, acceptable

    # dimensioni note dei file, riempite dopo repo_info: il downloader multi-stream le usa
    # per calcolare i confini dei segmenti e per sapere quando un file e' completo.
    # EN: known file sizes, filled after repo_info: the multi-stream downloader uses them
    # EN: to compute segment boundaries and to know when a file is complete.
    SIZES = {}

    def download_retry(repo, fn, dest, tries=999):
        """Downloader multi-stream con resume via Range. Apre N segmenti concorrenti
        (default 2, COLI_DL_STREAMS per cambiarli) e salva lo stato per-segmento in un
        sidecar .seg -> NESSUN byte perso comunque muoia la connessione. Un singolo stream
        HF e' limitato a ~2 MB/s (misurato); 2 stream ~ raddoppiano il throughput senza
        saturare una linea domestica. File piccoli, COLI_DL_STREAMS=1 o un vecchio .part
        legacy -> percorso a stream singolo (_download_single).
        EN: multi-stream Range-resume downloader. Opens N concurrent segments (default 2,
        EN: COLI_DL_STREAMS to change) and saves per-segment state in a .seg sidecar -> NO
        EN: byte is lost however the connection dies. A single HF stream is paced at
        EN: ~2 MB/s (measured); 2 streams roughly double throughput without saturating a
        EN: home line. Small files, COLI_DL_STREAMS=1 or a legacy .part -> single-stream
        EN: path (_download_single)."""
        import time as _t, threading, urllib.request, urllib.error
        url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
        out = os.path.join(dest, fn); part = out + ".part"; side = part + ".seg"
        os.makedirs(dest, exist_ok=True)
        expected = SIZES.get(fn)
        if os.path.exists(out) and (expected is None or os.path.getsize(out) == expected):
            return out
        NS = max(1, min(8, int(os.environ.get("COLI_DL_STREAMS", "2"))))
        # un .part senza sidecar l'ha scritto una versione precedente a stream singolo.
        # EN: a .part without a sidecar was written by an older single-stream version.
        legacy = os.path.exists(part) and not os.path.exists(side)
        if expected is None or expected < (256 << 20) or NS == 1 or legacy:
            return _download_single(url, fn, out, part, expected)
        # ---- multi-stream ----
        segs = [(expected * t // NS, expected * (t + 1) // NS) for t in range(NS)]
        done = [0] * NS
        # riprendi lo stato dei segmenti se il sidecar combacia (stesso N, stessa size).
        # EN: resume per-segment progress if the sidecar matches (same N, same size).
        if os.path.exists(side):
            try:
                st = json.loads(open(side).read())
                if st.get("n") == NS and st.get("size") == expected: done = st["done"]
            except Exception: pass
        if not os.path.exists(part):
            with open(part, "wb") as f: f.truncate(expected)   # file sparse / sparse file
        fd = os.open(part, os.O_WRONLY)
        t0 = _t.time(); nres = [0]; log_lock = threading.Lock(); stopfail = []
        def worker(t):
            s0, s1 = segs[t]
            while done[t] < s1 - s0 and not stopfail:
                pos = s0 + done[t]
                _hdrs = {"User-Agent": "colibri-convert", "Range": f"bytes={pos}-{s1-1}"}
                if os.environ.get("HF_TOKEN"): _hdrs["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
                req = urllib.request.Request(url, headers=_hdrs)
                try:
                    with urllib.request.urlopen(req, timeout=8) as r:
                        if r.status != 206:               # Range ignorato: multi-stream impossibile
                            stopfail.append(t); return    # EN: Range ignored: multi-stream impossible
                        while done[t] < s1 - s0:
                            chunk = r.read(1 << 20)
                            if not chunk: break
                            rem = (s1 - s0) - done[t]     # mai oltre il segmento / never past the segment
                            if len(chunk) > rem: chunk = chunk[:rem]
                            os.pwrite(fd, chunk, s0 + done[t])
                            done[t] += len(chunk)
                except KeyboardInterrupt: raise
                except Exception as ex:
                    with log_lock:
                        nres[0] += 1
                        print(f"    [dl] s{t}: {type(ex).__name__} at {(s0+done[t])/1e9:.2f} GB: "
                              f"resuming (#{nres[0]})", flush=True)
                    _t.sleep(min(15, 1 + nres[0] // NS))
        th = [threading.Thread(target=worker, args=(t,), daemon=True) for t in range(NS)]
        for x in th: x.start()
        print(f"    [dl {_t.strftime('%H:%M:%S')}] connected: {NS} streams, "
              f"{sum(done)/1e9:.2f} of {expected/1e9:.2f} GB", flush=True)
        mark = sum(done); tmark = t0
        while any(x.is_alive() for x in th):
            _t.sleep(5)
            have = sum(done)
            tmpside = side + ".tmp"                       # checkpoint atomico / atomic checkpoint
            open(tmpside, "w").write(json.dumps({"n": NS, "size": expected, "done": list(done)}))
            os.replace(tmpside, side)
            now = _t.time()
            if now - tmark >= 30:
                print(f"    [dl {_t.strftime('%H:%M:%S')}] {have/1e9:5.2f} GB "
                      f"({(have-mark)/max(now-tmark,1e-9)/1e6:5.1f} MB/s, {NS} stream)", flush=True)
                mark = have; tmark = now
        os.close(fd)
        if stopfail:                                      # il server non onora il Range: fallback
            for f2 in (part, side):                       # EN: server won't honor Range: fall back
                if os.path.exists(f2): os.remove(f2)
            return _download_single(url, fn, out, part, expected)
        assert sum(done) == expected
        if os.path.exists(side): os.remove(side)
        os.replace(part, out)
        dt = max(_t.time() - t0, 1e-9)
        print(f"    [dl] {fn}: {expected/1e9:.2f} GB in {dt/60:.1f} min "
              f"({expected/dt/1e6:.1f} MB/s avg, {NS} streams, {nres[0]} resumes)", flush=True)
        return out

    def _download_single(url, fn, out, part, expected):
        """Percorso a stream singolo con resume via Range (file piccoli / .part legacy /
        COLI_DL_STREAMS=1). Un EOF corto ma pulito conta come ripresa; se non arriva
        NESSUN byte nuovo, backoff invece di girare a vuoto.
        EN: single-stream path with Range resume (small files / legacy .part /
        EN: COLI_DL_STREAMS=1). A clean short EOF counts as a resume; if NO new byte
        EN: arrives, back off instead of spinning."""
        import time as _t, urllib.request, urllib.error
        t0 = _t.time(); nres = 0; mark = 0; tmark = t0
        while True:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if expected is not None and have >= expected: break
            have0 = have
            req = urllib.request.Request(url, headers={"User-Agent": "colibri-convert"})
            if have: req.add_header("Range", f"bytes={have}-")
            if os.environ.get("HF_TOKEN"): req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
            try:
                with urllib.request.urlopen(req, timeout=8) as r:
                    if have and r.status == 200:          # server ha ignorato il Range: riparti pulito
                        have = 0                          # EN: server ignored Range: restart clean
                    if expected is None:
                        cl = r.headers.get("Content-Length")
                        if cl: expected = have + int(cl)
                    if have == 0 or nres:                 # segnale di vita subito / immediate sign of life
                        print(f"    [dl {_t.strftime('%H:%M:%S')}] connected"
                              f"{f' @ {have/1e9:.2f} GB' if have else ''}"
                              f"{f' of {expected/1e9:.2f} GB' if expected else ''}", flush=True)
                    with open(part, "ab" if have else "wb") as f:
                        if not have: f.truncate(0)
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk: break
                            f.write(chunk); have += len(chunk)
                            if have - mark >= 512 * 1024 * 1024 or _t.time() - tmark >= 30:
                                now = _t.time()
                                print(f"    [dl {_t.strftime('%H:%M:%S')}] {have/1e9:5.2f} GB "
                                      f"({(have-mark)/max(now-tmark,1e-9)/1e6:5.1f} MB/s)", flush=True)
                                mark = have; tmark = now
                if expected is None: break                # lunghezza ignota: passata singola / unknown length
                if have < expected:                       # EOF corto ma pulito: conta come ripresa
                    nres += 1                             # EN: clean short EOF: counts as a resume
                    if have == have0: _t.sleep(min(15, 1 + nres))   # zero progresso -> backoff / zero progress -> back off
            except KeyboardInterrupt: raise
            except urllib.error.HTTPError as ex:
                if ex.code == 416: break                  # gia' completo / already complete
                nres += 1
                print(f"    [dl] HTTP {ex.code} at {have/1e9:.2f} GB: resuming (#{nres})", flush=True)
                _t.sleep(min(15, 1 + nres))
            except Exception as ex:
                nres += 1
                print(f"    [dl] {type(ex).__name__} at {have/1e9:.2f} GB: resuming (#{nres})", flush=True)
                _t.sleep(min(15, 1 + nres))
        os.replace(part, out)
        dt = max(_t.time() - t0, 1e-9); sz = os.path.getsize(out)
        print(f"    [dl] {fn}: {sz/1e9:.2f} GB in {dt/60:.1f} min "
              f"({sz/dt/1e6:.1f} MB/s avg, {nres} resumes)", flush=True)
        return out

    from safetensors.numpy import save_file
    import time as _t
    info = None
    for att in range(10):
        try:
            info = HfApi().repo_info(a.repo, files_metadata=True)
            # dimensioni note dallo store: abilitano il download multi-stream a segmenti.
            # EN: sizes known from the store: enable segmented multi-stream download.
            SIZES.update({s.rfilename: s.size for s in info.siblings if s.size})
            break
        except KeyboardInterrupt: raise
        except Exception as ex:
            w = min(60, 5*(att+1)); print(f"repo_info failed ({type(ex).__name__}); retrying in {w}s", flush=True); _t.sleep(w)
    if info is None:
        print("ERROR: could not reach the repository after 10 retries. Check your network and repo name.", flush=True)
        return
    shards = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
    if not shards:
        print("ERROR: no .safetensors shards found in this repository.", flush=True)
        return
    meta_dir = os.path.join(a.outdir, "_meta"); os.makedirs(meta_dir, exist_ok=True)
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
               "tiktoken.model"]:
        # hf_hub_download(..., local_dir=meta_dir) already returns meta_dir/fn; a shutil.copy
        # of that path onto meta_dir was copying the file onto itself (SameFileError, silently
        # swallowed by the except below) -- just download, nothing left to copy.
        try: hf_hub_download(a.repo, fn, local_dir=meta_dir)
        except Exception: pass
    # re-read n_layers now that config.json is present (repo path): the early read
    # above ran before download, when meta_dir's config.json didn't exist yet.
    if "--n-layers" not in sys.argv:
        n = read_n_layers_from_config(meta_dir)
        if n: a.n_layers = n
    _write_metadata(meta_dir, a.outdir)
    tmp = os.path.join(a.outdir, "_inflight"); os.makedirs(tmp, exist_ok=True)
    if a.mtp:
        params = {"ebits": a.ebits, "io_bits": a.io_bits, "xbits": a.xbits,
                  "group_size": a.group_size, "n_layers": a.n_layers, "bits_map": bits_map,
                  "proj_bits": dict(PROJ_BITS)}
        if not check_or_record_params(a.outdir, "out-mtp-", params): return
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        pref = f"model.layers.{a.n_layers}."
        mtp_shards = sorted(set(v for k, v in idx.items() if k.startswith(pref)))
        print(f"[MTP] head at layer {a.n_layers}: {len(mtp_shards)} shards to process: {mtp_shards}")
        for i, sh in enumerate(mtp_shards):
            outp = os.path.join(a.outdir, f"out-mtp-{i:05d}.safetensors")
            if os.path.exists(outp): print(f"[MTP] {outp} already done"); continue
            print(f"[MTP {i+1}/{len(mtp_shards)}] downloading {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_mtp=True, group_size=a.group_size, bits_map=bits_map)
            save_file(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB, {len(out)} tensors)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[MTP] DONE."); return
    if a.indexer:
        params = {"ebits": a.ebits, "io_bits": a.io_bits, "xbits": a.xbits,
                  "group_size": a.group_size, "n_layers": a.n_layers, "bits_map": bits_map,
                  "proj_bits": dict(PROJ_BITS)}
        if not check_or_record_params(a.outdir, "out-idx-", params): return
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        idx_shards = sorted(set(v for k, v in idx.items()
                                if "indexer" in k and 0 <= layer_idx(k) < a.n_layers))
        tot_gb = len(idx_shards) * 5.4
        print(f"[IDX] indexer weights across {len(idx_shards)} shards (~{tot_gb:.0f} GB total download, resumable)")
        for i, sh in enumerate(idx_shards):
            outp = os.path.join(a.outdir, f"out-idx-{i:05d}.safetensors")
            if os.path.exists(outp): continue             # gia' fatto -> ripartibile
            print(f"[IDX {i+1}/{len(idx_shards)}] downloading {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_idx=True, group_size=a.group_size, bits_map=bits_map)
            if out: save_file(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({len(out)} tensors)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[IDX] DONE."); return
    params = {"ebits": a.ebits, "io_bits": a.io_bits, "xbits": a.xbits,
              "group_size": a.group_size, "n_layers": a.n_layers, "bits_map": bits_map,
              "proj_bits": dict(PROJ_BITS)}
    if not check_or_record_params(a.outdir, "out-", params): return
    vision_totals = {}   # single-process download loop -- a plain shared dict is fine here
    # PIPELINE: while shard i converts (CPU/disk-bound, network idle), shard i+1
    # downloads in a background thread -- without this the link sits dead for the
    # whole conversion of every shard (~40-50% of wall time on a fast line).
    # download_retry is segment-resumable and writes only its own blob/.seg files,
    # so the worst a crash costs is re-joining a partial. The prefetch is skipped
    # when disk headroom couldn't hold blob + output + margin.
    import threading as _thr
    pre = {"sh": None, "path": None, "err": None, "t": None}
    def _prefetch(sh_next):
        try:
            pre["path"] = download_retry(a.repo, sh_next, tmp)
        except BaseException as e:                    # main loop retries synchronously
            pre["err"] = e
    def _next_missing(after):
        for j in range(after + 1, len(shards)):
            if not os.path.exists(os.path.join(a.outdir, f"out-{j:05d}.safetensors")):
                return j
        return -1
    for i, sh in enumerate(shards):
        if free_gb(a.outdir) < a.min_free_gb:
            print(f"STOP: free space is below {a.min_free_gb} GB. Free space and rerun to resume."); break
        outp = os.path.join(a.outdir, f"out-{i:05d}.safetensors")
        if os.path.exists(outp): continue                 # gia' fatto -> ripartibile
        if pre["t"]: pre["t"].join()
        if pre["sh"] == sh and pre["err"] is None and pre["path"]:
            p = pre["path"]                               # prefetched while the previous shard converted
        else:
            print(f"[{i+1}/{len(shards)}] downloading {sh} ({free_gb(a.outdir):.0f} GB free)...", flush=True)
            p = download_retry(a.repo, sh, tmp)           # resumes any prefetch partial
        pre.update(sh=None, path=None, err=None, t=None)
        nj = _next_missing(i)
        if nj >= 0 and free_gb(a.outdir) > a.min_free_gb + 40:
            print(f"[{i+1}/{len(shards)}] converting {sh} while prefetching {shards[nj]}...", flush=True)
            pre["sh"] = shards[nj]
            pre["t"] = _thr.Thread(target=_prefetch, args=(shards[nj],), daemon=True)
            pre["t"].start()
        out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, group_size=a.group_size,
                                 bits_map=bits_map, vision_counts=vision_totals)
        save_file(out, outp)
        os.remove(p)                                       # <-- cancella subito lo shard fp8
        for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
            # never touch the in-flight prefetch blob or its .seg resume sidecars
            if pre["sh"] and pre["sh"] in os.path.basename(blob): continue
            if os.path.isfile(blob): os.remove(blob)
        print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB)", flush=True)
    if pre["t"]: pre["t"].join()                           # never orphan a downloader thread
    shutil.rmtree(tmp, ignore_errors=True)
    total_dropped = sum(vision_totals.values())
    if total_dropped:
        print(f"[VISION] dropped {total_dropped} tensors "
              f"({vision_totals.get('vision_tower', 0)} vision_tower, "
              f"{vision_totals.get('mm_projector', 0)} mm_projector) — text-only container")
    print("DONE." if i == len(shards)-1 else "INTERRUPTED (rerun to resume).")

if __name__ == "__main__":
    main()
