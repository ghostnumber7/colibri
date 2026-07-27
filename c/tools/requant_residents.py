"""Requantize RESIDENT tensor classes of a finished colibri container IN PLACE
(int8 per-row -> int4-g64, fmt=4), with exact byte-level revert -- the A/B lever
for K3's resident-RAM-bandwidth problem without a second 1.35 TB container.

Why in place: a sibling container does not fit on disk next to the K3 one, and
only the residents change -- the mxfp4 experts (the bulk) are untouched. Why
revertible: the whole point is SCORE A/B (baseline int8 vs int4-g64 per class);
theory says the KDA projections dominate per-token RAM traffic and should take
int4-g64 like GLM's residents, but GLM's ablation also measured a real quality
cost (-2.2..-3.4pp acc_norm), so every class must be measured, kept or reverted.

Mechanics per shard (out-*.safetensors):
  - selected tensors currently stored int8 per-row (u8 codes [O*I] + f32 .qs [O])
    are dequantized and requantized with the converter's OWN quant_int4_grouped
    (identical math to a --ebits 4 --group-size 64 conversion);
  - the ORIGINAL weight+.qs tensors are saved to <shard>.residbak.safetensors
    BEFORE the shard is rewritten (tmp + atomic rename), so --revert restores
    byte-identical originals;
  - non-selected tensors are copied through byte-for-byte.

Classes (name-pattern -> engine role; KDA-vs-MLA layer kinds come from config):
  kdaqkv   q/k/v_proj on KDA layers          (the biggest RAM term: ~264MB/layer)
  kdagate  g_proj + f_b_proj on KDA layers   (~90MB/layer)
  gate     g_proj on MLA layers              (output gate)
  attn     q_a/q_b/kv_a on MLA layers
  kvb      kv_b_proj                          (KV reconstruction -- sensitive)
  o        o_proj everywhere
  sh       shared experts
  lat      routed_expert_{down,up}_proj       (every-token latent bracket)
  dmlp     dense mlp (first_k_dense_replace layers)
  io       embed_tokens + lm_head

Usage:
  requant_residents.py --container DIR --classes kdaqkv,kdagate      # apply
  requant_residents.py --container DIR --revert                      # restore all
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))
from convert_fp8_to_int4 import quant_int4_grouped          # noqa: E402

from safetensors.numpy import load_file, save_file          # noqa: E402

GS = 64
BAK_SUFFIX = ".residbak.safetensors"


def load_cfg(container):
    return json.load(open(Path(container) / "config.json"))


def kda_set(cfg):
    la = cfg.get("linear_attn_config") or {}
    return {i - 1 for i in la.get("kda_layers", [])}     # 1-indexed list


def tensor_class(name, kda):
    """Class label for a container tensor name, or None (untouchable)."""
    m = re.match(r"model\.layers\.(\d+)\.(.+)", name)
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "io"
    if not m:
        return None
    li, rest = int(m.group(1)), m.group(2)
    is_kda = li in kda
    if rest in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight"):
        return "kdaqkv" if is_kda else None
    if rest in ("self_attn.g_proj.weight", "self_attn.f_b_proj.weight"):
        return "kdagate" if is_kda else "gate"
    if rest in ("self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight"):
        return "attn"
    if rest == "self_attn.kv_b_proj.weight":
        return "kvb"
    if rest == "self_attn.o_proj.weight":
        return "o"
    if ".shared_experts." in rest and rest.endswith("_proj.weight"):
        return "sh"
    if rest.endswith(("routed_expert_down_proj.weight", "routed_expert_up_proj.weight")):
        return "lat"
    if rest in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
        return "dmlp"
    return None


def is_int8_pair(tensors, name):
    """True iff `name` is stored int8 per-row: u8 codes + f32 .qs with O entries."""
    qs = tensors.get(name + ".qs")
    w = tensors.get(name)
    return (w is not None and qs is not None and w.dtype == np.uint8
            and qs.dtype == np.float32 and w.size % qs.size == 0
            and w.size // qs.size > 1)


def apply(container, classes):
    cfg = load_cfg(container)
    kda = kda_set(cfg)
    total_before = total_after = n_req = 0
    for shard in sorted(glob.glob(str(Path(container) / "out-*.safetensors"))):
        if shard.endswith(BAK_SUFFIX):
            continue
        tensors = load_file(shard)
        picked = {}
        for name in list(tensors):
            if name.endswith(".qs"):
                continue
            cl = tensor_class(name, kda)
            if cl in classes and is_int8_pair(tensors, name):
                picked[name] = cl
        if not picked:
            continue
        bak_path = shard + BAK_SUFFIX
        if os.path.exists(bak_path):
            print(f"[requant] {os.path.basename(shard)}: backup already exists -- "
                  "already requantized? skipping (revert first to re-apply)")
            continue
        # backup FIRST, fsync'd, so a crash between backup and rewrite loses nothing
        bak = {}
        for name in picked:
            bak[name] = tensors[name]
            bak[name + ".qs"] = tensors[name + ".qs"]
        save_file(bak, bak_path)
        for name, cl in picked.items():
            w = tensors[name]; qs = tensors[name + ".qs"]
            O = qs.size; I = w.size // O
            deq = w.reshape(O, I).view(np.int8).astype(np.float32) * qs[:, None]
            q, s = quant_int4_grouped(deq, 4, GS)
            total_before += w.nbytes + qs.nbytes
            total_after += q.nbytes + s.nbytes
            tensors[name] = q
            tensors[name + ".qs"] = s
            n_req += 1
        tmp = shard + ".tmp"
        save_file(tensors, tmp)
        os.replace(tmp, shard)
        print(f"[requant] {os.path.basename(shard)}: {len(picked)} tensors "
              f"({', '.join(sorted(set(picked.values())))})", flush=True)
    print(f"[requant] DONE: {n_req} tensors, {total_before/1e9:.2f} GB -> "
          f"{total_after/1e9:.2f} GB (residents only; experts untouched)")


def revert(container):
    n = 0
    for bak_path in sorted(glob.glob(str(Path(container) / ("out-*" + BAK_SUFFIX)))):
        shard = bak_path[:-len(BAK_SUFFIX)]
        tensors = load_file(shard)
        bak = load_file(bak_path)
        tensors.update(bak)                     # byte-identical originals
        tmp = shard + ".tmp"
        save_file(tensors, tmp)
        os.replace(tmp, shard)
        os.remove(bak_path)
        n += len([k for k in bak if not k.endswith(".qs")])
        print(f"[revert] {os.path.basename(shard)}: {len(bak)//2} tensors restored", flush=True)
    print(f"[revert] DONE: {n} tensors restored to original int8 bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--classes", default="",
                    help="comma list: kdaqkv,kdagate,gate,attn,kvb,o,sh,lat,dmlp,io")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    if a.revert:
        revert(a.container)
        return
    classes = {c.strip() for c in a.classes.split(",") if c.strip()}
    valid = {"kdaqkv", "kdagate", "gate", "attn", "kvb", "o", "sh", "lat", "dmlp", "io"}
    bad = classes - valid
    if bad or not classes:
        raise SystemExit(f"--classes needed; unknown: {sorted(bad)}; valid: {sorted(valid)}")
    apply(a.container, classes)


if __name__ == "__main__":
    main()
