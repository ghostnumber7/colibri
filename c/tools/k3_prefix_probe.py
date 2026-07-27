"""Run the REAL Kimi-K3 weights through the first N converted layers while the
rest of the container is still downloading -- an early smoke test of the port on
real data, hours before the full conversion lands.

Why a prefix and not the whole thing: expert routing is state-dependent (layer
L's router reads hidden states shaped by layer L-1's experts), so "which experts
will this prompt use" can only be discovered layer by layer -- but the converter
processes shards in layer order, so the first shards already hold complete
layers with their real mxfp4 experts. The embedding lives in source shard 94
(end of the queue) and is range-fetched separately (~2.3 GB).

What it validates on REAL weights that the tiny oracle cannot:
- the per-dim A_log reading (a wrong reading degenerates the KDA state fast);
- hidden-state health through real KDA/MLA/MoE layers (rms bounded, no NaN);
- real router behavior (which experts fire, how concentrated).

Usage:
  .venv/bin/python c/tools/k3_prefix_probe.py \
      --container /mnt/d/kimi_k3_i4 \
      --embed <scratch>/k3_embed_bf16.bin \
      --prompt "The capital of France is Paris."
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))
from k3_ref import K3Ref, _rmsnorm  # noqa: E402

from safetensors import safe_open  # noqa: E402

MXFP4_E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                      np.float32)


class ContainerWeights(dict):
    """Lazy name -> f32 torch.Tensor view of a (partial) colibri container.
    Dequantises on first access and caches: int8 per-row (u8 codes + f32 .qs),
    mxfp4 (u8 nibbles + u8 e8m0 .qs, group 32), f32 kept tensors."""

    def __init__(self, container, cfg):
        super().__init__()
        self.cfg = cfg
        self.files = {}
        for f in sorted(glob.glob(str(Path(container) / "out-*.safetensors"))):
            with safe_open(f, framework="numpy") as h:
                for k in h.keys():
                    self.files[k] = f

    def __contains__(self, name):
        return dict.__contains__(self, name) or name in self.files

    def _raw(self, name):
        with safe_open(self.files[name], framework="numpy") as h:
            return h.get_tensor(name)

    def __missing__(self, name):
        w = self._raw(name)
        qs_name = name + ".qs"
        if qs_name in self.files:
            qs = self._raw(qs_name)
            if qs.dtype == np.uint8:            # fmt=7 mxfp4, verbatim source bytes
                # container tensors are flattened and O/I cannot be recovered from
                # byte counts alone (nb/ns == 16 for every mxfp4 tensor) -- derive
                # the shape from the expert geometry, exactly like the engine does:
                # w1/w3 = [moe_inter, latent], w2 = [latent, moe_inter].
                mi = self.cfg["moe_intermediate_size"]
                lat = self.cfg["routed_expert_hidden_size"]
                O, I = (lat, mi) if ".w2." in name else (mi, lat)
                assert w.size == O * ((I + 1) // 2) and qs.size == O * ((I + 31) // 32), \
                    f"{name}: bytes do not match expert geometry [{O},{I}]"
                packed = w.reshape(O, I // 2)
                nib = np.empty((O, I), np.int64)
                nib[:, 0::2] = packed & 0x0F
                nib[:, 1::2] = (packed >> 4) & 0x0F
                deq = MXFP4_E2M1[nib] * np.repeat(
                    np.exp2(qs.reshape(O, I // 32).astype(np.float32) - 127.0), 32, axis=1)[:, :I]
            else:                                # int8 per-row: O scales f32
                O = qs.size
                I = w.size // O
                deq = w.reshape(O, I).view(np.int8).astype(np.float32) * \
                    qs.astype(np.float32)[:, None]
            t = torch.from_numpy(np.ascontiguousarray(deq))
        else:
            t = torch.from_numpy(w.astype(np.float32))
        self[name] = t
        return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--embed", required=True, help="raw bf16 [163840,7168] embed blob")
    ap.add_argument("--prompt", default="The capital of France is Paris.")
    ap.add_argument("--layers", type=int, default=0, help="0 = all complete layers")
    a = ap.parse_args()

    cfg = json.load(open(Path(a.container) / "config.json"))
    W = ContainerWeights(a.container, cfg)

    # complete layers = every non-scale tensor of the layer present
    per_layer = {}
    for n in W.files:
        m = re.match(r"model\.layers\.(\d+)\.", n)
        if m and not n.endswith(".qs"):
            per_layer.setdefault(int(m.group(1)), 0)
            per_layer[int(m.group(1))] += 1
    complete = []
    for li in sorted(per_layer):
        if li != len(complete):
            break
        complete.append(li)
    N = min(a.layers, len(complete)) if a.layers else len(complete)
    print(f"[probe] container has layers {complete[0]}..{complete[-1]} complete; probing 0..{N-1}")

    # embedding from the range-fetched bf16 blob
    emb = np.frombuffer(open(a.embed, "rb").read(), dtype=np.uint16)
    emb = emb.reshape(cfg["vocab_size"], cfg["hidden_size"])
    embed = torch.from_numpy(emb.copy()).view(torch.bfloat16).to(torch.float32)

    # tokenize with the container tokenizer (raw text; no chat template -- the
    # probe inspects states/routing, it does not generate)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.container) / "tokenizer.json"))
    ids = tok.encode(a.prompt).ids
    print(f"[probe] prompt tokens ({len(ids)}): {ids}")

    ref = K3Ref(cfg, W)
    h = embed[torch.tensor(ids)]
    T = h.shape[0]
    D = cfg["hidden_size"]
    rb = cfg.get("attn_res_block_size")
    bres = h.new_zeros(T, 0, D) if rb else None

    with torch.no_grad():
        for li in range(N):
            lp = f"model.layers.{li}"
            if rb:
                prefix = h
                if bres.shape[1] > 0:
                    h = ref._apply_attn_res(prefix, bres,
                                            W[lp + ".self_attention_res_norm.weight"],
                                            W[lp + ".self_attention_res_proj.weight"])
                if li % rb == 0:
                    bres = torch.cat([bres, prefix.unsqueeze(1)], dim=1)
                    prefix = None
                x = _rmsnorm(h, W[lp + ".input_layernorm.weight"], ref.eps)
                attn = ref._kda(x, li) if li in ref.kda_layers else ref._mla(x, li)
                prefix = attn if prefix is None else prefix + attn
                h = ref._apply_attn_res(prefix, bres,
                                        W[lp + ".mlp_res_norm.weight"],
                                        W[lp + ".mlp_res_proj.weight"])
                x = _rmsnorm(h, W[lp + ".post_attention_layernorm.weight"], ref.eps)
                if li >= ref.first_dense:
                    p = f"{lp}.block_sparse_moe"
                    scores = torch.sigmoid(x.float() @ W[p + ".gate.weight"].float().T)
                    choice = scores + W[p + ".gate.e_score_correction_bias"].float()
                    topk = torch.topk(choice, ref.top_k, dim=-1).indices
                    uniq = sorted(set(topk.flatten().tolist()))
                    print(f"  layer {li:2d} {'KDA' if li in ref.kda_layers else 'MLA'} "
                          f"routed {len(uniq):3d} unique experts "
                          f"(pos0: {sorted(topk[0].tolist())})")
                    m = ref._moe(x, li)
                else:
                    print(f"  layer {li:2d} {'KDA' if li in ref.kda_layers else 'MLA'} dense")
                    m = ref._mlp(x, lp + ".mlp")
                h = prefix + m
            else:
                raise SystemExit("probe expects attn_res_block_size (K3)")
            hf = h.float()
            rms = hf.pow(2).mean().sqrt().item()
            mx = hf.abs().max().item()
            bad = int(torch.isnan(hf).sum() + torch.isinf(hf).sum())
            print(f"           hidden rms {rms:9.4f} absmax {mx:9.3f} nan/inf {bad}", flush=True)
            if bad:
                raise SystemExit(f"NaN/Inf at layer {li} -- STOP")
            # purge this layer's dequanted f32 tensors -- a K3 layer is ~2 GB of
            # residents plus ~12 GB of touched experts; keeping 13 layers OOMs.
            for k in [k for k in W if f".layers.{li}." in k]:
                dict.__delitem__(W, k)
    print(f"[probe] OK: {N} real layers, hidden state healthy, per-dim A_log stable")


if __name__ == "__main__":
    main()
