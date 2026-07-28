"""Run a REAL Kimi-K3 prompt end-to-end while the container is still
downloading: demand-driven, fetch-on-miss weight streaming.

The insight (and its limit): expert routing is state-dependent, so the expert
set is discovered layer by layer -- but that is exactly enough, because each
layer's forward only needs the 16 routed experts per position (~94 MB) plus
that layer's residents (~0.9 GB), not the layer's full 896-expert shard
(~17 GB). At TENSOR granularity (HTTP range requests against the source
shards) a first prompt costs ~90 GB instead of 1.45 TB. At SHARD granularity
prioritization buys nothing -- every MoE layer's shard would be needed anyway.

Weight resolution order per tensor:
  1. the local converted container (finished layers are free);
  2. a local fetch cache (previous runs' ranged downloads);
  3. an HTTP range request against the source repo (needs HF_TOKEN in env).

Generation is greedy full-recompute per token (no incremental cache): fine
for a handful of smoke-test tokens; the per-layer weight cache is purged each
layer to stay inside RAM, but the on-disk fetch cache makes later forwards
cheap on bandwidth.

Usage (token comes from the interactive env):
  bash -ic 'export HF_TOKEN; .venv/bin/python c/tools/k3_remote_run.py \
      --container /path/to/kimi_k3_i4 \
      --embed <scratch>/k3_embed_bf16.bin \
      --cache <scratch>/k3_fetch_cache \
      --prompt "The capital of France is" --ngen 3'
"""
import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))
from k3_prefix_probe import ContainerWeights, MXFP4_E2M1   # noqa: E402
from k3_ref import K3Ref, _rmsnorm                          # noqa: E402

REPO = "moonshotai/Kimi-K3"
BASE = f"https://huggingface.co/{REPO}/resolve/main"


class RemoteStore:
    """Ranged reads of individual tensors from the source repo's shards.
    Shard headers are fetched once and cached; tensor bytes are cached on disk
    so a re-run never refetches."""

    def __init__(self, index_path, cache_dir):
        self.wmap = json.load(open(index_path))["weight_map"]
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.headers = {}
        self.token = os.environ.get("HF_TOKEN", "")
        self.fetched_bytes = 0

    def _get(self, url, lo, hi):
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes={lo}-{hi}")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def _header(self, shard):
        if shard not in self.headers:
            hp = self.cache / (shard + ".header.json")
            if not hp.exists():
                n = struct.unpack("<Q", self._get(f"{BASE}/{shard}", 0, 7))[0]
                hp.write_bytes(self._get(f"{BASE}/{shard}", 8, 8 + n - 1))
            self.headers[shard] = json.loads(hp.read_text())
        return self.headers[shard]

    def raw(self, src_name):
        """Raw bytes + (dtype, shape) of a SOURCE-name tensor, disk-cached."""
        shard = self.wmap[src_name]
        h = self._header(shard)[src_name]
        fp = self.cache / (src_name.replace("/", "_") + ".bin")
        if not fp.exists():
            # data offsets are relative to the data section: 8 + header_len
            hp = self.cache / (shard + ".hlen")
            if not hp.exists():
                hp.write_text(str(struct.unpack("<Q", self._get(f"{BASE}/{shard}", 0, 7))[0]))
            base = 8 + int(hp.read_text())
            o = h["data_offsets"]
            data = self._get(f"{BASE}/{shard}", base + o[0], base + o[1] - 1)
            fp.write_bytes(data)
            self.fetched_bytes += len(data)
            print(f"    [fetch] {src_name} {len(data)/1e6:.1f} MB "
                  f"(total {self.fetched_bytes/1e9:.2f} GB)", flush=True)
        return fp.read_bytes(), h["dtype"], h["shape"]

    def tensor_f32(self, src_name):
        raw, dt, shape = self.raw(src_name)
        if dt == "BF16":
            u = np.frombuffer(raw, np.uint16).reshape(shape)
            return torch.from_numpy(u.copy()).view(torch.bfloat16).to(torch.float32)
        if dt == "F32":
            return torch.from_numpy(np.frombuffer(raw, np.float32).reshape(shape).copy())
        raise ValueError(f"{src_name}: unexpected dtype {dt}")


class HybridWeights(ContainerWeights):
    """Local converted container first, then remote fetch-on-miss."""

    def __init__(self, container, cfg, remote):
        super().__init__(container, cfg)
        self.remote = remote

    def __contains__(self, name):
        return True   # everything resolvable, one way or another

    def __missing__(self, name):
        if name in self.files:                       # local converted tensor
            return ContainerWeights.__missing__(self, name)
        src = "language_model." + name
        if src in self.remote.wmap:                  # remote bf16/f32 resident
            t = self.remote.tensor_f32(src)
            self[name] = t
            return t
        # remote mxfp4 expert: <base>.weight -> weight_packed + weight_scale
        if name.endswith(".weight"):
            base = "language_model." + name[:-len(".weight")]
            praw, pdt, pshape = self.remote.raw(base + ".weight_packed")
            sraw, sdt, sshape = self.remote.raw(base + ".weight_scale")
            assert pdt == "U8" and sdt == "U8", (name, pdt, sdt)
            packed = np.frombuffer(praw, np.uint8).reshape(pshape)
            scale = np.frombuffer(sraw, np.uint8).reshape(sshape)
            O, Ih = packed.shape
            I = Ih * 2
            nib = np.empty((O, I), np.int64)
            nib[:, 0::2] = packed & 0x0F
            nib[:, 1::2] = (packed >> 4) & 0x0F
            deq = MXFP4_E2M1[nib] * np.repeat(
                np.exp2(scale.astype(np.float32) - 127.0), 32, axis=1)[:, :I]
            t = torch.from_numpy(deq)
            self[name] = t
            return t
        raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--index", default=None, help="source model.safetensors.index.json")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ngen", type=int, default=3)
    a = ap.parse_args()

    cfg = json.load(open(Path(a.container) / "config.json"))
    idx = a.index or str(Path(a.cache) / "index.json")
    if not Path(idx).exists():
        raise SystemExit("pass --index (model.safetensors.index.json)")
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    remote = RemoteStore(idx, a.cache)
    W = HybridWeights(a.container, cfg, remote)

    emb = np.frombuffer(open(a.embed, "rb").read(), dtype=np.uint16)
    embed = torch.from_numpy(emb.reshape(cfg["vocab_size"], cfg["hidden_size"]).copy()) \
        .view(torch.bfloat16).to(torch.float32)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.container) / "tokenizer.json"))
    tmpl = (f'<|open|>message role="user"<|sep|>{a.prompt}<|close|>message<|sep|>'
            f'<|end_of_msg|><|open|>message role="assistant"<|sep|><|open|>response<|sep|>')
    ids = tok.encode(tmpl).ids
    print(f"[run] prompt tokens: {len(ids)}", flush=True)

    ref = K3Ref(cfg, W)
    NL = cfg["num_hidden_layers"]
    rb = cfg.get("attn_res_block_size")
    lm_head = None
    stops = {163585, 163586, 163593}   # [EOS], <|end_of_msg|>, [EOT]

    for step in range(a.ngen):
        h = embed[torch.tensor(ids)]
        T, D = h.shape
        bres = h.new_zeros(T, 0, D)
        with torch.no_grad():
            for li in range(NL):
                lp = f"model.layers.{li}"
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
                    # The router tells us THIS layer's full expert set the moment it
                    # runs -- prefetch every missing expert in parallel (the sequential
                    # fetch-on-miss inside _moe is latency-bound at ~2 MB/s; 16-way
                    # parallel ranged GETs are bandwidth-bound instead).
                    p = f"{lp}.block_sparse_moe"
                    scores = torch.sigmoid(x.float() @ W[p + ".gate.weight"].float().T)
                    choice = scores + W[p + ".gate.e_score_correction_bias"].float()
                    topk = torch.topk(choice, ref.top_k, dim=-1).indices
                    uniq = sorted(set(topk.flatten().tolist()))
                    missing = []
                    for e in uniq:
                        for mat in ("w1", "w3", "w2"):
                            nm = f"{p}.experts.{e}.{mat}.weight"
                            if not dict.__contains__(W, nm) and nm not in W.files:
                                src = "language_model." + nm[:-len(".weight")]
                                missing += [src + ".weight_packed", src + ".weight_scale"]
                    if missing:
                        for sh in {remote.wmap[n] for n in missing}:
                            remote._header(sh)        # headers single-threaded (dict/file writes)
                        from concurrent.futures import ThreadPoolExecutor
                        with ThreadPoolExecutor(16) as ex:
                            list(ex.map(remote.raw, missing))
                    m = ref._moe(x, li)
                else:
                    m = ref._mlp(x, lp + ".mlp")
                h = prefix + m
                if li % 8 == 0 or li == NL - 1:
                    print(f"  [fw{step}] layer {li:2d}/{NL} rms {h.float().pow(2).mean().sqrt():.4f}",
                          flush=True)
                for k in [k for k in W if f".layers.{li}." in k]:
                    dict.__delitem__(W, k)
            h = ref._apply_attn_res(h, bres,
                                    W["model.output_attn_res_norm.weight"],
                                    W["model.output_attn_res_proj.weight"])
            h = _rmsnorm(h, W["model.norm.weight"], ref.eps)
            if lm_head is None:
                lm_head = W["lm_head.weight"]
            logits = h[-1] @ lm_head.T
        nxt = int(logits.argmax())
        piece = tok.decode([nxt])
        print(f"[run] token {step+1}: id={nxt} {piece!r} "
              f"(fetched so far {remote.fetched_bytes/1e9:.2f} GB)", flush=True)
        ids.append(nxt)
        if nxt in stops:
            break
    print(f"[run] OUTPUT: {tok.decode(ids)!r}")
    print(f"[run] total remote bytes: {remote.fetched_bytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()
