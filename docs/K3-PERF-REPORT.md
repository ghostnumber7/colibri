# Kimi-K3 first decode measurements (reference box: 9950X3D · 192 GB · RTX 5090)

First tokens/second numbers for the Kimi-K3 port (1.45 TB container: mxfp4
experts + int8 residents; 93 layers, 896 experts × top-16 + 2 shared), and
where the time actually goes. Method: identical 30-token prompt, `--ngen 16`,
greedy, `KVSAVE=0`; single runs, so treat deltas under ~15% as noise. 16-token
decode never warms the LRU fully — sustained tps on long generations sits above
these numbers.

## Decode progression

| platform | config | decode tok/s | s/token | expert hit |
|---|---|---|---|---|
| WSL2 (drvfs model dir) | engine defaults | 0.013 | 78 | — |
| WSL2 | `PIPE=1 PIPE_WORKERS=24 RAM_GB=70` | 0.044 | 22.9 | 43.3% |
| Windows native | `RAM_GB=140 PIPE_WORKERS=24` + `DIRECT/PIPE/PILOT_REAL` defaults | 0.11 | 9.1 | 60.4% |
| Windows native | + `num_experts_per_token=8` (config edit) | 0.18 | 5.5 | 67.2% |

Prefill (30 tokens): 437.9 s → 334.9 s (PIPE) → 210.2 s (native) → 105.1 s (top-8).

## Where the time goes

Expert streaming dominates everything. A cold top-16 token reads
16 experts × 92 MoE layers × 17.55 MB ≈ **25.8 GB from disk**.

- WSL2's drvfs bridge caps a single reader at ~230 MB/s and saturates near
  ~840 MB/s with 16+ parallel streams — that is the whole story of the first
  two rows. `PIPE`/`DIRECT`/`PILOT_REAL` are win32-only defaults in `coli`;
  Linux runs must set them explicitly.
- Native NVMe removes the bridge cap: prefill expert-disk wait drops
  322.9 s → 15.8 s. Decode on the native row splits ~49% disk-miss wait /
  ~35% expert-matmul / ~10% KDA+attention.
- `matmul_mxfp4` (fmt=7 kernel) is not a bottleneck: 13.3 GB/s weight-read at
  S=1 on 4 cores in a standalone micro-bench.
- Cache size is exhausted for short runs (`RAM_GB` 140 → 155 changed nothing):
  16-token decode is first-touch misses. The usage histogram says 80% of hits
  need ~142 GB of experts resident.
- Top-8 routing is the cheapest big lever (1.64×) and produced the same greedy
  opening as top-16 on the test prompt — but it changes the model; it needs a
  real quality eval before being more than a demo knob.
- GPU: there is no fmt=7 CUDA kernel, so routed experts cannot offload;
  `CUDA_DENSE=1` would take every resident (selection is format-generic) but
  does a synchronous PCIe round-trip per matvec — expected net-negative at
  S=1. `CUDA_EXPERT_GB` must stay unset on K3: uploads fail per-format and the
  placement logic still inflates the host pin budget.

## Ceiling

1 tok/s is out of reach on a single 192 GB box, independent of engine quality:
the container is 7.5× host RAM, so every token streams experts from disk —
~8–10 GB/token at the measured hit rates, a ≥3 s/token floor at ~3 GB/s NVMe.
Getting there requires the expert working set in RAM (several hundred GB) or a
striped multi-drive array, not software. Software headroom that does exist:
router-pilot prefetch (fixed on this branch — the K3 residual-mixer path never
called `pilot_prefetch`, so `PILOT_REAL` was silently dead on kimi_linear;
mechanism gave GLM +11%), speculative k=2 batching to amortize expert reads,
and the resident int8 → int4-g64 requant (tools/requant_residents.py, unmeasured).
