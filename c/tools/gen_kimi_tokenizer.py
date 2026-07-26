"""Generate a colibri/tok.h-compatible tokenizer.json for Kimi K2.

K2 ships tiktoken.model (base64 token -> rank) + tokenizer_config.json (special
tokens), but NO tokenizer.json. colibri's tok.h loads a byte-level BPE
tokenizer.json (ignore_merges, o200k-style Split pretokenizer). This rebuilds
one: vocab from the tiktoken ranks (GPT-2 byte->unicode encoded), merges
reconstructed from ranks, special tokens appended, K2's exact pat_str emitted.
Validated by a round-trip vs the tiktoken reference (tests/test_kimi_tokenizer.py)."""
import os, sys, json, base64, argparse

# K2 pretokenizer pattern (verbatim from tokenization_kimi.py).
K2_PAT = (r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
          r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
          r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
          r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
          r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def bytes_to_unicode():
    """GPT-2 reversible byte<->unicode map (same table tok.h/HF use)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def load_ranks(tiktoken_model_path):
    ranks = {}
    with open(tiktoken_model_path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            tok_b64, rank = line.split()
            ranks[base64.b64decode(tok_b64)] = int(rank)
    return ranks


def reconstruct_merges(ranks):
    """Standard tiktoken->BPE merge recovery: for each multi-byte token (by rank),
    the merge is the split (a,b) into existing tokens with the lowest max-rank."""
    byte_to_rank = ranks
    merges = []
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) < 2: continue
        best = None
        for i in range(1, len(token)):
            a, b = token[:i], token[i:]
            ra, rb = byte_to_rank.get(a), byte_to_rank.get(b)
            if ra is not None and rb is not None:
                m = max(ra, rb)
                if best is None or m < best[0]: best = (m, a, b)
        if best: merges.append((best[1], best[2]))
    return merges


def gen_kimi_tokenizer(model_dir, out_path):
    b2u = bytes_to_unicode()
    enc = lambda bs: "".join(b2u[x] for x in bs)
    ranks = load_ranks(os.path.join(model_dir, "tiktoken.model"))
    vocab = {enc(tok): rank for tok, rank in ranks.items()}
    # tok.h (c/tok.h) requires each merge entry to be a JSON array of exactly
    # two strings ["a","b"] -- NOT a joined "a b" string. The Python
    # `tokenizers` library tolerates both, so this only surfaces at the C
    # consumer; see test_merges_are_two_element_string_arrays.
    merges = [[enc(a), enc(b)] for a, b in reconstruct_merges(ranks)]

    # special / added tokens: authoritative ids come from tokenizer_config.json
    cfg = json.load(open(os.path.join(model_dir, "tokenizer_config.json")))
    added = []
    for sid, meta in sorted((cfg.get("added_tokens_decoder") or {}).items(), key=lambda kv: int(kv[0])):
        added.append({"id": int(sid), "content": meta["content"], "single_word": False,
                      "lstrip": bool(meta.get("lstrip", False)), "rstrip": bool(meta.get("rstrip", False)),
                      "normalized": bool(meta.get("normalized", False)), "special": bool(meta.get("special", True))})
        vocab.setdefault(meta["content"], int(sid))

    tok = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": added, "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": K2_PAT}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": vocab, "merges": merges},
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(tok, fh, ensure_ascii=False)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gen_kimi_tokenizer(a.model_dir, a.out)
    print(f"wrote {a.out}")
