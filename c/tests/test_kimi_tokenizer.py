import os, json, importlib.util, tempfile, base64
import pytest

K2_DIR = os.environ.get("COLI_K2_SRC", "")
TOOLS = os.path.join(os.path.dirname(__file__), "..", "tools")

_NEEDS_K2_DATA = pytest.mark.skipif(
    not (K2_DIR and os.path.exists(os.path.join(K2_DIR, "tiktoken.model"))),
    reason="needs a local K2 checkpoint with tiktoken.model (set COLI_K2_SRC)")

spec = importlib.util.spec_from_file_location("genk", os.path.join(TOOLS, "gen_kimi_tokenizer.py"))
genk = importlib.util.module_from_spec(spec); spec.loader.exec_module(genk)

# K2 pretokenizer pattern (from tokenization_kimi.py), o200k + Han handling.
K2_PAT = (r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
          r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
          r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
          r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
          r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

NON_CJK_CASES = [
    "Hello, world!", "def foo(x): return x*2\n", "The quick brown fox.",
    "  leading spaces and\ttabs", "numbers 12345 and 6789", "CamelCaseIdentifier",
    "it's a test, don't you think?", "line1\nline2\nline3",
]


def _reference_ids(texts):
    """Ground truth: K2's own tiktoken Encoding built from the same ranks + pattern."""
    import tiktoken, base64
    ranks = {}
    for line in open(os.path.join(K2_DIR, "tiktoken.model")):
        tok, rank = line.split()
        ranks[base64.b64decode(tok)] = int(rank)
    enc = tiktoken.Encoding(name="kimi", pat_str=K2_PAT, mergeable_ranks=ranks, special_tokens={})
    return [enc.encode(t) for t in texts]


@_NEEDS_K2_DATA
def test_generated_tokenizer_matches_reference_non_cjk():
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("tiktoken")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "tokenizer.json")
        genk.gen_kimi_tokenizer(K2_DIR, out)
        tk = tokenizers.Tokenizer.from_file(out)
        ref = _reference_ids(NON_CJK_CASES)
        for text, exp in zip(NON_CJK_CASES, ref):
            got = tk.encode(text, add_special_tokens=False).ids
            assert got == exp, f"mismatch for {text!r}: {got} != {exp}"


def test_special_tokens_are_included(tmp_path):
    """Self-contained unit test (no checkpoint dependency, no full vocab needed):
    asserts that entries in tokenizer_config.json's added_tokens_decoder make
    it into the generated tokenizer.json's added_tokens AND model.vocab, at
    the exact ids the config specifies. This is what guards K2-Thinking's
    reasoning-only special tokens like <think>/</think> (ids 163606/163607)
    from silently being dropped or mis-numbered — the round-trip test above
    only checks the base BPE vocab/merges and would not catch this."""
    base_tokens = [b"a", b"b", b"c", b"d"]
    model_dir = tmp_path
    with open(model_dir / "tiktoken.model", "w") as fh:
        for rank, tok in enumerate(base_tokens):
            fh.write(f"{base64.b64encode(tok).decode()} {rank}\n")

    think_id = len(base_tokens)
    think_close_id = len(base_tokens) + 1
    cfg = {
        "added_tokens_decoder": {
            str(think_id): {"content": "<think>", "special": True},
            str(think_close_id): {"content": "</think>", "special": True},
        }
    }
    with open(model_dir / "tokenizer_config.json", "w") as fh:
        json.dump(cfg, fh)

    out = tmp_path / "tokenizer.json"
    genk.gen_kimi_tokenizer(str(model_dir), str(out))
    generated = json.loads(out.read_text())

    added_by_content = {t["content"]: t for t in generated["added_tokens"]}
    assert "<think>" in added_by_content, "generated tokenizer.json is missing <think> in added_tokens"
    assert "</think>" in added_by_content, "generated tokenizer.json is missing </think> in added_tokens"
    assert added_by_content["<think>"]["id"] == think_id
    assert added_by_content["</think>"]["id"] == think_close_id
    assert added_by_content["<think>"]["special"] is True
    assert added_by_content["</think>"]["special"] is True

    vocab = generated["model"]["vocab"]
    assert vocab.get("<think>") == think_id, "<think> missing from model.vocab at the expected id"
    assert vocab.get("</think>") == think_close_id, "</think> missing from model.vocab at the expected id"


def test_merges_are_two_element_string_arrays(tmp_path):
    """Self-contained structural test (no checkpoint dependency): colibri's C
    tokenizer (c/tok.h lines 151-153) requires each `model.merges` entry to be
    a JSON array of exactly two strings ["a","b"] -- it exit(1)s with
    "malformed merge entry" on anything else, including the joined "a b"
    string form that the Python `tokenizers` library happens to also accept.
    This guards against regressing to that string form, which the round-trip
    test alone would never catch (tokenizers silently tolerates both)."""
    base_tokens = [b"a", b"b", b"ab"]  # "ab" forces a real reconstructed merge
    model_dir = tmp_path
    with open(model_dir / "tiktoken.model", "w") as fh:
        for rank, tok in enumerate(base_tokens):
            fh.write(f"{base64.b64encode(tok).decode()} {rank}\n")
    with open(model_dir / "tokenizer_config.json", "w") as fh:
        json.dump({"added_tokens_decoder": {}}, fh)

    out = tmp_path / "tokenizer.json"
    genk.gen_kimi_tokenizer(str(model_dir), str(out))
    generated = json.loads(out.read_text())

    merges = generated["model"]["merges"]
    assert isinstance(merges, list) and len(merges) > 0, "expected at least one reconstructed merge"
    for entry in merges:
        assert isinstance(entry, list), f"merge entry must be a list, got {type(entry).__name__}: {entry!r}"
        assert len(entry) == 2, f"merge entry must have exactly 2 elements: {entry!r}"
        assert all(isinstance(x, str) for x in entry), f"merge entry elements must be str: {entry!r}"
