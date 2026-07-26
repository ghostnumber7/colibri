"""Regression test for the `--outdir`-derives-`ref_k2.json` bug (final whole-branch review,
fix wave item 1): `make_k2tiny.py` used to compute `ref_path = outdir.parent / "ref_k2.json"`,
so ANY `--outdir` sharing a parent with the default (e.g. `c/k2_tiny002`) still resolved to
`c/ref_k2.json` and silently clobbered the committed golden reference the engine's `REF=`/
`TF=` gates depend on. A corrupted golden makes every downstream gate vacuous while it still
reports a pass -- nothing catches it. `resolve_ref_path` is the fixed derivation; these tests
pin its cases (default outdir, several non-default outdirs, explicit --ref override) directly,
without paying for a full model build.

unittest.TestCase (not bare pytest functions) so this file is actually collected by
`make -C c test-python`'s `python -m unittest discover -s tests -p 'test_*.py'` -- the same
class of "looks covered but silently isn't" failure mode this bug itself was.
"""
import importlib.util
import os
import unittest
from pathlib import Path

TOOLS = os.path.join(os.path.dirname(__file__), "..", "tools")
spec = importlib.util.spec_from_file_location(
    "make_k2tiny", os.path.join(TOOLS, "make_k2tiny.py"))
make_k2tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_k2tiny)


class ResolveRefPathTest(unittest.TestCase):
    def test_default_outdir_resolves_to_committed_golden(self):
        """The one invocation shape used by CI/gates (`--outdir c/k2_tiny`, i.e. no
        --outdir override at all) must keep writing exactly c/ref_k2.json -- this must not
        regress for callers that already depend on the default path."""
        ref = make_k2tiny.resolve_ref_path("c/k2_tiny", None, "c/k2_tiny")
        self.assertEqual(ref, Path("c/ref_k2.json"))

    def test_nondefault_outdir_never_touches_the_committed_golden(self):
        """The actual bug: a non-default --outdir must NOT resolve to c/ref_k2.json."""
        ref = make_k2tiny.resolve_ref_path("c/k2_tiny002", None, "c/k2_tiny")
        self.assertNotEqual(ref, Path("c/ref_k2.json"))
        self.assertEqual(ref, Path("c/ref_k2_tiny002.json"))

    def test_various_nondefault_outdirs_each_derive_a_distinct_ref(self):
        for outdir_arg in ("c/k2_tiny_variant", "c/other", "c/k2_tiny/nested"):
            with self.subTest(outdir_arg=outdir_arg):
                ref = make_k2tiny.resolve_ref_path(outdir_arg, None, "c/k2_tiny")
                self.assertNotEqual(ref, Path("c/ref_k2.json"))
                self.assertEqual(ref, Path(outdir_arg).parent / f"ref_{Path(outdir_arg).name}.json")

    def test_explicit_ref_overrides_derivation_for_nondefault_outdir(self):
        ref = make_k2tiny.resolve_ref_path("c/k2_tiny002", "/tmp/custom_ref.json", "c/k2_tiny")
        self.assertEqual(ref, Path("/tmp/custom_ref.json"))

    def test_explicit_ref_overrides_even_for_default_outdir(self):
        ref = make_k2tiny.resolve_ref_path("c/k2_tiny", "c/somewhere_else.json", "c/k2_tiny")
        self.assertEqual(ref, Path("c/somewhere_else.json"))


if __name__ == "__main__":
    unittest.main()
