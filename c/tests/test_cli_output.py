import argparse
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import openai_server

HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "coli"

# `coli` has no .py extension (it's the launcher script), so it can't be `import`ed the
# normal way -- load it as a module directly from its source file, same trick
# test_env_defaults.py already uses to reach env_for() without a subprocess per assertion.
_loader = importlib.machinery.SourceFileLoader("coli_cli", str(CLI))
_spec = importlib.util.spec_from_loader("coli_cli", _loader)
coli = importlib.util.module_from_spec(_spec)
_loader.exec_module(coli)


class CliOutputLanguageTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def test_help_is_english(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run GLM-5.2 locally", result.stdout)
        self.assertIn("automatically apply the RAM/VRAM plan", result.stdout)
        self.assertNotIn("modello", result.stdout.lower())
        self.assertNotIn("motore", result.stdout.lower())

    def test_info_status_is_english(self):
        with tempfile.TemporaryDirectory() as model:
            result = self.run_cli("info", "--model", model)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("config.json is missing", result.stdout)
        self.assertIn("disk", result.stdout)
        self.assertIn("engine", result.stdout)

    def test_missing_model_error_is_english(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_model = str(Path(directory) / "missing-model")
            result = self.run_cli("run", "--model", missing_model, "hello")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model not found", result.stderr)
        self.assertIn("set COLI_MODEL or use --model", result.stderr)


class ModelTypeOfTest(unittest.TestCase):
    """model_type_of: the one config.json read that selects the chat template. Both tiny
    fixture containers ship no tokenizer.json by design, and `need_model` hard-exits
    without one -- so `coli run` end-to-end can't be exercised here. These tests pin the
    piece that actually changed: the launcher-side string construction, with no engine or
    tokenizer involved."""

    def test_reads_kimi_k2(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({"model_type": "kimi_k2"}))
            self.assertEqual(coli.model_type_of(model), "kimi_k2")

    def test_reads_glm(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
            self.assertEqual(coli.model_type_of(model), "glm_moe_dsa")

    def test_missing_config_returns_empty_string(self):
        with tempfile.TemporaryDirectory() as model:
            self.assertEqual(coli.model_type_of(model), "")

    def test_malformed_config_returns_empty_string(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text("not json")
            self.assertEqual(coli.model_type_of(model), "")

    def test_reads_kimi_k2_through_nested_text_config(self):
        """K2.6's unconverted source repo nests model_type inside text_config (the
        top-level model_type is "kimi_k25"). model_type_of must still report "kimi_k2",
        not "" or "kimi_k25" -- via the shared flatten_config accessor."""
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "kimi_k25",
                "text_config": {"model_type": "kimi_k2"},
                "vision_config": {},
            }))
            self.assertEqual(coli.model_type_of(model), "kimi_k2")


class FlattenConfigTest(unittest.TestCase):
    """flatten_config: the single shared accessor every c/coli config.json reader
    (model_type_of, cmd_info, _convert_features) funnels through, so a K2.6-shaped
    nested config and an already-flat config report identical values."""

    def test_nested_returns_text_config_object(self):
        text_config = {"model_type": "kimi_k2", "num_hidden_layers": 61}
        cfg = {"model_type": "kimi_k25", "text_config": text_config, "vision_config": {}}
        self.assertIs(coli.flatten_config(cfg), text_config)

    def test_flat_passthrough_unchanged(self):
        cfg = {"model_type": "kimi_k2", "num_hidden_layers": 61}
        self.assertIs(coli.flatten_config(cfg), cfg)

    def test_text_config_present_but_not_a_dict_is_ignored(self):
        cfg = {"model_type": "kimi_k2", "text_config": "not-a-dict"}
        self.assertIs(coli.flatten_config(cfg), cfg)

    def test_non_dict_input_passed_through(self):
        self.assertEqual(coli.flatten_config(None), None)


class IsK26OfTest(unittest.TestCase):
    """is_k26_of: the K2.6-vs-K2-Thinking discriminator, since both report the identical
    model_type "kimi_k2" once converted. Verified against both REAL checkpoints'
    config.json (K2-Thinking: beta_fast=1.0; K2.6: beta_fast=32.0), not assumed."""

    def test_k26_shaped_config_beta_fast_32(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "kimi_k2",
                "rope_scaling": {"type": "yarn", "beta_fast": 32.0, "beta_slow": 1.0},
            }))
            self.assertTrue(coli.is_k26_of(model))

    def test_k2_thinking_shaped_config_beta_fast_1_is_not_k26(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "kimi_k2",
                "rope_scaling": {"type": "yarn", "beta_fast": 1.0, "beta_slow": 1.0},
            }))
            self.assertFalse(coli.is_k26_of(model))

    def test_no_rope_scaling_at_all_is_not_k26(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
            self.assertFalse(coli.is_k26_of(model))

    def test_reads_through_nested_text_config(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "kimi_k25",
                "text_config": {"model_type": "kimi_k2",
                               "rope_scaling": {"type": "yarn", "beta_fast": 32.0}},
            }))
            self.assertTrue(coli.is_k26_of(model))

    def test_missing_config_fails_soft_to_false(self):
        with tempfile.TemporaryDirectory() as model:
            self.assertFalse(coli.is_k26_of(model))

    def test_malformed_config_fails_soft_to_false(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text("not json")
            self.assertFalse(coli.is_k26_of(model))

    def test_marker_wins_over_beta_fast_1_fallback(self):
        """rope_scaling.beta_fast is a YaRN tuning value with no semantic tie
        to template choice. The converter now stamps an explicit top-level
        `_colibri_source_variant: "kimi_k25"` marker; it must decide the answer even when
        beta_fast alone would say the opposite (K2-Thinking's beta_fast=1.0 here)."""
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "kimi_k2", "_colibri_source_variant": "kimi_k25",
                "rope_scaling": {"type": "yarn", "beta_fast": 1.0, "beta_slow": 1.0},
            }))
            self.assertTrue(coli.is_k26_of(model))

    def test_marker_present_but_wrong_value_is_decisive_not_k26(self):
        """A marker present but not "kimi_k25" is decisive as "not K2.6" -- only an
        ABSENT marker falls through to the beta_fast fallback, not merely a mismatch."""
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "kimi_k2", "_colibri_source_variant": "something_else",
                "rope_scaling": {"type": "yarn", "beta_fast": 32.0, "beta_slow": 1.0},
            }))
            self.assertFalse(coli.is_k26_of(model))

    def test_no_marker_falls_back_to_beta_fast(self):
        """Regression guard: containers converted before the marker existed have no such
        key and must keep working via the beta_fast fallback, unchanged."""
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "kimi_k2",
                "rope_scaling": {"type": "yarn", "beta_fast": 32.0, "beta_slow": 1.0},
            }))
            self.assertTrue(coli.is_k26_of(model))


class ConvertFeaturesTest(unittest.TestCase):
    """_convert_features: drives --ebits/1-vs-2-pass for `coli convert`. K2.6's source
    config nests both num_nextn_predict_layers and quantization_config inside
    text_config -- read flat (pre-Task-2 behavior), pack_quantized silently comes out
    False, forcing --ebits 4 onto resident tensors that must stay int8 (the same
    regression this branch already root-caused and fixed once for K2-Thinking)."""

    def _features(self, indir):
        import argparse
        return coli._convert_features(argparse.Namespace(indir=str(indir), repo=None))

    def test_k26_nested_config_reports_pack_quantized_true(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "kimi_k25",
                "text_config": {
                    "model_type": "kimi_k2",
                    "num_nextn_predict_layers": 0,
                    "quantization_config": {"format": "pack-quantized"},
                },
            }))
            has_mtp, pack_quantized = self._features(model)
            self.assertFalse(has_mtp)
            self.assertTrue(pack_quantized)

    def test_k2_thinking_flat_config_unchanged(self):
        """Regression guard: the flat (already-working) K2-Thinking shape must keep
        reporting exactly what it did before flatten_config existed."""
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "kimi_k2",
                "num_nextn_predict_layers": 0,
                "quantization_config": {"format": "pack-quantized"},
            }))
            has_mtp, pack_quantized = self._features(model)
            self.assertFalse(has_mtp)
            self.assertTrue(pack_quantized)

    def test_glm_flat_config_unchanged(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "glm_moe_dsa", "num_nextn_predict_layers": 1,
            }))
            has_mtp, pack_quantized = self._features(model)
            self.assertTrue(has_mtp)
            self.assertFalse(pack_quantized)

    def test_missing_config_fails_soft_to_glm_safe_default(self):
        with tempfile.TemporaryDirectory() as model:
            has_mtp, pack_quantized = self._features(model)
            self.assertTrue(has_mtp)
            self.assertFalse(pack_quantized)


class BuildRunPromptTest(unittest.TestCase):
    """coli:cmd_run builds this exact string and hands it to the engine via $PROMPT.
    K2 has neither [gMASK] nor <sop>: the engine already no-ops an unresolvable prefix
    (tok_id_of returns -1), but the launcher used to inject the literal GLM characters as
    text, which BPE-encodes into token sequences the model never saw."""

    def test_kimi_k2_template_is_exact(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("kimi_k2", "hello"),
                "<|im_system|>system<|im_middle|>"
                "You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"
                "<|im_user|>user<|im_middle|>hello<|im_end|>"
                "<|im_assistant|>assistant<|im_middle|>",
            )

    def test_glm_template_is_unchanged(self):
        """Regression pin: the K2 work must not alter GLM's byte-identical template."""
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("glm_moe_dsa", "hello"),
                "[gMASK]<sop><|user|>hello<|assistant|><think></think>",
            )

    def test_think_env_affects_only_glm(self):
        """K2's generation prompt has NO <think> marker at all -- K2-Thinking opens one (or
        not) on its own, unlike GLM's <think></think>-means-nothink convention. THINK=1 must
        not add anything to the K2 string."""
        with mock.patch.dict("os.environ", {"THINK": "1"}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("glm_moe_dsa", "hello"),
                "[gMASK]<sop><|user|>hello<|assistant|><think>",
            )
            self.assertEqual(
                coli.build_run_prompt("kimi_k2", "hello"),
                "<|im_system|>system<|im_middle|>"
                "You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"
                "<|im_user|>user<|im_middle|>hello<|im_end|>"
                "<|im_assistant|>assistant<|im_middle|>",
            )

    def test_unknown_model_type_falls_back_to_glm(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("", "hello"),
                "[gMASK]<sop><|user|>hello<|assistant|><think></think>",
            )

    def test_k26_no_default_preamble_nothink(self):
        """Byte-exact match against Kimi-K2.6's REAL chat_template.jinja, rendered through
        transformers' Jinja2 env. Unlike K2-Thinking, is_k26=True gets NO default system
        preamble at all, and ALWAYS a think marker in the generation prompt."""
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("kimi_k2", "hello", is_k26=True),
                "<|im_user|>user<|im_middle|>hello<|im_end|>"
                "<|im_assistant|>assistant<|im_middle|><think></think>",
            )

    def test_k26_think_env_opens_think_marker(self):
        """K2.6 DOES use THINK, exactly GLM's convention -- unlike K2-Thinking, which
        ignores it entirely (see test_think_env_affects_only_glm)."""
        with mock.patch.dict("os.environ", {"THINK": "1"}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("kimi_k2", "hello", is_k26=True),
                "<|im_user|>user<|im_middle|>hello<|im_end|>"
                "<|im_assistant|>assistant<|im_middle|><think>",
            )

    def test_k26_default_is_k26_false_is_k2_thinking_regression(self):
        """is_k26 defaults to False so every OLD call/test that only passes (mt, prompt)
        keeps K2-Thinking's exact byte-for-byte behavior -- this is the regression guard
        for that default."""
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("kimi_k2", "hello"),
                coli.build_run_prompt("kimi_k2", "hello", is_k26=False),
            )


class CmdServeTest(unittest.TestCase):
    """cmd_serve had zero coverage before these tests -- which is exactly how a
    regression slipped through: c/coli's cmd_serve calls openai_server.detect_arch(a.model)
    unguarded, and detect_arch used to catch only OSError. A syntactically-invalid
    config.json (with a valid tokenizer.json, so need_model passes) raised an uncaught
    json.JSONDecodeError, which (a) crashed `coli serve` where it previously started
    (wrongly, with GLM's template), and (b) landed after the pidfile write but before the
    try/finally that removes it, orphaning a stale pidfile. Fixed two ways: detect_arch now
    catches json.JSONDecodeError too (matching detect_k26, which already did), AND
    cmd_serve's try/finally was widened to cover the detect_arch/detect_k26 calls
    themselves, not just serve() -- so the pidfile can't be orphaned by any exception
    there, not only the one kind reproduced here."""

    def _args(self, model, port):
        return argparse.Namespace(
            model=model, port=port, host="127.0.0.1", model_id="glm-5.2-colibri",
            api_key=None, cap=8, ngen=1024, cors_origin=None, max_queue=8,
            queue_timeout=300.0, kv_slots=1, policy="quality", ram=0, topp=0,
            topk=0, temp=None, repin=0, ctx=0, auto_tier=False, gpu=None, vram=0,
        )

    def _model_dir(self, tmp, config_text):
        model = Path(tmp)
        (model / "tokenizer.json").write_text("{}")
        (model / "config.json").write_text(config_text)
        return str(model)

    def test_malformed_config_does_not_crash_and_cleans_up_pidfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model_dir(tmp, "not json")
            pidfile = coli.serve_pidfile(19734)
            self.addCleanup(lambda: os.path.exists(pidfile) and os.unlink(pidfile))
            served = {}
            with mock.patch.object(coli, "GLM", __file__), \
                 mock.patch.object(openai_server, "serve",
                                   side_effect=lambda *a, **k: served.setdefault("called", True)):
                coli.cmd_serve(self._args(model, 19734))
            self.assertTrue(served.get("called"),
                             "serve() should still run (fail-soft glm arch), not crash")
            self.assertFalse(os.path.exists(pidfile), "pidfile must not be orphaned")

    def test_unexpected_exception_in_detect_arch_still_cleans_up_pidfile(self):
        """Defense in depth, independent of which exception type triggered the review:
        the try/finally now wraps the whole detect+serve sequence, so even an exception
        type neither detect_arch nor detect_k26 catches today cannot orphan the pidfile."""
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model_dir(tmp, json.dumps({"model_type": "glm_moe_dsa"}))
            pidfile = coli.serve_pidfile(19735)
            self.addCleanup(lambda: os.path.exists(pidfile) and os.unlink(pidfile))
            with mock.patch.object(coli, "GLM", __file__), \
                 mock.patch.object(openai_server, "detect_arch", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    coli.cmd_serve(self._args(model, 19735))
            self.assertFalse(os.path.exists(pidfile),
                              "pidfile must not be orphaned even on an unexpected exception")


if __name__ == "__main__":
    unittest.main()
