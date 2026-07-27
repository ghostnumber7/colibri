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
    """flatten_config: the shared accessor every c/coli config.json reader
    (model_type_of, cmd_info, cmd_convert) funnels through, so a K2.6-shaped
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


class BuildRunPromptTest(unittest.TestCase):
    """coli:cmd_run builds this exact string and hands it to the engine via $PROMPT.
    kimi_k2 means Kimi-K2.6; the template is pinned byte-exact against the checkpoint's
    own chat_template.jinja rendered through transformers' Jinja2 env."""

    def test_kimi_k2_template_is_exact(self):
        """K2.6: no [gMASK]/<sop>, no system preamble, nothink marker by default."""
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("kimi_k2", "hello"),
                "<|im_user|>user<|im_middle|>hello<|im_end|>"
                "<|im_assistant|>assistant<|im_middle|><think></think>",
            )

    def test_glm_template_is_unchanged(self):
        """Regression pin: the K2 work must not alter GLM's byte-identical template."""
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("glm_moe_dsa", "hello"),
                "[gMASK]<sop><|user|>hello<|assistant|><think></think>",
            )

    def test_think_env_opens_think_marker_for_glm_and_kimi(self):
        """THINK=1 leaves <think> open; K2.6 follows GLM's THINK-env convention."""
        with mock.patch.dict("os.environ", {"THINK": "1"}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("glm_moe_dsa", "hello"),
                "[gMASK]<sop><|user|>hello<|assistant|><think>",
            )
            self.assertEqual(
                coli.build_run_prompt("kimi_k2", "hello"),
                "<|im_user|>user<|im_middle|>hello<|im_end|>"
                "<|im_assistant|>assistant<|im_middle|><think>",
            )

    def test_unknown_model_type_falls_back_to_glm(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                coli.build_run_prompt("", "hello"),
                "[gMASK]<sop><|user|>hello<|assistant|><think></think>",
            )


class CmdServeTest(unittest.TestCase):
    """cmd_serve wiring: detect_arch sets openai_server.ARCH, fail-soft on a bad
    config.json, and the pidfile try/finally covers the detect_arch call itself."""

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

    def test_serve_wires_arch_from_model_config(self):
        """cmd_serve must set openai_server.ARCH via detect_arch before serve() runs."""
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model_dir(tmp, json.dumps({"model_type": "kimi_k2"}))
            pidfile = coli.serve_pidfile(19736)
            self.addCleanup(lambda: os.path.exists(pidfile) and os.unlink(pidfile))
            original_arch = openai_server.ARCH
            self.addCleanup(lambda: setattr(openai_server, "ARCH", original_arch))
            seen = {}
            with mock.patch.object(coli, "GLM", __file__), \
                 mock.patch.object(openai_server, "serve",
                                   side_effect=lambda *a, **k: seen.setdefault("arch", openai_server.ARCH)):
                coli.cmd_serve(self._args(model, 19736))
            self.assertEqual(seen.get("arch"), "kimi_k2")

    def test_unexpected_exception_in_detect_arch_still_cleans_up_pidfile(self):
        """The try/finally wraps the whole detect+serve sequence, so even an exception
        type detect_arch doesn't catch today cannot orphan the pidfile."""
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
