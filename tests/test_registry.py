import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aigc.registry import load_registry


ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):
    def test_registry_loads_all_target_models(self) -> None:
        registry = load_registry()
        names = [model.name for model in registry.list_models()]
        self.assertEqual(len(names), 12)
        self.assertIn("Z-Image", names)
        self.assertIn("Wan2.2-T2V-A14B-Diffusers", names)
        self.assertIn("CogVideoX-5B", names)

    def test_registry_resolves_adapter_class(self) -> None:
        registry = load_registry()
        adapter_class = registry.resolve_adapter_class("Z-Image")
        self.assertEqual(adapter_class.__name__, "ZImageAdapter")

    def test_registry_merges_local_model_overrides(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        local_models_path = tmpdir / "local_models.yaml"
        local_models_path.write_text(
            json.dumps(
                {
                    "Z-Image": {
                        "conda_env_name": "zimage_cluster",
                        "local_path": "/models/Z-Image",
                        "hf_cache_dir": "/cache/hf",
                        "max_gpus": 2,
                    },
                    "LongCat-Video": {
                        "repo_path": "/repos/LongCat-Video",
                        "weights_path": "/models/LongCat-Video",
                    },
                }
            ),
            encoding="utf-8",
        )
        registry = load_registry(local_models_path=local_models_path)

        zimage = registry.get_model("Z-Image")
        longcat = registry.get_model("LongCat-Video")

        self.assertEqual(zimage.weights["local_path"], "/models/Z-Image")
        self.assertEqual(zimage.weights["hf_cache_dir"], "/cache/hf")
        self.assertEqual(zimage.runtime["conda_env_name"], "zimage_cluster")
        self.assertEqual(zimage.conda_env_name, "zimage_cluster")
        self.assertEqual(zimage.runtime["max_gpus"], 2)
        self.assertEqual(zimage.max_gpus, 2)
        self.assertEqual(zimage.local_paths["conda_env_name"], "zimage_cluster")
        self.assertEqual(zimage.local_paths["local_path"], "/models/Z-Image")
        self.assertEqual(zimage.local_paths["max_gpus"], 2)
        self.assertEqual(zimage.local_override_source, str(local_models_path))
        self.assertEqual(longcat.weights["repo_path"], "/repos/LongCat-Video")
        self.assertEqual(longcat.weights["weights_path"], "/models/LongCat-Video")

    def test_models_list_json_output(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "aigc", "models", "list", "--output", "json"],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertEqual(len(payload), 12)

    def test_models_inspect_text_output(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "aigc", "models", "inspect", "Z-Image"],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Model: Z-Image", result.stdout)
        self.assertIn("Adapter Class: ZImageAdapter", result.stdout)
        self.assertIn("Conda Env: zimage", result.stdout)
