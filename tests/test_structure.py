"""Regression checks for installation boundaries and runtime configuration."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from pricer.config import Settings, runtime_home


class ConfigurationTests(unittest.TestCase):
    def test_checkout_default_does_not_depend_on_working_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_home(), Path(__file__).resolve().parents[1])

    def test_relative_and_absolute_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            external = home / "external.sqlite3"
            with patch.dict(os.environ, {
                "PRICER_HOME": str(home),
                "PRICER_MEMORY_PATH": "state/memory.json",
                "PRICER_INTELLIGENCE_PATH": str(external),
                "PRICER_BM25_PATH": "indexes/products.sqlite3",
            }, clear=True):
                settings = Settings.from_env()
                self.assertEqual(settings.memory_path, home / "state/memory.json")
                self.assertEqual(settings.intelligence_path, external)
                self.assertEqual(settings.bm25_path, home / "indexes/products.sqlite3")
                self.assertEqual(settings.checkpoint_path, home / "artifacts/deep_neural_network.pth")

    def test_wheel_layout_defaults_to_user_data_not_site_packages(self):
        home = Path.home()
        with patch.dict(os.environ, {}, clear=True), patch("pathlib.Path.home", return_value=home), patch(
            "pricer.config.__file__", str(home / "env/Lib/site-packages/pricer/config.py")
        ):
            self.assertEqual(runtime_home(), home / ".pricer")


class PackageStructureTests(unittest.TestCase):
    def test_package_has_no_compatibility_modules(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("api.py", "deal_agent_framework.py", "price_is_right.py",
                     "src/api.py", "src/pricer/batch.py", "src/pricer/items.py",
                     "src/pricer/agents/items.py", "src/pricer/agents/evaluator.py",
                     "src/pricer/evaluation/legacy.py"):
            self.assertFalse((root / name).exists(), name)
        for path in (root / "src").rglob("*.py"):
            self.assertNotIn("sys.modules[__name__]", path.read_text(encoding="utf-8"), str(path))

    def test_product_model_preserves_numeric_and_text_identifiers(self):
        from pricer.data.items import Item
        for identifier in (42, "B001", None):
            item = Item(title="Product", category="Electronics", price=10, id=identifier)
            self.assertEqual(item.id, identifier)
            if identifier is not None:
                self.assertIs(type(item.id), type(identifier))
            item.make_prompt("A product")
            self.assertTrue(item.test_prompt().endswith("Price is $"))

    def test_canonical_code_does_not_import_legacy_entry_points(self):
        import ast
        source = Path(__file__).resolve().parents[1] / "src/pricer"
        legacy = {"agents", "deal_agent_framework", "llama", "pricer_ephemeral", "log_utils", "util"}
        for path in source.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], legacy, str(path))
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        self.assertNotIn(name.name.split(".")[0], legacy, str(path))

    def test_src_package_cli_works_outside_checkout_without_loading_ml(self):
        source = Path(__file__).resolve().parents[1] / "src"
        with tempfile.TemporaryDirectory() as directory:
            environment = {**os.environ, "PYTHONPATH": str(source)}
            result = subprocess.run(
                [sys.executable, "-B", "-m", "pricer", "--help"],
                cwd=directory, env=environment, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("build-index", result.stdout)
            probe = subprocess.run(
                [sys.executable, "-B", "-c",
                 "import pricer, sys; print('torch' in sys.modules)"],
                cwd=directory, env=environment, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertEqual(probe.stdout.strip(), "False")


class ApplicationFactoryTests(unittest.TestCase):
    def test_routes_use_injected_service_and_factory_is_independent(self):
        from fastapi.testclient import TestClient
        from pricer.interfaces.api import create_app
        from pricer.interfaces.dependencies import framework

        application = create_app()
        service = Mock()
        service.store.list_opportunities.return_value = []
        application.dependency_overrides[framework] = lambda: service
        with patch.dict(os.environ, {"PRICER_API_KEY": "test-key"}):
            with TestClient(application) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/opportunities").status_code, 401)
                response = client.get("/opportunities", headers={"X-API-Key": "test-key"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), [])
        service.store.list_opportunities.assert_called_once_with(None)
        self.assertEqual(create_app().dependency_overrides, {})

    def test_health_import_does_not_load_application_or_models(self):
        source = Path(__file__).resolve().parents[1] / "src"
        code = (
            "import sys; from fastapi.testclient import TestClient; "
            "from pricer.interfaces.api import create_app; "
            "assert TestClient(create_app()).get('/health').status_code == 200; "
            "assert 'pricer.application.framework' not in sys.modules; "
            "assert 'torch' not in sys.modules"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", "-c", code], cwd=directory,
                env={**os.environ, "PYTHONPATH": str(source)},
                text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


class EnvironmentTests(unittest.TestCase):
    def test_direct_asgi_auth_loads_dotenv_before_service_initialization(self):
        from fastapi import HTTPException
        from pricer.interfaces.dependencies import require_api_key
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / ".env").write_text("PRICER_API_KEY=dotenv-key\n", encoding="utf-8")
            with patch.dict(os.environ, {"PRICER_HOME": directory}, clear=True):
                with self.assertRaises(HTTPException) as error:
                    require_api_key(None)
                self.assertEqual(error.exception.status_code, 401)
                require_api_key("dotenv-key")

    def test_dotenv_never_overwrites_deployment_environment(self):
        from pricer.config import load_environment
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / ".env").write_text("PRICER_API_KEY=dotenv-key\n", encoding="utf-8")
            with patch.dict(os.environ, {
                "PRICER_HOME": directory, "PRICER_API_KEY": "deployment-key"
            }, clear=True):
                load_environment()
                self.assertEqual(os.environ["PRICER_API_KEY"], "deployment-key")
