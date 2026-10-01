"""Shared runtime paths. PRICER_HOME overrides the checkout or ~/.pricer default."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


def runtime_home() -> Path:
    configured = os.getenv("PRICER_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "src" / "pricer").is_dir():
        return checkout
    return Path.home() / ".pricer"


def _path(name: str, default: Path, home: Path) -> Path:
    configured = os.getenv(name, "").strip()
    path = Path(configured).expanduser() if configured else default
    return path if path.is_absolute() else home / path


@dataclass(frozen=True)
class Settings:
    home: Path
    vectorstore_path: Path
    memory_path: Path
    intelligence_path: Path
    bm25_path: Path
    checkpoint_path: Path
    alert_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        home = runtime_home()
        return cls(
            home=home,
            vectorstore_path=_path("PRICER_VECTORSTORE_PATH", home / "products_vectorstore", home),
            memory_path=_path("PRICER_MEMORY_PATH", home / "memory.json", home),
            intelligence_path=_path("PRICER_INTELLIGENCE_PATH", home / "artifacts/pricer_intelligence.sqlite3", home),
            bm25_path=_path("PRICER_BM25_PATH", home / "products_bm25.sqlite3", home),
            checkpoint_path=_path("PRICER_CHECKPOINT_PATH", home / "artifacts/deep_neural_network.pth", home),
            alert_dir=_path("PRICER_ALERT_DIR", home / "artifacts", home),
        )


def load_environment() -> None:
    """Load the selected runtime dotenv file without replacing deployment values."""
    from dotenv import load_dotenv
    load_dotenv(runtime_home() / ".env", override=False)
