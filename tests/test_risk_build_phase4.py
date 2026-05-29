from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from src.risk.build_phase4 import resolve_signal_path


def test_resolve_signal_path_prefers_selection_manifest(tmp_path: Path):
    selected_path = tmp_path / "alpha_signals" / "ridge.parquet"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_bytes(b"")

    selection_path = tmp_path / "alpha_signal_selection.parquet"
    pd.DataFrame([{"signal_path": str(selected_path)}]).to_parquet(selection_path)

    config = SimpleNamespace(
        alpha=SimpleNamespace(
            selection_path=selection_path,
            signals_path=tmp_path / "fallback.parquet",
        )
    )

    assert resolve_signal_path(config) == selected_path


def test_resolve_signal_path_falls_back_to_config_path(tmp_path: Path):
    config = SimpleNamespace(
        alpha=SimpleNamespace(
            selection_path=tmp_path / "missing.parquet",
            signals_path=tmp_path / "fallback.parquet",
        )
    )

    assert resolve_signal_path(config) == tmp_path / "fallback.parquet"
