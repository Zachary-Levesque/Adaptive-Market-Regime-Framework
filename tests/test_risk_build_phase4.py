from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from src.risk.build_phase4 import resolve_signal_path, resolve_signal_selection


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


def test_resolve_signal_selection_includes_execution_settings(tmp_path: Path):
    selected_path = tmp_path / "alpha_signals" / "ensemble.parquet"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_bytes(b"")

    selection_path = tmp_path / "alpha_signal_selection.parquet"
    pd.DataFrame(
        [
            {
                "signal_path": str(selected_path),
                "transaction_cost_bps": 10.0,
                "rebalance_interval_days": 5,
            }
        ]
    ).to_parquet(selection_path)

    config = SimpleNamespace(
        alpha=SimpleNamespace(
            selection_path=selection_path,
            signals_path=tmp_path / "fallback.parquet",
        )
    )

    selection = resolve_signal_selection(config)

    assert selection.signal_path == selected_path
    assert selection.transaction_cost_bps == 10.0
    assert selection.rebalance_interval_days == 5


def test_resolve_signal_selection_override_ignores_manifest_settings(tmp_path: Path):
    override_path = tmp_path / "override.parquet"
    selection_path = tmp_path / "alpha_signal_selection.parquet"
    pd.DataFrame(
        [
            {
                "signal_path": str(tmp_path / "selected.parquet"),
                "transaction_cost_bps": 10.0,
                "rebalance_interval_days": 5,
            }
        ]
    ).to_parquet(selection_path)

    config = SimpleNamespace(
        alpha=SimpleNamespace(
            selection_path=selection_path,
            signals_path=tmp_path / "fallback.parquet",
        )
    )

    selection = resolve_signal_selection(config, override=str(override_path))

    assert selection.signal_path == override_path
    assert selection.transaction_cost_bps is None
    assert selection.rebalance_interval_days is None


def test_resolve_signal_path_falls_back_to_config_path(tmp_path: Path):
    config = SimpleNamespace(
        alpha=SimpleNamespace(
            selection_path=tmp_path / "missing.parquet",
            signals_path=tmp_path / "fallback.parquet",
        )
    )

    assert resolve_signal_path(config) == tmp_path / "fallback.parquet"
