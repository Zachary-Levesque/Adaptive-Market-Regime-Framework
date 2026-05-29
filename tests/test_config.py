from pathlib import Path

from src.config import load_config


def test_load_config_parses_data_paths(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  universe: [SPY, QQQ]",
                "  start_date: '2020-01-01'",
                "  end_date: '2020-12-31'",
                "  benchmark: 'SPY'",
                "  cache_dir: 'data/raw'",
                "  processed_dir: 'data/processed'",
                "  local_data_dir: 'data/raw/manual'",
                "  allow_remote_downloads: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.universe == ["SPY", "QQQ"]
    assert config.data.benchmark == "SPY"
    assert config.data.cache_dir == Path("data/raw")
    assert config.data.processed_dir == Path("data/processed")
    assert config.data.local_data_dir == Path("data/raw/manual")
    assert config.data.allow_remote_downloads is False
    assert config.regime.n_regimes == 4
    assert config.regime.model_path == Path("src/regime/hmm_model.pkl")
    assert config.alpha.sequence_length == 60
    assert config.alpha.model_dir == Path("src/alpha/models")
    assert config.alpha.diagnostics_path == Path("data/processed/alpha_diagnostics.parquet")
    assert config.risk.output_dir == Path("data/results")
    assert config.risk.transaction_cost_bps == 10.0
