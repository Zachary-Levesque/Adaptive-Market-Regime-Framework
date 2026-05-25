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
