from pathlib import Path

from src.data.validate_phase1_inputs import main


def test_validate_phase1_inputs_runs_with_local_files(tmp_path: Path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "spy.csv").write_text(
        "\n".join(
            [
                "Date,Open,High,Low,Close,Volume",
                "2024-01-02,100,101,99,100.5,1000",
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  universe: [SPY, QQQ, ^VIX]",
                "  start_date: '2020-01-01'",
                "  end_date: '2020-12-31'",
                "  benchmark: 'SPY'",
                f"  cache_dir: '{raw_dir}'",
                "  processed_dir: 'data/processed'",
                f"  local_data_dir: '{raw_dir}'",
                "  allow_remote_downloads: false",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["validate_phase1_inputs", "--config", str(config_path)])
    main()
    output = capsys.readouterr().out

    assert "Resolved locally: 1" in output
    assert "Missing locally: 1" in output
    assert "SPY" in output
    assert "QQQ" in output
