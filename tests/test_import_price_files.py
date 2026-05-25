from pathlib import Path
import zipfile

from src.data.import_price_files import PriceFileImporter


def test_import_from_directory_copies_matching_files(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "spy.us.txt").write_text(
        "\n".join(
            [
                "Date,Open,High,Low,Close,Volume",
                "2024-01-02,100,101,99,100.5,1000",
            ]
        ),
        encoding="utf-8",
    )

    importer = PriceFileImporter(local_data_dir=tmp_path / "raw")
    imported, missing = importer.import_files(source, ["SPY", "QQQ"])

    assert len(imported) == 1
    assert imported[0].ticker == "SPY"
    assert imported[0].destination.exists()
    assert missing == ["QQQ"]


def test_import_from_zip_extracts_matching_files(tmp_path: Path):
    archive_path = tmp_path / "vendor.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "nested/aapl.us.txt",
            "\n".join(
                [
                    "Date,Open,High,Low,Close,Volume",
                    "2024-01-02,100,101,99,100.5,1000",
                ]
            ),
        )

    importer = PriceFileImporter(local_data_dir=tmp_path / "raw")
    imported, missing = importer.import_files(archive_path, ["AAPL"])

    assert len(imported) == 1
    assert imported[0].ticker == "AAPL"
    assert imported[0].destination.exists()
    assert missing == []
