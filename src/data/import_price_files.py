"""Import local vendor price files into the Phase 1 raw-data directory."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config
from src.data.ingestion import MarketDataIngester


@dataclass(frozen=True)
class ImportedFile:
    ticker: str
    source: str
    destination: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import local price files for AMRF Phase 1.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Directory or ZIP archive containing vendor price files.",
    )
    parser.add_argument(
        "--destination-subdir",
        default="imported",
        help="Subdirectory to create under local_data_dir for imported files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    price_universe = [ticker for ticker in config.data.universe if ticker not in {"^VIX", "VIX"}]

    importer = PriceFileImporter(
        local_data_dir=config.data.local_data_dir,
        destination_subdir=args.destination_subdir,
    )
    imported, missing = importer.import_files(Path(args.source), price_universe)

    print(f"Imported files: {len(imported)}")
    for item in imported:
        print(f"  {item.ticker}: {item.source} -> {item.destination}")

    print(f"Missing after import: {len(missing)}")
    for ticker in missing:
        candidates = ", ".join(MarketDataIngester._local_filename_candidates(ticker))
        print(f"  {ticker}: no source file matched [{candidates}]")

    print("\nNext:")
    print("  python -m src.data.validate_phase1_inputs --config configs/config.yaml")
    print("  python -m src.data.build_phase1 --config configs/config.yaml")


class PriceFileImporter:
    """Import vendor files into a consistent local raw-data layout."""

    def __init__(self, local_data_dir: Path, destination_subdir: str = "imported") -> None:
        self.local_data_dir = local_data_dir
        self.destination_dir = local_data_dir / destination_subdir

    def import_files(self, source: Path, tickers: list[str]) -> tuple[list[ImportedFile], list[str]]:
        self.destination_dir.mkdir(parents=True, exist_ok=True)

        if source.is_dir():
            return self._import_from_directory(source, tickers)
        if zipfile.is_zipfile(source):
            return self._import_from_zip(source, tickers)

        raise ValueError(f"Unsupported source: {source}. Expected a directory or ZIP archive.")

    def _import_from_directory(self, source_dir: Path, tickers: list[str]) -> tuple[list[ImportedFile], list[str]]:
        imported: list[ImportedFile] = []
        missing: list[str] = []

        for ticker in tickers:
            match = self._find_in_directory(source_dir, ticker)
            if match is None:
                missing.append(ticker)
                continue

            destination = self._destination_path(ticker, match.suffix or ".csv")
            shutil.copy2(match, destination)
            imported.append(ImportedFile(ticker=ticker, source=str(match), destination=destination))

        return imported, missing

    def _import_from_zip(self, zip_path: Path, tickers: list[str]) -> tuple[list[ImportedFile], list[str]]:
        imported: list[ImportedFile] = []
        missing: list[str] = []

        with zipfile.ZipFile(zip_path) as archive:
            lowered_names = {name.lower(): name for name in archive.namelist()}

            for ticker in tickers:
                archive_name = self._find_in_archive(lowered_names, ticker)
                if archive_name is None:
                    missing.append(ticker)
                    continue

                destination = self._destination_path(ticker, Path(archive_name).suffix or ".csv")
                with archive.open(archive_name) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                imported.append(ImportedFile(ticker=ticker, source=f"{zip_path}:{archive_name}", destination=destination))

        return imported, missing

    def _find_in_directory(self, source_dir: Path, ticker: str) -> Path | None:
        candidates = {candidate.lower() for candidate in MarketDataIngester._local_filename_candidates(ticker)}
        for path in source_dir.rglob("*"):
            if path.is_file() and path.name.lower() in candidates:
                return path
        return None

    def _find_in_archive(self, lowered_names: dict[str, str], ticker: str) -> str | None:
        candidates = {candidate.lower() for candidate in MarketDataIngester._local_filename_candidates(ticker)}
        for lowered, original in lowered_names.items():
            if Path(lowered).name in candidates:
                return original
        return None

    def _destination_path(self, ticker: str, suffix: str) -> Path:
        normalized_suffix = suffix if suffix in {".txt", ".csv"} else ".csv"
        return self.destination_dir / f"{ticker.lower()}{normalized_suffix}"


if __name__ == "__main__":
    main()
