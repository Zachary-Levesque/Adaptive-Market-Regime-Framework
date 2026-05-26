"""Factor and macro data loaders."""

from __future__ import annotations

import sys
import types
from typing import Mapping

import pandas as pd
from packaging.version import parse as parse_version

try:  # pragma: no cover - exercised indirectly depending on environment
    from loguru import logger
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    import logging

    logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised indirectly depending on environment
    # pandas-datareader 0.10 still imports distutils on Python 3.12.
    class _LooseVersion:
        def __init__(self, version: str) -> None:
            self.version = version
            self._parsed = parse_version(version)

        def __lt__(self, other: object) -> bool:
            if not isinstance(other, _LooseVersion):
                return NotImplemented
            return self._parsed < other._parsed

        def __le__(self, other: object) -> bool:
            if not isinstance(other, _LooseVersion):
                return NotImplemented
            return self._parsed <= other._parsed

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, _LooseVersion):
                return NotImplemented
            return self._parsed == other._parsed

        def __gt__(self, other: object) -> bool:
            if not isinstance(other, _LooseVersion):
                return NotImplemented
            return self._parsed > other._parsed

        def __ge__(self, other: object) -> bool:
            if not isinstance(other, _LooseVersion):
                return NotImplemented
            return self._parsed >= other._parsed

    distutils_module = types.ModuleType("distutils")
    distutils_version_module = types.ModuleType("distutils.version")
    distutils_version_module.LooseVersion = _LooseVersion
    distutils_module.version = distutils_version_module
    sys.modules.setdefault("distutils", distutils_module)
    sys.modules.setdefault("distutils.version", distutils_version_module)

    from pandas_datareader import data as web
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    web = None


class FactorLoader:
    """Load Fama-French factors and optional macro series."""

    FF5_DAILY_DATASET = "F-F_Research_Data_5_Factors_2x3_daily"
    MOMENTUM_DATASETS = (
        "F-F_Momentum_Factor_daily",
        "F-F_Momentum_Factor",
    )

    def download_ff5(self) -> pd.DataFrame:
        """Download daily Fama-French 5 factors and momentum."""
        if web is None:
            raise ImportError("pandas-datareader is required for download_ff5(). Install dependencies first.")

        ff5_raw = web.DataReader(self.FF5_DAILY_DATASET, "famafrench")[0]
        ff5 = self._clean_factor_frame(ff5_raw)

        momentum = None
        for dataset in self.MOMENTUM_DATASETS:
            try:
                momentum_raw = web.DataReader(dataset, "famafrench")[0]
                momentum = self._clean_factor_frame(momentum_raw)
                break
            except Exception:  # pragma: no cover - depends on remote dataset availability
                logger.warning("Momentum dataset {} not available", dataset)

        if momentum is None:
            raise RuntimeError("Unable to download a Fama-French momentum dataset.")

        if momentum.shape[1] != 1:
            momentum = momentum.iloc[:, :1]
        momentum.columns = ["UMD"]

        factors = ff5.join(momentum, how="inner").sort_index()
        logger.info("Loaded Fama-French factors with {} observations", len(factors))
        return factors

    def align_with_returns(self, factors: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Align factors to the dates present in returns."""
        aligned = factors.copy()
        aligned.index = pd.to_datetime(aligned.index).tz_localize(None)
        aligned = aligned.reindex(pd.to_datetime(returns.index).tz_localize(None))
        return aligned.ffill().dropna(how="all")

    def download_macro_series(
        self,
        series_map: Mapping[str, str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Download named FRED series into a single DataFrame."""
        if web is None:
            raise ImportError(
                "pandas-datareader is required for download_macro_series(). Install dependencies first."
            )

        macro = web.DataReader(list(series_map.values()), "fred", start, end)
        inverse = {fred_code: label for label, fred_code in series_map.items()}
        macro = macro.rename(columns=inverse)
        macro.index = pd.to_datetime(macro.index).tz_localize(None)
        return macro.sort_index()

    @staticmethod
    def _clean_factor_frame(frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.copy()
        parsed_index = pd.to_datetime(cleaned.index.astype(str), format="%Y%m%d", errors="coerce")
        if parsed_index.isna().any():
            parsed_index = pd.to_datetime(cleaned.index.astype(str), errors="raise")
        cleaned.index = parsed_index
        cleaned.columns = [column.strip() for column in cleaned.columns]
        return cleaned.astype(float).div(100.0)
