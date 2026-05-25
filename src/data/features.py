"""Feature engineering for regime and alpha models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Compute technical, market, and regime-level features."""

    TECHNICAL_WINDOWS = (1, 5, 21, 63)

    def compute_technical_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute asset-level technical features from OHLCV prices."""
        close = self._extract_field(prices, ("Adj Close", "Close"))
        volume = self._extract_field(prices, ("Volume",))

        vix_candidates = {"^VIX", "VIX"}
        vix_ticker = next((ticker for ticker in close.columns if ticker in vix_candidates), None)
        asset_tickers = [ticker for ticker in close.columns if ticker != vix_ticker]

        features: dict[tuple[str, str], pd.Series] = {}

        for ticker in asset_tickers:
            series = close[ticker]
            ticker_volume = volume[ticker] if ticker in volume.columns else pd.Series(index=series.index, dtype=float)
            log_price = np.log(series.where(series > 0))
            daily_returns = log_price.diff()

            for window in self.TECHNICAL_WINDOWS:
                features[(ticker, f"return_{window}d")] = log_price.diff(window)

            features[(ticker, "volatility_21d")] = daily_returns.rolling(21).std() * np.sqrt(252)
            features[(ticker, "volatility_63d")] = daily_returns.rolling(63).std() * np.sqrt(252)
            features[(ticker, "momentum_12_1")] = np.log(series.shift(21) / series.shift(252))
            features[(ticker, "volume_ratio_20d")] = ticker_volume / ticker_volume.rolling(20).mean()
            features[(ticker, "price_to_ma50")] = series / series.rolling(50).mean() - 1.0
            features[(ticker, "price_to_ma200")] = series / series.rolling(200).mean() - 1.0

            rolling_mean = series.rolling(20).mean()
            rolling_std = series.rolling(20).std()
            features[(ticker, "bollinger_zscore")] = (series - rolling_mean) / rolling_std

        technical = pd.DataFrame(features, index=close.index)
        technical.columns = pd.MultiIndex.from_tuples(technical.columns, names=["ticker", "feature"])

        if vix_ticker is not None:
            vix = close[vix_ticker]
            technical[("MARKET", "vix_level")] = vix
            technical[("MARKET", "vix_5d_change")] = vix.pct_change(5)

        return technical.sort_index(axis=1)

    def compute_market_features(self, returns: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
        """Compute daily market-level features from asset returns."""
        aligned_returns = returns.copy()
        aligned_returns.index = pd.to_datetime(aligned_returns.index).tz_localize(None)

        if benchmark not in aligned_returns.columns:
            raise KeyError(f"Benchmark {benchmark} not found in returns.")

        dispersion_frame = aligned_returns.drop(columns=[benchmark], errors="ignore")
        pairwise_corr = self._rolling_average_correlation(dispersion_frame, window=63)

        market_features = pd.DataFrame(
            {
                "market_return": aligned_returns[benchmark],
                "market_volatility_21d": aligned_returns[benchmark].rolling(21).std() * np.sqrt(252),
                "cross_sectional_dispersion": dispersion_frame.std(axis=1),
                "average_pairwise_correlation_63d": pairwise_corr,
            },
            index=aligned_returns.index,
        )
        return market_features

    def normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-sectional z-score normalization clipped to +/-3."""
        if isinstance(features.columns, pd.MultiIndex):
            normalized_blocks: list[pd.DataFrame] = []
            market_columns = []

            for feature_name in features.columns.get_level_values(1).unique():
                feature_block = features.xs(feature_name, axis=1, level=1)
                if "MARKET" in feature_block.columns:
                    market_columns.append(feature_name)
                    continue
                normalized_feature = self._normalize_block(feature_block)
                normalized_feature.columns = pd.MultiIndex.from_product(
                    [normalized_feature.columns, [feature_name]],
                    names=["ticker", "feature"],
                )
                normalized_blocks.append(normalized_feature)

            normalized = (
                pd.concat(normalized_blocks, axis=1).sort_index(axis=1) if normalized_blocks else pd.DataFrame(index=features.index)
            )

            for feature_name in market_columns:
                normalized[("MARKET", feature_name)] = features[("MARKET", feature_name)]

            return normalized.sort_index(axis=1)

        return self._normalize_block(features)

    def compute_regime_features(
        self,
        prices: pd.DataFrame,
        vix: pd.Series | None = None,
        macro: pd.DataFrame | None = None,
        benchmark: str = "SPY",
        high_yield_proxy: str = "HYG",
        investment_grade_proxy: str = "LQD",
    ) -> pd.DataFrame:
        """Build the market-level feature matrix used by the regime model."""
        close = self._extract_field(prices, ("Adj Close", "Close"))
        returns = np.log(close / close.shift(1))

        if benchmark not in close.columns:
            raise KeyError(f"Benchmark {benchmark} not found in prices.")

        market_features = self.compute_market_features(returns.dropna(how="all"), benchmark=benchmark)
        benchmark_prices = close[benchmark]

        vix_series = vix
        if vix_series is None:
            for candidate in ("^VIX", "VIX"):
                if candidate in close.columns:
                    vix_series = close[candidate]
                    break

        if vix_series is None:
            vix_series = pd.Series(np.nan, index=close.index, name="vix")

        regime_features = pd.DataFrame(index=close.index)
        regime_features["spy_return"] = returns[benchmark]
        regime_features["spy_volatility_21d"] = returns[benchmark].rolling(21).std() * np.sqrt(252)
        regime_features["vix_level"] = vix_series.reindex(close.index)
        regime_features["vix_5d_change"] = vix_series.reindex(close.index).pct_change(5)
        regime_features["spy_momentum_63d"] = np.log(benchmark_prices / benchmark_prices.shift(63))
        regime_features["cross_sectional_dispersion"] = market_features["cross_sectional_dispersion"].reindex(close.index)
        regime_features["yield_curve_slope"] = self._yield_curve_slope(close.index, macro)
        regime_features["credit_spread_proxy"] = self._credit_spread_proxy(
            close,
            high_yield_proxy=high_yield_proxy,
            investment_grade_proxy=investment_grade_proxy,
        )

        return regime_features.dropna(how="all")

    @staticmethod
    def _extract_field(prices: pd.DataFrame, fields: Iterable[str]) -> pd.DataFrame:
        if not isinstance(prices.columns, pd.MultiIndex):
            raise TypeError("Expected price data with (ticker, field) MultiIndex columns.")

        available_fields = prices.columns.get_level_values(1)
        field = next((candidate for candidate in fields if candidate in available_fields), None)
        if field is None:
            raise KeyError(f"None of the fields {tuple(fields)} were present in price data.")

        field_frame = prices.xs(field, axis=1, level=1).sort_index()
        field_frame.columns.name = None
        return field_frame

    @staticmethod
    def _normalize_block(block: pd.DataFrame) -> pd.DataFrame:
        mean = block.mean(axis=1)
        std = block.std(axis=1).replace(0.0, np.nan)
        normalized = block.sub(mean, axis=0).div(std, axis=0)
        return normalized.clip(-3.0, 3.0)

    @staticmethod
    def _rolling_average_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
        values = pd.Series(index=returns.index, dtype=float)

        for end_idx in range(window - 1, len(returns)):
            window_frame = returns.iloc[end_idx - window + 1 : end_idx + 1].dropna(axis=1, how="any")
            if window_frame.shape[1] < 2:
                continue
            corr = window_frame.corr().to_numpy()
            upper = corr[np.triu_indices_from(corr, k=1)]
            values.iloc[end_idx] = float(np.nanmean(upper))

        return values

    @staticmethod
    def _yield_curve_slope(index: pd.Index, macro: pd.DataFrame | None) -> pd.Series:
        if macro is None:
            return pd.Series(np.nan, index=index)

        candidates = [
            ("10Y", "2Y"),
            ("DGS10", "DGS2"),
            ("UST10Y", "UST2Y"),
        ]
        for long_key, short_key in candidates:
            if long_key in macro.columns and short_key in macro.columns:
                return (macro[long_key] - macro[short_key]).reindex(index).ffill()

        return pd.Series(np.nan, index=index)

    @staticmethod
    def _credit_spread_proxy(
        close: pd.DataFrame,
        high_yield_proxy: str,
        investment_grade_proxy: str,
    ) -> pd.Series:
        if high_yield_proxy not in close.columns or investment_grade_proxy not in close.columns:
            return pd.Series(np.nan, index=close.index)

        ratio = close[high_yield_proxy] / close[investment_grade_proxy]
        return np.log(ratio)
