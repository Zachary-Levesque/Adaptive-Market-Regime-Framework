"""Compare phase 3 alpha models against simple sklearn baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.alpha.baselines import BaselineSpec, build_default_baseline_specs
from src.alpha.dataset import RegimeDataset, extract_regime_series
from src.alpha.ensemble import RegimeAlphaEnsemble
from src.alpha.training import temporal_train_val_split
from src.alpha.walk_forward import WalkForwardValidator
from src.config import AlphaConfig, RegimeConfig

try:  # pragma: no cover - exercised indirectly depending on environment
    from loguru import logger
except ImportError:  # pragma: no cover - dependency may not be installed in CI/local env
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class AlphaComparisonArtifacts:
    fold_metrics: pd.DataFrame
    leaderboard: pd.DataFrame
    best_model: str
    best_signal_path: Path | None
    signal_paths: dict[str, Path]


class AlphaModelComparator:
    """Score the existing ensemble and a set of baseline models on walk-forward splits."""

    def __init__(
        self,
        alpha_config: AlphaConfig,
        regime_config: RegimeConfig,
        baseline_specs: list[BaselineSpec] | None = None,
        transaction_cost_bps: float = 10.0,
        max_gross_exposure: float = 1.0,
        long_fraction: float = 0.2,
        short_fraction: float = 0.2,
        rebalance_interval_days: int = 1,
    ) -> None:
        self.alpha_config = alpha_config
        self.regime_config = regime_config
        self.baseline_specs = baseline_specs or build_default_baseline_specs()
        self.transaction_cost_bps = transaction_cost_bps
        self.max_gross_exposure = max_gross_exposure
        self.long_fraction = long_fraction
        self.short_fraction = short_fraction
        self.rebalance_interval_days = max(1, int(rebalance_interval_days))

    def build(
        self,
        technical_features: pd.DataFrame,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        regime_labels: pd.DataFrame | pd.Series,
        epochs_override: int | None = None,
        include_ensemble: bool = True,
    ) -> AlphaComparisonArtifacts:
        regime_series = extract_regime_series(regime_labels)
        unique_regimes = sorted(int(regime) for regime in regime_series.dropna().unique())
        epochs = int(epochs_override) if epochs_override is not None else self.alpha_config.epochs
        validator = WalkForwardValidator(
            train_window=self.alpha_config.train_window,
            test_window=self.alpha_config.test_window,
            step_size=self.alpha_config.step_size,
        )

        model_specs: list[tuple[str, Callable[[int], object]]] = [
            *[(spec.name, spec.factory) for spec in self.baseline_specs],
        ]
        if include_ensemble:
            model_specs.append(
                (
                    "ensemble",
                    lambda input_size: RegimeAlphaEnsemble(
                        input_size=input_size,
                        hidden_size=self.alpha_config.hidden_size,
                        num_layers=self.alpha_config.num_layers,
                        dropout=self.alpha_config.dropout,
                        learning_rate=self.alpha_config.learning_rate,
                        weight_decay=self.alpha_config.weight_decay,
                        batch_size=self.alpha_config.batch_size,
                        patience=self.alpha_config.patience,
                        sequence_length=self.alpha_config.sequence_length,
                    ),
                )
            )

        fold_frames: list[pd.DataFrame] = []
        signal_frames: dict[str, pd.DataFrame] = {
            model_name: pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=float)
            for model_name, _ in model_specs
        }
        signal_frames["cash"] = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=float)

        for model_name, factory in model_specs:
            for regime in unique_regimes:
                regime_signal_frame, regime_metrics = self._project_regime_model(
                    model_name=model_name,
                    model_factory=factory,
                    regime=regime,
                    technical_features=technical_features,
                    returns=returns,
                    factors=factors,
                    regime_series=regime_series,
                    validator=validator,
                    epochs=epochs,
                )
                if not regime_metrics.empty:
                    fold_frames.append(regime_metrics)
                if not regime_signal_frame.empty:
                    signal_frames[model_name] = signal_frames[model_name].combine_first(regime_signal_frame)

        fold_metrics = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
        leaderboard = self._summarize(fold_metrics)
        leaderboard = self._with_cash_candidate(leaderboard)
        leaderboard = self._attach_signal_stats(leaderboard, signal_frames)
        leaderboard = self._attach_projected_backtest_stats(leaderboard, signal_frames, returns)
        signal_paths = self.save(signal_frames, fold_metrics, leaderboard)
        best_model = str(leaderboard.index[0]) if not leaderboard.empty else ""
        best_signal_path = signal_paths.get(best_model)
        return AlphaComparisonArtifacts(
            fold_metrics=fold_metrics,
            leaderboard=leaderboard,
            best_model=best_model,
            best_signal_path=best_signal_path,
            signal_paths=signal_paths,
        )

    def save(
        self,
        signal_frames: dict[str, pd.DataFrame],
        fold_metrics: pd.DataFrame,
        leaderboard: pd.DataFrame,
    ) -> dict[str, Path]:
        output_path = self.alpha_config.comparison_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fold_metrics.to_parquet(output_path)
        signal_dir = self.alpha_config.signals_dir
        signal_dir.mkdir(parents=True, exist_ok=True)

        signal_paths: dict[str, Path] = {}
        for model_name, frame in signal_frames.items():
            signal_path = signal_dir / f"{model_name}.parquet"
            frame.sort_index().to_parquet(signal_path)
            signal_paths[model_name] = signal_path

        leaderboard_to_save = leaderboard.copy()
        leaderboard_to_save["signal_path"] = pd.Series(
            {model_name: str(path) for model_name, path in signal_paths.items()}
        )
        leaderboard_to_save.to_parquet(output_path.with_name("alpha_model_comparison_summary.parquet"))

        selection = pd.DataFrame(
            [
                {
                    "model": leaderboard_to_save.index[0] if not leaderboard_to_save.empty else "",
                    "signal_path": str(signal_paths.get(leaderboard_to_save.index[0], "")) if not leaderboard_to_save.empty else "",
                }
            ]
        )
        if not leaderboard_to_save.empty:
            top_model = leaderboard_to_save.index[0]
            for column in leaderboard_to_save.columns:
                selection.loc[0, column] = leaderboard_to_save.loc[top_model, column]
            selection.loc[0, "model"] = top_model
            selection.loc[0, "signal_path"] = str(signal_paths[top_model])
        selection.to_parquet(self.alpha_config.selection_path)
        logger.info("Saved alpha model comparison to {}", output_path)
        return signal_paths

    def _project_regime_model(
        self,
        model_name: str,
        model_factory,
        regime: int,
        technical_features: pd.DataFrame,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        regime_series: pd.Series,
        validator: WalkForwardValidator,
        epochs: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        dataset = RegimeDataset(
            features=technical_features,
            returns=returns,
            regime_labels=regime_series,
            target_regime=regime,
            factors=factors,
            sequence_length=self.alpha_config.sequence_length,
            min_samples=self.alpha_config.min_samples_per_regime,
            augment_noise_std=self.alpha_config.augment_noise_std,
        )
        if len(dataset) < 2 or dataset.input_size == 0:
            logger.warning("Skipping regime {} due to insufficient samples", regime)
            return pd.DataFrame(), pd.DataFrame()

        regime_signal_frame = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=float)
        fold_rows: list[dict[str, float | int]] = []
        usable_dates = technical_features.index.intersection(returns.index).intersection(regime_series.index)
        splits = validator.generate_splits(usable_dates)

        for fold, (train_dates, test_dates) in enumerate(splits):
            train_dataset = RegimeDataset(
                features=technical_features,
                returns=returns,
                regime_labels=regime_series,
                target_regime=regime,
                factors=factors,
                sequence_length=self.alpha_config.sequence_length,
                min_samples=self.alpha_config.min_samples_per_regime,
                augment_noise_std=self.alpha_config.augment_noise_std,
                allowed_dates=train_dates,
            )
            test_dataset = RegimeDataset(
                features=technical_features,
                returns=returns,
                regime_labels=regime_series,
                target_regime=regime,
                factors=factors,
                sequence_length=self.alpha_config.sequence_length,
                min_samples=0,
                augment_noise_std=self.alpha_config.augment_noise_std,
                allowed_dates=test_dates,
            )
            if len(train_dataset) < 2 or len(test_dataset) == 0:
                continue

            train_subset, val_subset = temporal_train_val_split(train_dataset, self.alpha_config.validation_fraction)
            model = model_factory(train_dataset.input_size)
            model.fit(train_subset, val_subset, epochs=epochs, device=self.alpha_config.device)
            predictions = model.predict_dataset(test_dataset, device=self.alpha_config.device)
            actuals = test_dataset.targets.detach().cpu().numpy()
            trading_stats = self._compute_trading_stats(
                predictions=predictions,
                actuals=actuals,
                dates=test_dataset.sample_dates,
                tickers=test_dataset.sample_tickers,
            )

            fold_rows.append(
                {
                    "model": model_name,
                    "regime": regime,
                    "fold": fold,
                    "n_train": len(train_dataset),
                    "n_test": len(test_dataset),
                    "sharpe": trading_stats["gross_sharpe"],
                    "net_sharpe": trading_stats["net_sharpe"],
                    "mean_turnover": trading_stats["mean_turnover"],
                    "mean_transaction_cost": trading_stats["mean_transaction_cost"],
                    "ic": validator._safe_corr(predictions, actuals, method="pearson"),
                    "rank_ic": validator._safe_corr(predictions, actuals, method="spearman"),
                    "hit_rate": float(np.mean(np.sign(predictions) == np.sign(actuals))),
                }
            )

            for date, ticker, prediction in zip(test_dataset.sample_dates, test_dataset.sample_tickers, predictions):
                regime_signal_frame.at[pd.Timestamp(date), ticker] = float(prediction)

        return regime_signal_frame, pd.DataFrame(fold_rows)

    @staticmethod
    def _summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
        if fold_metrics.empty:
            empty = pd.DataFrame(
                columns=[
                    "n_rows",
                    "n_folds",
                    "n_regimes",
                    "mean_sharpe",
                    "median_sharpe",
                    "mean_net_sharpe",
                    "mean_ic",
                    "mean_rank_ic",
                    "mean_hit_rate",
                    "mean_turnover",
                    "mean_transaction_cost",
                    "mean_train_size",
                    "mean_test_size",
                ]
            )
            empty.index.name = "model"
            return empty

        rows: list[dict[str, float | int | str]] = []
        for model_name, group in fold_metrics.groupby("model"):
            rows.append(
                {
                    "model": model_name,
                    "n_rows": int(len(group)),
                    "n_folds": int(group["fold"].nunique()) if "fold" in group.columns else int(len(group)),
                    "n_regimes": int(group["regime"].nunique()) if "regime" in group.columns else 0,
                    "mean_sharpe": float(group["sharpe"].mean()) if "sharpe" in group.columns else 0.0,
                    "median_sharpe": float(group["sharpe"].median()) if "sharpe" in group.columns else 0.0,
                    "mean_net_sharpe": float(group["net_sharpe"].mean()) if "net_sharpe" in group.columns else 0.0,
                    "mean_ic": float(group["ic"].mean()) if "ic" in group.columns else 0.0,
                    "mean_rank_ic": float(group["rank_ic"].mean()) if "rank_ic" in group.columns else 0.0,
                    "mean_hit_rate": float(group["hit_rate"].mean()) if "hit_rate" in group.columns else 0.0,
                    "mean_turnover": float(group["mean_turnover"].mean()) if "mean_turnover" in group.columns else 0.0,
                    "mean_transaction_cost": float(group["mean_transaction_cost"].mean())
                    if "mean_transaction_cost" in group.columns
                    else 0.0,
                    "mean_train_size": float(group["n_train"].mean()) if "n_train" in group.columns else 0.0,
                    "mean_test_size": float(group["n_test"].mean()) if "n_test" in group.columns else 0.0,
                }
            )

        leaderboard = pd.DataFrame(rows).set_index("model")
        leaderboard = leaderboard.sort_values(["mean_net_sharpe", "mean_sharpe", "mean_ic"], ascending=False)
        return leaderboard

    @staticmethod
    def _with_cash_candidate(leaderboard: pd.DataFrame) -> pd.DataFrame:
        cash_row = pd.DataFrame(
            [
                {
                    "n_rows": 0,
                    "n_folds": 0,
                    "n_regimes": 0,
                    "mean_sharpe": 0.0,
                    "median_sharpe": 0.0,
                    "mean_net_sharpe": 0.0,
                    "mean_ic": 0.0,
                    "mean_rank_ic": 0.0,
                    "mean_hit_rate": 0.0,
                    "mean_turnover": 0.0,
                    "mean_transaction_cost": 0.0,
                    "mean_train_size": 0.0,
                    "mean_test_size": 0.0,
                }
            ],
            index=pd.Index(["cash"], name="model"),
        )
        combined = cash_row if leaderboard.empty else pd.concat([leaderboard, cash_row])
        return combined.sort_values(["mean_net_sharpe", "mean_sharpe", "mean_ic"], ascending=False)

    def _compute_trading_stats(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: pd.DatetimeIndex,
        tickers: list[str],
    ) -> dict[str, float]:
        frame = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(dates),
                "ticker": tickers,
                "prediction": predictions,
                "actual": actuals,
            }
        ).dropna()
        if frame.empty:
            return {
                "gross_sharpe": 0.0,
                "net_sharpe": 0.0,
                "mean_turnover": 0.0,
                "mean_transaction_cost": 0.0,
            }

        daily_returns: list[float] = []
        daily_net_returns: list[float] = []
        daily_turnover: list[float] = []
        previous_weights: pd.Series | None = None
        current_weights: pd.Series | None = None
        last_rebalance_pos: int | None = None

        for pos, (_, group) in enumerate(frame.sort_values("date").groupby("date", sort=True)):
            candidate_weights = self._construct_fold_weights(group)
            if candidate_weights.empty and current_weights is None:
                continue
            should_rebalance = (
                not candidate_weights.empty
                and (last_rebalance_pos is None or pos - last_rebalance_pos >= self.rebalance_interval_days)
            )
            if should_rebalance:
                current_weights = candidate_weights
                last_rebalance_pos = pos

            if current_weights is None:
                continue

            weights = current_weights.copy()
            actual_returns = group.set_index("ticker")["actual"].reindex(weights.index).fillna(0.0)
            gross_return = float((weights * actual_returns).sum())
            if previous_weights is None:
                aligned_previous = pd.Series(0.0, index=weights.index)
            else:
                aligned_index = weights.index.union(previous_weights.index)
                weights = weights.reindex(aligned_index).fillna(0.0)
                actual_returns = actual_returns.reindex(aligned_index).fillna(0.0)
                aligned_previous = previous_weights.reindex(aligned_index).fillna(0.0)
                gross_return = float((weights * actual_returns).sum())

            turnover = float((weights - aligned_previous).abs().sum())
            transaction_cost = turnover * (self.transaction_cost_bps / 10_000.0)
            daily_returns.append(gross_return)
            daily_net_returns.append(gross_return - transaction_cost)
            daily_turnover.append(turnover)
            previous_weights = weights

        return {
            "gross_sharpe": self._annualized_sharpe(daily_returns),
            "net_sharpe": self._annualized_sharpe(daily_net_returns),
            "mean_turnover": float(np.mean(daily_turnover)) if daily_turnover else 0.0,
            "mean_transaction_cost": float(np.mean(daily_turnover) * (self.transaction_cost_bps / 10_000.0))
            if daily_turnover
            else 0.0,
        }

    def _construct_fold_weights(self, group: pd.DataFrame) -> pd.Series:
        clean = group.set_index("ticker")["prediction"].dropna().sort_values()
        if clean.empty:
            return pd.Series(dtype=float)

        n_assets = len(clean)
        n_long = max(1, int(np.ceil(n_assets * self.long_fraction)))
        n_short = max(1, int(np.ceil(n_assets * self.short_fraction)))
        long_names = clean.tail(n_long).index
        short_names = clean.head(n_short).index
        if set(long_names) & set(short_names):
            return pd.Series(dtype=float)

        weights = pd.Series(0.0, index=clean.index, dtype=float)
        gross_side = self.max_gross_exposure / 2.0
        weights.loc[long_names] = gross_side / len(long_names)
        weights.loc[short_names] = -gross_side / len(short_names)
        return weights

    @staticmethod
    def _annualized_sharpe(returns: list[float]) -> float:
        if not returns:
            return 0.0
        series = pd.Series(returns)
        std = float(series.std(ddof=0))
        if std == 0.0:
            return 0.0
        return float(np.sqrt(252.0) * series.mean() / std)

    def _attach_projected_backtest_stats(
        self,
        leaderboard: pd.DataFrame,
        signal_frames: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        if leaderboard.empty:
            return leaderboard

        enriched = leaderboard.copy()
        for model_name, signals in signal_frames.items():
            if model_name not in enriched.index:
                continue

            stats = self._project_signal_backtest(signals, returns)
            for key, value in stats.items():
                enriched.loc[model_name, key] = value

        return enriched.sort_values(
            ["projected_backtest_sharpe", "projected_total_return", "mean_net_sharpe"],
            ascending=False,
        )

    def _project_signal_backtest(self, signals: pd.DataFrame, returns: pd.DataFrame) -> dict[str, float]:
        normalized_signals = self._normalize_frame(signals)
        normalized_returns = self._normalize_frame(returns)
        common_index = normalized_returns.index.intersection(normalized_signals.index).sort_values()
        common_columns = normalized_returns.columns.intersection(normalized_signals.columns).sort_values()
        if common_index.empty or common_columns.empty:
            return {
                "projected_backtest_sharpe": 0.0,
                "projected_total_return": 0.0,
                "projected_mean_turnover": 0.0,
            }

        signals = normalized_signals.loc[common_index, common_columns]
        returns = normalized_returns.loc[common_index, common_columns].fillna(0.0)
        active_rows = signals.notna().any(axis=1)
        if not active_rows.any():
            return {
                "projected_backtest_sharpe": 0.0,
                "projected_total_return": 0.0,
                "projected_mean_turnover": 0.0,
            }

        first_active = active_rows[active_rows].index[0]
        signals = signals.loc[signals.index >= first_active]
        returns = returns.loc[returns.index >= first_active]

        raw_weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for date, row in signals.iterrows():
            weights = self._construct_fold_weights(
                pd.DataFrame({"ticker": row.index, "prediction": pd.to_numeric(row, errors="coerce").to_numpy()})
            )
            if not weights.empty:
                raw_weights.loc[date, weights.index] = weights

        target_weights = self._apply_rebalance_schedule(raw_weights)
        applied_weights = target_weights.shift(1).reindex(returns.index).fillna(0.0)
        gross_returns = (applied_weights * returns).sum(axis=1)
        turnover = applied_weights.diff().abs().sum(axis=1).fillna(applied_weights.abs().sum(axis=1))
        transaction_cost = turnover * (self.transaction_cost_bps / 10_000.0)
        strategy_returns = gross_returns - transaction_cost

        return {
            "projected_backtest_sharpe": self._annualized_sharpe(strategy_returns.tolist()),
            "projected_total_return": float((1.0 + strategy_returns).prod() - 1.0),
            "projected_mean_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        }

    def _apply_rebalance_schedule(self, weights: pd.DataFrame) -> pd.DataFrame:
        interval = max(1, int(self.rebalance_interval_days))
        if interval <= 1 or weights.empty:
            return weights

        scheduled = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        current = pd.Series(0.0, index=weights.columns, dtype=float)
        last_rebalance_pos: int | None = None
        for pos, (date, row) in enumerate(weights.iterrows()):
            has_signal = bool(row.abs().sum() > 0.0)
            should_rebalance = has_signal and (
                last_rebalance_pos is None or pos - last_rebalance_pos >= interval
            )
            if should_rebalance:
                current = row.copy()
                last_rebalance_pos = pos
            scheduled.loc[date] = current
        return scheduled

    @staticmethod
    def _attach_signal_stats(leaderboard: pd.DataFrame, signal_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if leaderboard.empty:
            return leaderboard

        enriched = leaderboard.copy()
        for model_name, frame in signal_frames.items():
            if model_name not in enriched.index:
                continue

            active_rows = frame.notna().any(axis=1)
            active_days = int(active_rows.sum())
            enriched.loc[model_name, "active_signal_days"] = active_days
            enriched.loc[model_name, "mean_signal_coverage"] = float(frame.notna().mean().mean()) if not frame.empty else 0.0
            enriched.loc[model_name, "first_signal_date"] = (
                str(frame.index[active_rows][0].date()) if active_days else ""
            )
            enriched.loc[model_name, "last_signal_date"] = (
                str(frame.index[active_rows][-1].date()) if active_days else ""
            )

        return enriched

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
        normalized = normalized.apply(pd.to_numeric, errors="coerce")
        return normalized.replace([np.inf, -np.inf], np.nan).sort_index()
