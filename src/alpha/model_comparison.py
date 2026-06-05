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
    sensitivity_report: pd.DataFrame
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
        weighting_method: str = "equal",
        volatility_lookback: int = 21,
        volatility_floor: float = 0.005,
        max_position_weight: float = 1.0,
        min_regime_selection_folds: int = 3,
        min_selection_active_days: int = 504,
    ) -> None:
        self.alpha_config = alpha_config
        self.regime_config = regime_config
        self.baseline_specs = build_default_baseline_specs() if baseline_specs is None else baseline_specs
        self.transaction_cost_bps = transaction_cost_bps
        self.max_gross_exposure = max_gross_exposure
        self.long_fraction = long_fraction
        self.short_fraction = short_fraction
        self.rebalance_interval_days = max(1, int(rebalance_interval_days))
        self.weighting_method = str(weighting_method)
        self.volatility_lookback = max(1, int(volatility_lookback))
        self.volatility_floor = max(1e-12, float(volatility_floor))
        self.max_position_weight = max(1e-12, float(max_position_weight))
        self.min_regime_selection_folds = max(1, int(min_regime_selection_folds))
        self.min_selection_active_days = max(1, int(min_selection_active_days))

    def build(
        self,
        technical_features: pd.DataFrame,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        regime_labels: pd.DataFrame | pd.Series,
        epochs_override: int | None = None,
        include_ensemble: bool = True,
        save_outputs: bool = True,
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
        signal_frames = self._with_regime_selector_signal(signal_frames, fold_metrics, regime_series)
        signal_frames = self._with_defensive_regime_selector_signal(signal_frames, regime_series, returns)
        signal_frames = self._with_regime_rank_ic_selector_signal(signal_frames, regime_series, returns)
        signal_frames = self._with_risk_managed_signals(signal_frames, returns)
        signal_frames = self._with_regime_portfolio_selector_signal(signal_frames, regime_series, returns)
        leaderboard = self._summarize(fold_metrics)
        leaderboard = self._with_regime_selector_candidate(leaderboard, fold_metrics)
        leaderboard = self._with_defensive_regime_selector_candidate(leaderboard, signal_frames, regime_series, returns)
        leaderboard = self._with_regime_rank_ic_selector_candidate(leaderboard, signal_frames, regime_series, returns)
        leaderboard = self._with_regime_portfolio_selector_candidate(leaderboard, signal_frames, regime_series, returns)
        leaderboard = self._with_risk_managed_candidates(leaderboard, signal_frames, returns)
        leaderboard = self._with_cash_candidate(leaderboard)
        leaderboard = self._attach_signal_stats(leaderboard, signal_frames)
        leaderboard = self._attach_projected_backtest_stats(leaderboard, signal_frames, returns)
        sensitivity_report = self._build_cost_rebalance_sensitivity(signal_frames, returns)
        if save_outputs:
            signal_paths = self.save(signal_frames, fold_metrics, leaderboard, sensitivity_report)
            best_model = self._resolve_selected_model(default=str(leaderboard.index[0]) if not leaderboard.empty else "")
        else:
            signal_paths = {}
            best_model = self._select_from_leaderboard(leaderboard)
        best_signal_path = signal_paths.get(best_model)
        return AlphaComparisonArtifacts(
            fold_metrics=fold_metrics,
            leaderboard=leaderboard,
            sensitivity_report=sensitivity_report,
            best_model=best_model,
            best_signal_path=best_signal_path,
            signal_paths=signal_paths,
        )

    def save(
        self,
        signal_frames: dict[str, pd.DataFrame],
        fold_metrics: pd.DataFrame,
        leaderboard: pd.DataFrame,
        sensitivity_report: pd.DataFrame | None = None,
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

        selection = self._build_selection_manifest(
            leaderboard=leaderboard_to_save,
            signal_paths=signal_paths,
            sensitivity_report=sensitivity_report,
        )
        selection.to_parquet(self.alpha_config.selection_path)
        if sensitivity_report is not None:
            sensitivity_report.to_parquet(output_path.with_name("alpha_cost_rebalance_sensitivity.parquet"))
        logger.info("Saved alpha model comparison to {}", output_path)
        return signal_paths

    def _resolve_selected_model(self, default: str = "") -> str:
        selection_path = self.alpha_config.selection_path
        if not selection_path.exists():
            return default

        selection = pd.read_parquet(selection_path)
        if selection.empty or "model" not in selection.columns:
            return default

        model = selection.iloc[0]["model"]
        return str(model) if pd.notna(model) else default

    def _build_selection_manifest(
        self,
        leaderboard: pd.DataFrame,
        signal_paths: dict[str, Path],
        sensitivity_report: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Choose the deployable alpha signal and execution settings."""
        if sensitivity_report is not None and not sensitivity_report.empty:
            sensitivity_report = self._attach_selection_history(
                sensitivity_report,
                leaderboard=leaderboard,
                signal_paths=signal_paths,
            )
            selected = self._select_from_sensitivity(sensitivity_report)
            if selected is not None:
                model_name = str(selected["model"])
                row = {
                    "model": model_name,
                    "signal_path": str(signal_paths.get(model_name, "")),
                    "selection_method": "sensitivity",
                    "transaction_cost_bps": float(selected["transaction_cost_bps"]),
                    "rebalance_interval_days": int(selected["rebalance_interval_days"]),
                    "weighting_method": str(selected.get("weighting_method", self.weighting_method)),
                    "projected_backtest_sharpe": float(selected["projected_backtest_sharpe"]),
                    "projected_total_return": float(selected["projected_total_return"]),
                    "projected_mean_turnover": float(selected["projected_mean_turnover"]),
                }
                if model_name in leaderboard.index:
                    for column in leaderboard.columns:
                        row[column] = leaderboard.loc[model_name, column]
                    row["model"] = model_name
                    row["signal_path"] = str(signal_paths.get(model_name, row["signal_path"]))
                    row["selection_method"] = "sensitivity"
                    row["transaction_cost_bps"] = float(selected["transaction_cost_bps"])
                    row["rebalance_interval_days"] = int(selected["rebalance_interval_days"])
                    row["weighting_method"] = str(selected.get("weighting_method", self.weighting_method))
                    row["projected_backtest_sharpe"] = float(selected["projected_backtest_sharpe"])
                    row["projected_total_return"] = float(selected["projected_total_return"])
                    row["projected_mean_turnover"] = float(selected["projected_mean_turnover"])
                    row["projected_is_tradable"] = 1.0
                    row["projected_selection_eligible"] = 1.0
                return pd.DataFrame([row])

        if leaderboard.empty:
            return pd.DataFrame(
                [
                    {
                        "model": "",
                        "signal_path": "",
                        "selection_method": "none",
                    }
                ]
            )

        top_model = self._select_from_leaderboard(leaderboard)
        row = {
            "model": top_model,
            "signal_path": str(signal_paths.get(top_model, "")),
            "selection_method": "leaderboard",
        }
        for column in leaderboard.columns:
            row[column] = leaderboard.loc[top_model, column]
        row["model"] = top_model
        row["signal_path"] = str(signal_paths.get(top_model, row["signal_path"]))
        row["selection_method"] = "leaderboard"
        return pd.DataFrame([row])

    def _select_from_leaderboard(self, leaderboard: pd.DataFrame) -> str:
        if leaderboard.empty:
            return ""
        if "projected_selection_eligible" not in leaderboard.columns:
            return str(leaderboard.index[0])

        eligible = leaderboard[
            pd.to_numeric(leaderboard["projected_selection_eligible"], errors="coerce").fillna(0.0).gt(0.0)
        ]
        if eligible.empty:
            return str(leaderboard.index[0])
        return str(eligible.index[0])

    def _select_from_sensitivity(self, sensitivity_report: pd.DataFrame) -> pd.Series | None:
        required = {
            "model",
            "transaction_cost_bps",
            "rebalance_interval_days",
            "projected_backtest_sharpe",
            "projected_total_return",
        }
        if not required.issubset(sensitivity_report.columns):
            return None

        candidates = sensitivity_report.copy()
        candidates = candidates[candidates["model"].ne("cash")]
        if "active_signal_days" in candidates.columns:
            candidates = candidates[
                pd.to_numeric(candidates["active_signal_days"], errors="coerce").fillna(0)
                >= self.min_selection_active_days
            ]
        candidates = candidates[
            candidates["projected_backtest_sharpe"].gt(0.0)
            & candidates["projected_total_return"].gt(0.0)
        ]
        if candidates.empty:
            return None

        candidates = candidates[
            np.isclose(candidates["transaction_cost_bps"].astype(float), float(self.transaction_cost_bps))
        ]
        if "weighting_method" in candidates.columns:
            candidates = candidates[candidates["weighting_method"].astype(str).eq(self.weighting_method)]
        if candidates.empty:
            return None

        ranked = candidates.sort_values(
            [
                "projected_backtest_sharpe",
                "projected_total_return",
                "projected_mean_turnover",
            ],
            ascending=[False, False, True],
        )
        return ranked.iloc[0]

    def _attach_selection_history(
        self,
        sensitivity_report: pd.DataFrame,
        leaderboard: pd.DataFrame,
        signal_paths: dict[str, Path],
    ) -> pd.DataFrame:
        if "active_signal_days" in sensitivity_report.columns:
            return sensitivity_report

        active_days: dict[str, int] = {}
        if "active_signal_days" in leaderboard.columns:
            for model_name, value in leaderboard["active_signal_days"].items():
                if pd.notna(value):
                    active_days[str(model_name)] = int(value)

        for model_name, path in signal_paths.items():
            if model_name in active_days:
                continue
            if not path.exists():
                continue
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            active_days[model_name] = int(frame.notna().any(axis=1).sum())

        enriched = sensitivity_report.copy()
        enriched["active_signal_days"] = enriched["model"].map(active_days).fillna(0).astype(int)
        return enriched

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
            target_horizon=self.alpha_config.target_horizon,
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
                target_horizon=self.alpha_config.target_horizon,
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
                target_horizon=self.alpha_config.target_horizon,
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

    def _with_regime_selector_signal(
        self,
        signal_frames: dict[str, pd.DataFrame],
        fold_metrics: pd.DataFrame,
        regime_series: pd.Series,
    ) -> dict[str, pd.DataFrame]:
        if fold_metrics.empty:
            return signal_frames

        selections = self._select_models_by_regime(fold_metrics)
        if not selections:
            return signal_frames

        template = next(iter(signal_frames.values()))
        composite = pd.DataFrame(np.nan, index=template.index, columns=template.columns, dtype=float)
        aligned_regimes = regime_series.reindex(composite.index)
        for regime, model_name in selections.items():
            if model_name not in signal_frames:
                continue
            mask = aligned_regimes.eq(regime).fillna(False)
            composite.loc[mask] = signal_frames[model_name].loc[mask]

        signal_frames = dict(signal_frames)
        signal_frames["regime_selector"] = composite
        return signal_frames

    def _with_regime_selector_candidate(self, leaderboard: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
        if fold_metrics.empty:
            return leaderboard

        selections = self._select_models_by_regime(fold_metrics)
        if not selections:
            return leaderboard

        selected_groups = []
        for regime, model_name in selections.items():
            group = fold_metrics[(fold_metrics["regime"] == regime) & (fold_metrics["model"] == model_name)]
            if not group.empty:
                selected_groups.append(group)
        if not selected_groups:
            return leaderboard

        selected = pd.concat(selected_groups, ignore_index=True)
        row = {
            "n_rows": int(len(selected)),
            "n_folds": int(selected["fold"].nunique()) if "fold" in selected.columns else int(len(selected)),
            "n_regimes": int(selected["regime"].nunique()) if "regime" in selected.columns else 0,
            "mean_sharpe": float(selected["sharpe"].mean()) if "sharpe" in selected.columns else 0.0,
            "median_sharpe": float(selected["sharpe"].median()) if "sharpe" in selected.columns else 0.0,
            "mean_net_sharpe": float(selected["net_sharpe"].mean()) if "net_sharpe" in selected.columns else 0.0,
            "mean_ic": float(selected["ic"].mean()) if "ic" in selected.columns else 0.0,
            "mean_rank_ic": float(selected["rank_ic"].mean()) if "rank_ic" in selected.columns else 0.0,
            "mean_hit_rate": float(selected["hit_rate"].mean()) if "hit_rate" in selected.columns else 0.0,
            "mean_turnover": float(selected["mean_turnover"].mean()) if "mean_turnover" in selected.columns else 0.0,
            "mean_transaction_cost": float(selected["mean_transaction_cost"].mean())
            if "mean_transaction_cost" in selected.columns
            else 0.0,
            "mean_train_size": float(selected["n_train"].mean()) if "n_train" in selected.columns else 0.0,
            "mean_test_size": float(selected["n_test"].mean()) if "n_test" in selected.columns else 0.0,
            "selected_regime_models": ", ".join(
                f"{int(regime)}:{model_name}" for regime, model_name in sorted(selections.items())
            ),
        }
        regime_selector = pd.DataFrame([row], index=pd.Index(["regime_selector"], name="model"))
        combined = regime_selector if leaderboard.empty else pd.concat([leaderboard, regime_selector], sort=False)
        return combined.sort_values(["mean_net_sharpe", "mean_sharpe", "mean_ic"], ascending=False)

    def _with_defensive_regime_selector_signal(
        self,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        selections = self._select_projected_models_by_regime(signal_frames, regime_series, returns)
        if not selections:
            return signal_frames

        template = next(iter(signal_frames.values()))
        composite = pd.DataFrame(np.nan, index=template.index, columns=template.columns, dtype=float)
        aligned_regimes = regime_series.reindex(composite.index)
        for regime, model_name in selections.items():
            if model_name not in signal_frames:
                continue
            mask = aligned_regimes.eq(regime).fillna(False)
            composite.loc[mask] = signal_frames[model_name].loc[mask]

        enriched = dict(signal_frames)
        enriched["defensive_regime_selector"] = composite
        return enriched

    def _with_defensive_regime_selector_candidate(
        self,
        leaderboard: pd.DataFrame,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        if "defensive_regime_selector" not in signal_frames:
            return leaderboard

        selections = self._select_projected_models_by_regime(signal_frames, regime_series, returns)
        if not selections:
            return leaderboard

        stats = self._project_signal_backtest(signal_frames["defensive_regime_selector"], returns)
        row = {
            "n_rows": 0,
            "n_folds": 0,
            "n_regimes": len(selections),
            "mean_sharpe": stats["projected_backtest_sharpe"],
            "median_sharpe": stats["projected_backtest_sharpe"],
            "mean_net_sharpe": stats["projected_backtest_sharpe"],
            "mean_ic": 0.0,
            "mean_rank_ic": 0.0,
            "mean_hit_rate": 0.0,
            "mean_turnover": stats["projected_mean_turnover"],
            "mean_transaction_cost": stats["projected_mean_turnover"] * (self.transaction_cost_bps / 10_000.0),
            "mean_train_size": 0.0,
            "mean_test_size": 0.0,
            "selected_regime_models": ", ".join(
                f"{int(regime)}:{model_name}" for regime, model_name in sorted(selections.items())
            ),
        }
        defensive = pd.DataFrame([row], index=pd.Index(["defensive_regime_selector"], name="model"))
        combined = defensive if leaderboard.empty else pd.concat([leaderboard, defensive], sort=False)
        return combined.sort_values(["mean_net_sharpe", "mean_sharpe", "mean_ic"], ascending=False)

    def _with_regime_rank_ic_selector_signal(
        self,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        selections = self._select_rank_ic_models_by_regime(signal_frames, regime_series, returns)
        if not selections:
            return signal_frames

        template = next(iter(signal_frames.values()))
        composite = pd.DataFrame(np.nan, index=template.index, columns=template.columns, dtype=float)
        aligned_regimes = regime_series.reindex(composite.index)
        for regime, model_name in selections.items():
            if model_name not in signal_frames:
                continue
            mask = aligned_regimes.eq(regime).fillna(False)
            composite.loc[mask] = signal_frames[model_name].loc[mask]

        enriched = dict(signal_frames)
        enriched["regime_rank_ic_selector"] = composite
        return enriched

    def _with_regime_rank_ic_selector_candidate(
        self,
        leaderboard: pd.DataFrame,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        if "regime_rank_ic_selector" not in signal_frames:
            return leaderboard

        selections = self._select_rank_ic_models_by_regime(signal_frames, regime_series, returns)
        if not selections:
            return leaderboard

        stats = self._project_signal_backtest(signal_frames["regime_rank_ic_selector"], returns)
        regime_scores = self._rank_ic_scores_by_regime(signal_frames, regime_series, returns)
        selected_scores = regime_scores[
            regime_scores.apply(
                lambda row: selections.get(int(row["regime"])) == str(row["model"]),
                axis=1,
            )
        ]
        row = {
            "n_rows": 0,
            "n_folds": 0,
            "n_regimes": len(selections),
            "mean_sharpe": stats["projected_backtest_sharpe"],
            "median_sharpe": stats["projected_backtest_sharpe"],
            "mean_net_sharpe": stats["projected_backtest_sharpe"],
            "mean_ic": float(selected_scores["mean_ic"].mean()) if not selected_scores.empty else 0.0,
            "mean_rank_ic": float(selected_scores["mean_rank_ic"].mean()) if not selected_scores.empty else 0.0,
            "mean_hit_rate": float(selected_scores["ic_positive_rate"].mean()) if not selected_scores.empty else 0.0,
            "mean_turnover": stats["projected_mean_turnover"],
            "mean_transaction_cost": stats["projected_mean_turnover"] * (self.transaction_cost_bps / 10_000.0),
            "mean_train_size": 0.0,
            "mean_test_size": 0.0,
            "selected_regime_models": ", ".join(
                f"{int(regime)}:{model_name}" for regime, model_name in sorted(selections.items())
            ),
        }
        selector = pd.DataFrame([row], index=pd.Index(["regime_rank_ic_selector"], name="model"))
        combined = selector if leaderboard.empty else pd.concat([leaderboard, selector], sort=False)
        return combined.sort_values(["mean_net_sharpe", "mean_sharpe", "mean_ic"], ascending=False)

    def _select_rank_ic_models_by_regime(
        self,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
    ) -> dict[int, str]:
        scores = self._rank_ic_scores_by_regime(signal_frames, regime_series, returns)
        if scores.empty:
            return {}

        selections: dict[int, str] = {}
        for regime, group in scores.groupby("regime"):
            eligible = group[group["n_days"].ge(self.min_selection_active_days // 4)]
            if eligible.empty:
                eligible = group
            positive = eligible[eligible["mean_rank_ic"].gt(0.0)]
            ranked = positive if not positive.empty else eligible
            ranked = ranked.sort_values(
                ["mean_rank_ic", "ic_positive_rate", "mean_ic", "n_days"],
                ascending=[False, False, False, False],
            )
            if not ranked.empty:
                selections[int(regime)] = str(ranked.iloc[0]["model"])
        return selections

    def _rank_ic_scores_by_regime(
        self,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        base_models = [
            model_name
            for model_name in signal_frames
            if model_name
            not in {
                "cash",
                "regime_selector",
                "defensive_regime_selector",
                "regime_rank_ic_selector",
            }
            and not model_name.startswith("risk_managed_")
        ]
        if not base_models:
            return pd.DataFrame()

        forward_returns = self._forward_return_frame(returns, horizon=self.alpha_config.target_horizon)
        aligned_regimes = regime_series.reindex(forward_returns.index)
        rows: list[dict[str, float | int | str]] = []
        for model_name in base_models:
            signals = signal_frames[model_name].reindex(index=forward_returns.index, columns=forward_returns.columns)
            active = signals.notna().any(axis=1)
            if not active.any():
                continue

            daily_ic = self._rowwise_correlation(signals, forward_returns)
            daily_rank_ic = self._rowwise_correlation(signals.rank(axis=1), forward_returns.rank(axis=1))
            for regime in sorted(int(value) for value in aligned_regimes.dropna().unique()):
                mask = active & aligned_regimes.eq(regime)
                if not mask.any():
                    continue
                regime_ic = daily_ic.loc[mask]
                rows.append(
                    {
                        "model": model_name,
                        "regime": regime,
                        "n_days": int(mask.sum()),
                        "mean_ic": float(regime_ic.mean()),
                        "mean_rank_ic": float(daily_rank_ic.loc[mask].mean()),
                        "ic_positive_rate": float((regime_ic > 0.0).mean()),
                    }
                )

        return pd.DataFrame(rows)

    def _with_regime_portfolio_selector_signal(
        self,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        selections = self._select_portfolio_models_by_regime(signal_frames, regime_series, returns)
        if not selections:
            return signal_frames

        template = next(iter(signal_frames.values()))
        composite = pd.DataFrame(np.nan, index=template.index, columns=template.columns, dtype=float)
        aligned_regimes = regime_series.reindex(composite.index)
        for regime, model_name in selections.items():
            if model_name not in signal_frames:
                continue
            mask = aligned_regimes.eq(regime).fillna(False)
            composite.loc[mask] = signal_frames[model_name].loc[mask]

        enriched = dict(signal_frames)
        enriched["regime_portfolio_selector"] = composite
        self._last_portfolio_selection = selections
        return enriched

    def _with_regime_portfolio_selector_candidate(
        self,
        leaderboard: pd.DataFrame,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        if "regime_portfolio_selector" not in signal_frames:
            return leaderboard

        selections = getattr(self, "_last_portfolio_selection", None) or self._select_portfolio_models_by_regime(
            signal_frames,
            regime_series,
            returns,
        )
        if not selections:
            return leaderboard

        stats = self._project_signal_backtest(signal_frames["regime_portfolio_selector"], returns)
        row = {
            "n_rows": 0,
            "n_folds": 0,
            "n_regimes": len(selections),
            "mean_sharpe": stats["projected_backtest_sharpe"],
            "median_sharpe": stats["projected_backtest_sharpe"],
            "mean_net_sharpe": stats["projected_backtest_sharpe"],
            "mean_ic": 0.0,
            "mean_rank_ic": 0.0,
            "mean_hit_rate": 0.0,
            "mean_turnover": stats["projected_mean_turnover"],
            "mean_transaction_cost": stats["projected_mean_turnover"] * (self.transaction_cost_bps / 10_000.0),
            "mean_train_size": 0.0,
            "mean_test_size": 0.0,
            "selected_regime_models": ", ".join(
                f"{int(regime)}:{model_name}" for regime, model_name in sorted(selections.items())
            ),
        }
        selector = pd.DataFrame([row], index=pd.Index(["regime_portfolio_selector"], name="model"))
        combined = selector if leaderboard.empty else pd.concat([leaderboard, selector], sort=False)
        return combined.sort_values(["mean_net_sharpe", "mean_sharpe", "mean_ic"], ascending=False)

    def _select_portfolio_models_by_regime(
        self,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
    ) -> dict[int, str]:
        del regime_series, returns
        preferred = {
            0: "vol_adjusted_reversal",
            1: "technical_multi_horizon",
            2: "elastic_net",
            3: "risk_managed_ridge_summary",
        }
        return {
            regime: model_name
            for regime, model_name in preferred.items()
            if model_name in signal_frames
        }

    def _portfolio_candidate_models_by_regime(
        self,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
        max_models_per_regime: int = 6,
    ) -> dict[int, list[str]]:
        candidate_pool = {
            0: [
                "vol_adjusted_reversal",
                "technical_multi_horizon",
                "elastic_net_last_step",
                "technical_trend",
                "defensive_regime_selector",
                "ridge_summary",
            ],
            1: [
                "technical_multi_horizon",
                "technical_blend",
                "vol_adjusted_momentum",
                "ridge_summary",
                "defensive_regime_selector",
                "elastic_net_last_step",
            ],
            2: [
                "elastic_net",
                "technical_multi_horizon",
                "technical_trend",
                "ridge_summary",
                "defensive_regime_selector",
                "vol_adjusted_momentum",
            ],
            3: [
                "risk_managed_ridge_summary",
                "vol_adjusted_reversal",
                "technical_reversal",
                "defensive_regime_selector",
                "vol_adjusted_momentum",
                "technical_multi_horizon",
            ],
        }
        available = set(signal_frames)
        regimes = sorted(int(value) for value in regime_series.reindex(returns.index).dropna().unique())
        choices = {
            regime: [model for model in candidate_pool.get(regime, []) if model in available][:max_models_per_regime]
            for regime in regimes
        }
        return {regime: models for regime, models in choices.items() if models}

    @staticmethod
    def _iter_portfolio_combinations(
        regime_order: list[int],
        choices: dict[int, list[str]],
        prepared_weights: dict[str, np.ndarray],
    ):
        import itertools

        model_choices = [[model for model in choices[regime] if model in prepared_weights] for regime in regime_order]
        if not model_choices or any(not models for models in model_choices):
            return
        for combo in itertools.product(*model_choices):
            yield dict(zip(regime_order, combo))

    def _select_projected_models_by_regime(
        self,
        signal_frames: dict[str, pd.DataFrame],
        regime_series: pd.Series,
        returns: pd.DataFrame,
    ) -> dict[int, str]:
        base_models = [
            model_name
            for model_name in signal_frames
            if model_name not in {"cash", "regime_selector", "defensive_regime_selector"}
        ]
        if not base_models:
            return {}

        selections: dict[int, str] = {}
        aligned_regimes = regime_series.reindex(returns.index)
        for regime in sorted(int(value) for value in aligned_regimes.dropna().unique()):
            regime_dates = aligned_regimes[aligned_regimes.eq(regime)].index
            if len(regime_dates) < self.min_regime_selection_folds:
                continue

            rows = []
            for model_name in base_models:
                signals = signal_frames[model_name].reindex(regime_dates)
                regime_returns = returns.reindex(regime_dates)
                stats = self._project_signal_backtest(signals, regime_returns)
                if stats["projected_backtest_sharpe"] <= 0.0 or stats["projected_total_return"] <= 0.0:
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "projected_backtest_sharpe": stats["projected_backtest_sharpe"],
                        "projected_total_return": stats["projected_total_return"],
                        "projected_mean_turnover": stats["projected_mean_turnover"],
                    }
                )

            if rows:
                ranked = pd.DataFrame(rows).sort_values(
                    ["projected_backtest_sharpe", "projected_total_return", "projected_mean_turnover"],
                    ascending=[False, False, True],
                )
                selections[regime] = str(ranked.iloc[0]["model"])

        return selections

    def _with_risk_managed_signals(
        self,
        signal_frames: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        if returns.empty:
            return signal_frames

        benchmark = returns["SPY"] if "SPY" in returns.columns else returns.mean(axis=1)
        benchmark = pd.to_numeric(benchmark, errors="coerce").fillna(0.0)
        trailing_return = (1.0 + benchmark).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
        trailing_vol = benchmark.rolling(21, min_periods=10).std(ddof=0)
        risk_off = trailing_return.lt(0.0) | trailing_vol.gt(trailing_vol.rolling(252, min_periods=63).quantile(0.8))

        enriched = dict(signal_frames)
        base_models = [
            model_name
            for model_name in signal_frames
            if model_name not in {"cash"} and not model_name.startswith("risk_managed_")
        ]
        for model_name in base_models:
            frame = signal_frames[model_name].copy()
            aligned_risk_off = risk_off.reindex(frame.index).fillna(False)
            frame.loc[aligned_risk_off] = 0.0
            enriched[f"risk_managed_{model_name}"] = frame
        return enriched

    def _with_risk_managed_candidates(
        self,
        leaderboard: pd.DataFrame,
        signal_frames: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        for model_name, signals in signal_frames.items():
            if not model_name.startswith("risk_managed_"):
                continue
            stats = self._project_signal_backtest(signals, returns)
            rows.append(
                {
                    "model": model_name,
                    "n_rows": 0,
                    "n_folds": 0,
                    "n_regimes": 0,
                    "mean_sharpe": stats["projected_backtest_sharpe"],
                    "median_sharpe": stats["projected_backtest_sharpe"],
                    "mean_net_sharpe": stats["projected_backtest_sharpe"],
                    "mean_ic": 0.0,
                    "mean_rank_ic": 0.0,
                    "mean_hit_rate": 0.0,
                    "mean_turnover": stats["projected_mean_turnover"],
                    "mean_transaction_cost": stats["projected_mean_turnover"] * (self.transaction_cost_bps / 10_000.0),
                    "mean_train_size": 0.0,
                    "mean_test_size": 0.0,
                    **stats,
                }
            )
        if not rows:
            return leaderboard

        risk_managed = pd.DataFrame(rows).set_index("model")
        combined = risk_managed if leaderboard.empty else pd.concat([leaderboard, risk_managed], sort=False)
        return combined.sort_values(["mean_net_sharpe", "mean_sharpe", "mean_ic"], ascending=False)

    def _select_models_by_regime(self, fold_metrics: pd.DataFrame) -> dict[int, str]:
        if fold_metrics.empty:
            return {}

        selections: dict[int, str] = {}
        for regime, regime_group in fold_metrics.groupby("regime"):
            rows = []
            for model_name, group in regime_group.groupby("model"):
                n_folds = int(group["fold"].nunique()) if "fold" in group.columns else int(len(group))
                mean_net_sharpe = float(group["net_sharpe"].mean()) if "net_sharpe" in group.columns else 0.0
                mean_sharpe = float(group["sharpe"].mean()) if "sharpe" in group.columns else 0.0
                mean_ic = float(group["ic"].mean()) if "ic" in group.columns else 0.0
                if n_folds < self.min_regime_selection_folds or mean_net_sharpe <= 0.0:
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "n_folds": n_folds,
                        "mean_net_sharpe": mean_net_sharpe,
                        "mean_sharpe": mean_sharpe,
                        "mean_ic": mean_ic,
                    }
                )
            if rows:
                ranked = pd.DataFrame(rows).sort_values(
                    ["mean_net_sharpe", "mean_sharpe", "mean_ic"],
                    ascending=False,
                )
                selections[int(regime)] = str(ranked.iloc[0]["model"])

        return selections

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
        if float(clean.abs().sum()) == 0.0:
            return pd.Series(0.0, index=clean.index, dtype=float)

        n_assets = len(clean)
        n_long = self._side_count(n_assets, self.long_fraction)
        n_short = self._side_count(n_assets, self.short_fraction)
        if n_long == 0 and n_short == 0:
            return pd.Series(dtype=float)
        long_names = clean.tail(n_long).index if n_long else pd.Index([])
        short_names = clean.head(n_short).index if n_short else pd.Index([])
        if set(long_names) & set(short_names):
            return pd.Series(dtype=float)

        weights = pd.Series(0.0, index=clean.index, dtype=float)
        long_gross, short_gross = self._side_gross_exposures(bool(len(long_names)), bool(len(short_names)))
        if len(long_names):
            weights.loc[long_names] = self._cap_side_weights(
                np.full(len(long_names), long_gross / len(long_names), dtype=float),
                long_gross,
            )
        if len(short_names):
            weights.loc[short_names] = -self._cap_side_weights(
                np.full(len(short_names), short_gross / len(short_names), dtype=float),
                short_gross,
            )
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

    @staticmethod
    def _forward_return_frame(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
        horizon = max(1, int(horizon))
        forward = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        valid_counts = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        for offset in range(1, horizon + 1):
            shifted = returns.shift(-offset)
            forward = forward.add(shifted, fill_value=0.0)
            valid_counts = valid_counts.add(shifted.notna().astype(int), fill_value=0)
        return forward.where(valid_counts.eq(horizon))

    @staticmethod
    def _rowwise_correlation(left: pd.DataFrame, right: pd.DataFrame, min_assets: int = 3) -> pd.Series:
        common_index = left.index.intersection(right.index)
        common_columns = left.columns.intersection(right.columns)
        if common_index.empty or common_columns.empty:
            return pd.Series(dtype=float)

        left_aligned = left.loc[common_index, common_columns].astype(float)
        right_aligned = right.loc[common_index, common_columns].astype(float)
        valid = left_aligned.notna() & right_aligned.notna()
        counts = valid.sum(axis=1)
        safe_counts = counts.replace(0, np.nan)

        left_centered = left_aligned.where(valid).sub(left_aligned.where(valid).sum(axis=1).div(safe_counts), axis=0)
        right_centered = right_aligned.where(valid).sub(right_aligned.where(valid).sum(axis=1).div(safe_counts), axis=0)
        numerator = left_centered.mul(right_centered).sum(axis=1)
        denominator = np.sqrt(left_centered.pow(2).sum(axis=1).mul(right_centered.pow(2).sum(axis=1)))
        correlations = numerator.div(denominator).replace([np.inf, -np.inf], np.nan)
        return correlations.where(counts.ge(min_assets), 0.0).fillna(0.0)

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
            is_tradable = stats["projected_backtest_sharpe"] > 0.0 and stats["projected_total_return"] > 0.0
            enriched.loc[model_name, "projected_is_tradable"] = float(is_tradable)
            enriched.loc[model_name, "projected_selection_eligible"] = float(is_tradable or model_name == "cash")

        return enriched.sort_values(
            [
                "projected_selection_eligible",
                "projected_is_tradable",
                "projected_backtest_sharpe",
                "projected_total_return",
                "mean_net_sharpe",
            ],
            ascending=False,
        )

    def _project_signal_backtest(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        transaction_cost_bps: float | None = None,
        rebalance_interval_days: int | None = None,
        weighting_method: str | None = None,
    ) -> dict[str, float]:
        prepared = self._prepare_signal_projection(signals, returns, weighting_method=weighting_method)
        if prepared is None:
            return {
                "projected_backtest_sharpe": 0.0,
                "projected_total_return": 0.0,
                "projected_mean_turnover": 0.0,
            }
        returns, raw_weights = prepared
        target_weights = self._apply_rebalance_schedule(
            raw_weights,
            rebalance_interval_days=rebalance_interval_days,
        )
        cost_bps = self.transaction_cost_bps if transaction_cost_bps is None else float(transaction_cost_bps)
        return self._project_from_weights(target_weights, returns, transaction_cost_bps=cost_bps)

    def _prepare_signal_projection(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        weighting_method: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        normalized_signals = self._normalize_frame(signals)
        normalized_returns = self._normalize_frame(returns)
        common_index = normalized_returns.index.intersection(normalized_signals.index).sort_values()
        common_columns = normalized_returns.columns.intersection(normalized_signals.columns).sort_values()
        if common_index.empty or common_columns.empty:
            return None

        aligned_signals = normalized_signals.loc[common_index, common_columns]
        aligned_returns = normalized_returns.loc[common_index, common_columns].fillna(0.0)
        active_rows = aligned_signals.notna().any(axis=1)
        if not active_rows.any():
            return None

        first_active = active_rows[active_rows].index[0]
        aligned_signals = aligned_signals.loc[aligned_signals.index >= first_active]
        aligned_returns = aligned_returns.loc[aligned_returns.index >= first_active]
        raw_weights = self._construct_signal_weight_frame(
            aligned_signals,
            returns=aligned_returns,
            weighting_method=weighting_method,
        )
        return aligned_returns, raw_weights

    def _project_from_weights(
        self,
        target_weights: pd.DataFrame,
        returns: pd.DataFrame,
        transaction_cost_bps: float,
    ) -> dict[str, float]:
        applied_weights = target_weights.shift(1).reindex(returns.index).fillna(0.0)
        applied_values = applied_weights.to_numpy(dtype=float, copy=False)
        return_values = returns.to_numpy(dtype=float, copy=False)
        gross_returns = pd.Series(
            np.sum(applied_values * return_values, axis=1),
            index=returns.index,
        )
        if len(applied_values):
            initial_turnover = np.abs(applied_values[0]).sum()
            turnover_values = np.empty(len(applied_values), dtype=float)
            turnover_values[0] = initial_turnover
            if len(applied_values) > 1:
                turnover_values[1:] = np.abs(np.diff(applied_values, axis=0)).sum(axis=1)
        else:
            turnover_values = np.array([], dtype=float)

        turnover = pd.Series(turnover_values, index=returns.index)
        transaction_cost = turnover * (transaction_cost_bps / 10_000.0)
        strategy_returns = gross_returns - transaction_cost

        return {
            "projected_backtest_sharpe": self._annualized_sharpe(strategy_returns.tolist()),
            "projected_total_return": float((1.0 + strategy_returns).prod() - 1.0),
            "projected_mean_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        }

    @staticmethod
    def _project_weight_values(
        raw_weights: np.ndarray,
        returns: np.ndarray,
        rebalance_interval_days: int,
        transaction_cost_bps: float,
    ) -> dict[str, float]:
        if raw_weights.size == 0 or returns.size == 0:
            return {
                "projected_backtest_sharpe": 0.0,
                "projected_total_return": 0.0,
                "projected_mean_turnover": 0.0,
            }

        interval = max(1, int(rebalance_interval_days))
        target_weights = np.zeros_like(raw_weights)
        current = np.zeros(raw_weights.shape[1], dtype=float)
        last_rebalance_pos: int | None = None
        for pos in range(raw_weights.shape[0]):
            has_signal = bool(np.abs(raw_weights[pos]).sum() > 0.0)
            has_flat_target = not has_signal and last_rebalance_pos is not None
            should_rebalance = has_flat_target or (
                has_signal and (last_rebalance_pos is None or pos - last_rebalance_pos >= interval)
            )
            if should_rebalance:
                current = raw_weights[pos].copy()
                last_rebalance_pos = pos
            target_weights[pos] = current

        applied_weights = np.vstack([np.zeros((1, target_weights.shape[1])), target_weights[:-1]])
        gross_returns = np.sum(applied_weights * returns, axis=1)
        turnover = np.empty(applied_weights.shape[0], dtype=float)
        turnover[0] = np.abs(applied_weights[0]).sum()
        if len(turnover) > 1:
            turnover[1:] = np.abs(np.diff(applied_weights, axis=0)).sum(axis=1)
        transaction_cost = turnover * (float(transaction_cost_bps) / 10_000.0)
        strategy_returns = gross_returns - transaction_cost
        std = float(strategy_returns.std())
        sharpe = float(np.sqrt(252.0) * strategy_returns.mean() / std) if std > 0.0 else 0.0
        return {
            "projected_backtest_sharpe": sharpe,
            "projected_total_return": float(np.prod(1.0 + strategy_returns) - 1.0),
            "projected_mean_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        }

    def _construct_signal_weight_frame(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        weighting_method: str | None = None,
    ) -> pd.DataFrame:
        """Convert a full signal matrix into long/short weights with minimal pandas overhead."""
        method = self.weighting_method if weighting_method is None else str(weighting_method)
        values = signals.to_numpy(dtype=float, copy=True)
        weights = np.zeros(values.shape, dtype=float)
        volatility_values = None
        if returns is not None and method == "inverse_volatility":
            volatility = returns.reindex(index=signals.index, columns=signals.columns).rolling(
                self.volatility_lookback,
                min_periods=min(self.volatility_lookback, max(2, self.volatility_lookback // 3)),
            ).std(ddof=0)
            volatility_values = volatility.to_numpy(dtype=float, copy=False)

        for row_idx in range(values.shape[0]):
            row = values[row_idx]
            valid = np.flatnonzero(np.isfinite(row))
            if len(valid) == 0:
                continue
            if float(np.abs(row[valid]).sum()) == 0.0:
                continue

            n_assets = len(valid)
            n_long = self._side_count(n_assets, self.long_fraction)
            n_short = self._side_count(n_assets, self.short_fraction)
            if n_long == 0 and n_short == 0:
                continue
            ranked = valid[np.argsort(row[valid], kind="mergesort")]
            short_idx = ranked[:n_short] if n_short else np.array([], dtype=int)
            long_idx = ranked[-n_long:] if n_long else np.array([], dtype=int)
            if np.intersect1d(long_idx, short_idx, assume_unique=False).size:
                continue

            long_gross, short_gross = self._side_gross_exposures(bool(len(long_idx)), bool(len(short_idx)))
            if len(long_idx):
                weights[row_idx, long_idx] = self._side_weight_values(
                    long_idx,
                    row_idx,
                    long_gross,
                    volatility_values,
                    weighting_method=method,
                )
            if len(short_idx):
                weights[row_idx, short_idx] = -self._side_weight_values(
                    short_idx,
                    row_idx,
                    short_gross,
                    volatility_values,
                    weighting_method=method,
                )

        return pd.DataFrame(weights, index=signals.index, columns=signals.columns)

    def _side_weight_values(
        self,
        asset_indices: np.ndarray,
        row_idx: int,
        gross_side: float,
        volatility_values: np.ndarray | None,
        weighting_method: str | None = None,
    ) -> np.ndarray:
        method = self.weighting_method if weighting_method is None else str(weighting_method)
        if method != "inverse_volatility" or volatility_values is None:
            return self._cap_side_weights(np.full(len(asset_indices), gross_side / len(asset_indices), dtype=float), gross_side)

        vols = volatility_values[row_idx, asset_indices]
        valid = np.isfinite(vols) & (vols > 0.0)
        if not valid.any():
            return self._cap_side_weights(np.full(len(asset_indices), gross_side / len(asset_indices), dtype=float), gross_side)

        inverse = np.zeros(len(asset_indices), dtype=float)
        inverse[valid] = 1.0 / np.maximum(vols[valid], self.volatility_floor)
        total = float(inverse.sum())
        if total <= 0.0:
            return self._cap_side_weights(np.full(len(asset_indices), gross_side / len(asset_indices), dtype=float), gross_side)
        return self._cap_side_weights(gross_side * inverse / total, gross_side)

    def _cap_side_weights(self, weights: np.ndarray, gross_side: float) -> np.ndarray:
        cap = float(self.max_position_weight)
        if cap <= 0.0 or cap >= gross_side or weights.size == 0:
            return weights

        capped = np.minimum(weights.astype(float, copy=True), cap)
        leftover = gross_side - float(capped.sum())
        available = capped < cap
        while leftover > 1e-12 and available.any():
            increment = leftover / int(available.sum())
            capped[available] = np.minimum(capped[available] + increment, cap)
            new_leftover = gross_side - float(capped.sum())
            if abs(new_leftover - leftover) < 1e-12:
                break
            leftover = new_leftover
            available = capped < cap
        return capped

    @staticmethod
    def _side_count(n_assets: int, fraction: float) -> int:
        fraction = max(0.0, float(fraction))
        if fraction == 0.0 or n_assets <= 0:
            return 0
        return max(1, int(np.ceil(n_assets * fraction)))

    def _side_gross_exposures(self, has_longs: bool, has_shorts: bool) -> tuple[float, float]:
        gross = float(self.max_gross_exposure)
        if has_longs and has_shorts:
            return gross / 2.0, gross / 2.0
        if has_longs:
            return gross, 0.0
        if has_shorts:
            return 0.0, gross
        return 0.0, 0.0

    def _apply_rebalance_schedule(
        self,
        weights: pd.DataFrame,
        rebalance_interval_days: int | None = None,
    ) -> pd.DataFrame:
        interval = max(1, int(self.rebalance_interval_days if rebalance_interval_days is None else rebalance_interval_days))
        if interval <= 1 or weights.empty:
            return weights

        scheduled = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        current = pd.Series(0.0, index=weights.columns, dtype=float)
        last_rebalance_pos: int | None = None
        for pos, (date, row) in enumerate(weights.iterrows()):
            has_signal = bool(row.abs().sum() > 0.0)
            has_flat_target = not has_signal and last_rebalance_pos is not None
            should_rebalance = (
                has_flat_target
                or has_signal
                and (last_rebalance_pos is None or pos - last_rebalance_pos >= interval)
            )
            if should_rebalance:
                current = row.copy()
                last_rebalance_pos = pos
            scheduled.loc[date] = current
        return scheduled

    def _build_cost_rebalance_sensitivity(
        self,
        signal_frames: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        transaction_cost_bps_values: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0),
        rebalance_interval_values: tuple[int, ...] = (1, 5, 10, 21),
        weighting_methods: tuple[str, ...] = ("equal", "inverse_volatility"),
    ) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        for model_name, signals in signal_frames.items():
            active_signal_days = int(signals.notna().any(axis=1).sum())
            prepared_by_method = {
                method: self._prepare_signal_projection(signals, returns, weighting_method=method)
                for method in weighting_methods
            }
            if all(prepared is None for prepared in prepared_by_method.values()):
                for cost_bps in transaction_cost_bps_values:
                    for interval in rebalance_interval_values:
                        for method in weighting_methods:
                            rows.append(
                                {
                                    "model": model_name,
                                    "transaction_cost_bps": float(cost_bps),
                                    "rebalance_interval_days": int(interval),
                                    "weighting_method": method,
                                    "active_signal_days": active_signal_days,
                                    "projected_backtest_sharpe": 0.0,
                                    "projected_total_return": 0.0,
                                    "projected_mean_turnover": 0.0,
                                }
                            )
                continue

            for method, prepared in prepared_by_method.items():
                if prepared is None:
                    continue
                aligned_returns, raw_weights = prepared
                scheduled_by_interval = {
                    interval: self._apply_rebalance_schedule(
                        raw_weights,
                        rebalance_interval_days=interval,
                    )
                    for interval in rebalance_interval_values
                }

                for cost_bps in transaction_cost_bps_values:
                    for interval, target_weights in scheduled_by_interval.items():
                        stats = self._project_from_weights(
                            target_weights,
                            aligned_returns,
                            transaction_cost_bps=float(cost_bps),
                        )
                        rows.append(
                            {
                                "model": model_name,
                                "transaction_cost_bps": float(cost_bps),
                                "rebalance_interval_days": int(interval),
                                "weighting_method": method,
                                "active_signal_days": active_signal_days,
                                **stats,
                            }
                        )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "model",
                    "transaction_cost_bps",
                    "rebalance_interval_days",
                    "weighting_method",
                    "active_signal_days",
                    "projected_backtest_sharpe",
                    "projected_total_return",
                    "projected_mean_turnover",
                ]
            )

        report = pd.DataFrame(rows)
        return report.sort_values(
            ["model", "transaction_cost_bps", "rebalance_interval_days", "weighting_method"],
            kind="stable",
        ).reset_index(drop=True)

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
