import pandas as pd

from src.alpha.walk_forward import WalkForwardValidator


def test_walk_forward_generate_splits_orders_dates_without_overlap():
    dates = pd.date_range("2021-01-01", periods=300, freq="B")
    validator = WalkForwardValidator(train_window=120, test_window=60, step_size=30)

    splits = validator.generate_splits(dates)

    assert splits
    train_dates, test_dates = splits[0]
    assert len(train_dates) == 120
    assert len(test_dates) == 60
    assert train_dates.max() < test_dates.min()
