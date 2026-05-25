import pandas as pd

from src.data.factors import FactorLoader


def test_align_with_returns_reindexes_and_forward_fills():
    loader = FactorLoader()
    factor_index = pd.to_datetime(["2024-01-01", "2024-01-03"])
    factors = pd.DataFrame({"Mkt-RF": [0.01, 0.02], "SMB": [0.0, 0.01]}, index=factor_index)

    returns = pd.DataFrame(index=pd.date_range("2024-01-01", periods=4, freq="B"))
    aligned = loader.align_with_returns(factors, returns)

    assert list(aligned.index) == list(returns.index)
    assert aligned.loc[pd.Timestamp("2024-01-02"), "Mkt-RF"] == 0.01
    assert aligned.loc[pd.Timestamp("2024-01-04"), "Mkt-RF"] == 0.02
