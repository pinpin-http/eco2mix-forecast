import numpy as np
import pandas as pd
import pytest

from eco2mix_forecast.metrics import bias, mae, mape, rmse, score_table, season_of


def test_simple_values():
    y = [100.0, 200.0]
    p = [110.0, 180.0]
    assert mape(y, p) == pytest.approx(10.0)
    assert mae(y, p) == pytest.approx(15.0)
    assert rmse(y, p) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert bias(y, p) == pytest.approx(-5.0)


def test_perfect_forecast():
    y = np.array([50000.0, 60000.0, 70000.0])
    assert mape(y, y) == 0.0
    assert rmse(y, y) == 0.0


def test_nan_are_ignored():
    assert mape([100.0, np.nan, 200.0], [110.0, 5.0, 220.0]) == pytest.approx(10.0)


def test_mape_rejects_non_positive_truth():
    with pytest.raises(ValueError):
        mape([0.0, 100.0], [1.0, 100.0])


def test_score_table_uses_the_same_rows_for_all_models():
    df = pd.DataFrame({"y": [100.0, 100.0, 100.0], "a": [110.0, np.nan, 90.0],
                       "b": [100.0, 100.0, 100.0], "g": ["x", "x", "z"]})
    t = score_table(df, ["a", "b"])
    assert set(t["n"]) == {2}
    by = score_table(df, ["a", "b"], by="g")
    assert set(by["group"]) == {"x", "z"}
    assert by.loc[(by.group == "x") & (by.model == "a"), "mape"].item() == pytest.approx(10.0)


def test_seasons():
    s = season_of(pd.to_datetime(["2025-01-15", "2025-04-01", "2025-07-14", "2025-11-11", "2025-12-01"]))
    assert list(s) == ["hiver", "printemps", "été", "automne", "hiver"]
