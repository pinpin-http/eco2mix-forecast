import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_half_hourly

from eco2mix_forecast import config
from eco2mix_forecast.backtest import Fold, fold_masks, run_fold, yearly_folds
from eco2mix_forecast.calendar_fr import daily_calendar
from eco2mix_forecast.dataset import to_daily_matrix
from eco2mix_forecast.features import build_features
from eco2mix_forecast.models import LGBMForecaster, SeasonalMean, persistence_j7


def test_folds_are_expanding_and_exclude_test_year():
    folds, test = yearly_folds([2019, 2020, 2021], test_year=2025, first_train_year=2013)
    assert [f.test_year for f in folds] == [2019, 2020, 2021]
    for f in folds:
        assert f.train_years[0] == 2013
        assert f.train_years[1] == f.test_year - 1
        assert f.test_year != 2025
    assert test.test_year == 2025 and test.train_years == (2013, 2024)
    ends = [f.train_years[1] for f in folds]
    assert ends == sorted(ends)


def test_default_folds_respect_config():
    folds, test = yearly_folds()
    assert test.test_year == config.TEST_YEAR
    assert all(f.test_year < config.TEST_YEAR for f in folds)


def test_overlap_is_rejected():
    with pytest.raises(ValueError):
        yearly_folds([2024, 2025], test_year=2025)


def test_masks_do_not_overlap():
    dates = pd.date_range("2013-01-01", "2025-12-31", freq="D")
    _, test = yearly_folds([2020], test_year=2025)
    tr, te = fold_masks(dates, Fold("x", (2013, 2019), 2020))
    assert not (tr & te).any()
    assert pd.DatetimeIndex(dates[tr]).max() < pd.DatetimeIndex(dates[te]).min()
    tr, te = fold_masks(dates, test)
    assert pd.DatetimeIndex(dates[te]).year.unique().tolist() == [2025]
    assert pd.DatetimeIndex(dates[tr]).max() == pd.Timestamp("2024-12-31")


@pytest.fixture(scope="module")
def small_problem():
    hh = synthetic_half_hourly("2022-12-20", "2025-01-01", seed=3)
    load = to_daily_matrix(hh["load"])
    temp = to_daily_matrix(hh["temp_obs"])
    cal = daily_calendar(load.index.min(), load.index.max())
    return build_features(load, temp, temp, cal), cal


def test_baselines(small_problem):
    feat, cal = small_problem
    t = pd.Timestamp("2024-06-12")
    p = persistence_j7(feat).loc[(t, 10)]
    assert p == feat.loc[(t - pd.Timedelta(days=7), 10), "y"]
    train = feat.loc[:"2023-12-31"].dropna(subset=["y"])
    sm = SeasonalMean().fit(train, cal).predict(feat.loc["2024-01-01":], cal)
    assert sm.notna().all()


def test_lightgbm_smoke(small_problem):
    feat, cal = small_problem
    train = feat.loc["2023-01-01":"2023-12-31"]
    test = feat.loc["2024-01-01":"2024-12-31"]
    for target in ("ratio", "absolute"):
        m = LGBMForecaster(target=target, params={"num_boost_round": 30}).fit(train)
        p = m.predict(test)
        assert p.notna().all()
        err = np.nanmean(np.abs(p - test["y"]) / test["y"])
        assert err < 0.1


def test_run_fold_keeps_test_year_out_of_training(small_problem, monkeypatch):
    feat, cal = small_problem
    seen = {}
    orig = LGBMForecaster.fit

    def spy(self, f):
        seen["max_date"] = f.index.get_level_values("date").max()
        return orig(self, f)

    monkeypatch.setattr(LGBMForecaster, "fit", spy)
    monkeypatch.setitem(__import__("eco2mix_forecast.models", fromlist=["x"]).DEFAULT_PARAMS,
                        "num_boost_round", 20)
    preds, _ = run_fold(feat, cal, Fold("t", (2023, 2023), 2024), targets=("ratio",))
    assert seen["max_date"] < pd.Timestamp("2024-01-01")
    assert pd.DatetimeIndex(preds.index.get_level_values("date")).year.unique().tolist() == [2024]


def test_rte_bias_correction_uses_previous_year_only():
    from eco2mix_forecast.backtest import Matrices, rte_columns
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    load = pd.DataFrame(50000.0, index=days, columns=range(48))
    rte = load.copy()
    rte.loc["2023"] -= 1000.0
    rte.loc["2024"] -= 400.0
    empty = load * np.nan
    m = Matrices(load, empty, empty, empty, rte, pd.DataFrame(index=days))
    idx = pd.MultiIndex.from_product([days, range(48)], names=["date", "slot"])
    out = rte_columns(m, idx)
    # 2024 corrigée avec le biais de 2023 (-1000 MW) : 49600 + 1000
    assert out.loc[(pd.Timestamp("2024-06-01"), 10), "rte_j1_debiaisee"] == pytest.approx(50600.0)
    # 2023 : pas d'année précédente disponible
    assert np.isnan(out.loc[(pd.Timestamp("2023-06-01"), 10), "rte_j1_debiaisee"])
