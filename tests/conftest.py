"""Petites données synthétiques : aucun test ne télécharge quoi que ce soit."""

import numpy as np
import pandas as pd
import pytest

from eco2mix_forecast.calendar_fr import daily_calendar
from eco2mix_forecast.dataset import to_daily_matrix


def synthetic_half_hourly(start="2024-12-20", end="2025-04-10", seed=0) -> pd.DataFrame:
    """Consommation et température au pas de 30 min (index UTC), avec un profil réaliste."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz="Europe/Paris"), pd.Timestamp(end, tz="Europe/Paris"),
                        freq="30min", inclusive="left").tz_convert("UTC")
    local = idx.tz_convert("Europe/Paris")
    h = local.hour + local.minute / 60
    doy = local.day_of_year.to_numpy()
    temp = 8 - 6 * np.cos(2 * np.pi * (doy - 15) / 365) + 4 * np.sin(2 * np.pi * (h - 9) / 24)
    temp = temp + rng.normal(0, 1, len(idx))
    weekend = local.day_of_week >= 5
    load = 55000 + 1500 * np.maximum(15 - temp, 0) + 6000 * np.sin(np.pi * h / 24) ** 2
    load = load - 5000 * weekend + rng.normal(0, 500, len(idx))
    return pd.DataFrame({"load": load, "temp_obs": temp}, index=idx)


@pytest.fixture
def half_hourly():
    return synthetic_half_hourly()


@pytest.fixture
def matrices(half_hourly):
    load = to_daily_matrix(half_hourly["load"])
    temp = to_daily_matrix(half_hourly["temp_obs"])
    cal = daily_calendar(load.index.min(), load.index.max())
    return load, temp, cal
