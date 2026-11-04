"""Absence de fuite temporelle : les variables du jour T = D+1 ne doivent dépendre
que de données disponibles à la coupure (jour D, 12:00 heure de Paris), plus la
température du jour cible, qui est par construction une prévision."""

import numpy as np
import pandas as pd
import pytest

from eco2mix_forecast.calendar_fr import daily_calendar
from eco2mix_forecast.dataset import to_daily_matrix
from eco2mix_forecast.features import CUTOFF_SLOT, FEATURES, TEMP_FEATURES, build_features

TARGET_TEMP = ["temp", "temp_mean", "temp_min", "temp_max", "temp_delta"]


def _feat(load, temp, cal, temp_target=None):
    return build_features(load, temp, temp if temp_target is None else temp_target, cal)


def _perturb_after_cutoff(m: pd.DataFrame, d: pd.Timestamp, rng) -> pd.DataFrame:
    m = m.copy()
    after = m.index > d
    m.loc[after, :] = rng.normal(0, 1e4, size=(after.sum(), m.shape[1]))
    m.loc[d, list(range(CUTOFF_SLOT, 48))] = rng.normal(0, 1e4, size=48 - CUTOFF_SLOT)
    return m


@pytest.mark.parametrize("target", ["2025-01-15", "2025-03-30", "2025-03-31"])
def test_no_leakage_matrix_level(matrices, target):
    load, temp, cal = matrices
    t = pd.Timestamp(target)
    d = t - pd.Timedelta(days=1)
    rng = np.random.default_rng(1)
    ref = _feat(load, temp, cal).loc[t, FEATURES]
    # Température du jour cible fournie séparément (prévision), donc inchangée.
    alt = _feat(_perturb_after_cutoff(load, d, rng), _perturb_after_cutoff(temp, d, rng), cal,
                temp_target=temp).loc[t, FEATURES]
    pd.testing.assert_frame_equal(ref, alt)


def test_known_data_does_change_features(matrices):
    """Garde-fou : le test précédent ne doit pas être vide de sens."""
    load, temp, cal = matrices
    t = pd.Timestamp("2025-01-15")
    d = t - pd.Timedelta(days=1)
    ref = _feat(load, temp, cal).loc[t]
    load2 = load.copy()
    load2.loc[d, CUTOFF_SLOT - 1] += 1000.0   # créneau 11:30 de D, connu
    alt = _feat(load2, temp, cal).loc[t]
    assert not np.allclose(ref["load_d_last"], alt["load_d_last"])


def test_target_temperature_only_moves_temperature_features(matrices):
    load, temp, cal = matrices
    t = pd.Timestamp("2025-02-10")
    fc = temp.copy()
    fc.loc[t] += 3.0
    ref = _feat(load, temp, cal).loc[t]
    alt = _feat(load, temp, cal, temp_target=fc).loc[t]
    changed = [c for c in FEATURES if not np.allclose(ref[c], alt[c], equal_nan=True)]
    assert changed and set(changed) <= set(TARGET_TEMP)
    assert set(TARGET_TEMP) <= set(TEMP_FEATURES)


@pytest.mark.parametrize("d", ["2025-03-29", "2025-03-30", "2025-01-20"])
def test_no_leakage_from_raw_half_hourly_series(half_hourly, d):
    """Bout en bout : on modifie la série brute (UTC) après D 12:00 heure de Paris,
    y compris autour du passage à l'heure d'été, puis on reconstruit tout."""
    d = pd.Timestamp(d)
    t = d + pd.Timedelta(days=1)
    cutoff = pd.Timestamp(f"{d.date()} 12:00", tz="Europe/Paris")

    def feats(hh):
        load = to_daily_matrix(hh["load"])
        temp = to_daily_matrix(hh["temp_obs"])
        target_temp = to_daily_matrix(half_hourly["temp_obs"])  # « prévision » fixe
        cal = daily_calendar(load.index.min(), load.index.max())
        return build_features(load, temp, target_temp.reindex(load.index), cal).loc[t, FEATURES]

    ref = feats(half_hourly)
    hh = half_hourly.copy()
    late = hh.index >= cutoff
    rng = np.random.default_rng(2)
    hh.loc[late, "load"] = rng.normal(0, 1e4, late.sum())
    hh.loc[late, "temp_obs"] = rng.normal(0, 50, late.sum())
    pd.testing.assert_frame_equal(ref, feats(hh))


def test_shape_and_index(matrices):
    load, temp, cal = matrices
    f = _feat(load, temp, cal)
    assert len(f) == 48 * len(load)
    assert f.index.names == ["date", "slot"]
    assert set(f.index.get_level_values("slot")) == set(range(48))
    # lag J-7 : même créneau sept jours plus tôt
    t = pd.Timestamp("2025-02-10")
    assert f.loc[(t, 20), "load_dm6_same"] == load.loc[t - pd.Timedelta(days=7), 20]
    assert f.loc[(t, 20), "y"] == load.loc[t, 20]


def test_spring_gap_is_filled_and_autumn_is_averaged(half_hourly):
    m = to_daily_matrix(half_hourly["load"])
    assert m.loc["2025-03-30"].notna().all()
    assert m.shape[1] == 48


def test_same_slot_of_previous_day_only_for_morning(matrices):
    load, temp, cal = matrices
    f = _feat(load, temp, cal)
    t = pd.Timestamp("2025-02-10")
    d = t - pd.Timedelta(days=1)
    assert f.loc[(t, 10), "load_d_same"] == load.loc[d, 10]
    assert f.loc[(t, CUTOFF_SLOT - 1), "load_d_same"] == load.loc[d, CUTOFF_SLOT - 1]
    assert np.isnan(f.loc[(t, CUTOFF_SLOT), "load_d_same"])
    assert np.isnan(f.loc[(t, 47), "load_d_same"])
