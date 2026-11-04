"""Construction des variables explicatives dans le cadre J+1 strict.

Convention : on se place le jour D à midi (heure de Paris) et on prévoit les
48 demi-heures du jour cible T = D+1. Sont alors connus :
- la consommation jusqu'au créneau 11:30 de D inclus (créneaux 0 à 23) ;
- les températures observées jusqu'au même créneau ;
- le calendrier (fériés, vacances, changements d'heure), connu à l'avance ;
- une prévision de température pour T (ou, à défaut, l'observation utilisée
  comme approximation : c'est alors une limite assumée, voir README).

Toutes les entrées sont des matrices jours locaux × 48 créneaux, indexées par
des jours consécutifs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_fr import day_type

SLOTS = 48
CUTOFF_SLOT = 24  # créneaux 0..23 de D connus à midi
TEMP_SMOOTH_HALFLIFE = 2.0  # jours, inertie thermique des bâtiments

CALENDAR_FEATURES = [
    "dow", "doy", "month", "is_holiday", "holiday_id", "is_bridge", "is_off",
    "year_end", "school_zones", "dst_change", "summer_time",
]
LAG_DAY_FEATURES = ["daytype_d", "daytype_dm1", "daytype_dm6", "bridge_d", "bridge_dm1"]
LOAD_FEATURES = [
    "load_d_same", "load_d_last", "load_d_morning", "load_dm1_morning", "load_dm1_same", "load_dm1_mean",
    "load_dm6_same", "load_dm6_mean", "load_week_mean", "morning_ratio",
]
TEMP_FEATURES = [
    "temp", "temp_mean", "temp_min", "temp_max", "temp_dm1_mean", "temp_d_morning",
    "temp_smooth", "temp_delta",
]
FEATURES = ["slot", "hour"] + CALENDAR_FEATURES + LAG_DAY_FEATURES + LOAD_FEATURES + TEMP_FEATURES
CATEGORICAL = ["holiday_id", "daytype_d", "daytype_dm1", "daytype_dm6"]


def _check_daily_index(idx: pd.DatetimeIndex) -> None:
    if len(idx) > 1 and not (np.diff(idx.values) == np.timedelta64(1, "D")).all():
        raise ValueError("les matrices doivent couvrir des jours consécutifs")


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """Décale les lignes de k jours vers le bas (la ligne T reçoit la ligne T-k)."""
    out = np.full_like(a, np.nan, dtype=float)
    if k < len(a):
        out[k:] = a[: len(a) - k]
    return out


def _shift1d(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if k < len(a):
        out[k:] = a[: len(a) - k]
    return out


def build_features(load: pd.DataFrame, temp_obs: pd.DataFrame, temp_target: pd.DataFrame,
                   cal: pd.DataFrame) -> pd.DataFrame:
    """Renvoie un tableau long indexé par (date cible, créneau).

    `load` et `temp_obs` ne sont lus que jusqu'à la coupure de D.
    `temp_target` n'est lu que pour le jour cible T (prévision météo).
    La colonne `y` contient la valeur à prévoir et n'est pas une variable.
    """
    idx = load.index
    _check_daily_index(idx)
    L = load.reindex(idx).to_numpy(dtype=float)
    To = temp_obs.reindex(idx).to_numpy(dtype=float)
    Tt = temp_target.reindex(idx).to_numpy(dtype=float)
    c = cal.reindex(idx)
    n = len(idx)

    # Jour D (veille de la cible) : seule la matinée est connue.
    L_d = _shift(L, 1)[:, :CUTOFF_SLOT]
    L_dm1 = _shift(L, 2)
    L_dm6 = _shift(L, 7)
    To_d = _shift(To, 1)[:, :CUTOFF_SLOT]
    To_dm1 = _shift(To, 2)

    with np.errstate(invalid="ignore"), _quiet_nanmean():
        # Moyenne des six journées complètes D-6 .. D-1 (toutes connues à la coupure).
        daily_load = np.nanmean(L, axis=1)
        week = np.vstack([_shift1d(daily_load, k) for k in range(2, 8)])
        load_week_mean = np.nanmean(week, axis=0)

        day = {
            "load_d_last": L_d[:, CUTOFF_SLOT - 1],
            "load_d_morning": np.nanmean(L_d, axis=1),
            "load_dm1_morning": np.nanmean(L_dm1[:, :CUTOFF_SLOT], axis=1),
            "load_dm1_mean": np.nanmean(L_dm1, axis=1),
            "load_dm6_mean": np.nanmean(L_dm6, axis=1),
            "load_week_mean": load_week_mean,
            "temp_mean": np.nanmean(Tt, axis=1),
            "temp_min": np.nanmin(Tt, axis=1),
            "temp_max": np.nanmax(Tt, axis=1),
            "temp_dm1_mean": np.nanmean(To_dm1, axis=1),
            "temp_d_morning": np.nanmean(To_d, axis=1),
        }
        daily_temp = pd.Series(np.nanmean(To, axis=1))
    smooth = daily_temp.ewm(halflife=TEMP_SMOOTH_HALFLIFE, adjust=False, ignore_na=True).mean()
    day["temp_smooth"] = _shift1d(smooth.to_numpy(), 2)  # jours <= D-1
    day["morning_ratio"] = day["load_d_morning"] / day["load_dm1_morning"]
    day["temp_delta"] = day["temp_mean"] - day["temp_dm1_mean"]

    for col in CALENDAR_FEATURES:
        day[col] = c[col].to_numpy()
    dtype = day_type(c).to_numpy().astype(float)
    bridge = c["is_bridge"].to_numpy().astype(float)
    day["daytype_d"] = _shift1d(dtype, 1)
    day["daytype_dm1"] = _shift1d(dtype, 2)
    day["daytype_dm6"] = _shift1d(dtype, 7)
    day["bridge_d"] = _shift1d(bridge, 1)
    day["bridge_dm1"] = _shift1d(bridge, 2)

    out = pd.DataFrame({k: np.repeat(np.asarray(v, dtype=float), SLOTS) for k, v in day.items()})
    slots = np.tile(np.arange(SLOTS), n)
    out["slot"] = slots
    out["hour"] = slots // 2
    # Même demi-heure la veille : connue seulement pour les créneaux du matin.
    L_d_same = np.full((n, SLOTS), np.nan)
    L_d_same[:, :CUTOFF_SLOT] = L_d
    out["load_d_same"] = L_d_same.reshape(-1)
    out["load_dm1_same"] = L_dm1.reshape(-1)
    out["load_dm6_same"] = L_dm6.reshape(-1)
    out["temp"] = Tt.reshape(-1)
    out["y"] = L.reshape(-1)
    out.index = pd.MultiIndex.from_arrays([np.repeat(idx.values, SLOTS), slots], names=["date", "slot"])
    return out[FEATURES + ["y"]]


class _quiet_nanmean:
    """Masque les avertissements « Mean of empty slice » de numpy."""

    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
