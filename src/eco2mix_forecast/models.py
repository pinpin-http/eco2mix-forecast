"""Modèles : deux références naïves et un gradient boosting LightGBM."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .calendar_fr import day_type
from .features import CATEGORICAL, FEATURES


def persistence_j7(feat: pd.DataFrame) -> pd.Series:
    """Même demi-heure, sept jours plus tôt (connue à la coupure de D)."""
    return feat["load_dm6_same"].rename("persistance_j7")


class SeasonalMean:
    """Moyenne historique par (semaine ISO, type de jour, créneau) sur l'entraînement."""

    def fit(self, feat: pd.DataFrame, cal: pd.DataFrame) -> "SeasonalMean":
        week, dtype, slot = self._keys(feat.index, cal)
        y = feat["y"].to_numpy()
        self.table_ = pd.Series(y).groupby([week, dtype, slot]).mean()
        self.fallback_ = pd.Series(y).groupby([dtype, slot]).mean()
        return self

    @staticmethod
    def _keys(index: pd.MultiIndex, cal: pd.DataFrame) -> list[np.ndarray]:
        dates = pd.DatetimeIndex(index.get_level_values("date"))
        week = np.minimum(dates.isocalendar().week.to_numpy(dtype=int), 52)
        dtype = day_type(cal).reindex(dates).to_numpy(dtype=int)
        slot = index.get_level_values("slot").to_numpy(dtype=int)
        return [week, dtype, slot]

    def predict(self, feat: pd.DataFrame, cal: pd.DataFrame) -> pd.Series:
        week, dtype, slot = self._keys(feat.index, cal)
        mi = pd.MultiIndex.from_arrays([week, dtype, slot])
        pred = self.table_.reindex(mi).to_numpy(dtype=float, copy=True)
        miss = np.isnan(pred)
        if miss.any():
            fb = pd.MultiIndex.from_arrays([dtype, slot])
            pred[miss] = self.fallback_.reindex(fb).to_numpy(dtype=float)[miss]
        return pd.Series(pred, index=feat.index, name="moyenne_saisonniere")


DEFAULT_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
    "num_boost_round": 1500,
    "verbosity": -1,
}


class LGBMForecaster:
    """LightGBM sur les variables J+1.

    `target="ratio"` : on apprend y / load_week_mean (consommation rapportée au
    niveau moyen des six derniers jours connus), ce qui rend le modèle moins
    sensible aux changements de niveau d'une année à l'autre.
    `target="absolute"` : on apprend directement la consommation en MW.
    """

    def __init__(self, target: str = "ratio", params: dict | None = None, seed: int = 0):
        if target not in ("ratio", "absolute"):
            raise ValueError(target)
        self.target = target
        self.params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed,
                       "num_threads": config.N_JOBS, "deterministic": True}

    def _y(self, feat):
        return feat["y"] / feat["load_week_mean"] if self.target == "ratio" else feat["y"]

    def fit(self, feat: pd.DataFrame) -> "LGBMForecaster":
        d = feat.dropna(subset=["y", "load_week_mean"])
        params = dict(self.params)
        rounds = params.pop("num_boost_round")
        data = lgb.Dataset(d[FEATURES], label=self._y(d), categorical_feature=CATEGORICAL,
                           free_raw_data=True)
        self.booster_ = lgb.train(params, data, num_boost_round=rounds)
        return self

    def predict(self, feat: pd.DataFrame) -> pd.Series:
        raw = self.booster_.predict(feat[FEATURES])
        if self.target == "ratio":
            raw = raw * feat["load_week_mean"].to_numpy()
        return pd.Series(raw, index=feat.index, name="lightgbm")

    def shap_values(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Contributions TreeSHAP calculées nativement par LightGBM (pred_contrib).

        Avec la cible « ratio », elles sont exprimées en unités du ratio.
        """
        contrib = self.booster_.predict(feat[FEATURES], pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=feat.index)
