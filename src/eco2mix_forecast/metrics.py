"""Métriques d'erreur : MAPE (%), RMSE et MAE (MW), globales ou par groupe."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASONS = {12: "hiver", 1: "hiver", 2: "hiver", 3: "printemps", 4: "printemps", 5: "printemps",
           6: "été", 7: "été", 8: "été", 9: "automne", 10: "automne", 11: "automne"}
SEASON_ORDER = ["hiver", "printemps", "été", "automne"]


def _clean(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    return y[ok], p[ok]


def mape(y, p) -> float:
    y, p = _clean(y, p)
    if np.any(y <= 0):
        raise ValueError("MAPE indéfini pour des valeurs réelles nulles ou négatives")
    return float(np.mean(np.abs(y - p) / y) * 100) if len(y) else np.nan


def rmse(y, p) -> float:
    y, p = _clean(y, p)
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else np.nan


def mae(y, p) -> float:
    y, p = _clean(y, p)
    return float(np.mean(np.abs(y - p))) if len(y) else np.nan


def bias(y, p) -> float:
    """Erreur moyenne prévision - réel (MW) ; positive = surestimation."""
    y, p = _clean(y, p)
    return float(np.mean(p - y)) if len(y) else np.nan


def scores(y, p) -> dict[str, float]:
    return {"mape": mape(y, p), "rmse": rmse(y, p), "mae": mae(y, p), "bias": bias(y, p)}


def score_table(df: pd.DataFrame, models: list[str], by: str | None = None,
                target: str = "y") -> pd.DataFrame:
    """Une ligne par (groupe, modèle). `df` contient la cible et une colonne par modèle.

    Les lignes où une des prévisions manque sont retirées pour tous les modèles,
    afin que chaque modèle soit évalué exactement sur les mêmes créneaux.
    """
    d = df.dropna(subset=[target] + models)
    groups = [("all", d)] if by is None else list(d.groupby(by, sort=True))
    rows = []
    for key, g in groups:
        for m in models:
            rows.append({"group": key, "model": m, "n": len(g), **scores(g[target], g[m])})
    return pd.DataFrame(rows)


def season_of(dates) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(dates).month).map(SEASONS).to_numpy()
