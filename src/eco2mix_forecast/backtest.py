"""Backtest annuel à fenêtre croissante, test final sur l'année tenue à l'écart.

Déroulé :
1. backtest : pour chaque année Y de BACKTEST_YEARS, entraînement sur
   [2013, Y-1] et prévision de Y, avec la température observée (ERA5) comme
   approximation de la prévision météo (seule option avant 2024) ;
2. choix de la cible du modèle (ratio ou MW) sur la moyenne du backtest ;
3. test : entraînement sur [2013, 2024], prévision de 2025, avec trois sources
   de température pour le jour cible : prévision émise 48 h avant (strictement
   disponible à midi la veille), prévision émise 24 h avant, et observation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, dataset
from .features import CUTOFF_SLOT, build_features
from .metrics import score_table, season_of
from .models import LGBMForecaster, SeasonalMean, persistence_j7

log = logging.getLogger(__name__)

FIRST_TRAIN_YEAR = 2013
TARGET_MODES = ("ratio", "absolute")


@dataclass
class Fold:
    name: str
    train_years: tuple[int, int]
    test_year: int


def yearly_folds(backtest_years=config.BACKTEST_YEARS, test_year=config.TEST_YEAR,
                 first_train_year=FIRST_TRAIN_YEAR) -> tuple[list[Fold], Fold]:
    """Plis annuels à fenêtre croissante ; l'année de test n'apparaît dans aucun pli."""
    if any(y >= test_year for y in backtest_years):
        raise ValueError("une année de backtest recouvre l'année de test")
    folds = [Fold(str(y), (first_train_year, y - 1), y) for y in sorted(backtest_years)]
    test = Fold(f"test_{test_year}", (first_train_year, test_year - 1), test_year)
    return folds, test


def fold_masks(dates: pd.Index, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    years = pd.DatetimeIndex(dates).year
    train = (years >= fold.train_years[0]) & (years <= fold.train_years[1])
    test = years == fold.test_year
    return np.asarray(train), np.asarray(test)


# --- Données -------------------------------------------------------------------

@dataclass
class Matrices:
    load: pd.DataFrame
    temp_obs: pd.DataFrame
    temp_fc_d1: pd.DataFrame
    temp_fc_d2: pd.DataFrame
    rte_j1: pd.DataFrame | None
    cal: pd.DataFrame


def load_matrices(name: str) -> Matrices:
    df, cal = dataset.load_processed(name)
    m = {c: dataset.to_daily_matrix(df[c]) for c in df.columns}
    idx = m["load"].index
    # La dernière heure UTC de l'année tombe le 1er janvier en heure de Paris.
    idx = idx[idx <= pd.Timestamp(f"{config.LAST_YEAR}-12-31")]
    m = {k: v.reindex(idx) for k, v in m.items()}
    return Matrices(m["load"], m["temp_obs"], m["temp_fc_d1"], m["temp_fc_d2"],
                    m.get("rte_j1"), cal.reindex(idx))


def forecast_bias(obs: pd.DataFrame, fc: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Écart moyen (observé - prévu) par créneau sur une période passée."""
    d = (obs - fc).loc[start:end]
    return d.mean(axis=0)


def long(m: pd.DataFrame, name: str) -> pd.Series:
    return dataset.from_daily_matrix(m).rename(name)


def rte_columns(mats: Matrices, index: pd.MultiIndex) -> pd.DataFrame:
    """Prévision J-1 de RTE, brute et corrigée de son biais moyen de l'année précédente.

    La version corrigée n'est pas un produit de RTE : c'est un contrôle. Mesurée
    contre les données définitives, la prévision J-1 sous-estime en moyenne la
    consommation (d'environ 1 GW depuis 2023) ; on vérifie ce que donnerait une
    simple correction de ce biais, estimée sur l'année Y-1 pour prévoir Y.
    """
    rte = long(mats.rte_j1, "rte_j1")
    y = long(mats.load, "y")
    err = (rte - y).dropna()
    yearly_bias = err.groupby(pd.DatetimeIndex(err.index.get_level_values("date")).year).mean()
    out = pd.DataFrame({"rte_j1": rte.reindex(index)})
    years = pd.DatetimeIndex(index.get_level_values("date")).year
    prev_bias = pd.Series(years - 1).map(yearly_bias).to_numpy()
    out["rte_j1_debiaisee"] = out["rte_j1"].to_numpy() - prev_bias
    return out


# --- Exécution --------------------------------------------------------------------

def run_fold(feat: pd.DataFrame, cal: pd.DataFrame, fold: Fold, targets=TARGET_MODES,
             test_feats: dict[str, pd.DataFrame] | None = None):
    """Entraîne sur le pli et prédit l'année de test.

    `test_feats` permet de prédire la même année avec d'autres variables (par
    exemple une autre source de température) sans réentraîner.
    """
    dates = feat.index.get_level_values("date")
    tr, te = fold_masks(dates, fold)
    train = feat[tr].dropna(subset=["y"])
    test = feat[te]
    preds = pd.DataFrame({"y": test["y"]})
    preds["persistance_j7"] = persistence_j7(test)
    preds["moyenne_saisonniere"] = SeasonalMean().fit(train, cal).predict(test, cal)
    models = {}
    for t in targets:
        model = LGBMForecaster(target=t).fit(train)
        models[t] = model
        if test_feats is None:
            preds[f"lightgbm_{t}"] = model.predict(test)
        else:
            for label, f in test_feats.items():
                preds[f"lightgbm_{label}"] = model.predict(f.loc[test.index])
    return preds, models


def _add_groups(p: pd.DataFrame) -> pd.DataFrame:
    dates = p.index.get_level_values("date")
    p = p.copy()
    p["season"] = season_of(dates)
    p["hour"] = p.index.get_level_values("slot") // 2
    p["year"] = pd.DatetimeIndex(dates).year
    return p


def daily_win_rate(p: pd.DataFrame, a: str, b: str) -> float:
    """Part des jours où le modèle `a` a un MAPE journalier plus faible que `b`."""
    d = p.dropna(subset=["y", a, b])
    err = d[[a, b]].sub(d["y"], axis=0).abs().div(d["y"], axis=0)
    daily = err.groupby(level="date").mean()
    return float((daily[a] < daily[b]).mean())


def run_region(name: str, has_rte: bool, targets=TARGET_MODES) -> dict:
    log.info("=== %s ===", name)
    mats = load_matrices(name)
    feat = build_features(mats.load, mats.temp_obs, mats.temp_obs, mats.cal)
    folds, test_fold = yearly_folds()

    # 1. Backtest annuel (température observée comme approximation)
    bt = []
    for fold in folds:
        log.info("pli %s : entraînement %s-%s", fold.name, *fold.train_years)
        p, _ = run_fold(feat, mats.cal, fold, targets=targets)
        p["fold"] = fold.name
        bt.append(p)
    bt = _add_groups(pd.concat(bt))
    if has_rte:
        bt = bt.join(rte_columns(mats, bt.index))
    bt_models = ["persistance_j7", "moyenne_saisonniere"] + [f"lightgbm_{t}" for t in targets]
    if has_rte:
        bt_models += ["rte_j1", "rte_j1_debiaisee"]
    by_year = score_table(bt, bt_models, by="year")
    mean_mape = by_year.groupby("model")["mape"].mean()
    best = min(targets, key=lambda t: mean_mape[f"lightgbm_{t}"])
    log.info("MAPE moyen du backtest : %s ; cible retenue : %s", mean_mape.round(3).to_dict(), best)

    # 2. Test final : prévision météo corrigée du biais moyen observé sur 2024
    bias_start, bias_end = config.FORECAST_START, f"{config.TEST_YEAR - 1}-12-31"
    tf = {}
    for label, fc in (("fc_d2", mats.temp_fc_d2), ("fc_d1", mats.temp_fc_d1)):
        b = forecast_bias(mats.temp_obs, fc, bias_start, bias_end)
        tf[label] = build_features(mats.load, mats.temp_obs, fc + b, mats.cal)
    tf["obs"] = feat
    test_p, models = run_fold(feat, mats.cal, test_fold, targets=(best,), test_feats=tf)
    test_p = _add_groups(test_p)
    if has_rte:
        test_p = test_p.join(rte_columns(mats, test_p.index))

    test_models = ["persistance_j7", "moyenne_saisonniere", "lightgbm_fc_d2", "lightgbm_fc_d1",
                   "lightgbm_obs"] + (["rte_j1", "rte_j1_debiaisee"] if has_rte else [])
    out_dir = config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if name == "national" else f"{name}_"
    by_year.to_csv(out_dir / f"{prefix}backtest_by_year.csv", index=False, float_format="%.4f")
    score_table(bt, bt_models).to_csv(out_dir / f"{prefix}backtest_overall.csv", index=False,
                                      float_format="%.4f")
    test_all = score_table(test_p, test_models)
    test_all.to_csv(out_dir / f"{prefix}test_overall.csv", index=False, float_format="%.4f")
    score_table(test_p, test_models, by="season").to_csv(out_dir / f"{prefix}test_by_season.csv",
                                                         index=False, float_format="%.4f")
    score_table(test_p, test_models, by="hour").to_csv(out_dir / f"{prefix}test_by_hour.csv",
                                                       index=False, float_format="%.4f")

    # Importance des variables : SHAP natif de LightGBM sur l'année de test
    model = models[best]
    sample = tf["fc_d2"].loc[test_p.index].dropna(subset=["y"])
    sample = sample.sample(n=min(len(sample), 8000), random_state=0)
    shap = model.shap_values(sample).abs().mean().sort_values(ascending=False)
    gain = pd.Series(model.booster_.feature_importance("gain"),
                     index=model.booster_.feature_name())
    imp = pd.DataFrame({"mean_abs_shap": shap, "gain": gain.reindex(shap.index)})
    imp["shap_share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp.to_csv(out_dir / f"{prefix}feature_importance.csv", float_format="%.6f")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(config.PROCESSED_DIR / f"{prefix}pred_backtest.parquet")
    test_p.to_parquet(config.PROCESSED_DIR / f"{prefix}pred_test.parquet")

    # Erreur des prévisions météo sur l'année de test (après correction du biais)
    y0, y1 = f"{config.TEST_YEAR}-01-01", f"{config.TEST_YEAR}-12-31"
    temp_err = {}
    for label, fc in (("fc_d2", mats.temp_fc_d2), ("fc_d1", mats.temp_fc_d1)):
        b = forecast_bias(mats.temp_obs, fc, bias_start, bias_end)
        e = ((fc + b) - mats.temp_obs).loc[y0:y1].to_numpy().ravel()
        e = e[np.isfinite(e)]
        temp_err[label] = {"mae_degC": float(np.mean(np.abs(e))), "bias_correction_mean_degC":
                           float(b.mean())}

    summary = {
        "region": name,
        "target_mode": best,
        "backtest_mean_mape": {k: float(v) for k, v in mean_mape.items()},
        "test": {r["model"]: {k: float(r[k]) for k in ("mape", "rmse", "mae", "bias")} | {"n": int(r["n"])}
                 for _, r in test_all.iterrows()},
        "temperature_forecast_error_test": temp_err,
        "cutoff": f"jour D, créneau {CUTOFF_SLOT} (12:00 heure de Paris)",
    }
    if has_rte:
        summary["daily_win_rate_vs_rte"] = {
            f"{m}_vs_{r}": daily_win_rate(test_p, m, r)
            for m in ("lightgbm_fc_d2", "lightgbm_fc_d1", "lightgbm_obs")
            for r in ("rte_j1", "rte_j1_debiaisee")
        }
    (out_dir / f"{prefix}summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("test %s : %s", name, test_all[["model", "mape", "rmse"]].round(3).to_dict("records"))
    return summary


def run_all() -> None:
    national = run_region("national", has_rte=True)
    # Pour la région, on reprend la cible choisie sur le backtest national
    # (moitié moins d'entraînements).
    run_region("nouvelle_aquitaine", has_rte=False, targets=(national["target_mode"],))
