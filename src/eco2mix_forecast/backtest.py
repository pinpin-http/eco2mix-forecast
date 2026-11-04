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

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, dataset
from .features import CUTOFF_SLOT, FEATURES, build_features
from .metrics import score_table, season_of
from .models import DEFAULT_PARAMS, LGBMForecaster, SeasonalMean, persistence_j7

log = logging.getLogger(__name__)

FIRST_TRAIN_YEAR = 2013
TARGET_MODES = ("ratio", "absolute")
RTE_COLUMNS = ["rte_j1", "rte_j1_debiaisee", "rte_j1_profil"]


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
    """Prévision J-1 de RTE, brute et corrigée de son écart moyen de l'année précédente.

    Les versions corrigées ne sont pas des produits de RTE : ce sont des
    contrôles. Mesurée contre les données définitives, la prévision J-1
    sous-estime la consommation (d'environ 1 GW depuis 2023), avec un écart qui
    dépend de l'heure et du mois. Pour prévoir l'année Y, on retire :
    - `rte_j1_debiaisee` : l'écart moyen de l'année Y-1 (une constante) ;
    - `rte_j1_profil` : l'écart moyen de Y-1 par mois et par demi-heure.
    """
    rte = long(mats.rte_j1, "rte_j1")
    y = long(mats.load, "y")
    err = (rte - y).dropna()
    err_dates = pd.DatetimeIndex(err.index.get_level_values("date"))
    err_slots = err.index.get_level_values("slot")
    yearly = err.groupby(err_dates.year).mean()
    profile = err.groupby([err_dates.year, err_dates.month, err_slots]).mean()

    out = pd.DataFrame({"rte_j1": rte.reindex(index)})
    dates = pd.DatetimeIndex(index.get_level_values("date"))
    slots = index.get_level_values("slot")
    prev = dates.year - 1
    out["rte_j1_debiaisee"] = out["rte_j1"].to_numpy() - pd.Series(prev).map(yearly).to_numpy()
    keys = pd.MultiIndex.from_arrays([prev, dates.month, slots])
    out["rte_j1_profil"] = out["rte_j1"].to_numpy() - profile.reindex(keys).to_numpy()
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


# --- Cache des plis ---------------------------------------------------------------
# Un pli déjà calculé avec les mêmes variables et les mêmes réglages est relu
# depuis data/processed/cache/ : une coupure en cours de backtest ne fait perdre
# que le pli en cours.

def _cache_key(targets) -> str:
    blob = json.dumps({"features": FEATURES, "params": DEFAULT_PARAMS, "targets": list(targets),
                       "last_year": config.LAST_YEAR}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _fsync(path: Path) -> None:
    with open(path, "rb+") as f:
        os.fsync(f.fileno())


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    df.to_parquet(tmp)
    _fsync(tmp)
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text)
    _fsync(tmp)
    tmp.replace(path)


def _write_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    _write_text(path, df.to_csv(**kw))


def cached_fold(name: str, fold: Fold, compute, targets):
    path = config.PROCESSED_DIR / "cache" / f"{name}_{fold.name}_{_cache_key(targets)}.parquet"
    if path.exists():
        log.info("pli %s relu depuis le cache", fold.name)
        return pd.read_parquet(path)
    preds = compute()
    _write_parquet(preds, path)
    return preds


def feature_importance(model: LGBMForecaster, feat: pd.DataFrame, n: int = 8000) -> pd.DataFrame:
    """Part moyenne de chaque variable dans les contributions SHAP, et gain LightGBM."""
    sample = feat.dropna(subset=["y"])
    sample = sample.sample(n=min(len(sample), n), random_state=0)
    shap = model.shap_values(sample).abs().mean().sort_values(ascending=False)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
    imp = pd.DataFrame({"mean_abs_shap": shap, "gain": gain.reindex(shap.index)})
    imp["shap_share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp.index.name = "feature"
    return imp


def cached_test(name: str, fold: Fold, compute, targets):
    """Comme `cached_fold`, pour le test : prévisions et importance des variables."""
    base = config.PROCESSED_DIR / "cache" / f"{name}_{fold.name}_{_cache_key(targets)}"
    pred_path, imp_path = Path(f"{base}.parquet"), Path(f"{base}_importance.parquet")
    if pred_path.exists() and imp_path.exists():
        log.info("test %s relu depuis le cache", fold.name)
        return pd.read_parquet(pred_path), pd.read_parquet(imp_path)
    preds, imp = compute()
    _write_parquet(imp, imp_path)
    _write_parquet(preds, pred_path)
    return preds, imp


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
        p = cached_fold(name, fold, lambda f=fold: run_fold(feat, mats.cal, f, targets=targets)[0],
                        targets)
        p["fold"] = fold.name
        bt.append(p)
    bt = _add_groups(pd.concat(bt))
    if has_rte:
        bt = bt.join(rte_columns(mats, bt.index))
    bt_models = ["persistance_j7", "moyenne_saisonniere"] + [f"lightgbm_{t}" for t in targets]
    if has_rte:
        bt_models += RTE_COLUMNS
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

    def compute_test():
        preds, models = run_fold(feat, mats.cal, test_fold, targets=(best,), test_feats=tf)
        return preds, feature_importance(models[best], tf["fc_d2"].loc[preds.index])

    test_p, imp = cached_test(name, test_fold, compute_test, (best, "test"))
    test_p = _add_groups(test_p)
    if has_rte:
        test_p = test_p.join(rte_columns(mats, test_p.index))

    test_models = ["persistance_j7", "moyenne_saisonniere", "lightgbm_fc_d2", "lightgbm_fc_d1",
                   "lightgbm_obs"] + (RTE_COLUMNS if has_rte else [])
    out_dir = config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if name == "national" else f"{name}_"
    _write_csv(by_year, out_dir / f"{prefix}backtest_by_year.csv", index=False, float_format="%.4f")
    _write_csv(score_table(bt, bt_models), out_dir / f"{prefix}backtest_overall.csv", index=False,
               float_format="%.4f")
    test_all = score_table(test_p, test_models)
    _write_csv(test_all, out_dir / f"{prefix}test_overall.csv", index=False, float_format="%.4f")
    _write_csv(score_table(test_p, test_models, by="season"), out_dir / f"{prefix}test_by_season.csv",
               index=False, float_format="%.4f")
    _write_csv(score_table(test_p, test_models, by="hour"), out_dir / f"{prefix}test_by_hour.csv",
               index=False, float_format="%.4f")

    _write_csv(imp, out_dir / f"{prefix}feature_importance.csv", float_format="%.6f")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _write_parquet(bt, config.PROCESSED_DIR / f"{prefix}pred_backtest.parquet")
    _write_parquet(test_p, config.PROCESSED_DIR / f"{prefix}pred_test.parquet")

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
            for r in RTE_COLUMNS
        }
    _write_text(out_dir / f"{prefix}summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("test %s : %s", name, test_all[["model", "mape", "rmse"]].round(3).to_dict("records"))
    return summary


def run_all() -> None:
    national = run_region("national", has_rte=True)
    # Pour la région, on reprend la cible choisie sur le backtest national
    # (moitié moins d'entraînements).
    run_region("nouvelle_aquitaine", has_rte=False, targets=(national["target_mode"],))
