"""Figures PNG (matplotlib, backend Agg) écrites dans docs/."""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config, dataset  # noqa: E402

log = logging.getLogger(__name__)

# Palette catégorielle validée (ordre fixe) et encre neutre pour le réel.
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

LABELS = {
    "y": "Réalisé",
    "lightgbm_fc_d2": "LightGBM (météo prévue à 48 h)",
    "lightgbm_fc_d1": "LightGBM (météo prévue à 24 h)",
    "lightgbm_obs": "LightGBM (météo observée)",
    "rte_j1": "RTE, prévision J-1",
    "rte_j1_debiaisee": "RTE J-1 corrigée du biais (contrôle)",
    "persistance_j7": "Persistance J-7",
    "moyenne_saisonniere": "Moyenne saisonnière",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True,
    })


def _save(fig, name: str) -> None:
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DOCS_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("figure %s", path)


def _to_time(p: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(p.index.get_level_values("date"))
    slots = p.index.get_level_values("slot").to_numpy()
    return dates + pd.to_timedelta(slots * 30, unit="min")


def fig_week(test: pd.DataFrame) -> None:
    """Semaine (lundi-dimanche) de 2025 contenant le pic de consommation."""
    daily = test["y"].groupby(level="date").mean()
    peak_day = daily.idxmax()
    monday = peak_day - pd.Timedelta(days=peak_day.dayofweek)
    w = test.loc[monday:monday + pd.Timedelta(days=6)]
    t = _to_time(w)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(t, w["y"] / 1000, color=INK, lw=2.4, label=LABELS["y"])
    ax.plot(t, w["rte_j1"] / 1000, color=ORANGE, lw=1.8, label=LABELS["rte_j1"])
    ax.plot(t, w["lightgbm_fc_d2"] / 1000, color=BLUE, lw=1.8, label=LABELS["lightgbm_fc_d2"])
    ax.plot(t, w["persistance_j7"] / 1000, color=AQUA, lw=1.4, label=LABELS["persistance_j7"])
    ax.set_ylabel("GW")
    ax.set_title(f"Semaine du {monday:%d/%m/%Y} : réalisé et prévisions J+1")
    ax.legend(loc="upper left", ncol=4, fontsize=9, bbox_to_anchor=(0, -0.12))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    _save(fig, f"semaine_pointe_{config.TEST_YEAR}.png")


def fig_error_by_hour(by_hour: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    series = [("lightgbm_fc_d2", BLUE, "-"), ("lightgbm_obs", AQUA, "--"),
              ("rte_j1", ORANGE, "-"), ("rte_j1_debiaisee", YELLOW, "--")]
    for m, color, ls in series:
        d = by_hour[by_hour["model"] == m].sort_values("group")
        ax.plot(d["group"], d["mape"], color=color, ls=ls, label=LABELS[m], marker="o", ms=4)
    ax.set_xlabel("Heure de la journée (heure de Paris)")
    ax.set_ylabel("MAPE (%)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Erreur moyenne par heure, année de test {config.TEST_YEAR}")
    ax.legend(loc="upper left", ncol=2, fontsize=9, bbox_to_anchor=(0, -0.12))
    _save(fig, "erreur_par_heure.png")


FEATURE_LABELS = {
    "load_week_mean": "conso. moyenne des 6 derniers jours",
    "load_d_morning": "conso. matin de D (0h-11h30)",
    "load_d_last": "conso. D à 11h30",
    "load_d_same": "conso. D, même demi-heure (matin)",
    "load_dm1_same": "conso. D-1, même demi-heure",
    "load_dm1_mean": "conso. moyenne de D-1",
    "load_dm1_morning": "conso. matin de D-1",
    "load_dm6_same": "conso. J-7, même demi-heure",
    "load_dm6_mean": "conso. moyenne de J-7",
    "morning_ratio": "ratio matin D / matin D-1",
    "temp": "température prévue (demi-heure)",
    "temp_mean": "température moyenne prévue",
    "temp_min": "température min prévue",
    "temp_max": "température max prévue",
    "temp_dm1_mean": "température moyenne de D-1",
    "temp_d_morning": "température matin de D",
    "temp_smooth": "température lissée (inertie)",
    "temp_delta": "écart de température cible / D-1",
    "slot": "demi-heure", "hour": "heure", "dow": "jour de la semaine", "doy": "jour de l'année",
    "month": "mois", "is_holiday": "férié", "holiday_id": "quel férié", "is_bridge": "pont",
    "is_off": "jour non ouvré", "year_end": "fin d'année", "school_zones": "zones en vacances",
    "dst_change": "changement d'heure", "summer_time": "heure d'été",
    "daytype_d": "type de jour de D", "daytype_dm1": "type de jour de D-1",
    "daytype_dm6": "type de jour de J-7", "bridge_d": "D est un pont", "bridge_dm1": "D-1 est un pont",
}


def fig_importance(imp: pd.DataFrame, top: int = 15) -> None:
    d = imp.sort_values("shap_share", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    labels = [FEATURE_LABELS.get(i, i) for i in d.index]
    ax.barh(labels, d["shap_share"] * 100, color=BLUE, height=0.6)
    for y, v in enumerate(d["shap_share"] * 100):
        ax.text(v + 0.3, y, f"{v:.1f} %".replace(".", ","), va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Part de la contribution absolue moyenne (SHAP), en %")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Variables les plus utilisées par LightGBM ({config.TEST_YEAR})")
    _save(fig, "importance_variables.png")


def fig_temperature(df: pd.DataFrame, cal: pd.DataFrame) -> None:
    """Courbe en V : consommation journalière moyenne contre température moyenne."""
    load = dataset.to_daily_matrix(df["load"]).mean(axis=1)
    temp = dataset.to_daily_matrix(df["temp_obs"]).mean(axis=1)
    d = pd.DataFrame({"load": load / 1000, "temp": temp}).join(cal[["is_off", "year_end"]], how="inner")
    d = d[~d["is_off"].astype(bool) & ~d["year_end"].astype(bool)].dropna()
    d["year"] = d.index.year
    periods = [("2014-2019", 2014, 2019, BLUE), ("2023-2025", 2023, 2025, ORANGE)]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for label, a, b, color in periods:
        s = d[(d["year"] >= a) & (d["year"] <= b)]
        ax.scatter(s["temp"], s["load"], s=7, color=color, alpha=0.25, linewidths=0)
        bins = np.arange(-6, 31, 1.0)
        g = s.groupby(pd.cut(s["temp"], bins), observed=True)["load"].median()
        centers = [iv.mid for iv in g.index]
        ax.plot(centers, g.to_numpy(), color=color, lw=2.2, label=f"{label} (médiane par degré)")
    ax.set_xlabel("Température moyenne journalière pondérée par la population (°C, ERA5)")
    ax.set_ylabel("Consommation moyenne journalière (GW)")
    ax.set_title("Jours ouvrés : effet de la température sur la consommation")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "effet_temperature.png")


def fig_backtest(by_year: pd.DataFrame, best: str) -> None:
    models = [(f"lightgbm_{best}", BLUE, "LightGBM (météo observée)"),
              ("rte_j1", ORANGE, LABELS["rte_j1"]),
              ("rte_j1_debiaisee", YELLOW, LABELS["rte_j1_debiaisee"])]
    years = sorted(by_year["group"].unique())
    x = np.arange(len(years))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (m, color, label) in enumerate(models):
        d = by_year[by_year["model"] == m].set_index("group").reindex(years)
        ax.bar(x + (i - 1) * (width + 0.02), d["mape"], width=width, color=color, label=label)
    ax.set_xticks(x, [str(y) for y in years])
    ax.set_ylabel("MAPE (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Backtest annuel, fenêtre croissante (LightGBM avec la météo observée)")
    ax.legend(loc="upper left", ncol=3, fontsize=9, bbox_to_anchor=(0, -0.08))
    _save(fig, "backtest_par_annee.png")


def make_all() -> None:
    _style()
    res = config.RESULTS_DIR
    test = pd.read_parquet(config.PROCESSED_DIR / "pred_test.parquet")
    fig_week(test)
    fig_error_by_hour(pd.read_csv(res / "test_by_hour.csv"))
    fig_importance(pd.read_csv(res / "feature_importance.csv", index_col=0))
    df, cal = dataset.load_processed("national")
    fig_temperature(df, cal)
    summary = json.loads((res / "summary.json").read_text())
    fig_backtest(pd.read_csv(res / "backtest_by_year.csv"), summary["target_mode"])
    write_tables()


# --- Tableaux Markdown (repris tels quels dans le README) ----------------------

def _fr(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def _pivot(df: pd.DataFrame, models: list[str], metric: str, nd: int,
           labels: dict | None = None) -> str:
    labels = {**LABELS, **(labels or {})}
    d = df[df["model"].isin(models)].pivot(index="group", columns="model", values=metric)
    d = d[models]
    head = "| | " + " | ".join(labels.get(m, m) for m in models) + " |"
    sep = "|---" * (len(models) + 1) + "|"
    rows = [f"| {g} | " + " | ".join(_fr(v, nd) for v in r) + " |" for g, r in d.iterrows()]
    return "\n".join([head, sep] + rows)


def write_tables() -> None:
    res = config.RESULTS_DIR
    out = []
    for prefix, title, rte in (("", "France", True), ("nouvelle_aquitaine_", "Nouvelle-Aquitaine", False)):
        if not (res / f"{prefix}test_overall.csv").exists():
            continue
        summary = json.loads((res / f"{prefix}summary.json").read_text())
        best = f"lightgbm_{summary['target_mode']}"
        test = pd.read_csv(res / f"{prefix}test_overall.csv")
        out.append(f"## {title}, test {config.TEST_YEAR}\n")
        out.append("| Modèle | MAPE (%) | RMSE (MW) | MAE (MW) | Biais (MW) |\n|---|---|---|---|---|")
        for _, r in test.sort_values("mape").iterrows():
            out.append(f"| {LABELS.get(r['model'], r['model'])} | {_fr(r['mape'])} | {_fr(r['rmse'], 0)} | "
                       f"{_fr(r['mae'], 0)} | {_fr(r['bias'], 0)} |")
        season = pd.read_csv(res / f"{prefix}test_by_season.csv")
        season["group"] = pd.Categorical(season["group"], ["hiver", "printemps", "été", "automne"])
        season = season.sort_values("group")
        models = ["lightgbm_fc_d2", "lightgbm_obs"] + (["rte_j1", "rte_j1_debiaisee"] if rte else []) \
            + ["persistance_j7"]
        out.append(f"\n### MAPE (%) par saison, {title}\n")
        out.append(_pivot(season, models, "mape", 2))
        by_year = pd.read_csv(res / f"{prefix}backtest_by_year.csv")
        models = [best] + (["rte_j1", "rte_j1_debiaisee"] if rte else []) \
            + ["persistance_j7", "moyenne_saisonniere"]
        alt = "lightgbm_absolute" if best == "lightgbm_ratio" else "lightgbm_ratio"
        names = {"lightgbm_absolute": "LightGBM, cible en MW", "lightgbm_ratio": "LightGBM, cible ratio"}
        out.append(f"\n### Backtest, MAPE (%) par année, {title} (température observée)\n")
        cols = models[:1] + ([alt] if alt in set(by_year["model"]) else []) + models[1:]
        out.append(_pivot(by_year, cols, "mape", 2, names))
        out.append("")
    path = res / "tables.md"
    path.write_text("\n".join(out) + "\n")
    log.info("tableaux %s", path)
