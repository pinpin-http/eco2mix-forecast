"""Lecture des fichiers bruts et construction des tables de travail.

Représentation centrale : une matrice « jours locaux × 48 demi-heures ».
Les jours de changement d'heure n'ont pas 48 demi-heures en heure légale :
- fin mars (46), les deux créneaux absents (02:00 et 02:30) sont interpolés ;
- fin octobre (50), les deux passages par 02:00 et 02:30 sont moyennés.
On applique exactement le même traitement aux valeurs réalisées et à la
prévision de RTE, donc la comparaison reste équitable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .calendar_fr import daily_calendar, load_school_zones

log = logging.getLogger(__name__)

SLOTS = 48


# --- Lecture des fichiers bruts ---------------------------------------------

def read_eco2mix_csv(paths: list[Path], columns: dict[str, str]) -> pd.DataFrame:
    """Concatène les exports annuels et garde le pas de 30 minutes (index UTC)."""
    frames = [pd.read_csv(p, sep=";", encoding="utf-8-sig") for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    df = df[df["date_heure"].dt.minute.isin([0, 30])]
    df = df.drop_duplicates("date_heure").set_index("date_heure").sort_index()
    out = df[list(columns)].rename(columns=columns).astype(float)
    full = pd.date_range(out.index.min(), out.index.max(), freq="30min", tz="UTC")
    return out.reindex(full)


def read_open_meteo(path: Path, variables: list[str]) -> pd.DataFrame:
    raw = json.loads(path.read_text())
    h = raw["hourly"]
    idx = pd.to_datetime(h["time"]).tz_localize("UTC")
    return pd.DataFrame({v: np.asarray(h[v], dtype=float) for v in variables}, index=idx)


def population_weights(populations: dict, cities, by: str) -> dict[str, float]:
    """Poids de chaque ville = population de la zone qu'elle représente.

    `by="region"` : population de la région de la ville (agrégée sur ses départements) ;
    `by="na"` : population des départements de Nouvelle-Aquitaine rattachés à la ville.
    """
    if by == "region":
        w = {c.name: sum(v["population"] for v in populations.values() if v["region"] == c.region)
             for c in cities}
    elif by == "na":
        w = {name: sum(populations[d]["population"] for d in depts)
             for name, depts in config.NA_CITY_DEPARTMENTS.items()}
    else:
        raise ValueError(by)
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def weighted_mean(series: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    df = pd.DataFrame({k: series[k] for k in weights})
    w = pd.Series(weights)
    valid = df.notna()
    num = (df.fillna(0) * w).sum(axis=1)
    den = (valid * w).sum(axis=1)
    out = num / den
    out[den < 0.99] = np.nan  # pas de moyenne si une ville manque
    return out


def hourly_to_half_hourly(s: pd.Series) -> pd.Series:
    idx = pd.date_range(s.index.min(), s.index.max(), freq="30min", tz="UTC")
    return s.reindex(idx).interpolate(method="time", limit=1, limit_area="inside")


# --- Matrice jours × demi-heures ---------------------------------------------

def to_daily_matrix(s: pd.Series, tz: str = config.TZ) -> pd.DataFrame:
    """Série UTC au pas de 30 min → DataFrame (jour local) × (créneau 0..47)."""
    local = s.index.tz_convert(tz)
    frame = pd.DataFrame({
        "date": local.tz_localize(None).normalize(),
        "slot": local.hour * 2 + local.minute // 30,
        "v": s.to_numpy(),
    })
    m = frame.groupby(["date", "slot"])["v"].mean().unstack("slot")
    m = m.reindex(columns=range(SLOTS))
    full_days = pd.date_range(m.index.min(), m.index.max(), freq="D")
    m = m.reindex(full_days)
    m.index.name = "date"
    m.columns.name = "slot"
    return fill_spring_gap(m)


def fill_spring_gap(m: pd.DataFrame) -> pd.DataFrame:
    """Comble 02:00 et 02:30 (créneaux 4 et 5) par interpolation entre 01:30 et 03:00.

    On ne touche à aucun autre trou : une interpolation plus large pourrait
    utiliser une valeur d'après la coupure de midi pour remplir la matinée.
    """
    m = m.copy()
    gap = m[4].isna() & m[5].isna() & m[3].notna() & m[6].notna()
    a, b = m.loc[gap, 3], m.loc[gap, 6]
    m.loc[gap, 4] = a + (b - a) / 3
    m.loc[gap, 5] = a + 2 * (b - a) / 3
    return m


def from_daily_matrix(m: pd.DataFrame) -> pd.Series:
    """Format long indexé par (date, slot)."""
    m = m.rename_axis(index="date", columns="slot")
    return m.stack(future_stack=True).rename("value")


# --- Construction --------------------------------------------------------------

def _years(first: int) -> range:
    return range(first, config.LAST_YEAR + 1)


def build_temperatures(cities, weights) -> pd.DataFrame:
    obs, d1, d2 = {}, {}, {}
    fc_end = config.FORECAST_END
    for c in cities:
        paths = [config.RAW_DIR / "weather" / f"era5_{c.name}_{y}.json"
                 for y in _years(config.FIRST_WEATHER_YEAR)]
        obs[c.name] = pd.concat([read_open_meteo(p, ["temperature_2m"]) for p in paths])["temperature_2m"]
        fc = read_open_meteo(
            config.RAW_DIR / "weather" / f"forecast_{config.FORECAST_MODEL}_{c.name}_{config.FORECAST_START}_{fc_end}.json",
            ["temperature_2m_previous_day1", "temperature_2m_previous_day2"])
        d1[c.name] = fc["temperature_2m_previous_day1"]
        d2[c.name] = fc["temperature_2m_previous_day2"]
    out = pd.DataFrame({
        "temp_obs": hourly_to_half_hourly(weighted_mean(obs, weights)),
    })
    out["temp_fc_d1"] = hourly_to_half_hourly(weighted_mean(d1, weights)).reindex(out.index)
    out["temp_fc_d2"] = hourly_to_half_hourly(weighted_mean(d2, weights)).reindex(out.index)
    return out


def build_all() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pops = json.loads((config.RAW_DIR / "populations.json").read_text())

    # National
    nat_paths = [config.RAW_DIR / "eco2mix" / f"national_{y}.csv" for y in _years(config.FIRST_LOAD_YEAR)]
    nat = read_eco2mix_csv(nat_paths, {"consommation": "load", "prevision_j1": "rte_j1",
                                       "prevision_j": "rte_j"})
    w_nat = population_weights(pops, config.NATIONAL_CITIES, by="region")
    log.info("poids nationaux : %s", {k: round(v, 3) for k, v in w_nat.items()})
    temps = build_temperatures(config.NATIONAL_CITIES, w_nat)
    nat = nat.join(temps, how="left")
    nat.to_parquet(config.PROCESSED_DIR / "national.parquet")

    # Nouvelle-Aquitaine
    na_paths = [config.RAW_DIR / "eco2mix" / f"region{config.NA_REGION}_{y}.csv"
                for y in _years(config.FIRST_LOAD_YEAR + 1)]
    na = read_eco2mix_csv(na_paths, {"consommation": "load"})
    na_cities = [c for c in config.ALL_CITIES if c.name in config.NA_CITY_DEPARTMENTS]
    w_na = population_weights(pops, na_cities, by="na")
    log.info("poids Nouvelle-Aquitaine : %s", {k: round(v, 3) for k, v in w_na.items()})
    na = na.join(build_temperatures(na_cities, w_na), how="left")
    na.to_parquet(config.PROCESSED_DIR / "nouvelle_aquitaine.parquet")

    # Calendrier
    zones = load_school_zones(config.RAW_DIR / "calendrier_scolaire.json")
    cal = daily_calendar(f"{config.FIRST_LOAD_YEAR}-01-01", f"{config.LAST_YEAR}-12-31", zones)
    cal.to_parquet(config.PROCESSED_DIR / "calendar.parquet")

    json.dump({"national": w_nat, "nouvelle_aquitaine": w_na},
              open(config.PROCESSED_DIR / "weights.json", "w"), indent=1)
    log.info("tables écrites dans %s", config.PROCESSED_DIR)


def load_processed(name: str = "national") -> tuple[pd.DataFrame, pd.DataFrame]:
    fname = {"national": "national.parquet", "nouvelle_aquitaine": "nouvelle_aquitaine.parquet"}[name]
    df = pd.read_parquet(config.PROCESSED_DIR / fname)
    cal = pd.read_parquet(config.PROCESSED_DIR / "calendar.parquet")
    return df, cal
