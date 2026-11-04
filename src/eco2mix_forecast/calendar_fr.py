"""Calendrier français : jours fériés, ponts, vacances scolaires, changements d'heure."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TZ

# Identifiant stable par jour férié, pour que le modèle puisse apprendre un effet
# propre à chacun (le 25 décembre ne ressemble pas au lundi de Pentecôte).
HOLIDAY_IDS = {
    "jour_de_l_an": 1,
    "lundi_de_paques": 2,
    "fete_du_travail": 3,
    "victoire_1945": 4,
    "ascension": 5,
    "lundi_de_pentecote": 6,
    "fete_nationale": 7,
    "assomption": 8,
    "toussaint": 9,
    "armistice": 10,
    "noel": 11,
}


def easter(year: int) -> dt.date:
    """Dimanche de Pâques (calendrier grégorien, algorithme dit « anonyme »)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def public_holidays(year: int) -> dict[dt.date, str]:
    """Les 11 jours fériés nationaux de métropole (hors spécificités d'Alsace-Moselle)."""
    e = easter(year)
    return {
        dt.date(year, 1, 1): "jour_de_l_an",
        e + dt.timedelta(days=1): "lundi_de_paques",
        dt.date(year, 5, 1): "fete_du_travail",
        dt.date(year, 5, 8): "victoire_1945",
        e + dt.timedelta(days=39): "ascension",
        e + dt.timedelta(days=50): "lundi_de_pentecote",
        dt.date(year, 7, 14): "fete_nationale",
        dt.date(year, 8, 15): "assomption",
        dt.date(year, 11, 1): "toussaint",
        dt.date(year, 11, 11): "armistice",
        dt.date(year, 12, 25): "noel",
    }


def bridge_days(holidays: set[dt.date]) -> set[dt.date]:
    """Ponts : lundi avant un mardi férié, vendredi après un jeudi férié."""
    out = set()
    for h in holidays:
        if h.weekday() == 1:  # mardi
            out.add(h - dt.timedelta(days=1))
        elif h.weekday() == 3:  # jeudi
            out.add(h + dt.timedelta(days=1))
    return {d for d in out if d not in holidays}


def half_hours_in_local_day(day: dt.date, tz: str = TZ) -> int:
    """46, 48 ou 50 selon que le jour local perd, garde ou gagne une heure."""
    start = pd.Timestamp(day).tz_localize(tz)
    end = pd.Timestamp(day + dt.timedelta(days=1)).tz_localize(tz)
    return int((end - start) / pd.Timedelta(minutes=30))


def daily_calendar(start: str | dt.date, end: str | dt.date,
                   school_zones: pd.Series | None = None) -> pd.DataFrame:
    """Une ligne par jour local, avec toutes les variables calendaires utiles."""
    days = pd.date_range(start, end, freq="D")
    years = range(days.min().year - 1, days.max().year + 2)
    hol: dict[dt.date, str] = {}
    for y in years:
        hol.update(public_holidays(y))
    bridges = bridge_days(set(hol))

    dates = [d.date() for d in days]
    cal = pd.DataFrame(index=days)
    cal.index.name = "date"
    cal["dow"] = days.day_of_week
    cal["doy"] = days.day_of_year
    cal["month"] = days.month
    cal["is_holiday"] = [d in hol for d in dates]
    cal["holiday_id"] = [HOLIDAY_IDS[hol[d]] if d in hol else 0 for d in dates]
    cal["is_bridge"] = [d in bridges for d in dates]
    cal["is_off"] = cal["is_holiday"] | (cal["dow"] >= 5)
    # Fin d'année : du 24 décembre au 1er janvier, activité très réduite.
    cal["year_end"] = ((cal["month"] == 12) & (days.day >= 24)) | ((cal["month"] == 1) & (days.day == 1))
    cal["dst_change"] = [(half_hours_in_local_day(d) - 48) // 2 for d in dates]
    offsets = days.tz_localize(TZ).map(lambda t: t.utcoffset().total_seconds() / 3600)
    cal["summer_time"] = np.asarray(offsets) == 2
    if school_zones is not None:
        cal["school_zones"] = school_zones.reindex(days).fillna(0).astype(int).to_numpy()
    else:
        cal["school_zones"] = 0
    for c in ["is_holiday", "is_bridge", "is_off", "year_end", "summer_time"]:
        cal[c] = cal[c].astype(bool)
    return cal


def day_type(cal: pd.DataFrame) -> pd.Series:
    """Type de jour pour la moyenne saisonnière : lun, mar-jeu, ven, sam, dim/férié."""
    t = pd.Series(1, index=cal.index)
    t[cal["dow"] == 0] = 0
    t[cal["dow"] == 4] = 2
    t[cal["dow"] == 5] = 3
    t[(cal["dow"] == 6) | cal["is_holiday"]] = 4
    return t


def school_holiday_zones(records: list[dict]) -> pd.Series:
    """Nombre de zones (A, B, C) en vacances pour chaque jour local.

    Les dates du jeu de données sont des instants UTC correspondant à minuit
    heure de Paris ; la période de vacances est [début, reprise).
    """
    spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for r in records:
        if r.get("population") == "Enseignants":
            continue
        zone = r.get("zones")
        if zone not in ("Zone A", "Zone B", "Zone C"):
            continue
        s = pd.Timestamp(r["start_date"]).tz_convert(TZ).normalize().tz_localize(None)
        e = pd.Timestamp(r["end_date"]).tz_convert(TZ).normalize().tz_localize(None)
        if e > s:
            spans.setdefault(zone, []).append((s, e))
    if not spans:
        return pd.Series(dtype=int)
    lo = min(s for v in spans.values() for s, _ in v)
    hi = max(e for v in spans.values() for _, e in v)
    days = pd.date_range(lo, hi, freq="D")
    total = pd.Series(0, index=days)
    for zone, periods in spans.items():
        on = pd.Series(False, index=days)
        for s, e in periods:
            on[(days >= s) & (days < e)] = True
        total += on.astype(int)
    return total


def load_school_zones(path: Path) -> pd.Series:
    return school_holiday_zones(json.loads(path.read_text()))
