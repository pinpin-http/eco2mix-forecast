"""Téléchargement et mise en cache des données sources dans `data/raw/`.

Sources :
- éCO2mix national et régional (RTE, plateforme ODRE, API Opendatasoft v2.1) ;
- températures horaires ERA5 et prévisions archivées ECMWF IFS (Open-Meteo) ;
- populations légales par commune (API Découpage administratif, geo.api.gouv.fr) ;
- calendrier scolaire (ministère de l'Éducation nationale, data.education.gouv.fr).

Chaque fichier n'est téléchargé qu'une fois : s'il existe déjà, il est réutilisé.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path

import requests

from . import config

log = logging.getLogger(__name__)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    return s


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


def _get(session: requests.Session, url: str, params: dict | None = None,
         retries: int = 5, timeout: int = 180) -> requests.Response:
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            wait = 10 * (attempt + 1)
            log.warning("erreur réseau (%s), nouvel essai dans %ss", exc, wait)
            time.sleep(wait)
            continue
        if r.status_code == 429:
            reason = r.text[:200]
            if "Daily" in reason:
                raise RuntimeError(f"quota journalier atteint : {reason}")
            wait = 600 if "Hourly" in reason else 65
            log.warning("limite de débit (%s), pause de %ss", reason, wait)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            wait = 15 * (attempt + 1)
            log.warning("erreur serveur %s, nouvel essai dans %ss", r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"échec après {retries} essais : {url}")


# --- éCO2mix (ODRE) ---------------------------------------------------------

def _odre_year(session, dataset: str, year: int, select: str, extra_where: str = "") -> bytes:
    where = (f'date_heure >= "{year}-01-01T00:00:00Z" '
             f'and date_heure < "{year + 1}-01-01T00:00:00Z"')
    if extra_where:
        where += f" and {extra_where}"
    params = {
        "select": select,
        "where": where,
        "order_by": "date_heure",
        "timezone": "UTC",
        "delimiter": ";",
    }
    url = f"{config.ODRE_BASE}/{dataset}/exports/csv"
    return _get(session, url, params).content


def fetch_eco2mix_national(years, session=None) -> list[Path]:
    session = session or _session()
    paths = []
    for year in years:
        path = config.RAW_DIR / "eco2mix" / f"national_{year}.csv"
        if not path.exists():
            log.info("éCO2mix national %s", year)
            content = _odre_year(session, config.ODRE_NATIONAL, year,
                                 "date_heure,nature,consommation,prevision_j1,prevision_j")
            _atomic_write(path, content)
        paths.append(path)
    return paths


def fetch_eco2mix_region(region: str, years, session=None) -> list[Path]:
    session = session or _session()
    paths = []
    for year in years:
        path = config.RAW_DIR / "eco2mix" / f"region{region}_{year}.csv"
        if not path.exists():
            log.info("éCO2mix région %s, %s", region, year)
            content = _odre_year(session, config.ODRE_REGIONAL, year,
                                 "date_heure,nature,consommation",
                                 extra_where=f'code_insee_region = "{region}"')
            _atomic_write(path, content)
        paths.append(path)
    return paths


# --- Open-Meteo -----------------------------------------------------------

class Throttle:
    """Limiteur simple pour rester sous les quotas gratuits d'Open-Meteo.

    Open-Meteo compte une requête longue comme plusieurs appels (environ un appel
    par tranche de 14 jours et de 10 variables). On garde de la marge par rapport
    aux limites publiées (600/min, 5 000/h, 10 000/jour).
    """

    def __init__(self, per_minute: float = 400, per_hour: float = 4000):
        self.per_minute = per_minute
        self.per_hour = per_hour
        self.events: deque[tuple[float, float]] = deque()

    @staticmethod
    def weight(n_days: int, n_vars: int = 1) -> float:
        return max(1.0, n_days / 14) * max(1.0, n_vars / 10)

    def wait(self, w: float) -> None:
        while True:
            now = time.monotonic()
            while self.events and now - self.events[0][0] > 3600:
                self.events.popleft()
            last_min = sum(x for t, x in self.events if now - t <= 60)
            last_hour = sum(x for _, x in self.events)
            if last_min + w <= self.per_minute and last_hour + w <= self.per_hour:
                self.events.append((now, w))
                return
            time.sleep(5)


def fetch_era5(city: config.City, year: int, session=None, throttle=None) -> Path:
    path = config.RAW_DIR / "weather" / f"era5_{city.name}_{year}.json"
    if path.exists():
        return path
    session = session or _session()
    throttle = throttle or Throttle()
    throttle.wait(Throttle.weight(366))
    log.info("ERA5 %s %s", city.name, year)
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "temperature_2m",
        "timezone": "GMT",
        "models": config.ARCHIVE_MODEL,
    }
    r = _get(session, config.OPEN_METEO_ARCHIVE, params)
    _atomic_write(path, r.content)
    return path


def fetch_forecast(city: config.City, start: str, end: str, session=None, throttle=None) -> Path:
    """Prévisions archivées : valeur prévue 24 h (day1) et 48 h (day2) avant l'heure cible."""
    path = config.RAW_DIR / "weather" / f"forecast_{config.FORECAST_MODEL}_{city.name}_{start}_{end}.json"
    if path.exists():
        return path
    session = session or _session()
    throttle = throttle or Throttle()
    n_days = (int(end[:4]) - int(start[:4]) + 1) * 366
    throttle.wait(Throttle.weight(n_days))
    log.info("prévisions %s %s → %s", city.name, start, end)
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
        "timezone": "GMT",
        "models": config.FORECAST_MODEL,
    }
    r = _get(session, config.OPEN_METEO_PREVIOUS_RUNS, params)
    _atomic_write(path, r.content)
    return path


# --- Populations et calendrier scolaire ------------------------------------

def fetch_populations(regions, session=None) -> Path:
    """Population légale par département, sommée à partir des communes."""
    path = config.RAW_DIR / "populations.json"
    if path.exists():
        return path
    session = session or _session()
    by_dept: dict[str, dict] = {}
    for region in regions:
        r = _get(session, f"{config.GEO_API}/communes",
                 {"codeRegion": region, "fields": "population,codeDepartement", "format": "json"})
        for commune in r.json():
            d = commune["codeDepartement"]
            entry = by_dept.setdefault(d, {"region": region, "population": 0})
            entry["population"] += commune.get("population") or 0
        time.sleep(1)
    _atomic_write(path, json.dumps(by_dept, indent=1, sort_keys=True).encode())
    return path


def fetch_school_holidays(session=None) -> Path:
    path = config.RAW_DIR / "calendrier_scolaire.json"
    if path.exists():
        return path
    session = session or _session()
    params = {
        "select": "description,population,start_date,end_date,location,zones,annee_scolaire",
        "where": 'zones in ("Zone A", "Zone B", "Zone C")',
        "timezone": "UTC",
    }
    r = _get(session, config.SCHOOL_HOLIDAYS_URL, params)
    _atomic_write(path, r.content)
    return path


def download_all() -> None:
    session = _session()
    throttle = Throttle()
    fetch_eco2mix_national(range(config.FIRST_LOAD_YEAR, config.LAST_YEAR + 1), session)
    fetch_eco2mix_region(config.NA_REGION, range(config.FIRST_LOAD_YEAR + 1, config.LAST_YEAR + 1), session)
    regions = sorted({c.region for c in config.NATIONAL_CITIES})
    fetch_populations(regions, session)
    fetch_school_holidays(session)
    fc_end = config.FORECAST_END
    for city in config.ALL_CITIES:
        fetch_forecast(city, config.FORECAST_START, fc_end, session, throttle)
    for year in range(config.FIRST_WEATHER_YEAR, config.LAST_YEAR + 1):
        for city in config.ALL_CITIES:
            fetch_era5(city, year, session, throttle)
    log.info("téléchargements terminés")
