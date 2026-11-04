"""Constantes du projet : chemins, périodes, villes et paramètres des sources."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ECO2MIX_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = DOCS_DIR / "results"

TZ = "Europe/Paris"

USER_AGENT = "eco2mix-forecast/0.1 (+https://github.com/pinpin-http/eco2mix-forecast)"

# Période couverte. 2012 ne sert qu'à alimenter les retards de début 2013.
FIRST_LOAD_YEAR = 2012
FIRST_WEATHER_YEAR = 2013
LAST_YEAR = 2025  # dernière année civile complète, tenue à l'écart pour le test final

# Backtest : chaque année de cette liste est prédite par un modèle entraîné
# sur toutes les années précédentes (fenêtre croissante).
BACKTEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
TEST_YEAR = LAST_YEAR

# Calcul limité pour cohabiter avec d'autres travaux sur la machine.
N_JOBS = int(os.environ.get("ECO2MIX_N_JOBS", "2"))

# --- Sources ---------------------------------------------------------------

ODRE_BASE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
ODRE_NATIONAL = "eco2mix-national-cons-def"
ODRE_REGIONAL = "eco2mix-regional-cons-def"

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_MODEL = "era5"
FORECAST_MODEL = "ecmwf_ifs025"
# Les prévisions archivées d'ECMWF IFS 0,25° commencent le 2024-02-04.
FORECAST_START = "2024-02-04"
FORECAST_END = f"{LAST_YEAR}-12-31"

GEO_API = "https://geo.api.gouv.fr"
SCHOOL_HOLIDAYS_URL = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-calendrier-scolaire/exports/json"
)


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    region: str  # code INSEE de la région représentée


# Une ville par région métropolitaine (Corse exclue : elle n'est pas raccordée
# au réseau de RTE). Chaque ville est pondérée par la population de sa région.
NATIONAL_CITIES = [
    City("Paris", 48.8566, 2.3522, "11"),
    City("Orleans", 47.9029, 1.9093, "24"),
    City("Dijon", 47.3220, 5.0415, "27"),
    City("Rouen", 49.4432, 1.0999, "28"),
    City("Lille", 50.6292, 3.0573, "32"),
    City("Strasbourg", 48.5734, 7.7521, "44"),
    City("Nantes", 47.2184, -1.5536, "52"),
    City("Rennes", 48.1173, -1.6778, "53"),
    City("Bordeaux", 44.8378, -0.5792, "75"),
    City("Toulouse", 43.6047, 1.4442, "76"),
    City("Lyon", 45.7640, 4.8357, "84"),
    City("Marseille", 43.2965, 5.3698, "93"),
]

# Nouvelle-Aquitaine : trois villes, chacune pondérée par la population des
# départements qu'on lui rattache (rattachement géographique simple).
NA_REGION = "75"
NA_CITY_DEPARTMENTS = {
    "Bordeaux": ["24", "33", "40", "47", "64"],
    "Poitiers": ["16", "17", "79", "86"],
    "Limoges": ["19", "23", "87"],
}
NA_EXTRA_CITIES = [
    City("Poitiers", 46.5802, 0.3404, "75"),
    City("Limoges", 45.8336, 1.2611, "75"),
]

ALL_CITIES = NATIONAL_CITIES + NA_EXTRA_CITIES
