import datetime as dt

import pandas as pd
import pytest

from eco2mix_forecast.calendar_fr import (
    bridge_days,
    daily_calendar,
    day_type,
    easter,
    half_hours_in_local_day,
    public_holidays,
    school_holiday_zones,
)


@pytest.mark.parametrize("year,expected", [
    (2013, dt.date(2013, 3, 31)),
    (2019, dt.date(2019, 4, 21)),
    (2024, dt.date(2024, 3, 31)),
    (2025, dt.date(2025, 4, 20)),
    (2038, dt.date(2038, 4, 25)),
])
def test_easter(year, expected):
    assert easter(year) == expected


def test_holidays_2025():
    h = public_holidays(2025)
    assert len(h) == 11
    assert h[dt.date(2025, 4, 21)] == "lundi_de_paques"
    assert h[dt.date(2025, 5, 29)] == "ascension"
    assert h[dt.date(2025, 6, 9)] == "lundi_de_pentecote"
    assert h[dt.date(2025, 7, 14)] == "fete_nationale"
    assert dt.date(2025, 12, 26) not in h  # férié seulement en Alsace-Moselle


def test_bridges():
    hol = set(public_holidays(2025)) | set(public_holidays(2026))
    b = bridge_days(hol)
    assert dt.date(2025, 5, 30) in b    # vendredi après l'Ascension
    assert dt.date(2025, 11, 10) in b   # lundi avant le mardi 11 novembre
    assert dt.date(2026, 7, 13) in b    # lundi avant le mardi 14 juillet
    assert dt.date(2025, 8, 15) not in b
    assert all(d.weekday() in (0, 4) for d in b)


def test_dst_days():
    assert half_hours_in_local_day(dt.date(2025, 3, 30)) == 46
    assert half_hours_in_local_day(dt.date(2025, 10, 26)) == 50
    assert half_hours_in_local_day(dt.date(2025, 6, 1)) == 48
    cal = daily_calendar("2025-03-29", "2025-10-27")
    assert cal.loc["2025-03-30", "dst_change"] == -1
    assert cal.loc["2025-10-26", "dst_change"] == 1
    assert not cal.loc["2025-03-29", "summer_time"]
    assert cal.loc["2025-03-31", "summer_time"]
    assert cal.loc["2025-10-27", "summer_time"] == False  # noqa: E712


def test_daily_calendar_flags():
    cal = daily_calendar("2025-05-26", "2025-06-09")
    assert cal.loc["2025-05-29", "is_holiday"] and cal.loc["2025-05-29", "is_off"]
    assert cal.loc["2025-05-30", "is_bridge"] and not cal.loc["2025-05-30", "is_off"]
    assert cal.loc["2025-05-31", "is_off"]  # samedi
    assert cal.loc["2025-06-09", "holiday_id"] > 0
    t = day_type(cal)
    assert t.loc["2025-05-26"] == 0 and t.loc["2025-05-29"] == 4 and t.loc["2025-05-30"] == 2


def test_school_holidays_count_zones():
    records = [
        # dates UTC = minuit heure de Paris ; vacances [début, reprise)
        {"zones": "Zone A", "population": "-", "start_date": "2025-02-07T23:00:00+00:00",
         "end_date": "2025-02-23T23:00:00+00:00"},
        {"zones": "Zone B", "population": "Élèves", "start_date": "2025-02-14T23:00:00+00:00",
         "end_date": "2025-03-02T23:00:00+00:00"},
        {"zones": "Zone B", "population": "Enseignants", "start_date": "2025-01-01T23:00:00+00:00",
         "end_date": "2025-01-05T23:00:00+00:00"},
        {"zones": "Corse", "population": "-", "start_date": "2025-02-14T23:00:00+00:00",
         "end_date": "2025-03-02T23:00:00+00:00"},
    ]
    z = school_holiday_zones(records)
    assert z.loc["2025-02-08"] == 1
    assert z.loc["2025-02-15"] == 2
    assert z.loc["2025-02-23"] == 2
    assert z.loc["2025-02-24"] == 1    # reprise en zone A
    assert z.loc["2025-03-03"] == 0    # reprise en zone B
    assert pd.Timestamp("2025-01-03") not in z.index  # enseignants ignorés
