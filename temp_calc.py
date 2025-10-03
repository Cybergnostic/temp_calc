#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Almuten Figuris (traditional) + Temperament.
Includes fixes:
- Correct weekday → day ruler mapping.
- Planetary hour ruler from sunrise/sunset (unequal hours).
- Triplicity for ALL three Dorothean rulers on every Place (Morinus AF style).
- Robust Whole Sign house calculation.
- Syzygy time format '%H:%M'.

Modified for Streamlit web application.
"""

import math
# import argparse # Removed for Streamlit
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime as py_datetime, timedelta, timezone

import streamlit as st # New Streamlit import

from flatlib.chart import Chart
from flatlib.datetime import Datetime
from flatlib.geopos import GeoPos
from flatlib import const

# Try to import solar utilities; provide a fallback if unavailable.
try:
    from flatlib import solar
    FL_SOLAR_AVAILABLE = True
except Exception:
    FL_SOLAR_AVAILABLE = False

# ----------------------------
# Traditional Tables
# ----------------------------

SIGN_ORDER = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo",
              "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]

DOMICILE = {
    "Aries":"Mars","Taurus":"Venus","Gemini":"Mercury","Cancer":"Moon",
    "Leo":"Sun","Virgo":"Mercury","Libra":"Venus","Scorpio":"Mars",
    "Sagittarius":"Jupiter","Capricorn":"Saturn","Aquarius":"Saturn","Pisces":"Jupiter"
}

EXALT = {
    "Aries":"Sun","Taurus":"Moon","Gemini":None,"Cancer":"Jupiter",
    "Leo":None,"Virgo":"Mercury","Libra":"Saturn","Scorpio":None,
    "Sagittarius":None,"Capricorn":"Mars","Aquarius":None,"Pisces":"Venus"
}

# Dorothean Triplicity (day, night, participating)
TRIPLICITY = {
    "Fire":     ("Sun","Jupiter","Saturn"),
    "Earth":    ("Venus","Moon","Mars"),
    "Air":      ("Saturn","Mercury","Jupiter"),
    "Water":    ("Venus","Mars","Moon"),
}
SIGN_ELEMENT = {
    "Aries":"Fire","Leo":"Fire","Sagittarius":"Fire",
    "Taurus":"Earth","Virgo":"Earth","Capricorn":"Earth",
    "Gemini":"Air","Libra":"Air","Aquarius":"Air",
    "Cancer":"Water","Scorpio":"Water","Pisces":"Water",
}

# Egyptian Terms (bounds)
TERMS_EGYPT = {
    "Aries":[(6,"Jupiter"),(14,"Venus"),(21,"Mercury"),(26,"Mars"),(30,"Saturn")],
    "Taurus":[(8,"Venus"),(14,"Mercury"),(22,"Jupiter"),(27,"Saturn"),(30,"Mars")],
    "Gemini":[(7,"Mercury"),(13,"Jupiter"),(20,"Venus"),(25,"Mars"),(30,"Saturn")],
    "Cancer":[(6,"Mars"),(13,"Venus"),(20,"Mercury"),(27,"Jupiter"),(30,"Saturn")],
    "Leo":[(6,"Saturn"),(13,"Mercury"),(20,"Venus"),(26,"Jupiter"),(30,"Mars")],
    "Virgo":[(7,"Mercury"),(13,"Venus"),(17,"Jupiter"),(20,"Mercury"),(27,"Saturn"),(30,"Mars")],
    "Libra":[(6,"Saturn"),(14,"Mercury"),(21,"Jupiter"),(28,"Venus"),(30,"Mars")],
    "Scorpio":[(7,"Mars"),(13,"Venus"),(19,"Mercury"),(24,"Jupiter"),(30,"Saturn")],
    "Sagittarius":[(12,"Jupiter"),(19,"Venus"),(26,"Mercury"),(30,"Saturn")],
    "Capricorn":[(7,"Mercury"),(14,"Jupiter"),(22,"Venus"),(28,"Saturn"),(30,"Mars")],
    "Aquarius":[(7,"Mercury"),(13,"Venus"),(20,"Jupiter"),(25,"Mars"),(30,"Saturn")],
    "Pisces":[(12,"Venus"),(19,"Jupiter"),(28,"Mercury"),(30,"Mars")],
}

# Chaldean order for decans and planetary hours
CHALDEAN_ORDER_NAMES = ["Saturn", "Jupiter", "Mars", "Sun", "Venus", "Mercury", "Moon"]

# House points (Morinus default)
HOUSE_POINTS = {1:12, 10:11, 7:10, 4:9, 11:8, 5:7, 2:6, 8:5, 9:4, 3:3, 12:2, 6:1}

# Flatlib keys → names
PLANET_KEYS = [const.SUN, const.MOON, const.MERCURY, const.VENUS,
               const.MARS, const.JUPITER, const.SATURN]
PLANET_NAMES = {
    const.SUN:"Sun", const.MOON:"Moon", const.MERCURY:"Mercury", const.VENUS:"Venus",
    const.MARS:"Mars", const.JUPITER:"Jupiter", const.SATURN:"Saturn",
    const.ASC:"Asc", const.MC:"MC", const.NORTH_NODE:"North Node", const.SOUTH_NODE:"South Node"
}
PLANET_NAME_LIST = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]

# Accidental Dignity Scores (Morinus-style used in your screenshot)
RULER_SCORES = {"Day": 7, "Hour": 6}
PHASE_SCORES = {"Greatest": 3, "Least": 1}  # superior planets simple oriental/occidental

# ----------------------------
# Temperament Tables
# ----------------------------

HOT, COLD, MOIST, DRY = "Hot", "Cold", "Moist", "Dry"

ELEMENT_QUALITIES = {
    "Fire": (HOT, DRY), "Air": (HOT, MOIST), "Earth": (COLD, DRY), "Water": (COLD, MOIST)
}
LUNAR_PHASE_QUALITIES = {
    (0, 90): (HOT, MOIST),
    (90, 180): (HOT, DRY),
    (180, 270): (COLD, DRY),
    (270, 360): (COLD, MOIST),
}
SEASON_QUALITIES = {
    "Aries": (HOT, MOIST), "Taurus": (HOT, MOIST), "Gemini": (HOT, MOIST),
    "Cancer": (HOT, DRY),  "Leo": (HOT, DRY),      "Virgo": (HOT, DRY),
    "Libra": (COLD, DRY),  "Scorpio": (COLD, DRY), "Sagittarius": (COLD, DRY),
    "Capricorn": (COLD, MOIST), "Aquarius": (COLD, MOIST), "Pisces": (COLD, MOIST)
}
PLANET_QUALITIES = {
    "Saturn": {"oriental": (COLD, DRY), "occidental": (DRY,)},
    "Jupiter": {"oriental": (HOT, MOIST), "occidental": (MOIST,)},
    "Mars": {"oriental": (HOT, DRY), "occidental": (DRY,)},
    "Venus": {"occidental": (COLD, MOIST), "oriental": (HOT, MOIST)},
    "Mercury": {"occidental": (COLD, DRY), "oriental": (HOT, MOIST)},
}

# ----------------------------
# Utils
# ----------------------------

def norm360(x: float) -> float:
    return x % 360.0

def sign_from_lon(lon: float) -> str:
    idx = int(norm360(lon) // 30)
    return SIGN_ORDER[idx]

def deg_in_sign(lon: float) -> float:
    return norm360(lon) % 30.0

def _house_num_value(h):
    if isinstance(h, int):
        return h
    num = getattr(h, "num", None)
    if callable(num):
        return int(num())
    if num is not None:
        return int(num)
    return int(h)

def is_day_chart(chart: Chart) -> bool:
    sun = chart.get(const.SUN)
    try:
        h = _house_num_value(chart.houses.getObjectHouse(sun))
    except Exception:
        asc = float(chart.get(const.ASC).lon)
        asc_sign = int(norm360(asc) // 30)
        sun_sign = int(norm360(float(sun.lon)) // 30)
        h = ((sun_sign - asc_sign) % 12) + 1
    return 7 <= h <= 12

def part_of_fortune(asc: float, sun: float, moon: float, day: bool) -> float:
    lot = asc + (moon - sun) if day else asc + (sun - moon)
    return norm360(lot)

def term_ruler(sign: str, deg: float) -> str:
    """Egyptian terms, inclusive at boundary."""
    for enddeg, ruler in TERMS_EGYPT[sign]:
        if deg <= enddeg:   # inclusive boundaries
            return ruler
    return TERMS_EGYPT[sign][-1][1]

def decan_ruler(sign: str, deg: float) -> str:
    """Chaldean Faces: Aries 0 - 10 Mars, then Sun, Venus, Mercury, Moon, Saturn, Jupiter cycling every 10° through the zodiac."""
    sign_idx = SIGN_ORDER.index(sign)
    decan_idx_in_sign = int(deg // 10)  # 0,1,2
    sequence = CHALDEAN_ORDER_NAMES     # ["Saturn","Jupiter","Mars","Sun","Venus","Mercury","Moon"]
    base = sequence.index("Mars")       # Aries 0 - 10 starts with Mars
    abs_decan = sign_idx * 3 + decan_idx_in_sign
    ruler_index = (base + abs_decan) % len(sequence)
    return sequence[ruler_index]

def triplicity_rulers_for_AF(sign: str) -> List[str]:
    elem = SIGN_ELEMENT[sign]
    dayr, nightr, part = TRIPLICITY[elem]
    return [dayr, nightr, part]  # all three, Morinus AF

def house_num_of_obj(chart, obj) -> int:
    """Whole Sign house number based on sign distance from ASC."""
    try:
        asc_lon = float(chart.get(const.ASC).lon)
        obj_lon = float(obj.lon)
        asc_sign_idx = int(norm360(asc_lon) // 30)
        obj_sign_idx = int(norm360(obj_lon) // 30)
        return ((obj_sign_idx - asc_sign_idx) % 12) + 1
    except Exception:
        return 0

def house_points_for(chart: Chart) -> Dict[str,int]:
    pts: Dict[str,int] = {}
    for key in PLANET_KEYS:
        obj = chart.get(key)
        nm = PLANET_NAMES[key]
        h = house_num_of_obj(chart, obj)
        pts[nm] = HOUSE_POINTS.get(h, 0)
    return pts

# ----------------------------
# Day/Hour rulers and phases
# ----------------------------

def get_day_of_week_ruler(dt: Datetime) -> str:
    """Planetary day ruler, Python weekday(): 0=Mon..6=Sun."""
    y, m, d = map(int, dt.date.toString().split('/'))
    dow = py_datetime(y, m, d).weekday()
    weekday_to_ruler = {
        0: "Moon",     # Monday
        1: "Mars",     # Tuesday
        2: "Mercury",  # Wednesday
        3: "Jupiter",  # Thursday
        4: "Venus",    # Friday
        5: "Saturn",   # Saturday
        6: "Sun",      # Sunday
    }
    return weekday_to_ruler[dow]

def _to_minutes(dtime: Datetime) -> int:
    H, M, *_ = map(int, dtime.time.toString().split(':'))
    return H * 60 + M

def _sunrise_sunset(dt: Datetime, pos: GeoPos) -> Tuple[int, int]:
    """Return (sunrise_min, sunset_min) minutes from 00:00 local. Fallback: 06:00/18:00."""
    if FL_SOLAR_AVAILABLE:
        y, m, d = map(int, dt.date.toString().split('/'))
        sr = solar.sunrise(Datetime(f"{y:04d}/{m:02d}/{d:02d}", "12:00", dt.time.offset), pos)
        ss = solar.sunset (Datetime(f"{y:04d}/{m:02d}/{d:02d}", "12:00", dt.time.offset), pos)
        return _to_minutes(sr), _to_minutes(ss)
    # crude fallback if solar not available
    return 6*60, 18*60

def get_hour_ruler(dt: Datetime, pos: GeoPos) -> str:
    """Planetary hour ruler using sunrise/sunset (unequal hours)."""
    day_ruler = get_day_of_week_ruler(dt)
    rise_min, set_min = _sunrise_sunset(dt, pos)
    cur_min = _to_minutes(dt)
    chaldean = CHALDEAN_ORDER_NAMES
    day_start_idx = chaldean.index(day_ruler)

    if rise_min <= cur_min < set_min:
        # Day series: sunrise → sunset
        span = set_min - rise_min
        hour_len = span / 12.0
        idx_in_series = int((cur_min - rise_min) // hour_len)
        ruler_idx = (day_start_idx + idx_in_series) % 7
    else:
        # Night series starts at sunset; first night hour is next in sequence
        # Night span from sunset → next sunrise
        if cur_min >= set_min:
            span = (rise_min + 24*60) - set_min
            idx_in_series = int((cur_min - set_min) // (span / 12.0))
        else:
            span = (rise_min + 24*60) - set_min # Correction: full 24h cycle
            if cur_min < rise_min:
                 # It's the period between midnight and sunrise of the chart's day
                cur_min_adj = cur_min + 24*60 # Treat as late night/early morning of the next day
                idx_in_series = int((cur_min_adj - set_min) // (span / 12.0))
            else:
                # Should be caught by rise_min <= cur_min < set_min, but defensively
                idx_in_series = 0 # Fallback
                
        ruler_idx = (day_start_idx + 12 + idx_in_series) % 7
        
        # Simpler night ruler calculation (ensuring it wraps correctly)
        if not (rise_min <= cur_min < set_min):
            # Calculate total minutes from sunset/sunrise boundary
            if cur_min >= set_min:
                minutes_from_start = cur_min - set_min
                night_span = (rise_min + 24*60) - set_min
            else:
                # Cur_min is between 00:00 and sunrise
                minutes_from_start = (24*60 - set_min) + cur_min
                night_span = (rise_min + 24*60) - set_min

            hour_len = night_span / 12.0
            idx_in_series = int(minutes_from_start // hour_len)
            
            # The *first* night hour follows the day ruler. Day ruler is index N. 
            # The night series starts at N+1. Hour 13 is N+1, hour 14 is N+2, etc.
            ruler_idx = (day_start_idx + 12 + idx_in_series) % 7


    return chaldean[ruler_idx]

def is_oriental(sun_lon: float, planet_lon: float) -> bool:
    elongation = norm360(planet_lon - sun_lon)
    return 0 <= elongation < 180

def get_planet_phase_score(chart: Chart, planet_name: str) -> int:
    """Simple oriental/occidental score for superior planets."""
    if planet_name not in ["Saturn", "Jupiter", "Mars"]:
        return 0
    planet_key = next((k for k,v in PLANET_NAMES.items() if v == planet_name), None)
    if not planet_key:
        return 0
    sun_lon = float(chart.get(const.SUN).lon)
    planet_lon = float(chart.get(planet_key).lon)
    return PHASE_SCORES["Greatest"] if is_oriental(sun_lon, planet_lon) else PHASE_SCORES["Least"]

# ----------------------------
# Prenatal Syzygy
# ----------------------------

def lunar_elongation(chart: Chart) -> float:
    sun = chart.get(const.SUN)
    moon = chart.get(const.MOON)
    return norm360(float(moon.lon) - float(sun.lon))

def syzygy_distance_deg(elong: float) -> float:
    # Syzygy is 0 (New) or 180 (Full). Min distance to either.
    return min(elong, 360.0-elong, abs(elong-180.0))

def back_dt(dt: Datetime, tz_str: str, minutes: int) -> Datetime:
    # Extract UTC offset in minutes from the timezone string
    sign = 1 if tz_str[0] == '+' else -1
    tzmins = sign * (int(tz_str[1:3]) * 60 + int(tz_str[4:6]))
    
    # Create a timezone object for the chart time
    tz = timezone(timedelta(minutes=tzmins))
    
    # Reconstruct the python datetime object
    y, m, d = map(int, dt.date.toString().split('/'))
    H, M, *_ = map(int, dt.time.toString().split(':'))
    py = py_datetime(y, m, d, H, M, tzinfo=tz)
    
    # Subtract the minutes
    py_back = py - timedelta(minutes=minutes)
    
    # Format the timezone offset string back to "+HH:MM"
    tzs = py_back.strftime('%z'); 
    tz_out = f"{tzs[:3]}:{tzs[3:]}"
    
    # Create a new flatlib Datetime object
    return Datetime(py_back.strftime('%Y/%m/%d'), py_back.strftime('%H:%M'), tz_out)

@st.cache_data
def find_prenatal_syzygy(dt: Datetime, pos: GeoPos, tz_str: str,
                         hsys=const.HOUSES_WHOLE_SIGN, max_back_days=40):
    step = 180  # minutes
    total = int((max_back_days*24*60)//step)
    prev_dt = dt
    
    # Check current chart for syzygy
    chart_prev = Chart(prev_dt, pos, hsys=hsys)
    prev_d = syzygy_distance_deg(lunar_elongation(chart_prev))
    
    turning_found = False
    best_dt, best_val = prev_dt, prev_d

    for i in range(1, total+1):
        cur_dt = back_dt(dt, tz_str, i*step)
        chart_cur = Chart(cur_dt, pos, hsys=hsys)
        cur_d = syzygy_distance_deg(lunar_elongation(chart_cur))
        
        if cur_d < best_val:
            best_val, best_dt = cur_d, cur_dt
        
        # Look for the turn-around (distance starts increasing)
        if i > 2 and cur_d > prev_d and not turning_found:
            turning_found = True
            t1, t2 = cur_dt, prev_dt
            break
            
        prev_d, prev_dt = cur_d, cur_dt

    if not turning_found:
        # If no clear turning point found, bracket around the best step
        t1 = back_dt(best_dt, tz_str, step)
        t2 = back_dt(best_dt, tz_str, -step)

    def to_py(d: Datetime):
        y, m, d2 = map(int, d.date.toString().split('/'))
        H, M, *_ = map(int, d.time.toString().split(':'))
        sign = 1 if tz_str[0] == '+' else -1
        tzmins = sign * (int(tz_str[1:3]) * 60 + int(tz_str[4:6]))
        tz = timezone(timedelta(minutes=tzmins))
        return py_datetime(y, m, d2, H, M, tzinfo=tz)

    # Begin trisection for high-precision time
    left, right = to_py(t1), to_py(t2)
    for _ in range(30): # 30 iterations gives high precision
        mid = left + (right - left) / 2
        
        tzs = mid.strftime('%z'); tz_out = f"{tzs[:3]}:{tzs[3:]}"
        mid_dt = Datetime(mid.strftime('%Y/%m/%d'), mid.strftime('%H:%M'), tz_out)
        val_mid = syzygy_distance_deg(lunar_elongation(Chart(mid_dt, pos, hsys=hsys)))

        # Use 1-minute offset for bracketing evaluation
        left_test = mid - timedelta(minutes=1)
        right_test = mid + timedelta(minutes=1)

        tzs_l = left_test.strftime('%z'); tz_out_l = f"{tzs_l[:3]}:{tzs_l[3:]}"
        left_dt  = Datetime(left_test.strftime('%Y/%m/%d'),  left_test.strftime('%H:%M'),  tz_out_l)
        val_left  = syzygy_distance_deg(lunar_elongation(Chart(left_dt,  pos, hsys=hsys)))

        tzs_r = right_test.strftime('%z'); tz_out_r = f"{tzs_r[:3]}:{tzs_r[3:]}"
        right_dt = Datetime(right_test.strftime('%Y/%m/%d'), right_test.strftime('%H:%M'), tz_out_r)
        val_right = syzygy_distance_deg(lunar_elongation(Chart(right_dt, pos, hsys=hsys)))

        if val_left < val_mid:
            right = mid
        elif val_right < val_mid:
            left = mid
        else:
            # If mid is the minimum, narrow the search space
            left = left_test
            right = right_test
            # break # Could break, but continuing to 30 ensures max precision

    # Final result
    mid = left + (right - left) / 2
    tzs = mid.strftime('%z'); tz_out = f"{tzs[:3]}:{tzs[3:]}"
    
    # flatlib time string needs to be HH:MM
    syz_dt = Datetime(mid.strftime('%Y/%m/%d'), mid.strftime('%H:%M'), tz_out)
    
    ch = Chart(syz_dt, pos, hsys=hsys)
    elong = lunar_elongation(ch)
    kind = "New" if min(elong, 330.0-elong) <= abs(elong-180.0) else "Full" # Adjusted threshold for safety
    lon = float(ch.get(const.SUN).lon) if kind == "New" else float(ch.get(const.MOON).lon)
    return kind, norm360(lon)

# ----------------------------
# Dignities & Temperament
# ----------------------------

def dignity_contributions(sign: str, deg: float) -> Dict[str, Dict[str,int]]:
    """Morinus AF: dom(5), exalt(4), trip(all three ×3), term(2), decan(1)."""
    out: Dict[str, Dict[str,int]] = {}
    def add(p, kind, pts):
        if p is None: return
        d = out.setdefault(p, {})
        d[kind] = d.get(kind, 0) + pts

    add(DOMICILE[sign], "dom", 5)
    add(EXALT[sign],   "exalt", 4)
    for r in triplicity_rulers_for_AF(sign):
        add(r, "trip", 3)
    add(term_ruler(sign, deg),  "term", 2)
    add(decan_ruler(sign, deg), "decan", 1)
    return out

def get_sign_qualities(sign: str) -> Tuple[str, str]:
    elem = SIGN_ELEMENT[sign]
    return ELEMENT_QUALITIES.get(elem, ('',''))

def get_planet_qualities(planet_key: str, sun_lon: float, planet_lon: float) -> Tuple[str, ...]:
    planet_name = PLANET_NAMES.get(planet_key, planet_key)
    if planet_name not in PLANET_QUALITIES:
        return ()
    elongation = norm360(planet_lon - sun_lon)
    is_superior = planet_key in [const.SATURN, const.JUPITER, const.MARS]
    # Oriental/Occidental definition differs for superior vs inferior/luminaries
    is_oriental_flag = (is_superior and elongation < 180) or (not is_superior and elongation >= 180)
    key = "oriental" if is_oriental_flag else "occidental"
    qualities = PLANET_QUALITIES[planet_name].get(key, ())
    if isinstance(qualities, str):
        return (qualities,)
    return qualities

def get_lunar_phase_qualities(elongation: float) -> Tuple[str, str]:
    for (start, end), qualities in LUNAR_PHASE_QUALITIES.items():
        if start <= elongation < end:
            return qualities
    return ('', '')

def get_seasonal_qualities(sign: str) -> Tuple[str, str]:
    return SEASON_QUALITIES.get(sign, ('',''))

def get_node_qualities(node_key: str) -> Tuple[str, str]:
    if node_key == const.NORTH_NODE: return (HOT, MOIST)
    if node_key == const.SOUTH_NODE: return (COLD, DRY)
    return ('','')

def calculate_temperament(chart: Chart, syz_lon: float, almuten_figuris: str, max_orb: float = 6.0) -> Dict[str, int]:
    qualities = {HOT: 0, COLD: 0, MOIST: 0, DRY: 0}
    def add(qs: Tuple[str, ...]):
        for q in qs:
            if q in qualities: qualities[q] += 1

    sun_lon = float(chart.get(const.SUN).lon)
    asc = chart.get(const.ASC)
    asc_lon = float(asc.lon)
    asc_sign = sign_from_lon(asc_lon)

    # 1. Asc sign
    add(get_sign_qualities(asc_sign))

    # 2. Asc ruler
    ruler_name = DOMICILE[asc_sign]
    ruler_key = next(k for k,v in PLANET_NAMES.items() if v == ruler_name)
    ruler = chart.get(ruler_key)
    add(get_planet_qualities(ruler_key, sun_lon, float(ruler.lon)))
    add(get_sign_qualities(sign_from_lon(float(ruler.lon))))

    # 3. Aspects to Asc ruler (0,60,90,120,180; orb ~6)
    for p_key in PLANET_KEYS:
        if p_key == ruler_key: continue
        p = chart.get(p_key); p_lon = float(p.lon)
        for asp in [0, 60, 90, 120, 180]:
            diff = abs(norm360(p_lon - float(ruler.lon)))
            orb = min(abs(diff - asp), abs(360 - abs(diff - asp)))
            if orb <= max_orb:
                if asp == 0: add(get_planet_qualities(p_key, sun_lon, p_lon))
                else: add(get_sign_qualities(sign_from_lon(p_lon)))

    # 4. Planets/Nodes in 1st
    for key in PLANET_KEYS + [const.NORTH_NODE, const.SOUTH_NODE]:
        obj = chart.get(key)
        if house_num_of_obj(chart, obj) == 1:
            if key in [const.NORTH_NODE, const.SOUTH_NODE]: add(get_node_qualities(key))
            else: add(get_planet_qualities(key, sun_lon, float(obj.lon)))

    # 5. Duad (simplified: ruler again)
    add(get_planet_qualities(ruler_key, sun_lon, float(ruler.lon)))

    # 6. Aspects to Asc
    for p_key in PLANET_KEYS:
        p = chart.get(p_key); p_lon = float(p.lon)
        for asp in [60, 90, 120, 180]:
            diff = abs(norm360(p_lon - asc_lon))
            orb = min(abs(diff - asp), abs(360 - abs(diff - asp)))
            if orb <= max_orb:
                add(get_sign_qualities(sign_from_lon(p_lon)))

    # 7. Moon + phase + dispositor + aspects
    moon = chart.get(const.MOON); moon_lon = float(moon.lon)
    add(get_sign_qualities(sign_from_lon(moon_lon)))
    add(get_lunar_phase_qualities(lunar_elongation(chart)))
    moon_disp_name = DOMICILE[sign_from_lon(moon_lon)]
    moon_disp_key = next(k for k,v in PLANET_NAMES.items() if v == moon_disp_name)
    add(get_sign_qualities(sign_from_lon(float(chart.get(moon_disp_key).lon))))
    for p_key in PLANET_KEYS:
        if p_key == const.MOON: continue
        p = chart.get(p_key); p_lon = float(p.lon)
        for asp in [0, 60, 90, 120, 180]:
            diff = abs(norm360(p_lon - moon_lon))
            orb = min(abs(diff - asp), abs(360 - abs(diff - asp)))
            if orb <= max_orb:
                if asp == 0: add(get_planet_qualities(p_key, sun_lon, p_lon))
                else: add(get_sign_qualities(sign_from_lon(p_lon)))

    # 8. Sun's season (sign nature)
    add(get_seasonal_qualities(sign_from_lon(sun_lon)))

    # 9. Almuten figuris sign + qualities
    alm_key = next((k for k,v in PLANET_NAMES.items() if v == almuten_figuris), None)
    if alm_key is None: alm_key = const.MARS
    alm = chart.get(alm_key); alm_lon = float(alm.lon)
    add(get_sign_qualities(sign_from_lon(alm_lon)))
    add(get_planet_qualities(alm_key, sun_lon, alm_lon))
    return qualities

# ----------------------------
# Data Class
# ----------------------------

@dataclass
class PlaceScore:
    name: str
    lon: float
    sign: str
    deg: float
    contributions: Dict[str, Dict[str,int]] = field(default_factory=dict)

# ----------------------------
# Core Calculation Function (replaces original main)
# ----------------------------

def run_calculation(date_str, time_str, tz_str, lat, lon, hsys):
    """The main logic function, called by the Streamlit UI."""
    try:
        dt = Datetime(date_str.replace("-", "/"), time_str, tz_str)
        pos = GeoPos(lat, lon)
    except Exception as e:
        st.error(f"Error creating Datetime/GeoPos. Check your inputs: {e}")
        return

    try:
        chart = Chart(dt, pos, hsys=hsys)
    except Exception as e:
        st.error(f"Error calculating chart (flatlib error): {e}")
        return

    isday = is_day_chart(chart)

    asc_lon = float(chart.get(const.ASC).lon)
    sun_lon = float(chart.get(const.SUN).lon)
    moon_lon = float(chart.get(const.MOON).lon)

    # --- 1) Places for Almuten Figuris ---
    places: List[PlaceScore] = []
    def add_place(name, lon):
        s = sign_from_lon(lon); d = deg_in_sign(lon)
        pc = PlaceScore(name=name, lon=lon, sign=s, deg=d)
        pc.contributions = dignity_contributions(s, d)  # AF style: all triplicity rulers
        places.append(pc)

    add_place("Sun", sun_lon)
    add_place("Moon", moon_lon)
    add_place("Asc", asc_lon)

    try:
        kind, syz_lon = find_prenatal_syzygy(dt, pos, tz_str, hsys=hsys)
    except Exception as e:
        st.error(f"Could not calculate Prenatal Syzygy. Error: {e}")
        return
        
    add_place(f"Prenatal {kind}", syz_lon)

    lot_lon = part_of_fortune(asc_lon, sun_lon, moon_lon, isday)
    add_place("Fortune", lot_lon)

    # --- 2) Scoring ---
    per_planet: Dict[str, Dict[str,int]] = { k:{} for k in PLANET_NAME_LIST }
    total: Dict[str,int] = {k:0 for k in PLANET_NAME_LIST}
    essential_total: Dict[str,int] = {k:0 for k in PLANET_NAME_LIST}
    accidental_total: Dict[str,int] = {k:0 for k in PLANET_NAME_LIST}

    # Essential (dom/exalt/trip/term/decan) from the five Places
    for pl in places:
        # Debugging message is now optional or removed for clean UI
        for p, kinds in pl.contributions.items():
            if p not in total: 
                continue
            parts = per_planet.setdefault(p, {})
            for kind, pts in kinds.items():
                parts[kind] = parts.get(kind, 0) + pts
                essential_total[p] += pts
                total[p] += pts

    # Accidental: House points
    hpts = house_points_for(chart)
    for p, pts in hpts.items():
        total[p] += pts
        accidental_total[p] += pts
        per_planet[p]["house"] = pts

    # Accidental: Day ruler (+7)
    day_ruler_name = get_day_of_week_ruler(dt)
    total[day_ruler_name] += RULER_SCORES["Day"]
    accidental_total[day_ruler_name] += RULER_SCORES["Day"]
    per_planet[day_ruler_name]["Day Ruler"] = RULER_SCORES["Day"]

    # Accidental: Hour ruler (+6)
    hour_ruler_name = get_hour_ruler(dt, pos)
    total[hour_ruler_name] += RULER_SCORES["Hour"]
    accidental_total[hour_ruler_name] += RULER_SCORES["Hour"]
    per_planet[hour_ruler_name]["Hour Ruler"] = RULER_SCORES["Hour"]

    # Accidental: Phase (+3 / +1) for superiors
    for p_name in ["Saturn", "Jupiter", "Mars"]:
        phase_score = get_planet_phase_score(chart, p_name)
        total[p_name] += phase_score
        accidental_total[p_name] += phase_score
        if phase_score:
            per_planet[p_name]["Phase"] = phase_score

    # Winner rows preparation
    rows = []
    for p in PLANET_NAME_LIST:
        parts = per_planet.get(p, {})
        detail_parts = []
        for k in ["dom","exalt","trip","term","decan"]:
            if parts.get(k, 0) > 0: detail_parts.append(f"{k}:{parts[k]}")
        for k in ["house","Day Ruler","Hour Ruler","Phase"]:
            if parts.get(k, 0) > 0: detail_parts.append(f"{k}:{parts[k]}")
        
        # Prepare data for a clean Streamlit table/dataframe
        rows.append({
            "Planet": p,
            "Essential": essential_total[p],
            "Accidental": accidental_total[p],
            "Grand Total": total[p],
            "Detail": " ".join(detail_parts)
        })

    # Sort the list by Grand Total
    rows.sort(key=lambda x: x["Grand Total"], reverse=True)

    almuten_figuris_name = rows[0]["Planet"]
    best = rows[0]["Grand Total"]
    winners = [p["Planet"] for p in rows if p["Grand Total"]==best]

    # --- Output (Using Streamlit Functions) ---
    st.header("✨ Chart & Sect Details")
    st.markdown(f"**Date/Time:** {date_str} {time_str} {tz_str} | **House System:** {hsys} | **Sect:** {'Day' if isday else 'Night'}")
    
    st.subheader("Key Points")
    st.markdown(f"**Asc:** {sign_from_lon(asc_lon)} {deg_in_sign(asc_lon):.2f}°")
    st.markdown(f"**Sun:** {sign_from_lon(sun_lon)} {deg_in_sign(sun_lon):.2f}°")
    st.markdown(f"**Moon:** {sign_from_lon(moon_lon)} {deg_in_sign(moon_lon):.2f}°")
    st.markdown(f"**Day Ruler:** {day_ruler_name} | **Hour Ruler:** {hour_ruler_name}")
    
    st.header("🏛️ Places of Life (for Dignity Scoring)")
    for pl in places:
        st.markdown(f"- **{pl.name}**: {pl.sign} {pl.deg:.2f}° (lon {pl.lon:.3f}°)")

    st.header("🏆 Almuten Figuris Totals")
    st.markdown("Scores: **Dom:5, Exalt:4, Trip(all):3 each, Term:2, Decan:1** | **House:12..1** | **DayRuler:7, HourRuler:6** | **Phase:3/1**")
    
    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df.style.highlight_max(subset=['Grand Total'], axis=0, color='gold'), use_container_width=True)

    st.success(f"**The Almuten Figuris is: {'Tie: ' if len(winners)>1 else ''}{', '.join(winners)}** (Total score: {best})")

    # --- Temperament ---
    temperament_scores = calculate_temperament(chart, syz_lon, almuten_figuris_name)
    hot_cold = temperament_scores[HOT] - temperament_scores[COLD]
    moist_dry = temperament_scores[MOIST] - temperament_scores[DRY]
    quality_to_humor = {
        (HOT, MOIST): "Sanguine (Blood)",
        (HOT, DRY): "Choleric (Yellow Bile)",
        (COLD, DRY): "Melancholic (Black Bile)",
        (COLD, MOIST): "Phlegmatic (Phlegm)",
    }
    if   hot_cold > 0 and moist_dry > 0: temperament_type = quality_to_humor[(HOT, MOIST)]
    elif hot_cold > 0 and moist_dry < 0: temperament_type = quality_to_humor[(HOT, DRY)]
    elif hot_cold < 0 and moist_dry < 0: temperament_type = quality_to_humor[(COLD, DRY)]
    elif hot_cold < 0 and moist_dry > 0: temperament_type = quality_to_humor[(COLD, MOIST)]
    elif hot_cold == 0 and moist_dry == 0: temperament_type = "Balanced (Equal)"
    elif hot_cold == 0: temperament_type = f"Balanced (Hot/Cold), Dominated by {'Moist' if moist_dry > 0 else 'Dry'}"
    else: temperament_type = f"Balanced (Moist/Dry), Dominated by {'Hot' if hot_cold > 0 else 'Cold'}"

    st.header("🌡️ Temperament Analysis")
    st.markdown(f"**Total Quality Scores:** `{temperament_scores}`")
    st.markdown(f"**Hot - Cold Balance:** **{hot_cold}**")
    st.markdown(f"**Moist - Dry Balance:** **{moist_dry}**")
    st.warning(f"**Temperament Type:** **{temperament_type}**")


# ----------------------------
# Streamlit UI Function
# ----------------------------

def app():
    st.title("Almuten Figuris & Temperament Calculator 🌌")
    st.markdown("A traditional astrological calculator based on Morinus' method for determining the ruler of the nativity and the resulting temperament.")
    
    # --- Input Section ---
    st.header("1. Birth Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_input = st.date_input("Date of Birth", value=py_datetime(2023, 10, 23).date(), key="dob_date")
    with col2:
        time_input = st.time_input("Time of Birth (Local)", value=py_datetime(2023, 10, 23, 14, 30).time(), key="dob_time")
    with col3:
        tz_str = st.text_input("Timezone Offset (+HH:MM)", value="+02:00", key="tz_offset", help="e.g., +02:00 for Berlin, -05:00 for New York.")

    st.header("2. Location Data")
    col4, col5 = st.columns(2)
    with col4:
        lat = st.number_input("Latitude", value=48.86, format="%.2f", key="latitude") # Paris
    with col5:
        lon = st.number_input("Longitude", value=2.35, format="%.2f", key="longitude")

    st.header("3. House System")
    house_system = st.radio(
        "Select House System for AF/Temperament Calculations:",
        ('Whole Sign', 'Placidus'),
        horizontal=True,
        key="house_sys"
    )
    hsys_const = const.HOUSES_WHOLE_SIGN if house_system == 'Whole Sign' else const.HOUSES_PLACIDUS

    st.markdown("---")
    
    # --- Calculation Button ---
    if st.button("Calculate Almuten Figuris & Temperament", type="primary"):
        # Format the inputs for your logic
        date_str = date_input.strftime("%Y-%m-%d")
        time_str = time_input.strftime("%H:%M")
        
        # Display a spinner while calculation runs
        with st.spinner('Calculating Chart, Prenatal Syzygy, Dignities, and Temperament...'):
            run_calculation(date_str, time_str, tz_str, lat, lon, hsys_const)

# Execute the Streamlit UI function
if __name__ == "__main__":
    app()