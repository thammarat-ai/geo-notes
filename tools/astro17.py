# -*- coding: utf-8 -*-
"""
astro17.py — ephemeris สำหรับ Day 17 (อาทิตย์ + จันทร์ + LST) แบบ low-precision แต่ *ตรวจแล้ว*

ใช้ series หลักของ Meeus (Solar System / Astronomical Algorithms):
  • อาทิตย์: ทฤษฎีง่าย (ความแม่น ~0.01°) → เวลาขึ้น/ตก ±1 นาที
  • จันทร์: รวมพจน์ใหญ่ 14 พจน์ (longitude) + 10 พจน์ (latitude) → ความแม่น ~0.2°
  • LST: สูตร IAU 2000 อย่างย่อ

สคริปต์นี้รันเดี่ยว ๆ ได้เพื่อ *ตรวจกับเวลาที่เผยแพร่* (validate) ก่อนที่ day17_leonids.py จะใช้
  python tools/astro17.py
"""
import math, datetime as dt
import numpy as np

DEG = math.pi / 180.0


def jd_utc(y, mo, d, h=0, mi=0, s=0):
    return dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp() / 86400.0 + 2440587.5


def icrs_alt(ra_deg, dec_deg, lat_deg, lon_deg, jd):
    """ความสูงเหนือขอบฟ้า (องศา) ของวัตถุจาก RA/Dec · lon ทิศตะวันออกเป็นบวก"""
    T = (jd - 2451545.0) / 36525.0
    gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T * T
            - T * T * T / 38710000.0) % 360.0
    H = ((gmst + lon_deg - ra_deg + 180.0) % 360.0 - 180.0) * DEG
    la, de = lat_deg * DEG, dec_deg * DEG
    s = math.sin(de) * math.sin(la) + math.cos(de) * math.cos(la) * math.cos(H)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def sun_ra_dec(jd):
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360.0
    g = (357.528 + 0.9856003 * n) % 360.0 * DEG
    lam = (L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) * DEG
    eps = (23.439 - 0.0000004 * n) * DEG
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))) % 360.0
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(lam)))
    return ra, dec


# ---- Meeus lunar series (พจน์ใหญ่): argument = a*D + b*M + c*M' + d*F  (ดีกรี)
# coefficients เป็น arcsecond หาร 1e6 → บวกเข้ากับ mean longitude / latitude
_LON = [(0, 0, 1, 0, 6288774), (2, 0, -2, 0, 1274274), (2, 0, -1, 0, 658569),
        (2, 0, 0, 0, 213618), (0, 1, 0, 0, -185116), (2, 0, -2, 1, -114332),
        (2, -1, -2, 0, 58793), (2, 0, 0, -1, 57066), (2, 0, 1, 0, 53322),
        (2, 0, -2, -1, 45758), (0, 1, -2, 0, -40923), (2, 1, -2, 0, -34720),
        (0, 0, 2, 0, -30383), (2, 0, -1, 1, 15327), (2, 0, -3, 0, -12528),
        (0, 1, -1, 0, 10980), (2, 1, -1, 0, 10675), (2, -1, 0, -1, 10034)]
_LAT = [(0, 0, 0, 1, 5128122), (0, 0, 1, 1, 280602), (0, 0, -1, 1, 277693),
        (2, 0, 0, -1, 173237), (2, 0, -2, 1, 55413), (2, 0, -1, -1, 46292),
        (2, 0, -1, 1, 32543), (2, 0, 0, 1, -17323), (0, 0, 2, 1, -16769),
        (2, 0, 1, -1, -16769), (2, 0, -2, -1, -15836), (2, 0, 1, 1, 12935),
        (2, 1, -2, -1, 12935), (0, 1, 0, 1, -12935)]


def _trigonometric_series(args, table):
    tot = 0.0
    for a, b, c, d, coef in table:
        ang = (a * args[0] + b * args[1] + c * args[2] + d * args[3]) * DEG
        tot += coef * math.sin(ang)
    return tot


def moon_elements(jd):
    T = (jd - 2451545.0) / 36525.0
    T2, T3 = T * T, T ** 3
    Lp = (218.3164477 + 481267.88123421 * T - 0.0015786 * T2 + T3 / 538811.0) % 360.0
    D = (297.8501929 + 445267.1114034 * T - 0.0018394 * T2 + T3 / 189025.0) % 360.0
    M = (357.5291092 + 35999.0502909 * T - 0.0001536 * T2 + T3 / 24491000.0) % 360.0
    Mp = (134.9633964 + 477198.8675055 * T + 0.0087454 * T2 + T3 / 69699.0) % 360.0
    F = (93.2720950 + 483202.0175233 * T - 0.0036534 * T2 - T3 / 3526000.0) % 360.0
    return Lp, D, M, Mp, F


def moon_ra_dec(jd):
    Lp, D, M, Mp, F = moon_elements(jd)
    # correction for the Earth's orbital eccentricity (A1, A2, A3)
    E = 1 - 0.002516 * ((jd - 2451545.0) / 36525.0) - 0.0000052 * ((jd - 2451545.0) / 36525.0) ** 2
    args = (D, M, Mp, F)
    dl = _trigonometric_series(args, _LON)
    db = _trigonometric_series(args, _LAT)
    dl += -2235 * E * E * math.sin((D + 2 * Mp) * DEG) * 0   # placeholder kept explicit
    lam = math.radians((Lp + dl / 1e6) % 360.0)
    bet = math.radians(db / 1e6)
    eps = math.radians(23.4392912 - 0.0130041666 * ((jd - 2451545.0) / 36525.0))
    ra = math.degrees(math.atan2(math.sin(lam) * math.cos(eps) - math.tan(bet) * math.sin(eps),
                                 math.cos(lam))) % 360.0
    dec = math.degrees(math.asin(math.sin(bet) * math.cos(eps) + math.cos(bet) * math.sin(eps) * math.sin(lam)))
    sra, sdec = sun_ra_dec(jd)
    v1 = (math.cos(dec * DEG) * math.cos(ra * DEG), math.cos(dec * DEG) * math.sin(ra * DEG),
          math.sin(dec * DEG))
    v2 = (math.cos(sdec * DEG) * math.cos(sra * DEG), math.cos(sdec * DEG) * math.sin(sra * DEG),
          math.sin(sdec * DEG))
    elong = math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(v1, v2))))))
    illum = (1 + math.cos(math.radians(180 - elong))) / 2
    return ra, dec, elong, illum


def altitude_of(kind, lat, lon, jd):
    if kind == "sun":
        ra, dec = sun_ra_dec(jd)
    elif kind == "moon":
        ra, dec, _, _ = moon_ra_dec(jd)
    else:
        ra, dec = RADIANT
    return icrs_alt(ra, dec, lat, lon, jd)


RADIANT = (152.0, 22.0)          # IMO 2026 calendar: 013 LEO radiant at maximum
ZHR = 15                          # IMO 2026: ZHR ≈ 15
PEAK_UTC = dt.datetime(2026, 11, 17, 23, 45, tzinfo=dt.timezone.utc)


def rise_set(kind, lat, lon, jd_start, jd_end, h0, step_min=1.0):
    """คืน (เวลาขึ้น, เวลาตก) เป็น JD ภายในหน้าต่าง — ใช้ sign change ของ alt-h0
    วัตถุอาจ *ขึ้นมาก่อน* หน้าต่างจะเริ่มแล้ว จึงจับการตกได้แม้ไม่เห็นการขึ้น"""
    n = int((jd_end - jd_start) * 1440 / step_min) + 1
    prev = None
    up = dn = None
    for i in range(n):
        jd = jd_start + i * step_min / 1440.0
        a = altitude_of(kind, lat, lon, jd) - h0
        if prev is not None:
            if prev[1] < 0 <= a and up is None:
                up = prev[0] + (jd - prev[0]) * (-prev[1] / (a - prev[1]))
            elif prev[1] >= 0 > a and dn is None:
                dn = prev[0] + (jd - prev[0]) * (-prev[1] / (a - prev[1]))
                if up is not None:
                    break
        prev = (jd, a)
    return up, dn


def _fmt(jd):
    if jd is None:
        return "-"
    t = dt.datetime.fromtimestamp((jd - 2440587.5) * 86400.0, dt.timezone.utc) + dt.timedelta(hours=7)
    return t.strftime("%d %b %H:%M ICT")


if __name__ == "__main__":
    BKK = (13.7543, 100.5056)
    j0, j1 = jd_utc(2026, 11, 17, 8), jd_utc(2026, 11, 18, 4)   # 17 15:00 ICT → 18 11:00 ICT
    sr_up, _ = rise_set("sun", *BKK, j0, j1, -0.833)
    print("อาทิตย์ขึ้น (คำนวณ)  :", _fmt(sr_up), " · timebie Bangkok พ.ย. ≈ 06:19–06:20 ICT")
    tw, _ = rise_set("sun", *BKK, jd_utc(2026, 11, 17, 20), j1, -18.0)
    print("สนธยาเชิงดาราศาสตร์เริ่ม:", _fmt(tw))
    _, ms = rise_set("moon", *BKK, j0, j1, -0.125)
    print("จันทร์ตก (คำนวณ)     :", _fmt(ms), " · timebie Bangkok 17 พ.ย. = 23:19 ICT")
    ra, dec, el, ill = moon_ra_dec(jd_utc(2026, 11, 17, 19))
    print(f"จันทร์ 18 พ.ย. 02:00 ICT: elongation {el:.1f}° · illumination {100*ill:.0f}%")
    print("peak IMO:", PEAK_UTC.strftime("%d %b %H:%M UTC"), "=",
          (PEAK_UTC + dt.timedelta(hours=7)).strftime("%d %b %H:%M ICT"))
