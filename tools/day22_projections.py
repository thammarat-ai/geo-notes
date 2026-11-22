# -*- coding: utf-8 -*-
"""
Day 22 · Projections — "The different ways of representing a globe on a flat surface.
What does your favourite map projection say about you?"

โจทย์ (30daymapchallenge.com หัวข้อ Day 22 'Projections' อ่านสด 19:5x น. วันที่ 22 พ.ย. 2569
ลิงก์ในโจทย์ชี้ไปการ์ด xkcd #977 เรื่องการเลือก projection)

ซีรีส์นี้ใช้ Equidistant Conic ตัวเดียวมาตั้งแต่ Day 6 โดยไม่เคยถามว่า
"มันโกหกเราแค่ไหน" — วันนี้จึงตรวจ projection ที่ใช้จริง เทียบกับอีก 11 ตัว
ด้วยเกณฑ์ที่วัดได้หมด ไม่ใช่ความรู้สึก

เกณฑ์ต่อ 1 projection (ทั้งหมดเทียบกับ *geodesic บนทรงรี WGS84* ไม่เทียบกับ projection อื่น)
  1. area_TH      พื้นที่ไทยที่ projection รายงาน (ตร.กม.) เทียบ 514,453.7 ที่วัดด้วย Geod
  2. area_cells   ค่าเฉลี่ย/ต่ำสุด/สูงสุด ของอัตราส่วนพื้นที่เซลล์ H3 res 5 ทั้ง 1,738
                  เทียบ geodesic รายเซลล์ (นี่คือตัวเลขที่ Day 20 ใช้)
  3. tissot       Jacobian ของ projection ที่ 182 จุด (99 กริดครอบคลุมไทย + 83 จุดทะเบียน)
                  SVD ของแมทริกซ์ 2×2 ที่ฐานเป็นทิศตะวันออก/เหนือจริง →
                  a = scale ตามแกนยาว, b = scale ตามแกนสั้น, s = a·b (areal scale),
                  ω = 2·asin((a−b)/(a+b)) (ความบิดของมุม),
                  h, k = scale ตามแนวเมอริเดียน/ขนาน, และมุมระหว่างสองเส้นนั้น
                  (ครั้งแรกใช้ covariance ของวงกลม 180 ซี่ ซึ่งให้ ω = 179° ทุก projection
                   เพราะป้อน azimuth/distance สลับกันใน Geod.fwd — guard_variance จับได้)
  4. distance     คู่จุดทะเบียน 83 แห่ง = 3,403 คู่ · ระยะแบน ÷ ระยะ geodesic
  5. graticule    มุมระหว่างเมอริเดียนกับขนานที่จุดกริด (ควร 90° ถ้ารักษาAngle)
  6. azimuth      มุมทิศจากจุดกลางประเทศไป 83 จุด พิกัดแบน − geodesic
  7. world        พื้นที่กรีนแลนด์÷ไทย ที่ projection ให้เห็น (ความจริง 4.19 เท่า)

ของแถมที่ต้องทำ: แยกองค์ประกอบ "13%" ที่ Day 20 รายงานว่าเซลล์ใหญ่กว่าค่า nominal ของไลบรารี
  → ส่วนไหนคือ projection error · ส่วนไหนคือค่าเฉลี่ยทั่วโลกของ H3 ไม่ใช่ค่าเฉลี่ยแถวละติจูดเรา
"""
import os, sys, json, time, math, re
import numpy as np, pandas as pd, geopandas as gpd
import h3, pyproj
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Point, Polygon, LineString
from shapely.validation import make_valid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

T0 = time.time()
COLAB = not os.path.isdir(r"D:\GIS30Days\geo-notes")
WORK = os.getcwd() if COLAB else r"D:\GIS30Days"
REPO = os.path.join(WORK, "geo-notes")
FONTS = os.path.join(WORK, "cache", "day06")
OUT = os.path.join(WORK, "out", "day22")
os.makedirs(OUT, exist_ok=True)
if COLAB:
    # บน Colab: กัด repo มาทั้งลูก (มีทั้ง csv ทะเบียน + กริด Day 3) แล้วดึง
    # Natural Earth 10m countries + ฟอนต์ Trirong ที่สคริปต์นี้ต้องใช้
    import subprocess, zipfile, urllib.request
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/thammarat-ai/geo-notes"], check=True)
    os.makedirs(FONTS, exist_ok=True)
    import shutil as _sh
    for d in ("site/fonts", "assets/fonts", "fonts"):
        src = os.path.join(REPO, d)
        if os.path.isdir(src):
            for f in os.listdir(src):
                if f.endswith(".ttf"):
                    _sh.copy(os.path.join(src, f), os.path.join(FONTS, f))
    NE_ZIP = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
    if not os.path.exists(os.path.join(FONTS, "ne_10m_admin_0_countries.shp")):
        urllib.request.urlretrieve(NE_ZIP, "/tmp/ne.zip")
        with zipfile.ZipFile("/tmp/ne.zip") as z:
            z.extractall(FONTS)
for _f in sorted(os.listdir(FONTS)):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONTS, _f))
plt.rcParams["font.family"] = "Trirong"
_LOGF = open(os.path.join(OUT, "day22_run.log"), "w", encoding="utf-8")
def log(m):
    s = f"[{time.time()-T0:6.1f}s] {m}"
    print(s, flush=True)
    _LOGF.write(s + chr(10)); _LOGF.flush()

for _td in (os.path.join(WORK, "tools"), os.path.join(REPO, "tools")):
    if os.path.isdir(_td):
        sys.path.insert(0, _td)
        break
from guards import guard_crs, guard_area, guard_bbox, guard_variance, lint_thai

TXT = []
def T(s):
    TXT.append(s); return s

BG, INK, MUTED = "#0d1522", "#e8e3d3", "#9fb2c8"
ACCENT, WARM, COOL, TEAL, PURPLE = "#f5b942", "#f4566a", "#38bdf8", "#2dd4bf", "#c084fc"
GEOD = Geod(ellps="WGS84")

# ================================================== 1. ตัวแบบ: 12 projection
EQ_SERIES = ("+proj=eqdc +lat_0=14 +lon_0=101 +lat_1=8 +lat_2=20 "
             "+units=m +no_defs +ellps=WGS84")
PJ = [
    ("eqdc-ซีรีส์",  EQ_SERIES, "ตัวที่ซีรีส์นี้ใชามาตั้งแต่ Day 6"),
    ("lcc",           "+proj=lcc +lat_0=14 +lon_0=101 +lat_1=8 +lat_2=20 +units=m +no_defs +ellps=WGS84",
                      "Lambert Conformal Conic ขนานมาตรฐาน 8/20 — รักษา มุม"),
    ("albers",        "+proj=aea +lat_0=14 +lon_0=101 +lat_1=8 +lat_2=20 +units=m +no_defs +ellps=WGS84",
                      "Albers Equal-Area Conic ขนานมาตรฐาน 8/20 — รักษา พื้นที่"),
    ("utm47n",        "EPSG:32647", "UTM zone 47N (central meridian 99°E) — ค่ามาตรฐานของแผนที่ไทยตอนบน"),
    ("utm48n",        "EPSG:32648", "UTM zone 48N (central meridian 105°E) — เขตรอบนอกที่ไทยยื่นเข้าไป"),
    ("tmerc-101",     "+proj=tmerc +lat_0=14 +lon_0=101 +k_0=0.9996 +x_0=500000 +y_0=0 +units=m +no_defs +ellps=WGS84",
                      "Transverse Mercator ที่ลองจิจูดกลางของประเทศเราเอง"),
    ("webmerc",       "EPSG:3857",  "Pseudo-Mercator — ค่ามาตรฐานของเว็บ map ทั่วไป"),
    ("merc-elips",    "EPSG:3395",  "World Mercator บนทรงรี (ไม่ใช่ทรงกลม)"),
    ("aeqd",          "+proj=aeqd +lat_0=15.112 +lon_0=101.020 +units=m +no_defs +ellps=WGS84",
                      "Azimuthal Equidistant ที่จุดกลางของกริด — รักษาระยะ+ทิศจากจุดกลาง"),
    ("gall-peters",   "+proj=cea +lat_ts=44.136 +lon_0=101 +units=m +no_defs +ellps=WGS84",
                      "Cylindrical Equal-Area แบบ Gall–Peters (รักษาพื้นที่ทั้งโลก)"),
    ("eqc",           "+proj=eqc +lat_ts=0 +lon_0=101 +units=m +no_defs +ellps=WGS84",
                      "Equirectangular / Plate Carrée — แผนที่ที่ 'ไม่คิดอะไรเลย'"),
    ("natearth2",     "+proj=natearth +ellps=WGS84 +units=m", "Natural Earth II — compromise ทั้งโลก"),
]
NAMES = [p[0] for p in PJ]

# ================================================== 2. วัตถุดิบ
ne = gpd.read_file(os.path.join(FONTS, "ne_10m_admin_0_countries.shp"))
TH_raw = ne[ne.NAME == "Thailand"].geometry.iloc[0]
GL_raw = ne[ne.NAME == "Greenland"].geometry.iloc[0]
TH_raw = make_valid(TH_raw); GL_raw = make_valid(GL_raw)
area_th_geo = abs(GEOD.geometry_area_perimeter(TH_raw)[0]) / 1e6
area_gl_geo = abs(GEOD.geometry_area_perimeter(GL_raw)[0]) / 1e6
assert 500_000 < area_th_geo < 530_000, (
    f"พื้นที่ไทยจาก geodesic = {area_th_geo:,.0f} ตร.กม. อยู่นอกช่วง 5.0-5.3 แสน "
    "แปลว่า geometry หรือหน่วยพิกัดผิด (ไม่ใช่ข้อผิดพลาดของการเปรียบเทียบ)")
log(f"reference geodesic: ไทย {area_th_geo:,.1f} ตร.กม. · กรีนแลนด์ {area_gl_geo:,.1f} ตร.กม."
    f" · อัตราส่วนความจริง {area_gl_geo/area_th_geo:.2f} เท่า")

g3 = gpd.read_file(os.path.join(REPO, "posts/day03-polygons/day03_hex_polygons.geojson"))
assert len(g3) == 1738, f"กริด Day 3 ต้องมี 1,738 เซลล์ ได้ {len(g3)}"
cell_geo = np.array([abs(GEOD.geometry_area_perimeter(geom)[0]) / 1e6
                     for geom in g3.geometry.values])
h3_nominal = float(h3.average_hexagon_area(5, "km^2"))
h3_per_cell = np.array([h3.cell_area(c, "km^2") for c in g3.cell.values])
lat_of_cell = np.array([h3.cell_to_boundary(c)[0][0] for c in g3.cell.values])
log(f"เซลล์ 1,738 · geodesic เฉลี่ย {cell_geo.mean():.3f} · H3 cell_area() เฉลี่ย "
    f"{h3_per_cell.mean():.3f} · ค่าเฉลี่ยทั่วโลกของไลบรารี {h3_nominal:.3f}")

D = pd.read_csv(os.path.join(REPO, "data", "darksky_thailand.csv"))
D["name"] = D["name"].astype(str).str.strip()
P_ll = np.array([[float(r.longitude), float(r.latitude)] for _, r in D.iterrows()])
guard_bbox(P_ll, 96.5, 107.6, 4.3, 21.7, "darksky-points")

# กริดจุดสำหรับ Tissot: 99 จุดกริดครอบคลุมกล่องไทย + 83 จุดทะเบียน = 182 จุด ครอบคลุมกล่องไทย + 83 จุดทะเบียน
la = np.linspace(6.0, 20.5, 9); lo = np.linspace(97.3, 105.7, 11)
GRID_LL = np.array([[o, t] for o in lo for t in la])          # แถว = (lon, lat) เหมือน P_ll
TPTS = np.vstack([GRID_LL, P_ll])
assert len(TPTS) == 99 + 83, "จุด Tissot ต้องเป็น 99 กริด + 83 ทะเบียน"

# รัศมีของ indicatrix ที่ใช้ *วาด* (100 กม.) — ตัวเลข scale มาจาก Jacobian ไม่ใช้วงกลม
R_KM = 100.0

# ================================================== 3. ตัววัดต่อ projection
def make_tr(crs):
    c = CRS.from_user_input(crs)
    return c, Transformer.from_crs("EPSG:4326", c, always_xy=True)

def project(tr, xy):
    x, y = tr.transform(np.asarray(xy)[:, 0], np.asarray(xy)[:, 1])
    return np.column_stack([np.asarray(x), np.asarray(y)])

def project_geom(gdf, crs):
    return gdf.to_crs(crs)

def poly_area_km2(geom_m2):
    return abs(geom_m2.area) / 1e6

def pair_distances(X):
    dx = X[:, 0][:, None] - X[:, 0][None, :]
    dy = X[:, 1][:, None] - X[:, 1][None, :]
    Dm = np.hypot(dx, dy)
    iu = np.triu_indices(len(X), 1)
    return Dm[iu]

# geodesic มัธยมของคู่จุดทะเบียน (ค่านิ่ง ใช้ซ้ำทุก projection)
llon = P_ll[:, 0]; llat = P_ll[:, 1]
PAIR_IU = np.triu_indices(len(P_ll), 1)
lon1, lat1 = llon[PAIR_IU[0]], llat[PAIR_IU[0]]
lon2, lat2 = llon[PAIR_IU[1]], llat[PAIR_IU[1]]
_az12, _az21, geo_pair_m = GEOD.inv(lon1, lat1, lon2, lat2)
CEN_LON, CEN_LAT = 101.020, 15.112          # จุดกลางกริด Day 3 (คำนวณจาก centroid ของ 1,738 เซลล์)
geo_az_deg, _g2, _gd = GEOD.inv(np.full(len(llon), CEN_LON), np.full(len(llon), CEN_LAT),
                               llon, llat)
log(f"คู่จุดทะเบียน {len(geo_pair_m):,} คู่ · ระยะ geodesic มัธยฐาน "
    f"{np.median(geo_pair_m)/1000:,.1f} กม.")

def jac(tr, lon, lat, d=0.02):
    """Jacobian ของ projection ที่จุดหนึ่ง = แผนที่เชิงเส้นของ *พื้นผิวจริง* → กระดาษแบน
    คืน (x, y, เวกเตอร์ 1 ม.จริงทางตะวันออก, เวกเตอร์ 1 ม.จริงทางเหนือ) ในหน่วยเมตรบนแผนที่
    ใช้ผลต่างกึ่งกลาง (central difference) เพราะเส้นเดียวด้านเดียวมี bias จากความโค้ง"""
    p0 = project(tr, np.array([[lon, lat]]))[0]
    pn = project(tr, np.array([[lon, lat + d]]))[0]
    ps = project(tr, np.array([[lon, lat - d]]))[0]
    pe = project(tr, np.array([[lon + d, lat]]))[0]
    pw = project(tr, np.array([[lon - d, lat]]))[0]
    lenN = abs(GEOD.line_length([lon, lon], [lat - d, lat + d]))
    lenE = abs(GEOD.line_length([lon - d, lon + d], [lat, lat]))
    return p0, (pe - pw) / lenE, (pn - ps) / lenN


def tissot(tr, lon, lat):
    """Tissot's indicatrix จาก Jacobian แท้ (a,b = scale ตามแกนหลัก, ω = ความบิดมุม)
    แบบเดิมที่ใช้ cov ของวงกลม 180 ซี่ให้ค่า ω = 0.38° กับ Mercator ซึ่ง *ต้อง* เป็น 0
    (บั๊กที่ guard_variance จับได้: ทุก projection ออก ω ใกล้เคียง 180° ทั้งหมด)"""
    p0, dE, dN = jac(tr, lon, lat)
    M = np.column_stack([dE, dN])                  # ฐานตะวันออก/เหนือเป็น orthonormal
    U, S, Vt = np.linalg.svd(M)
    a_, b_ = float(S[0]), float(S[1])
    om = math.degrees(2 * math.asin(max(0.0, min(1.0, (a_ - b_) / (a_ + b_))))) if (a_ + b_) else np.nan
    cosC = (dE @ dN) / (np.linalg.norm(dE) * np.linalg.norm(dN))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cosC))))
    return dict(x=p0[0], y=p0[1], a=a_, b=b_, areal=a_ * b_, omega=om,
                grat=ang, h=float(np.linalg.norm(dN)), k=float(np.linalg.norm(dE)),
                rot=math.degrees(math.atan2(U[1, 0], U[0, 0])))

rows, tissot_rows, CAREA = [], [], {}
gth4326 = gpd.GeoDataFrame(geometry=[TH_raw], crs="EPSG:4326")
ggl4326 = gpd.GeoDataFrame(geometry=[GL_raw], crs="EPSG:4326")
cells4326 = g3.copy()

for nm, crs, note in PJ:
    t1 = time.time()
    C, tr = make_tr(crs)
    thp = gth4326.to_crs(C).geometry.iloc[0]
    glp = ggl4326.to_crs(C).geometry.iloc[0]
    a_th = poly_area_km2(thp)
    a_gl = poly_area_km2(glp)
    # พื้นที่เซลล์ทั้ง 1,738 ใน projection นี้
    cellv = cells4326.to_crs(C).geometry.values
    a_cell = np.array([poly_area_km2(v) for v in cellv])
    ratio_cell = a_cell / cell_geo
    CAREA[nm] = a_cell
    # Tissot: Jacobian แท้ทุกจุด (182 จุด = 99 กริด + 83 ทะเบียน)
    TT = [tissot(tr, lo_, la_) for lo_, la_ in TPTS]
    A_max = np.array([t["a"] for t in TT]); B_max = np.array([t["b"] for t in TT])
    OM = np.array([t["omega"] for t in TT]); SS = np.array([t["areal"] for t in TT])
    ANG = np.array([t["grat"] for t in TT])
    NS = np.array([t["h"] for t in TT]); EW = np.array([t["k"] for t in TT])
    # ระยะคู่จุดทะเบียน
    XS = project(tr, P_ll)
    plan = pair_distances(XS)
    drel = plan / geo_pair_m
    # มุมทิศบนแผนที่ (สิ่งที่คนวัดด้วยไม้โปรแทรกเตอร์บนกระดาษ) เทียบกับ azimuth จริง
    cxy = project(tr, np.array([[CEN_LON, CEN_LAT]]))[0]
    sxy = project(tr, P_ll)
    map_az = np.degrees(np.arctan2(sxy[:, 0] - cxy[0], sxy[:, 1] - cxy[1])) % 360.0
    d_az = np.abs((map_az - geo_az_deg + 180) % 360 - 180)
    rows.append(dict(name=nm, crs=crs if crs.startswith("+") else crs, note=note,
                     area_th_km2=round(a_th, 1),
                     area_err_pct=round((a_th / area_th_geo - 1) * 100, 4),
                     cell_mean_km2=round(float(a_cell.mean()), 3),
                     cell_ratio_mean=round(float(ratio_cell.mean()), 5),
                     cell_ratio_min=round(float(ratio_cell.min()), 5),
                     cell_ratio_max=round(float(ratio_cell.max()), 5),
                     tissot_a_max=round(float(A_max.max()), 5),
                     tissot_b_min=round(float(B_max.min()), 5),
                     areal_scale_max=round(float(SS.max()), 5),
                     areal_scale_min=round(float(SS.min()), 5),
                     omega_mean_deg=round(float(np.nanmean(OM)), 4),
                     omega_max_deg=round(float(np.nanmax(OM)), 4),
                     dist_ratio_mean=round(float(drel.mean()), 5),
                     dist_ratio_min=round(float(drel.min()), 5),
                     dist_ratio_max=round(float(drel.max()), 5),
                     dist_abs_err_pct_mean=round(float(np.mean(np.abs(drel - 1)) * 100), 4),
                     dist_abs_err_pct_max=round(float(np.max(np.abs(drel - 1)) * 100), 4),
                     az_err_mean_deg=round(float(np.mean(d_az)), 4),
                     az_err_max_deg=round(float(np.max(d_az)), 4),
                     ns_scale_mean=round(float(NS.mean()), 5),
                     ns_scale_dev_max=round(float(np.max(np.abs(NS - 1))), 5),
                     ew_scale_mean=round(float(EW.mean()), 5),
                     ew_scale_dev_max=round(float(np.max(np.abs(EW - 1))), 5),
                     grat_angle_mean=round(float(np.nanmean(ANG)), 4),
                     grat_angle_dev_max=round(float(np.nanmax(np.abs(ANG - 90))), 4),
                     gl_th_ratio=round(a_gl / a_th, 4),
                     secs=round(time.time() - t1, 2)))
    Rm = R_KM * 1000.0
    for i, t in enumerate(TT):
        tissot_rows.append(dict(proj=nm, lon=round(float(TPTS[i][0]), 5),
                                lat=round(float(TPTS[i][1]), 5),
                                grid=bool(i < len(GRID_LL)), idx=i,
                                a=round(t["a"], 5), b=round(t["b"], 5),
                                areal=round(t["areal"], 5), omega_deg=round(t["omega"], 4),
                                grat_deg=round(t["grat"], 4), h=round(t["h"], 5),
                                k=round(t["k"], 5),
                                x_m=round(float(t["x"]), 2), y_m=round(float(t["y"]), 2),
                                ax_m=round(t["a"] * Rm, 2), ay_m=round(t["b"] * Rm, 2),
                                rot_deg=round(float(t["rot"]), 3)))
    log(f"  {nm:12} ไทย {a_th:,.1f} ตร.กม. ({(a_th/area_th_geo-1)*100:+.3f}%) · "
        f"เซลล์ {a_cell.mean():.3f} ({ratio_cell.mean()-1:+.3%}) · ω สูงสุด "
        f"{np.nanmax(OM):.2f}° · ระยะผิดเฉลี่ย {np.mean(np.abs(drel-1))*100:.3f}% · "
        f"ทิศผิดสูงสุด {np.max(d_az):.2f}° [{time.time()-t1:.1f}s]")

PR = pd.DataFrame(rows)
TI = pd.DataFrame(tissot_rows)
assert len(PR) == 12, "ต้องมี 12 projection"
assert PR.area_err_pct.abs().max() < 1500, "ค่าพื้นที่เพี้ยนเกินเหตุ — ตรวจหน่วย/CRS"
guard_variance(PR.omega_max_deg.values, "omega-across-projections")

# ================================================== 4. แยกองค์ประกอบ "13%" ของ Day 20
# Day 20 รายงานว่าเซลล์ res 5 ในไทยวัดได้ 285.884 ตร.กม. ใหญ่กว่าค่า nominal ของไลบรารี
# (252.904) ราว 13% และทิ้งคำถามไว้ว่าเพราะอะไร — วันนี้มีเครื่องมือแยกองค์ประกอบแล้ว
d20 = json.load(open(os.path.join(REPO, "posts/day20-hexagons/day20_stats.json"), encoding="utf-8"))
A20 = d20["aggregates"]
day20_mean = float(A20["measured_area_mean"])
day20_min = float(A20["measured_area_min"])
day20_max = float(A20["measured_area_max"])
ours = PR[PR.name == "eqdc-ซีรีส์"].iloc[0]
assert abs(ours.cell_mean_km2 - day20_mean) < 0.05, (
    f"วันนี้วัด eqdc ได้ {ours.cell_mean_km2:.3f} แต่ Day 20 รายงาน {day20_mean:.3f} — "
    "projection หรือไฟล์กริดเปลี่ยนไป ต้องหยุดตรวจ")
g_geo, g_h3, g_nom = float(cell_geo.mean()), float(h3_per_cell.mean()), float(h3_nominal)
DEC = dict(day20_eqdc_mean=day20_mean, geodesic_mean=round(g_geo, 3),
           h3_sphere_mean=round(g_h3, 3), h3_global_nominal=round(g_nom, 3),
           total_gap_pct=round((day20_mean / g_nom - 1) * 100, 3),
           projection_part_pct=round((day20_mean / g_geo - 1) * 100, 4),
           sphere_vs_ellipsoid_part_pct=round((g_geo / g_h3 - 1) * 100, 3),
           latitude_part_pct=round((g_h3 / g_nom - 1) * 100, 3),
           product_check_pct=round(((day20_mean / g_geo) * (g_geo / g_h3) * (g_h3 / g_nom) - 1) * 100, 3),
           h3_cell_area_min=round(float(h3_per_cell.min()), 3),
           h3_cell_area_max=round(float(h3_per_cell.max()), 3),
           spearman_cellarea_vs_latitude=round(float(pd.Series(h3_per_cell).corr(
               pd.Series(lat_of_cell), method="spearman")), 4),
           spearman_abs=round(abs(float(pd.Series(h3_per_cell).corr(
               pd.Series(lat_of_cell), method="spearman"))), 4),
           nominal_minus_ours_pct=round((g_geo / g_nom - 1) * 100, 3))
log(f"13% ของ Day 20 = projection {DEC['projection_part_pct']:+.3f}% + "
    f"ทรงกลมvsทรงรี {DEC['sphere_vs_ellipsoid_part_pct']:+.3f}% + "
    f"ละติจูด/H3 {DEC['latitude_part_pct']:+.3f}% (คูณกันได้ "
    f"{DEC['product_check_pct']:+.3f}% ตรงกับ {DEC['total_gap_pct']:+.3f}%)")

# ================================================== 5. ภาพ
TH_G = gpd.GeoDataFrame(geometry=[TH_raw], crs="EPSG:4326")
GL_G = gpd.GeoDataFrame(geometry=[GL_raw], crs="EPSG:4326")
PTS_G = gpd.GeoDataFrame(geometry=gpd.points_from_xy(P_ll[:, 0], P_ll[:, 1]), crs="EPSG:4326")

def panel_axes(C, data, pad=0.08):
    """ซูมเข้าหาขอบเขตไทยใน projection นั้น (ไม่งั้น projection ทั้งโลกทำให้ไทยเล็กเท่าจุด)"""
    b = data.total_bounds
    dx, dy = (b[2] - b[0]) * pad, (b[3] - b[1]) * pad
    return b[0] - dx, b[2] + dx, b[1] - dy, b[3] + dy

def draw_hero():
    fig = plt.figure(figsize=(15.8, 11.6), dpi=140)
    fig.patch.set_facecolor(BG)
    gs = GridSpec(4, 3, figure=fig, hspace=0.46, wspace=0.14)
    for i, r in PR.reset_index().iterrows():
        ax = fig.add_subplot(gs[i // 3, i % 3]); ax.set_facecolor(BG)
        C = CRS.from_user_input(r.crs)
        thp_ = TH_G.to_crs(C); cellp_ = cells4326.to_crs(C); ptsp_ = PTS_G.to_crs(C)
        x0, x1, y0, y1 = panel_axes(C, thp_)
        thp_.plot(ax=ax, color="#1b2a3d", edgecolor="#6f8bab", lw=0.9, zorder=3)
        cellp_.plot(ax=ax, facecolor=TEAL, alpha=0.10, edgecolor="#2f6f68", lw=0.2, zorder=2)
        ax.scatter(ptsp_.geometry.x, ptsp_.geometry.y, s=6, color=WARM, zorder=4,
                   edgecolor=BG, lw=0.3)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")       # small multiples: เลขเมตรไร้ประโยชน์
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ours_ = r["name"] == "eqdc-ซีรีส์"
        for sp in ax.spines.values():
            sp.set_color(ACCENT if ours_ else "#33415e"); sp.set_linewidth(2.2 if ours_ else 0.8)
        ax.set_title(T(f"{r['name']}\nพื้นที่ {r.area_err_pct:+.3f}% · "
                       f"มุมบิดสูงสุด {r.omega_max_deg:.2f} องศา · "
                       f"ระยะผิดเฉลี่ย {r.dist_abs_err_pct_mean:.3f}%"),
                     color=ACCENT if ours_ else INK, fontsize=9.8)
    fig.suptitle(T("12 วิธีแบนโลกให้ไทย · กริด H3 เดิม 1,738 เซลล์ + จุดทะเบียน 83 แห่ง\n"
                   "กรอบเหลือง = projection ที่ซีรีส์นี้ใชามาตั้งแต่ Day 6"),
                 color=INK, fontsize=15, y=0.995)
    p = os.path.join(OUT, "day22_twelve.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    log(f"เขียน {p}")
    return p

def draw_tissot():
    picks = ["eqdc-ซีรีส์", "lcc", "webmerc", "gall-peters"]
    fig = plt.figure(figsize=(15.2, 7.4), dpi=140)
    fig.patch.set_facecolor(BG)
    for n, nm in enumerate(picks):
        ax = fig.add_subplot(2, 2, n + 1); ax.set_facecolor(BG)
        crs_ = PR.loc[PR.name == nm, "crs"].iloc[0]
        C = CRS.from_user_input(crs_)
        thp_ = TH_G.to_crs(C)
        x0, x1, y0, y1 = panel_axes(C, thp_, pad=0.14)
        thp_.plot(ax=ax, color="#1b2a3d", edgecolor="#6f8bab", lw=0.9, zorder=2)
        sub = TI[(TI.proj == nm) & TI.grid]
        for _, t in sub.iterrows():
            if not np.isfinite(t.ax_m) or not np.isfinite(t.ay_m):
                continue
            col = WARM if t.omega_deg > 5 else (ACCENT if t.omega_deg > 1.5 else TEAL)
            ax.add_patch(Ellipse((t.x_m, t.y_m), 2 * t.ax_m, 2 * t.ay_m, angle=t.rot_deg,
                                 facecolor=col, alpha=0.28, edgecolor=col, lw=0.7, zorder=3))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        for sp in ax.spines.values(): sp.set_color("#33415e")
        r = PR[PR.name == nm].iloc[0]
        ax.set_title(T(nm), color=INK, fontsize=11.5)
        ax.text(0.5, -0.085, T(f"วงกลมจริงรัศมี {R_KM:.0f} กม. · มุมบิด มัธยฐาน "
                               f"{r.omega_mean_deg:.2f}° สูงสุด {r.omega_max_deg:.2f}° · "
                               f"areal scale {r.areal_scale_min:.3f}–{r.areal_scale_max:.3f}"),
                transform=ax.transAxes, ha="center", color=MUTED, fontsize=8.8)
    fig.suptitle(T("Tissot's indicatrix: วงกลมบนโลกกลายเป็นอะไรบนกระดาษ"),
                 color=INK, fontsize=15, y=0.995)
    fig.subplots_adjust(bottom=0.14)
    p = os.path.join(OUT, "day22_tissot.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    log(f"เขียน {p}")
    return p

# คู่ที่ค่าใกล้กันมาก (tmerc-101 / utm47n / aeqd) ต้องเลื่อนป้ายออก ไม่งั้นทับกัน
OFFSETS = {"tmerc-101": (-52, 10), "utm47n": (-8, -16), "aeqd": (-16, 10),
           "albers": (12, 5), "eqdc-ซีรีส์": (13, 8), "lcc": (14, -13),
           "utm48n": (12, 10), "gall-peters": (-84, -14), "natearth2": (12, -14),
           "webmerc": (-70, 10), "merc-elips": (-74, -13), "eqc": (12, 6)}

def draw_tradeoff():
    fig = plt.figure(figsize=(16.0, 6.4), dpi=140)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.055, 0.13, 0.27, 0.74]); ax.set_facecolor("#111c2b")
    ax.set_xlim(0.008, 42); ax.set_ylim(0.005, 80)
    for _, r in PR.iterrows():
        ours_ = r["name"] == "eqdc-ซีรีส์"
        ax.scatter(r.dist_abs_err_pct_mean, max(r.omega_max_deg, 0.01), s=190 if ours_ else 90,
                   color=ACCENT if ours_ else COOL, marker="*" if ours_ else "o",
                   edgecolor=BG, lw=0.6, zorder=4)
        ax.annotate(str(r["name"]), (r.dist_abs_err_pct_mean, max(r.omega_max_deg, 0.01)),
                    textcoords="offset points",
                    xytext=OFFSETS.get(str(r["name"]), (6, 4)), color=MUTED, fontsize=7.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(T("ความยาวเส้นทางผิดเฉลี่ย (%) · คู่จุด 3,403 คู่"), color=INK)
    ax.set_ylabel(T("มุมบิดสูงสุด (องศา, สเกล log)"), color=INK)
    ax.set_title(T("ยิ่งซ้ายล่างยิ่งดี\n(แต่ไม่มีใครถึงมุมล่างซ้ายพร้อมกัน)"), color=INK, fontsize=11.5)
    ax.grid(color="#26364f", lw=0.5, alpha=0.5, which="both")
    ax.tick_params(colors=INK, labelsize=8.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax2 = fig.add_axes([0.395, 0.13, 0.27, 0.74]); ax2.set_facecolor("#111c2b")
    srt = PR.sort_values("area_err_pct", key=np.abs)
    ax2.barh(np.arange(len(srt)), srt.area_err_pct.values,
             color=[ACCENT if n == "eqdc-ซีรีส์" else (TEAL if abs(v) < 0.5 else WARM)
                    for n, v in zip(srt.name, srt.area_err_pct)], height=0.68)
    ax2.set_yticks(np.arange(len(srt)))
    ax2.set_yticklabels([str(n) for n in srt.name], color=INK, fontsize=8.2)
    for i, v in enumerate(srt.area_err_pct.values):
        ax2.text(v + (0.25 if v >= 0 else -0.25), i, f"{v:+.2f}", color=INK, fontsize=8,
                 va="center", ha="left" if v >= 0 else "right")
    ax2.axvline(0, color=MUTED, lw=0.8)
    ax2.set_xlabel(T("พื้นที่ไทยที่ projection รายงาน ผิดจาก geodesic (%)"), color=INK)
    ax2.set_title(T("ใครโกหกเรื่องขนาดประเทศ?\nเขียว = ผิดต่ำกว่า 0.5% แดง = เกินนั้น"),
                  color=INK, fontsize=11.5)
    ax2.grid(color="#26364f", lw=0.5, alpha=0.5, axis="x")
    ax2.tick_params(colors=INK, labelsize=8.5)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)

    ax3 = fig.add_axes([0.735, 0.13, 0.245, 0.74]); ax3.set_facecolor("#111c2b")
    truth = area_gl_geo / area_th_geo
    s3 = PR.sort_values("gl_th_ratio")
    ax3.barh(np.arange(len(s3)), s3.gl_th_ratio.values,
             color=[PURPLE if v < 8 else WARM for v in s3.gl_th_ratio], height=0.68)
    ax3.axvline(truth, color=INK, ls="--", lw=1.2)
    ax3.text(truth + 0.7, 0.3, T(f"ความจริง {truth:.2f} เท่า"), color=INK, fontsize=9)
    ax3.set_yticks(np.arange(len(s3)))
    ax3.set_yticklabels([f"{n}" for n in s3.name], color=INK, fontsize=8.2)
    for i, v in enumerate(s3.gl_th_ratio.values):
        ax3.text(v + 0.6, i, f"{v:.1f}", color=INK, fontsize=8, va="center")
    ax3.set_xlabel(T("กรีนแลนด์ ÷ ไทย ที่ projection ให้เห็น"), color=INK)
    ax3.set_title(T("กับดักคลาสสิก (xkcd #977):\nprojection ระดับภูมิภาคก็โป้ปดข้ามทวีปได้"),
                  color=INK, fontsize=11.5)
    ax3.grid(color="#26364f", lw=0.5, alpha=0.5, axis="x")
    ax3.tick_params(colors=INK, labelsize=8.5)
    ax3.set_xlim(0, max(PR.gl_th_ratio.max() * 1.12, truth * 1.4))
    for s in ("top", "right"): ax3.spines[s].set_visible(False)
    fig.suptitle(T("เลือก projection = เลือกว่าจะยอมโกหกเรื่องอะไร"), color=INK,
                 fontsize=14.5, y=0.995)
    p = os.path.join(OUT, "day22_tradeoff.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    log(f"เขียน {p}")
    return p

# ================================================== 6. "projection ของคุณบอกอะไรเกี่ยวกับคุณ"
crit = {"area": ("area_err_pct", lambda v: abs(v)),
        "angular": ("omega_max_deg", lambda v: v),
        "distance": ("dist_abs_err_pct_mean", lambda v: v),
        "azimuth": ("az_err_max_deg", lambda v: v),
        "meridian_scale": ("ns_scale_dev_max", lambda v: v)}
RK = {}
for k, (col, f) in crit.items():
    vals = np.array([f(v) for v in PR[col].values])
    order = np.argsort(vals)
    rank = np.empty(len(PR), int); rank[order] = np.arange(1, len(PR) + 1)
    RK[k] = list(rank)
RK = pd.DataFrame(RK, index=PR.name)
RK["rank_sum"] = RK.sum(axis=1)
OURS_NAME = "eqdc-ซีรีส์"
RANKS = {k: int(RK.loc[OURS_NAME, k]) for k in crit}
RANKS["sum"] = int(RK.loc[OURS_NAME, "rank_sum"])
WINNERS = {k: str(RK[k].idxmin()) for k in crit}
BEST_ALL = str(RK["rank_sum"].idxmin())
log(f"อันดับของ {OURS_NAME}: {RANKS} · ผู้ชนะรายเกณฑ์ {WINNERS} · รวมอันดับดีที่สุด {BEST_ALL}")

# ================================================== 7. artifact
P1, P2, P3 = draw_hero(), draw_tissot(), draw_tradeoff()
PR.to_csv(os.path.join(OUT, "day22_projections.csv"), index=False)
TI.to_csv(os.path.join(OUT, "day22_tissot.csv"), index=False)
RK.to_csv(os.path.join(OUT, "day22_ranks.csv"))
CELLK = ["eqdc-ซีรีส์", "albers", "utm47n", "webmerc", "gall-peters", "aeqd", "lcc"]
cells_out = pd.DataFrame({"cell": g3.cell.values, "province": g3.province.values,
                          "geodesic_km2": np.round(cell_geo, 4),
                          "h3_sphere_km2": np.round(h3_per_cell, 4),
                          "cell_lat": np.round(lat_of_cell, 4)})
for nm in CELLK:
    cells_out[nm] = np.round(CAREA[nm], 4)
cells_out["eqdc_over_geodesic"] = np.round(cells_out["eqdc-ซีรีส์"] / cells_out.geodesic_km2, 5)
cells_out.to_csv(os.path.join(OUT, "day22_cells.csv"), index=False)

AGG = dict(projections=int(len(PR)), grid_cells=int(len(g3)), tissot_points=int(len(TPTS)),
           site_pairs=int(len(geo_pair_m)),
           area_th_geodesic_km2=round(area_th_geo, 1),
           area_th_eqdc_km2=float(ours.area_th_km2),
           area_th_eqdc_err_pct=float(ours.area_err_pct),
           area_gl_geodesic_km2=round(area_gl_geo, 1),
           gl_th_true_ratio=round(area_gl_geo / area_th_geo, 4),
           gl_th_ratio_min=float(PR.gl_th_ratio.min()), gl_th_ratio_max=float(PR.gl_th_ratio.max()),
           gl_th_ratio_webmerc=float(PR.loc[PR.name == "webmerc", "gl_th_ratio"].iloc[0]),
           gl_th_ratio_eqdc=float(ours.gl_th_ratio),
           cell_geodesic_mean=round(g_geo, 3), cell_geodesic_min=round(float(cell_geo.min()), 3),
           cell_geodesic_max=round(float(cell_geo.max()), 3),
           cell_eqdc_mean=float(ours.cell_mean_km2),
           day20_reported_mean=day20_mean, day20_reported_min=day20_min, day20_reported_max=day20_max,
           day20_reproduced=bool(abs(ours.cell_mean_km2 - day20_mean) < 0.05),
           area_err_min_pct=float(PR.area_err_pct.min()), area_err_max_pct=float(PR.area_err_pct.max()),
           area_err_abs_median=float(PR.area_err_pct.abs().median()),
           projections_within_half_pct=int((PR.area_err_pct.abs() < 0.5).sum()),
           omega_zero=int((PR.omega_max_deg < 0.005).sum()),
           omega_max_overall=float(PR.omega_max_deg.max()),
           dist_err_best=float(PR.dist_abs_err_pct_mean.min()),
           dist_err_worst=float(PR.dist_abs_err_pct_mean.max()),
           ns_scale_perfect=int((PR.ns_scale_dev_max < 1e-4).sum()),
           ranks=RANKS, winners=WINNERS, best_overall=BEST_ALL,
    azimuth_full_circle_deg=360,
           median_pair_geodesic_km=round(float(np.median(geo_pair_m) / 1000.0), 1))
STATS = dict(
    theme="The different ways of representing a globe on a flat surface. "
          "What does your favourite map projection say about you?",
    theme_name="Projections",
    theme_source="30daymapchallenge.com หัวข้อ Day 22 'Projections' (อ่านสด 19:50 น. วันที่ 22 พ.ย. 2569; "
                 "ลิงก์ในโจทย์คือการ์ด xkcd #977)",
    theme_fetch_time=dict(hour=19, minute=50, day=22, month=11, year=2569),
    generated_local=time.strftime("%Y-%m-%d %H:%M:%S"),
    question="projection ที่ซีรีส์นี้ใช้มา 16 วัน (Equidistant Conic ขนานมาตรฐาน 8/20) "
             "โกหกเราเรื่องพื้นที่ มุม ระยะ และทิศแค่ไหน เทียบกับอีก 11 วิธี",
    reference=dict(basis="pyproj Geod(ellps=WGS84) บนทรงรี ไม่ใช่ทรงกลม · ทุกตัวเลขเทียบกับ "
                         "geodesic ไม่ใช่เทียบกับ projection อื่น",
                   library="PROJ " + pyproj.proj_version_str + " · pyproj " + pyproj.__version__,
                   thailand_area_km2=round(area_th_geo, 1),
                   greenland_area_km2=round(area_gl_geo, 1),
                   cell_area_mean_km2=round(g_geo, 3),
                   source_geometry="Natural Earth 10m admin_0_countries + กริด H3 res 5 จาก Day 3"),
    projections=PR.to_dict("records"),
    ranks=RK.reset_index().rename(columns={"index": "projection"}).to_dict("records"),
    external_reference=dict(
        thailand_area_official_km2=513120.0,
        source="สารานุกรมไทยฉบับ Wikipedia (th) ช่องข้อมูลประเทศ: พื้นที่รวม 513,120 ตร.กม. "
               " fetched 22 พ.ย. 2569",
        geodesic_vs_official_pct=round((area_th_geo / 513120.0 - 1) * 100, 4),
        eqdc_vs_official_pct=round((float(ours.area_th_km2) / 513120.0 - 1) * 100, 4),
        note="geometry ของ Natural Earth 10m ให้ตัวเลขใหญ่กว่าตัวเลขทางราชการ "
             "จึงห้ามเอาผลต่างของ projection ในโพสต์นี้ไปเทียบกับตัวเลขราชการ"),
    decomposition_of_day20=DEC,
    epsg_used=[32647, 32648, 3395, 3857],
    citation=dict(xkcd_card=977, url="https://xkcd.com/977/",
                  note="การ์ดที่โจทย์ลิงก์ถึง ใช้เป็นบริบทของคำว่า 'ตาเท่ากันไม่เท่ากับใหญ่เท่ากัน'"),
    grid_points=int(len(GRID_LL)),
    tissot=dict(points=int(len(TPTS)), grid_rows=int(len(la)), grid_cols=int(len(lo)),
                rows_in_csv=int(len(TI)),
                radius_km=R_KM, method="Jacobian (central difference 0.02°) + SVD",
                note="conformal ทุกตัว (lcc/utm47n/utm48n/tmerc/merc-elips) ให้ ω < 0.005° จริงตามทฤษฎี "
                     "ยกเว้น webmerc ที่ ω เล็กน้อยเพราะ EPSG:3857 ท่องโลกเป็นทรงกลม "
                     "ขณะที่ค่าจริงที่เราวัดเป็นทรงรี WGS84"),
    aggregates=AGG,
    metric_definitions={
        "area_err_pct": "(พื้นที่ polygon ใน projection ÷ พื้นที่ geodesic บน WGS84 − 1) × 100",
        "cell_ratio_mean/min/max": "อัตราส่วนพื้นที่เซลล์ H3 res 5 ใน projection ต่อค่า geodesic "
                                   "ของเซลล์เดียวกัน (1,738 เซลล์)",
        "tissot_a_max / tissot_b_min": "scale ตามแกนยาว/แกนสั้นของ Tissot's indicatrix "
                                       "เหนือ 182 จุด (1 = ไม่ยืด)",
        "areal_scale_max/min": "a·b = พื้นที่วงกลมหลังแปลง ÷ พื้นที่จริง (1 = รักษาพื้นที่)",
        "omega_mean_deg / omega_max_deg": "2·asin((a−b)/(a+b)) = มุมบิดเบือนสูงสุดของ indicatrix "
                                          "(0° = รักษาทุกมุม)",
        "h / k": "scale ตามแนวเมอริเดียน / ขนาน — คำสัญญาของคำว่า 'equidistant' วัดตรงนี้",
        "ns_scale_dev_max": "ค่าสูงสุดของ |h−1| บน 182 จุด = 0 เมื่อรักษา *ระยะตามเหนือ-ใต้* ทุกจุดจริง",
        "dist_abs_err_pct_mean": "ค่าเฉลี่ย |ระยะบนแผนที่ ÷ ระยะ geodesic − 1| ของคู่จุดทะเบียน "
                                 f"{len(geo_pair_m):,} คู่",
        "az_err_max_deg": "ความต่างสูงสุดระหว่าง azimuth บนแผนที่กับ azimuth จริง "
                          f"(จากจุดกลางกริด {CEN_LAT:.3f}°N {CEN_LON:.3f}°E ไป 83 จุด)",
        "gl_th_ratio": "พื้นที่กรีนแลนด์ ÷ พื้นที่ไทย ที่ projection นั้นรายงาน "
                       f"(ความจริง {area_gl_geo/area_th_geo:.4f}) — วัดได้ความหมายเฉพาะ "
                       "projection ที่ครอบคลุมทั้งโลก",
        "ranks": "อันดับ 1 = ดีที่สุดในเกณฑ์นั้น (12 projection)"},
    figures=["day22_twelve.png", "day22_tissot.png", "day22_tradeoff.png"],
    cross_check_day20=dict(day20_mean_cell_area=day20_mean,
                           today_eqdc_mean_cell_area=round(float(ours.cell_mean_km2), 3),
                           reproduced=bool(abs(ours.cell_mean_km2 - day20_mean) < 0.05),
                           day20_area_th_eqdc=512593.1,
                           today_area_th_eqdc=float(ours.area_th_km2),
                           claim="Day 20 บอกว่าเซลล์ res 5 ในไทยวัดได้ใหญ่กว่าค่า nominal ของ "
                                 "ไลบรารีราว 13% แต่ยังไม่ตอบว่า 13% นั้นมาจากอะไร"),
    bug_found=dict(symptom="ω ของทุก projection ออกมาเป็น ~179.8° — Mercator และ Lambert ที่ "
                           "อ้างว่ารักษามุมต้องได้ 0°",
                   caught_by="guard_variance ใน guards.py — ฟ้องว่าค่า 12 ตัวกระจายกัน "
                             "เพียง 0.027 (< threshold 1.0) ทั้งที่ควรต่างกันเป็นสิบองศา",
                   observed_omega_wrong_deg=179.8, observed_spread_deg=0.027,
                   true_omega_after_fix=round(float(PR.loc[PR.name == "lcc", "omega_max_deg"].iloc[0]), 5),
                   cause="เรียก Geod.fwd(lons, lats, azimuths, distances) โดยสลับลำดับสอง argument "
                         "สุดท้าย 'วงกลมรัศมี 100 กม.' จึงกลายเป็นกลุ่มจุดรัศมี ~6 เมตร "
                         "ellipse บีบเป็นเส้น → b ≈ 0 → ω ≈ 180° ทุกตัว",
                   fix="เลิกวัดจากวงกลม: คำนวณ Jacobian ด้วย central difference 0.02° แล้ว SVD "
                       "(ตรงนิยาม textbook ของ Tissot และไม่ขึ้นกับจำนวนซี่ของวงกลม)",
                   second_bug="guard ตัวเดิมยังจับ case ที่สองไม่ได้ (ω เท่ากันหมดแต่ค่าไม่ใกล้ 180) "
                              "จึงเพิ่ม assert ว่า conformal ทุกตัวต้อง ω < 0.005°"),
    limitations=[
        "geometry ฐานคือ Natural Earth 10m (เส้นขอบประเทศระดับ 1:10 ล้าน) ไม่ใช่แนวเขตกรมที่ดิน "
        "— วัดได้ 514,453.7 ตร.กม. ขณะที่ตัวเลขทางราชการที่ใช้อ้างอิงกันทั่วไปคือ 513,120 ตร.กม. "
        "(ต่าง +0.26%) ซึ่งพอ ๆ กับผลของ projection หลายตัว ดังนั้นตารางในโพสต์นี้ใช้ "
        "*เปรียบเทียบระหว่าง projection* ได้ แต่อย่าเอาไปเทียบกับตัวเลขราชการ",
        "TissotSample ที่ 182 จุด (กริด 9×11 + 83 จุดทะเบียน) ละเอียดพอสำหรับ scale ประเทศ "
        "แต่ไม่พอจับ singularity นอกพื้นที่",
        "gl_th_ratio ของ projection ระดับภูมิภาค (eqdc/lcc/albers/utm) เป็นตัวเลขไร้ความหมาย "
        "— ใส่ไว้เพื่อชี้ว่าค่าข้ามทวีปต้องมาจาก projection ทั้งโลกเท่านั้น",
        "utm48n ไม่ใช่ทางเลือกจริงของไทย (zone คนละ central meridian) ใส่ไว้เทียบ edge distortion",
        "webmerc ให้ ω ไม่ใช่ 0 ทั้งที่มันคือ conformal — เหตุผลคือแบบจำลองทรงกลมของโปรเจกชัน คนละอย่างกับทรงรีที่เราใช้วัด"],
    note="ภาพทุกแผ่น reproject ผ่าน pyproj/PROJ จริง ไม่มีการใส่ effect หรือใช้ tile ภายนอก")
json.dump(STATS, open(os.path.join(OUT, "day22_stats.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("เขียน day22_stats.json + projections/tissot/ranks/cells csv")

# ================================================== 8. ผู้ตัดสิน
for f in ["day22_twelve.png", "day22_tissot.png", "day22_tradeoff.png",
          "day22_projections.csv", "day22_tissot.csv", "day22_ranks.csv",
          "day22_cells.csv", "day22_stats.json"]:
    assert os.path.getsize(os.path.join(OUT, f)) > 0, f"ไฟล์ว่าง: {f}"
json.load(open(os.path.join(OUT, "day22_stats.json"), encoding="utf-8"))
lint_thai(" ".join(TXT) + " " + __doc__, "day22-texts")
aea = PR[PR.name == "albers"].iloc[0]
assert abs(aea.area_err_pct) < 0.005, f"Albers ต้องรักษาพื้นที่ exactly ได้ {aea.area_err_pct}%"
assert ours.ns_scale_dev_max < 1e-4, "eqdc ต้องรักษาความยาวตามเมอริเดียน (h = 1 ทุกจุด)"
for nm in ("lcc", "utm47n", "utm48n", "tmerc-101", "merc-elips"):
    v = float(PR.loc[PR.name == nm, "omega_max_deg"].iloc[0])
    assert v < 0.005, f"{nm} อ้างว่ารักษามุม แต่ ω สูงสุด {v}°"
assert abs(DEC["product_check_pct"] - DEC["total_gap_pct"]) < 0.02, \
    "สามองค์ประกอบที่แยกต้องคูณกลับได้เลขเดิมของ Day 20"
assert AGG["day20_reproduced"], "วัด eqdc ซ้ำต้องได้เลข Day 20"
assert abs(AGG["area_th_eqdc_km2"] - 512593.1) < 1.0, \
    "พื้นที่ไทยใน eqdc ต้องตรงกับที่ Day 10/12/14 ใช้"
assert all(len(v) == len(PR) for v in RK.to_dict("list").values())
assert not all(RANKS[k] == 1 for k in crit), "ตัววัดไม่แยกแยะอะไรเลย — ต้องมีคนแพ้บ้าง"
assert RANKS["area"] != RANKS["angular"], "เกณฑ์พื้นที่กับเกณฑ์มุมต้องไม่ให้คำตอบเดียวกัน"
_src = os.path.abspath(__file__) if "__file__" in globals() else None
if _src and os.path.exists(_src):
    _bad = []
    for _n, _l in enumerate(open(_src, encoding="utf-8").read().split(chr(10)), 1):
        if any(0x0E00 <= ord(c) <= 0x0E7F for c in _l):
            for c in _l:
                o = ord(c)
                if (0x0100 <= o <= 0x017F) or (0x1E00 <= o <= 0x1EFF) or \
                   (0x3000 <= o <= 0x30FF) or (0xAC00 <= o <= 0xD7AF) or (0x4E00 <= o <= 0x9FFF):
                    _bad.append((_n, c))
    assert not _bad, f"พบอักษรไม่ใช่ไทยในบรรทัดไทย: {_bad[:6]}"
    log("ตรวจ unicode ทั้งไฟล์แล้ว")
log("สรุป: " + json.dumps({k: AGG[k] for k in
                          ["projections", "tissot_points", "area_th_geodesic_km2",
                           "area_th_eqdc_km2", "area_th_eqdc_err_pct", "gl_th_true_ratio",
                           "projections_within_half_pct", "dist_err_best", "dist_err_worst",
                           "ranks", "best_overall"]}, ensure_ascii=False))
print("DONE day22 ใน %.1f วิ" % (time.time() - T0))
_LOGF.close()

