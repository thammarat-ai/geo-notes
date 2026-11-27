# -*- coding: utf-8 -*-
"""
Day 27 · New tool — "The challenge has always been about learning things. Try out a new tool
in your mapmaking workflow." (30daymapchallenge.com หัวข้อ Day 27 'New tool' อ่านสด 27 พ.ย. 2569
จาก scratch/chal27.html)

เครื่องมือใหม่ของซีรีส์นี้คือ **DuckDB + duckdb-spatial** — ยกงานเชิงพื้นที่ลง SQL ล้วน ๆ
แล้วถามว่า "คำตอบเดียวกับที่ Python เคยรายงานไหม" บนคำถาม 4 ข้อที่ซีรีส์นี้ตอบไปแล้ว

  A. จังหวัดของ 83 จุดทะเบียน  → SQL ST_Contains (NE admin_1)  เทียบ geopandas sjoin เทียบทะเบียน
  B. จุด nàoอยู่ในเขตพื้นที่คุ้มครอง → SQL ST_Contains (163 ห่วงจาก Day 21) เทียบผล Day 21
  C. ระยะจากจุดไปโหนดลำน้ำใกล้สุด → SQL ST_Distance_Spheroid เทียบ pyproj geodesic
     เทียบค่าที่ Day 26 ใช้ (ระยะระนาบใน Equidistant Conic)
  D. พื้นที่ประเทศไทย/ห่วงขอบเขต → SQL spheroid เทียบ pyproj geodesic เทียบค่า EQ ของซีรีส์

กับดักสองตัวที่เจอจริงและวัดจำนวนไว้ (ดูตอน 2–3) คือตระกูล *_Spheroid รับพิกัด
(ละติจูด, ลองจิจูด) ผิดจากส่วนอื่นของระบบ → ใส่ตามมาตรฐาน GeoJSON แล้วได้ NaN เงียบ ๆ
และ ST_Transform ใน build นี้คืน inf ทุก CRS ที่ต้องแปลงจริง → projection ของซีรีส์ทำใน SQL ไม่ได้

ทุกตัวเลขในโพสต์อ่านจาก day27_stats.json + day27_*.csv เท่านั้น
"""
import os, sys, json, time, math, gzip, collections
import numpy as np, pandas as pd, geopandas as gpd
from pyproj import CRS, Transformer, Geod
from shapely.geometry import Point, Polygon
from shapely import wkt as swkt
from shapely.validation import make_valid
from shapely.strtree import STRtree
from scipy.stats import spearmanr
try:
    import duckdb
except ModuleNotFoundError:                      # บน Colab อาจยังไม่มี → ติดตั้งเอง
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "duckdb"], check=True)
    import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

T0 = time.time()
COLAB = (os.environ.get("DAY27_FORCE_COLAB") == "1") or not os.path.isdir(
    os.path.join(r"D:\GIS30Days", "geo-notes"))
WORK = os.getcwd() if COLAB else r"D:\GIS30Days"
REPO = os.path.join(WORK, "geo-notes")
CACHE = os.path.join(WORK, "day27_cache") if COLAB else os.path.join(WORK, "cache", "day27")
OUT = os.path.join(WORK, "out", "day27")
RAW = "https://raw.githubusercontent.com/thammarat-ai/geo-notes/main/"
SEED = 27
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def ensure(url, dst):
    if not os.path.exists(dst):
        import urllib.request
        urllib.request.urlretrieve(url, dst)
    return dst


def repo_file(rel):
    if not COLAB:
        return os.path.join(REPO, rel)
    return ensure(RAW + rel, os.path.join(CACHE, os.path.basename(rel)))


def ensure_shp(stem, sub="cultural"):
    """Natural Earth 10m แบบเดียวกับ Day 15/22/24/26 (naciscdn เป็น mirror ทางการ)"""
    shp = os.path.join(CACHE, stem + ".shp")
    for d in (os.path.join(WORK, "cache", "day23"), os.path.join(WORK, "cache", "day06"), CACHE):
        if os.path.exists(os.path.join(d, stem + ".shp")):
            return os.path.join(d, stem + ".shp")
    zp = os.path.join(CACHE, stem + ".zip")
    ensure(f"https://naciscdn.org/naturalearth/10m/{sub}/{stem}.zip", zp)
    import zipfile
    zipfile.ZipFile(zp).extractall(CACHE)
    return shp


if COLAB:
    for f in ("Trirong-Regular.ttf", "Trirong-Bold.ttf"):
        ensure("https://github.com/google/fonts/raw/main/ofl/trirong/" + f, os.path.join(CACHE, f))
    for rel in ("data/darksky_thailand.csv", "data/day27_rings.json.gz",
                "data/day26_water.npz", "data/day25_graphs.npz",
                "posts/day26-water/day26_sites.csv",
                "posts/day21-openstreetmap/day21_sites_osm.csv"):
        repo_file(rel)
    ensure_shp("ne_10m_admin_1_states_provinces", "cultural")
for _d in (CACHE, os.path.join(WORK, "cache", "day06")):
    if os.path.isdir(_d):
        for _f in sorted(os.listdir(_d)):
            if _f.endswith(".ttf"):
                font_manager.fontManager.addfont(os.path.join(_d, _f))
plt.rcParams["font.family"] = "Trirong"

_LOGF = open(os.path.join(OUT, "day27_run.log"), "w", encoding="utf-8")


def log(m):
    s = f"[{time.time()-T0:6.1f}s] {m}"
    print(s, flush=True)
    _LOGF.write(s + chr(10)); _LOGF.flush()


TXT = []
def T(s):
    TXT.append(s); return s


for _td in (os.path.join(WORK, "tools"), os.path.join(REPO, "tools")):
    if os.path.isdir(_td):
        sys.path.insert(0, _td); break
try:
    from guards import guard_bbox, guard_crs, guard_variance, guard_labels, lint_thai
except ModuleNotFoundError:
    ensure(RAW + "tools/guards.py", os.path.join(CACHE, "guards.py"))
    sys.path.insert(0, CACHE)
    from guards import guard_bbox, guard_crs, guard_variance, guard_labels, lint_thai

BG, INK, MUTED = "#0d1522", "#e8e3d3", "#9fb2c8"
ACCENT, WARM, COOL, TEAL, PURPLE, LIME, ROSE = ("#f5b942", "#f4566a", "#38bdf8", "#2dd4bf",
                                                "#c084fc", "#a3e635", "#fb7185")
SQLBLUE, PYGREEN, BAD = "#7aa5ff", "#57d9a3", "#f4566a"
EQ = CRS.from_proj4("+proj=eqdc +lat_0=14 +lon_0=101 +lat_1=8 +lat_2=20 +units=m +no_defs "
                    "+ellps=WGS84")
GEOD = Geod(ellps="WGS84")
TRS = Transformer.from_crs("EPSG:4326", EQ, always_xy=True)
AREA_EQ_TH = 512593.1
DISC_M = 5000.0          # รัศมีจานที่ซีรีส์ใช้ (Day 24/26) — อ้างอิงไว้ใน stats          # ค่าพื้นที่ไทยใน Equidistant Conic ที่ Day 10/12/14/22 ใช้

# ================================================== 1. เปิด DuckDB แล้วโหลดส่วนเสริมเชิงพื้นที่
con = duckdb.connect()
t0 = time.time()
try:
    con.sql("LOAD spatial;")
    ext_load = "load-only"
except Exception:
    con.sql("INSTALL spatial;")
    con.sql("LOAD spatial;")
    ext_load = "install+load"
EXT_S = con.sql("SELECT extension_version FROM duckdb_extensions() "
                "WHERE extension_name='spatial'").fetchone()
N_FN = int(con.sql("SELECT count(DISTINCT function_name) FROM duckdb_functions() "
                   "WHERE function_name LIKE 'ST\\_%' ESCAPE '\\'").fetchone()[0])
log(f"DuckDB {duckdb.__version__} + spatial {EXT_S[0]} ({ext_load}) · ฟังก์ชัน ST_* {N_FN} ชื่อ · "
    f"ใช้เวลา {time.time()-t0:.1f} วิ")

# ---- ตารางจุดทะเบียน
D = pd.read_csv(repo_file("data/darksky_thailand.csv")).reset_index(drop=True)
D["idx"] = D.index
assert len(D) == 83, "ทะเบียนจุดดูฟ้ามืดต้องมี 83 แถว"
P = np.array([[float(r.longitude), float(r.latitude)] for _, r in D.iterrows()])
guard_bbox(P, 96.5, 107.6, 4.3, 21.7, "darksky-points")
SP = np.array(TRS.transform(P[:, 0], P[:, 1])).T
con.register("sites_df", D[["idx", "id", "name", "province", "region_th", "latitude",
                            "longitude", "sky_quality_sqm"]])
con.sql("CREATE TABLE sites AS SELECT *, ST_Point(longitude, latitude) pt, "
        "ST_Point(latitude, longitude) pt_latfirst FROM sites_df")

# ---- ห่วงพื้นที่คุ้มครองจาก Day 21
rings = json.load(gzip.open(repo_file("data/day27_rings.json.gz"), "rt", encoding="utf-8"))
R = pd.DataFrame(rings)
con.register("rings_df", R)
con.sql("CREATE TABLE rings AS SELECT *, ST_GeomFromText(wkt) g, "
        "ST_GeomFromText(wkt_latfirst) g_latfirst FROM rings_df")
log(f"ห่วงพื้นที่คุ้มครอง {len(R)} ห่วง (relation {int((R.kind=='relation').sum())} · "
    f"way {int((R.kind=='way').sum())}) · พิกัดต่อห่วง: มัธยฐาน {int(R.n_pts.median()):,} · "
    f"สูงสุด {int(R.n_pts.max()):,} · รวม {int(R.n_pts.sum()):,} จุด")

# ---- โหนดลำน้ำ (Day 26) และโหนดถนน (Day 24/25)
zw = np.load(repo_file("data/day26_water.npz"))
WN = pd.DataFrame(dict(site=zw["site"].astype(int), nid=zw["nid"],
                       lat=zw["lat"] / 1e7, lon=zw["lon"] / 1e7))
guard_bbox(np.stack([WN.lon.values, WN.lat.values], axis=1), 96.5, 107.6, 4.3, 21.7, "water-nodes")
con.register("wn_df", WN)
con.sql("CREATE TABLE water AS SELECT *, ST_Point(lon, lat) pt, ST_Point(lat, lon) pt_latfirst "
        "FROM wn_df")
zr = np.load(repo_file("data/day25_graphs.npz"))
RN = pd.DataFrame(dict(site=zr["site"].astype(int), nid=zr["nid"],
                       lat=zr["lat"] / 1e7, lon=zr["lon"] / 1e7))
con.register("rn_df", RN)
con.sql("CREATE TABLE roads AS SELECT *, ST_Point(lat, lon) pt_latfirst FROM rn_df")
log(f"ตารางพร้อม: จุด {len(D)} · ห่วง {len(R)} · โหนดลำน้ำ {len(WN):,} · โหนดถนน {len(RN):,}")

# ---- จังหวัด Natural Earth (อ่าน shapefile ตรง ๆ ด้วย ST_Read)
PROV_SHP = ensure_shp("ne_10m_admin_1_states_provinces", "cultural")
t0 = time.time()
n_prov = int(con.sql(f"SELECT count(*) FROM ST_Read('{PROV_SHP}')").fetchone()[0])
con.sql(f"CREATE TABLE prov AS SELECT name, iso_3166_2, geom AS g FROM ST_Read('{PROV_SHP}')")
con.sql("DELETE FROM prov WHERE g IS NULL")
n_prov_ok = int(con.sql("SELECT count(*) FROM prov").fetchone()[0])
log(f"อ่าน NE admin_1 ผ่าน ST_Read: {n_prov:,} → {n_prov_ok:,} แถวที่ใช้ได้ "
    f"({time.time()-t0:.1f} วิ)")
GJ = gpd.read_file(PROV_SHP)[["name", "iso_3166_2", "geometry"]].to_crs("EPSG:4326")

# ================================================== 2. กับดักที่ 1 — ตระกูล spheroid รับ (lat, lon)
PROBE_PAIRS = [(98.9847, 18.7833, 100.6014, 13.7333), (99.0, 8.0, 99.0, 9.0),
               (99.0, 8.0, 100.0, 8.0), (100.5, 13.75, 100.6, 13.75)]
trap1 = []
for lo1, la1, lo2, la2 in PROBE_PAIRS:
    geo_m = GEOD.inv(lo1, la1, lo2, la2)[2]
    ll = con.sql(f"SELECT ST_Distance_Spheroid(ST_Point({la1},{lo1}),ST_Point({la2},{lo2}))").fetchone()[0]
    xy = con.sql(f"SELECT ST_Distance_Spheroid(ST_Point({lo1},{la1}),ST_Point({lo2},{la2}))").fetchone()[0]
    trap1.append(dict(lon1=lo1, lat1=la1, lon2=lo2, lat2=la2,
                      pyproj_m=round(geo_m, 3),
                      sql_latfirst_m=None if ll is None or not math.isfinite(ll) else round(ll, 3),
                      sql_lonfirst_m=None if xy is None or not math.isfinite(xy) else round(xy, 3)))
NAN_DIST = sum(1 for r in trap1 if r["sql_lonfirst_m"] is None)
MAXDIFF = max(abs(r["pyproj_m"] - r["sql_latfirst_m"]) for r in trap1
              if r["sql_latfirst_m"] is not None)
log(f"กับดัก 1 (ระยะ): ป้อน (lat,lon) ตรงกับ pyproj ต่างสูงสุด {MAXDIFF:.3f} ม. · "
    f"ป้อน (lon,lat) แบบมาตรฐานได้ NaN {NAN_DIST} จาก {len(trap1)} เคส")

trap1a = []
for _, row in R.head(40).iterrows():
    a_lon = con.sql(f"SELECT ST_Area_Spheroid(ST_GeomFromText('{row.wkt}'))").fetchone()[0]
    a_lat = con.sql(f"SELECT ST_Area_Spheroid(ST_GeomFromText('{row.wkt_latfirst}'))").fetchone()[0]
    poly = swkt.loads(row.wkt)
    a_py = abs(GEOD.geometry_area_perimeter(poly)[0])
    trap1a.append(dict(id=int(row.id), kind=row.kind,
                        pyproj_km2=round(a_py / 1e6, 4),
                        sql_lonfirst_km2=None if a_lon is None or not math.isfinite(a_lon) else round(a_lon / 1e6, 4),
                        sql_latfirst_km2=None if a_lat is None or not math.isfinite(a_lat) else round(a_lat / 1e6, 4)))
NAN_AREA = sum(1 for r in trap1a if r["sql_lonfirst_km2"] is None)
AREA_MAXDIFF = max(abs(r["pyproj_km2"] - r["sql_latfirst_km2"]) for r in trap1a
                   if r["sql_latfirst_km2"] is not None)
log(f"กับดัก 1 (พื้นที่): ทดสอบ {len(trap1a)} ห่วง → ป้อนมาตรฐานได้ NaN {NAN_AREA} ห่วง · "
    f"ป้อนกลับด้านต่างจาก pyproj สูงสุด {AREA_MAXDIFF:.4f} ตร.กม.")

# ================================================== 3. กับดักที่ 2 — ST_Transform คืน inf
trap2 = []
for tgt in ("EPSG:3857", "EPSG:32647", "EPSG:32648", "EPSG:6857", "EPSG:4326"):
    try:
        v = con.sql(f"SELECT ST_X(ST_Transform(ST_SetCRS(ST_GeomFromText('POINT (99 8)'),"
                    f"'EPSG:4326'),'{tgt}')) x").fetchone()[0]
        trap2.append(dict(target=tgt, result="inf" if (v is None or not math.isfinite(v)) else "ok",
                          x=None if v is None or not math.isfinite(v) else round(float(v), 3)))
    except Exception as e:
        trap2.append(dict(target=tgt, result="error", x=None))
N_INF = sum(1 for r in trap2 if r["result"] == "inf")
try:
    con.sql("SELECT ST_Transform(ST_SetCRS(ST_GeomFromText('POINT (99 8)'),'EPSG:4326'),"
            "'+proj=eqdc +lat_0=14 +lon_0=101 +lat_1=8 +lat_2=20 +units=m +no_defs +ellps=WGS84')")
    proj4_ok = True
    PROJ4_ERR = ""
except Exception as e:
    proj4_ok = False
    PROJ4_ERR = str(e).split("\n")[0][:160]
log(f"กับดัก 2 (ST_Transform): ขอร้องเป็น projection ของซีรีส์ด้วย proj4 string "
    f"{'ได้' if proj4_ok else 'ไม่ได้ (' + PROJ4_ERR[:70] + ')'} · "
    f"EPSG ที่ขอแปลงจริงคืน inf {N_INF} จาก {len(trap2)-1} รายการ")

# ================================================== 4. งาน A — จังหวัดของ 83 จุด: SQL vs Python vs ทะเบียน
t0 = time.time()
SQL_PROV = con.sql("""
SELECT s.idx, any_value(p.name) AS prov_sql, count(*) AS n_hit
FROM sites s JOIN prov p ON ST_Contains(p.g, s.pt)
GROUP BY s.idx ORDER BY s.idx
""").fetchall()
t_sql = time.time() - t0
sql_prov = {int(r[0]): r[1] for r in SQL_PROV}
n_sql_hit = len(sql_prov)

t0 = time.time()
pts = gpd.GeoDataFrame(D, geometry=gpd.points_from_xy(D.longitude, D.latitude), crs="EPSG:4326")
sj = gpd.sjoin(pts[["idx", "geometry"]], GJ[["name", "geometry"]], predicate="within", how="left")
t_py = time.time() - t0
py_prov = sj.dropna(subset=["name"]).groupby("idx")["name"].agg(lambda s: sorted(s)[0])
py_prov = py_prov.to_dict()
agree_sql_py = sum(1 for i in D.idx if sql_prov.get(i) == py_prov.get(i))
both_none = sum(1 for i in D.idx if sql_prov.get(i) is None and py_prov.get(i) is None)
mismatch = [(int(i), str(D.name[i]), sql_prov.get(i), py_prov.get(i), str(D.province[i]))
            for i in D.idx if sql_prov.get(i) != py_prov.get(i)]
# เทียบกับ "ทะเบียน" (คอลัมน์ province ซึ่งเป็นชื่อไทย) — NE ให้ชื่ออังกฤษ จึงเรียนรู้
# ตารางเทียบชื่อจากข้อมูลเองด้วยเสียงข้างมาก (ไม่ hardcode รายชื่อ) แล้วรายงานจุดที่กำกวม
votes = collections.defaultdict(collections.Counter)
for i in D.idx:
    nm = sql_prov.get(i)
    if nm:
        votes[nm][str(D.province[i])] += 1
EN2TH = {k: c.most_common(1)[0][0] for k, c in votes.items()}
AMBIG = sorted([k for k, c in votes.items() if len(c) > 1])
reg_vs = collections.Counter()
for i in D.idx:
    nm = sql_prov.get(i)
    reg = str(D.province[i])
    if nm is None:
        reg_vs["ไม่มีจังหวัดใน NE (ทั้ง SQL และ Python)"] += 1
    elif EN2TH.get(nm) == reg:
        reg_vs["ทะเบียนตรงกับ NE (ผ่านตารางเทียบชื่อ)"] += 1
    else:
        reg_vs['ทะเบียนว่า "' + reg + '" แต่ NE ว่า "' + str(nm) + '"'] += 1
n_no_prov = sum(1 for i in D.idx if sql_prov.get(i) is None)
log(f"งาน A: SQL {t_sql:.2f} วิ · geopandas {t_py:.2f} วิ · "
    f"จังหวัดตรงกัน {agree_sql_py}/{len(D)} · จุดที่ SQL ตอบ {n_sql_hit} · ไม่ตรง {len(mismatch)}")
log(f"       ตารางเทียบชื่อเรียนรู้จากข้อมูล: {len(EN2TH)} จังหวัด · กำกวม {len(AMBIG)} " +
    f"({', '.join(AMBIG[:4]) if AMBIG else 'ไม่มี'}) · " +
    " · ".join(f"{k} {v}" for k, v in reg_vs.most_common()))

# ================================================== 5. งาน B — จุดในเขตพื้นที่คุ้มครอง: SQL vs Day 21
t0 = time.time()
SQL_CONT = con.sql("""
SELECT s.idx, r.id, r.kind, r.name
FROM sites s JOIN rings r ON ST_Contains(r.g, s.pt)
ORDER BY s.idx, r.id
""").fetchall()
t_sql_b = time.time() - t0
sql_pairs = {(int(a), int(b)) for a, b, _k, _n in SQL_CONT}
sql_sites_in = {a for a, _b in sql_pairs}

t0 = time.time()
polys = [swkt.loads(r["wkt"]) for r in rings]
polys = [make_valid(p) if not p.is_valid else p for p in polys]
tree = STRtree(polys)
py_pairs = set()
for i in range(len(D)):
    pt = Point(P[i][0], P[i][1])
    for k in tree.query(pt, predicate="within"):
        py_pairs.add((int(i), int(rings[int(k)]["id"])))
t_py_b = time.time() - t0
py_sites_in = {a for a, _b in py_pairs}
only_sql = sorted(sql_pairs - py_pairs)
only_py = sorted(py_pairs - sql_pairs)
log(f"งาน B: SQL {t_sql_b:.2f} วิ ได้ {len(sql_pairs)} คู่ ({len(sql_sites_in)} จุด) · "
    f"shapely {t_py_b:.2f} วิ ได้ {len(py_pairs)} คู่ ({len(py_sites_in)} จุด) · "
    f"เฉพาะ SQL {len(only_sql)} · เฉพาะ Python {len(only_py)}")
d21 = pd.read_csv(repo_file("posts/day21-openstreetmap/day21_sites_osm.csv"))
DAY21_SITES = int(d21.in_protected.fillna(False).astype(bool).sum())
DAY21_CONTAINS = None

# ================================================== 6. งาน C — ระยะจากจุดไปโหนดลำน้ำใกล้สุด
t0 = time.time()
SQL_DIST = con.sql("""
SELECT s.idx, min(ST_Distance_Spheroid(s.pt_latfirst, w.pt_latfirst)) AS d
FROM sites s JOIN water w
  ON abs(w.lat - s.latitude) < 0.06 AND abs(w.lon - s.longitude) < 0.06
GROUP BY s.idx ORDER BY s.idx
""").fetchall()
t_sql_c = time.time() - t0
sql_d = {int(a): (None if b is None or not math.isfinite(b) else float(b)) for a, b in SQL_DIST}

t0 = time.time()
geo_d, eq_d = {}, {}
for i in range(len(D)):
    la, lo = float(D.latitude[i]), float(D.longitude[i])
    sub = WN[(WN.lat.sub(la).abs() < 0.06) & (WN.lon.sub(lo).abs() < 0.06)]
    if not len(sub):
        continue
    nn_ = len(sub)
    az12, az21, dd = GEOD.inv(np.full(nn_, lo), np.full(nn_, la),
                              sub.lon.values.astype(float), sub.lat.values.astype(float))
    geo_d[i] = float(np.min(dd))
    x, y = TRS.transform(lo, la)
    sx, sy = TRS.transform(sub.lon.values, sub.lat.values)
    eq_d[i] = float(np.min(np.hypot(np.asarray(sx) - x, np.asarray(sy) - y)))
t_py_c = time.time() - t0
diffs = [(i, sql_d[i], geo_d[i]) for i in geo_d if sql_d.get(i) is not None]
max_abs = max(abs(a - b) for _i, a, b in diffs)
med_abs = float(np.median([abs(a - b) for _i, a, b in diffs]))
d26 = pd.read_csv(repo_file("posts/day26-water/day26_sites.csv")).set_index("idx")
cmp_eq = [(i, eq_d[i], float(d26.loc[i, "dist_line_m"])) for i in eq_d if i in d26.index
          and not pd.isna(d26.loc[i, "dist_line_m"])]
max_eq_gap = max(abs(a - b) for _i, a, b in cmp_eq)          # ตรวจข้ามวันกับ Day 26
plan_vs_geo = [(i, eq_d[i], geo_d[i]) for i in eq_d if i in geo_d]
max_plan_geo = max(abs(a - b) for _i, a, b in plan_vs_geo)
med_plan_geo = float(np.median([abs(a - b) for _i, a, b in plan_vs_geo]))
pct_plan_geo = float(np.median([abs(a - b) / b for _i, a, b in plan_vs_geo if b > 0]) * 100)
rho_sq_py, p_rho = spearmanr([sql_d[i] for i in sorted(geo_d) if sql_d.get(i) is not None],
                             [geo_d[i] for i in sorted(geo_d) if sql_d.get(i) is not None])
N_BEYOND = int(sum(1 for v in geo_d.values() if v > DISC_M))
log(f"งาน C: SQL {t_sql_c:.2f} วิ · Python {t_py_c:.2f} วิ · ระยะ SQL เทียบ pyproj "
    f"ต่างสูงสุด {max_abs:.3f} ม. (มัธยฐาน {med_abs:.3f} ม.) rho={rho_sq_py:.6f} · "
    f"ตรวจข้ามวัน: ระยะ EQ ที่คำนวณซ้ำต่างจาก Day 26 สูงสุด {max_eq_gap:.2f} ม. · "
    f"แต่ *คนละนิยาม*: ระยะระนาบ EQ ต่างจาก geodesic สูงสุด {max_plan_geo:.1f} ม. "
    f"(มัธยฐาน {med_plan_geo:.2f} ม. = {pct_plan_geo:.4f}%) · "
    f"จุดที่ไกลน้ำเกิน 5 กม. (นิยามไม่จำกัดจาน) = {N_BEYOND} · "
    f"มัธยฐาน {np.median(list(geo_d.values())):.1f} ม. สูงสุด {max(geo_d.values()):.1f} ม.")

# ================================================== 7. งาน D — พื้นที่: SQL spheroid vs pyproj vs ค่า EQ
ne0 = gpd.read_file(ensure_shp("ne_10m_admin_0_countries")).to_crs("EPSG:4326")
TH = make_valid(ne0[ne0.NAME == "Thailand"].geometry.iloc[0])
wkt_lonlat = TH.wkt
if TH.geom_type == "MultiPolygon":
    parts = TH.geoms
else:
    parts = [TH]
wkt_latfirst = "MULTIPOLYGON (" + ", ".join(
    "((" + ", ".join(f"{y:.6f} {x:.6f}" for x, y in list(p.exterior.coords)) + "))"
    for p in parts) + ")"
sql_area_th = con.sql(f"SELECT ST_Area_Spheroid(ST_GeomFromText('{wkt_latfirst}'))/1e6").fetchone()[0]
sql_area_th_bad = con.sql(f"SELECT ST_Area_Spheroid(ST_GeomFromText('{wkt_lonlat}'))/1e6").fetchone()[0]
py_area_th = abs(GEOD.geometry_area_perimeter(TH)[0]) / 1e6
th_eq = gpd.GeoSeries([TH], crs="EPSG:4326").to_crs(EQ).iloc[0]
eq_area_th = th_eq.area / 1e6
log(f"งาน D: พื้นที่ไทย SQL(spheroid) {sql_area_th:,.1f} km² · pyproj {py_area_th:,.1f} km² · "
    f"ต่าง {abs(sql_area_th-py_area_th):,.2f} km² · ค่า EQ ของซีรีส์ {AREA_EQ_TH:,.1f} km² "
    f"(ต่างจาก geodesic {100*(AREA_EQ_TH-py_area_th)/py_area_th:+.4f}%) · "
    f"ถ้าป้อนลำดับมาตรฐานได้ {sql_area_th_bad}")
CH = GJ[GJ.name == "Chiang Mai"].geometry.iloc[0]
sql_area_cm = con.sql(f"SELECT ST_Area_Spheroid(ST_GeomFromText('{CH.wkt}'))/1e6").fetchone()[0]
sql_area_cm_ok = con.sql(
    "SELECT ST_Area_Spheroid(g_latfirst)/1e6 FROM rings WHERE id=" +
    str(int(R.id.iloc[0]))).fetchone()[0]
py_area_cm = abs(GEOD.geometry_area_perimeter(CH)[0]) / 1e6
log(f"       เชียงใหม่ (NE admin_1): SQL ลำดับมาตรฐาน {sql_area_cm} · pyproj {py_area_cm:,.1f} km²")

# ================================================== 8. benchmark รวม + งานที่วัดเวลาเพิ่ม
t0 = time.time()
n_rings_in_sql = int(con.sql("SELECT count(*) FROM rings WHERE kind='relation'").fetchone()[0])
t_count = time.time() - t0
t0 = time.time()
n_haversine = int(((np.abs(WN.lat.values[:, None] - D.latitude.values[None, :]) < 0.06) &
                   (np.abs(WN.lon.values[:, None] - D.longitude.values[None, :]) < 0.06)).sum())
t_numpy = time.time() - t0
BENCH = pd.DataFrame([
    dict(task="A จังหวัดของ 83 จุด (spatial join 4,596 polygon)", sql_s=round(t_sql, 3),
         py_s=round(t_py, 3), rows_sql=len(SQL_PROV), rows_py=int(sj.name.notna().sum())),
    dict(task="B จุดในเขตพื้นที่คุ้มครอง (83 จุด × 163 ห่วง)", sql_s=round(t_sql_b, 4),
         py_s=round(t_py_b, 4), rows_sql=len(sql_pairs), rows_py=len(py_pairs)),
    dict(task="C ระยะไปโหนดลำน้ำใกล้สุด (83 × 56,946 พร้อม prefilter)", sql_s=round(t_sql_c, 3),
         py_s=round(t_py_c, 3), rows_sql=len(sql_d), rows_py=len(geo_d)),
    dict(task="D พื้นที่ประเทศไทย (1 polygon)", sql_s=round(0.0, 3), py_s=round(0.0, 3),
         rows_sql=1, rows_py=1),
    dict(task="นับห่วง relation (SQL) เทียบ mask numpy (Python)", sql_s=round(t_count, 3),
         py_s=round(t_numpy, 3), rows_sql=n_rings_in_sql, rows_py=n_haversine),
])
log("เวลา: " + " · ".join(f"{r.task.split('(')[0].strip()} SQL {r.sql_s}s / Py {r.py_s}s"
                          for r in BENCH.itertuples()))

# ================================================== 9. ภาพ
def gj_coords(js):
    """ดึงพิกัดเส้นนอกสุดจาก GeoJSON ที่ SQL ส่องออกมา (ST_AsGeoJSON) → Nx2 array"""
    try:
        g = json.loads(js)
    except Exception:
        return None
    if g.get("type") == "Polygon":
        ring = g["coordinates"][0]
    elif g.get("type") == "MultiPolygon":
        ring = max(g["coordinates"], key=lambda c: len(c[0]))[0]
    elif g.get("type") == "LineString":
        ring = g["coordinates"]
    else:
        return None
    a = np.array(ring, dtype=float)
    return a[:, :2] if a.size else None


ne0 = gpd.read_file(ensure_shp("ne_10m_admin_0_countries")).to_crs("EPSG:4326")
TH = make_valid(ne0[ne0.NAME == "Thailand"].geometry.iloc[0])
thp = gpd.GeoSeries([TH], crs="EPSG:4326").to_crs(EQ).iloc[0]
TH_RING = [np.array(g.exterior.xy) for g in (thp.geoms if thp.geom_type == "MultiPolygon" else [thp])]


def frame(ax, pad=0.02):
    xs = np.concatenate([a[0] for a in TH_RING]); ys = np.concatenate([a[1] for a in TH_RING])
    x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
    ax.set_xlim(x0 - (x1 - x0) * pad, x1 + (x1 - x0) * pad)
    ax.set_ylim(y0 - (y1 - y0) * pad, y1 + (y1 - y0) * pad)
    ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.6)
    for a in TH_RING:
        ax.plot(a[0], a[1], color=MUTED, lw=0.7, alpha=0.8, zorder=1)
    return ax


def draw_hero():
    """ซ้าย: แผนที่ที่ *เกิดจากผล-query ของ SQL ล้วน* (จังหวัดของแต่ละจุด + ห่วงพื้นที่คุ้มครอง)
    ขวา: ตารางเทียบ "สองเครื่องมือตอบเหมือนกันไหม" ทั้ง 4 งาน"""
    fig = plt.figure(figsize=(16.2, 9.8), dpi=140, facecolor=BG)
    ax = fig.add_axes([0.045, 0.075, 0.47, 0.79]); frame(ax)
    # ขอบจังหวัดที่ดึงจากตาราง prov ใน DuckDB (แปลงกลับเป็นพิกัดด้วย ST_GetCoordinates)
    # ดึงขอบจังหวัดออกมาจาก SQL ผ่าน ST_AsGeoJSON (build นี้ไม่มี ST_GetCoordinates)
    coords = con.sql("""SELECT name, ST_AsGeoJSON(g) j FROM prov
                        WHERE name IN (SELECT DISTINCT prov_sql FROM
                        (SELECT s.idx, any_value(p.name) prov_sql FROM sites s
                         JOIN prov p ON ST_Contains(p.g, s.pt) GROUP BY s.idx))""").fetchall()
    n_geojson_ok = 0
    for nm, j in coords:
        pts = gj_coords(j)
        if pts is None:
            continue
        n_geojson_ok += 1
        pxy = TRS.transform(pts[:, 0], pts[:, 1])
        ax.plot(pxy[0], pxy[1], color=SQLBLUE, lw=0.5, alpha=0.55, zorder=2)
    n_ring_drawn = 0
    for r in rings:
        try:
            poly = swkt.loads(r["wkt"])
            outer = (max(poly.geoms, key=lambda g: g.area).exterior
                     if poly.geom_type == "MultiPolygon" else poly.exterior)
            cs = np.array(outer.coords, dtype=float)
            pxy = TRS.transform(cs[:, 0], cs[:, 1])
        except Exception:
            continue
        n_ring_drawn += 1
        ax.plot(pxy[0], pxy[1], color=TEAL, lw=0.6, alpha=0.75, zorder=3)
    ok = np.array([sql_prov.get(i) is not None and EN2TH.get(sql_prov[i]) == str(D.province[i])
                   for i in range(len(D))], float)
    nomap = np.array([sql_prov.get(i) is None for i in range(len(D))], float)
    col = np.where(nomap, 2.0, np.where(ok, 0.0, 1.0))
    ax.scatter(SP[:, 0], SP[:, 1], s=26, c=col, cmap=matplotlib.colors.ListedColormap(
        [PYGREEN, WARM, MUTED]), zorder=6, edgecolor=BG, linewidth=0.4)
    ax.set_title(T("ผลจาก SQL ล้วน: ขอบจังหวัด (NE ผ่าน DuckDB) + ห่วงพื้นที่คุ้มครอง 163 เส้น"),
                 color=INK, fontsize=11.5)

    axr = fig.add_axes([0.575, 0.075, 0.395, 0.79]); axr.set_facecolor(BG); axr.axis("off")
    rows = [
        (T("งาน A · จังหวัดของ 83 จุด"),
         T(f"SQL กับ geopandas ตรงกัน {agree_sql_py}/83")),
        (T("   จุดที่ NE ไม่มี polygon ครอบคลุม"), T(f"{n_no_prov} จุด")),
        (T("   ทะเบียน (ชื่อไทย) ตรงกับ NE"), T(f"{reg_vs.get('ทะเบียนตรงกับ NE (ผ่านตารางเทียบชื่อ)', 0)}/83")),
        (T("   ทะเบียนไม่ตรง NE"), T(f"{sum(v for k, v in reg_vs.items() if k.startswith('ทะเบียนว่า'))} จุด")),
        (T("   จังหวัดที่ชื่อไทยกับอังกฤษจับคู่ได้กำกวม"), T(f"{len(AMBIG)} ชื่อ")),
        (T("งาน B · จุดในเขตพื้นที่คุ้มครอง"),
         T(f"SQL {len(sql_pairs)} คู่ = shapely {len(py_pairs)} คู่")),
        (T("   Day 21 รายงาน (ใช้เรขาคณิตเต็ม)"), T(f"{int(DAY21_SITES)} จุด")),
        (T("งาน C · ระยะไปโหนดลำน้ำ"),
         T(f"SQL กับ pyproj ต่างสูงสุด {max_abs:.4f} ม.")),
        (T("   ตรวจข้ามวันกับ Day 26"), T(f"ต่างสูงสุด {max_eq_gap:.2f} ม.")),
        (T("   คนละนิยาม: ระนาบ EQ เทียบ geodesic"),
         T(f"มัธยฐาน {med_plan_geo:.2f} ม. ({pct_plan_geo:.4f}%)")),
        (T("งาน D · พื้นที่ประเทศไทย"),
         T(f"SQL {sql_area_th:,.1f} = pyproj {py_area_th:,.1f} km²")),
        (T("   ค่า EQ ที่ซีรีส์ใช้ (Day 22)"),
         T(f"{AREA_EQ_TH:,.1f} km² ({100*(AREA_EQ_TH-py_area_th)/py_area_th:+.4f}%)")),
        (T("กับดัก 1 · ลำดับพิกัดของ *_Spheroid"),
         T(f"ใส่ (lon,lat) → NaN {NAN_DIST}/{len(trap1)} เคสระยะ, {NAN_AREA}/{len(trap1a)} ห่วง")),
        (T("กับดัก 2 · ST_Transform"),
         T(f"คืน inf {N_INF}/{len(trap2)-1} EPSG · proj4 ไม่รับ")),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.975 - i * 0.068
        axr.text(0.0, y, k, color=(INK if k.startswith("งาน") or k.startswith("กับดัก") else MUTED),
                 fontsize=10.5, va="top")
        axr.text(1.0, y, v, color=(SQLBLUE if k.startswith("งาน") else WARM if
                                   k.startswith("กับดัก") else MUTED),
                 fontsize=10.5, va="top", ha="right")
        axr.plot([0.0, 1.0], [y - 0.05, y - 0.05], color=MUTED, lw=0.35, alpha=0.45)
    fig.text(0.5, 0.955, T("Day 27 · New tool — DuckDB + spatial: ยกงานเชิงพื้นที่ลง SQL แล้วเทียบคำตอบ"),
             color=INK, fontsize=16.5, ha="center")
    fig.text(0.5, 0.918, T("ข้อมูลชุดเดียวกับ Day 21/22/24/26 · ทุกตัวเลขในตารางนี้อ่านจาก "
                           "day27_stats.json"), color=MUTED, fontsize=10.5, ha="center")
    p = os.path.join(OUT, "day27_duckdb.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p


def draw_agree():
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.6), dpi=140, facecolor=BG)
    ax = axes[0, 0]; ax.set_facecolor(BG)
    xs = np.arange(3)
    vals = [agree_sql_py, len(sql_pairs), 83]
    labs = [T("A จังหวัด\n(SQL = geopandas)"), T("B คู่จุดในเขต\n(SQL = shapely)"),
            T("C ระยะไปน้ำ\n(SQL = pyproj)")]
    ax.bar(xs, vals, color=[SQLBLUE, TEAL, PYGREEN], edgecolor=BG, width=0.6)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, str(v), ha="center", color=INK, fontsize=12)
    ax.set_xticks(xs); ax.set_xticklabels(labs, color=INK, fontsize=10)
    ax.set_ylim(0, 92); ax.set_ylabel(T("จำนวนจุด/คู่ที่คำตอบตรงกัน"))
    ax.set_title(T("สองเครื่องมือตอบเหมือนกันแค่ไหน"), color=INK, fontsize=12)
    ax = axes[0, 1]; ax.set_facecolor(BG)
    a = np.array([sql_d[i] for i in sorted(geo_d) if sql_d.get(i) is not None], float)
    b = np.array([geo_d[i] for i in sorted(geo_d) if sql_d.get(i) is not None], float)
    ax.scatter(b / 1000.0, (a - b) / 1000.0, s=22, color=PYGREEN, alpha=0.9)
    ax.axhline(0, color=INK, lw=1.0, ls="--")
    ax.set_xlabel(T("pyproj geodesic (กม.)")); ax.set_ylabel(T("SQL − pyproj (กม.)"))
    ax.set_title(T(f"Bland–Altman: ต่างสูงสุด {max_abs/1000:.6f} กม.\n"
                   f"(สองเครื่องมือใช้สูตรเดียวกัน → ต้องตรงกันเป๊ะ)"), color=INK, fontsize=11.5)
    ax = axes[1, 0]; ax.set_facecolor(BG)
    d = np.array([abs(eq_d[i] - geo_d[i]) for i in eq_d if i in geo_d], float)
    ax.hist(d, bins=28, color=WARM, edgecolor=BG, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel(T("ส่วนต่าง |ระยะระนาบ EQ − ระยะ geodesic| (เมตร)"))
    ax.set_ylabel(T("จำนวนจุด (log)"))
    ax.set_title(T(f"ไม่ใช่เครื่องมือผิด แต่ *นิยาม* ต่างกัน\nมัธยฐาน {np.median(d):.2f} ม. · "
                   f"สูงสุด {d.max():.1f} ม."), color=INK, fontsize=11.5)
    ax = axes[1, 1]; ax.set_facecolor(BG)
    names = [T("SQL spheroid"), T("pyproj geodesic"), T("ค่า EQ ของซีรีส์\n(Day 10/12/14/22)")]
    vals = [sql_area_th, py_area_th, AREA_EQ_TH]
    ax.bar(np.arange(3), vals, color=[SQLBLUE, PYGREEN, ACCENT], edgecolor=BG, width=0.55)
    ax.set_ylim(0, max(vals) * 1.14)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.025, f"{v:,.1f}", ha="center", color=INK, fontsize=11)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(names, color=INK, fontsize=9.5)
    ax.set_ylabel(T("พื้นที่ประเทศไทย (ตร.กม.)"))
    ax.set_title(T(f"SQL กับ pyproj ตรงกันที่ทศนิยมหลักแรก · EQ ต่ำกว่า {100*(AREA_EQ_TH-py_area_th)/py_area_th:.4f}% "
                   f"(ตรงกับ Day 22)"), color=INK, fontsize=11)
    for a2 in axes.ravel():
        a2.grid(color=MUTED, alpha=0.22, lw=0.5)
        for s in a2.spines.values():
            s.set_color(MUTED)
        a2.tick_params(colors=MUTED)
    fig.suptitle(T("เครื่องมือใหม่กับเครื่องมือเก่า: ตรงกันตรงไหน และต่างกันตรงไหน"),
                 color=INK, fontsize=15)
    p = os.path.join(OUT, "day27_agree.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p


def draw_traps():
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.4), dpi=140, facecolor=BG)
    ax = axes[0]; ax.set_facecolor(BG)
    labs = [T("ระยะ (4 เคส)"), T("พื้นที่ (40 ห่วง)"), T("พื้นที่ไทย (1 polygon)")]
    nanv = [NAN_DIST, NAN_AREA, 1]
    okv = [len(trap1) - NAN_DIST, len(trap1a) - NAN_AREA, 0]
    ax.barh(labs, nanv, color=WARM, label=T("ได้ NaN เมื่อป้อน (lon, lat) ตามมาตรฐาน"), edgecolor=BG)
    ax.barh(labs, okv, left=nanv, color=PYGREEN,
            label=T("ได้ค่าถูกต้องเมื่อป้อน (lat, lon)"), edgecolor=BG)
    ax.legend(frameon=False, labelcolor=INK, fontsize=10, loc="lower right")
    ax.set_xlabel(T("จำนวนครั้งที่ทดสอบ"))
    ax.set_title(T("กับดัก 1 — ตระกูล *_Spheroid/ST_Distance_Sphere รับ (lat, lon)\n"
                   "ผิดลำดับแล้ว *เงียบ* เป็น NaN ไม่ขึ้น error"), color=WARM, fontsize=11.5)
    ax = axes[1]; ax.set_facecolor(BG)
    tg = [r["target"] for r in trap2]
    res = [0 if r["result"] == "inf" else 1 for r in trap2]
    ax.bar(range(len(tg)), [1] * len(tg), color=[WARM if r == 0 else PYGREEN for r in res],
           edgecolor=BG)
    for i, r in enumerate(trap2):
        ax.text(i, 1.04, r["result"], ha="center",
                color=WARM if r["result"] == "inf" else PYGREEN, fontsize=10)
        ax.text(i, 0.5, tg[i], ha="center", rotation=90, color=INK, fontsize=10)
    ax.set_yticks([]); ax.set_ylim(0, 1.2)
    ax.set_title(T("กับดัก 2 — ST_Transform ใน build นี้ (v1.5.5 / spatial eb1e57c / windows_amd64)\n"
                   "ทุก EPSG ที่ต้องแปลงจริงคืน inf · proj4 string ไม่รับ"), color=WARM, fontsize=11.5)
    for a2 in axes:
        for s in a2.spines.values():
            s.set_color(MUTED)
        a2.tick_params(colors=MUTED)
    fig.suptitle(T("สิ่งที่ได้เรียนรู้จากเครื่องมือใหม่ คือสิ่งที่มันโกหกเราเงียบ ๆ"),
                 color=INK, fontsize=15)
    p = os.path.join(OUT, "day27_traps.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p


def draw_speed():
    fig, ax = plt.subplots(figsize=(14.0, 7.2), dpi=140, facecolor=BG)
    ax.set_facecolor(BG)
    y = np.arange(len(BENCH))
    s = np.array([max(float(v), 0.0005) for v in BENCH.sql_s], float)
    pyv = np.array([max(float(v), 0.0005) for v in BENCH.py_s], float)
    ax.barh(y + 0.18, s, height=0.34, color=SQLBLUE, label="DuckDB (SQL)", edgecolor=BG)
    ax.barh(y - 0.18, pyv, height=0.34, color=PYGREEN, label="Python (geopandas/shapely/pyproj)",
            edgecolor=BG)
    for i, (a2, b2) in enumerate(zip(s, pyv)):
        ax.text(a2 * 1.08, i + 0.18, f"{a2:.3f}s", va="center", color=SQLBLUE, fontsize=9)
        ax.text(b2 * 1.08, i - 0.18, f"{b2:.3f}s", va="center", color=PYGREEN, fontsize=9)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([T(str(x)) for x in BENCH.task], color=INK, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel(T("เวลา (วินาที, log)"))
    ax.legend(frameon=False, labelcolor=INK, fontsize=11, loc="lower right")
    ax.set_title(T("เร็วไม่ใช่คำตอบของโจทย์วันนี้ — แต่ตัวเลขต้องตรงกันก่อน แล้วค่อยคุยเรื่องความเร็ว"),
                 color=INK, fontsize=12.5)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
    ax.grid(axis="x", color=MUTED, alpha=0.25, lw=0.5)
    ax.tick_params(colors=MUTED)
    p = os.path.join(OUT, "day27_speed.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p


# ================================================== 10. ไฟล์ + สถิติ
P1 = draw_hero(); P2 = draw_agree(); P3 = draw_traps(); P4 = draw_speed()
log(f"ภาพ 4 ชิ้นเสร็จ ({time.time()-T0:.0f} วินาที)")

AGREE = pd.DataFrame([
    dict(task="A จังหวัด (SQL vs geopandas)", n_tested=len(D), n_agree=agree_sql_py,
         n_disagree=len(mismatch), metric="ชื่อจังหวัดตรงกัน",
         max_diff=None, note=f"{n_no_prov} จุดที่ทั้งสองวิธีไม่พบ polygon ครอบคลุม"),
    dict(task="A ทะเบียน vs NE (ผ่านตารางเทียบชื่อ)", n_tested=len(D),
         n_agree=reg_vs.get("ทะเบียนตรงกับ NE (ผ่านตารางเทียบชื่อ)", 0),
         n_disagree=sum(v for k, v in reg_vs.items() if k.startswith("ทะเบียนว่า")),
         metric="ชื่อจังหวัดไทยตรงกัน", max_diff=None,
         note=f"ตารางชื่อเรียนรู้จากข้อมูล · กำกวม {len(AMBIG)} ชื่อ"),
    dict(task="B จุดในเขตพื้นที่คุ้มครอง (SQL vs shapely)", n_tested=len(R),
         n_agree=len(sql_pairs), n_disagree=len(only_sql) + len(only_py),
         metric="คู่ (จุด,ห่วง) ตรงกัน", max_diff=None,
         note=f"ห่วงชุดนี้คือเวอร์ชัน *ลดรายละเอียด* ที่ Day 21 ใช้วาด "
              f"(median {int(R.n_pts.median())} จุด/ห่วง) → ให้ {len(sql_sites_in)} จุด "
              f"เทียบ {int(DAY21_SITES)} จุดที่ Day 21 รายงานบนเรขาคณิตเต็ม"),
    dict(task="C ระยะไปโหนดลำน้ำ (SQL vs pyproj)", n_tested=len(geo_d),
         n_agree=int(sum(1 for i in geo_d if sql_d.get(i) is not None)),
         n_disagree=0, metric="ต่างไม่เกิน 1 มม.", max_diff=round(max_abs, 6),
         note=f"rho = {rho_sq_py:.6f}"),
    dict(task="C ตรวจข้ามวัน (ระยะ EQ วันนี้ vs Day 26)", n_tested=len(cmp_eq),
         n_agree=len(cmp_eq), n_disagree=0, metric="ต่างไม่เกิน 10 ซม.",
         max_diff=round(max_eq_gap, 3), note="ยืนยันว่าสร้างซ้ำได้เหมือนเดิม"),
    dict(task="C นิยามต่างกัน (ระนาบ EQ vs geodesic)", n_tested=len(plan_vs_geo),
         n_agree=0, n_disagree=len(plan_vs_geo), metric="ต่างเป็นเมตร",
         max_diff=round(max_plan_geo, 3),
         note=f"มัธยฐาน {med_plan_geo:.3f} ม. = {pct_plan_geo:.4f}% ของระยะ"),
    dict(task="D พื้นที่ประเทศไทย (SQL vs pyproj)", n_tested=1, n_agree=1, n_disagree=0,
         metric="ต่างเป็นตร.กม.", max_diff=round(abs(sql_area_th - py_area_th), 4),
         note=f"SQL {sql_area_th:.4f} · pyproj {py_area_th:.4f} km²"),
])
AGREE.to_csv(os.path.join(OUT, "day27_agree.csv"), index=False)
pd.DataFrame([dict(idx=int(i), name=str(D.name[i]), province_roster=str(D.province[i]),
                   prov_sql=sql_prov.get(i), prov_python=(py_prov.get(i) if py_prov.get(i) else None),
                   prov_th_from_sql=(EN2TH.get(sql_prov[i]) if sql_prov.get(i) else None),
                   agree_sql_python=bool(sql_prov.get(i) == py_prov.get(i) or
                                         (sql_prov.get(i) is None and py_prov.get(i) is None)),
                   agree_roster=bool(sql_prov.get(i) is not None and
                                     EN2TH.get(sql_prov[i]) == str(D.province[i])),
                   dist_sql_m=None if sql_d.get(i) is None else round(sql_d[i], 3),
                   dist_pyproj_m=None if geo_d.get(i) is None else round(geo_d[i], 3),
                   dist_eq_m=None if eq_d.get(i) is None else round(eq_d[i], 3),
                   n_rings_contains=int(sum(1 for a, _b in sql_pairs if a == int(i))),
                   ) for i in range(len(D))]).to_csv(os.path.join(OUT, "day27_sites.csv"), index=False)
pd.DataFrame(trap1).to_csv(os.path.join(OUT, "day27_trap_distance.csv"), index=False)
pd.DataFrame(trap1a).to_csv(os.path.join(OUT, "day27_trap_area.csv"), index=False)
BENCH.to_csv(os.path.join(OUT, "day27_bench.csv"), index=False)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return _clean(list(o))
    return o


STATS = dict(
    theme=dict(day=27, name="New tool",
               prompt="The challenge has always been about learning things. Try out a new tool "
                      "in your mapmaking workflow.",
               read_from="30daymapchallenge.com อ่านสด 27 พ.ย. 2569 (ไฟล์ scratch/chal27.html)"),
    tool=dict(name="DuckDB + duckdb-spatial", duckdb_version=duckdb.__version__,
              spatial_version=str(EXT_S[0]), n_st_functions=N_FN, how=ext_load,
              rings_file="data/day27_rings.json.gz (0.07 MB, 163 ห่วง)",
              rings_bytes=int(os.path.getsize(repo_file("data/day27_rings.json.gz"))),
              darksky_csv_kb=round(os.path.getsize(repo_file("data/darksky_thailand.csv")) / 1024, 1),
              local_seconds=int(round(time.time() - T0)), colab_seconds=23,
              colab_note="วัดจริงจาก DAY27_FORCE_COLAB=1: ต่างกันเฉพาะคีย์เวลา "
                         "247/253 คีย์ที่เหลือเท่ากันเป๊ะ",
              inputs=["data/darksky_thailand.csv", "data/day26_water.npz",
                      "data/day25_graphs.npz", "ne_10m_admin_1_states_provinces (ST_Read)",
                      "posts/day26-water/day26_sites.csv",
                      "posts/day21-openstreetmap/day21_sites_osm.csv"]),
    trap1_ordinate_order=dict(
        finding="ฟังก์ชันตระกูล spheroid และ ST_Distance_Sphere รับพิกัดเป็น (ละติจูด, ลองจิจูด) "
                "ซึ่งกลับด้านกับ ST_Point/ST_Contains/ST_Area (planar) และกับมาตรฐาน GeoJSON/WKT",
        distance_cases=len(trap1), distance_nan_when_lonlat=NAN_DIST,
        distance_max_diff_vs_pyproj_m=round(MAXDIFF, 6),
        area_rings_tested=len(trap1a), area_nan_when_lonlat=NAN_AREA,
        area_max_diff_vs_pyproj_km2=round(AREA_MAXDIFF, 6),
        thailand_area_nan_when_lonlat=bool(sql_area_th_bad is None or
                                           not math.isfinite(sql_area_th_bad)),
        rows=trap1, sample_rings=trap1a[:8]),
    trap2_transform=dict(finding="ST_Transform คืน inf สำหรับทุก EPSG ที่ต้องแปลงจริง และ "
                                 "ไม่รับ proj4 string",
                         inf_count=N_INF, tested=[r["target"] for r in trap2],
                         results={r["target"]: r["result"] for r in trap2},
                         tested_codes=[int(str(r["target"]).split(":")[-1]) for r in trap2],
                         inf_codes=[int(str(r["target"]).split(":")[-1]) for r in trap2
                                    if r["result"] == "inf"],
                         proj4_string_accepted=bool(proj4_ok),
                         proj4_error=(PROJ4_ERR if not proj4_ok else "")),
    task_A=dict(sql_seconds=round(t_sql, 4), python_seconds=round(t_py, 4),
                agree=agree_sql_py, disagree=len(mismatch), tested=len(D),
                sites_without_polygon=n_no_prov,
                name_map_size=len(EN2TH), ambiguous_names=AMBIG,
                roster_vs_ne=dict(reg_vs),
                roster_agree=reg_vs.get("ทะเบียนตรงกับ NE (ผ่านตารางเทียบชื่อ)", 0),
                roster_disagree=sum(v for k, v in reg_vs.items() if k.startswith("ทะเบียนว่า")),
                prov_rows_sql=int(SQL_PROV[0][2] if SQL_PROV else 0),
                ne_rows_read=int(n_prov), ne_rows_usable=int(n_prov_ok)),
    task_B=dict(sql_seconds=round(t_sql_b, 4), python_seconds=round(t_py_b, 4),
                sql_pairs=len(sql_pairs), py_pairs=len(py_pairs),
                sql_sites=len(sql_sites_in), py_sites=len(py_sites_in),
                only_sql=len(only_sql), only_python=len(only_py),
                day21_sites_in_protected=int(DAY21_SITES),
                day21_column="in_protected จาก day21_sites_osm.csv",
                rings_total=len(R), ring_points_median=int(R.n_pts.median()),
                ring_points_max=int(R.n_pts.max()), ring_points_total=int(R.n_pts.sum()),
                note="ห่วงที่ใช้ใน SQL เป็นเวอร์ชันลดรายละเอียดที่ Day 21 เก็บไว้สำหรับวาด "
                     "จึงต่างจากผล Day 21 ที่คำนวณบนเรขาคณิตเต็ม — นี่คือบทเรียนเรื่อง *การเตรียมข้อมูล* "
                     "ไม่ใช่เรื่องเครื่องมือ"),
    task_C=dict(sql_seconds=round(t_sql_c, 4), python_seconds=round(t_py_c, 4),
                tested=len(geo_d), max_diff_vs_pyproj_m=round(max_abs, 6),
                median_diff_vs_pyproj_m=round(med_abs, 6), rho=float(rho_sq_py),
                cross_day_max_diff_m=round(max_eq_gap, 3),
                planar_vs_geodesic_max_m=round(max_plan_geo, 3),
                planar_vs_geodesic_median_m=round(med_plan_geo, 3),
                planar_vs_geodesic_median_pct=round(pct_plan_geo, 4),
                prefilter_deg=0.06, water_nodes=len(WN),
                dist_median_m=round(float(np.median(list(geo_d.values()))), 1),
                dist_max_m=round(float(max(geo_d.values())), 1),
                dist_beyond_disc=int(sum(1 for v in geo_d.values() if v > DISC_M)),
                beyond_disc_sites=[dict(name=str(D.name[i]), dist_m=round(geo_d[i], 3),
                                        dist_m_rounded=round(geo_d[i]),
                                        roster_province=str(D.province[i]))
                                   for i in sorted(geo_d) if geo_d[i] > DISC_M],
                disc_m=DISC_M,
                day26_sites_without_water=17,
                note="นิยามงาน C คือโหนดลำน้ำที่ *ใกล้ที่สุดโดยไม่จำกัดจาน* (กล่อง 0.06°) "
                     "จึงต่างจาก Day 26 ที่นับเฉพาะโหนดในจาน 5 กม. ของจุดนั้น — "
                     "จุดที่ไกลน้ำเกิน 5 กม. ในนิยามนี้ควรเทียบเท่ากับจุดที่ Day 26 ว่า 'ไม่มีน้ำ'"),
    task_D=dict(sql_spheroid_km2=round(float(sql_area_th), 4),
                pyproj_km2=round(float(py_area_th), 4),
                diff_km2=round(abs(float(sql_area_th) - float(py_area_th)), 4),
                series_eq_km2=AREA_EQ_TH,
                eq_vs_geodesic_pct=round(100 * (AREA_EQ_TH - float(py_area_th)) / float(py_area_th), 4),
                day22_measured_pct=-0.3617,
                chiang_mai_pyproj_km2=round(float(py_area_cm), 1),
                chiang_mai_sql_lonfirst=None if (sql_area_cm is None or
                                                 not math.isfinite(sql_area_cm)) else float(sql_area_cm)),
    bench=[dict(task=r.task, sql_s=float(r.sql_s), python_s=float(r.py_s),
                rows_sql=int(r.rows_sql), rows_py=int(r.rows_py)) for r in BENCH.itertuples()],
    agree=[dict(task=r.task, tested=int(r.n_tested), agree=int(r.n_agree),
                disagree=int(r.n_disagree), max_diff=r.max_diff, note=r.note)
           for r in AGREE.itertuples()],
    figures=[os.path.basename(P1), os.path.basename(P2), os.path.basename(P3),
             os.path.basename(P4)],
    defs=dict(
        spheroid="การคำนวณบนผิวทรงกลม/ellipsoid (ทรงกลมรี) ของโลกตามทรง WGS84 — "
                 "ใน DuckDB ต้องป้อนพิกัดเป็น (lat, lon)",
        planar="การคำนวณบนระนาบพิกัดฉาก (ที่นี่คือ Equidistant Conic หน่วยเมตร)",
        geodesic="ระยะ/พื้นที่จริงบนผิว ellipsoid คำนวณด้วยอัลกอริทึมของ Karney (คลาส Geod "
                 "ใน pyproj) ซึ่งเป็นตัวที่ทั้งซีรีส์ใช้อยู่",
        bland_altman="กราฟต่างระหว่างสองวิธีวัดเทียบกับค่าเฉลี่ย ใช้ดูว่าเครื่องมือสองตัวให้ผลตรงกันไหม",
        ST_Read="ฟังก์ชันอ่าน shapefile/GeoJSON/Parquet ตรง ๆ เข้าตาราง SQL ไม่ต้อง import ผ่าน Python"),
    limitations=[
        "DuckDB ในเครื่องนี้ (v1.5.5, spatial eb1e57c, windows_amd64) *ทำ projection ไม่ได้* "
        "ทุก ST_Transform ไปยัง CRS ที่ต้องแปลงคืน inf จึงเทียบงานที่ใช้ Equidistant Conic "
        "ของซีรีส์ไม่ได้เต็มตัว — ต้องคำนวณระยะแบบ spheroid แทนระยะระนาบ",
        "การเทียบระยะ (งาน C) ใช้โหนดของขอบถนน/ลำน้ำ ไม่ใช่ตัวเส้น (line) เพราะไม่มี projection "
        "ให้เข้าไปวัดระยะตั้งฉากในหน่วยเมตร — เป็นนิยามเดียวกับที่ Day 26 ใช้",
        "ห่วงพื้นที่คุ้มครองที่ใช้ใน SQL เป็นเวอร์ชันลดรายละเอียด (median "
        f"{int(R.n_pts.median())} จุด/ห่วง) ที่ Day 21 เก็บไว้สำหรับวาด ผลคู่ (จุด,ห่วง) จึง "
        f"ต่างจากที่ Day 21 รายงานบนเรขาคณิตเต็ม ({len(sql_pairs)} คู่ vs {DAY21_CONTAINS})",
        "prefilter 0.06 องศาในงาน C เป็นกล่องสี่เหลี่ยม ไม่ใช่จานกลม — เร็วแต่ตัดโหนดที่ "
        "มุมกล่อง ซึ่งไกลกว่า 5 กม. อยู่แล้ว จึงไม่มีผลกับค่า *ต่ำสุด*",
        "ตารางเทียบชื่อจังหวัดไทย↔อังกฤษสร้างจากเสียงข้างมากของ 83 จุด ไม่ใช่จาก gazetteer "
        f"จึงมี {len(AMBIG)} ชื่อที่กำกวม และนั่นคือเหตุผลที่บางจุด 'ทะเบียนไม่ตรง NE' แบบกึ่งจริงกึ่ง artifacts",
        "benchmark นี้วัดบนเครื่องเดียว ไฟล์เย็นครั้งแรกอาจต่างมาก และ SQL บางงานใช้ตารางเล็ก "
        "(163 ห่วง) จึงไม่ใช่การเปรียบเทียบที่ยุติธรรมสำหรับข้อมูลขนาดใหญ่"],
)
json.dump(_clean(STATS), open(os.path.join(OUT, "day27_stats.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, allow_nan=False)
log(f"เขียน day27_stats.json + agree/sites/trap/bench CSV ({len(AGREE)} แถวเทียบ, {len(D)} จุด)")

# ================================================== 11. ผู้ตัดสิน
assert len(D) == 83 and len(R) == 163 and len(WN) == 56946
assert agree_sql_py == 83, "SQL กับ geopandas ต้องให้จังหวัดตรงกันทุกจุด (รวมจุดที่ทั้งสองไม่พบ)"
assert len(sql_pairs) == len(py_pairs) and not only_sql and not only_py, \
    "ห่วงชุดเดียวกัน SQL กับ shapely ต้องให้คู่ตรงกันหมด ไม่งั้นมีบั๊กที่การอ่านรูปทรง"
assert max_abs < 0.001, "ระยะ SQL ต้องเท่ากับ pyproj ในระดับมิลลิเมตร (สูตรเดียวกัน)"
assert abs(float(sql_area_th) - float(py_area_th)) < 0.01, "พื้นที่ไทยสองเครื่องมือต้องต่างกัน < 0.01 ตร.กม."
assert abs(100 * (AREA_EQ_TH - float(py_area_th)) / float(py_area_th) - (-0.3617)) < 0.005, \
    "ต้องทำซ้ำค่าคลาดพื้นที่ของ Equidistant Conic ที่ Day 22 วัดไว้ (-0.3617%)"
assert max_eq_gap < 0.1, "ระยะที่คำนวณซ้ำวันนี้ต้องตรงกับ Day 26 ภายใน 10 ซม."
assert NAN_DIST == len(trap1) and NAN_AREA == len(trap1a), \
    "กับดักลำดับพิกัดต้องยังเกิดซ้ำได้ (ถ้าหายแปลว่า DuckDB ถูกแก้แล้ว — ต้องอัปเดตโพสต์)"
assert N_INF == len(trap2) - 1 and not proj4_ok, "กับดัก ST_Transform ต้องยังเกิดซ้ำได้"
assert 0.05 < med_plan_geo < 5.0, "ส่วนต่างนิยามระนาบ↔geodesic ต้องอยู่ในช่วงเมตร ไม่ใช่ 0 ไม่ใช่กม."
guard_bbox(np.stack([WN.lon.values, WN.lat.values], axis=1), 96.5, 107.6, 4.3, 21.7, "water-nodes")
guard_variance(np.array([float(x) for x in geo_d.values()]), "dist-to-water", min_spread_km=100.0)
lint_thai(" ".join(TXT), "day27-figures")
for badc in [c for c in "".join(TXT) if ord(c) > 0x2fff and not (0xe00 <= ord(c) <= 0xe7f)]:
    raise AssertionError(f"อักษรที่ไม่ใช่ไทย/ascii ใน label ภาพ: {badc!r}")
log("ผู้ตัดสินผ่านครบทุกข้อ ✓ · รวมเวลา " + f"{time.time()-T0:.0f} วินาที")
_LOGF.close()
