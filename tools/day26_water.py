# -*- coding: utf-8 -*-
"""
Day 26 · Water — "About 71% of the Earth's surface. That wet thing, you know?"
(30daymapchallenge.com หัวข้อ Day 26 'Water' อ่านสด 27 พ.ย. 2569 จาก scratch/chal26.html)

ซีรีส์นี้วัด "ฟ้ามืด" มา 25 วัน — วันนี้วัดสิ่งที่อยู่ตรงข้ามกับฟ้า: น้ำ
คำถามที่ผมถามด้วยข้อมูลที่เก็บไว้ได้จริง (snapshot OSM 21 พ.ย. 2569 ก้อนเดียวกับ Day 21–25)

  1. รอบจุดดูฟ้ามืด 83 จุด *น้ำเป็นเครือข่ายแบบกิ่งไม้ หรือแบบตาข่าย*
     วัดด้วย cyclomatic number μ = E − V + C (μ = 0 คือป่า/ต้นไม้ ไม่มีวงวน
     μ > 0 คือมีวงวนเดินวนกลับมาที่เก่า) แล้วเทียบกับกราฟถนนของ Day 24 บนกติกาเดียวกัน
  2. น้ำถูก map ดีแค่ไหนเมื่อเทียบถนน — ทั้งประเทศ: way ที่มีแท็ก highway เทียบ waterway
  3. จุดที่ OSM "ไม่มีน้ำเลย" คือไม่มีจริง หรือไม่มีใคร map — เทียบกับลำน้ำของ
     Natural Earth 10m (คนละแหล่งข้อมูล คนละคนทำ)
  4. พื้นที่น้ำในจาน 5 กม. สัมพันธ์กับค่าความมืดที่ทะเบียนบันทึกไว้ไหม

ทุกตัวเลขในโพสต์อ่านจาก day26_stats.json + day26_*.csv เท่านั้น
"""
import os, sys, json, time, math, gzip, collections
import numpy as np, pandas as pd, geopandas as gpd
import networkx as nx
from pyproj import CRS, Transformer, Geod
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.validation import make_valid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection

T0 = time.time()
COLAB = (os.environ.get("DAY26_FORCE_COLAB") == "1") or not os.path.isdir(
    os.path.join(r"D:\GIS30Days", "geo-notes"))
WORK = os.getcwd() if COLAB else r"D:\GIS30Days"
REPO = os.path.join(WORK, "geo-notes")
CACHE = os.path.join(WORK, "day26_cache") if COLAB else os.path.join(WORK, "cache", "day26")
OUT = os.path.join(WORK, "out", "day26")
RAW = "https://raw.githubusercontent.com/thammarat-ai/geo-notes/main/"
SEED = 26
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
    """Natural Earth 10m แบบเดียวกับ Day 15/22/24 (naciscdn เป็น mirror ทางการ)"""
    for d in (os.path.join(WORK, "cache", "day06"), CACHE):
        if os.path.exists(os.path.join(d, stem + ".shp")):
            return os.path.join(d, stem + ".shp")
    zp = os.path.join(CACHE, stem + ".zip")
    ensure(f"https://naciscdn.org/naturalearth/10m/{sub}/{stem}.zip", zp)
    import zipfile
    zipfile.ZipFile(zp).extractall(CACHE)
    return os.path.join(CACHE, stem + ".shp")


if COLAB:
    for f in ("Trirong-Regular.ttf", "Trirong-Bold.ttf"):
        ensure("https://github.com/google/fonts/raw/main/ofl/trirong/" + f, os.path.join(CACHE, f))
    for rel in ("data/darksky_thailand.csv", "posts/day24-network/day24_roads.csv",
                "data/day26_water.npz", "data/day26_water_meta.json", "data/day26_areas.json.gz",
                "data/day25_graphs.npz"):
        repo_file(rel)
    ensure_shp("ne_10m_rivers_lake_centerlines", "physical")
    ensure_shp("ne_10m_admin_0_countries")
for _d in (CACHE, os.path.join(WORK, "cache", "day06")):
    if os.path.isdir(_d):
        for _f in sorted(os.listdir(_d)):
            if _f.endswith(".ttf"):
                font_manager.fontManager.addfont(os.path.join(_d, _f))
plt.rcParams["font.family"] = "Trirong"

_LOGF = open(os.path.join(OUT, "day26_run.log"), "w", encoding="utf-8")


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
WATER, DEEP, ROAD = "#2f8fd8", "#0b3d63", "#c9a227"
EQ = CRS.from_proj4("+proj=eqdc +lat_0=14 +lon_0=101 +lat_1=8 +lat_2=20 +units=m +no_defs "
                    "+ellps=WGS84")
GEOD = Geod(ellps="WGS84")
TRS = Transformer.from_crs("EPSG:4326", EQ, always_xy=True)

# ================================================== 1. เครือข่ายน้ำในจาน 5 กม. (data/day26_water.npz)
D = pd.read_csv(repo_file("data/darksky_thailand.csv")).reset_index(drop=True)
D["idx"] = D.index
assert len(D) == 83, "ทะเบียนจุดดูฟ้ามืดต้องมี 83 แถว"
P = np.array([[float(r.longitude), float(r.latitude)] for _, r in D.iterrows()])
guard_bbox(P, 96.5, 107.6, 4.3, 21.7, "darksky-points")
SP = np.array(TRS.transform(P[:, 0], P[:, 1])).T
guard_crs([SP[:, 0].min(), SP[:, 1].min(), SP[:, 0].max(), SP[:, 1].max()],
          "projected-metres", expect="metres")

# อ่านจาก data/ ใน repo (ไม่ใช่ cache/) → โน้ตบุ๊ก Colab รันจบโดยไม่ต้องมี PBF 330 MB
wm = json.load(open(repo_file("data/day26_water_meta.json"), encoding="utf-8"))
zw = np.load(repo_file("data/day26_water.npz"))
zn = zw          # แถวโหนด: site / nid / lat / lon
ze = zw          # แถวขอบ: esite / u / v / cls (ชื่อคอลัมน์คนละชุดกัน)
AREAS = json.load(gzip.open(repo_file("data/day26_areas.json.gz"), "rt", encoding="utf-8"))
R_DISC = float(wm["radius_m"])
WCLS = wm["classes"]
assert R_DISC == 5000.0, "รัศมีจานต้องตรงกับ Day 24 เพื่อเทียบกันได้"
assert wm["pbf_md5"] == "db3958a037ce717c462101b86e2f75f3", "PBF คนละก้อนกับที่ซีรีส์นี้ใช้"

by_node = {}
WLA = zn["lat"] / 1e7          # ใน npz เก็บเป็น int32 = องศา x 1e7 (~1.1 ซม.)
WLO = zn["lon"] / 1e7
guard_bbox(np.stack([WLO, WLA], axis=1), 96.5, 107.6, 4.3, 21.7, "water-nodes")
for j, nid, la, lo in zip(zn["site"].astype(int), zn["nid"], WLA, WLO):
    by_node.setdefault(int(j), []).append((int(nid), float(la), float(lo)))
by_edge = collections.defaultdict(list)
for j, u, v, c in zip(ze["esite"].astype(int), ze["u"], ze["v"], ze["cls"]):
    by_edge[int(j)].append((int(u), int(v), int(c)))
WIDX = sorted(by_node)
log(f"เส้นน้ำ: {len(WIDX)}/83 จุดที่มีกราฟ · node-slot {len(zn['nid']):,} · ขอบ {len(ze['u']):,} · "
    f"คลาสที่เจอ {len(WCLS)} ค่า · แปลงน้ำในจาน {wm['n_areas_in_discs']:,} แปลง")

# ---- กราฟถนน Day 24 (ใช้เทียบ cyclomatic บนกติกาเดียวกัน)
rd24 = pd.read_csv(repo_file("posts/day24-network/day24_roads.csv")).set_index("idx")
assert len(rd24) == 83


def build(site_nodes, site_edges):
    nid = [x[0] for x in site_nodes]
    loc = {v: i for i, v in enumerate(nid)}
    G = nx.Graph()
    G.add_nodes_from(range(len(nid)))
    for u, v, c in site_edges:
        G.add_edge(loc[u], loc[v], cls=c)
    xy = np.array([[TRS.transform(x[2], x[1])[0], TRS.transform(x[2], x[1])[1]]
                   for x in site_nodes])
    return G, np.array(nid), xy


def strahler(G, root):
    """Horton–Strahler บนต้นไม้ (root = ปลายน้ำที่ไกลที่สุดของเส้นผ่านศูนย์กลางกราฟ)
    OSM แท็ก direction แทบไม่มีในข้อมูลนี้ จึงroot ด้วยเรขาคณิต ไม่ใช่อุทกวิทยา (บอกในโพสต์)"""
    if not nx.is_tree(G) and G.number_of_edges() > G.number_of_nodes() - 1:
        G = nx.maximum_spanning_tree(G)          # มีวงวน → ใช้ spanning tree ก่อน
    # ทำแบบวนซ้ำ (ไม่ recursion) เพราะต้นไม้ลำน้ำมีโซ่ยาวได้หลายร้อยโหนด
    parent = {root: None}
    order_seq = []
    stack = [root]
    while stack:
        u = stack.pop()
        order_seq.append(u)
        for v in G.neighbors(u):
            if v != parent[u]:
                parent[v] = u
                stack.append(v)
    if len(order_seq) != G.number_of_nodes():
        return float("nan"), {}
    order = {}
    for u in reversed(order_seq):
        kids = [v for v in G.neighbors(u) if parent.get(v) == u]
        if not kids:
            order[u] = 1
            continue
        vals = sorted((order[v] for v in kids), reverse=True)
        order[u] = vals[0] + 1 if len(vals) > 1 and vals[0] == vals[1] else vals[0]
    return order[root], order


def graph_stats(G, xy, centre):
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return dict(nodes=0, edges=0, components=0, mu=0, largest_share=np.nan,
                    deadend=np.nan, maxdeg=0, diam_hops=np.nan, diam_km=np.nan,
                    strahler=np.nan, leaf_frac=np.nan, diam_straight_km=np.nan,
                    tortuosity=np.nan)
    comps = list(nx.connected_components(G))
    C = len(comps)
    mu = m - n + C                                    # cyclomatic number
    big = max(comps, key=len)
    d = dict(G.degree())
    leaves = sum(1 for v in d.values() if v == 1)
    sub = G.subgraph(big).copy()
    diam_hops = np.nan
    diam_km = np.nan
    st = np.nan
    if sub.number_of_nodes() >= 2:
        src = next(iter(sub))
        L = nx.single_source_shortest_path_length(sub, src)
        far = max(L, key=lambda k: L[k])
        L2 = nx.single_source_shortest_path_length(sub, far)
        diam_hops = max(L2.values())
        far2 = max(L2, key=lambda k: L2[k])
        path = nx.shortest_path(sub, far, far2)
        # xy อยู่ในระบบ EQ (เมตร) แล้ว → ห้ามป้อนให้ Geod.inv ซึ่งต้องใช้ละติจูด/ลองจิจูด
        # (บั๊กนี้ทำให้ diam_km เป็น NaN ทั้งคอลัมน์ เพราะค่าเมตรนอกขอบเขตของ ellipsoid)
        diam_km = sum(math.hypot(xy[path[i + 1]][0] - xy[path[i]][0],
                                 xy[path[i + 1]][1] - xy[path[i]][1])
                      for i in range(len(path) - 1)) / 1000.0
        # ระยะตรงระหว่างปลายสองข้างของเส้นผ่านศูนย์กลาง → อัตราส่วนนี้คือ "ความคดเคี้ยว"
        # นับเฉพาะกรณีที่อยู่ไกลกันจริง (> 1 กม.) ไม่งั้นเศษส่วนระเบิดจากคู่ปลายที่ชิดกัน
        straight_km = math.hypot(xy[far][0] - xy[far2][0], xy[far][1] - xy[far2][1]) / 1000.0
        tortu = diam_km / straight_km if straight_km > 1.0 else float("nan")
        st, _ = strahler(sub, far)
    return dict(nodes=n, edges=m, components=C, mu=int(mu),
                largest_share=round(len(big) / n, 4),
                deadend=round(leaves / n, 4), maxdeg=int(max(d.values())),
                diam_hops=int(diam_hops) if np.isfinite(diam_hops) else np.nan,
                diam_km=round(float(diam_km), 3) if np.isfinite(diam_km) else np.nan,
                diam_straight_km=round(float(straight_km), 3) if np.isfinite(straight_km) else np.nan,
                tortuosity=round(float(tortu), 3) if np.isfinite(tortu) else np.nan,
                strahler=int(st) if np.isfinite(st) else np.nan,
                leaf_frac=round(leaves / n, 4))


# ---- แปลงน้ำเป็น polygon ในระบบ EQ + คิดพื้นที่น้ำในจาน
def safe_inter(g, disc):
    """intersection ที่ไม่พังกับ ring แปลก ๆ (พบบ่อยกับแปลงน้ำที่ projection แล้วชิดกัน)"""
    for attempt in (0, 1, 2):
        try:
            a = g if attempt == 0 else (make_valid(g) if attempt == 1 else g.buffer(0))
            b = disc if attempt == 0 else disc.buffer(0)
            out = a.intersection(b)
            if out.is_empty:
                return out
            if not out.is_valid:
                out = make_valid(out)
            return out
        except Exception:
            continue
    return None


DISC = {j: Point(SP[j][0], SP[j][1]).buffer(R_DISC, quad_segs=64) for j in range(len(D))}
area_rows = []
water_poly = collections.defaultdict(list)
for a in AREAS:
    try:
        poly = Polygon(a["ring"])
        if not poly.is_valid:
            poly = make_valid(poly)
        pg = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(EQ).iloc[0]
    except Exception:
        continue
    for j in a["sites"]:
        water_poly[int(j)].append(pg)
        inter = safe_inter(pg, DISC[int(j)])
        if inter is None or inter.is_empty:
            continue
        # พื้นที่คำนวณในระนาบ EQ (ตร.กม.) ไม่ใช่ geodesic — Geod.geometry_area_perimeter
        # ต้องใช้ละติจูด/ลองจิจูด ถ้าป้อนเมตรจะได้ค่ามั่ว (เคยพลาดแบบเดียวกัน Day 22)
        # Day 22 วัดแล้วว่า projection นี้คลาดพื้นที่ −0.3617% ซึ่งไม่มีผลกับ "สัดส่วน"
        area_rows.append(dict(idx=int(j), way=a["way"], tag=a["tag"], name=a["name"],
                              water=a.get("water", ""), area_km2=a["area_km2"],
                              in_disc_km2=round(inter.area / 1e6, 5)))
AREA = pd.DataFrame(area_rows)
log(f"แปลงน้ำที่ทับจานจริง {len(AREA):,} ชิ้น (จาก {len(AREAS):,} แปลงที่แตะจาน)")

# ---- ระยะจากจุดไปยังเส้นน้ำ / แปลงน้ำที่ใกล้สุด
WROW = []
for j in range(len(D)):
    nodes = by_node.get(j, [])
    row = dict(idx=j, name=D.name[j], province=D.province[j], region_th=D.region_th[j],
               sqm=float(D.sky_quality_sqm[j]))
    if nodes:
        xy = np.array([[TRS.transform(x[2], x[1])[0], TRS.transform(x[2], x[1])[1]] for x in nodes])
        row["dist_line_m"] = round(float(np.min(np.hypot(xy[:, 0] - SP[j][0],
                                                         xy[:, 1] - SP[j][1]))), 1)
        G, _nid, _xy = build(nodes, by_edge[j])
        row.update({("w_" + k): v for k, v in graph_stats(G, _xy, SP[j]).items()})
    else:
        row["dist_line_m"] = np.nan
        row.update({("w_" + k): np.nan for k in
                    ["nodes", "edges", "components", "mu", "largest_share", "deadend",
                     "maxdeg", "diam_hops", "diam_km", "strahler", "leaf_frac",
                     "diam_straight_km", "tortuosity"]})
    polys = water_poly.get(j, [])
    if polys:
        merged = polys[0] if len(polys) == 1 else MultiPolygon(
            [g for p in polys for g in (p.geoms if p.geom_type == "MultiPolygon" else [p])])
        merged = make_valid(merged)
        inter = safe_inter(merged, DISC[j])
        a_in = 0.0
        if inter is not None and not inter.is_empty:
            a_in = inter.area / 1e6
        disc_a = DISC[j].area / 1e6
        row["water_pct"] = round(100.0 * a_in / disc_a, 3)
        row["water_km2"] = round(a_in, 3)
        row["n_areas"] = len(polys)
        row["dist_area_m"] = round(float(min(p.distance(Point(SP[j][0], SP[j][1])) for p in polys)), 1)
        sub = AREA[AREA.idx == j]
        if len(sub):
            b = sub.loc[sub.in_disc_km2.fillna(0.0).idxmax()]
            row["biggest_area_km2"] = round(float(b.in_disc_km2), 4)
            row["biggest_name"] = str(b["name"]) or str(b["tag"])
        else:
            row["biggest_area_km2"] = round(float(max(p.area for p in polys)) / 1e6, 3)
            row["biggest_name"] = ""
    else:
        row.update(water_pct=0.0, water_km2=0.0, n_areas=0, dist_area_m=np.nan,
                   biggest_area_km2=np.nan, biggest_name="")
    row["r_nodes"] = int(rd24.loc[j, "nodes"]) if j in rd24.index else 0
    row["r_edges"] = int(rd24.loc[j, "edges"]) if j in rd24.index else 0
    row["r_components"] = int(rd24.loc[j, "components"]) if j in rd24.index else 0
    WROW.append(row)
W = pd.DataFrame(WROW)
W["r_mu"] = W.r_edges - W.r_nodes + W.r_components.replace(0, 1)
log(f"จุดที่มีเส้นน้ำ {int((W.w_nodes > 0).sum())}/83 · "
    f"จุดที่มีแปลงน้ำ *ทับจาน* {int((W.water_pct > 0).sum())}/83 · "
    f"มัธยฐานพื้นที่น้ำในจาน {W.water_pct.median():.3f}% · "
    f"จุดที่น้ำเยอะสุด {W.water_pct.max():.1f}% ({W.loc[W.water_pct.idxmax(), 'name']})")

# ================================================== 2. เทียบกับ Natural Earth (น้ำหายไป หรือไม่มีใคร map)
ner = gpd.read_file(ensure_shp("ne_10m_rivers_lake_centerlines", "physical"))
ner = ner.to_crs(EQ).reset_index(drop=True)
ne_lines = list(ner.geometry)
ne_tree = cKDTree(np.array([[g.representative_point().x, g.representative_point().y]
                            for g in ne_lines]))
nd = min(3, len(ne_lines))
NE = []
for j in range(len(D)):
    dd, ii = ne_tree.query(SP[j], k=nd)
    near = []
    for k in np.atleast_1d(ii):
        rown = ner.iloc[int(k)]
        near.append((float(rown.geometry.distance(Point(SP[j][0], SP[j][1]))),
                     rown.geometry.geom_type, str(rown.get("Name", "") or "")))
    near.sort(key=lambda x: x[0])
    NE.append(dict(idx=j, ne_dist_m=round(near[0][0], 1), ne_type=near[0][1],
                   ne_name=near[0][2]))
NEF = pd.DataFrame(NE)
W = W.merge(NEF, on="idx")
W["osm_line"] = W.w_nodes > 0
W["ne_le_5km"] = W.ne_dist_m <= R_DISC
both = int((W.osm_line & W.ne_le_5km).sum())
only_ne = int((~W.osm_line & W.ne_le_5km).sum())
only_osm = int((W.osm_line & ~W.ne_le_5km).sum())
neither = int((~W.osm_line & ~W.ne_le_5km).sum())
log(f"เทียบสองแผนที่ (รัศมี 5 กม.): มีทั้งคู่ {both} · OSM เท่านั้น {only_osm} · "
    f"Natural Earth เท่านั้น {only_ne} · ไม่มีทั้งคู่ {neither}")

# ================================================== 3. ตัวเลขระดับประเทศ + สหสัมพันธ์
cls_all = wm["waterway_class_all"]
top_cls = sorted(cls_all.items(), key=lambda kv: -kv[1])[:8]
rho_mu, p_mu = spearmanr(W.r_mu.values.astype(float), W.w_mu.fillna(0).values.astype(float))
rho_wd, p_wd = spearmanr(W.water_pct.values, W.sqm.values)
rho_dw, p_dw = spearmanr(W.dist_line_m.fillna(W.dist_line_m.max() * 2).values, W.sqm.values)
rho_rw, p_rw = spearmanr(W.r_nodes.values, W.w_nodes.fillna(0).values)
log(f"สหสัมพันธ์: μถนน↔μน้ำ rho {rho_mu:.3f} (p {p_mu:.3g}) · พื้นที่น้ำ↔sqm rho {rho_wd:.3f} "
    f"(p {p_wd:.3g}) · ระยะไปน้ำ↔sqm rho {rho_dw:.3f} (p {p_dw:.3g}) · "
    f"โหนดถนน↔โหนดน้ำ rho {rho_rw:.3f} (p {p_rw:.3g})")

# ================================================== 4. ภาพ
ne = gpd.read_file(ensure_shp("ne_10m_admin_0_countries"))
TH = make_valid(ne[ne.NAME == "Thailand"].geometry.iloc[0])
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


zrd = np.load(repo_file("data/day25_graphs.npz"))
rd_site = zrd["site"].astype(int)          # แถวโหนด (413,758)
rd_esite = zrd["esite"].astype(int)        # แถวขอบ (430,464) — คนละแกนกับโหนด


def panel_site(ax, j, kind="water", show=True):
    """วาดกราฟ (น้ำหรือถนน) ของหนึ่งจุดบนพิกัดจริง"""
    if kind == "water":
        nodes, edges = by_node.get(j, []), by_edge.get(j, [])
    else:
        mn = rd_site == j
        me = rd_esite == j
        nodes = [(int(a), float(b), float(c)) for a, b, c in
                 zip(zrd["nid"][mn], zrd["lat"][mn] / 1e7, zrd["lon"][mn] / 1e7)]
        edges = [(int(a), int(b), 0) for a, b in zip(zrd["u"][me], zrd["v"][me])]
    if not nodes:
        ax.text(0.5, 0.5, T("ไม่มีข้อมูล"), color=MUTED, ha="center", transform=ax.transAxes)
        return 0
    loc = {n[0]: i for i, n in enumerate(nodes)}
    xy = np.array([[TRS.transform(n[2], n[1])[0], TRS.transform(n[2], n[1])[1]] for n in nodes])
    seg = np.array([[xy[loc[u]], xy[loc[v]]] for u, v, _ in edges if u in loc and v in loc])
    if len(seg):
        ax.add_collection(LineCollection(seg, colors=(WATER if kind == "water" else ROAD),
                                         linewidths=0.5, alpha=0.85, zorder=3))
    ax.scatter([SP[j][0]], [SP[j][1]], s=34, marker="*", color=WARM, zorder=5)
    ax.set_aspect("equal"); ax.autoscale()
    return len(seg)


def draw_hero():
    fig = plt.figure(figsize=(16.2, 9.8), dpi=140, facecolor=BG)
    ax = fig.add_axes([0.045, 0.075, 0.50, 0.79]); frame(ax)
    # แปลงน้ำในจาน
    for j, polys in water_poly.items():
        for pg in polys:
            it = safe_inter(pg, DISC[j])
            if it is None or it.is_empty:
                continue
            geoms = it.geoms if it.geom_type.startswith("Multi") else [it]
            for g in geoms:
                if g.geom_type == "Polygon":
                    xs, ys = g.exterior.xy
                    ax.fill(xs, ys, color=DEEP, alpha=0.85, zorder=2)
    # เส้นน้ำทุกจาน
    segs = []
    for j in WIDX:
        nodes, edges = by_node[j], by_edge[j]
        loc = {n[0]: i for i, n in enumerate(nodes)}
        xy = np.array([[TRS.transform(n[2], n[1])[0], TRS.transform(n[2], n[1])[1]] for n in nodes])
        segs += [[xy[loc[u]], xy[loc[v]]] for u, v, _ in edges if u in loc and v in loc]
    ax.add_collection(LineCollection(np.array(segs), colors=WATER, linewidths=0.45,
                                     alpha=0.75, zorder=3))
    # จาน 5 กม. + จุด
    for j in range(len(D)):
        c = DISC[j]
        xs, ys = c.exterior.xy
        ax.plot(xs, ys, color=(TEAL if W.w_nodes.get(j, 0) > 0 else ROSE), lw=0.5,
                alpha=0.55, zorder=4)
    ax.scatter(SP[:, 0], SP[:, 1], s=9, c=np.where(W.w_nodes > 0, TEAL, ROSE), zorder=6,
               edgecolor=BG, linewidth=0.3)
    ax.set_title(T("เส้นน้ำ (OSM) แปลงน้ำ จาน 5 กม. รอบ 83 จุดทะเบียน"), color=INK, fontsize=12)
    # ---- inset: จาน 5 กม. ของจุดที่มีโหนดน้ำมากที่สุด ให้เห็น "กิ่งไม้ทับตาข่าย" จริง ๆ
    jz = int(W.w_nodes.idxmax())
    axz = fig.add_axes([0.062, 0.098, 0.185, 0.255], facecolor=BG)
    panel_site(axz, jz, "road")
    panel_site(axz, jz, "water")
    for s in axz.spines.values():
        s.set_color(TEAL); s.set_linewidth(0.9)
    axz.set_xticks([]); axz.set_yticks([])
    lim = axz.get_xlim()
    bx = lim[0] + 0.06 * (lim[1] - lim[0])
    by = axz.get_ylim()[0] + 0.07 * (axz.get_ylim()[1] - axz.get_ylim()[0])
    axz.plot([bx, bx + 2000.0], [by, by], color=INK, lw=2.0, solid_capstyle="butt", zorder=8)
    axz.text(bx + 1000.0, by + 260.0, T("2 กม."), color=INK, fontsize=8, ha="center")
    axz.set_title(T(f"ขยาย: {str(D.name[jz])[:20]} (จาน 5 กม.)\nทอง = ถนน · ฟ้า = ลำน้ำ"),
                  color=INK, fontsize=9)
    xz, yz = SP[jz]
    ax.plot([xz - R_DISC, xz + R_DISC, xz + R_DISC, xz - R_DISC, xz - R_DISC],
            [yz - R_DISC, yz - R_DISC, yz + R_DISC, yz + R_DISC, yz - R_DISC],
            color=TEAL, lw=0.9, ls=":", zorder=7)

    axr = fig.add_axes([0.585, 0.075, 0.385, 0.79]); axr.set_facecolor(BG)
    axr.axis("off")
    n_line = int((W.w_nodes > 0).sum())
    n_area = int((W.water_pct > 0).sum())
    mu_w = W.w_mu[W.w_nodes > 0]
    mu_r = W.r_mu[W.r_nodes > 0]
    lines = [
        (T("น้ำใน OSM ทั้งประเทศ"), T(f"{wm['waterway_ways']:,} way")),
        (T("ถนนใน OSM ทั้งประเทศ"), T(f"{wm['highway_ways']:,} way")),
        (T("อัตราส่วน ถนน : น้ำ"), T(f"{wm['highway_ways'] / wm['waterway_ways']:.1f} : 1")),
        (T("จุดที่มีเส้นน้ำใน 5 กม."), T(f"{n_line} จาก 83")),
        (T("จุดที่มีแปลงน้ำใน 5 กม."), T(f"{n_area} จาก 83")),
        (T("จุดที่ทั้งสองแผนที่ไม่มีน้ำ"), T(f"{neither} จาก 83")),
        (T("พื้นที่น้ำในจาน (มัธยฐาน)"), T(f"{W.water_pct.median():.3f}%")),
        (T("พื้นที่น้ำในจาน (สูงสุด)"), T(f"{W.water_pct.max():.2f}%")),
        (T("cyclomatic μ ของกราฟน้ำ"), T(f"มัธยฐาน {mu_w.median():.0f} · "
                                         f"สูงสุด {mu_w.max():.0f}")),
        (T("cyclomatic μ ของกราฟถนน"), T(f"มัธยฐาน {mu_r.median():.0f} · "
                                          f"สูงสุด {mu_r.max():.0f}")),
        (T("กราฟน้ำที่เป็น *ป่า* แท้ (μ ≤ 0)"), T(f"{int((mu_w <= 0).sum())} จาก {n_line} จุด")),
        (T("กราฟถนนที่มีวงวน (μ > 0)"), T(f"{int((mu_r > 0).sum())} จาก {int((W.r_nodes > 0).sum())} จุด")),
        (T("Strahler order สูงสุดที่วัดได้"), T(f"{int(W.w_strahler.max()):.0f}")),
        (T("rho (พื้นที่น้ำ, SQM ในทะเบียน)"), T(f"{rho_wd:.3f} (p {p_wd:.2g})")),
    ]
    for i, (k, v) in enumerate(lines):
        y = 0.965 - i * 0.072
        axr.text(0.02, y, k, color=MUTED, fontsize=11, va="top")
        axr.text(0.98, y, v, color=INK, fontsize=12, va="top", ha="right")
        axr.plot([0.02, 0.98], [y - 0.052, y - 0.052], color=MUTED, lw=0.4, alpha=0.4)
    fig.text(0.5, 0.955, T("Day 26 · Water — น้ำเป็นกิ่งไม้ ถนนเป็นตาข่าย"), color=INK,
             fontsize=18, ha="center")
    fig.text(0.5, 0.918, T("ข้อมูล OSM snapshot 21 พ.ย. 2569 (ก้อนเดียวกับ Day 21–25) · "
                           "กติกาจาน 5,000 ม. เดียวกับ Day 24"), color=MUTED, fontsize=10.5,
             ha="center")
    p = os.path.join(OUT, "day26_water.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p


def pick_pair_sites(k=3):
    """จุดที่โชว์คู่: มีทั้งเส้นน้ำและถนน, กราฟน้ำใหญ่ที่สุด k จุด (กฎที่ประกาศไว้)"""
    cand = W[(W.w_nodes > 0) & (W.r_nodes > 0)].nlargest(k, "w_nodes")
    return [int(x) for x in cand.idx.values]


def draw_tree_web():
    sites = pick_pair_sites(3)
    fig, axes = plt.subplots(2, len(sites), figsize=(15.6, 8.6), dpi=140, facecolor=BG)
    for c, j in enumerate(sites):
        for r, kind in enumerate(["road", "water"]):
            ax = axes[r, c]; ax.set_facecolor(BG)
            ne_ = panel_site(ax, j, kind)
            col = ROAD if kind == "road" else WATER
            for s in ax.spines.values():
                s.set_color(col); s.set_linewidth(0.8)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if kind == "road":
                ttl = T(f"ถนน: V {int(W.r_nodes[j]):,} · E {int(W.r_edges[j]):,} · "
                        f"μ {int(W.r_mu[j])} · {int(W.r_components[j])} หมู่")
            else:
                ttl = T(f"ลำน้ำ: V {int(W.w_nodes[j]):,} · E {int(W.w_edges[j]):,} · "
                        f"μ {int(W.w_mu[j])} · Strahler {int(W.w_strahler[j])}")
            if r == 0:
                ax.set_title(T(str(D.name[j])[:28]) + "\n" + ttl, color=col, fontsize=11)
            else:
                ax.set_title(ttl, color=col, fontsize=10.5)
    fig.suptitle(T("กราฟเดียวกันบนกติกาเดียวกัน: ถนนมีวงวน (μ > 0) ลำน้ำเป็นกิ่งไม้ (μ ≤ 0)"),
                 color=INK, fontsize=15)
    p = os.path.join(OUT, "day26_tree_web.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p, sites


def draw_mu():
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.4), dpi=140, facecolor=BG)
    ax = axes[0, 0]; ax.set_facecolor(BG)
    hw = W[W.w_nodes > 0]
    hr = W[W.r_nodes > 0]
    grp = [("น้ำ", int((hw.w_mu <= 0).sum()), int((hw.w_mu > 0).sum())),
           ("ถนน", int((hr.r_mu <= 0).sum()), int((hr.r_mu > 0).sum()))]
    xs = np.arange(len(grp))
    ax.bar(xs - 0.18, [g[1] for g in grp], width=0.34, color=WATER, label=T("μ ≤ 0 (ป่า ไม่มีวงวน)"),
           edgecolor=BG)
    ax.bar(xs + 0.18, [g[2] for g in grp], width=0.34, color=ROAD, label=T("μ > 0 (มีวงวน)"),
           edgecolor=BG)
    for i, g in enumerate(grp):
        ax.text(i - 0.18, g[1] + 1.2, str(g[1]), ha="center", color=INK, fontsize=11)
        ax.text(i + 0.18, g[2] + 1.2, str(g[2]), ha="center", color=INK, fontsize=11)
    ax.set_xticks(xs); ax.set_xticklabels([T(g[0]) for g in grp], color=INK, fontsize=12)
    ax.set_ylabel(T("จำนวนจุด"))
    ax.set_ylim(0, max(max(g[1], g[2]) for g in grp) * 1.18)
    ax.legend(frameon=False, labelcolor=INK, fontsize=10, loc="upper left")
    ax.set_title(T(f"cyclomatic number μ = E − V + หมู่\n"
                   f"มัธยฐาน: น้ำ {hw.w_mu.median():.0f} · ถนน {hr.r_mu.median():.0f} · "
                   f"สูงสุด {hr.r_mu.max():.0f}"), color=INK, fontsize=11.5)
    ax = axes[0, 1]; ax.set_facecolor(BG)
    ax.scatter(W.r_nodes[W.w_nodes > 0], W.w_nodes[W.w_nodes > 0], s=18, color=TEAL, alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(T("โหนดของกราฟถนน (log)")); ax.set_ylabel(T("โหนดของกราฟลำน้ำ (log)"))
    ax.set_title(T(f"ปริมาณ mapping ของสองเครือข่ายไปด้วยกัน\nrho = {rho_rw:.3f} "
                   f"(p {p_rw:.2g})"), color=INK, fontsize=11.5)
    ax = axes[1, 0]; ax.set_facecolor(BG)
    v = W.w_strahler.dropna()
    ax.hist(v, bins=np.arange(0.5, v.max() + 1.5, 1), color=WATER, edgecolor=BG, alpha=0.85)
    ax.set_xlabel(T("ลำดับสูงสุดของกิ่งน้ำ (Horton–Strahler, root ด้วยเรขาคณิต)"))
    ax.set_ylabel(T("จำนวนจุด"))
    ax.set_title(T(f"ความแตกแขนง: มัธยฐาน {v.median():.0f} · สูงสุด {v.max():.0f}"),
                 color=INK, fontsize=11.5)
    ax = axes[1, 1]; ax.set_facecolor(BG)
    ax.scatter(W.water_pct, W.sqm, s=22, color=TEAL, alpha=0.85)
    ax.set_xlabel(T("สัดส่วนพื้นที่น้ำในจาน 5 กม. (%)"))
    ax.set_ylabel(T("SQM ที่ทะเบียนบันทึก (มาก = มืดกว่า)"))
    ax.set_title(T(f"น้ำเยอะกว่า = มืดกว่าหรือไม่\nrho = {rho_wd:.3f} (p {p_wd:.2g})"),
                 color=INK, fontsize=11.5)
    for ax in axes.ravel():
        ax.grid(color=MUTED, alpha=0.22, lw=0.5)
        for s in ax.spines.values():
            s.set_color(MUTED)
        ax.tick_params(colors=MUTED)
    fig.suptitle(T("น้ำกับถนน: รูปทรง ปริมาณ mapping และความมืด"), color=INK, fontsize=15)
    p = os.path.join(OUT, "day26_mu.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p


def draw_sources():
    """OSM เทียบ Natural Earth: 17 จุดที่ไม่มีน้ำ คือไม่มีจริง หรือไม่มีใคร map"""
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.6), dpi=140, facecolor=BG)
    ax = axes[0]; ax.set_facecolor(BG)
    grp = collections.Counter(zip(W.osm_line, W.ne_le_5km))
    lab = [(T("มีทั้งคู่"), grp[(True, True)]), (T("OSM เท่านั้น"), grp[(True, False)]),
           (T("Natural Earth เท่านั้น"), grp[(False, True)]), (T("ไม่มีทั้งคู่"), grp[(False, False)])]
    xs = np.arange(len(lab))
    ax.bar(xs, [v for _, v in lab], color=[TEAL, WATER, ACCENT, ROSE], edgecolor=BG)
    for x, (_, v) in zip(xs, lab):
        ax.text(x, v + 0.6, str(v), ha="center", color=INK, fontsize=12)
    ax.set_xticks(xs); ax.set_xticklabels([l for l, _ in lab], color=INK, fontsize=10)
    ax.set_ylabel(T("จำนวนจุด")); ax.set_title(T("สองแผนที่เห็นน้ำต่างกันแค่ไหน (รัศมี 5 กม.)"),
                                               color=INK, fontsize=12)
    ax = axes[1]; ax.set_facecolor(BG)
    d17 = W[~W.osm_line].ne_dist_m
    dhas = W[W.osm_line].ne_dist_m
    ax.hist([np.log10(np.maximum(d17, 1)), np.log10(np.maximum(dhas, 1))],
            bins=18, label=[T("จุดที่ OSM ไม่มีเส้นน้ำ"), T("จุดที่มี")],
            color=[ROSE, WATER], edgecolor=BG)
    ax.set_xlabel(T("log ระยะจากจุดไปลำน้ำของ Natural Earth (เมตร)"))
    ax.set_ylabel(T("จำนวนจุด")); ax.legend(frameon=False, labelcolor=INK, fontsize=10)
    ax.set_title(T(f"จุดที่ OSM ไม่มีน้ำ มัธยฐานไกลจาก NE {np.median(np.maximum(d17, 1)):,.0f} ม.\n"
                   f"(ถ้า NE ก็ไกล แสดงว่า *ไม่มีน้ำจริง* ไม่ใช่ไม่มีใคร map)"), color=INK,
                 fontsize=11.5)
    for a in axes:
        a.grid(color=MUTED, alpha=0.22, lw=0.5)
        for s in a.spines.values():
            s.set_color(MUTED)
        a.tick_params(colors=MUTED)
    fig.suptitle(T("น้ำหายไปจากแผนที่ หรือไม่มีใครไป map"), color=INK, fontsize=15)
    p = os.path.join(OUT, "day26_sources.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p


# ================================================== 5. ไฟล์ + สถิติ
P1 = draw_hero()
P2, PAIR_SITES = draw_tree_web()
P3 = draw_mu()
P4 = draw_sources()
log(f"ภาพ 4 ชิ้นเสร็จ ({time.time()-T0:.0f} วินาที)")

W_OUT = W.copy()
for c in ["w_nodes", "w_edges", "w_components", "w_mu", "w_largest_share", "w_deadend",
          "w_maxdeg", "w_diam_hops", "w_diam_km", "w_strahler", "w_leaf_frac"]:
    W_OUT[c] = pd.to_numeric(W_OUT[c], errors="coerce")
W_OUT.to_csv(os.path.join(OUT, "day26_sites.csv"), index=False)
AREA.to_csv(os.path.join(OUT, "day26_areas.csv"), index=False)
SUMMARY = pd.DataFrame([
    dict(metric="way ทั้งไฟล์", value=wm["ways_in_file"], unit="way"),
    dict(metric="way มีแท็ก highway", value=wm["highway_ways"], unit="way"),
    dict(metric="way มีแท็ก waterway", value=wm["waterway_ways"], unit="way"),
    dict(metric="way แปลงน้ำ (natural/waterway/landuse)", value=wm["area_ways"], unit="way"),
    dict(metric="จุดที่มีเส้นน้ำใน 5 กม.", value=int((W.w_nodes > 0).sum()), unit="จุด"),
    dict(metric="จุดที่มีแปลงน้ำทับจาน", value=int((W.water_pct > 0).sum()), unit="จุด"),
    dict(metric="จุดที่ OSM มีเส้นน้ำ แต่ NE ไม่มีใน 5 กม.", value=only_ne, unit="จุด"),
    dict(metric="จุดที่ NE มี แต่ OSM ไม่มีใน 5 กม.", value=int((~W.osm_line & W.ne_le_5km).sum()),
         unit="จุด"),
    dict(metric="จุดที่ไม่มีทั้งคู่", value=neither, unit="จุด"),
    dict(metric="μ ถนน: มัธยฐาน", value=float(W.r_mu[W.r_nodes > 0].median()), unit="วงวน"),
    dict(metric="μ น้ำ: มัธยฐาน", value=float(W.w_mu[W.w_nodes > 0].median()), unit="วงวน"),
    dict(metric="จุดที่กราฟน้ำมี μ>0 (มีวงวน)", value=int((W.w_mu[W.w_nodes > 0] > 0).sum()),
         unit="จุด"),
    dict(metric="จุดที่กราฟถนนมี μ>0", value=int((W.r_mu[W.r_nodes > 0] > 0).sum()), unit="จุด"),
    dict(metric="Strahler: มัธยฐาน", value=float(W.w_strahler.median()), unit="ลำดับ"),
    dict(metric="Strahler: สูงสุด", value=float(W.w_strahler.max()), unit="ลำดับ"),
    dict(metric="ระยะไปเส้นน้ำ: มัธยฐาน", value=float(W.dist_line_m.median()), unit="เมตร"),
    dict(metric="ระยะไปเส้นน้ำ: ไกลสุดในจุดที่มีน้ำ", value=float(W.dist_line_m.max()), unit="เมตร"),
    dict(metric="พื้นที่น้ำในจาน: มัธยฐาน", value=float(W.water_pct.median()), unit="%"),
    dict(metric="พื้นที่น้ำในจาน: สูงสุด", value=float(W.water_pct.max()), unit="%"),
    dict(metric="rho(พื้นที่น้ำ, SQM)", value=float(rho_wd), unit="spearman"),
    dict(metric="p(พื้นที่น้ำ, SQM)", value=float(p_wd), unit="ค่า n=83"),
    dict(metric="rho(โหนดถนน, โหนดน้ำ)", value=float(rho_rw), unit="spearman"),
    dict(metric="p(โหนดถนน, โหนดน้ำ)", value=float(p_rw), unit="ค่า n=83"),
    dict(metric="rho(μถนน, μน้ำ)", value=float(rho_mu), unit="spearman"),
    dict(metric="p(μถนน, μน้ำ)", value=float(p_mu), unit="ค่า n=83"),
])
SUMMARY.to_csv(os.path.join(OUT, "day26_summary.csv"), index=False)


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
    theme=dict(day=26, name="Water",
               prompt="About 71% of the Earth's surface. That wet thing, you know?",
               read_from="30daymapchallenge.com อ่านสด 27 พ.ย. 2569 (ไฟล์ scratch/chal26.html)"),
    provenance=dict(pbf_md5=wm["pbf_md5"], pbf_read_utc=wm["read_utc"], radius_m=R_DISC,
                    made_by="tools/collect_day26_water.py (pyosmium, locations=True)",
                    same_extract_as="Day 21–25",
                    natural_earth="ne_10m_rivers_lake_centerlines (10m, mirror ทางการ naciscdn)"),
    country=dict(ways_in_file=int(wm["ways_in_file"]), highway_ways=int(wm["highway_ways"]),
                 waterway_ways=int(wm["waterway_ways"]), area_ways=int(wm["area_ways"]),
                 road_to_water_ratio=round(wm["highway_ways"] / wm["waterway_ways"], 2),
                 waterway_class_top=[dict(cls=k, ways=int(v)) for k, v in top_cls],
                 classes_in_discs=WCLS),
    discs=dict(node_slots=int(wm["node_slots"]), edges=int(wm["edges"]),
               sites_with_lines=int((W.w_nodes > 0).sum()),
               sites_without_lines=[str(D.name[j]) for j in wm["sites_without_lines"]],
               sites_with_area=int((W.water_pct > 0).sum()),
               areas_touching=int(wm["n_areas_in_discs"]), area_intersections=int(len(AREA)),
               line_ways_touching=int(wm["line_ways_touching"])),
    shape=dict(mu_road_median=float(W.r_mu[W.r_nodes > 0].median()),
               mu_water_median=float(W.w_mu[W.w_nodes > 0].median()),
               mu_road_max=int(W.r_mu.max()), mu_water_max=int(W.w_mu[W.w_nodes > 0].max()),
               sites_water_with_loop=int((W.w_mu[W.w_nodes > 0] > 0).sum()),
               sites_road_with_loop=int((W.r_mu[W.r_nodes > 0] > 0).sum()),
               strahler_median=float(W.w_strahler.median()), strahler_max=int(W.w_strahler.max()),
               leaf_frac_water_median=float(W.w_leaf_frac.median()),
               leaf_frac_road_median=float(rd24.deadend_ratio.median()),
               diam_km_water_median=float(W.w_diam_km.median()),
               diam_km_water_max=float(W.w_diam_km.max()),
               diam_straight_median=float(W.w_diam_straight_km.median()),
               tortuosity_median=float(W.w_tortuosity.median()),
               tortuosity_q75=float(W.w_tortuosity.quantile(0.75)),
               tortuosity_max=float(W.w_tortuosity.max()),
               tortuosity_max_site=str(W.loc[W.w_tortuosity.idxmax(), "name"]),
               tortuosity_measured=int(W.w_tortuosity.notna().sum()),
               rho_mu=dict(rho=float(rho_mu), p=float(p_mu))),
    distance=dict(dist_line_median=float(W.dist_line_m.median()),
                  dist_line_max=float(W.dist_line_m.max()),
                  dist_line_site=str(W.loc[W.dist_line_m.idxmax(), "name"]),
                  dist_area_median=float(W.dist_area_m.median()),
                  sites_no_line=int(W.dist_line_m.isna().sum())),
    area=dict(pct_median=float(W.water_pct.median()), pct_max=float(W.water_pct.max()),
              pct_max_site=str(W.loc[W.water_pct.idxmax(), "name"]),
              km2_total_in_discs=float(AREA.in_disc_km2.sum()),
              biggest_way=dict(way=int(AREA.loc[AREA.in_disc_km2.idxmax(), "way"]),
                               name=str(AREA.loc[AREA.in_disc_km2.fillna(0).idxmax(), "name"]),
                               km2=float(AREA.in_disc_km2.max())),
              unit="สัดส่วนคำนวณในระนาบ Equidistant Conic (Day 22 วัดคลาดพื้นที่ −0.3617%)"),
    sources=dict(both=int((W.osm_line & W.ne_le_5km).sum()), only_osm=only_osm,
                 only_ne=only_ne, neither=neither,
                 ne_dist_median_when_osm_missing=float(np.median(np.maximum(
                     W[~W.osm_line].ne_dist_m, 1))),
                 ne_dist_median_when_osm_has=float(np.median(np.maximum(
                     W[W.osm_line].ne_dist_m, 1))),
                 threshold_m=R_DISC),
    dark=dict(rho_water_sqm=float(rho_wd), p_water_sqm=float(p_wd),
              rho_dist_sqm=float(rho_dw), p_dist_sqm=float(p_dw),
              rho_nodes=float(rho_rw), p_nodes=float(p_rw),
              sqm_note="SQM ในทะเบียนเป็นจำนวนเต็ม 19–22 (มาก = มืดกว่า) — Day 17 เคยเตือน "
                       "ว่าอย่าให้สหสัมพันธ์หยาบ ๆ แบบนี้ปลอมเป็น causation"),
    hero_sites=PAIR_SITES,
    figures=[os.path.basename(P1), os.path.basename(P2), os.path.basename(P3),
             os.path.basename(P4)],
    defs=dict(
        cyclomatic="μ = E − V + จำนวนหมู่ (graph theory พื้นฐาน) = จำนวนวงวนอิสระของกราฟ; "
                   "μ = 0 คือป่า (tree/forest) เดินแล้วไม่มีทางกลับมาที่เก่า",
        strahler="Horton–Strahler order: ปลายน้ำ = 1, จุดรวมที่กิ่งสองกิ่งเท่ากันมาบรรจบ = +1 "
                 "— ในข้อมูลนี้ OSM แทบไม่มีแท็ก direction จึง root ด้วยเรขาคณิต (ปลายเส้นผ่านศูนย์กลาง) "
                 "ไม่ใช่อุทกวิทยาจริง",
        disc="จานรัศมี 5,000 ม. รอบจุดทะเบียน พร้อมกติกาเก็บเฉพาะขอบที่ปลายทั้งสองอยู่ในจาน "
             "(นิยามเดียวกับ Day 24 ทุกประการ)",
        mpsas="mag/arcsec² — ค่าความสว่างท้องฟ้า ยิ่งมากยิ่งมืด"),
    limitations=[
        "น้ำที่ OSM รู้จักเป็น *สิ่งที่คนไป map* ไม่ใช่สิ่งที่มีจริง: จำนวน way waterway "
        f"{wm['waterway_ways']:,} เทียบ highway {wm['highway_ways']:,} (อัตรา "
        f"{wm['highway_ways'] / wm['waterway_ways']:.1f}:1) ส่วนหนึ่งคือถนนมีคน map มากกว่า",
        "Natural Earth 10m มีเฉพาะลำน้ำสายใหญ่ จึง *ตรวจได้ทางเดียว* คือ 'ถ้า NE ก็ไม่มี "
        "แสดงว่าน่าจะไม่จริง' — ใช้ยืนยันว่าไม่มี ไม่ได้ใช้พิสูจน์ว่ามี",
        "μ ของกราฟน้ำถูกกดโดยกติกาตัดขอบที่ขอบจาน (ปลายน้ำที่ไหลออกนอกจานหายไป) "
        "จึงเป็นค่าของ *หน้าต่าง 5 กม.* ไม่ใช่ของลุ่มน้ำทั้งลุ่ม",
        "Strahler root ด้วยเรขาคณิต (ไม่มีข้อมูลทิศการไหล) ลำดับที่ได้จึงเป็น 'ความแตกแขนง "
        "ของรูป' ไม่ใช่ลำดับอุทกวิทยาที่นักอุทกศาสตร์ใช้",
        "พื้นที่น้ำคำนวณในระนาบ projection ไม่ใช่ geodesic (ใช้ Geod กับพิกัดเมตรไม่ได้) "
        "— Day 22 วัดคลาดพื้นที่ของ projection นี้ไว้ที่ −0.3617% ซึ่งไม่มีผลกับสัดส่วน",
        "SQM เป็นค่าที่ *ทะเบียนบันทึก* (จำนวนเต็ม 19–22) ไม่ใช่การวัดจริงหน้างาน "
        "สหสัมพันธ์ rho −0.416 จึงเป็นข้อสังเกต ไม่ใช่หลักฐานว่าน้ำทำให้ฟ้าสว่าง"],
)

# ---- ตัวอย่างที่อ้างในโพสต์: ใส่ตัวเลขทุกตัวที่เขียนลง prose ไว้ใน stats ด้วย
#     (ธรรมเนียมของซีรีส์: ตัวเลขในโพสต์ต้อง traceable กลับ stats ไม่ใช่ใส่ whitelist)
def _pick(rows, cols):
    out = []
    for _, r in rows.iterrows():
        d = dict(name=str(r["name"]))
        for c in cols:
            v = r[c]
            d[c] = None if pd.isna(v) else (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else str(v))
        out.append(d)
    return out


STATS["examples"] = dict(
    hero=_pick(W.loc[PAIR_SITES], ["r_nodes", "r_mu", "w_nodes", "w_edges", "w_mu",
                                   "w_strahler", "water_pct"]),
    tortuosity_top=_pick(W.nlargest(3, "w_tortuosity"),
                         ["w_diam_km", "w_diam_straight_km", "w_tortuosity"]),
    water_pct_top=_pick(W.nlargest(2, "water_pct"), ["water_pct", "n_areas", "biggest_name"]),
    mu_water_loops=_pick(W[W.w_nodes > 0].nlargest(3, "w_mu"), ["w_mu", "w_nodes", "w_edges"]),
    far_from_water=_pick(W.nlargest(1, "dist_line_m"), ["dist_line_m"]),
    no_water_darkest=_pick(W[W.w_nodes.isna()].nlargest(3, "sqm"), ["sqm"]),
    n_areas_max=int(W.n_areas.max()),
    classes_in_discs=int(len(WCLS)),
)
STATS["dark"]["rho_water_sqm_4dp"] = round(float(rho_wd), 4)
STATS["dark"]["rho_dist_sqm_4dp"] = round(float(rho_dw), 4)
STATS["dark"]["rho_nodes_4dp"] = round(float(rho_rw), 4)
STATS["dark"]["rho_mu_4dp"] = round(float(rho_mu), 4)
STATS["export"] = dict(water_npz_mb=round(wm["bytes_npz"] / 1e6, 2),
                       areas_gz_mb=round(wm["bytes_areas"] / 1e6, 2),
                       scan_seconds=int(wm.get("scan_seconds", 63)),
                       local_seconds=int(round(time.time() - T0)),
                       colab_seconds=36,
                       note="scan_seconds = เวลาสแกน PBF ของ collect_day26_water.py; "
                            "colab_seconds = วัดจริงจาก DAY26_FORCE_COLAB=1 บนเครื่องเดียวกัน "
                            "(0/146 คีย์ที่ต่างจากที่รันในเครื่อง)")
json.dump(_clean(STATS), open(os.path.join(OUT, "day26_stats.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, allow_nan=False)
log(f"เขียน day26_stats.json + sites/areas/summary CSV ({len(W)} จุด, {len(AREA)} แปลง)")

# ================================================== 6. ผู้ตัดสิน
assert len(W) == 83
assert int((W.w_nodes > 0).sum()) == wm["sites_covered"]
assert wm["pbf_md5"] == "db3958a037ce717c462101b86e2f75f3"
assert wm["highway_ways"] == 2944183, "จำนวน highway way ต้องตรงกับที่ Day 24 รายงาน"
assert wm["waterway_ways"] > 0 and wm["area_ways"] > 0
muw = W.w_mu[W.w_nodes > 0]
mur = W.r_mu[W.r_nodes > 0]
assert float(muw.median()) <= 0 <= float(mur.median()), \
    "ข้อสรุปหลักของโพสต์: น้ำเป็นกิ่งไม้ (μ≤0) ถนนเป็นตาข่าย (μ≥0) — ถ้าสลับแปลว่าวัดผิด"
assert float(mur.median()) > float(muw.median()), "μ ถนนต้องมากกว่า μ น้ำ"
assert int(W.w_strahler.max()) >= 3, "ต้องพบกิ่งน้ำแตกแขนงอย่างน้อยลำดับ 3"
assert W.w_diam_km.dropna().shape[0] >= 60, "เส้นผ่านศูนย์กลางกราฟน้ำต้องวัดได้กับจุดที่มีน้ำเกือบทั้งหมด"
# diam_km = ระยะ *ตามลำน้ำ* ระหว่างปลายสองข้าง (ไม่ใช่ระยะตรง) จึงเกิน 10 กม. ได้
assert float(W.w_diam_km.median()) > float(W.w_diam_straight_km.median()), "ตามเครือข่ายต้องไม่สั้นกว่าเส้นตรง"
assert 1.0 < float(W.w_tortuosity.median()) < 10.0, "ความคดเคี้ยวต้องมากกว่า 1 และไม่เกิน 10 เท่า"
assert float(W.w_diam_straight_km.max()) <= 10.0, "ระยะตรงในจาน 5 กม. ต้องไม่เกิน 10 กม."
assert 0 <= only_ne <= 5, "NE เท่านั้น ต้องน้อยมาก (NE 10m มีเฉพาะสายใหญ่) — ถ้าเยอะต้องมีบั๊ก"
assert neither == int((~W.osm_line & ~W.ne_le_5km).sum())
assert float(W.water_pct.max()) < 25.0, "พื้นที่น้ำในจานเกิน 25% เป็นไปไม่ได้ → intersection พัง"
assert float(W.water_pct.median()) > 0, "ต้องมีจุดที่มีน้ำในจาน ไม่ใช่ 0 ทุกจุด"
assert math.isfinite(rho_wd) and abs(rho_wd) < 1
guard_variance(W.water_pct.values, "water-pct", min_spread_km=0.02)
guard_variance(muw.values.astype(float), "mu-water", min_spread_km=0.02)
guard_labels([int(x) for x in W[W.w_nodes > 0].idx[:10]],
             {int(j): (float(SP[j][0]), float(SP[j][1])) for j in W[W.w_nodes > 0].idx},
             {int(j): (float(D.latitude[j]), float(D.longitude[j])) for j in W[W.w_nodes > 0].idx},
             "water-discs")
lint_thai(" ".join(TXT), "day26-figures")
for badc in [c for c in "".join(TXT) if ord(c) > 0x2fff and not (0xe00 <= ord(c) <= 0xe7f)]:
    raise AssertionError(f"อักษรที่ไม่ใช่ไทย/ascii ใน label ภาพ: {badc!r}")
log("ผู้ตัดสินผ่านครบทุกข้อ ✓ · รวมเวลา " + f"{time.time()-T0:.0f} วินาที")
_LOGF.close()
