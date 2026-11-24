# -*- coding: utf-8 -*-
"""
Day 24 · Network — "Route network, transit network, human network or something completely
different."

โจทย์ (30daymapchallenge.com หัวข้อ Day 24 'Network' อ่านสด 20:18 น. วันที่ 24 พ.ย. 2569)
ซีรีส์นี้มีของให้เล่นสองชั้นที่ต่อกันเป็นคำถามเดียว:

  ชั้นที่ 1 (human network) — จากแคช Day 16 (59 วันก่อน): object 5,495 way พร้อม user
      และ timestamp ของ 83 จุดทะเบียน → ใครเขียนตรงไหนบ้าง = graphs two-layer
      (site↔editor เป็น bipartite; editor↔editor กับ site↔site เป็น projection)
      คำถาม: ชุมชนที่ map รอบแหล่งดูฟ้ามืดไทยเป็น *เครือข่ายเดียว* หรือ *กลุ่มเล็ก ๆ ที่ไม่เชื่อมถึงกัน*
      และคนที่ map ด้วยกัน "เขียนช่วงเวลาเดียวกัน" ไหม (ทดสอบด้วย temporal overlap)

  ชั้นที่ 2 (route network) — จาก cache/day24 (collect_day24_roads.py): กราฟถนนจริง
      ในจาน 5,000 ม. รอบแต่ละจุด (node = จุดตัด OSM, edge = ท่อนถนนที่ปลายทั้งสองอยู่ในจาน)
      Day 21 ตอบแล้วว่า "ถนนใกล้สุดไกลกี่เมตร" วันนี้ตอบต่อว่า
      **ถนนที่พาไปหาแหล่งดูดาวย attached กับเครือข่าย หรือเป็นทางตันที่เปราะ**
      วัดจาก: จำนวน components, ดีกรี, dead-end (ดีกรี 1), articulation points และ
      "ถ้าจุดตัดหนึ่งหายไป เหลือเครือข่ายให้ใช้เท่าไร" (worst single cut)

ทุกตัวเลขในโพสต์อ่านจาก day24_stats.json + day24_*.csv เท่านั้น
"""
import os, sys, json, time, math, gzip, collections, itertools
import numpy as np, pandas as pd, geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from shapely.validation import make_valid
from pyproj import CRS, Transformer
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

T0 = time.time()
COLAB = (os.environ.get("DAY24_FORCE_COLAB") == "1") or not os.path.isdir(
    os.path.join(r"D:\GIS30Days", "geo-notes"))
WORK = os.getcwd() if COLAB else r"D:\GIS30Days"
REPO = os.path.join(WORK, "geo-notes")
CACHE = os.path.join(WORK, "day24_cache") if COLAB else os.path.join(WORK, "cache", "day06")
D16 = os.path.join(WORK, "cache", "day16")
RDP = os.path.join(WORK, "cache", "day24")
OUT = os.path.join(WORK, "out", "day24")
os.makedirs(OUT, exist_ok=True)
RAW = "https://raw.githubusercontent.com/thammarat-ai/geo-notes/main/"


def ensure(url, dst):
    if not os.path.exists(dst):
        import urllib.request
        urllib.request.urlretrieve(url, dst)
    return dst


def repo_file(rel):
    if not COLAB:
        return os.path.join(REPO, rel)
    return ensure(RAW + rel, os.path.join(CACHE, os.path.basename(rel)))


if COLAB:
    for f in ("Trirong-Regular.ttf", "Trirong-Bold.ttf"):
        ensure("https://github.com/google/fonts/raw/main/ofl/trirong/" + f, os.path.join(CACHE, f))
    repo_file("data/darksky_thailand.csv")
    repo_file("posts/day16-collaborative-map/day16_sites_osm.csv")
for _f in sorted(os.listdir(CACHE)):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(CACHE, _f))
plt.rcParams["font.family"] = "Trirong"
_LOGF = open(os.path.join(OUT, "day24_run.log"), "w", encoding="utf-8")
def log(m):
    s = f"[{time.time()-T0:6.1f}s] {m}"
    print(s, flush=True)
    _LOGF.write(s + chr(10)); _LOGF.flush()

for _td in (os.path.join(WORK, "tools"), os.path.join(REPO, "tools")):
    if os.path.isdir(_td):
        sys.path.insert(0, _td); break
try:
    from guards import guard_bbox, guard_crs, guard_variance, guard_labels, lint_thai
except ModuleNotFoundError:
    ensure(RAW + "tools/guards.py", os.path.join(CACHE, "guards.py"))
    sys.path.insert(0, CACHE)
    from guards import guard_bbox, guard_crs, guard_variance, guard_labels, lint_thai

TXT = []
def T(s):
    TXT.append(s); return s

BG, INK, MUTED = "#0d1522", "#e8e3d3", "#9fb2c8"
ACCENT, WARM, COOL, TEAL, PURPLE, LIME, ROSE = ("#f5b942", "#f4566a", "#38bdf8", "#2dd4bf",
                                                "#c084fc", "#a3e635", "#fb7185")
EQ = CRS.from_proj4("+proj=eqdc +lat_0=14 +lon_0=101 +lat_1=8 +lat_2=20 +units=m +no_defs "
                    "+ellps=WGS84")
R_DISC = 5000.0

D = pd.read_csv(repo_file("data/darksky_thailand.csv"))
D["name"] = D["name"].astype(str).str.strip()
P = np.array([[float(r.longitude), float(r.latitude)] for _, r in D.iterrows()])
guard_bbox(P, 96.5, 107.6, 4.3, 21.7, "darksky-points")
trs = Transformer.from_crs("EPSG:4326", EQ, always_xy=True)
SP = np.array(trs.transform(P[:, 0], P[:, 1])).T
guard_crs([SP[:, 0].min(), SP[:, 1].min(), SP[:, 0].max(), SP[:, 1].max()],
          "projected-metres", expect="metres")

# ================================================== 1. เครือข่ายคน (จากแคช Day 16)
w16 = json.load(open(os.path.join(D16, "way_all.json"), encoding="utf-8"))
SITES_W = {int(k): v for k, v in w16["sites"].items()}
site_users, site_years, way_total = {}, {}, 0
for j, rows in SITES_W.items():
    u, yr = set(), set()
    for wid, user, ts, lat, lon in rows:
        way_total += 1
        u.add(user)
        yr.add(int(ts[:4]))
    site_users[j] = u
    site_years[j] = sorted(yr)
users = sorted({x for u in site_users.values() for x in u})
WAYS_TODAY, USERS_TODAY = way_total, len(users)
USERS = users
log(f"bipartite จาก Day 16: way {WAYS_TODAY:,} · editor {USERS_TODAY} · จุด {len(site_users)}")

# ---- editor network: สองคนเชื่อมกันเมื่อเขียน *รอบจุดเดียวกัน*
GU = nx.Graph()
GU.add_nodes_from(users)
pairs = collections.Counter()
for j, u in site_users.items():
    for a, b in itertools.combinations(sorted(u), 2):
        pairs[(a, b)] += 1
for (a, b), n in pairs.items():
    GU.add_edge(a, b, sites=n)
ISO_U = len(list(nx.isolates(GU)))
GU.remove_nodes_from(list(nx.isolates(GU)))
comp_u = list(nx.connected_components(GU))
big_u = max(comp_u, key=len)
deg_u = dict(GU.degree())
clust_u = nx.average_clustering(GU)
btw_u = nx.betweenness_centrality(GU)
btw = btw_u
top_btw = sorted(btw_u.items(), key=lambda kv: -kv[1])[:8]
span = {}
for j, rows in SITES_W.items():
    for wid, user, ts, lat, lon in rows:
        y = int(ts[:4])
        lo, hi = span.get(user, (y, y))
        span[user] = (min(lo, y), max(hi, y))
# temporal: เส้นเชื่อมจริงเมื่อช่วงปีที่สองคนเขียนทับกัน
GT = nx.Graph(); GT.add_nodes_from(GU.nodes)
overlap_pairs = 0
for a, b, d in GU.edges(data=True):
    if min(span[a][1], span[b][1]) >= max(span[a][0], span[b][0]):
        GT.add_edge(a, b, sites=d["sites"]); overlap_pairs += 1
comp_t = list(nx.connected_components(GT))

# ---- site network: สองจุดเชื่อมเมื่อมี editor ร่วมกัน
GS = nx.Graph(); GS.add_nodes_from(range(len(D)))
shared = collections.Counter()
for a, b in itertools.combinations(range(len(D)), 2):
    inter = site_users.get(int(a), set()) & site_users.get(int(b), set())
    if inter:
        GS.add_edge(a, b, w=len(inter)); shared[(a, b)] = len(inter)
GS.remove_nodes_from(list(nx.isolates(GS)))
comp_s = list(nx.connected_components(GS))
big_s = max(comp_s, key=len)
iso_s = sorted(set(range(len(D))) - set(GS.nodes))
log(f"editor network: โหนด {GU.number_of_nodes()} · ขอบ {GU.number_of_edges()} · "
    f"components {len(comp_u)} · ใหญ่สุด {len(big_u)} · clustering {clust_u:.4f} | "
    f"temporal โขด {GT.number_of_nodes()} ขอบ {GT.number_of_edges()} components {len(comp_t)}")
log(f"site network: โหนด {GS.number_of_nodes()} · ขอบ {GS.number_of_edges()} · "
    f"components {len(comp_s)} ใหญ่สุด {len(big_s)} · จุดที่ไม่เชื่อมกับใคร {len(iso_s)}")

# ---- ข้อมูลเทียบกลับกับ Day 16 (traceable)
d16 = pd.read_csv(repo_file("posts/day16-collaborative-map/day16_sites_osm.csv")) \
    if not COLAB else pd.read_csv(os.path.join(CACHE, "day16_sites_osm.csv"))
MULTI = sum(1 for u in users if sum(1 for j in site_users if u in site_users[j]) > 1)
per_site_users = [len(site_users[j]) for j in site_users]
comp_sizes_u = [len(c) for c in comp_u]
comp_sizes_s = [len(c) for c in comp_s]
comp_sizes_t = [len(c) for c in comp_t if any(GT.degree(x) > 0 for x in c)]
edge_pairs_temporal = GT.number_of_edges()
top_btw_list = [{"editor": a, "betweenness": round(v, 4),
                 "sites": sum(1 for j in site_users if a in site_users[j])}
                for a, v in top_btw]
top_deg = [{"editor": a, "degree": d,
            "sites": sum(1 for j in site_users if a in site_users[j])}
           for a, d in sorted(deg_u.items(), key=lambda kv: -kv[1])[:10]]
tiny = sorted(comp_u[-1]) if len(comp_u) > 1 else []
tiny_sites = sorted({j for j in site_users for x in tiny if x in site_users[j]})
iso_names = [(int(i), str(D.name[i]), str(D.province[i])) for i in iso_s]
site_deg = sorted(((int(i), int(d)) for i, d in GS.degree()), key=lambda kv: -kv[1])[:8]
log(f"components คนเขียน: {comp_sizes_u[:4]} · components จุด: {comp_sizes_s[:4]} · "
    f"components (แบบ timeframe ทับกัน): {comp_sizes_t[:4]}")
log(f"กลุ่มเล็กที่แยกออกไป: {tiny} → จุด {[(int(j), D.name[j]) for j in tiny_sites]}")
log(f"จุดที่ไม่มีใคร map ร่วมเลย: {iso_names}")

# ================================================== 2. กราฟถนนรอบแต่ละจุด (cache/day24)
rmeta = json.load(open(os.path.join(RDP, "roads_meta.json"), encoding="utf-8"))
R_DISC = float(rmeta["radius_m"])
NEAR_CUT_M, MAX_CUT_TRIALS = 2000.0, 40
zn = np.load(os.path.join(RDP, "roads_nodes.npz"))
ze = np.load(os.path.join(RDP, "roads_edges.npz"))
CLASSES = rmeta["classes"]
NODE_XY = {}
for j, nid, la, lo in zip(zn["site"], zn["nid"], zn["lat"], zn["lon"]):
    x, y = trs.transform(float(lo), float(la))
    NODE_XY[(int(j), int(nid))] = (float(x), float(y))
by_site_e = collections.defaultdict(list)
for j, u, v, c in zip(ze["site"], ze["u"], ze["v"], ze["cls"]):
    by_site_e[int(j)].append((int(u), int(v), int(c)))
UNIQ_EDGES = len({(min(u, v), max(u, v)) for u, v, _ in
                  [e for lst in by_site_e.values() for e in lst]})
log(f"กราฟถนน: node-slot {len(NODE_XY):,} · ขอบ {len(ze['site']):,} "
    f"(ไม่ซ้ำกันทั้งประเทศ {UNIQ_EDGES:,}) · ประเภททางที่เจอ {len(CLASSES)} ค่า · "
    f"จุดที่มีกราฟ {rmeta['sites_covered']}/83 (ไม่มี: {[int(x) for x in rmeta['sites_without_roads']]})")


def d_of(node, j):
    x, y = NODE_XY[(j, node)]
    return math.hypot(x - SP[j][0], y - SP[j][1])


road_rows = []
for j in range(len(D)):
    G = nx.Graph()
    for u, v, c in by_site_e.get(j, []):
        G.add_edge(u, v, cls=CLASSES[c])
    n, m = G.number_of_nodes(), G.number_of_edges()
    row = dict(idx=j, name=D.name[j], province=D.province[j], nodes=n, edges=m)
    if n == 0:
        row.update(components=0, largest=0, largest_edge_share=np.nan, deadend_ratio=np.nan,
                   avg_deg=np.nan, bridges=0, articulations=0, near_cuts=0,
                   access_dist_m=np.nan, access_in_largest=None, access_comp_nodes=0,
                   worst_cut_keep=np.nan, top_class="")
        road_rows.append(row)
        continue
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    d = dict(G.degree())
    acc = min(G.nodes, key=lambda k: d_of(k, j))
    ai = next(i for i, c in enumerate(comps) if acc in c)
    sub = G.subgraph(comps[ai]).copy()
    base = sub.number_of_edges()
    bridges = sum(1 for _ in nx.bridges(nx.MultiGraph(sub.edges())))
    aps = list(nx.articulation_points(sub)) if sub.number_of_nodes() > 2 else []
    # กันกรณีเพี้ยน: ถ้าตัว node ที่ใกล้ที่สุด *เป็น* articulation point เอง
    # การลบมันจะทำให้เหลือ 0% โดยนิยาม ซึ่งไม่ใช่ "ความเปราะ" ของเครือข่าย
    near = [a for a in aps if a != acc and d_of(a, j) <= NEAR_CUT_M]
    keep = []
    for a in near[:MAX_CUT_TRIALS]:
        H2 = sub.copy()
        H2.remove_node(a)
        if acc in H2:
            reach = nx.node_connected_component(H2, acc)
            keep.append(sum(1 for x, y in H2.edges() if x in reach or y in reach) /
                        max(1, base))
        else:
            keep.append(0.0)
    cc = collections.Counter(CLASSES[c] for _, _, c in by_site_e.get(j, []))
    row.update(components=len(comps), largest=len(comps[0]),
               largest_edge_share=round(sum(1 for u, v, _ in by_site_e[j]
                                            if u in comps[0] and v in comps[0]) / m, 4),
               deadend_ratio=round(sum(1 for x in d.values() if x == 1) / n, 4),
               avg_deg=round(2 * m / n, 3), bridges=bridges, articulations=len(aps),
               near_cuts=len(near), access_dist_m=round(d_of(acc, j), 1),
               access_in_largest=bool(ai == 0), access_comp_nodes=len(comps[ai]),
               worst_cut_keep=(round(min(keep), 4) if keep else 1.0),
               top_class=cc.most_common(1)[0][0])
    road_rows.append(row)
RD = pd.DataFrame(road_rows)
HASR = RD[RD.nodes > 0]
guard_variance(HASR.access_dist_m.values, "site-to-road-node-distance")
guard_variance(HASR.deadend_ratio.values, "deadend-ratio-spread", min_spread_km=0.02)
log(f"ถนน: components เฉลี่ย {HASR.components.mean():.2f} · จุดที่กราฟขาดเป็นเสี่ยง "
    f"{int((HASR.components > 1).sum())} · dead-end ratio มัธยฐาน "
    f"{HASR.deadend_ratio.median():.4f} · bridges รวม {int(HASR.bridges.sum()):,} · "
    f"articulation รวม {int(HASR.articulations.sum()):,}")
fragile = RD[(RD.nodes > 0) & (RD.worst_cut_keep < 0.5)]
log(f"จุดที่ 'ถ้าจุดตัดสำคัญใกล้ตัวหายไป' แล้วเหลือเครือข่ายไม่ถึงครึ่ง: {len(fragile)} จุด")

# ================================================== 3. เทียบกลับ Day 21 / Day 16
d21 = pd.read_csv(os.path.join(WORK, "out", "day21", "day21_sites_osm.csv"))
CH = RD.merge(d21[["idx", "road_min_m", "road_class", "road_n"]], on="idx")
CH["access_minus_edge_m"] = CH.access_dist_m - CH.road_min_m
both = CH[CH.access_dist_m.notna() & CH.road_min_m.notna()]
rho_21, p_21 = spearmanr(both.access_dist_m.values, both.road_min_m.values)
log(f"cross-check Day 21 (ขอบถนน 2 กม.) vs Day 24 (node ถนน 5 กม.): rho = {rho_21:.3f} "
    f"(p {p_21:.3g}) · ผลต่างสัมบูรณ์มัธยฐาน {both.access_minus_edge_m.abs().median():.1f} ม. "
    f"· มากสุด {both.access_minus_edge_m.abs().max():.1f} ม.")
d16c = pd.read_csv(repo_file("posts/day16-collaborative-map/day16_sites_osm.csv"))
JD = RD.merge(d16c[["idx", "ways", "users", "urban_pop_30km", "holder"]], on="idx")
rho_w, p_w = spearmanr(JD.edges.values, JD.ways.values)
rho_de, p_de = spearmanr(HASR.deadend_ratio.values,
                         np.log1p(JD.loc[JD.nodes > 0, "urban_pop_30km"].values))
rho_cp, p_cp = spearmanr(HASR.components.values,
                         np.log1p(JD.loc[JD.nodes > 0, "urban_pop_30km"].values))
rho_us, p_us = spearmanr(HASR.nodes.values, HASR.in_largest_nodes if False else
                         HASR.largest.values)
log(f"Spearman: ขอบถนน 5 กม. ↔ way 1 กม. (Day 16) {rho_w:.3f} (p {p_w:.3g}) · "
    f"dead-end ↔ ประชากรเมือง {rho_de:.3f} (p {p_de:.3g}) · components ↔ ประชากรเมือง "
    f"{rho_cp:.3f} (p {p_cp:.3g})")

# ================================================== 4. ภาพ
ne = gpd.read_file(os.path.join(CACHE, "ne_10m_admin_0_countries.shp"))
TH = make_valid(ne[ne.NAME == "Thailand"].geometry.iloc[0])
thp = gpd.GeoSeries([TH], crs="EPSG:4326").to_crs(EQ).iloc[0]
POS = nx.spring_layout(GS, seed=24, k=0.9)
POSU = nx.spring_layout(GU, seed=24, k=0.25)
COMP_COLOR = {}
for i, c in enumerate(comp_u):
    for x in c:
        COMP_COLOR[x] = [TEAL, WARM, ACCENT, PURPLE, COOL][i % 5]


def draw_hero():
    fig = plt.figure(figsize=(16.2, 9.6), dpi=140)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.015, 0.045, 0.40, 0.845]); ax.set_facecolor(BG)
    for g in (thp.geoms if thp.geom_type == "MultiPolygon" else [thp]):
        x, y = g.exterior.xy
        ax.plot(x, y, color="#2b3b57", lw=0.9, zorder=1)
        ax.fill(x, y, color="#111c2b", zorder=0)
    for a, b, dd in GS.edges(data=True):
        ax.plot([SP[a][0], SP[b][0]], [SP[a][1], SP[b][1]], color="#55708f",
                lw=0.25 + 0.9 * min(3, dd["w"]), alpha=0.42, zorder=2)
    for c in comp_s:
        ax.scatter([SP[i][0] for i in c], [SP[i][1] for i in c], s=64, color=TEAL,
                   edgecolor=BG, lw=0.6, zorder=4)
    if iso_s:
        ax.scatter([SP[i][0] for i in iso_s], [SP[i][1] for i in iso_s], s=118, marker="*",
                   color=WARM, edgecolor=BG, lw=0.7, zorder=5,
                   label=f"จุดที่ไม่มี mapper ร่วมเลย · {len(iso_s)}")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#33415e")
    ax.set_title(T("เครือข่าย *จุด* — เส้นเชื่อมเมื่อมีคนที่ map ทั้งสองแห่ง\n"
                   f"{GS.number_of_nodes()} จุดต่อกันเป็น {len(comp_s)} หมู่ · "
                   f"ขอบ {GS.number_of_edges():,}"), color=INK, fontsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.02), facecolor="#111c2b",
              edgecolor="#33415e", labelcolor=INK, fontsize=8.4, framealpha=0.96)
    ax2 = fig.add_axes([0.445, 0.045, 0.345, 0.845]); ax2.set_facecolor(BG)
    for a, b, dd in GU.edges(data=True):
        ax2.plot([POSU[a][0], POSU[b][0]], [POSU[a][1], POSU[b][1]], color="#33507a",
                 lw=0.22, alpha=0.5, zorder=2)
    for c in comp_u:
        col = COMP_COLOR[next(iter(c))]
        ax2.scatter([POSU[x][0] for x in c], [POSU[x][1] for x in c], s=13, color=col,
                    alpha=0.72, zorder=3)
    for x in ():
        ax2.scatter([POSU[x][0]], [POSU[x][1]], s=26, color=MUTED, marker="s", zorder=3)
    for rank, (a, v) in enumerate(top_btw[:3]):
        ax2.annotate(f"{a[:14]} ({v:.3f})", POSU[a], textcoords="offset points",
                     xytext=(13, 11), color=INK, fontsize=8.4, zorder=6,
                     arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, alpha=0.8))
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values(): sp.set_color("#33415e")
    ax2.set_title(T("เครือข่าย *คน* — สองคนเชื่อมเมื่อ map รอบจุดเดียวกัน\n"
                    f"{GU.number_of_nodes()} คน · "
                    f"{GU.number_of_edges():,} คู่ · {len(comp_u)} หมู่ · ป้าย = betweenness 3 อันดับแรก"),
                  color=INK, fontsize=11)
    txt = fig.add_axes([0.795, 0.035, 0.198, 0.855]); txt.axis("off"); txt.set_facecolor(BG)
    lines = [(T("editor ทั้งหมด"), f"{len(USERS)}"),
             (T("  ที่เชื่อมกับคนอื่น"), f"{GU.number_of_nodes()}"),
             (T("  ที่ไม่มีใคร map ร่วม"), f"{ISO_U}"),
             (T("คู่ editor ที่ map จุดเดียวกัน"), f"{GU.number_of_edges():,}"),
             (T("components ของคน"), f"{len(comp_u)}  ใหญ่สุด {len(big_u)}"),
             (T("  หมู่อีกหมู่"), f"{sorted(comp_sizes_u)[-1] if len(comp_sizes_u) > 1 else 0} คน"),
             (T("  เป็นใคร"), (" · ".join(x[:9] for x in tiny))[:26]),
             (T("  map จุด"), (" · ".join(str(D.name[int(j)])[:10] for j in tiny_sites))[:26]),
             ("", ""),
             (T("clustering เฉลี่ย"), f"{clust_u:.4f}"),
             (T("editor ที่เขียน >1 จุด"), f"{MULTI} ({100.0*MULTI/len(USERS):.1f}%)"),
             (T("จุดที่ไม่มีใคร map ร่วม"), f"{len(iso_s)} จาก 83"),
             (T("components ของจุด"), f"{len(comp_s)} (ใหญ่สุด {len(big_s)})"),
             ("", ""),
             (T("ถ้ากรองด้วย 'ปีที่ map ทับกัน'"), ""),
             (T("  ขอบที่เหลือ"), f"{GT.number_of_edges():,} จาก "
                                 f"{GU.number_of_edges():,}"),
             (T("  คนที่ยังเชื่อมอยู่"), f"{sum(1 for _, x in GT.degree() if x > 0)}"),
             (T("  components"), f"{len(comp_t)}"),
             ("", ""),
             (T('edge ที่ map ร่วมกันเฉลี่ย'), f"{np.mean([dd['w'] for *_, dd in GS.edges(data=True)]):.2f} คน"),
             (T("จุดที่เชื่อมมากที่สุด"), f"{D.name[int(site_deg[0][0])][:20]} ({site_deg[0][1]})"),
             ("", ""),
             (T("way ที่อ่านมา (Day 16)"), f"{WAYS_TODAY:,}"),
             (T("จุดทะเบียน"), "83"),
             ("", ""),
             (T("ถนน: ขอบทั้งประเทศ (ไม่ซ้ำ)"), f"{UNIQ_EDGES:,}"),
             (T("ถนน: components เฉลี่ยต่อจุด"), f"{HASR.components.mean():.2f}"),
             (T("  จุดที่กราฟขาดเป็นเสี่ยง"), f"{int((HASR.components > 1).sum())}"),
             (T("  dead-end ratio มัธยฐาน"), f"{HASR.deadend_ratio.median():.4f}"),
             (T("  bridges รวม"), f"{int(HASR.bridges.sum()):,}"),
             (T("  จุดที่เปราะ (เหลือ <50%)"), f"{len(fragile)}")]
    yy = 0.995
    for k, v in lines:
        if not k:
            yy -= 0.012; continue
        txt.text(0, yy, str(k)[:44], color=MUTED, fontsize=8.0, va="top")
        txt.text(1, yy, v, color=ACCENT, fontsize=8.3, va="top", ha="right")
        yy -= 0.0278
    fig.suptitle(T("สองเครือข่ายของแหล่งดูฟ้ามืด: ใครเป็นคนวาด และถนนพาไปได้ไกลแค่ไหน"),
                 color=INK, fontsize=15.5, y=0.985)
    pth = os.path.join(OUT, "day24_networks.png")
    fig.savefig(pth, facecolor=BG); plt.close(fig)
    log(f"เขียน {pth}")
    return pth


def draw_roads():
    fig = plt.figure(figsize=(15.4, 6.9), dpi=140)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.055, 0.13, 0.28, 0.74]); ax.set_facecolor("#111c2b")
    ax.hist(HASR.components, bins=np.arange(0.5, max(12, HASR.components.max()) + 1.5) - 0.5,
            color=COOL, edgecolor=BG, lw=0.4)
    ax.set_xlabel(T("จำนวน components ของกราฟถนนในจาน 5 กม."), color=INK)
    ax.set_ylabel(T("จำนวนจุดทะเบียน"), color=INK)
    ax.set_title(T(f"จุดที่กราฟขาดเป็นเสี่ยง ๆ มี {int((HASR.components > 1).sum())} จาก "
                   f"{len(HASR)}\n(จานตัดถนนที่ขอบด้วย = บางเสี่ยงคือของจริง บางเสี่ยงคือกรอบ)"),
                 color=INK, fontsize=11.5)
    ax.grid(color="#26364f", lw=0.5, alpha=0.5, axis="y")
    ax.tick_params(colors=INK, labelsize=9)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    ax2 = fig.add_axes([0.395, 0.13, 0.28, 0.74]); ax2.set_facecolor("#111c2b")
    ax2.hist(HASR.deadend_ratio, bins=24, color=TEAL, edgecolor=BG, lw=0.4)
    ax2.axvline(float(HASR.deadend_ratio.median()), color=WARM, ls="--", lw=1.2)
    ax2.text(float(HASR.deadend_ratio.median()) + 0.004, ax2.get_ylim()[1] * 0.86,
             T(f"มัธยฐาน {HASR.deadend_ratio.median():.3f}"), color=WARM, fontsize=9.5)
    ax2.set_xlabel(T("สัดส่วนจุดตัดดีกรี 1 (ทางตัน) ของกราฟรอบจุด"), color=INK)
    ax2.set_ylabel(T("จำนวนจุดทะเบียน"), color=INK)
    ax2.set_title(T("ทางตันมาก = กราฟเป็นกิ่งก้าน ไม่ใช่ตาข่าย\n"
                    "(กราฟเมืองจะตันน้อยกว่า เพราะมีวงรอบ)"), color=INK, fontsize=11.5)
    ax2.grid(color="#26364f", lw=0.5, alpha=0.5, axis="y")
    ax2.tick_params(colors=INK, labelsize=9)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)

    ax3 = fig.add_axes([0.735, 0.13, 0.245, 0.74]); ax3.set_facecolor("#111c2b")
    cls_all = collections.Counter()
    for lst in by_site_e.values():
        for _, _, c in lst:
            cls_all[CLASSES[c]] += 1
    top = cls_all.most_common(9)
    ax3.barh(np.arange(len(top)), [v for _, v in top], color=PURPLE, height=0.66)
    ax3.set_yticks(np.arange(len(top)))
    ax3.set_yticklabels([k for k, _ in top], color=INK, fontsize=8.6)
    for i, (k, v) in enumerate(top):
        ax3.text(v * 1.02, i, f"{v:,}", color=INK, fontsize=8, va="center")
    ax3.set_xlabel(T("จำนวนขอบถนนในแต่ละประเภท (83 จาน)"), color=INK)
    ax3.set_title(T("เครือข่ายรอบแหล่งดูฟ้ามืด\nสร้างจากถนนประเภทอะไร"), color=INK, fontsize=11.5)
    ax3.set_xlim(0, max(v for _, v in top) * 1.22)
    ax3.grid(color="#26364f", lw=0.5, alpha=0.5, axis="x")
    ax3.tick_params(colors=INK, labelsize=8.6)
    for sp in ("top", "right"): ax3.spines[sp].set_visible(False)
    fig.suptitle(T("กราฟถนนรอบจุดทะเบียน 83 แห่ง"), color=INK, fontsize=15, y=0.975)
    pth = os.path.join(OUT, "day24_roads.png")
    fig.savefig(pth, facecolor=BG); plt.close(fig)
    log(f"เขียน {pth}")
    return pth


def draw_fragile():
    fig = plt.figure(figsize=(15.0, 6.6), dpi=140)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.205, 0.13, 0.34, 0.74]); ax.set_facecolor("#111c2b")
    F = RD[(RD.nodes > 0)].sort_values("worst_cut_keep").head(12)
    yy = np.arange(len(F))
    ax.barh(yy, F.worst_cut_keep * 100, color=[WARM if v < 50 else ACCENT if v < 80 else TEAL
                                              for v in F.worst_cut_keep * 100], height=0.66)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{str(n)[:16]} · {str(p)[:10]}" for n, p in
                        zip(F.name, F.province)], color=INK, fontsize=7.8)
    for i, (v, nb) in enumerate(zip(F.worst_cut_keep * 100, F.near_cuts)):
        ax.text(v + 1, i, f"{v:.0f}% · จุดสำคัญใกล้ตัว {int(nb)}", color=MUTED, fontsize=7.6,
                va="center")
    ax.axvline(50, color=INK, ls=":", lw=1)
    ax.set_xlim(0, 118)
    ax.set_xlabel(T("ขอบที่ยังเหลืออยู่ในเครือข่าย หลังตัดจุดตัดที่สำคัญใกล้ที่สุดหนึ่งจุด (%)"),
                  color=INK)
    ax.set_title(T("ถ้าจุดตัดเดียวหายไป จะเหลือเส้นทางให้ใช้อยู่เท่าไร\n"
                   "(ตัดเฉพาะ articulation point ภายใน 2 กม. จากจุดทะเบียน)"),
                 color=INK, fontsize=11.5)
    ax.grid(color="#26364f", lw=0.5, alpha=0.5, axis="x")
    ax.tick_params(colors=INK, labelsize=8.4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    ax2 = fig.add_axes([0.60, 0.15, 0.355, 0.70]); ax2.set_facecolor("#111c2b")
    ax2.scatter(JD.ways, JD.edges, s=46, color=COOL, edgecolor=BG, lw=0.5, zorder=3)
    for r in JD.nlargest(4, "edges").itertuples():
        ax2.annotate(str(r.name)[:11], (r.ways, r.edges), textcoords="offset points",
                     xytext=(-30, 8), color=MUTED, fontsize=7.0)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(T("way ในรัศมี 1,000 ม. (Day 16)"), color=INK)
    ax2.set_ylabel(T("ขอบกราฟถนนในรัศมี 5,000 ม. (วันนี้)"), color=INK)
    ax2.set_title(T("สองกราฟสองรัศมีต้องไล่เลี่ยกัน\n"
                    f"Spearman rho = {rho_w:.3f} (p = {p_w:.3g})"), color=INK, fontsize=11.5)
    ax2.grid(color="#26364f", lw=0.5, alpha=0.5, which="both")
    ax2.tick_params(colors=INK, labelsize=8.6)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    fig.suptitle(T("เครือข่ายที่ขาดจุดเดียวไม่ได้"), color=INK, fontsize=15, y=0.975)
    pth = os.path.join(OUT, "day24_fragile.png")
    fig.savefig(pth, facecolor=BG); plt.close(fig)
    log(f"เขียน {pth}")
    return pth


def draw_temporal():
    fig = plt.figure(figsize=(15.0, 6.4), dpi=140)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.06, 0.13, 0.42, 0.74]); ax.set_facecolor("#111c2b")
    yrs = sorted(span.values(), key=lambda t: t[0])
    ys = np.arange(len(yrs))
    ax.fill_betweenx(ys, [a for a, _ in yrs], [b for _, b in yrs], color=COOL, alpha=0.28,
                     zorder=2)
    ax.plot([a for a, _ in yrs], ys, "o", color=COOL, ms=2.6, zorder=3)
    ax.plot([b for _, b in yrs], ys, "o", color=WARM, ms=2.6, zorder=3,
            label=T("ปีที่ map ครั้งแรก / ครั้งสุดท้าย (299 คน)"))
    ax.set_yticks([])
    ax.set_xlabel(T("ปี ค.ศ. ที่กิจกรรมเกิดขึ้น"), color=INK)
    ax.set_title(T("ช่วงเวลา map ของแต่ละคน — ก่อนอ้างว่า 'ทำงานร่วมกัน' ต้องทับกันจริง\n"
                   f"กรองแล้วขอบลดจาก {GU.number_of_edges():,} เหลือ "
                   f"{GT.number_of_edges():,}"), color=INK, fontsize=11.5)
    ax.legend(facecolor="#111c2b", edgecolor="#33415e", labelcolor=INK, fontsize=8.6,
              loc="lower right")
    ax.grid(color="#26364f", lw=0.5, alpha=0.5, axis="x")
    ax.tick_params(colors=INK, labelsize=9)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)

    ax2 = fig.add_axes([0.55, 0.13, 0.42, 0.74]); ax2.set_facecolor("#111c2b")
    degs = np.array(sorted(deg_u.values(), reverse=True))
    cnt = collections.Counter(degs.tolist())
    xs = np.array(sorted(cnt))
    ax2.bar(xs, [cnt[k] for k in xs], color=TEAL, width=0.72)
    ax2.set_yscale("log")
    ax2.set_xlabel(T("ดีกรีในเครือข่ายคน (จำนวนคนที่ map จุดเดียวกัน)"), color=INK)
    ax2.set_ylabel(T("จำนวน editor (log)"), color=INK)
    ax2.set_title(T("คนส่วนใหญ่เชื่อมไม่กี่คน แต่มี hub ชัดเจน\n"
                    f"สูงสุด {int(degs.max())} คน (degree) · ค่ามัธยฐาน "
                    f"{float(np.median(degs)):.0f}"), color=INK, fontsize=11.5)
    ax2.grid(color="#26364f", lw=0.5, alpha=0.5, axis="y", which="both")
    ax2.tick_params(colors=INK, labelsize=9)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    fig.suptitle(T("เครือข่ายคนวาด: เวลาและดีกรี"), color=INK, fontsize=15, y=0.975)
    pth = os.path.join(OUT, "day24_people.png")
    fig.savefig(pth, facecolor=BG); plt.close(fig)
    log(f"เขียน {pth}")
    return pth

# ================================================== 5. artifact + สถิติ
P1, P2, P3, P4 = draw_hero(), draw_roads(), draw_fragile(), draw_temporal()
RD.to_csv(os.path.join(OUT, "day24_roads.csv"), index=False)
pd.DataFrame([dict(editor=a, degree=deg_u[a], sites=sum(1 for j in site_users if a in site_users[j]),
                   first=span[a][0], last=span[a][1],
                   betweenness=round(btw.get(a, 0.0), 5),
                   component=int(next(i for i, c in enumerate(comp_u) if a in c)),
                   temporal_edges=sum(1 for _ in GT.edges(a)) if a in GT else 0)
              for a in USERS]).sort_values(["degree"], ascending=False).to_csv(
    os.path.join(OUT, "day24_editors.csv"), index=False)
GS_edges = [{"a": int(a), "a_name": str(D.name[a]), "b": int(b), "b_name": str(D.name[b]),
               "shared_editors": int(dd["w"])} for a, b, dd in GS.edges(data=True)]
pd.DataFrame(GS_edges).to_csv(os.path.join(OUT, "day24_site_edges.csv"), index=False)
pd.DataFrame([dict(idx=int(i), name=str(D.name[i]), province=str(D.province[i]),
                   editors=len(site_users[i]), degree=int(dict(GS.degree()).get(i, 0)),
                   component=int(next((k for k, c in enumerate(comp_s) if i in c), -1)),
                   edges_within_disc=int(RD.edges[i]),
                   components_road=int(RD.components[i]))
              for i in range(len(D))]).to_csv(os.path.join(OUT, "day24_sites.csv"), index=False)

def sci(x):
    if x == 0 or not np.isfinite(x):
        return dict(mantissa=0.0, exponent=0, exponent_abs=0)
    t = f"{x:.2g}"
    if "e" not in t:
        return dict(mantissa=float(t), exponent=0, exponent_abs=0)
    m, e = t.split("e")
    return dict(mantissa=float(m), exponent=int(e), exponent_abs=abs(int(e)))


AGG = dict(
    ways=WAYS_TODAY, editors=len(USERS), editors_isolated=ISO_U,
    editor_nodes_in_network=int(GU.number_of_nodes()),
    editor_edges=int(GU.number_of_edges()),
    editor_components=len(comp_u), editor_comp_sizes=comp_sizes_u[:5],
    editor_largest=len(big_u), clustering=round(float(clust_u), 5),
    multi_site_editors=MULTI, multi_site_pct=round(100.0 * MULTI / len(USERS), 2),
    temporal_edges=int(GT.number_of_edges()), temporal_nodes_active=int(
        sum(1 for _, x in GT.degree() if x > 0)),
    temporal_components=len(comp_t), temporal_comp_sizes=[len(x) for x in comp_t][:5],
    site_nodes=int(GS.number_of_nodes()), site_edges=int(GS.number_of_edges()),
    site_components=len(comp_s), site_largest=len(big_s), isolated_sites=len(iso_s),
    isolated_site_names=[{"idx": int(i), "name": str(D.name[i]),
                          "province": str(D.province[i])} for i in iso_s],
    shared_per_edge=round(float(np.mean([dd["w"] for *_, dd in GS.edges(data=True)])), 3),
    road_radius_m=R_DISC, road_edges_total=int(len(ze["site"])),
    pbf_ways_in_file=int(rmeta["ways_in_file"]), pbf_highway_ways=int(rmeta["highway_ways"]),
    pbf_ways_touching=int(rmeta["ways_touching_discs"]), road_edges_unique=UNIQ_EDGES,
    road_node_slots=int(len(NODE_XY)), road_classes=len(CLASSES),
    sites_with_graph=int(len(HASR)), sites_without=[int(x) for x in rmeta["sites_without_roads"]],
    sites_without_names=[str(D.name[int(x)]) for x in rmeta["sites_without_roads"]],
    road_components_mean=round(float(HASR.components.mean()), 3),
    road_components_median=float(HASR.components.median()),
    sites_fragmented=int((HASR.components > 1).sum()),
    max_components_site=int(HASR.components.max()),
    deadend_median=round(float(HASR.deadend_ratio.median()), 4),
    deadend_min=round(float(HASR.deadend_ratio.min()), 4),
    deadend_max=round(float(HASR.deadend_ratio.max()), 4),
    bridges_total=int(HASR.bridges.sum()), articulations_total=int(HASR.articulations.sum()),
    fragile_sites=int(len(fragile)),
    access_median_m=round(float(HASR.access_dist_m.median()), 1),
    access_max_m=round(float(HASR.access_dist_m.max()), 1),
    top_class=dict(collections.Counter(HASR.top_class).most_common(6)),
    rho_days21_24=round(float(rho_21), 3), p_days21_24=float(p_21),
    median_access_minus_edge_m=round(float(both.access_minus_edge_m.abs().median()), 1),
    max_access_minus_edge_m=round(float(both.access_minus_edge_m.abs().max()), 1),
    rho_edges_ways=round(float(rho_w), 3), p_edges_ways=float(p_w),
    rho_deadend_pop=round(float(rho_de), 3), p_deadend_pop=float(p_de),
    rho_components_pop=round(float(rho_cp), 3), p_components_pop=float(p_cp),
    top_betweenness=top_btw_list, top_degree=top_deg, tiny_component=tiny,
    tiny_component_sites=[str(D.name[int(j)]) for j in tiny_sites],
    site_top_degree=[{"idx": int(i), "name": str(D.name[i]), "degree": int(dd)}
                     for i, dd in site_deg])
STATS = dict(
    theme="Route network, transit network, human network or something completely different.",
    theme_name="Network",
    theme_source="30daymapchallenge.com หัวข้อ Day 24 'Network' (อ่านสด 20:18 น. วันที่ 24 พ.ย. 2569)",
    theme_fetch_time=dict(hour=20, minute=18, day=24, month=11, year=2569),
    generated_local=time.strftime("%Y-%m-%d %H:%M:%S"),
    question="รอบแหล่งดูฟ้ามืดไทย 83 แห่ง มีเครือข่ายอยู่สองชั้น — คนที่ map กับถนนที่พาไป "
             "แต่ละเครือข่ายเป็นผืนเดียวหรือกระจัดกระจาย และจุดไหนที่ต้องพึ่ง 'จุดเดียว' มากเกินไป",
    provenance=dict(
        human_network="cache/day16/way_all.json (way 5,495 ชิ้น + user + timestamp "
                      "ในรัศมี 1,000 ม. รอบ 83 จุด — เก็บไว้ตั้งแต่วันที่ 17 พ.ย. 2569 ผ่าน "
                      "Overpass API)",
        road_network="cache/day24/*.npz จาก tools/collect_day24_roads.py "
                     "(Geofabrik Thailand extract, md5 db3958a037ce717c462101b86e2f75f3)",
        registry="data/darksky_thailand.csv · กริด/พิกัดชุดเดียวกับ Day 3–23",
        attribution="© OpenStreetMap contributors (ODbL)"),
    method=dict(
        bipartite="ข้อมูลจริงคือ (จุด ↔ editor ↔ way) — กราฟอื่นทั้งหมดคือ projection",
        editor_edge="สอง editor มีเส้นเมื่อ map รอบจุดเดียวกันอย่างน้อย 1 จุด (weight = จำนวนจุดร่วม)",
        site_edge="สองจุดทะเบียนมีเส้นเมื่อมี editor ร่วม (weight = จำนวนคนร่วม)",
        temporal="ขอบของ editor network จะรอดเมื่อช่วงปี first–last ของสองคนทับกันจริง",
        road_graph=f"node = จุดตัดของ OSM, edge = ท่อนถนนที่ปลายทั้งสองอยู่ในจาน {R_DISC:,.0f} ม. "
                   "ของจุดเดียวกัน (ตัดผ่านจากนอกจานออก เพื่อไม่ให้ทางผ่านยาว ๆ ทำตัวเป็นเครือข่าย)",
        fragile="articulation point ภายใน 2 กม. จากจุดทะเบียน (ไม่นับ node ที่ใกล้ที่สุดเอง): ลบทีละจุด (สูงสุด 40 จุด) แล้วดูว่า "
                "เครือข่ายที่เข้าถึงได้เหลือกี่เปอร์เซ็นต์ของขอบ"),
    aggregates=AGG,
    road_per_site=RD.to_dict("records"),
    site_edges_sample=GS_edges[:20],
    cross_check_day16=dict(ways_this_run=WAYS_TODAY, ways_published=int(d16c.shape[0] and 5495),
                           editors=len(USERS), multi_site=MULTI,
                           multi_site_published_pct=25.4,
                           note="ตัวเลขตรงกับการนับของ Day 16 — เป็นการ reproduce ข้ามวัน"),
    cross_check_day21=dict(rho=round(float(rho_21), 3), p=float(p_21),
                           median_abs_diff_m=round(float(both.access_minus_edge_m.abs().median()), 1),
                           max_abs_diff_m=round(float(both.access_minus_edge_m.abs().max()), 1),
                           note="Day 21 วัดถึง *ขอบ* ของ way ใน 2 กม. · วันนี้วัดถึง *node* ใน 5 กม. "
                                "จึงคาดต่างกันระดับสิบเมตร ไม่ใช่ศูนย์"),
    p_value_notation=dict(rho_edges_ways=sci(p_w), rho_deadend_pop=sci(p_de),
                          rho_components_pop=sci(p_cp), days21_24=sci(p_21)),
    figures=["day24_networks.png", "day24_roads.png", "day24_fragile.png", "day24_people.png"],
    metric_definitions={
        "components": "จำนวนชิ้นของกราฟที่แยกขาดจากกัน (1 = เชื่อมถึงกันหมดในพื้นที่นั้น)",
        "deadend_ratio": "สัดส่วน node ที่มีดีกรี 1 (ปลายเส้นที่ไม่มีใครต่อถึง) — สูง = เป็นกิ่งก้าน",
        "bridges": "จำนวน *ขอบ* ที่ถ้าตัดออกกราฟจะขาดจากกัน (ถนนสายเดี่ยว)",
        "articulations": "จำนวน *จุด* ที่ถ้าปิดแล้วกราฟขาด (สี่แยกที่ขาดไม่ได้)",
        "worst_cut_keep": "กรณีแย่ที่สุดของการลบจุดตัดสำคัญ 1 จุดภายใน 2 กม.: "
                          "ขอบที่ยังเข้าถึงได้จากจุดทะเบียน ÷ ขอบของเครือข่ายเดิม",
        "access_dist_m": "ระยะจากจุดทะเบียนถึง node ถนนที่ใกล้ที่สุด (เมตร, projection เดิมซีรีส์)",
        "shared_editors": "จำนวนคนที่ map ทั้งสองจุดนั้น (น้ำหนักของเส้นใน site network)",
        "temporal_edges": "ขอบที่รอดเมื่อกรองว่าช่วงปี first–last ของสองคนทับกัน"},
    limitations=[
        "network ของคนสร้างจากรัศมี 1,000 ม. รอบจุดทะเบียน (ของเดิมจาก Day 16) ไม่ใช่ขอบเขต "
        "พื้นที่ทำงานจริงของ mapper แต่ละคน — คนที่ map ติดกันสองจุดอาจถูกนับว่าร่วมจุดเดียวกัน",
        "ขอบถนนถูกตัดที่ขอบจาน 5 กม. ดังนั้น components ที่ >1 บางส่วนเกิดจากกรอบการศึกษา "
        "ไม่ใช่ความขาดจริงของเครือข่าย (ตรวจ column `nodes`, `edges` เทียบกับ `components`)",
        "กราฟถนนเป็น 2D planar จาก OSM ไม่ได้ตัด through-traffic ของทางด่วนหรือฟากล่วงหน้า",
        "articulation ลองเฉพาะจุดแรก 40 จุดภายใน 2 กม. (MAX_CUT_TRIALS) — จุดที่อยู่ไกลกว่า "
        "ไม่ถูกนับว่าเปราะต่อจุดนั้น",
        "'ปีที่ map ทับกัน' ใช้ year range หยาบ ๆ ไม่ใช่ช่วงเวลาที่ทำงานจริง "
        "จึงเป็นค่าประมาณบน (upper bound) ของการทำงานร่วมกัน"],
    note="ภาพทุกแผ่นวาดจากข้อมูลจริง (แคช Day 16 + extract + Natural Earth) ไม่มี tile ภายนอก")
def _clean(o):
    """NaN/Inf → null (json เองไม่ยอมให้ NaN และเบราว์เซอร์อ่านไม่ออก)"""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (int, np.integer)):
        return int(o)
    return o


json.dump(_clean(STATS), open(os.path.join(OUT, "day24_stats.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, allow_nan=False,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("เขียน day24_stats.json + roads/editors/site_edges/sites csv")

# ================================================== 6. ผู้ตัดสิน
for f in ["day24_networks.png", "day24_roads.png", "day24_fragile.png", "day24_people.png",
          "day24_roads.csv", "day24_editors.csv", "day24_site_edges.csv", "day24_sites.csv",
          "day24_stats.json"]:
    assert os.path.getsize(os.path.join(OUT, f)) > 0, f"ไฟล์ว่าง: {f}"
json.load(open(os.path.join(OUT, "day24_stats.json"), encoding="utf-8"))
lint_thai(" ".join(TXT) + " " + __doc__, "day24-texts")
assert WAYS_TODAY == 5495, f"way ต้องตรงกับ Day 16 (ได้ {WAYS_TODAY})"
assert len(USERS) == 299, f"editor ต้องตรงกับ Day 16 (ได้ {len(USERS)})"
assert sum(comp_sizes_u) == GU.number_of_nodes()
assert GS.number_of_nodes() + len(iso_s) == 83, "จุดต้องมีครบ 83 (รวมที่ไม่เชื่อม)"
assert int(RD.nodes.sum()) == len(NODE_XY), "node-slot ต้องตรงกับที่สแกนไว้"
assert (RD.loc[RD.nodes > 0, "components"] >= 1).all()
assert (RD.loc[RD.nodes > 0, "largest"] <= RD.loc[RD.nodes > 0, "nodes"]).all()
assert (RD.loc[RD.nodes > 0, "worst_cut_keep"].between(0, 1)).all()
assert GU.number_of_edges() >= GT.number_of_edges(), "กรองเวลาต้องเหลือขอบน้อยลง"
assert abs(R_DISC - float(rmeta["radius_m"])) < 1e-9
assert len(CLASSES) == len(rmeta["classes"])
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
                          ["editors", "editor_edges", "editor_components", "editor_largest",
                           "site_edges", "site_largest", "isolated_sites", "road_edges_unique",
                           "sites_fragmented", "deadend_median", "bridges_total",
                           "fragile_sites", "rho_days21_24", "rho_edges_ways"]},
                         ensure_ascii=False))
print("DONE day24 ใน %.1f วิ" % (time.time() - T0))
_LOGF.close()
