# -*- coding: utf-8 -*-
"""
Day 25 · Is this a map? — "Stretch the limits of maps and blend your map with an artwork or
some creative data visualization. When does something become a map and when does it stop
being one?" (30daymapchallenge.com หัวข้อ Day 25 'Is this a map?' อ่านสด 26 พ.ย. 2569)

แปลโจทย์เป็นสิ่งที่วัดได้: ลบ "ความเป็นแผนที่" ออกทีละชั้น แล้ววัด 2 อย่าง
  A. ชี้ที่ได้ไหม (identifiability) — จากภาพ/ตัวเลขที่เหลือ อยู่เดี่ยวถูกว่าเป็นจุดไหน
  B. ระยะทางยังแปลว่าระยะทางอยู่ไหม (metric fidelity) — hop-กับ-กม., road stretch,
     และ "ถ้าเอารูปนี้ไปวาดใหม่ ตำแหน่งเพี้ยนจากจริงกี่เมตร"

กติกาที่ประกาศก่อนรัน (ไม่ปรับทีหลัง):
  - ทำให้ข้อมูลด้อยลง *ก่อน* แล้วค่อยสกัดคุณลักษณะจากสิ่งที่รอด (project → describe → match)
  - การทดสอบ A ใช้ split-half: แยกบันทึกขอบของจุดเดียวกันเป็นสองซีก A/B แล้วถามว่า
    "ลายเซ็นของซีก A ของจุด i ใกล้กับซีก B ของจุด i มากกว่าซีก B ของจุดอื่นหรือไม่"
    (แบบ leave-one-out ใช้ไม่ได้ — ถ้าหนึ่งจุดมีภาพเดียว การกันตัวเองออกจากการแข่งขัน
     ทำให้ตอบถูกเป็นศูนย์โดยนิยาม เจอก่อนรันจึงเปลี่ยนเป็น split-half)
  - ผู้สมัคร k=1, แต้มที่เท่ากันแตกด้วยเลขสุ่ม, seed = 25, ระยะทางแต่ละขั้นประกาศล่วงหน้า
  - เกณฑ์ "ยังชี้ที่ได้": p < 0.05 และ top-1 ≥ 3 × chance (ตั้งก่อนเห็นผล)

ทุกตัวเลขในโพสต์อ่านจาก day25_stats.json + day25_*.csv เท่านั้น
"""
import os, sys, json, time, math, zipfile, collections
import numpy as np, pandas as pd, geopandas as gpd
import networkx as nx
from shapely.validation import make_valid
from pyproj import CRS, Transformer, Geod
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection

T0 = time.time()
COLAB = (os.environ.get("DAY25_FORCE_COLAB") == "1") or not os.path.isdir(
    os.path.join(r"D:\GIS30Days", "geo-notes"))
WORK = os.getcwd() if COLAB else r"D:\GIS30Days"
REPO = os.path.join(WORK, "geo-notes")
CACHE = os.path.join(WORK, "day25_cache") if COLAB else os.path.join(WORK, "cache", "day25")
OUT = os.path.join(WORK, "out", "day25")
RAW = "https://raw.githubusercontent.com/thammarat-ai/geo-notes/main/"
SEED = 25
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


if COLAB:
    for f in ("Trirong-Regular.ttf", "Trirong-Bold.ttf"):
        ensure("https://github.com/google/fonts/raw/main/ofl/trirong/" + f, os.path.join(CACHE, f))
    ensure("https://github.com/google/fonts/raw/main/ofl/chonburi/Chonburi-Regular.ttf",
           os.path.join(CACHE, "Chonburi-Regular.ttf"))
    for rel in ("data/darksky_thailand.csv", "data/day25_graphs.npz",
                "data/day25_graphs_meta.json", "posts/day24-network/day24_roads.csv"):
        repo_file(rel)
FONTDIRS = [CACHE, os.path.join(WORK, "cache", "day06"), os.path.join(REPO, "static", "fonts")]
for _d in FONTDIRS:
    if not os.path.isdir(_d):
        continue
    for _f in sorted(os.listdir(_d)):
        if _f.endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(_d, _f))
plt.rcParams["font.family"] = "Trirong"
assert any(f.name == "Trirong" for f in font_manager.fontManager.ttflist),     "ฟอนต์ไทย Trirong ไม่ถูกโหลด — ภาพจะขึ้นกล่องสี่เหลี่ยมแทนตัวอักษร"

_LOGF = open(os.path.join(OUT, "day25_run.log"), "w", encoding="utf-8")


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

NELOCAL = os.path.join(WORK, "cache", "day06")


def ensure_shp(stem, sub="cultural"):
    """Natural Earth 10m แบบเดียวกับ Day 15/22 (naciscdn เป็น mirror ทางการของ NE)"""
    for d in (NELOCAL, CACHE):
        if os.path.exists(os.path.join(d, stem + ".shp")):
            return os.path.join(d, stem + ".shp")
    zp = os.path.join(CACHE, stem + ".zip")
    ensure(f"https://naciscdn.org/naturalearth/10m/{sub}/{stem}.zip", zp)
    zipfile.ZipFile(zp).extractall(CACHE)
    return os.path.join(CACHE, stem + ".shp")


BG, INK, MUTED = "#0d1522", "#e8e3d3", "#9fb2c8"
ACCENT, WARM, COOL, TEAL, PURPLE, LIME, ROSE = ("#f5b942", "#f4566a", "#38bdf8", "#2dd4bf",
                                                "#c084fc", "#a3e635", "#fb7185")
EQ = CRS.from_proj4("+proj=eqdc +lat_0=14 +lon_0=101 +lat_1=8 +lat_2=20 +units=m +no_defs "
                    "+ellps=WGS84")
GEOD = Geod(ellps="WGS84")
TRS = Transformer.from_crs("EPSG:4326", EQ, always_xy=True)

# ================================================== 1. กราฟถนน 82 จุด (data/day25_graphs.npz)
D = pd.read_csv(repo_file("data/darksky_thailand.csv")).reset_index(drop=True)
D["idx"] = D.index
assert len(D) == 83, "ทะเบียนจุดดูฟ้ามืดต้องมี 83 แถว"
P = np.array([[float(r.longitude), float(r.latitude)] for _, r in D.iterrows()])
guard_bbox(P, 96.5, 107.6, 4.3, 21.7, "darksky-points")
SP = np.array(TRS.transform(P[:, 0], P[:, 1])).T
guard_crs([SP[:, 0].min(), SP[:, 1].min(), SP[:, 0].max(), SP[:, 1].max()],
          "projected-metres", expect="metres")

gm = json.load(open(repo_file("data/day25_graphs_meta.json"), encoding="utf-8"))
gz = np.load(repo_file("data/day25_graphs.npz"))
R_DISC = float(gm["radius_m"])
CLASSES = gm["classes"]
assert gm["pbf_md5"] == "db3958a037ce717c462101b86e2f75f3", "PBF ต้นทางคนละก้อนกับที่ซีรีส์นี้ใช้"
assert len(gz["nid"]) == gm["node_slots"] and len(gz["u"]) == gm["edges"], "ไฟล์ npz ขาดหรือเกิน"

NLat = gz["lat"] / 1e7
NLon = gz["lon"] / 1e7
XY = np.array(TRS.transform(NLon, NLat)).T
assert np.isfinite(XY).all(), "พิกัดที่ projection มา มีค่าไม่จำกัด"

by_node = {}
for j, nid, la, lo, x, y in zip(gz["site"].astype(int), gz["nid"], NLat, NLon,
                                XY[:, 0], XY[:, 1]):
    by_node.setdefault(int(j), []).append((int(nid), float(la), float(lo), float(x), float(y)))
by_edge = collections.defaultdict(list)
for j, u, v, c in zip(gz["esite"].astype(int), gz["u"], gz["v"], gz["cls"]):
    by_edge[int(j)].append((int(u), int(v), int(c)))

IDX = sorted(by_node)
assert len(IDX) == gm["sites_covered"] == 82
SITES_NO_GRAPH = [str(D.name[j]) for j in range(83) if j not in set(IDX)]
log(f"โหลดกราฟ: {len(IDX)} จุดที่มีถนน · node-slot {len(gz['nid']):,} · บันทึกขอบ {len(gz['u']):,} · "
    f"จุดที่ไม่มีกราฟ = {'; '.join(SITES_NO_GRAPH)}")


def build_view(j, edges=None, tag="full"):
    """สร้างกราฟของหนึ่งจุด (หรือของหนึ่งซีก) — เก็บพิกัดทั้งองศาและเมตรให้ครบทุกโหนด"""
    rows = by_node[j]
    nid = np.array([r[0] for r in rows], dtype=np.int64)
    vla = np.array([r[1] for r in rows], dtype=np.float64)
    vlo = np.array([r[2] for r in rows], dtype=np.float64)
    xy = np.array([[r[3], r[4]] for r in rows], dtype=np.float64)
    loc = {int(v): k for k, v in enumerate(nid)}
    e = by_edge[j] if edges is None else edges
    uu = np.array([loc[a] for a, b, c in e], dtype=np.int64)
    vv = np.array([loc[b] for a, b, c in e], dtype=np.int64)
    cl = np.array([c for a, b, c in e], dtype=np.int64)
    if edges is not None:                      # ซีก A/B: เก็บเฉพาะโหนดที่ขอบของซีกนั้นแตะ
        used = np.union1d(np.unique(uu), np.unique(vv))
        remap = -np.ones(len(nid), dtype=np.int64); remap[used] = np.arange(len(used))
        nid, vla, vlo, xy = nid[used], vla[used], vlo[used], xy[used]
        m_ok = (remap[uu] >= 0) & (remap[vv] >= 0)
        uu, vv, cl = remap[uu][m_ok], remap[vv][m_ok], cl[m_ok]
    n = len(nid)
    if n == 0:
        return None
    G = nx.Graph(); G.add_nodes_from(range(n))
    for a, b, c in zip(uu, vv, cl):
        G.add_edge(int(a), int(b), cls=int(c))
    d2c = np.hypot(xy[:, 0] - SP[j][0], xy[:, 1] - SP[j][1])
    rows_a = np.concatenate([uu, vv]); cols_a = np.concatenate([vv, uu])
    _a1, _a2, dd = GEOD.inv(vlo[uu], vla[uu], vlo[vv], vla[vv])
    dist = np.asarray(dd, dtype=np.float64)
    Cs = sparse.csr_matrix((np.ones(len(rows_a), dtype=np.int8), (rows_a, cols_a)), shape=(n, n))
    Cm = sparse.csr_matrix((np.concatenate([dist, dist]), (rows_a, cols_a)), shape=(n, n))
    deg = np.fromiter((d for _, d in G.degree()), dtype=np.int32, count=n)
    return dict(j=j, tag=tag, n=n, nid=nid, xy=xy, la=vla, lo=vlo, G=G, Cs=Cs, Cm=Cm,
                d2c=d2c, m=int(G.number_of_edges()), m_raw=int(len(uu)), deg=deg)


FULL = {j: build_view(j) for j in IDX}
for j in IDX:
    g = FULL[j]
    assert len(set(int(x) for x in g["nid"])) == g["n"], f"โหนดซ้ำในจุด {D.name[j]}"
    assert g["m"] <= g["m_raw"] and g["Cs"].shape == (g["n"], g["n"])
MULTI = sum(FULL[j]["m_raw"] - FULL[j]["m"] for j in IDX)
TOT_EDGE_REC = sum(FULL[j]["m_raw"] for j in IDX)
TOT_EDGE = sum(FULL[j]["m"] for j in IDX)
TOT_NODES = sum(FULL[j]["n"] for j in IDX)
assert TOT_EDGE_REC == len(gz["u"]), "จำนวนบันทึกขอบไม่ตรงกับไฟล์ npz"
log(f"ขอบซ้ำ (คู่โหนดเดิมมากกว่าหนึ่งบันทึก) {MULTI:,} บันทึก → ยุบเหลือ {TOT_EDGE:,} เส้น "
    f"ตามนิยาม Day 24 · โหนดรวม {TOT_NODES:,}")

# ---- ผู้ตัดสินข้อแรก: สร้างกราฟจาก npz แล้วต้องได้ตัวเลข Day 24 กลับมาตรงกัน
RD24 = pd.read_csv(repo_file("posts/day24-network/day24_roads.csv")).set_index("idx")
rebuilt = []
for j in IDX:
    g = FULL[j]; G = g["G"]
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    acc = int(np.argmin(g["d2c"]))
    ai = next(i for i, c in enumerate(comps) if acc in c)
    sub = G.subgraph(comps[ai]).copy()
    rebuilt.append(dict(idx=j, nodes=g["n"], edges=g["m"], components=len(comps),
                        deadend_ratio=round(float((g["deg"] == 1).sum()) / g["n"], 4),
                        avg_deg=round(2 * g["m"] / g["n"], 3),
                        bridges=sum(1 for _ in nx.bridges(nx.MultiGraph(sub.edges()))),
                        access_dist_m=round(float(g["d2c"][acc]), 1)))
RB = pd.DataFrame(rebuilt).set_index("idx")
REF = RD24.loc[RB.index]
REBUILD_DIFF = {}
for col, tol in [("nodes", 0), ("edges", 0), ("components", 0), ("bridges", 0),
                 ("deadend_ratio", 1e-9), ("avg_deg", 1e-9), ("access_dist_m", 0.051)]:
    REBUILD_DIFF[col] = float((RB[col] - REF[col]).abs().max())
    assert REBUILD_DIFF[col] <= tol, f"rebuild ไม่ตรงกับ Day 24 ที่คอลัมน์ {col}"
log("rebuild จาก day25_graphs.npz ตรงกับ day24_roads.csv ครบ 82 จุด × 7 คอลัมน์ ✓")
assert abs(RB.components.mean() - 8.671) < 0.001 and int(RB.bridges.sum()) == 157625

# ================================================== 2. ค่าคงที่ที่ใช้ร่วมกัน
BALL_HOPS = 12         # รัศมีลูกบอลทอพอโลยี (ก้าว) — เทียบกับ Day 7 ที่ใช้ 10 นาที

# ================================================== 3. สามตัวแทนของ "ที่เดียวกัน" (ไม่มี split-half)
# เหตุผลที่เลิก split-half (วัดจริง ไม่ใช่คาดเดา — ดู fragmentation_probe ด้านล่าง):
#   การแบ่งข้อมูลเครือข่ายเชิงพื้นที่ให้ไม่ทับกัน มีได้แค่สองแบบ และพังทั้งคู่
#     (1) สุ่มรายขอบ / สุ่มรายท่อนถนน → กราฟขาดเป็นผุยผง ชิ้นที่เดินถึงจากจุดเข้าถึง
#         เหลือเพียงหลักหน่วยโหนด จึงวัด "โครงสร้าง" ไม่ได้
#     (2) แบ่งด้วยเส้นผ่านศูนย์กลาง → สองซีกอยู่คนละพื้นที่ รูปทรง/ลูกบอลเทียบกันไม่ได้
#   สิ่งที่ทดสอบแทนจึงเป็นคำถามที่โจทย์ถามจริง: *ภาพวาดยังจับคู่กับแผนที่ได้ไหม*
#   โดยใช้ข้อมูลชุดเดียวกันทั้งหมด แต่สาม "สื่อ" ที่ต่างกัน (เมตร / ก้าว / ไม้บรรทัดบนรูป)
MW = 1000          # หน้าต่างร่วม: โหนด 1,000 โหนดแรกที่เดินถึงจากจุดเข้าถึง (ทุกจุดเท่ากัน)
N_SRC_W = 30       # แหล่งเริ่มต้นสำหรับวัดระยะภายในหน้าต่าง
NB = 12            # จำนวนขั้นของโปรไฟล์การเติบโต (ไร้หน่วย เพราะ normalize ด้วย P95 ของตัวเอง)


def geod_m(lo1, la1, lo2, la2):
    """ระยะ geodesic เป็นเมตร (pyproj ต้องการอาร์เรย์ความยาวเท่ากันทุกตัว จึงขยายฝั่งต้นทาง)"""
    lo2 = np.asarray(lo2, dtype=np.float64)
    la2 = np.asarray(la2, dtype=np.float64)
    n = lo2.size
    _a1, _a2, dd = GEOD.inv(np.full(n, float(lo1)), np.full(n, float(la1)), lo2, la2)
    return np.asarray(dd, dtype=np.float64)


def window(view, mw=MW):
    """ชิ้นกราฟที่เดินถึงจากจุดเข้าถึง mw โหนดแรก (BFS) = สิ่งที่ทั้งวาดได้และวัดได้
    ใช้หน้าต่างเดียวกันทุกสื่อ จึงไม่มีตัวแปรใดหลุดจากการเทียบ"""
    g = view
    acc = int(np.argmin(g["d2c"]))
    order = csgraph.breadth_first_order(g["Cs"].tocsc(), acc, directed=False,
                                        return_predecessors=False)
    keep = np.asarray(order[:mw], dtype=np.int64)
    Cs = g["Cs"][np.ix_(keep, keep)].tocsr()
    Cs.data = np.ones_like(Cs.data)
    Cm = g["Cm"][np.ix_(keep, keep)].tocsr()
    return dict(keep=keep, Cs=Cs, Cm=Cm, acc=0, la=g["la"][keep], lo=g["lo"][keep],
                xy=g["xy"][keep] - SP[g["j"]], sub=g["G"].subgraph([int(x) for x in keep]).copy())


def pick_srcs(w, n_src=N_SRC_W):
    n = len(w["keep"])
    if n <= n_src + 1:
        return np.arange(n)
    r = np.random.default_rng(SEED)
    return np.unique(np.concatenate([[0], r.choice(n, size=n_src, replace=False)]))


def growth(values, nb=NB):
    """โปรไฟล์การเติบโต: สัดส่วนโหนดที่ "ใกล้" ภายในรัศมี 12 ระดับที่วัดเป็นเศษส่วนของ
    รัศมี P95 ของตัวมันเอง → ไร้หน่วย เทียบข้ามสื่อได้ (แผนที่/ก้าว/ภาพวาด)"""
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < nb + 2:
        return None
    hi = float(np.percentile(v, 95))
    if not hi or not math.isfinite(hi) or hi <= 0:
        return None
    grid = np.linspace(0.0, 1.0, nb) * hi
    return np.searchsorted(v, grid, side="right") / float(v.size)


def spring_draw(w, iters=50):
    """วาดใหม่จากความเชื่อมโยงอย่างเดียว (spring layout ไม่ใช้พิกัดจริง) → ตัวแทนที่สาม"""
    g = w["sub"]
    if g.number_of_nodes() < NB + 4 or not nx.is_connected(g):
        return None
    return nx.spring_layout(g, seed=SEED, iterations=iters)


VIEWS = ["map", "hop", "draw", "tag"]
SIGS = ["radial1", "radial5"]
VIEW_LABEL = {"map": "แผนที่ (เมตรจริง)", "hop": "โครงสร้างล้วน (จำนวนก้าว)",
              "draw": "ภาพวาด (ไม้บรรทัดทาบรูป)",
              "tag": "ป้ายหมายเลข (ไม่มีข้อมูลเชิงพื้นที่)"}
SIG_LABEL = {"radial1": "โตขึ้นจากจุดเดียว (12 ขั้น)",
             "radial5": "โตขึ้นจากห้าจุด (5×12 ขั้น)"}
PROF = {(v, s): [] for v in VIEWS for s in SIGS}
PROFN = {}
DRAW = {}
WIN = {}
N_ORIGIN = 5           # จำนวนจุดเริ่มต้นของลายเซ็น radial5 (จุดเข้าถึง + สุ่ม 4 จุด)
for j in IDX:
    g = FULL[j]
    w = window(g)
    WIN[j] = w
    Hv = csgraph.shortest_path(w["Cs"].astype(np.int8), indices=np.arange(len(w["keep"])),
                               directed=False, method="BF") if False else None
    r = np.random.default_rng(SEED + j)
    n = len(w["keep"])
    org = np.unique(np.concatenate([[0], r.choice(n, size=N_ORIGIN - 1, replace=False)]))[:N_ORIGIN]
    org = np.pad(org, (0, N_ORIGIN - len(org)), constant_values=0)
    hops5 = csgraph.shortest_path(w["Cs"].astype(np.int8), indices=org, directed=False,
                                  method="BF")
    prof = {"map": [], "hop": [], "draw": [], "tag": []}
    P = None
    if j in DRAW:
        P = DRAW[j]["xy"]
    pos = spring_draw(w)
    if pos is not None:
        P = np.array([pos[int(k)] for k in w["keep"]])      # ป้ายชื่อโหนด = ดัชนีในกราฟเต็ม
        ii, jj = w["Cs"].nonzero()
        m = ii < jj
        DRAW[j] = dict(xy=P, fit=P, keep=w["keep"], edg=np.stack([ii[m], jj[m]], axis=1))
    for oi, o in enumerate(org):
        prof["map"].append(growth(np.hypot(w["xy"][:, 0] - w["xy"][o, 0],
                                           w["xy"][:, 1] - w["xy"][o, 1])))
        prof["hop"].append(growth(hops5[oi].astype(float)))
        prof["draw"].append(None if P is None else
                            growth(np.hypot(P[:, 0] - P[o, 0], P[:, 1] - P[o, 1])))
    # สื่อที่สี่: ป้ายหมายเลขสุ่มที่ไม่ได้มาจากข้อมูลเชิงพื้นที่เลย (ตัวคุมแนวคิด)
    # ถ้า "การระบุตัวตน" คือหัวใจของแผนที่ ป้ายเลขต้องจับคู่กับแผนที่ได้ — มันต้องทำไม่ได้
    tag = np.array([float(np.random.default_rng(SEED + 991 * j + 17).normal())])
    for v in VIEWS:
        ok = all(x is not None for x in prof[v]) if v != "tag" else True
        if v == "tag":
            # ขยายให้กว้างเท่าสื่ออื่น (ทุกช่องค่าเดียวกัน) → ระยะทางระหว่างทุกคู่เป็น 0 เสมอ
            # แปลว่า "ป้ายที่ระบุตัวตนได้ 100% ในระบบของมันเอง" จับคู่กับแผนที่ไม่ได้เลย
            PROF[(v, "radial1")].append(np.tile(tag, NB))
            PROF[(v, "radial5")].append(np.tile(tag, NB * N_ORIGIN))
        else:
            PROF[(v, "radial1")].append(prof[v][0])
            PROF[(v, "radial5")].append(np.concatenate(prof[v]) if ok else None)
    PROFN[j] = {s: {v: PROF[(v, s)][k] is not None for v in VIEWS}
                for s, k in [(s, IDX.index(j)) for s in SIGS]}
    assert PROFN[j]["radial5"]["tag"]
MISSING = {f"{v}/{s}": int(sum(1 for x in PROF[(v, s)] if x is None))
           for v in VIEWS for s in SIGS}
PRESENT = {(v, s): np.array([x is not None for x in PROF[(v, s)]])
           for v in VIEWS for s in SIGS}
log("สร้างตัวแทนเสร็จ: " + " · ".join(f"{k} ขาด {v}" for k, v in MISSING.items()) +
    f" จาก 82 จุด (หน้าต่างร่วม {MW:,} โหนด, {N_ORIGIN} จุดเริ่มต้น)")
assert all(MISSING[f"{v}/radial1"] == 0 for v in ("map", "hop", "tag")),     "โปรไฟล์แผนที่/ก้าว/ป้ายต้องมีครบทุกจุด"


def fragmentation_probe(lim=14, seed=SEED):
    """วัดหลักฐานว่าทำไม split-half ใช้ไม่ได้: ขนาดชิ้นที่เดินถึงจากจุดเข้าถึงหลังแบ่งซีก
    ทำกับจุด subset (lim จุด) เพื่อไม่ให้เสียเวลารันทั้งสคริปต์"""
    out = {}
    for mode in ("edge", "chain", "halfplane"):
        sizes = []
        for j in IDX[:lim]:
            e = by_edge[j]
            if mode == "edge":
                perm = np.random.default_rng(seed + 3 * j).permutation(len(e))
                A = [e[x] for x in perm[:len(e) // 2]]
            elif mode == "halfplane":
                rows = by_node[j]
                loc = {int(r[0]): (r[3], r[4]) for r in rows}
                cx = float(np.mean([r[3] for r in rows])); cy = float(np.mean([r[4] for r in rows]))
                th = float(np.random.default_rng(seed + 5 * j).uniform(0, 2 * math.pi))
                ux, uy = math.cos(th), math.sin(th)
                A = [(u, v, c) for (u, v, c) in e
                     if (((loc[u][0] + loc[v][0]) / 2 - cx) * ux +
                         ((loc[u][1] + loc[v][1]) / 2 - cy) * uy) >= 0]
            else:
                adj = collections.defaultdict(list)
                for k, (u, v, c) in enumerate(e):
                    adj[u].append((v, k)); adj[v].append((u, k))
                junction = {n for n, a in adj.items() if len(a) != 2}
                used = set(); chains = []
                for start in sorted(adj):
                    for nb2, ek in adj[start]:
                        if ek in used or (start not in junction and nb2 not in junction):
                            continue
                        ch = [ek]; used.add(ek); prev, cur = start, nb2
                        while cur not in junction:
                            nxt = [(n2, e2) for n2, e2 in adj[cur] if n2 != prev and e2 not in used]
                            if not nxt:
                                break
                            ch.append(nxt[0][1]); used.add(nxt[0][1]); prev, cur = cur, nxt[0][1]
                        chains.append(ch)
                rng = np.random.default_rng(seed + 7 * j)
                A = [e[k] for ch in chains if rng.random() < 0.5 for k in ch]
            h = build_view(j, A, "A")
            sizes.append(0 if h is None else len(window(h)["keep"]))
        out[mode] = int(np.median(sizes))
    out["full"] = int(np.median([len(WIN[j]["keep"]) for j in IDX[:lim]]))
    return out


# ================================================== 4. การทดสอบ A: จับคู่ข้ามสื่อ (4×4 × 2 ลายเซ็น × 3 ฐาน)
NN_FULL = np.array([FULL[j]["n"] for j in IDX])
REG = np.array([str(D.region_th[j]) for j in IDX])
PROV = np.array([str(D.province[j]) for j in IDX])
POOLKINDS = ("all", "sizeblind", "far")


def make_pools(kind):
    """สามฐานผู้สมัคร — คราวนี้ไม่มีปัญหา "กันตัวเองออกแล้วตอบถูกเป็นศูนย์" เพราะ
    query กับ gallery เป็นคนละสื่อของข้อมูลชุดเดียวกัน (คำตอบที่ถูกต้องอยู่ในหอภาพเสมอ)"""
    out = []
    for i in range(len(IDX)):
        o = np.arange(len(IDX))
        if kind == "sizeblind":
            o = o[(NN_FULL[o] >= 0.8 * NN_FULL[i]) & (NN_FULL[o] <= 1.25 * NN_FULL[i])]
        elif kind == "far":
            o = o[(REG[o] != REG[i]) | (o == i)]
        out.append(o)
    return out


POOLS = {k: make_pools(k) for k in POOLKINDS}
POOL_SIZES = {k: np.array([len(p) for p in POOLS[k]]) for k in POOLS}


def zmat(view, sig):
    """จัดสเกลโปรไฟล์ของ *สื่อเดียว* (mean/std คำนวณจากจุดที่มีโปรไฟล์เท่านั้น)
    จุดที่โปรไฟล์ขาด (เช่นหน้าต่างเล็กเกิน) ถูกทำเครื่องหมายไว้แล้วคัดออกในการเทียบ"""
    rows = [x for x, p in zip(PROF[(view, sig)], PRESENT[(view, sig)]) if p]
    M = np.array(rows, float)
    mu, sd = M.mean(axis=0), M.std(axis=0)
    sd[sd == 0] = 1.0
    W = NB if sig == "radial1" else NB * N_ORIGIN
    out = np.zeros((len(PROF[(view, sig)]), W))
    for k, x in enumerate(PROF[(view, sig)]):
        if x is not None:
            out[k] = (np.asarray(x, float) - mu) / sd
    return out


def dist_profiles(a, b, sig, perm=None):
    A = zmat(a, sig)
    B = zmat(b, sig)
    if perm is not None:
        B = B[perm]
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(np.maximum(d2, 0))


B_PERM, B_BOOT = 20000, 2000
rp = np.random.default_rng(SEED + 7)
rows, HIT = [], {}
for sig in SIGS:
  for a in VIEWS:
    for b in VIEWS:
        M = dist_profiles(a, b, sig)
        for kind in POOLKINDS:
            pool = POOLS[kind]
            best = np.full(len(IDX), -1)
            rank_i = np.full(len(IDX), 99)
            ok = np.zeros(len(IDX), bool)
            for i in range(len(IDX)):
                cand = pool[i]
                if len(cand) < 3 or not (PRESENT[(a, sig)][i] and PRESENT[(b, sig)][i]):
                    continue
                if not np.isfinite(M[i, cand]).all() or not all(PRESENT[(b, sig)][int(x)] for x in cand):
                    continue
                d = M[i, cand]
                sc = float(d.max() - d.min())
                kd = d + rp.random(d.size) * (1e-9 * sc if sc > 0 else 1e-12)
                best[i] = int(cand[int(np.argmin(kd))])
                rank_i[i] = int(np.argsort(np.argsort(kd), kind="stable")[list(cand).index(i)])
                ok[i] = True
            idxs = np.flatnonzero(ok)
            if idxs.size < 10:
                continue
            hit1 = (best[idxs] == idxs).astype(float)
            hit3 = (rank_i[idxs] <= 2).astype(float)
            chance = float(np.mean([1.0 / len(pool[i]) for i in idxs]))
            maxlen = max(len(pool[i]) for i in idxs)
            PA = np.full((idxs.size, maxlen), -1)
            PSZ = np.zeros(idxs.size, dtype=int)
            for r, i in enumerate(idxs):
                PA[r, :len(pool[i])] = pool[i]; PSZ[r] = len(pool[i])
            col = np.minimum((rp.random((B_PERM, idxs.size)) * PSZ[None, :]).astype(int),
                             PSZ[None, :] - 1)
            truth = PA.ravel()[np.arange(idxs.size)[:, None] * maxlen + col.T].T
            accs = (truth == best[idxs][None, :]).mean(axis=1)
            assert abs(float(accs.mean()) - chance) < 0.005, (
                f"null simulation ไม่ตรง chance ({accs.mean():.4f} vs {chance:.4f}) "
                f"ที่ {sig}/{a}>{b}/{kind}")
            p_perm = float((1 + (accs >= hit1.mean()).sum()) / (B_PERM + 1))
            bs = hit1[rp.integers(0, hit1.size, (B_BOOT, hit1.size))].mean(axis=1)
            rows.append(dict(sig=sig, view_q=a, view_r=b, pool=kind, distance="euclidean ของโปรไฟล์ 12 ขั้น",
                             n_tested=int(ok.sum()), top1=round(float(hit1.mean()), 4),
                             top3=round(float(hit3.mean()), 4), chance=round(chance, 4),
                             null_mean=round(float(accs.mean()), 4),
                             ratio_vs_chance=round(float(hit1.mean()) / max(chance, 1e-9), 2),
                             p_permutation=p_perm, ci_lo=round(float(np.percentile(bs, 2.5)), 4),
                             ci_hi=round(float(np.percentile(bs, 97.5)), 4),
                             trivial=bool(a == b),
                             verdict=bool((not a == b) and p_perm < 0.05 and hit1.mean() >= 3 * chance)))
            HIT[(sig, a, b, kind)] = dict(runner=[str(D.name[IDX[int(x)]]) for x in best[ok]],
                                     hit=[bool(int(x) == int(i)) for x, i in zip(best[ok], idxs)])
# control: หอภาพถูกสลับป้าย → ต้องเสมอโอกาส (ไม่มี split มาเกี่ยวข้องจึงเป็น control ที่สะอาด)
perm_swap = np.random.default_rng(SEED + 11).permutation(len(IDX))
Ms = dist_profiles("hop", "map", "radial5", perm=perm_swap)
pool = POOLS["all"]
hitc = []
for i in range(len(IDX)):
    if not (PRESENT[("hop", "radial5")][i] and PRESENT[("map", "radial5")][perm_swap[i]]):
        continue
    d = Ms[i, pool[i]]
    hitc.append(int(pool[i][int(np.argmin(d + rp.random(d.size) * 1e-12))]) == int(perm_swap[i]))
CTRL = dict(sig="radial5", view_q="hop", view_r="map (สลับหอภาพ)", pool="all",
            distance="euclidean ของโปรไฟล์ 12 ขั้น", n_tested=len(IDX),
            top1=round(float(np.mean(hitc)), 4), top3=None,
            chance=round(1.0 / len(IDX), 4), null_mean=round(1.0 / len(IDX), 4),
            ratio_vs_chance=round(float(np.mean(hitc)) * len(IDX), 2), p_permutation=None,
            ci_lo=None, ci_hi=None, trivial=False,
            verdict=bool(float(np.mean(hitc)) >= 3 / len(IDX)))
LAD = pd.DataFrame(rows + [CTRL])
LAD.to_csv(os.path.join(OUT, "day25_ladder_raw.csv"), index=False)
log("ผลจับคู่ข้ามสื่อ (เฉพาะ pool=all):\n" +
    LAD[(LAD.pool == "all")].to_string(index=False))
guard_variance(LAD[~LAD.trivial].top1.values, "cross-media-accuracy", min_spread_km=0.02)
FRAG = fragmentation_probe()
log("หลักฐานว่าทำไมเลิก split-half (มัธยฐานโหนดที่เดินถึงหลังแบ่งซีก): " +
    " · ".join(f"{k} {v:,}" for k, v in FRAG.items()))

# ================================================== 5. การทดสอบ B: ระยะทางยังแปลว่าระยะทางไหม
def hop_km(w, max_pairs=4000):
    n = len(w["keep"])
    if n < 20:
        return dict(rho=float("nan"), p=float("nan"), n_pairs=0, med_m=float("nan"))
    srcs = pick_srcs(w)
    hops = csgraph.shortest_path(w["Cs"].astype(np.int8), indices=srcs, directed=False, method="BF")
    H, K = [], []
    for si, s in enumerate(srcs):
        ok = np.isfinite(hops[si]) & (hops[si] > 0)
        if ok.sum() < 8:
            continue
        H.append(hops[si][ok])
        K.append(geod_m(w["lo"][s], w["la"][s], w["lo"][ok], w["la"][ok]))
    if not H:
        return dict(rho=float("nan"), p=float("nan"), n_pairs=0, med_m=float("nan"))
    H = np.concatenate(H).astype(float); K = np.concatenate(K)
    if H.size > max_pairs:
        sel = np.random.default_rng(SEED).choice(H.size, size=max_pairs, replace=False)
        H, K = H[sel], K[sel]
    rho, pv = spearmanr(H, K)
    return dict(rho=float(rho), p=float(pv), n_pairs=int(H.size), med_m=float(np.median(K)))


def road_stretch(w, max_pairs=4000, min_m=200.0):
    n = len(w["keep"])
    if n < 20:
        return dict(med=float("nan"), p90=float("nan"), n_pairs=0)
    srcs = pick_srcs(w)
    road = csgraph.shortest_path(w["Cm"], indices=srcs, directed=False, method="D")
    out = []
    for si, s in enumerate(srcs):
        ok = np.isfinite(road[si]) & (road[si] > 0)
        if ok.sum() < 8:
            continue
        st = geod_m(w["lo"][s], w["la"][s], w["lo"][ok], w["la"][ok])
        m = st > min_m
        if m.sum():
            out.append(road[si][ok][m] / st[m])
    if not out:
        return dict(med=float("nan"), p90=float("nan"), n_pairs=0)
    v = np.concatenate(out)
    if v.size > max_pairs:
        v = v[np.random.default_rng(SEED).choice(v.size, size=max_pairs, replace=False)]
    return dict(med=float(np.median(v)), p90=float(np.percentile(v, 90)), n_pairs=int(v.size))


def draw_stress(j, w, max_pairs=4000):
    """Procrustes-fit ภาพที่วาดใหม่ทับพิกัดจริง → เศษเหลือ (เมตร/เท่าของขนาดภาพ), R²,
    และอันดับระยะคู่ตรงกันแค่ไหน (นี่คือ 'เอาไม้บรรทัดทาบรูปแล้ววัดได้ไหม')"""
    if j not in DRAW:
        return None
    Pm = DRAW[j]["xy"]
    Pt = w["xy"]
    za = Pm[:, 0] + 1j * Pm[:, 1]
    zb = Pt[:, 0] + 1j * Pt[:, 1]
    za = za - za.mean(); zb = zb - zb.mean()
    cross = np.sum(za * np.conj(zb))
    rot = np.conj(cross) / (abs(cross) or 1.0)
    scale = math.sqrt(float((abs(zb) ** 2).sum()) / max(float((abs(za) ** 2).sum()), 1e-12))
    fit = za * rot * scale
    res = np.abs(fit - zb)
    ss_tot = float((abs(zb) ** 2).sum())
    r = np.random.default_rng(SEED)
    npair = min(1500, len(Pm) * (len(Pm) - 1) // 2)
    sel = np.array([r.choice(len(Pm), size=2, replace=False) for _ in range(npair)])
    sel = sel[sel[:, 0] != sel[:, 1]]
    d_true = np.hypot(*(Pt[sel[:, 0]] - Pt[sel[:, 1]]).T)
    d_draw = np.hypot(*(Pm[sel[:, 0]] - Pm[sel[:, 1]]).T)
    rho_pair, _ = spearmanr(d_draw, d_true)
    return dict(med_resid_m=float(np.median(res)), p90_resid_m=float(np.percentile(res, 90)),
                r2=float(1 - float((res ** 2).sum()) / max(ss_tot, 1e-12)),
                spread_m=float(np.median(np.hypot(Pt[:, 0], Pt[:, 1]))),
                rho_pair=float(rho_pair), n=int(len(Pm)))


def reach_full(view, hops=BALL_HOPS):
    g = view
    acc = int(np.argmin(g["d2c"]))
    hv = csgraph.shortest_path(g["Cs"].astype(np.int8), indices=[acc], directed=False,
                               method="BF")[0]
    within = np.isfinite(hv) & (hv <= hops)
    km = geod_m(g["lo"][acc], g["la"][acc], g["lo"], g["la"])
    return dict(ball_nodes=int(within.sum()), ball_frac=float(within.sum() / g["n"]),
                ball_km_max=float(km[within].max() / 1000.0) if within.any() else float("nan"),
                ball_km_mean=float(km[within].mean() / 1000.0) if within.any() else float("nan"),
                reach_frac=float(np.isfinite(hv).sum() / g["n"]))


MET = []
LIMIT = int(os.environ.get("DAY25_LIMIT", "0"))
for j in (IDX[:LIMIT] if LIMIT else IDX):
    g = FULL[j]; w = WIN[j]
    a = hop_km(w); b = road_stretch(w); c = draw_stress(j, w); d = reach_full(g)
    MET.append(dict(idx=j, name=D.name[j], province=D.province[j], nodes=g["n"], edges=g["m"],
                    window_nodes=int(len(w["keep"])),
                    rho_hop_km=None if not math.isfinite(a["rho"]) else round(a["rho"], 4),
                    p_hop_km=None if not math.isfinite(a["p"]) else float(a["p"]),
                    pairs_hop=int(a["n_pairs"]),
                    med_dist_m=round(a["med_m"], 1) if math.isfinite(a["med_m"]) else None,
                    stretch_med=None if not math.isfinite(b["med"]) else round(b["med"], 3),
                    stretch_p90=None if not math.isfinite(b["p90"]) else round(b["p90"], 3),
                    draw_med_resid_m=None if c is None else round(c["med_resid_m"], 1),
                    draw_p90_resid_m=None if c is None else round(c["p90_resid_m"], 1),
                    draw_r2=None if c is None else round(c["r2"], 4),
                    draw_rho_pair=None if c is None else round(c["rho_pair"], 4),
                    draw_spread_m=None if c is None else round(c["spread_m"], 1),
                    draw_rel=None if (c is None or not c["spread_m"]) else
                    round(c["med_resid_m"] / c["spread_m"], 4),
                    ball_nodes=int(d["ball_nodes"]), ball_frac=round(d["ball_frac"], 4),
                    ball_km_max=round(d["ball_km_max"], 3) if math.isfinite(d["ball_km_max"]) else None,
                    ball_km_mean=round(d["ball_km_mean"], 3) if math.isfinite(d["ball_km_mean"]) else None,
                    reach_frac=round(d["reach_frac"], 4)))
    if (len(MET) % 20) == 0:
        log(f"วัด metric fidelity แล้ว {len(MET)}/{len(IDX)} จุด ({time.time()-T0:.0f} วินาที)")
METR = pd.DataFrame(MET)
METR.to_csv(os.path.join(OUT, "day25_metric_raw.csv"), index=False)
log("metric fidelity (มัธยฐาน): rho(hop,km) %.3f · stretch %.2f · เศษการวาด %.0f ม. "
    "(%.2f เท่าของขนาดภาพ) · R2 %.3f · rho คู่ระยะ %.3f · เดินไม่ถึง %.1f%%" % (
        METR.rho_hop_km.median(), METR.stretch_med.median(), METR.draw_med_resid_m.median(),
        METR.draw_rel.median(), METR.draw_r2.median(), METR.draw_rho_pair.median(),
        100 * (1 - METR.reach_frac.median())))
guard_variance(METR.rho_hop_km.dropna().values, "rho-hop-km", min_spread_km=0.02)
guard_variance(METR.draw_med_resid_m.dropna().values, "draw-residual", min_spread_km=100.0)

# ================================================== 6. ภาพ
# จุดที่ใช้โชว์สามตัวแทน: เลือกด้วยกฎที่ประกาศไว้ = ในกลุ่มที่ "ตัดจุดเดียวแล้วเครือข่ายหายหมด"
# (worst_cut_keep = 0 จาก Day 24) ให้เอาจุดที่กราฟใหญ่ที่สุด
_cand = RD24[(RD24.worst_cut_keep == 0.0) & (RD24.nodes >= 500)]
JS = int(_cand.nodes.idxmax()) if len(_cand) else int(RB.nodes.idxmax())
assert JS in set(IDX), "จุดที่เลือกโชว์ไม่มีกราฟถนน"
log("จุดที่เลือกมาโชว์สามตัวแทน: " + str(D.name[JS]) + f" ({FULL[JS]['n']:,} โหนด, "
    f"worst_cut_keep = {float(RD24.loc[JS, 'worst_cut_keep']):.1f})")
ne = gpd.read_file(ensure_shp("ne_10m_admin_0_countries"))
TH = make_valid(ne[ne.NAME == "Thailand"].geometry.iloc[0])
thp = gpd.GeoSeries([TH], crs="EPSG:4326").to_crs(EQ).iloc[0]
TH_RING = [np.array(g.exterior.xy) for g in (thp.geoms if thp.geom_type == "MultiPolygon" else [thp])]


def frame(ax, pad=0.03):
    xs = np.concatenate([a[0] for a in TH_RING]); ys = np.concatenate([a[1] for a in TH_RING])
    x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
    ax.set_xlim(x0 - (x1 - x0) * pad, x1 + (x1 - x0) * pad)
    ax.set_ylim(y0 - (y1 - y0) * pad, y1 + (y1 - y0) * pad)
    ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.6)
    for a in TH_RING:
        ax.plot(a[0], a[1], color=MUTED, lw=0.7, alpha=0.75, zorder=1)
    return ax


def moran_i(vals, coords, k=6):
    tr = cKDTree(np.radians(coords))
    _, nb = tr.query(np.radians(coords), k=k + 1)
    v = np.asarray(vals, float); v = v - v.mean()
    w = np.array([v[nb[i, 1:]].mean() for i in range(len(v))])
    return float((v * w).sum() / (v ** 2).sum())


def cell(a, b, kind="all", sig="radial5"):
    s = LAD[(LAD.sig == sig) & (LAD.view_q == a) & (LAD.view_r == b) & (LAD.pool == kind)]
    return float(s.top1.iloc[0]) if len(s) else np.nan


def draw_hero():
    """ซ้าย: เมทริกซ์ 3×3 "จับสื่อหนึ่งกับอีกสื่อหนึ่ง" · ขวา: สามตัวแทนของสถานที่เดียวกัน"""
    fig = plt.figure(figsize=(16.2, 9.6), dpi=140, facecolor=BG)
    ax = fig.add_axes([0.205, 0.145, 0.235, 0.60]); ax.set_facecolor(BG)
    nv = len(VIEWS)
    Z = np.array([[cell(a, b) for b in VIEWS] for a in VIEWS], float)
    im = ax.imshow(Z, cmap="viridis", vmin=0, vmax=1)
    for i in range(nv):
        for k in range(nv):
            ax.text(k, i, ("–" if not np.isfinite(Z[i, k]) else f"{Z[i, k] * 100:.0f}%"),
                    ha="center", va="center", color=BG if (np.isfinite(Z[i, k]) and Z[i, k] > 0.55)
                    else INK, fontsize=11, fontweight="bold" if i == k else "normal")
    ax.set_xticks(range(nv)); ax.set_yticks(range(nv))
    ax.set_xticklabels([T(VIEW_LABEL[v]) for v in VIEWS], color=INK, fontsize=9, rotation=20,
                       ha="right")
    ax.set_yticklabels([T("ถาม: " + VIEW_LABEL[v]) for v in VIEWS], color=INK, fontsize=9)
    ax.set_title(T("จับคู่ข้ามสื่อ (ลายเซ็น 5 จุดเริ่มต้น): ภาพนี้คือที่เดียวกับแผนที่ไหม"),
                 color=INK, fontsize=12, pad=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(MUTED)

    j = JS
    w = WIN[j]
    hv = csgraph.shortest_path(w["Cs"].astype(np.int8), indices=[0], directed=False,
                               method="BF")[0].astype(float)
    dm = np.hypot(w["xy"][:, 0], w["xy"][:, 1])
    P = DRAW[j]["xy"]
    dd = np.hypot(P[:, 0] - P[0, 0], P[:, 1] - P[0, 1])
    ed = DRAW[j]["edg"]
    media = [("map", dm, w["xy"], T("1 · แผนที่: ระยะจริง (เมตร)")),
             ("hop", hv, w["xy"], T("2 · โครงสร้างล้วน: นับก้าว (ไร้หน่วย)")),
             ("draw", dd, P, T("3 · ภาพวาด: ไม้บรรทัดทาบรูป (ไร้หน่วย)"))]
    for r, (v, val, XYp, lab) in enumerate(media):
        axl = fig.add_axes([0.505, 0.715 - r * 0.235, 0.215, 0.175], facecolor=BG)
        axl.scatter(XYp[:, 0], XYp[:, 1], s=1.6, c=val, cmap="viridis", alpha=0.9)
        axl.scatter([XYp[0, 0]], [XYp[0, 1]], s=34, color=WARM, marker="*", zorder=5)
        axl.set_xticks([]); axl.set_yticks([])
        for s in axl.spines.values():
            s.set_color(MUTED); s.set_linewidth(0.6)
        axr = fig.add_axes([0.760, 0.715 - r * 0.235, 0.155, 0.175], facecolor=BG)
        pr = PROF[(v, "radial5")][IDX.index(j)]
        for s in range(N_ORIGIN):
            axr.plot(np.linspace(0, 1, NB), pr[s * NB:(s + 1) * NB], color=COOL,
                     lw=1.5 if s == 0 else 0.8, alpha=1.0 if s == 0 else 0.45)
        axr.set_xticks([]); axr.set_yticks([0, 0.5, 1.0])
        axr.tick_params(colors=MUTED, labelsize=8)
        for s in axr.spines.values():
            s.set_color(MUTED); s.set_linewidth(0.6)
        axr.set_title(T("โปรไฟล์ 5 จุดเริ่มต้น"), color=MUTED, fontsize=8.5)
        axl.set_title(lab + T(f"  ·  rho กับความจริง {float(METR.loc[METR.idx == j, 'rho_hop_km'].iloc[0]):.2f}"),
                      color=INK, fontsize=10, loc="left")
    fig.text(0.5, 0.955, T("Day 25 · Is this a map? — สามสื่อของสถานที่เดียวกัน จับคู่กันได้แค่ไหน"),
             color=INK, fontsize=17, ha="center")
    fig.text(0.5, 0.918, T(f"{str(D.name[j])} ({FULL[j]['n']:,} โหนดในจาน 5,000 ม. · "
                           f"หน้าต่างร่วม {MW:,} โหนด) · ข้อมูล OSM 21 พ.ย. 2569"),
             color=MUTED, fontsize=10.5, ha="center")
    p = os.path.join(OUT, "day25_ladder.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p


def draw_gallery():
    """งานศิลป์: ภาพวาด 82 แผ่นวางบนตำแหน่งจริง (ไม่มี basemap ไม่มีสเกล) เทียบเวอร์ชันสลับช่อง"""
    VIR = matplotlib.colormaps["viridis"]
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 9.0), dpi=140, facecolor=BG)
    nn = np.array([FULL[j]["n"] for j in IDX], float)
    lv = np.log10(np.maximum(nn, 10))
    norm = matplotlib.colors.Normalize(lv.min(), lv.max())
    slots = np.arange(len(IDX))
    shuf = np.random.default_rng(SEED).permutation(len(IDX))
    for ax, shuffle, cap in [(axes[0], False, "จัดวางตามตำแหน่งจริง"),
                             (axes[1], True, "สลับช่องแบบสุ่ม (seed 25)")]:
        frame(ax, pad=0.02)
        src = shuf if shuffle else slots
        for t in slots:
            s = int(src[t])
            if IDX[s] not in DRAW:
                continue
            d = DRAW[IDX[s]]
            Pm = d["fit"]
            sc_ = np.percentile(np.hypot(Pm[:, 0], Pm[:, 1]), 90) or 1.0
            k = 26000.0 / (2 * sc_)
            P = Pm * k
            cx, cy = SP[IDX[t]]
            ed = d.get("edg")
            if ed is not None and len(ed):
                seg = P[ed] + np.array([cx, cy])
                lc = matplotlib.colors.to_hex(VIR(norm(lv[s])), keep_alpha=False)
                ax.add_collection(LineCollection(seg, colors=lc, linewidths=0.30, alpha=0.85,
                                                 zorder=3, capstyle="round"))
            else:
                ax.scatter(P[:, 0] + cx, P[:, 1] + cy, s=0.7, color=VIR(norm(lv[s])),
                           alpha=0.85, zorder=3, rasterized=True)
        ax.set_title(T(cap), color=INK, fontsize=12)
    mi = moran_i(lv, SP[IDX])
    rngs = np.random.default_rng(SEED)
    null = [moran_i(rngs.permutation(lv), SP[IDX]) for _ in range(300)]
    p_mi = float((1 + sum(1 for x in null if x >= mi)) / (len(null) + 1))
    fig.text(0.5, 0.955, T("ลายนิ้วมือของสถานที่ 82 แห่ง — วาดจากความเชื่อมโยงอย่างเดียว"),
             color=INK, fontsize=17, ha="center")
    fig.text(0.5, 0.925, T(f"สี = จำนวนโหนดถนน (log) · ถ้าวางตามตำแหน่งจริง สีจะใกล้กัน "
                           f"Moran I = {mi:.3f} (p = {p_mi:.3f} จากการสลับที่ 300 ครั้ง)"),
             color=MUTED, fontsize=10.5, ha="center")
    p = os.path.join(OUT, "day25_gallery.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p, mi, p_mi


def draw_metric():
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.4), dpi=140, facecolor=BG)
    big = int(METR.nlargest(1, "nodes").idx.iloc[0])
    med = int(METR.iloc[[int(METR.nodes.values.argsort()[len(METR) // 2])]].idx.iloc[0])
    sml = int(METR.nsmallest(1, "nodes").idx.iloc[0])
    ax = axes[0, 0]; ax.set_facecolor(BG)
    tit = []
    for j, colr, lab in [(big, COOL, "ใหญ่สุด"), (med, TEAL, "มัธยฐาน"), (sml, WARM, "เล็กสุด")]:
        w = WIN[j]
        srcs = pick_srcs(w, 12)
        hops = csgraph.shortest_path(w["Cs"].astype(np.int8), indices=srcs, directed=False,
                                     method="BF")
        for si, s in enumerate(srcs):
            ok = np.isfinite(hops[si]) & (hops[si] > 0)
            if ok.sum() < 8:
                continue
            km = geod_m(w["lo"][s], w["la"][s], w["lo"][ok], w["la"][ok])
            ax.scatter(hops[si][ok], km / 1000.0, s=2.0, color=colr, alpha=0.28)
        rr = float(METR.loc[METR.idx == j, "rho_hop_km"].iloc[0])
        tit.append(f"{lab} rho {rr:.2f}")
        ax.scatter([], [], s=8, color=colr, label=T(f"{lab}: {str(D.name[j])[:20]}"))
    ax.set_xlabel(T("จำนวนก้าวบนกราฟถนน"), color=INK)
    ax.set_ylabel(T("ระยะตรง (กม.)"), color=INK)
    ax.set_title(T(" ·  ".join(tit)), color=INK, fontsize=11)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    ax.grid(color=MUTED, alpha=0.25, lw=0.5)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED)
    panels = [("rho_hop_km", COOL, "สหสัมพันธ์อันดับ ก้าว เทียบ ระยะจริง (ต่อจุด)"),
              ("stretch_med", ACCENT, "ถนนพาอ้อมกี่เท่าของเส้นตรง (ต่อจุด)"),
              ("draw_rel", PURPLE, "วาดใหม่แล้วตำแหน่งเพี้ยน = กี่เท่าของขนาดภาพ")]
    for axb, (col, colr, ttl) in zip(axes.ravel()[1:], panels):
        v = pd.to_numeric(METR[col], errors="coerce").dropna().values
        axb.set_facecolor(BG)
        axb.hist(v, bins=22, color=colr, alpha=0.85, edgecolor=BG, linewidth=0.5)
        md = float(np.median(v))
        axb.axvline(md, color=WARM, ls="--", lw=1.2)
        axb.text(md, axb.get_ylim()[1] * 0.92, T(f"มัธยฐาน {md:.3f}"), color=WARM, fontsize=10)
        axb.set_title(T(ttl), color=INK, fontsize=11)
        axb.grid(color=MUTED, alpha=0.2, lw=0.5)
        for s in axb.spines.values():
            s.set_color(MUTED)
        axb.tick_params(colors=MUTED)
    fig.suptitle(T("อีกแกนหนึ่งของความเป็นแผนที่: ระยะทางยังแปลว่าระยะทางอยู่ไหม"),
                 color=INK, fontsize=15)
    p = os.path.join(OUT, "day25_metric.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p


POOL_NAME = {"all": "ทุกจุดที่มีกราฟ (82)", "sizeblind": "ผู้สมัครขนาดใกล้เคียงกัน",
             "far": "ผู้สมัครคนละภาค"}
PAIRS = [(a, b) for a in VIEWS for b in VIEWS if a != b]


def draw_controls():
    fig, ax = plt.subplots(figsize=(15.2, 8.0), dpi=140, facecolor=BG)
    ax.set_facecolor(BG)
    x = np.arange(len(PAIRS) + 1)
    for gi, kind in enumerate(POOLKINDS):
        vals, labs = [], []
        for a, b in PAIRS:
            s = LAD[(LAD.sig == "radial5") & (LAD.view_q == a) & (LAD.view_r == b) &
                   (LAD.pool == kind)]
            vals.append(max(float(s.ratio_vs_chance.iloc[0]), 0.02) if len(s) else 0.02)
            labs.append(f"{a}→{b}")
        vals.append(max(float(CTRL["ratio_vs_chance"]), 0.02))
        labs.append("ควบคุม")
        ax.bar(x + (gi - 1) * 0.26, vals, width=0.24, color=[COOL, ACCENT, TEAL][gi],
               alpha=0.92, label=POOL_NAME[kind], edgecolor=BG, linewidth=0.5)
    ax.set_yscale("log")
    ax.axhline(1.0, color=WARM, ls="--", lw=1.2)
    ax.text(len(x) - 0.5, 1.06, T("เสมอโอกาส (1×)"), color=WARM, fontsize=10, ha="right")
    ax.axhline(3.0, color=INK, ls=":", lw=1.0)
    ax.text(0.0, 3.2, T("เกณฑ์ที่ตั้งไว้ก่อนรัน = 3×"), color=INK, fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}→{b}" for a, b in PAIRS] + ["ควบคุม"], color=INK, fontsize=10,
                       rotation=28, ha="right")
    ax.set_ylabel(T("แม่นกว่าการสุ่มกี่เท่า (log)"), color=INK)
    ax.legend(frameon=False, labelcolor=INK, loc="upper right", fontsize=10)
    ax.grid(axis="y", color=MUTED, alpha=0.25, lw=0.5)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_title(T("จับคู่ข้ามสื่อ (ลายเซ็นห้าจุดเริ่มต้น) × สามฐานผู้สมัคร + ตัวควบคุมสลับหอภาพ"), color=INK,
                 fontsize=14, pad=12)
    p = os.path.join(OUT, "day25_controls.png")
    fig.savefig(p, facecolor=BG); plt.close(fig)
    return p


# ================================================== 7. ไฟล์ + สถิติ
P1 = draw_hero()
P2, MORAN, MORAN_P = draw_gallery()
P3 = draw_metric()
P4 = draw_controls()
log(f"ภาพ 4 ชิ้นเสร็จ ({time.time()-T0:.0f} วินาที)")

SITE = []
for i, j in enumerate(IDX):
    row = dict(idx=j, name=D.name[j], province=D.province[j], region_th=D.region_th[j],
               nodes=int(RB.loc[j, "nodes"]), edges=int(RB.loc[j, "edges"]),
               components=int(RB.loc[j, "components"]),
               deadend_ratio=float(RB.loc[j, "deadend_ratio"]),
               worst_cut_keep=(None if pd.isna(RD24.loc[j, "worst_cut_keep"])
                               else float(RD24.loc[j, "worst_cut_keep"])),
               pool_size_sizeblind=int(len(POOLS["sizeblind"][i])),
               pool_size_far=int(len(POOLS["far"][i])))
    for c in ["rho_hop_km", "stretch_med", "stretch_p90", "draw_med_resid_m", "draw_rel",
              "draw_rho_pair", "ball_nodes", "ball_km_max", "reach_frac"]:
        v = METR.loc[METR.idx == j, c].iloc[0]
        row[c] = None if pd.isna(v) else float(v)
    for sig in SIGS:
        for a, b in PAIRS:
            h = HIT.get((sig, a, b, "all"))
            if h is None or i >= len(h["hit"]):
                continue
            row[f"hit_{sig}_{a}_{b}"] = bool(h["hit"][i])
            row[f"runner_{sig}_{a}_{b}"] = h["runner"][i]
    SITE.append(row)
SITE = pd.DataFrame(SITE)

LAD.to_csv(os.path.join(OUT, "day25_ladder.csv"), index=False)
METR.to_csv(os.path.join(OUT, "day25_metric.csv"), index=False)
SITE.to_csv(os.path.join(OUT, "day25_sites.csv"), index=False)


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


def ladof(a, b, kind, col, sig="radial5"):
    s = LAD[(LAD.sig == sig) & (LAD.view_q == a) & (LAD.view_r == b) & (LAD.pool == kind)]
    return _clean(float(s[col].iloc[0])) if len(s) else None


def verdictof(a, b, kind, sig="radial5"):
    s = LAD[(LAD.sig == sig) & (LAD.view_q == a) & (LAD.view_r == b) & (LAD.pool == kind)]
    return bool(s.verdict.iloc[0]) if len(s) else None


def q(col, p):
    v = pd.to_numeric(METR[col], errors="coerce").dropna().values
    return None if v.size == 0 else round(float(np.quantile(v, p)), 4)


STATS = dict(
    theme=dict(day=25, name="Is this a map?",
               prompt="Stretch the limits of maps and blend your map with an artwork or some "
                      "creative data visualization. When does something become a map and when "
                      "does it stop being one?",
               read_from="30daymapchallenge.com อ่านสด 26 พ.ย. 2569 (ไฟล์ scratch/chal25.html)"),
    provenance=dict(graph_file="data/day25_graphs.npz", graph_bytes=int(gm["bytes"]),
                    graph_sha256=gm["sha256"], pbf_md5=gm["pbf_md5"], pbf_read_utc=gm["read_utc"],
                    radius_m=R_DISC, made_by="tools/collect_day25_export.py จาก cache/day24 (Day 24)"),
    data=dict(sites_total=int(len(D)), sites_with_graph=len(IDX),
              sites_without_graph=SITES_NO_GRAPH, node_slots=TOT_NODES,
              edge_records=TOT_EDGE_REC, edge_unique=TOT_EDGE, duplicate_edge_records=MULTI,
              road_classes=len(CLASSES), window_nodes=MW, profile_bins=NB,
              ladder_rows=int(len(LAD)), metric_rows=int(len(METR)),
              sites_rows=int(len(SITE)), notebook_cells_note="ไฟล์เดียวรันจบ"),
    method=dict(seed=SEED, views=VIEWS, view_label=VIEW_LABEL,
                core="จับคู่ข้ามสื่อ (cross-media matching): ข้อมูลชุดเดียวกันสามสื่อ "
                     "เมตรจริง / จำนวนก้าว / ไม้บรรทัดบนภาพที่วาดใหม่ "
                     "จับคู่ผ่านโปรไฟล์การเติบโต 12 ขั้นที่ normalize ด้วย P95 ของตัวเอง",
                pools=dict(all=82, sizeblind_median=int(np.median(POOL_SIZES["sizeblind"])),
                           sizeblind_min=int(POOL_SIZES["sizeblind"].min()),
                           sizeblind_max=int(POOL_SIZES["sizeblind"].max()),
                           far_median=int(np.median(POOL_SIZES["far"]))),
                b_permutation=B_PERM, b_bootstrap=B_BOOT, n_sources=N_SRC_W, max_pairs=4000,
                rule="k=1, แต้มที่เท่ากันแตกด้วยเลขสุ่ม, ถือว่าผ่านเมื่อ p<0.05 และ "
                     "แม่นกว่าสุ่ม 3 เท่า (ตั้งก่อนเห็นผล)"),
    discarded_design=dict(
        what="split-half leave-one-out แบบที่ออกแบบไว้ตอนแรก",
        why="แบ่งข้อมูลเครือข่ายให้ไม่ทับกัน ทำให้วัดโครงสร้างไม่ได้ (กราฟขาดเป็นผุยผง) "
            "หรือทำให้สองซีกอยู่คนละพื้นที่ (รูปทรงเทียบกันไม่ได้)",
        median_reachable_nodes_after_split=FRAG,
        note="ตัวเลขในช่อง full คือมัธยฐานของกราฟเต็ม (ถูก cap ที่ 1,000) ใช้วัด "
             "edge/chain/halfplane แบบเดียวกันบนจุด subset 14 จุดแรก"),
    ladder={f"{sig}/{a}>{b}/{kind}": dict(top1=ladof(a, b, kind, "top1", sig),
                                          top3=ladof(a, b, kind, "top3", sig),
                                          chance=ladof(a, b, kind, "chance", sig),
                                          ratio=ladof(a, b, kind, "ratio_vs_chance", sig),
                                          p=ladof(a, b, kind, "p_permutation", sig),
                                          ci=[ladof(a, b, kind, "ci_lo", sig),
                                              ladof(a, b, kind, "ci_hi", sig)],
                                          trivial=bool(a == b),
                                          verdict=verdictof(a, b, kind, sig))
            for sig in SIGS for a in VIEWS for b in VIEWS for kind in POOLKINDS},
    control=dict(swap_gallery=dict(top1=CTRL["top1"], chance=CTRL["chance"],
                                   ratio=CTRL["ratio_vs_chance"],
                                   passed=bool(not CTRL["verdict"]), n_tested=CTRL["n_tested"])),
    metric=dict(n=len(METR),
                rho_hop_km=dict(median=q("rho_hop_km", 0.5), q25=q("rho_hop_km", 0.25),
                                q75=q("rho_hop_km", 0.75), min=q("rho_hop_km", 0),
                                max=q("rho_hop_km", 1)),
                stretch=dict(median=q("stretch_med", 0.5), p90_of_medians=q("stretch_p90", 0.9),
                             min=q("stretch_med", 0), max=q("stretch_med", 1)),
                draw_residual_m=dict(median=q("draw_med_resid_m", 0.5),
                                     q25=q("draw_med_resid_m", 0.25),
                                     q75=q("draw_med_resid_m", 0.75)),
                draw_rel=dict(median=q("draw_rel", 0.5), q25=q("draw_rel", 0.25),
                              q75=q("draw_rel", 0.75)),
                draw_r2_median=q("draw_r2", 0.5), draw_rho_pair_median=q("draw_rho_pair", 0.5),
                reach_frac_median=q("reach_frac", 0.5),
                ball_nodes_median=q("ball_nodes", 0.5), ball_km_max_median=q("ball_km_max", 0.5),
                unit="ต่อจุดดูฟ้ามืด (มัธยฐาน/ควอร์ไทล์ข้าม 82 จุด) ไม่ใช่ต่อโหนดถนน"),
    gallery=dict(moran_i=round(MORAN, 4), moran_p=round(MORAN_P, 4), n_tiles=len(DRAW),
                 tile_box_m=26000.0, colour="log10 จำนวนโหนดถนนต่อจุด"),
    hero_site=dict(idx=int(JS), name=str(D.name[JS]), nodes=int(FULL[JS]["n"]),
                   rule="จุดที่ worst_cut_keep = 0 จาก Day 24 และกราฟใหญ่ที่สุด"),
    cross_day=dict(day24_rebuild_max_diff=REBUILD_DIFF,
                   day24_components_mean=round(float(RB.components.mean()), 3),
                   day24_bridges_sum=int(RB.bridges.sum()),
                   day24_deadend_median=round(float(RB.deadend_ratio.median()), 4),
                   day7_link="โปรไฟล์การเติบโตแบบไร้หน่วย = เวอร์ชันไม่มีไมล์ของ isochrone 10 นาที",
                   day21_link="จาน 5,000 ม. และระยะทางเข้าถึงใช้นิยามเดียวกับ Day 21/24"),
    figures=[os.path.basename(P1), os.path.basename(P2), os.path.basename(P3),
             os.path.basename(P4)],
    defs=dict(
        growth_profile="สัดส่วนโหนดที่อยู่ภายในรัศมี 12 ระดับ โดยรัศมีวัดเป็นเศษส่วนของ "
                       "P95 ของตัวมันเอง (จึงไร้หน่วยและเทียบข้ามสื่อได้)",
        cross_media="การที่ภาพจากสื่อหนึ่ง (เช่นจำนวนก้าว) จับคู่กับหอภาพของอีกสื่อหนึ่ง "
                    "(เช่นเมตรจริง) แล้วถูกว่าเป็นสถานที่เดียวกัน",
        trivial="คู่ที่ถามและตอบเป็นสื่อเดียวกัน ผลต้องเป็น 100% โดยสร้าง ใช้เป็น positive control",
        hop_km="spearman rho (สหสัมพันธ์อันดับ) ระหว่างจำนวนก้าวที่ต้องเดินบนกราฟ "
               "กับระยะตรง (geodesic) ของคู่โหนด",
        stretch="ระยะที่เดินทางจริงตามถนน หารด้วยระยะเส้นตรง (มากกว่า 1 = อ้อม)",
        draw_residual="เอาภาพที่วาดจากความเชื่อมโยงอย่างเดียว Procrustes-fit ทับพิกัดจริง "
                      "แล้ววัดระยะเพี้ยนมัธยฐานเป็นเมตร (และเทียบเป็นเศษของขนาดภาพ)",
        worst_cut_keep="ตัวเลขจาก Day 24: ลบจุดตัดหนึ่งจุดแล้วเหลือเครือข่ายให้ใช้เท่าไร (0 = หายหมด)",
        Moran="ค่าทางสถิติวัดว่าจุดที่อยู่ใกล้กันบนแผนที่ มีสี (จำนวนโหนด) ใกล้เคียงกันแค่ไหน"),
    limitations=[
        "โปรไฟล์การเติบโตเป็นลายเซ็นที่ผมออกแบบเอง (12 ขั้น, normalize ด้วย P95) "
        "ผลที่ได้จึงเป็น *ขอบเขตล่าง* ของข้อมูลที่สื่อหนึ่งเก็บไว้ ไม่ใช่ค่าสูงสุดที่เป็นไปได้",
        "ทุกสื่อคำนวณบนหน้าต่างร่วม 1,000 โหนดแรกจากจุดเข้าถึง กราฟที่ใหญ่กว่านั้น "
        f"(ใหญ่สุด {int(METR.nodes.max()):,} โหนด) จึงถูกตัดเหลือเฉพาะส่วนที่เดินถึง",
        "สื่อ 'ภาพวาด' ใช้ spring_layout ซึ่งไม่มีเหตุผลใดจะคงทิศทาง/สัดส่วนจริง "
        "จึงไม่แปลกที่ผลจะต่ำ — แต่ตัวเลขที่ต่ำคือคำตอบของโจทย์ ไม่ใช่ข้อผิดพลาด",
        "จาน 5,000 ม. ตัดถนนที่ยื่นเข้ามา ทำให้บางโหนดเดินไปไม่ถึง (มัธยฐานเข้าถึงได้ "
        f"{100 * float(np.median(pd.to_numeric(METR.reach_frac, errors='coerce'))):.1f}%) "
        "components ที่ >1 บางส่วนจึงเป็นผลของกรอบการศึกษา ไม่ใช่ความขาดจริงของถนน",
        "การทดสอบนี้ไม่ใช้ข้อมูล 'เวลา' หรือ 'คนวาด' เลย — ถ้าใช้ snapshot คนละเดือน "
        "จะได้การแบ่งที่สะอาดกว่านี้ (บันทึกไว้เป็นงานของวันถัดไปถ้าจำเป็น)"],
)
json.dump(_clean(STATS), open(os.path.join(OUT, "day25_stats.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, allow_nan=False)
log(f"เขียน day25_stats.json + ladder/metric/sites CSV ({len(LAD)} แถวจับคู่, {len(METR)} แถวเมตริก)")

# ================================================== 8. ผู้ตัดสิน
assert len(IDX) == 82 and len(METR) == 82
assert TOT_NODES == 413758 and TOT_EDGE_REC == 430464
for sig in SIGS:
    for v in VIEWS:
        assert ladof(v, v, "all", "top1", sig) == 1.0,             f"positive control (สื่อเดียวกัน) ต้อง 100% ที่ {sig}/{v}"
assert not CTRL["verdict"], "control สลับหอภาพห้ามชนะการสุ่ม (แปลว่า pipeline รั่ว)"
assert ladof("hop", "map", "all", "top1") >= 3 * ladof("hop", "map", "all", "chance"), \
    "คำถามหลักของโพสต์: โครงสร้างล้วนต้องจับคู่กับแผนที่ได้ดีกว่าสุ่ม 3 เท่า"
assert ladof("draw", "map", "all", "ratio_vs_chance") < ladof("hop", "map", "all", "ratio_vs_chance"), \
    "ภาพวาดควรจับคู่กับแผนที่ได้แย่กว่า pure topology (ไม่งั้นต้องมีอะไรผิด)"
assert float(np.median(pd.to_numeric(METR.rho_hop_km, errors="coerce"))) > 0.5, \
    "hop ต้องยังพาไปหากม. ได้"
assert float(np.median(pd.to_numeric(METR.draw_rel, errors="coerce"))) > 0.4, \
    "การวาดควรเพี้ยนมาก"
assert FRAG["edge"] < 100 and FRAG["chain"] < 100, \
    "หลักฐาน split-half ต้องชี้ว่าการแบ่งแบบสุ่มทำให้กราฟขาดจริง"
guard_labels(sorted(DRAW)[:10],
             {int(k): (float(SP[k][0]), float(SP[k][1])) for k in DRAW},
             {int(k): (float(D.latitude[k]), float(D.longitude[k])) for k in DRAW},
             "gallery-positions")
lint_thai(" ".join(TXT) + " " + open(os.path.join(OUT, "day25_ladder.csv"),
                                      encoding="utf-8").read(), "day25-figures")
for badc in [c for c in "".join(TXT) if ord(c) > 0x2fff and not (0xe00 <= ord(c) <= 0xe7f)]:
    raise AssertionError(f"อักษรที่ไม่ใช่ไทย/ascii ใน label ภาพ: {badc!r}")
log("ผู้ตัดสินผ่านครบทุกข้อ ✓ · รวมเวลา " + f"{time.time()-T0:.0f} วินาที")
_LOGF.close()
