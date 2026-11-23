# -*- coding: utf-8 -*-
"""
Day 23 · Senses: Taste — "Food, drink or something else which you can taste."
ธีมที่อาจารย์ขอ: **อาหารภาคเหนือ** — ลาบ · ข้าวซอย · ไส้อั่ว · น้ำพริกหนุ่ม · น้ำพริกอ่อง ·
แคบหมู · ฮังเล · น้ำเงี้ยว · แกงแค · ตำขนุน · แอ็บ · จิ้นส้ม
(+ คำควบคุม ลาบ / หน่อไม้ ที่ไม่ได้มีเฉพาะภาคเหนือ เพื่อทดสอบว่าวิธีนี้ไม่ได้ "เห็นเป็นเหนือหมด")

โจทย์ (30daymapchallenge.com หัวข้อ Day 23 'Senses: Taste' อ่านสด 20:55 น. วันที่ 23 พ.ย. 2569)
เป็น senses วันที่สี่ของซีรีส์ (Day 5 sight · Day 12 sound · Day 19 smell · Day 23 taste)

ข้อค้นพบที่กำหนดวิธีทั้งหมด: **OSM ไม่ได้บันทึก "รส" ของอาหารไทยไว้เป็นหมวดหมู่**
  แท็ก `cuisine` มีคนใช้จริง 665 ค่าทั่วประเทศไทย แต่ค่าที่เอ่ยถึงอาหารเมือง
  (ข้าวซอย / ลาบ /ไส้อั่ว / แกงฮังเล / แคบหมู / Lanna / khantoke) แทบเป็นศูนย์
  → สิ่งที่ OSM บันทึกเรื่อง "รส" ไว้จริง ๆ คือ *ป้ายชื่อร้าน* (คน map ท้องที่เขียนตามที่ป้ายเขียน)
  แผ่นนี้จึงวัด "รส" จาก name ของ object ใน OSM ไม่ใช่จากหมวดอาหาร

คำถามที่ตอบได้
  1) `cuisine` ของไทยบันทึกอะไรไว้จริง (665 ค่า vs คำอาหารเมือง = กี่ครั้ง)
  2) ชื่อร้านที่โฆษณาอาหารเมือง กระจายอยู่ตรงไหน และสัดส่วนอยู่ใน 17 จังหวัดเหนือเท่าไร
  3) จังหวัดเหนือไหน "ขายรสเมืองในชื่อร้าน" เข้มข้นสุด (ทั้งจำนวนจริงและต่อ 1,000 ตร.กม.)
  4) แหล่งดูฟ้ามืดภาคเหนือ 23 แห่งอยู่ไกลจากร้านแบบนั้นกี่กิโลเมตร
  5) ร้านของย่าง (ลาบ/ไส้อั่ว/แคบหมู) เกี่ยวอะไรกับฝุ่น/CO ที่วัดไว้ใน Day 19 ไหม

ทุกตัวเลขในโพสต์อ่านจาก day23_stats.json + day23_*.csv ที่สคริปต์นี้เขียนเท่านั้น
"""
import os, sys, json, time, math, re, gzip, collections
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.validation import make_valid
from shapely.strtree import STRtree
from pyproj import CRS, Geod, Transformer
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

T0 = time.time()
COLAB = (os.environ.get("DAY23_FORCE_COLAB") == "1") or not os.path.isdir(
    os.path.join(r"D:\GIS30Days", "geo-notes"))
WORK = os.getcwd() if COLAB else r"D:\GIS30Days"
REPO = os.path.join(WORK, "geo-notes")
CACHE = os.path.join(WORK, "day23_cache") if COLAB else os.path.join(WORK, "cache", "day06")
FOOD = os.path.join(WORK, "cache", "day23")
OUT = os.path.join(WORK, "out", "day23")
os.makedirs(OUT, exist_ok=True)
RAW = "https://raw.githubusercontent.com/thammarat-ai/geo-notes/main/"
ADM1 = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"


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
    import zipfile
    for f in ("Trirong-Regular.ttf", "Trirong-Bold.ttf", "Trirong-SemiBold.ttf"):
        ensure("https://github.com/google/fonts/raw/main/ofl/trirong/" + f, os.path.join(CACHE, f))
    repo_file("data/darksky_thailand.csv")
for _f in sorted(os.listdir(CACHE)):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(CACHE, _f))
plt.rcParams["font.family"] = "Trirong"
_LOGF = open(os.path.join(OUT, "day23_run.log"), "w", encoding="utf-8")
def log(m):
    s = f"[{time.time()-T0:6.1f}s] {m}"
    print(s, flush=True)
    _LOGF.write(s + chr(10)); _LOGF.flush()

for _td in (os.path.join(WORK, "tools"), os.path.join(REPO, "tools")):
    if os.path.isdir(_td):
        sys.path.insert(0, _td); break
try:
    from guards import guard_bbox, guard_crs, guard_area, guard_variance, lint_thai
except ModuleNotFoundError:
    ensure(RAW + "tools/guards.py", os.path.join(CACHE, "guards.py"))
    sys.path.insert(0, CACHE)
    from guards import guard_bbox, guard_crs, guard_area, guard_variance, lint_thai

TXT = []
def T(s):
    TXT.append(s); return s

BG, INK, MUTED = "#0d1522", "#e8e3d3", "#9fb2c8"
ACCENT, WARM, COOL, TEAL, PURPLE, LIME, ROSE = ("#f5b942", "#f4566a", "#38bdf8", "#2dd4bf",
                                                "#c084fc", "#a3e635", "#fb7185")
EQ = CRS.from_proj4("+proj=eqdc +lat_0=14 +lon_0=101 +lat_1=8 +lat_2=20 "
                    "+units=m +no_defs +ellps=WGS84")
GEOD = Geod(ellps="WGS84")

# ภาคเหนือ 17 จังหวัด = 9 เหนือบน + 8 เหนือล่าง (ตามที่ Wikipedia: Northern Thailand ระบุ)
NORTH_UPPER = ["Chiang Mai", "Lamphun", "Lampang", "Uttaradit", "Phrae", "Nan", "Phayao",
               "Chiang Rai", "Mae Hong Son"]
NORTH_LOWER = ["Kamphaeng Phet", "Nakhon Sawan", "Phetchabun", "Phichit", "Phitsanulok",
               "Sukhothai", "Uthai Thani", "Tak"]
NORTH = NORTH_UPPER + NORTH_LOWER
PROV_TH = {"Chiang Mai": "เชียงใหม่", "Chiang Rai": "เชียงราย", "Lamphun": "ลำพูน",
           "Lampang": "ลำปาง", "Nan": "น่าน", "Phayao": "พะเยา", "Phrae": "แพร่",
           "Uttaradit": "อุตรดิตถ์", "Mae Hong Son": "แม่ฮ่องสอน", "Tak": "ตาก",
           "Sukhothai": "สุโขทัย", "Phitsanulok": "พิษณุโลก", "Phichit": "พิจิตร",
           "Kamphaeng Phet": "กำแพงเพชร", "Nakhon Sawan": "นครสวรรค์",
           "Uthai Thani": "อุทัยธานี", "Phetchabun": "เพชรบูรณ์"}
LANNA = ["ข้าวซอย", "ไส้อั่ว", "น้ำพริกหนุ่ม", "น้ำพริกอ่อง", "แคบหมู", "ฮังเล",
         "น้ำเงี้ยว", "แกงแค", "ตำขนุน", "แอ็บ", "จิ้นส้ม"]
CONTROL = ["ลาบ", "หน่อไม้"]
WORDS = LANNA + CONTROL
GLOSS = {"ข้าวซอย": "ข้าวซอย · บะหมี่น้ำแกงกะทิ", "ไส้อั่ว": "ไส้อั่ว · ไส้กรอกสมุนไพรย่าง",
         "น้ำพริกหนุ่ม": "น้ำพริกหนุ่ม · พริกย่างเขียวอ่อน", "น้ำพริกอ่อง": "น้ำพริกอ่อง",
         "แคบหมู": "แคบหมู · หนังหมูดิบ", "ฮังเล": "แกงฮังเล · แกงเนื้อตุ๋น (รากพม่า)",
         "น้ำเงี้ยว": "น้ำเงี้ยว · บะหมี่น้ำกะทิ-มะเขือ", "แกงแค": "แกงแค · แกงผักหลายชนิด",
         "ตำขนุน": "ตำขนุน · ขนุนอ่อนตำ", "แอ็บ": "แอ็บ · ไข่/เนื้อห่อใบตองย่าง",
         "จิ้นส้ม": "จิ้นส้ม · หมูหมักเปรี้ยว", "ลาบ": "ลาบ · เนื้อสับ (เหนือและอีสาน)",
         "หน่อไม้": "หน่อไม้ · ชุดควบคุม (ภาคอื่นก็ใช้)"}
# สีต่อคำ (13 คำ) — ไล่โทนให้แยกได้ด้วยตา
COLOR = {"ข้าวซอย": ACCENT, "ไส้อั่ว": WARM, "น้ำพริกหนุ่ม": TEAL, "น้ำพริกอ่อง": COOL,
         "แคบหมู": PURPLE, "ฮังเล": LIME, "น้ำเงี้ยว": ROSE, "แกงแค": "#f9a8d4",
         "ตำขนุน": "#fbbf24", "แอ็บ": "#34d399", "จิ้นส้ม": "#a78bfa",
         "ลาบ": "#e2e8f0", "หน่อไม้": "#64748b"}

# ================================================== 1. ข้อมูลตั้งต้น
D = pd.read_csv(repo_file("data/darksky_thailand.csv"))
D["name"] = D["name"].astype(str).str.strip()
P = np.array([[float(r.longitude), float(r.latitude)] for _, r in D.iterrows()])
guard_bbox(P, 96.5, 107.6, 4.3, 21.7, "darksky-points")
trs = Transformer.from_crs("EPSG:4326", EQ, always_xy=True)
SP = np.array(trs.transform(P[:, 0], P[:, 1])).T
guard_crs([SP[:, 0].min(), SP[:, 1].min(), SP[:, 0].max(), SP[:, 1].max()],
          "projected-metres", expect="metres")
SITE_PT = {j: Point(SP[j][0], SP[j][1]) for j in range(len(D))}
NSITES = D.index[D.region_th == "ภาคเหนือ"].tolist()
log(f"ทะเบียน {len(D)} จุด · ภาคเหนือ {len(NSITES)} จุด · จังหวัด "
    f"{sorted(set(str(x) for x in D.province[NSITES]))}")


def load_gz(nm):
    with gzip.open(os.path.join(FOOD, nm), "rb") as f:
        return json.loads(f.read().decode())


food_nodes, food_ways = load_gz("food_nodes.json.gz"), load_gz("food_ways.json.gz")
dn, dw = load_gz("dish_nodes.json.gz"), load_gz("dish_ways.json.gz")
dm = json.load(open(os.path.join(FOOD, "dish_meta.json"), encoding="utf-8"))
zf = np.load(os.path.join(FOOD, "food_node_locs.npz"))
FLOC = {int(i): (float(x), float(y)) for i, x, y in zip(zf["id"], zf["lon"], zf["lat"])}
zd = np.load(os.path.join(FOOD, "dish_node_locs.npz"))
DLOC = {int(i): (float(x), float(y)) for i, x, y in zip(zd["id"], zd["lon"], zd["lat"])}
FOOD_ALL = len(food_nodes) + len(food_ways)
NODE_SHARE = round(100.0 * len(food_nodes) / FOOD_ALL, 2)
log(f"object อาหารทั้งประเทศ: node {len(food_nodes):,} + way {len(food_ways):,} "
    f"(node = {NODE_SHARE}%) | ร้านที่ชื่อตรงคำอาหารเมือง: node {len(dn):,} + way {len(dw):,}")

# ================================================== 2. "รส" ในแท็กหมวดหมู่เป็นอย่างไร
cuis, diet, amen = collections.Counter(), collections.Counter(), collections.Counter()
oh_n = 0
for e in food_nodes + food_ways:
    t = e["tags"]
    for v in str(t.get("cuisine", "")).split(";"):
        if v.strip():
            cuis[v.strip().lower()] += 1
    for v in str(t.get("diet", "")).split(";"):
        if v.strip():
            diet[v.strip().lower()] += 1
    if str(t.get("opening_hours", "")).strip():
        oh_n += 1
    if t.get("amenity"):
        amen[t["amenity"]] += 1
CUIS_TOTAL = int(sum(cuis.values()))
PROBE = {"ข้าวซอย": ["ข้าวซอย", "khao soi", "khao-soi", "khawsao"],
         "ลาบ": ["ลาบ", "larb", "laap", "lab"],
         "ไส้อั่ว": ["ไส้อั่ว", "sai ua", "saiua", "sai-ua"],
         "น้ำพริกหนุ่ม": ["น้ำพริกหนุ่ม", "nam phrik num", "namphriknum"],
         "แคบหมู": ["แคบหมู", "kaep mu", "chaepmu", "crackling"],
         "ฮังเล": ["ฮังเล", "hang le", "hanglay", "hung lay", "aruna"],
         "น้ำเงี้ยว": ["น้ำเงี้ยว", "ngiao", "ngaew"],
         "จิ้นส้ม": ["จิ้นส้ม", "jin som", "chinsom"],
         "แกงแค": ["แกงแค", "gaeng kae", "kaeng khae"],
         "อาหารเหนือ / Lanna": ["northern", "lanna", "lan na", "khantoke",
                                  "kantoke", "อาหารเมือง", "เมือง"],
         "อาหารอีสาน (เทียบ)": ["อีสาน", "isaan", "isarn", "northeastern"]}
cuisine_probe = {}
for th, vs in PROBE.items():
    hit = {k: c for k, c in cuis.items() if any(v in k for v in vs)}
    cuisine_probe[th] = dict(uses=int(sum(hit.values())),
                             values=dict(sorted(hit.items(), key=lambda kv: -kv[1])[:3]))
LANNA_CUIS = sum(v["uses"] for k, v in cuisine_probe.items() if "อีสาน" not in k)
log(f"cuisine ที่ใช้จริง {len(cuis):,} ค่า รวม {CUIS_TOTAL:,} ครั้ง · top5 "
    f"{[k for k, _ in cuis.most_common(5)]} · diet {len(diet)} ค่า "
    f"({dict(diet.most_common(4))}) · คำอาหารเมืองใน cuisine = {LANNA_CUIS} ครั้ง")

# ================================================== 3. จังหวัด (Natural Earth admin_1)
adp = os.path.join(FOOD, "ne_10m_admin_1_states_provinces.shp")
if not os.path.exists(adp):
    import zipfile
    ensure(ADM1, os.path.join(FOOD, "adm1.zip"))
    zipfile.ZipFile(os.path.join(FOOD, "adm1.zip")).extractall(FOOD)
A1 = gpd.read_file(adp)
A1 = A1[A1.admin == "Thailand"][["name", "geometry"]].reset_index(drop=True)
assert len(A1) == 77, f"Natural Earth ต้องมี 77 จังหวัดไทย ได้ {len(A1)}"
A1["area_km2"] = [abs(GEOD.geometry_area_perimeter(g)[0]) / 1e6 for g in A1.geometry]
AREA = dict(zip(A1.name, A1.area_km2.round(1)))
# ไม่เทียบกับ "เนื้อที่ราชการ" เพราะยังไม่ยืนยันแหล่งในปีนี้ — ใช้ band หยาบ ๆ กันหน่วยผิด
# (Natural Earth admin_1 ให้พื้นที่หยาบกว่าตัวเลขกรมที่ดิน แต่ใช้ค่าเดียวกันทั้ง 17 จังหวัด
#  อัตราส่วนระหว่างจังหวัดจึงใช้ได้ ส่วนค่าสัมบูรณ์ให้ถือว่าเป็นตัวเลขของ NE)
for _n in ("Chiang Mai", "Nakhon Sawan", "Mae Hong Son"):
    assert 5_000 < float(AREA[_n]) < 50_000, f"พื้นที่จังหวัด {_n} = {AREA[_n]} ตร.กม. ผิดหน่วย"
NORTH_AREA = round(float(sum(AREA[n] for n in NORTH if n in AREA)), 1)
missing_prov = [n for n in NORTH if n not in AREA]
assert not missing_prov, f"จังหวัดเหนือไม่พบใน admin_1: {missing_prov}"
a1t = STRtree(list(A1.geometry.values))


def province_of(g):
    """จุดแทนวัตถุ (representative point) ตกในจังหวัดไหน — ถ้าไม่เจอนับกับขอบที่ใกล้ที่สุด"""
    c = g.representative_point()
    for i in a1t.query(c, predicate="within"):
        return str(A1.name.iat[int(i)]), False
    cand = [int(i) for i in a1t.query(c.buffer(0.04))]
    if cand:
        d = min((A1.geometry.iat[i].distance(c), i) for i in cand)
        return str(A1.name.iat[d[1]]), True
    return "", True


# ---- กรอง false positive ก่อน (บั๊กที่ตรวจเจอด้วยตา: "ลาบ" ชนกับ "กุหลาบ"/"สะลาบ"/"แม่สลาบ"
# ซึ่งเป็นดอกไม้กับชื่อหมู่บ้าน ไม่ใช่ร้านลาบ) → ใช้ negative lookbehind เฉพาะคำนั้น
WORD_RE = {w: re.compile(re.escape(w)) for w in WORDS}
# "ลาบ" เป็น substring ของ คำที่ไม่เกี่ยว: ปลาบึก/ตลาดปลาบ่น (ป+ลาบ), ศาลาบู (า-ล-า-บ),
# เลาบราเดอร์ส (เ+ลาบ), พลาบูรพาพัฒน์ (พ+ลาบ), สะลาบ/แม่สลาบ/กุหลาบ (ส/ะ/ห)
# → ห้ามตัวหน้าเหล่านี้ (บาลานซ์ระหว่าง false positive กับการตัด "ปลาบ" ของจริงออก)
# ห้ามเฉพาะตัวหน้าที่ได้ตรวจเห็นจริงในข้อมูลไทย: ห (กุหลาบ), ส (แม่สลาบ), ะ (สะลาบ),
# า (ศาลาบู/นาลาบี), ำ, ป (ปลาบึก/ปลาบ่น), พ (พลาบูรพา), เ (เลาบราเดอร์ส), ู (คูลาบน)
WORD_RE["ลาบ"] = re.compile("(?<![หสะาำปพเื])ลาบ")


def norm_th(x):
    return "".join(str(x).split())


def refine(blob, ws):
    """ตรวจซ้ำบน string เดียวกับที่สแกนใช้ (name + alt_name + official_name + sign + brand)
    — ถ้าตรวจแค่ name จะเผลอตัดร้านที่ตรงกับคำผ่าน alt_name ทิ้ง (บั๊กที่เจอจริงรอบแรก)"""
    n = norm_th(blob)
    return [w for w in ws if WORD_RE[w].search(n)]


# ================================================== 4. ร้านที่ "ชื่อ" พูดถึงอาหารเมือง
recs = [dict(id=e["id"], words=e["words"], tags=e["tags"], geom=Point(e["lon"], e["lat"]),
             kind="node", blob=e.get("blob", norm_th(e["tags"].get("name", "")))) for e in dn]
bad_way = 0
for e in dw:
    co = [DLOC[i] for i in e["nodes"] if i in DLOC]
    if len(co) < 2:
        bad_way += 1
        continue
    g = None
    if len(co) >= 4 and co[0] == co[-1]:
        try:
            g = Polygon(co)
            if not g.is_valid:
                g = make_valid(g)
            if g.geom_type == "MultiPolygon":
                g = max(g.geoms, key=lambda q: q.area)
            if g.geom_type != "Polygon":
                g = None
        except Exception:
            g = None
    if g is None:
        g = LineString(co)
    recs.append(dict(id=e["id"], words=e["words"], tags=e["tags"], geom=g, kind="way",
                     blob=e.get("blob", norm_th(e["tags"].get("name", "")))))
H = gpd.GeoDataFrame(recs, geometry="geom", crs="EPSG:4326")
H["names"] = H.tags.map(lambda t: str(t.get("name", "")))

_before = collections.Counter()
for ws in H.words:
    for w in ws:
        _before[w] += 1
H["words"] = [refine(bl, ws) for bl, ws in zip(H.blob, H.words)]
dropped = H.words.map(len).eq(0)
_after = collections.Counter()
for ws in H.words[~dropped]:
    for w in ws:
        _after[w] += 1
FP_NAMES = H.names[dropped].head(12).tolist()
H = H[~dropped].reset_index(drop=True)
H_ALL = H.copy()                                # ไว้ทำ audit sheet ทุกชิ้น
FOOD_AMEN = {"restaurant", "cafe", "fast_food", "food_court", "bar", "pub", "biergarten",
             "nightclub", "ice_cream", "bakery", "deli", "butchery", "concession_stand",
             "canteen", "market", "juice", "drinking_water"}
FOOD_SHOP = {"supermarket", "convenience", "bakery", "butcher", "deli", "greengrocer", "cheese",
             "pastry", "seafood", "health_food", "organic", "wine", "alcohol", "beverage",
             "delicatessen", "chocolate", "coffee", "tea", "spice"}
H["is_food"] = H.tags.map(lambda t: (t.get("amenity") in FOOD_AMEN) or
                          (t.get("shop") in FOOD_SHOP) or bool(str(t.get("cuisine", "")).strip()))
N_FOOD = int(H.is_food.sum())
H = H[H.is_food].reset_index(drop=True)          # ← ทุกตัวเลขต่อจากนี้คือ "ร้านที่ OSM ระบุว่าเป็นอาหาร"
NONFOOD_NAMES = H.names[~H.is_food].head(12).tolist()
log(f"กรอง false positive: ก่อน {sum(_before.values()):,} คู่(คำ×ร้าน) → หลัง "
    f"{sum(_after.values()):,} (ตัดร้านทิ้ง {int(dropped.sum()):,} แห่ง เช่น "
    f"{FP_NAMES[:3]}) · เหลือ {len(H):,} ร้าน ที่เป็นตัวอาหารจริง {N_FOOD:,}")
prv = [province_of(g) for g in H.geometry.values]
H["province"] = [p for p, _ in prv]
H["edge"] = [e for _, e in prv]
H["in_north"] = H.province.isin(NORTH)
Hm = H.to_crs(EQ)
guard_bbox(np.column_stack([H.geometry.representative_point().x,
                              H.geometry.representative_point().y]),
               96.5, 107.6, 4.3, 21.7, "dish-hits")
N_HIT = len(H)
N_NORTH = int(H.in_north.sum())
log(f"ร้านที่ชื่อตรงคำอาหารเมือง {N_HIT:,} ชิ้น (node {len(dn):,} · way ที่สร้าง geometry ได้ "
    f"{N_HIT - len(dn):,} · way ที่ node ไม่ครบ {bad_way:,}) · ใน 17 จังหวัดเหนือ {N_NORTH:,} "
    f"({100.0 * N_NORTH / max(1, N_HIT):.1f}%) · area ภาคเหนือ {NORTH_AREA:,.1f} ตร.กม.")

rows = []
for w in WORDS:
    sub = H[H.words.map(lambda x: w in x)]
    subf = sub[sub.is_food]
    pvn = sub[sub.in_north].province.value_counts()
    pvx = sub[~sub.in_north].province.value_counts()
    rows.append(dict(word=w, gloss=GLOSS[w], group="lanna" if w in LANNA else "control",
                     hits=int(len(sub)), in_north=int(sub.in_north.sum()),
                     north_share_pct=round(100.0 * sub.in_north.mean(), 2) if len(sub) else 0.0,
                     provinces=int(sub.province.nunique()), north_provinces=int(pvn.size),
                     north_per_1000km2=round(1000.0 * sub.in_north.sum() / NORTH_AREA, 4),
                     top_north="; ".join([f"{PROV_TH.get(k, k)} {v}" for k, v in pvn.head(4).items()]),
                     top_outside="; ".join([f"{k} {v}" for k, v in pvx.head(4).items()]),
                     as_node=int((sub.kind == "node").sum()), as_way=int((sub.kind == "way").sum()),
                     food_in_north=int(sub.in_north.sum()),
                     with_amenity=int(sub.tags.map(lambda t: bool(t.get("amenity"))).sum()),
                     with_cuisine=int(sub.tags.map(lambda t: bool(t.get("cuisine"))).sum()),
                     with_oh=int(sub.tags.map(lambda t: bool(t.get("opening_hours"))).sum()),
                     example_names=" | ".join(sub.names[sub.names.ne("")].head(5).tolist())))
WS = pd.DataFrame(rows)
LANNA_HITS = int(H.words.map(lambda x: any(w in LANNA for w in x)).sum())
NORTH_LANNA = int((H.words.map(lambda x: any(w in LANNA for w in x)) & H.in_north).sum())
print("\n=== ต่อคำ (ทั้งประเทศ vs 17 จังหวัดเหนือ) ===")
print(WS[["word", "group", "hits", "in_north", "north_share_pct", "north_provinces",
          "north_per_1000km2", "with_amenity", "with_cuisine"]].to_string(index=False))

# ================================================== 5. จังหวัดเหนือ: base อาหาร vs ชื่ออาหารเมือง
FP = gpd.GeoDataFrame(geometry=[Point(e["lon"], e["lat"]) for e in food_nodes], crs="EPSG:4326")
FP["province"] = [province_of(g)[0] for g in FP.geometry.values]
fp = collections.Counter(FP.province.tolist())
lanna_mask = H.words.map(lambda x: any(w in LANNA for w in x))
pv_rows = []
for nm in NORTH:
    sub = H[(H.province == nm) & lanna_mask]
    pv_rows.append(dict(province=nm, province_th=PROV_TH.get(nm, ""), area_km2=float(AREA.get(nm)),
                        food_nodes=int(fp.get(nm, 0)), lanna_named=int(len(sub)),
                        lanna_pct_of_food=round(100.0 * len(sub) / max(1, fp.get(nm, 0)), 3),
                        lanna_per_1000km2=round(1000.0 * len(sub) / max(1.0, AREA.get(nm, 1.0)), 3),
                        **{f"d_{w}": int(H[(H.province == nm)]
                                         .words.map(lambda x, ww=w: ww in x).sum())
                           for w in WORDS}))
PV = pd.DataFrame(pv_rows).sort_values("lanna_named", ascending=False)
print("\n=== จังหวัดภาคเหนือ ===")
print(PV[["province_th", "area_km2", "food_nodes", "lanna_named", "lanna_pct_of_food",
          "lanna_per_1000km2", "d_ข้าวซอย", "d_ไส้อั่ว", "d_ลาบ"]].to_string(index=False))

# ================================================== 6. แหล่งฟ้ามืดเหนือ ↔ ระยะไปหารสเมือง
FG = gpd.GeoDataFrame(
    geometry=[Point(e["lon"], e["lat"]) for e in food_nodes] +
             [LineString([FLOC[i] for i in e["nodes"] if i in FLOC])
              for e in food_ways if len([i for i in e["nodes"] if i in FLOC]) >= 2],
    crs="EPSG:4326").to_crs(EQ)
Hf = H[H.is_food].reset_index(drop=True)          # ใช้ร้านที่ OSM ระบุว่าเป็นอาหารจริง
Hfm = Hf.to_crs(EQ)
tree_h = STRtree(list(Hfm.geometry.values))
tree_f = STRtree(list(FG.geometry.values))
srows = []
for j in NSITES:
    pt = SITE_PT[j]
    ch = [int(i) for i in tree_h.query(pt.buffer(40000, quad_segs=6), predicate="intersects")]
    dh = np.array([Hfm.geometry.iat[i].distance(pt) for i in ch]) if ch else np.array([])
    cf = [int(i) for i in tree_f.query(pt.buffer(40000, quad_segs=6), predicate="intersects")]
    dfd = np.array([FG.geometry.iat[i].distance(pt) for i in cf]) if cf else np.array([np.nan])
    row = dict(idx=j, name=D.name[j], province=D.province[j], lat=float(D.latitude[j]),
               lon=float(D.longitude[j]),
               nearest_food_km=round(float(np.nanmin(dfd)) / 1000.0, 2) if len(dfd) else np.nan,
               nearest_lanna_km=round(float(np.min(dh)) / 1000.0, 2) if len(dh) else np.nan,
               lanna_within_10=int(np.sum(dh <= 10000)) if len(dh) else 0,
               lanna_within_40=int(len(dh)) if len(dh) else 0,
               lanna_within_10_name="")
    if len(dh):
        # บั๊กที่เจอตอน audit: `dh` เรียงตามลำดับของ ch ไม่ใช่ index ของ Hfm
        # → ต้อง map กลับผ่าน ch ไม่งั้น "ชื่อร้านที่ใกล้ที่สุด" จะสลับกันคนละร้าน
        k = ch[int(np.argmin(dh))]
        row["nearest_lanna_name"] = Hfm.names.iat[k]
        row["nearest_lanna_province"] = Hfm.province.iat[k]
        row["nearest_lanna_words"] = ", ".join(sorted(Hfm.words.iat[k]))
    for w in WORDS:
        sel = [d for i, d in zip(ch, dh) if w in Hfm.words.iat[i]]
        row[f"near_{w}_km"] = round(min(sel) / 1000.0, 2) if sel else np.nan
        row[f"cnt10_{w}"] = int(sum(1 for d in sel if d <= 10000))
    srows.append(row)
NS = pd.DataFrame(srows)
d16 = pd.read_csv(os.path.join(WORK, "out", "day16", "day16_sites_osm.csv"))
d19 = pd.read_csv(os.path.join(WORK, "out", "day19", "day19_smell_sites.csv"))
NJ = (NS.merge(d16[["idx", "ways", "users", "urban_pop_30km"]], on="idx")
        .merge(d19[["idx", "smell_index", "signature", "CO", "PM2.5"]], on="idx",
               suffixes=("", "_d19")))
GRILL = ["ลาบ", "ไส้อั่ว", "แคบหมู"]
NJ["grill_cnt10"] = NJ[[f"cnt10_{w}" for w in GRILL]].sum(axis=1)
NJ["no_lanna_within_40"] = NJ.nearest_lanna_km.isna() | (NJ.nearest_lanna_km > 40)
rho_co, p_co = spearmanr(NJ.grill_cnt10.values, NJ.CO.values)
rho_pm, p_pm = spearmanr(NJ.grill_cnt10.values, NJ["PM2.5"].values)
rho_way, p_way = spearmanr(NJ.lanna_within_40.values, NJ.ways.values)
rho_sm, p_sm = spearmanr(NJ.lanna_within_40.values, NJ.smell_index.values)
rho_fd, p_fd = spearmanr(NJ.nearest_lanna_km.fillna(999).values,
                         NJ.nearest_food_km.fillna(999).values)
# ตัวควบคุม: ทั้ง "ร้านชื่อย่าง" และ CO ตามความหนาแน่นของที่ตั้ง จึงต้องรายงาน go-between ด้วย
rho_gp, p_gp = spearmanr(NJ.grill_cnt10.values, np.log1p(NJ.urban_pop_30km.values))
rho_pc, p_pc = spearmanr(np.log1p(NJ.urban_pop_30km.values), NJ.CO.values)
log(f"ตัวควบคุม: ร้านย่าง ↔ ประชากรเมือง (log) {rho_gp:.3f} (p {p_gp:.3g}) และ "
    f"ประชากรเมือง ↔ CO {rho_pc:.3f} (p {p_pc:.3g}) — ถ้าสองคู่นี้สูงมาก "
    "ความสัมพันธ์กับ CO คือผลของความเป็นเมือง ไม่ใช่กลิ่นควัน")
print("\n=== แหล่งฟ้ามืดภาคเหนือ (กม.) ===")
print(NJ[["name", "province", "nearest_food_km", "nearest_lanna_km", "lanna_within_10",
          "lanna_within_40", "near_ข้าวซอย_km", "near_ลาบ_km", "ways"]].to_string(index=False))
log(f"Spearman (เหนือ {len(NJ)} จุด): ร้านของย่าง 10 กม. ↔ CO {rho_co:.3f} (p {p_co:.3g}) · "
    f"↔ PM2.5 {rho_pm:.3f} (p {p_pm:.3g}) · ร้านชื่ออาหารเมือง 40 กม. ↔ ways Day 16 "
    f"{rho_way:.3f} (p {p_way:.3g}) · ↔ smell_index {rho_sm:.3f} (p {p_sm:.3g}) · "
    f"ไกลจากร้านอาหารเมือง ↔ ไกลจากร้านอาหารทั่วไป {rho_fd:.3f} (p {p_fd:.3g})")
guard_variance(NJ.nearest_food_km.fillna(999).values, "site-to-food-distance")
guard_variance(NJ.nearest_lanna_km.fillna(999).values, "site-to-lanna-distance")

# ================================================== 7. ภาพ
MED_FOOD_KM = float(np.nanmedian(NJ.nearest_food_km.values))
LAN_KM = NJ.nearest_lanna_km.dropna()
MED_LANNA_KM = float(np.median(LAN_KM.values)) if len(LAN_KM) else None
A1N = A1[A1.name.isin(NORTH)]
A1O = A1[~A1.name.isin(NORTH)]
Hn = H[H.in_north].to_crs(EQ)
FPn = FP[FP.province.isin(NORTH)].to_crs(EQ)


def draw_hero():
    fig = plt.figure(figsize=(15.6, 9.4), dpi=140)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.015, 0.03, 0.62, 0.86]); ax.set_facecolor(BG)
    A1O.to_crs(EQ).plot(ax=ax, facecolor="#131e2c", edgecolor="#25344a", lw=0.4, zorder=1)
    A1N.to_crs(EQ).plot(ax=ax, facecolor="#1b2f3f", edgecolor="#3d6d7c", lw=0.9, zorder=2)
    if len(FPn):
        rf = FPn.geometry.representative_point()
        ax.scatter(rf.x, rf.y, s=1.7, color="#5b7186", alpha=0.55, zorder=3)
    for w in WORDS:
        sub = Hn[Hn.words.map(lambda x: w in x)]
        if not len(sub):
            continue
        rp = sub.geometry.representative_point()
        ax.scatter(rp.x, rp.y, s=34, color=COLOR[w], edgecolor=BG,
                   lw=0.4, alpha=0.95, zorder=5, label=f"{w} · {len(sub)}")
    sp = [SITE_PT[j] for j in NSITES]
    ax.scatter([q.x for q in sp], [q.y for q in sp], s=78, marker="*", color=INK,
               edgecolor=WARM, lw=0.8, zorder=7,
               label=f"แหล่งดูฟ้ามืดที่ราชการรับรอง · {len(NSITES)}")
    b = A1N.to_crs(EQ).total_bounds
    ax.set_xlim(b[0] - 14000, b[2] + 14000); ax.set_ylim(b[1] - 14000, b[3] + 14000)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for x in ax.spines.values(): x.set_color("#33415e")
    ax.set_title(T(f"ร้านที่ \"ชื่อร้าน\" พูดถึงอาหารเมือง — อ่านจาก OpenStreetMap ทั้งประเทศ\n"
                   f"จุดสี = ร้านใน 17 จังหวัดเหนือ ({N_NORTH:,} จาก {N_HIT:,} ร้านทั้งประเทศ) · "
                   f"ดาว = แหล่งดูฟ้ามืดภาคเหนือ {len(NSITES)} แห่ง"), color=INK, fontsize=11.5, pad=10)
    ax.legend(loc="center left", bbox_to_anchor=(0.005, 0.30), facecolor="#111c2b",
              edgecolor="#33415e", labelcolor=INK, fontsize=8.0, ncol=1, framealpha=0.95,
              title=T("คำที่หาใน name"), title_fontproperties={"size": 9.2})
    fig.text(0.035, 0.014, "ข้อมูล © OpenStreetMap contributors (ODbL) · เส้นจังหวัด "
             "Natural Earth admin_1 · ระยะวัดใน Equidistant Conic เดียวกับ Day 22",
             color=MUTED, fontsize=8.2)
    ws_i = WS.set_index("word")
    med_food, lan = MED_FOOD_KM, LAN_KM
    lines = [(T("object อาหารทั้งประเทศ"), f"{FOOD_ALL:,}"),
             (T("  ที่เป็น node (จุด)"), f"{NODE_SHARE}%"),
             (T("ค่า cuisine ที่ใช้จริง"), f"{len(cuis):,} ค่า"),
             (T("  เอ่ยถึงอาหารเมือง"), f"{LANNA_CUIS} ครั้ง"),
             (T("แท็ก diet (เจ/มังสวิรัติ/ฮาลาล)"), f"{len(diet)} ค่า"),
             ("", ""),
             (T("ร้านที่ชื่อมีคำอาหารเมือง"), f"{N_HIT:,}"),
             (T("  ใน 17 จังหวัดเหนือ"), f"{N_NORTH:,} ({100.0*N_NORTH/max(1,N_HIT):.1f}%)"),
             (T("  เฉพาะคำ Lanna แท้"), f"{LANNA_HITS:,}"),
             (T("    ที่อยู่ในเหนือ"), f"{NORTH_LANNA:,}"),
             ("", "")]
    for w in WORDS:
        lines.append((T(w), f"{int(ws_i.loc[w,'hits'])} · เหนือ {int(ws_i.loc[w,'in_north'])} "
                           f"({ws_i.loc[w,'north_share_pct']:.0f}%)"))
    lines += [("", ""),
              (T("อาหารทั่วไป · ไกลระดับมัธยฐาน"), f"{med_food:,.1f} กม."),
              (T("ร้านรสเมือง · ไกลระดับมัธยฐาน"),
               (f"{float(np.median(lan.values)):,.1f} กม." if len(lan) else "ไม่มีใน 40 กม.")),
              (T("จุดที่หาร้านรสเมืองไม่เจอใน 40 กม."), f"{int(NJ.no_lanna_within_40.sum())}/{len(NJ)}"),
              (T("ของย่าง 10 กม. กับ CO (Day 19)"), f"{rho_co:.3f} (p {p_co:.3g})"),
              (T("  ตัวควบคุม: ย่างกับประชากรเมือง"), f"{rho_gp:.3f}"),
              (T("  ประชากรเมืองกับ CO"), f"{rho_pc:.3f}"),
              (T("รสเมือง 40 กม. กับ ways (Day 16)"), f"{rho_way:.3f} (p {p_way:.3g})")]
    txt = fig.add_axes([0.660, 0.035, 0.33, 0.845]); txt.axis("off"); txt.set_facecolor(BG)
    yy = 0.995
    for k, v in lines:
        if not k:
            yy -= 0.012; continue
        txt.text(0, yy, k, color=MUTED, fontsize=8.6, va="top")
        txt.text(1, yy, v, color=ACCENT, fontsize=8.6, va="top", ha="right")
        yy -= 0.0312
    ex = NJ[NJ.nearest_lanna_km.notna()].sort_values("nearest_lanna_km").head(2)
    txt.text(0, max(yy - 0.004, 0.042), T("ใกล้รสเมืองที่สุด 2 จุด (กม.)"), color=INK, fontsize=9.2)
    for i, r in enumerate(ex.itertuples()):
        txt.text(0, max(yy - 0.033 - 0.020 * i, 0.008),
                 f"{str(r.name)[:15]} → {str(r.nearest_lanna_name)[:16]} "
                 f"({r.nearest_lanna_km:,.1f} กม.)", color=MUTED, fontsize=7.6, va="top")
    fig.suptitle(T("รสเมืองอยู่ใน \"ชื่อร้าน\" ไม่ใช่ในหมวดอาหารของแผนที่"), color=INK,
                 fontsize=15.5, y=0.985)
    pth = os.path.join(OUT, "day23_lanna_map.png")
    fig.savefig(pth, facecolor=BG); plt.close(fig)
    log(f"เขียน {pth}")
    return pth


def draw_words():
    fig = plt.figure(figsize=(15.2, 6.8), dpi=140)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.055, 0.13, 0.42, 0.74]); ax.set_facecolor("#111c2b")
    y = np.arange(len(WS))
    ax.barh(y - 0.2, WS.in_north, height=0.38, color=ACCENT, label=T("ใน 17 จังหวัดเหนือ"))
    ax.barh(y + 0.2, WS.hits - WS.in_north, height=0.38, color="#5b7186", label=T("นอกภาคเหนือ"))
    ax.set_yticks(y); ax.set_yticklabels(WS.word, color=INK, fontsize=10)
    for i, r in WS.iterrows():
        ax.text(max(r.hits, 1) * 1.04, i, T(f"เหนือ {r.north_share_pct:.0f}% · "
                                            f"{r.north_provinces} จังหวัด"),
                color=MUTED, fontsize=8.2, va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, max(WS.hits.max() * 1.5, 6))
    ax.set_xlabel(T("จำนวนร้านที่ชื่อมีคำนี้ (ทั้งประเทศ)"), color=INK)
    ax.set_title(T("คำที่ \"เป็นเมืองแท้\" (ข้าวซอย ไส้อั่ว แคบหมู) เทียบกับคำร่วมภาค (ลาบ)\n"
                   "แถบเทาที่ยาวขึ้น = คำนั้นไม่ได้การันตีว่าเป็นเหนือ"), color=INK, fontsize=11)
    ax.legend(facecolor="#111c2b", edgecolor="#33415e", labelcolor=INK, fontsize=8.6,
              loc="lower right")
    ax.grid(color="#26364f", lw=0.5, alpha=0.5, axis="x")
    ax.tick_params(colors=INK, labelsize=9)
    for x in ("top", "right"): ax.spines[x].set_visible(False)

    ax2 = fig.add_axes([0.555, 0.13, 0.415, 0.74]); ax2.set_facecolor("#111c2b")
    PVS = PV.sort_values("lanna_per_1000km2", ascending=True)
    yy = np.arange(len(PVS))
    ax2.barh(yy, PVS.lanna_per_1000km2, color=TEAL, height=0.62)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([f"{r.province_th or r.province}" for r in PVS.itertuples()],
                        color=INK, fontsize=8.8)
    for i, r in enumerate(PVS.itertuples()):
        ax2.text(r.lanna_per_1000km2 + max(PVS.lanna_per_1000km2) * 0.02, i,
                 f"{r.lanna_per_1000km2:.2f} · {r.lanna_named} ร้าน / POI {r.food_nodes:,}",
                 color=MUTED, fontsize=7.5, va="center")
    ax2.set_xlabel(T("ร้านชื่ออาหารเมืองต่อ 1,000 ตร.กม. (เฉพาะ 17 จังหวัดเหนือ)"), color=INK)
    ax2.set_title(T("นับแบบความหนาแน่นแล้วลำดับเปลี่ยน\n"
                    "(จังหวัดใหญ่มีร้านมากกว่า แต่ไม่ได้แน่นกว่า)"), color=INK, fontsize=11)
    ax2.grid(color="#26364f", lw=0.5, alpha=0.5, axis="x")
    ax2.tick_params(colors=INK, labelsize=8.6)
    ax2.set_xlim(0, max(PVS.lanna_per_1000km2.max() * 1.7, 0.2))
    for x in ("top", "right"): ax2.spines[x].set_visible(False)
    fig.suptitle(T("นับคำ: ในเหนือ vs ทั้งประเทศ"), color=INK, fontsize=15, y=0.975)
    pth = os.path.join(OUT, "day23_words.png")
    fig.savefig(pth, facecolor=BG); plt.close(fig)
    log(f"เขียน {pth}")
    return pth


def draw_sites():
    fig = plt.figure(figsize=(15.4, 7.2), dpi=140)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.185, 0.12, 0.395, 0.75]); ax.set_facecolor("#111c2b")
    Q = NJ.sort_values("nearest_food_km", ascending=True)
    y = np.arange(len(Q))
    for i, r in enumerate(Q.itertuples()):
        hi = r.nearest_lanna_km if r.nearest_lanna_km == r.nearest_lanna_km else np.nan
        ax.plot([min(r.nearest_food_km, hi if hi == hi else r.nearest_food_km),
                 max(r.nearest_food_km, hi if hi == hi else r.nearest_food_km)], [i, i],
                color="#41577a", lw=0.9, zorder=2)
    ax.scatter(Q.nearest_food_km, y, s=44, color=COOL, zorder=3,
               label=T("ร้านอาหารที่ OSM บันทึก (ทุกประเภท)"))
    ax.scatter(Q.nearest_lanna_km, y, s=46, color=ACCENT, marker="D", zorder=4,
               label=T("ร้านที่ชื่อโฆษณาอาหารเมือง"))
    ax.set_yticks(y); ax.set_yticklabels([str(n)[:24] for n in Q.name], color=INK, fontsize=7.2)
    ax.set_xlim(0, max(float(np.nanmax(Q.nearest_food_km.max())), float(np.nanmax(Q.nearest_lanna_km.max())) if Q.nearest_lanna_km.notna().any() else 1) * 1.12)
    ax.set_xlabel(T("ระยะจากจุดทะเบียนถึงขอบวัตถุที่ใกล้ที่สุด (กม.)"), color=INK)
    ax.set_title(T(f"แหล่งดูฟ้ามืดภาคเหนือ {len(Q)} แห่ง: ต้องขับไกลแค่ไหนกว่าจะได้รสเมือง\n"
                   f"มัธยฐาน {MED_FOOD_KM:,.1f} กม. (ร้านอาหาร) vs "
                   + (f"{MED_LANNA_KM:,.1f} กม. (รสเมือง)"
                      if Q.nearest_lanna_km.notna().any() else "ไม่พบรสเมืองใน 40 กม.")),
                 color=INK, fontsize=11)
    ax.legend(facecolor="#111c2b", edgecolor="#33415e", labelcolor=INK, fontsize=8.2,
              loc="upper left", framealpha=0.95)
    ax.grid(color="#26364f", lw=0.5, alpha=0.5, axis="x")
    ax.tick_params(colors=INK, labelsize=8.2)
    for x in ("top", "right"): ax.spines[x].set_visible(False)

    ax2 = fig.add_axes([0.635, 0.12, 0.33, 0.75]); ax2.set_facecolor("#111c2b")
    jit = np.random.RandomState(23).uniform(-0.16, 0.16, len(NJ))
    ax2.scatter(NJ.grill_cnt10.values + jit, NJ.CO.values, s=54, color=WARM, edgecolor=BG,
                lw=0.5, zorder=3)
    for r in NJ.nlargest(4, "grill_cnt10").itertuples():
        ax2.annotate(str(r.name)[:13], (r.grill_cnt10, r.CO), textcoords="offset points",
                     xytext=(6, 5), color=MUTED, fontsize=7.2, annotation_clip=False)
    ax2.set_xlabel(T("ร้านลาบ/ไส้อั่ว/แคบหมู ใน 10 กม. (นับจากชื่อร้าน)"), color=INK)
    ax2.set_ylabel(T("CO เฉลี่ย 16–19 พ.ย. (ไมโครกรัม/ลบ.ม.) — Day 19"), color=INK)
    ax2.set_title(T("ของย่างกับควัน: ข้อมูลพูดตรงกันไหม\n"
                    f"Spearman rho = {rho_co:.3f} (p = {p_co:.3g})"), color=INK, fontsize=11)
    ax2.margins(x=0.14, y=0.18)
    ax2.grid(color="#26364f", lw=0.5, alpha=0.5)
    ax2.tick_params(colors=INK, labelsize=8.6)
    for x in ("top", "right"): ax2.spines[x].set_visible(False)
    fig.suptitle(T("จากจุดดูดาวยาวไปหารสเมือง"), color=INK, fontsize=15, y=0.975)
    pth = os.path.join(OUT, "day23_sites.png")
    fig.savefig(pth, facecolor=BG); plt.close(fig)
    log(f"เขียน {pth}")
    return pth

# ================================================== 8. artifact + สถิติ
P1, P2, P3 = draw_hero(), draw_words(), draw_sites()
WS.to_csv(os.path.join(OUT, "day23_words.csv"), index=False)
PV.to_csv(os.path.join(OUT, "day23_provinces.csv"), index=False)
NJ.to_csv(os.path.join(OUT, "day23_sites_north.csv"), index=False)
H[["id", "kind", "province", "in_north", "names", "geom"]].rename(
    columns={"geom": "geometry"}).to_file(os.path.join(OUT, "day23_lanna_places.geojson"),
                                          driver="GeoJSON")
TOP_CUIS = cuis.most_common(12)
AGG = dict(probes=int(len(PROBE)), food_objects=FOOD_ALL, food_nodes=len(food_nodes),
           food_ways=len(food_ways), node_share_pct=NODE_SHARE,
           cuisine_values=len(cuis), cuisine_uses=CUIS_TOTAL,
           lanna_cuisine_uses=LANNA_CUIS, diet_values=len(diet), oh_objects=oh_n,
           dish_hits=N_HIT, dish_in_north=N_NORTH,
           dish_north_share_pct=round(100.0 * N_NORTH / max(1, N_HIT), 2),
           bad_way_geom=bad_way, lanna_only_hits=LANNA_HITS, lanna_only_in_north=NORTH_LANNA,
           north_provinces=len(NORTH), north_area_km2=NORTH_AREA,
           north_sites=len(NSITES),
           median_food_km=round(MED_FOOD_KM, 2),
           median_lanna_km=(round(MED_LANNA_KM, 2) if MED_LANNA_KM is not None else None),
           sites_without_lanna_40km=int(NJ.no_lanna_within_40.sum()),
           max_lanna_km=(round(float(np.max(LAN_KM.values)), 2) if len(LAN_KM) else None),
           rho_grill_co=round(float(rho_co), 3), p_grill_co=float(p_co),
           rho_grill_pm=round(float(rho_pm), 3), p_grill_pm=float(p_pm),
           rho_lanna_ways=round(float(rho_way), 3), p_lanna_ways=float(p_way),
           rho_lanna_smell=round(float(rho_sm), 3), p_lanna_smell=float(p_sm),
           rho_lanna_food=round(float(rho_fd), 3), p_lanna_food=float(p_fd),
           rho_grill_pop=round(float(rho_gp), 3), p_grill_pop=float(p_gp),
           rho_pop_co=round(float(rho_pc), 3), p_pop_co=float(p_pc),
           top_cuisine=TOP_CUIS, top_amenity=amen.most_common(8),
           false_positive_filter=dict(
               word_regex_pairs_before=int(sum(_before.values())),
               word_regex_pairs_after=int(sum(_after.values())),
               places_dropped=int(dropped.sum()), places_left=int(len(H)),
               food_tagged_left=N_FOOD, kept_after_food_filter=int(len(H)),
           dropped_not_food=int(N_FOOD and (len(H_ALL) - len(H)) or 0), laab_before=int(_before.get("ลาบ", 0)),
               laab_after=int(_after.get("ลาบ", 0)),
               dropped_examples=[str(x) for x in FP_NAMES],
               nonfood_examples=[str(x) for x in NONFOOD_NAMES]),
           word_totals={r.word: int(r.hits) for r in WS.itertuples()},
           word_north={r.word: int(r.in_north) for r in WS.itertuples()},
           word_north_share={r.word: float(r.north_share_pct) for r in WS.itertuples()})
STATS = dict(
    theme="Food, drink or something else which you can taste.",
    theme_name="Senses: Taste",
    theme_source="30daymapchallenge.com หัวข้อ Day 23 'Senses: Taste' (อ่านสด 20:55 น. วันที่ 23 พ.ย. 2569)",
    theme_fetch_time=dict(hour=20, minute=55, day=23, month=11, year=2569),
    generated_local=time.strftime("%Y-%m-%d %H:%M:%S"),
    brief="อาจารย์ขอธีม \"อาหารภาคเหนือ\" (ลาบ ข้าวซอย) — จึงวัดว่า OSM บันทึก 'รสเมือง' ไว้ตรงไหน "
          "และแหล่งดูฟ้ามืดภาคเหนืออยู่ไกลจากรสเมืองแค่ไหน",
    question="OpenStreetMap รู้จักรสชาติอาหารเหนือแค่ไหน และแหล่งดูฟ้ามืด 23 แห่งของภาคเหนือ"
             "อยู่ไกลจากร้านที่ OSM รู้จักว่าขายอาหารเมืองกี่กิโลเมตร",
    provenance=dict(source="OpenStreetMap ผ่านไฟล์ Geofabrik Thailand extract (อ่านในเครื่อง)",
                    pbf_md5="db3958a037ce717c462101b86e2f75f3",
                    read_by="tools/collect_day23_food.py (หมวดอาหารทั้งประเทศ) และ "
                            "tools/collect_day23_dishes.py (ค้นคำอาหารเมืองใน name)",
                    provinces="Natural Earth admin_1 (77 จังหวัดไทย, นับพื้นที่ด้วย WGS84 geodesic)",
                    attribution="© OpenStreetMap contributors (ODbL)"),
    method=dict(keyword_source="คำอาหารเมืองตั้งจากเมนูที่รู้จักทั่วไป (11 คำ) + คำควบคุม 2 คำ "
                              "(ลาบ, หน่อไม้) ที่ใช้ทั้งภาค — ใส่ไว้เพื่อทดสอบว่าวิธีนี้ไม่ได้ "
                              "\"เห็นเป็นเหนือไปหมด\"",
                match="NFKC + ตัดช่องว่าง แล้ว substring search ใน name / name:en / name:th / "
                      "alt_name / official_name / sig / sign / brand",
                geometry="way สร้างจาก node ที่หาได้ (node ขาด = ข้าม) แล้ววัดระยะถึงขอบวัตถุ "
                         "ใน Equidistant Conic (method เดียวกับ Day 22)",
                province="representative point + STRtree predicate='within' (shapely 2.x ต้องใช้ "
                         "'within' ไม่ใช่ 'contains')",
                distance_radius="ตามหาวัตถุภายใน 40 กม. รอบจุดทะเบียน — เกินนั้นถือว่าไม่พบ"),
    aggregates=AGG,
    per_word=WS.to_dict("records"),
    per_province=PV.to_dict("records"),
    per_site_north=NJ.drop(columns=["lat", "lon"]).to_dict("records"),
    cuisine_probe=cuisine_probe,
    confounder=dict(
        grill_vs_co=dict(rho=round(float(rho_co), 3), p=float(p_co)),
        grill_vs_urbanpop=dict(rho=round(float(rho_gp), 3), p=float(p_gp)),
        urbanpop_vs_co=dict(rho=round(float(rho_pc), 3), p=float(p_pc)),
        conclusion="ลำดับชั้นของสหสัมพันธ์ชี้ว่าความเป็นเมืองเป็นตัวขับทั้งคู่ "
                   "ไม่ใช่ควันจากเตาย่าง — จึงไม่ควรเขียนว่า 'ลาบทำให้ CO สูง'"),
    cross_check_series=dict(
        day16_source="out/day16/day16_sites_osm.csv (ways/users/ประชากรเมือง 30 กม.)",
        day19_source="out/day19/day19_smell_sites.csv (CO / PM2.5 / smell_index)",
        day22_source="projection เดียวกัน (eqdc 8/20) ที่ Day 22 วัดว่าพื้นที่คลาด −0.3617% และ "
                     "มุมบิดสูงสุด 0.31° ซึ่งเล็กกว่าความคลาดของตัวข้อมูล OSM เองหลายเท่า",
        note="การเทียบเป็น correlation ไม่ใช่สาเหตุ — ร้านชื่อกับค่าฝุ่นมาจากคนละที่มา "
             "(OSM snapshot vs แบบจำลอง CAMS)"),
    bug_found=dict(
        symptom="ชื่อร้านที่ใกล้ที่สุดที่รายงานแต่ละจุด ไม่ตรงกับความไกลที่คำนวณได้ "
                "(เช่น ซอมพอ วิลเลจ ได้ 'สมชัยลาบยโส' ซึ่งอยู่ที่ละติจูด 14.07° ห่างไป ~500 กม.)",
        cause="นำผลลัพธ์ของ argmin บน array ที่กรองด้วย candidate list (ch) ไปใช้ index กับ "
              "GeoFrame ทั้งหมด (Hfm) — ตำแหน่งคนละระบบกัน",
        fix="mapกลับผ่าน ch ก่อน: k = ch[argmin(dh)]",
        caught_by="ตรวจ chéoด้วย script แยก (recompute nearest ของ site 1 = ไส้อั่วป้าตุ๋ย 10.71 กม.) "
                  "หลังตัวเลขในตารางไม่สมเหตุสมผล",
        effect="ระยะทาง (km) ถูกต้องมาตั้งแต่ต้น — ที่ผิดคือ *ชื่อ/จังหวัดของร้านที่ใกล้ที่สุด* "
               "และคอลัมน์คำที่ตรง"),
    figures=["day23_lanna_map.png", "day23_words.png", "day23_sites.png"],
    metric_definitions={
        "hits": "จำนวน object ที่ OSM ระบุว่าเป็นอาหาร (amenity/shop หมวดอาหาร หรือมี cuisine) "
            "และชื่อมีคำนั้น (นับทั้งประเทศ · node + way) หลังกรอง "
                "false positive ของคำว่า ลาบ (กุหลาบ/สะลาบ/แม่สลาบ ไม่ถือ) แล้ว",
        "food_tagged": "ในจำนวนนั้น object ที่ OSM ระบุว่าเป็นอาหารจริง "
                       "(amenity/shop หมวดอาหาร หรือมีแท็ก cuisine)",
        "in_north": "จำนวนที่ตกใน 17 จังหวัดภาคเหนือ (ตาม Natural Earth admin_1)",
        "north_share_pct": "in_north ÷ hits × 100 — ถ้าคำไหน share ต่ำ แปลว่าเป็นคำร่วมภาค",
        "north_per_1000km2": "จำนวนร้านในเหนือ ÷ เนื้อที่ 17 จังหวัด "
                              f"({NORTH_AREA:,.1f} ตร.กม.) × 1,000",
        "lanna_pct_of_food": "ร้านที่ชื่อมีคำ Lanna ÷ object อาหารที่ OSM รู้จักในจังหวัดนั้น × 100",
        "nearest_lanna_km": "ระยะ (กม.) จากจุดทะเบียนถึงขอบวัตถุที่ใกล้ที่สุดที่มีคำอาหารเมืองในชื่อ "
                            "(ตามหาในรัศมี 40 กม. เท่านั้น)",
        "nearest_food_km": "ระยะ (กม.) ถึง object อาหารใกล้สุด (ทุกประเภท) ในรัศมี 40 กม.",
        "grill_cnt10": "จำนวนร้านลาบ/ไส้อั่ว/แคบหมู ภายใน 10 กม.",
        "cuisine_uses": f"จำนวนครั้งที่แท็ก `cuisine` ถูกใช้จริงทั้งประเทศ ({CUIS_TOTAL:,} ครั้ง, "
                        f"{len(cuis):,} ค่า)"},
    limitations=[
        "นี่คือ \"รสที่ OSM บันทึก\" ไม่ใช่ \"รสที่มีอยู่จริง\" — ร้านค้าไทยจำนวนมากไม่ถูก map "
        "และร้านที่ขายลาบ/ข้าวซอยอาจไม่ได้เอาชื่ออาหารมาใส่ใน name",
        "การ match เป็น substring: 'แอ็บ' สั้นจึงอาจชนคำอื่น, 'ลาบ' ชน 'ลาบ' ในความหมายอื่นได้ "
        "(ตรวจชื่อที่ตรงทุกแถวใน day23_lanna_places.geojson + per_word.example_names)",
        "จังหวัดมาจาก Natural Earth admin_1 (1:10 ล้าน) ซึ่งเส้นขอบไม่ตรงเขตกรมที่ดิน 100% "
        "ร้านใกล้ขอบจังหวัดอาจถูกนับผิดจังหวัด (ติดธง edge=True)",
        "CO/PM2.5 จาก Day 19 เป็นค่าเฉลี่ย 96 ชม. ของแบบจำลอง CAMS (grid ~0.1°) ไม่ใช่กลิ่นจริง",
        "17 จังหวัดเหนือใช้เกณฑ์ Wikipedia (9 เหนือบน + 8 เหนือล่าง) — แยก Lanna/ไทยวน/ Tai "
        "ไม่ได้จากข้อมูลนี้",
        "การค้นคำทำ ณ เวลา snapshot เดียวกับ Day 21/22 (extract 21 พ.ย. 2569) — "
        "ร้านอาหารเปิด-ปิดเร็วกว่าที่ OSM อัปเดตมาก"],
    note="ภาพทุกแผ่นวาดจากข้อมูล extract + admin_1 ของจริง ไม่ใช้ tile ภายนอก; "
         "ตัวเลขที่โพสต์อ้างอ่านจาก day23_stats.json + csv 4 ไฟล์เท่านั้น")
def _clean(o):
    """NaN/Inf ออกมาเป็น null (json ที่ allow_nan=False ไม่ยอมให้มี NaN และเบราว์เซอร์อ่านไม่ออก)"""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    return o


json.dump(_clean(STATS), open(os.path.join(OUT, "day23_stats.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, allow_nan=False,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("เขียน day23_stats.json + words/provinces/sites csv + geojson")

# ================================================== 9. ผู้ตัดสิน
for f in ["day23_lanna_map.png", "day23_words.png", "day23_sites.png", "day23_words.csv",
          "day23_provinces.csv", "day23_sites_north.csv", "day23_lanna_places.geojson",
          "day23_stats.json"]:
    assert os.path.getsize(os.path.join(OUT, f)) > 0, f"ไฟล์ว่าง: {f}"
json.load(open(os.path.join(OUT, "day23_stats.json"), encoding="utf-8"))
lint_thai(" ".join(TXT) + " " + __doc__, "day23-texts")
assert len(NORTH) == 17 and len(NORTH_UPPER) == 9 and len(NORTH_LOWER) == 8
assert int(WS.in_north.sum()) >= N_NORTH, "ผลรวมต่อคำ ≥ จำนวนร้าน (ร้านหนึ่งมีได้หลายคำ)"
assert N_NORTH <= N_HIT and (WS.in_north <= WS.hits).all()
assert (WS.north_share_pct <= 100.0).all()
assert len(PV) == 17 and PV.food_nodes.sum() > 0
assert len(NJ) == len(NSITES) == 23, f"จุดภาคเหนือต้อง 23 ได้ {len(NJ)}"
assert NJ.nearest_food_km.notna().all(), "ทุกจุดต้องมีร้านอาหารใน 40 กม. ไม่งั้นรัศมีเล็กไป"
assert abs(sum(v for _, v in TOP_CUIS[:1]) - cuis[TOP_CUIS[0][0]]) < 1
if LANNA_CUIS == 0:
    log("ยืนยัน: OSM ไม่ได้บันทึกรสเมืองไว้เป็นหมวด cuisine เลย (0 ครั้งจาก "
        f"{CUIS_TOTAL:,} ครั้ง)")
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
                          ["food_objects", "cuisine_values", "lanna_cuisine_uses", "dish_hits",
                           "dish_in_north", "dish_north_share_pct", "lanna_only_hits",
                           "median_food_km", "median_lanna_km", "sites_without_lanna_40km",
                           "rho_grill_co", "rho_lanna_ways"]}, ensure_ascii=False))
print("DONE day23 ใน %.1f วิ" % (time.time() - T0))
_LOGF.close()
