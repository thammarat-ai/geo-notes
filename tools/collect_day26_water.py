# -*- coding: utf-8 -*-
"""
collect_day26_water.py — สร้าง *เครือข่ายน้ำ* รอบแหล่งดูฟ้ามืด 83 จุด จาก OSM extract
ชุดเดียวกับ Day 21–25 (md5 db3958a0…) + นับทั้งประเทศเพื่อเทียบ "น้ำถูก map ดีเท่าถนนไหม"

สิ่งที่เก็บ (3 กอง)
  1) เส้นน้ำ (waterway = river/stream/canal/drain/ditch ...): โหนด + ขอบ ในจาน 5,000 ม.
     ใช้กติกาเดียวกับ Day 24 ทุกประการ = เก็บเฉพาะขอบที่ปลายทั้งสองอยู่ในจานของจุดเดียวกัน
     → เทียบกราฟ "ถนน vs ลำน้ำ" ได้ตรง ๆ (components, dead-end, ความอ้อม ฯลฯ)
  2) แปลงน้ำ (area): way ปิดที่มี natural=water / waterway=riverbank / landuse=reservoir /
     natural=strait|bay|lagoon / waterway=dam|dock|boatyard → เก็บ ring ข้างนอก + แท็ก + พื้นที่
     (geodesic) เพื่อคำนวณ "สัดส่วนพื้นที่น้ำในจาน"
  3) ตัวนับทั้งประเทศ: จำนวน way ที่มีแท็ก highway เทียบ waterway และแยกตามคลาส
     (นี่คือแกน "ข้อมูล" ของโพสต์ ไม่ใช่แกนภูมิศาสตร์)

รัน:  python tools/collect_day26_water.py
ออก: cache/day26/{water_nodes.npz, water_edges.npz, water_areas.json.gz, water_meta.json}
"""
import os, sys, json, time, math, gzip, collections
import numpy as np, pandas as pd
import osmium
from pyproj import Geod
from shapely.geometry import Polygon
from shapely.validation import make_valid

WORK = r"D:\GIS30Days"
REPO = os.path.join(WORK, "geo-notes")
CACHE = os.path.join(WORK, "cache", "day26")
PBF = os.path.join(WORK, "cache", "day21", "thailand.osm.pbf")
os.makedirs(CACHE, exist_ok=True)
R = 5000.0          # รัศมีจานต่อจุด (เมตร) — ค่าเดียวกับ Day 24 เพื่อให้เทียบกันได้ตรง ๆ
CELL = 0.10
GEOD = Geod(ellps="WGS84")
AREA_TAGS = {"natural": {"water", "strait", "bay", "lagoon"},
             "waterway": {"riverbank", "dam", "dock", "boatyard", "reservoir"},
             "landuse": {"reservoir", "basin", "aquaculture"},
             "leisure": {"marina", "swimming_pool"}}
LINE_TAGS = {"river", "stream", "canal", "drain", "ditch", "tidal_channel", "channel",
             "riverbank", "moat", "wadi", "pressurised"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def hav_m(a, b):
    la1, lo1 = math.radians(a[0]), math.radians(a[1])
    la2, lo2 = math.radians(b[0]), math.radians(b[1])
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * \
        math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(d))


D = pd.read_csv(os.path.join(REPO, "data", "darksky_thailand.csv"))
SITES = [(float(r.latitude), float(r.longitude)) for _, r in D.iterrows()]
log(f"จุดทะเบียน {len(SITES)} จุด · รัศมี {R:,.0f} ม.")

grid = collections.defaultdict(list)
box = []
for j, (la, lo) in enumerate(SITES):
    dla = R / 111320.0
    dlo = R / (111320.0 * math.cos(math.radians(la)))
    box.append((la - dla, la + dla, lo - dlo, lo + dlo))
    for i in range(int(math.floor((la - dla) / CELL)), int(math.ceil((la + dla) / CELL)) + 1):
        for k in range(int(math.floor((lo - dlo) / CELL)), int(math.ceil((lo + dlo) / CELL)) + 1):
            grid[(i, k)].append(j)


def sites_of(la, lo):
    out = []
    for j in grid.get((int(math.floor(la / CELL)), int(math.floor(lo / CELL))), ()):
        b = box[j]
        if b[0] <= la <= b[1] and b[2] <= lo <= b[3] and hav_m((la, lo), SITES[j]) <= R:
            out.append(j)
    return out


NODES = {}
EDGES = []
CLS = {}
AREAS = []
CNT = collections.Counter()
CLS_ALL = collections.Counter()


def is_area(w):
    for key, vals in AREA_TAGS.items():
        v = w.tags.get(key)
        if v and v in vals:
            return f"{key}={v}"
    if w.tags.get("water"):
        return "water=" + w.tags.get("water")
    return None


class WaterScan(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.ways = 0

    def way(self, w):
        self.ways += 1
        if w.tags.get("highway"):
            CNT["highway_ways"] += 1
        ww = w.tags.get("waterway")
        if ww:
            CNT["waterway_ways"] += 1
            CLS_ALL[ww] += 1
        area_tag = is_area(w)
        if not ww and not area_tag:
            return
        pts = []
        for nr in w.nodes:
            if not nr.location.valid():
                return
            pts.append((nr.ref, nr.location.lat, nr.location.lon))
        if len(pts) < 2:
            return
        own = {}
        for nid, la, lo in pts:
            ss = sites_of(la, lo)
            if ss:
                own[nid] = set(ss)
        touched = bool(own)
        # แปลงน้ำ: นับทั้งประเทศ + เก็บ ring เฉพาะที่แตะจาน
        if area_tag:
            CNT["area_ways"] += 1
            if touched and len(pts) >= 4:
                la = [p[1] for p in pts]
                lo = [p[2] for p in pts]
                try:
                    poly = Polygon(zip(lo, la))
                    if not poly.is_valid:
                        poly = make_valid(poly)
                    a, _p = GEOD.geometry_area_perimeter(poly)
                except Exception:
                    CNT["area_ring_broken"] += 1
                    return
                AREAS.append(dict(way=int(w.id), tag=area_tag, name=w.tags.get("name") or "",
                                  water=w.tags.get("water") or "",
                                  area_km2=round(abs(a) / 1e6, 4),
                                  ring=[[round(x, 6), round(y, 6)] for y, x in zip(la, lo)],
                                  sites=sorted(own[pts[0][0]] if pts[0][0] in own
                                              else set().union(*own.values()) if own else [])))
        if not ww or ww == "riverbank":
            return
        ci = CLS.setdefault(ww, len(CLS))
        if not touched:
            return
        CNT["line_ways_touching"] += 1
        for (n1, la1, lo1), (n2, la2, lo2) in zip(pts, pts[1:]):
            for j in own.get(n1, set()) & own.get(n2, set()):
                NODES[(j, n1)] = (la1, lo1)
                NODES[(j, n2)] = (la2, lo2)
                EDGES.append((j, n1, n2, ci))
        if touched:
            CNT["water_ways_in_discs"] += 1


t0 = time.time()
log("สแกน way ที่มีแท็ก waterway / แปลงน้ำ (locations=True) ...")
s = WaterScan()
s.apply_file(PBF, locations=True)
log(f"  way ทั้งไฟล์ {s.ways:,} · highway {CNT['highway_ways']:,} · waterway {CNT['waterway_ways']:,}"
    f" · แปลงน้ำ {CNT['area_ways']:,} · way ในจานที่แตะ {CNT['line_ways_touching']:,} · "
    f"node-slot {len(NODES):,} · ขอบ {len(EDGES):,} ({time.time()-t0:.0f} วิ)")

keys = list(NODES.keys())
np.savez_compressed(os.path.join(CACHE, "water_nodes.npz"),
                    site=np.array([k[0] for k in keys], np.int16),
                    nid=np.array([k[1] for k in keys], np.int64),
                    lat=np.array([NODES[k][0] for k in keys], np.float64),
                    lon=np.array([NODES[k][1] for k in keys], np.float64))
E = np.array(EDGES, dtype=np.int64) if EDGES else np.zeros((0, 4), np.int64)
np.savez_compressed(os.path.join(CACHE, "water_edges.npz"),
                    site=E[:, 0], u=E[:, 1], v=E[:, 2], cls=E[:, 3])
json.dump(AREAS, gzip.open(os.path.join(CACHE, "water_areas.json.gz"), "wt", encoding="utf-8"),
          ensure_ascii=False)

covered = sorted({int(k[0]) for k in keys})
area_cov = sorted({j for a in AREAS for j in a["sites"]})
cls_list = [k for k, _ in sorted(CLS.items(), key=lambda kv: kv[1])]
meta = dict(radius_m=R, ways_in_file=s.ways,
            highway_ways=CNT["highway_ways"], waterway_ways=CNT["waterway_ways"],
            area_ways=CNT["area_ways"], waterway_class_all=dict(CLS_ALL),
            line_ways_touching=CNT["line_ways_touching"],
            water_ways_in_discs=CNT["water_ways_in_discs"],
            node_slots=len(NODES), edges=len(EDGES), classes=cls_list,
            sites_covered=len(covered), site_list=covered,
            sites_without_lines=[j for j in range(len(SITES)) if j not in set(covered)],
            sites_with_area=area_cov, n_areas_in_discs=len(AREAS),
            pbf_md5="db3958a037ce717c462101b86e2f75f3",
            read_utc=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            edge_rule="กติกาเดียวกับ Day 24: เก็บเฉพาะขอบที่ปลายทั้งสองอยู่ในจานของจุดเดียวกัน")
json.dump(meta, open(os.path.join(CACHE, "water_meta.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
log(f"เขียน cache/day26 · จุดที่มีเส้นน้ำ {len(covered)}/83 · จุดที่มีแปลงน้ำ {len(area_cov)}/83")
print("WATER_DONE", len(NODES), len(EDGES), len(AREAS))
