# -*- coding: utf-8 -*-
"""
collect_day24_roads.py — สร้าง *กราฟถนน* รอบแหล่งดูฟ้ามืด 83 จุด จากไฟล์ OSM extract

Day 24 · Network ("Route network, transit network, human network or something completely
different.") — Day 21 เคยตอบว่า "ถนนที่ใกล้ที่สุดอยู่ไกลแค่ไหน" วันนี้ถามต่อว่า
**ถนนสายนั้นเป็นเครือข่าย หรือเป็นทางตันที่ขาดออกจากเครือข่าย**
จึงต้องเก็บ graph (node = จุดตัด OSM, edge = ท่อนถนน) ไม่ใช่แค่เรขาคณิต

อ่าน way ที่มีแท็ก `highway` รอบจานรัศมี R ของแต่ละจุดทะเบียน ด้วย `locations=True`
ของ pyosmium (libosmium สร้าง index ของ node ลงไฟล์ชั่วคราวเอง) → สแกนรอบเดียวจบ
เก็บเฉพาะขอบที่ปลายทั้งสองอยู่ในจานของจุดเดียวกัน
(กัน "ทางผ่านเส้นยาวจากจังหวัดข้างเคียง" แล่นเข้ามาเป็นเครือข่ายของจุดนี้)

รัน:  python tools/collect_day24_roads.py
ออก:  cache/day24/{roads_nodes.npz, roads_edges.npz, roads_meta.json}
"""
import os, sys, json, time, math, collections
import numpy as np, pandas as pd
import osmium

WORK = r"D:\GIS30Days"
REPO = os.path.join(WORK, "geo-notes")
CACHE = os.path.join(WORK, "cache", "day24")
PBF = os.path.join(WORK, "cache", "day21", "thailand.osm.pbf")
os.makedirs(CACHE, exist_ok=True)
R = 5000.0          # รัศมีจานต่อจุด (เมตร)
CELL = 0.10         # grid กรองเร็ว (องศา)


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
    """จุดทะเบียนที่พิกัดนี้ตกในจาน (ตรวจ bbox ก่อนแล้วจึงวัดระยะจริง)"""
    out = []
    for j in grid.get((int(math.floor(la / CELL)), int(math.floor(lo / CELL))), ()):
        b = box[j]
        if b[0] <= la <= b[1] and b[2] <= lo <= b[3] and hav_m((la, lo), SITES[j]) <= R:
            out.append(j)
    return out


NODES = {}      # (site, node) -> (lat, lon)
EDGES = []      # (site, u, v, class_index)
CLS = {}        # ค่า highway -> index


class RoadScan(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.ways = 0
        self.high = 0
        self.kept = 0

    def way(self, w):
        self.ways += 1
        cls = w.tags.get("highway")
        if not cls:
            return
        self.high += 1
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
        if not own:
            return
        self.kept += 1
        ci = CLS.setdefault(cls, len(CLS))
        for (n1, la1, lo1), (n2, la2, lo2) in zip(pts, pts[1:]):
            common = own.get(n1, set()) & own.get(n2, set())
            for j in common:
                NODES[(j, n1)] = (la1, lo1)
                NODES[(j, n2)] = (la2, lo2)
                EDGES.append((j, n1, n2, ci))


t0 = time.time()
log("สแกน way ที่มีแท็ก highway (locations=True, index ลงไฟล์ชั่วคราว) ...")
s = RoadScan()
s.apply_file(PBF, locations=True)
log(f"  way ทั้งไฟล์ {s.ways:,} · มี highway {s.high:,} · way ที่แตะจาน {s.kept:,} · "
    f"node-slot {len(NODES):,} · ขอบ {len(EDGES):,} ({time.time()-t0:.0f} วิ)")

keys = list(NODES.keys())
np.savez_compressed(os.path.join(CACHE, "roads_nodes.npz"),
                    site=np.array([k[0] for k in keys], np.int16),
                    nid=np.array([k[1] for k in keys], np.int64),
                    lat=np.array([NODES[k][0] for k in keys], np.float64),
                    lon=np.array([NODES[k][1] for k in keys], np.float64))
E = np.array(EDGES, dtype=np.int64) if EDGES else np.zeros((0, 4), np.int64)
np.savez_compressed(os.path.join(CACHE, "roads_edges.npz"),
                    site=E[:, 0], u=E[:, 1], v=E[:, 2], cls=E[:, 3])
cls_list = [k for k, _ in sorted(CLS.items(), key=lambda kv: kv[1])]
covered = sorted({int(k[0]) for k in keys})
json.dump({"radius_m": R, "ways_in_file": s.ways, "highway_ways": s.high,
           "ways_touching_discs": s.kept, "node_slots": len(NODES), "edges": len(EDGES),
           "classes": cls_list, "sites_covered": len(covered), "site_list": covered,
           "sites_without_roads": [j for j in range(len(SITES)) if j not in covered],
           "pbf_md5": "db3958a037ce717c462101b86e2f75f3",
           "read_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "edge_rule": "เก็บเฉพาะขอบที่ปลายทั้งสองอยู่ในจานของจุดเดียวกัน = ตัดถนนที่ "
                        "วิ่งมาจากนอกจาน (กราฟจึงขาดที่ขอบจานโดยเจตนา)"},
          open(os.path.join(CACHE, "roads_meta.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
log(f"เขียน cache/day24 · จุดที่มีกราฟถนน {len(covered)}/83 · จุดที่ไม่มีเลย "
    f"{len([j for j in range(len(SITES)) if j not in covered])} ({time.time()-t0:.0f} วิ)")
print("ROADS_DONE", len(NODES), len(EDGES))
