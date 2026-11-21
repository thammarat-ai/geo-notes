# -*- coding: utf-8 -*-
"""
collect_day21_polygons.py — ปิดบั๊กของ collect_day21_pbf.py (ต้องมีก่อนสรุปผล)

บั๊กที่เจอหลังรันครั้งแรก: จานสแกน 2 กม. เห็นเฉพาะ node/way ที่ *แตะ* จาน
ขอบอุทยานแห่งชาติที่ใหญ่กว่าจานจึงหลุดทั้งผืน จุดทะเบียนกลางป่าเลยโดนจัดเป็น class 5
"ไม่มีอะไรเลยใน 1,200 ม." ซึ่งเท็จ — อุทยานแห่งชาติเขาใหญ่มี Khao Yai National Park
อยู่ใน OSM ชัด ๆ แค่เส้นขอบอยู่ไกลกว่า 2 กม. จากจุดทะเบียน
(สรุป:รัศมี search เป็นนิยามของคำถาม ไม่ใช่หลักฐานของการไม่มีอยู่)

บั๊กชั้นที่สองที่เจอตอนแก้ชั้นแรก: ตัวกรอง "พื้นที่คุ้มครอง" รอบแรกเอาเฉพาะ
`boundary=protected_area` ซึ่ง *ไม่ครอบคลุม* ประเทศไทย — อุทยานแห่งชาติเขาใหญ่
(rel 2971358) และอุทยานอีกหลายแห่งใช้ `boundary=national_park` แบบ legacy
จึงต้องขยายค่าที่ยอมรับ (protected_area / national_park / nature_reserve /
leisure=park / tourism=national_park / landuse=forest) แล้วสแกนใหม่ทั้งรอบ
บทเรียน: ค่าที่เป็นมาตรฐานสากลไม่ได้เท่ากับมาตรฐานของข้อมูลจริงในพื้นที่

วิธี: สแกนพื้นที่คุ้มครอง (boundary=protected_area / leisure=nature_reserve|park /
tourism=national_park|park / landuse=forest) ทั้งแบบ way และ relation ทั่วประเทศ
ประกอบ polygon จาก node (สาม passes + แคชผลไว้เพื่อไม่ต้องสแกนซ้ำ) แล้วถามว่า
"จุดทะเบียน 83 จุด อยู่ภายในผืนไหนบ้าง" ด้วย STRtree

ผลลัพธ์ cache/day21/:
  prot_ways.json.gz / prot_nodes.npz  = แคชของ pass A/B/C (สร้างครั้งเดียว)
  polygons.json      = รายการ (จุด × ผืน) contains + ระยะถึงขอบผืนใกล้สุด
  poly_draw.json     = ขอบผืนที่เกี่ยวกับจุดทะเบียน (สำหรับวาดในแผนที่)

รัน:  python tools/collect_day21_polygons.py          # สแกนใหม่ถ้ายังไม่มีแคช
     python tools/collect_day21_polygons.py --rescan  # ทิ้งแคชแล้วสแกนใหม่
"""
import os, sys, json, time, gzip
import numpy as np, pandas as pd
import osmium
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely import transform as shp_transform
from pyproj import Transformer

WORK = r"D:\GIS30Days"
REPO = os.path.join(WORK, "geo-notes")
CACHE = os.path.join(WORK, "cache", "day21")
PBF = os.path.join(CACHE, "thailand.osm.pbf")
# ค่าที่ OSM ใช้กับ "พื้นที่คุ้มครอง" ในประเทศไทย — ไม่วัดจาก protected_area ค่าเดียว
# เพราะอุทยานแห่งชาติหลายแห่ง (รวมเขาใหญ่, relation 2971358) ใส่ boundary=national_park
# แบบ legacy และอีกหลายแห่งใช้ leisure=nature_reserve / tourism=national_park / landuse=forest
AREA = {"boundary": {"protected_area", "national_park", "nature_reserve", "forest_reserve"},
        "leisure": {"nature_reserve", "park", "protected_area"},
        "tourism": {"national_park", "park"},
        "landuse": {"forest", "meadow", "village_green"},
        "natural": {"protected_area"}}
KEEP_KEYS = ("boundary", "leisure", "tourism", "landuse", "natural", "protection_object",
             "designation", "operator", "wikidata", "website", "phone", "name", "name:en")
EQ_STR = "+proj=eqdc +lat_0=14 +lon_0=101 +lat_1=8 +lat_2=20 +units=m +no_defs +ellps=WGS84"
CACHE_AB = os.path.join(CACHE, "prot_ways.json.gz")
CACHE_C = os.path.join(CACHE, "prot_nodes.npz")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def is_protected(t):
    return any(t.get(k) in vs for k, vs in AREA.items())


D = pd.read_csv(os.path.join(REPO, "data", "darksky_thailand.csv"))
t0 = time.time()
if "--rescan" in sys.argv:
    for f in (CACHE_AB, CACHE_C):
        if os.path.exists(f):
            os.rename(f, f + ".old")
    log("ทิ้งแคชเดิม (ย้ายเป็น .old) แล้วสแกนใหม่")

# ---------------------------------------------------------------- pass A/B/C (มีแคช)
class RelH(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.n = 0; self.rel = {}; self.need = set()

    def relation(self, r):
        self.n += 1
        t = dict(r.tags)
        if not is_protected(t):
            return
        ws = tuple(m.ref for m in r.members if m.type == "w")
        if not ws:
            return
        self.rel[r.id] = dict(id=r.id, tags=t, ways=ws)
        self.need.update(ws)


class WayH(osmium.SimpleHandler):
    def __init__(self, need):
        super().__init__()
        self.need = need; self.n = 0; self.nodes = {}; self.tags = {}

    def way(self, w):
        self.n += 1
        t = dict(w.tags)
        if w.id not in self.need and not is_protected(t):
            return
        self.nodes[w.id] = tuple(nr.ref for nr in w.nodes)
        self.tags[w.id] = t


class NodeH(osmium.SimpleHandler):
    def __init__(self, need):
        super().__init__()
        self.need = need; self.n = 0; self.loc = {}

    def node(self, n):
        self.n += 1
        if n.id in self.need and n.location.valid():
            self.loc[n.id] = (n.location.lon, n.location.lat)


if os.path.exists(CACHE_AB) and os.path.exists(CACHE_C):
    log("อ่านแคช pass A/B/C ...")
    with gzip.open(CACHE_AB, "rb") as f:
        cd = json.loads(f.read().decode())
    a = RelH(); a.n = cd["n_rels"]
    a.rel = {int(k): dict(id=int(k), tags=v[0], ways=tuple(v[1]))
             for k, v in cd["rel"].items()}
    a.need = {w for v in a.rel.values() for w in v["ways"]}
    b = WayH(a.need); b.n = cd["n_ways"]
    b.nodes = {int(k): tuple(v) for k, v in cd["ways"].items()}
    b.tags = {int(k): v for k, v in cd["wtags"].items()}
    NPZ = np.load(CACHE_C)
    LOC = {int(i): (float(x), float(y)) for i, x, y in zip(NPZ["id"], NPZ["lon"], NPZ["lat"])}
    NEED = set(LOC)
    log(f"  แคช: relation {a.n:,} · ผืนคุ้มครอง {len(a.rel):,} · way {len(b.nodes):,} · "
        f"node {len(LOC):,}")
else:
    log("pass A relation ...")
    a = RelH(); a.apply_file(PBF, locations=False)
    log(f"  relation ทั้งไฟล์ {a.n:,} · พื้นที่คุ้มครอง {len(a.rel):,} · member way "
        f"{len(a.need):,} ({time.time()-t0:.0f} วิ)")
    log("pass B way ...")
    b = WayH(a.need); b.apply_file(PBF, locations=False)
    NEED = set()
    for ids in b.nodes.values():
        NEED.update(ids)
    log(f"  way ทั้งไฟล์ {b.n:,} · เก็บ {len(b.nodes):,} · node ที่ต้องมี {len(NEED):,} "
        f"({time.time()-t0:.0f} วิ)")
    log("pass C node ...")
    c = NodeH(NEED); c.apply_file(PBF, locations=False)
    LOC = c.loc
    log(f"  node ทั้งไฟล์ {c.n:,} · ได้ location {len(LOC):,}/{len(NEED):,} "
        f"({time.time()-t0:.0f} วิ)")
    with gzip.open(CACHE_AB, "wb") as f:
        f.write(json.dumps({"n_rels": a.n, "n_ways": b.n,
                            "rel": {str(k): [v["tags"], list(v["ways"])]
                                    for k, v in a.rel.items()},
                            "ways": {str(k): list(v) for k, v in b.nodes.items()},
                            "wtags": {str(k): v for k, v in b.tags.items()}}).encode())
    ids_arr = np.fromiter(LOC.keys(), dtype=np.int64, count=len(LOC))
    np.savez_compressed(CACHE_C, id=ids_arr,
                        lon=np.array([LOC[i][0] for i in ids_arr], float),
                        lat=np.array([LOC[i][1] for i in ids_arr], float))
    log("  บันทึกแคช pass A/B/C แล้ว")

# ---------------------------------------------------------------- ประกอบ polygon
rings, ids, tags = [], [], []
for wid, nds in b.nodes.items():
    if len(nds) < 4 or any(i not in LOC for i in nds):
        continue
    co = [LOC[i] for i in nds]
    if co[0] != co[-1]:
        co = co + [co[0]]
    try:
        p = Polygon(co)
        if not p.is_valid:
            p = make_valid(p)
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda q: q.area)
        if p.geom_type != "Polygon" or p.is_empty or p.area <= 0:
            continue
    except Exception:
        continue
    rings.append(p); ids.append(wid); tags.append(b.tags[wid])
del LOC
log(f"  way ที่ประกอบเป็นผืนได้ {len(rings):,} ผืน จาก {len(b.nodes):,} way ({time.time()-t0:.0f} วิ)")

idx_of = {wid: i for i, wid in enumerate(ids)}
rel_geo, rel_meta = [], []
for rid, v in a.rel.items():
    gi = [idx_of[w] for w in v["ways"] if w in idx_of]
    if not gi:
        continue
    try:
        u = rings[gi[0]] if len(gi) == 1 else unary_union([rings[i] for i in gi])
    except Exception:
        continue
    if u.is_empty:
        continue
    rel_geo.append(u); rel_meta.append((rid, v["tags"], len(gi)))
log(f"  relation ที่ประกอบเป็นผืนได้ {len(rel_geo):,} ผืน")

trs = Transformer.from_crs("EPSG:4326", EQ_STR, always_xy=True)
PTS = [Point(float(lo), float(la)) for la, lo in zip(D.latitude, D.longitude)]
SP = [Point(*trs.transform(float(D.longitude.iloc[j]), float(D.latitude.iloc[j])))
      for j in range(len(D))]


def to_m(g):
    """แปลง geometry จาก degree เป็น projection เมตรของซีรีส์ (shapely 2.x: fn (N,2)->(N,2))"""
    return shp_transform(g, lambda co: np.column_stack(
        trs.transform(co[:, 0], co[:, 1])))


def area_km2(g):
    """พื้นที่ตร.กม. จาก degree^2 ปรับด้วย cos(lat) ของจุดศูนย์กลางผืน (ประมาณ)"""
    c = g.centroid.y
    return g.area * (111.32 ** 2) * abs(np.cos(np.radians(c)))


tree_w, tree_r = STRtree(rings), STRtree(rel_geo)
recs = []
for j, pt in enumerate(PTS):
    hit_w = set(int(i) for i in tree_w.query(pt, predicate="within"))
    hit_r = set(int(i) for i in tree_r.query(pt, predicate="within"))
    for i in sorted(hit_w):
        recs.append(dict(site=j, kind="way", id=int(ids[i]), name=tags[i].get("name", "") or
                         tags[i].get("name:en", ""),
                         keys={k: v for k, v in tags[i].items() if k in KEEP_KEYS},
                         area_km2=round(area_km2(rings[i]), 3), contains=True, dist_edge_m=0.0))
    for i in sorted(hit_r):
        rid, t, nway = rel_meta[i]
        recs.append(dict(site=j, kind="relation", id=int(rid),
                         name=t.get("name", "") or t.get("name:en", ""),
                         keys={k: v for k, v in t.items() if k in KEEP_KEYS},
                         area_km2=round(area_km2(rel_geo[i]), 3), contains=True,
                         dist_edge_m=0.0, n_member_ways=nway))
    # ขอบผืน protection ที่ใกล้ที่สุด (เมตรจริงใน projection ของซีรีส์)
    cand = [(int(i), "w") for i in tree_w.query(pt.buffer(0.10, quad_segs=6),
                                                 predicate="intersects")]
    cand += [(int(i), "r") for i in tree_r.query(pt.buffer(0.10, quad_segs=6),
                                                 predicate="intersects")]
    bd, bref = None, None
    for i, kind in cand:
        g = rings[i] if kind == "w" else rel_geo[i]
        d = to_m(g).distance(SP[j])
        if bd is None or d < bd:
            bd = d
            bref = (kind, i)
    if bref:
        kind, i = bref
        nm = (tags[i].get("name", "") or tags[i].get("name:en", "")) if kind == "w" \
            else (rel_meta[i][1].get("name", "") or rel_meta[i][1].get("name:en", ""))
        recs.append(dict(site=j, kind="nearest",
                         id=int(ids[i]) if kind == "w" else int(rel_meta[i][0]),
                         name=nm or "", contains=False,
                         dist_edge_m=round(float(bd), 1), keys={}, area_km2=None))

# ---------------------------------------------------------------- ขอบผืนสำหรับวาด
draw, seen = [], set()
for r in recs:
    if r["kind"] == "nearest" or r["contains"]:
        seen.add((r["kind"], r["id"]))
for i, wid in enumerate(ids):
    if ("way", int(wid)) not in seen:
        continue
    g = rings[i]
    tol = 0.0015 if g.area < 1 else (0.006 if g.area < 30 else 0.02)
    gs = g.simplify(tol, preserve_topology=True)
    for q in (gs.geoms if gs.geom_type == "MultiPolygon" else [gs]):
        if q.is_empty or q.area <= 0:
            continue
        draw.append(dict(kind="way", id=int(wid), name=tags[i].get("name", "") or
                         tags[i].get("name:en", ""), area_km2=round(area_km2(g), 3),
                         sites=[r["site"] for r in recs if r["kind"] == "way" and r["id"] == wid],
                         ring=[[round(x, 5), round(y, 5)] for x, y in q.exterior.coords]))
for i, (rid, t, nway) in enumerate(rel_meta):
    if ("relation", int(rid)) not in seen:
        continue
    g = rel_geo[i]
    tol = 0.0015 if g.area < 1 else (0.006 if g.area < 30 else 0.02)
    gs = g.simplify(tol, preserve_topology=True)
    for q in (gs.geoms if gs.geom_type == "MultiPolygon" else [gs]):
        if q.is_empty or q.area <= 0:
            continue
        draw.append(dict(kind="relation", id=int(rid),
                         name=t.get("name", "") or t.get("name:en", ""),
                         area_km2=round(area_km2(g), 3),
                         sites=[r["site"] for r in recs
                                if r["kind"] == "relation" and r["id"] == rid],
                         ring=[[round(x, 5), round(y, 5)] for x, y in q.exterior.coords]))
json.dump(draw, open(os.path.join(CACHE, "poly_draw.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
json.dump({"polygons_way": len(rings), "polygons_relation": len(rel_geo),
           "protected_relations_all": len(a.rel), "file_ways": b.n, "file_relations": a.n,
           "read_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "run_secs": round(time.time() - t0, 1), "records": recs},
          open(os.path.join(CACHE, "polygons.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
n_cont = sum(1 for r in recs if r.get("contains"))
log(f"เขียน polygons.json + poly_draw.json · {len(recs):,} บันทึก · contains {n_cont:,} บันทึก "
    f"ครอบคลุม {len({r['site'] for r in recs if r.get('contains')})}/83 จุด · "
    f"ขอบผืนสำหรับวาด {len(draw):,} วง ({time.time()-t0:.0f} วิ)")
print("POLY_DONE", len(rings), len(rel_geo), n_cont)
