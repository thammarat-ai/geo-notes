# -*- coding: utf-8 -*-
"""
collect_day21_pbf.py — อ่าน OSM จาก *แหล่งข้อมูลต้นทางจริง* ไม่ต้องผ่าน API

คืนนี้ Overpass เต็มทุก mirror (502/504 ต่อเนื่อง) จึงเปลี่ยนมาใช้เส้นทางที่คนที่ใช้ OSM
ทั้งประเทศใช้กันจริง: ดาวน์โหลด extract ของประเทศไทยจาก Geofabrik แล้วสแกนในเครื่อง

ผลลัพธ์ = วัตถุที่มีแท็กน่ารู้ (name / tourism / highway / landuse / leisure / building /
amenity / shop / natural / waterway) ที่แตะจานรัศมี R_DISC รอบจุดทะเบียน 83 จุด
พร้อม *แท็กทั้งหมด* (Day 16 เอาแค่ id+user) และ *geometry จริงของ way*
→ วันนี้จึงวัดระยะจากจุดทะเบียนถึง "ขอบวัตถุ" ได้ ไม่ใช่ถึงจุดศูนย์กลางของ way

สาม passes (pbf อ่านตามลำดับ จึงต้องสแกนทั้งไฟล์สามรอบ):
  pass 1 node → พิกัด node ในจาน (ไว้ประกอบ line) + node ที่มีแท็ก
  pass 2 way  → ประกอบ geometry จาก node ที่เก็บไว้ (way ลากออกนอกจาน = จบเป็นเศษ)
  pass 3 rel  → เฉพาะ relation ที่มี name/tourism/boundary และ member ตกในจาน

รัน:  python tools/collect_day21_pbf.py
"""
import os, sys, json, time, math, gzip, hashlib, collections
import numpy as np, pandas as pd
import osmium

WORK = r"D:\GIS30Days"
REPO = os.path.join(WORK, "geo-notes")
CACHE = os.path.join(WORK, "cache", "day21")
PBF = os.path.join(CACHE, "thailand.osm.pbf")
R_DISC = 2000.0          # รัศมีจานสแกน (เมตร) — ใหญ่พอจะเห็นถนนสายใกล้ที่สุด
CELL = 0.02              # องศา ~2.2 กม. grid กรอง node ที่ไม่เกี่ยวทิ้งเร็ว
INTEREST = ("name", "tourism", "highway", "landuse", "leisure", "building", "amenity",
            "shop", "natural", "waterway", "historic", "boundary", "place", "man_made",
            "sport", "craft", "office", "aeroway", "railway", "route")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def hav_m(a, b):
    la1, lo1 = math.radians(a[0]), math.radians(a[1])
    la2, lo2 = math.radians(b[0]), math.radians(b[1])
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(d))


D = pd.read_csv(os.path.join(REPO, "data", "darksky_thailand.csv"))
SITES = [(float(r.latitude), float(r.longitude)) for _, r in D.iterrows()]
log(f"จุดทะเบียน {len(SITES)} จุด · รัศมีจาน {R_DISC:,.0f} ม.")

grid = collections.defaultdict(list)     # cell -> list ของ index จุดทะเบียนที่จานแตะ cell
box = []
for j, (la, lo) in enumerate(SITES):
    dla = R_DISC / 111320.0
    dlo = R_DISC / (111320.0 * math.cos(math.radians(la)))
    box.append((la - dla, la + dla, lo - dlo, lo + dlo))
    for i in range(int(math.floor((la - dla) / CELL)), int(math.ceil((la + dla) / CELL)) + 1):
        for k in range(int(math.floor((lo - dlo) / CELL)), int(math.ceil((lo + dlo) / CELL)) + 1):
            grid[(i, k)].append(j)


def site_of(la, lo):
    """จุดทะเบียนที่ใกล้ที่สุดของพิกัดนี้ (ในจานเท่านั้น) — คืน None ถ้าไม่อยู่ในจาน"""
    cand = grid.get((int(math.floor(la / CELL)), int(math.floor(lo / CELL))))
    if not cand:
        return None
    best, bd = None, 1e18
    for j in cand:
        b = box[j]
        if b[0] <= la <= b[1] and b[2] <= lo <= b[3]:
            d = hav_m((la, lo), SITES[j])
            if d <= R_DISC and d < bd:
                best, bd = j, d
    return best


t0 = time.time()


class Nodes(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.loc = {}
        self.tagged = []
        self.total = 0
        self.in_disc = 0

    def node(self, n):
        self.total += 1
        if not n.location.valid():
            return
        la, lo = n.location.lat, n.location.lon
        j = site_of(la, lo)
        if j is None:
            return
        self.in_disc += 1
        self.loc[n.id] = (la, lo)
        if len(n.tags):
            self.tagged.append({"type": "node", "id": n.id, "site": j, "lat": la, "lon": lo,
                                "tags": dict(n.tags)})


class Ways(osmium.SimpleHandler):
    def __init__(self, loc):
        super().__init__()
        self.loc = loc
        self.total = 0
        self.kept = []
        self.coords = {}
        self.site_of_way = {}

    def way(self, w):
        self.total += 1
        tags = dict(w.tags)
        interested = any(k in tags for k in INTEREST)
        pts, best_j, best_d = [], None, 1e18
        for nr in w.nodes:
            p = self.loc.get(nr.ref)
            if p is None:
                continue                     # node นอกจาน = way ลากออกนอกพื้นที่ของเรา
            if best_j is None:
                best_j = p
            pts.append(p)
        if not pts or not interested:
            return
        j = site_of(best_j[0], best_j[1])
        if j is None:
            return
        # way ยาวอาจแตะจานหลายจุด — มอบให้จุดที่ *ใกล้ geometry ที่สุด* (ตรวจทุก vertex)
        dmin = min(hav_m(p, SITES[j]) for p in pts)
        for k in grid.get((int(math.floor(best_j[0] / CELL)), int(math.floor(best_j[1] / CELL))), []):
            d = min(hav_m(p, SITES[k]) for p in pts)
            if d < dmin:
                dmin, j = d, k
        self.site_of_way[w.id] = j
        self.coords[w.id] = pts
        self.kept.append({"type": "way", "id": w.id, "site": j, "tags": tags,
                          "n_nodes": len(pts), "min_dist_m": round(dmin, 1)})


class Rels(osmium.SimpleHandler):
    def __init__(self, way_site):
        super().__init__()
        self.ws = way_site
        self.total = 0
        self.kept = []

    def relation(self, r):
        self.total += 1
        tags = dict(r.tags)
        if not tags:
            return
        if not (tags.get("name") or tags.get("tourism") or tags.get("boundary")
                or tags.get("leisure") or tags.get("landuse")):
            return
        hit = None
        for m in r.members:
            if m.type == "w" and m.ref in self.ws:
                hit = self.ws[m.ref]
                break
        if hit is None:
            return
        self.kept.append({"type": "relation", "id": r.id, "site": hit,
                          "n_members": len(r.members), "tags": tags})


if not os.path.exists(PBF):
    raise SystemExit("ยังไม่มี thailand.osm.pbf ใน cache/day21")
log(f"pbf {os.path.getsize(PBF)/1e6:,.0f} MB · คำนวณ md5 ...")
h = hashlib.md5()
with open(PBF, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
        h.update(chunk)
log(f"md5 = {h.hexdigest()}")

p1 = Nodes()
p1.apply_file(PBF, locations=False)
log(f"pass1 node: ทั้งไฟล์ {p1.total:,} · ในจาน {p1.in_disc:,} · มีแท็ก {len(p1.tagged):,}"
    f" ({time.time()-t0:.0f} วิ)")

p2 = Ways(p1.loc)
p2.apply_file(PBF, locations=False)
log(f"pass2 way: ทั้งไฟล์ {p2.total:,} · เก็บ {len(p2.kept):,} "
    f"({time.time()-t0:.0f} วิ)")

p3 = Rels(p2.site_of_way)
p3.apply_file(PBF, locations=False)
log(f"pass3 relation: ทั้งไฟล์ {p3.total:,} · เก็บ {len(p3.kept):,} "
    f"({time.time()-t0:.0f} วิ)")

out = {"source": "Geofabrik Thailand extract (OSM PBF)",
       "url": "https://download.geofabrik.de/asia/thailand/thailand-latest.osm.pbf",
       "last_modified_header": "Sat, 21 Nov 2026 01:00:00 GMT",
       "md5": h.hexdigest(), "bytes": os.path.getsize(PBF),
       "read_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
       "radius_m": R_DISC, "cell_deg": CELL, "interest_keys": list(INTEREST),
       "counts": {"file_nodes": p1.total, "disc_nodes": p1.in_disc, "tagged_nodes": len(p1.tagged),
                  "file_ways": p2.total, "kept_ways": len(p2.kept),
                  "file_relations": p3.total, "kept_relations": len(p3.kept)},
       "sites_covered": len({e["site"] for e in p1.tagged} | {e["site"] for e in p2.kept} |
                           {e["site"] for e in p3.kept}),
       "attribution": "© OpenStreetMap contributors (ODbL)"}
json.dump(out, open(os.path.join(CACHE, "scan_meta.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
log("เขียน scan_meta.json")

geo = {str(wid): [[round(p[0], 7), round(p[1], 7)] for p in pts]
       for wid, pts in p2.coords.items()}
with gzip.open(os.path.join(CACHE, "way_geom.json.gz"), "wb") as f:
    f.write(json.dumps(geo).encode())
log(f"เขียน way_geom.json.gz ({len(geo):,} ways)")

for nm, lst in (("poi_nodes", p1.tagged), ("ways", p2.kept), ("relations", p3.kept)):
    with gzip.open(os.path.join(CACHE, f"{nm}.json.gz"), "wb") as f:
        f.write(json.dumps(lst, ensure_ascii=False).encode())
    log(f"เขียน {nm}.json.gz ({len(lst):,} ชิ้น)")
log(f"ครอบคลุมจุดทะเบียน {out['sites_covered']}/83")
print("PBF_SCAN_DONE", out["counts"])
