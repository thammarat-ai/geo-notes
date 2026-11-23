# -*- coding: utf-8 -*-
"""
collect_day23_food.py — สแกน "สิ่งที่ OSM รู้เรื่องกิน-ดื่ม ของทั้งประเทศไทย"

โจทย์ Day 23 = Senses: Taste ("Food, drink or something else which you can taste.")
ของกินคือสิ่งที่ taste วัดได้ผ่าน OSM จริง เพราะ OSM มีแท็กเฉพาะเรื่องนี้:
    amenity = restaurant / cafe / fast_food / food_court / bar / pub / ice_cream / canteen ...
    shop    = supermarket / convenience / bakery / butcher / greengrocer / wine / tea ...
    craft   = brewery / winery / cheese / pastry
    cuisine = *            ← แท็กเดียวที่บอกว่า "รสอะไร" (ค่าควบคุมจาก OSM wiki)
    diet    = vegetarian / vegan / halal / kosher ...

สอง passes (ต้องอ่าน pbf ตามลำดับ):
  A node+way → เก็บ element ที่เข้าหมวดอาหาร (node มีพิกัดเลย, way เก็บ node refs + tags)
               พร้อม counters ระดับประเทศ (cuisine / amenity / shop)
  B node     → พิกัดของ node ที่ way ใน A ต้องการ (ไว้สร้าง geometry จริงของร้านอาหารที่เป็น polygon)

ผลลัพธ์ cache/day23/{food_nodes.json.gz, food_ways.json.gz, food_nodes_needed.npz, national.json}
รัน ~5 นาที (ไฟล์ 330 MB อ่านสองรอบ) — ไม่เรียก API ใด ๆ

หมายเหตุ: ใช้ไฟล์ PBF เดียวกับ Day 21/22 (md5 db3958a037ce717c462101b86e2f75f3)
"""
import os, sys, json, time, gzip, collections
import numpy as np
import osmium

WORK = r"D:\GIS30Days"
CACHE = os.path.join(WORK, "cache", "day23")
PBF = os.path.join(WORK, "cache", "day21", "thailand.osm.pbf")
os.makedirs(CACHE, exist_ok=True)

FOOD_AMENITY = {"restaurant", "cafe", "fast_food", "food_court", "bar", "pub", "biergarten",
                "nightclub", "ice_cream", "bakery", "deli", "butchery", "concession_stand",
                "canteen", "drinking_water", "market", "juice", "food_court"}
FOOD_SHOP = {"supermarket", "convenience", "bakery", "butcher", "deli", "greengrocer",
             "cheese", "pastry", "seafood", "health_food", "organic", "wine", "alcohol",
             "beverage", "delicatessen", "chocolate", "coffee", "tea", "spice", "butchers"}
FOOD_CRAFT = {"brewery", "winery", "cheese", "dairy", "pastry", "butcher", "sausage", "coffee"}
TAGS_KEEP = ("name", "name:en", "name:th", "amenity", "shop", "craft", "cuisine", "diet",
             "opening_hours", "phone", "contact:phone", "website", "addr:street",
             "addr:city", "addr:housenumber", "outdoor_seating", "reservation",
             "internet_access", "toilets", "highchair", "microbrewery", "takeaway",
             "drive_through", "delivery", "fast_food")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def is_food(t):
    return bool(t.get("cuisine")) or t.get("amenity") in FOOD_AMENITY or \
           t.get("shop") in FOOD_SHOP or t.get("craft") in FOOD_CRAFT


def keep(t):
    return {k: t[k] for k in TAGS_KEEP if k in t}


t0 = time.time()
ncuis, namen, shopn, craftn, dietn = (collections.Counter(), collections.Counter(),
                                      collections.Counter(), collections.Counter(),
                                      collections.Counter())
FOOD_NODES, FOOD_WAYS = [], []
NEED = set()


class ScanA(osmium.SimpleHandler):
    def node(self, n):
        t = dict(n.tags)
        if not is_food(t):
            return
        if not n.location.valid():
            return
        FOOD_NODES.append(dict(id=n.id, lat=n.location.lat, lon=n.location.lon, tags=keep(t)))
        bump(t)

    def way(self, w):
        t = dict(w.tags)
        if not is_food(t):
            return
        ids = [nr.ref for nr in w.nodes]
        if not ids:
            return
        NEED.update(ids)
        FOOD_WAYS.append(dict(id=w.id, nodes=ids, tags=keep(t)))
        bump(t)


def bump(t):
    for v in str(t.get("cuisine", "")).split(";"):
        if v.strip():
            ncuis[v.strip().lower()] += 1
    if t.get("amenity") in FOOD_AMENITY:
        namen[t["amenity"]] += 1
    if t.get("shop") in FOOD_SHOP:
        shopn[t["shop"]] += 1
    if t.get("craft") in FOOD_CRAFT:
        craftn[t["craft"]] += 1
    for v in str(t.get("diet", "")).split(";"):
        if v.strip() in ("vegetarian", "vegan", "halal", "kosher"):
            dietn[v.strip()] += 1


log(f"เปิด {os.path.getsize(PBF)/1e6:,.0f} MB (cache/day21/thailand.osm.pbf)")
a = ScanA()
a.apply_file(PBF, locations=False)
log(f"pass A: food nodes {len(FOOD_NODES):,} · food ways {len(FOOD_WAYS):,} · node refs {len(NEED):,} "
    f"({time.time()-t0:.0f} วิ)")

LOC = {}


class ScanB(osmium.SimpleHandler):
    def node(self, n):
        if n.id in NEED and n.location.valid():
            LOC[n.id] = (n.location.lon, n.location.lat)


b = ScanB()
b.apply_file(PBF, locations=False)
log(f"pass B: ได้พิกัด {len(LOC):,}/{len(NEED):,} node ({time.time()-t0:.0f} วิ)")

with gzip.open(os.path.join(CACHE, "food_nodes.json.gz"), "wb") as f:
    f.write(json.dumps(FOOD_NODES, ensure_ascii=False).encode())
with gzip.open(os.path.join(CACHE, "food_ways.json.gz"), "wb") as f:
    f.write(json.dumps(FOOD_WAYS, ensure_ascii=False).encode())
np.savez_compressed(os.path.join(CACHE, "food_node_locs.npz"),
                    id=np.fromiter(LOC.keys(), np.int64, count=len(LOC)),
                    lon=np.array([v[0] for v in LOC.values()], float),
                    lat=np.array([v[1] for v in LOC.values()], float))
json.dump({"cuisine": ncuis.most_common(120), "amenity": namen.most_common(30),
           "shop": shopn.most_common(30), "craft": craftn.most_common(20),
           "diet": dietn.most_common(10),
           "distinct_cuisine_values": len(ncuis),
           "food_nodes": len(FOOD_NODES), "food_ways": len(FOOD_WAYS),
           "pbf_md5": "db3958a037ce717c462101b86e2f75f3",
           "read_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "tag_filter": {"amenity": sorted(FOOD_AMENITY), "shop": sorted(FOOD_SHOP),
                          "craft": sorted(FOOD_CRAFT), "any_cuisine": True}},
          open(os.path.join(CACHE, "national.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
log(f"เขียน cache/day23 · Cuisine ทั้งประเทศ {sum(ncuis.values()):,} ครั้งจาก {len(ncuis)} ค่า "
    f"({time.time()-t0:.0f} วิ)")
print("FOOD_SCAN_DONE", len(FOOD_NODES), len(FOOD_WAYS), len(ncuis))
