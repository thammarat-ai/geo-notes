# -*- coding: utf-8 -*-
"""
collect_day23_dishes.py — หา "ชื่อร้านที่โฆษณาอาหารเมือง" ใน OpenStreetMap ทั้งประเทศ

โจทย์ Day 23 = Senses: Taste แต่ OSM ไม่ได้บันทึก "รส" ไว้ตรง ๆ (แท็ก `cuisine` ในไทย
เกือบทั้งหมดคือ `thai` — ดู national.json ของ collect_day23_food.py)
สิ่งที่ OSM บันทึกไว้จริงคือ *ป้ายชื่อร้าน* — ร้านภาคเหนือมักเอาชื่ออาหารมาตั้งเป็นชื่อร้าน
("ข้าวซอยแม่...", "ลาบ...", "ไส้อั่วแม่...") จึงใช้เป็นหลักฐานของรสชาติได้

ค้นคำ (ตัดช่อง + NFKC แล้ว match แบบ substring ใน name / name:en / alt_name / official_name /
sig / brand) แบ่งเป็น 2 กลุ่มตามที่โพสต์จะอ้างอิง:
  เฉพาะถิ่น (Lanna) : ข้าวซอย, ไส้อั่ว, น้ำพริกหนุ่ม, น้ำพริกอ่อง, แคบหมู, ฮังเล,
                     น้ำเงี้ยว, แกงแค, ตำขนุน, แอ็บ, จิ้นส้ม
  ใช้ร่วมกันทั่ว     : ลาบ, หน่อไม้   ← ใช้เป็นชุดควบคุมว่าวิธีนี้ไม่ได้ "เห็นของเหนืออยู่ทุกแห่ง"

สอง passes: A node+way (way เก็บ node refs) · B เก็บพิกัด node ที่ way ต้องการ
ออก: cache/day23/{dish_nodes.json.gz, dish_ways.json.gz, dish_node_locs.npz}

รัน:  python tools/collect_day23_dishes.py
"""
import os, sys, json, time, gzip, collections
import numpy as np
import osmium

WORK = r"D:\GIS30Days"
CACHE = os.path.join(WORK, "cache", "day23")
PBF = os.path.join(WORK, "cache", "day21", "thailand.osm.pbf")
os.makedirs(CACHE, exist_ok=True)

LANNA = ["ข้าวซอย", "ไส้อั่ว", "น้ำพริกหนุ่ม", "น้ำพริกอ่อง", "แคบหมู", "ฮังเล",
         "น้ำเงี้ยว", "แกงแค", "ตำขนุน", "แอ็บ", "จิ้นส้ม"]
SHARED = ["ลาบ", "หน่อไม้"]
WORDS = LANNA + SHARED
NAME_KEYS = ("name", "name:en", "name:th", "alt_name", "official_name", "sig", "sign", "brand")
KEEP_TAGS = ("name", "name:en", "amenity", "shop", "cuisine", "diet", "opening_hours",
             "tourism", "addr:street", "addr:city", "addr:province")


def norm(s):
    return "".join(str(s).split()).replace(" ", "")


def hits(tags):
    # ทางลัด: element ส่วนใหญ่ใน OSM ไม่มี name เลย — ข้ามก่อนไม่ต้องประกอบ string
    if not any(k in tags for k in NAME_KEYS):
        return "", []
    blob = norm(" ".join(str(tags.get(k, "")) for k in NAME_KEYS if k in tags))
    if not blob:
        return "", []
    return blob, [w for w in WORDS if w in blob]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


t0 = time.time()
NODES, WAYS = [], []
NEED = set()
counter = collections.Counter()


class A(osmium.SimpleHandler):
    def node(self, n):
        if "name" not in n.tags and "alt_name" not in n.tags:
            return
        t = dict(n.tags)
        blob, h = hits(t)
        if not h or not n.location.valid():
            return
        NODES.append(dict(id=n.id, lat=n.location.lat, lon=n.location.lon, words=h, blob=blob,
                          tags={k: t[k] for k in KEEP_TAGS if k in t}))
        for w in h:
            counter[w] += 1

    def way(self, w):
        if "name" not in w.tags and "alt_name" not in w.tags:
            return
        t = dict(w.tags)
        blob, h = hits(t)          # คืน (string ที่ค้นหา, list ของคำที่เจอ)
        if not h:
            return
        ids = [nr.ref for nr in w.nodes]
        if not ids:
            return
        NEED.update(ids)
        WAYS.append(dict(id=w.id, nodes=ids, words=h, blob=blob,
                         tags={k: t[k] for k in KEEP_TAGS if k in t}))
        for w2 in h:
            counter[w2] += 1


log("pass A node+way (หาชื่อที่มีคำอาหารเมือง) ...")
a = A(); a.apply_file(PBF, locations=False)
log(f"  node {len(NODES):,} · way {len(WAYS):,} · node refs ที่ต้องมี {len(NEED):,} "
    f"({time.time()-t0:.0f} วิ)")

LOC = {}


class B(osmium.SimpleHandler):
    def node(self, n):
        if n.id in NEED and n.location.valid():
            LOC[n.id] = (n.location.lon, n.location.lat)


log("pass B node location ...")
b = B(); b.apply_file(PBF, locations=False)
log(f"  ได้พิกัด {len(LOC):,}/{len(NEED):,} ({time.time()-t0:.0f} วิ)")

with gzip.open(os.path.join(CACHE, "dish_nodes.json.gz"), "wb") as f:
    f.write(json.dumps(NODES, ensure_ascii=False).encode())
with gzip.open(os.path.join(CACHE, "dish_ways.json.gz"), "wb") as f:
    f.write(json.dumps(WAYS, ensure_ascii=False).encode())
ids = np.fromiter(LOC.keys(), np.int64, count=len(LOC))
np.savez_compressed(os.path.join(CACHE, "dish_node_locs.npz"), id=ids,
                    lon=np.array([LOC[i][0] for i in ids]),
                    lat=np.array([LOC[i][1] for i in ids]))
json.dump({"lanna": LANNA, "shared": SHARED, "hits_by_word": counter.most_common(),
           "nodes": len(NODES), "ways": len(WAYS),
           "pbf_md5": "db3958a037ce717c462101b86e2f75f3",
           "read_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "matched_on": list(NAME_KEYS), "match_rule": "NFKC + ตัดช่อง + substring"},
          open(os.path.join(CACHE, "dish_meta.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
log("คำที่เจอ (ทั้งประเทศ): " + json.dumps(counter.most_common(), ensure_ascii=False))
print("DISH_DONE", len(NODES), len(WAYS))
