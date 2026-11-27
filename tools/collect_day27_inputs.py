# -*- coding: utf-8 -*-
"""
collect_day27_inputs.py — ส่งออก "ห่วงขอบเขตพื้นที่คุ้มครอง" ที่ Day 21 สแกนไว้
เป็นไฟล์ใน repo เพื่อให้ Day 27 (DuckDB) รันบน Colab ได้โดยไม่ต้องมีแคช PBF

สิ่งที่ export: 163 ห่วง (ring) จาก cache/day21/poly_draw.json พร้อม id/ชื่อ/พื้นที่/จุดที่ทับ
รูปแบบ: JSON gzip ที่มี WKT ต่อห่วง (ลองจิจูด, ละติจูด ตามมาตรฐาน WKT/GeoJSON)
        + คอลัมน์ lat_first ที่กลับลำดับไว้สำหรับฟังก์ชันตระกูล spheroid ของ DuckDB
"""
import os, json, gzip, time, hashlib

WORK = r"D:\GIS30Days"
SRC = os.path.join(WORK, "cache", "day21", "poly_draw.json")
DST = os.path.join(WORK, "geo-notes", "data", "day27_rings.json.gz")


def main():
    rings = json.load(open(SRC, encoding="utf-8"))
    out = []
    for r in rings:
        ring = r["ring"]
        if len(ring) < 4:
            continue
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        wkt_lonlat = "POLYGON ((" + ", ".join(f"{x:.5f} {y:.5f}" for x, y in ring) + "))"
        wkt_latlon = "POLYGON ((" + ", ".join(f"{y:.5f} {x:.5f}" for x, y in ring) + "))"
        out.append(dict(kind=r["kind"], id=int(r["id"]), name=str(r.get("name") or ""),
                        area_km2=r.get("area_km2"), sites=[int(s) for s in r.get("sites") or []],
                        n_pts=len(ring), wkt=wkt_lonlat, wkt_latfirst=wkt_latlon))
    with gzip.open(DST, "wt", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    h = hashlib.sha256(open(DST, "rb").read()).hexdigest()
    print(f"เขียน {DST} — {os.path.getsize(DST)/1e6:.2f} MB · {len(out)} ห่วง "
          f"({sum(1 for o in out if o['kind']=='relation')} relation, "
          f"{sum(1 for o in out if o['kind']=='way')} way) · sha256 {h[:16]}…")
    assert len(out) >= 150, "ห่วงขอบเขตต้องครบเกือบ 163 เส้น"
    assert all(o["n_pts"] >= 4 for o in out)


if __name__ == "__main__":
    main()
