# -*- coding: utf-8 -*-
"""
Day 25 · collect — แปลงแคชกราฟถนนของ Day 24 ให้เป็นไฟล์เดียวที่ Colab ดึงไปใช้ได้

ทำไมต้องมีไฟล์นี้: Day 24 เก็บกราฟไว้ใน cache/day24 (workspace เท่านั้น ไม่ลง repo)
คนที่รันโน้ตบุ๊กบน Colab จึงต้องสแกน PBF 330 MB เองก่อน ซึ่งไม่มีใครทำ
วันนี้โจทย์คือ "ยังเรียกว่าแผนที่ไหม" ซึ่ง *ต้องมีกราฟเต็ม* ถึงจะวัดได้
เลย export เป็น npz บีบอัด (วัดจริง 4.45 MB) ลง data/ ใน repo แทน

สิ่งที่เก็บ (ข้อมูลเดิมทุกตัว ไม่ได้สุมป์หรือกรองใหม่):
  nodes: site, nid, lat, lon        (จานถนน 5,000 ม. รอบแต่ละจุดดูฟ้ามืด)
  edges: site, u, v, cls            (เก็บเฉพาะขอบที่ปลายทั้งสองอยู่ในจานเดียวกัน)

การเข้ารหัสพิกัด: lat/lon × 1e7 เป็น int32 = ความละเอียด ~1.1 ซม.
(ทศนิยม 7 ตำแหน่งพอดีกับที่ OSM เก็บ จึงไม่เสียข้อมูลจริง)

รัน:  python tools/collect_day25_export.py
ออก: geo-notes/data/day25_graphs.npz + geo-notes/data/day25_graphs_meta.json
"""
import os, json, time, hashlib
import numpy as np

T0 = time.time()
WORK = r"D:\GIS30Days"
SRC = os.path.join(WORK, "cache", "day24")
DST = os.path.join(WORK, "geo-notes", "data")
PBF_MD5_EXPECT = "db3958a037ce717c462101b86e2f75f3"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    os.makedirs(DST, exist_ok=True)
    meta24 = json.load(open(os.path.join(SRC, "roads_meta.json"), encoding="utf-8"))
    zn = np.load(os.path.join(SRC, "roads_nodes.npz"))
    ze = np.load(os.path.join(SRC, "roads_edges.npz"))

    # --- กันพิกัดหลุดขอบ int32 (21° × 1e7 = 2.1e8 < 2^31) และเช็คว่าไม่มีการสูญเสียที่ยอมรับไม่ได้
    lat_i = np.rint(zn["lat"] * 1e7).astype(np.int64)
    lon_i = np.rint(zn["lon"] * 1e7).astype(np.int64)
    err_lat = np.abs(lat_i / 1e7 - zn["lat"]).max()
    err_lon = np.abs(lon_i / 1e7 - zn["lon"]).max()
    assert err_lat <= 1e-7 + 1e-12 and err_lon <= 1e-7 + 1e-12, "ปัดเศษพิกัดเกิน 1e-7 องศา"
    assert lat_i.max() < 2**31 - 1 and lon_i.max() < 2**31 - 1, "พิกัดล้น int32"
    assert int(zn["lat"].min()) >= 5 and int(zn["lat"].max()) <= 21, "ละติจูดหลุดขอบประเทศไทย"
    assert int(zn["lon"].min()) >= 96 and int(zn["lon"].max()) <= 108, "ลองจิจูดหลุดขอบประเทศไทย"

    site_n = zn["site"].astype(np.int16)
    site_e = ze["site"].astype(np.int16)
    assert site_n.min() >= 0 and max(site_n.max(), site_e.max()) < 83, "ดัชนีจุดเกินทะเบียน 83 แถว"
    assert ze["cls"].max() < len(meta24["classes"]), "รหัสประเภททางเกินรายการ classes"

    out_npz = os.path.join(DST, "day25_graphs.npz")
    np.savez_compressed(out_npz,
                        site=site_n, nid=zn["nid"].astype(np.int64),
                        lat=lat_i.astype(np.int32), lon=lon_i.astype(np.int32),
                        esite=site_e, u=ze["u"].astype(np.int64),
                        v=ze["v"].astype(np.int64), cls=ze["cls"].astype(np.int16))
    size = os.path.getsize(out_npz)

    out_meta = os.path.join(DST, "day25_graphs_meta.json")
    meta = dict(
        source="cache/day24/*.npz สร้างโดย tools/collect_day24_roads.py (Day 24)",
        pbf="Geofabrik thailand-latest.osm.pbf",
        pbf_bytes=int(meta24["pbf_bytes"]) if "pbf_bytes" in meta24 else 330640775,
        pbf_md5=meta24["pbf_md5"],
        pbf_last_modified="2026-11-21 01:00 GMT (อ่านจาก HTTP header วันเก็บ 24 พ.ย. 2569)",
        read_utc=meta24["read_utc"],
        radius_m=float(meta24["radius_m"]),
        edge_rule=meta24["edge_rule"],
        node_slots=int(len(zn["nid"])),
        edges=int(len(ze["u"])),
        sites_covered=int(len(set(site_n.tolist()))),
        sites_without_roads=[int(x) for x in meta24["sites_without_roads"]],
        classes=meta24["classes"],
        ways_in_file=int(meta24["ways_in_file"]),
        highway_ways=int(meta24["highway_ways"]),
        ways_touching_discs=int(meta24["ways_touching_discs"]),
        encoding=dict(latlon="int32 = องศา x 1e7 (~1.1 ซม.)",
                      max_rounding_error_deg=float(max(err_lat, err_lon))),
        sha256=sha(out_npz),
        bytes=size,
        made_utc=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    )
    json.dump(meta, open(out_meta, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    assert meta["pbf_md5"] == PBF_MD5_EXPECT, "md5 ของ PBF ต้นทางไม่ตรงกับที่ Day 21-24 ใช้"
    assert meta["node_slots"] == 413758 and meta["edges"] == 430464, "จำนวนโหนด/ขอบเปลี่ยนจาก Day 24"
    assert meta["sites_covered"] == 82, "จำนวนจุดที่มีกราฟไม่ตรง Day 24"
    print(f"เขียน {out_npz}  {size/1e6:.2f} MB  ({meta['node_slots']:,} node-slots, "
          f"{meta['edges']:,} edges, {meta['sites_covered']} จุด)")
    print(f"เขียน {out_meta}")
    print(f"sha256 {meta['sha256'][:16]}…  ใช้เวลา {time.time()-T0:.1f} วินาที")


if __name__ == "__main__":
    main()
