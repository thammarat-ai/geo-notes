# -*- coding: utf-8 -*-
"""
collect_day26_export.py — แปลงแคชเครือข่ายน้ำของ Day 26 เป็นไฟล์ใน repo
เพื่อว่า โน้ตบุ๊ก Colab จะรันจบได้โดยไม่ต้องสแกน PBF 330 MB (บทเรียนจาก Day 24)

รัน:  python tools/collect_day26_export.py   (หลัง collect_day26_water.py)
ออก: geo-notes/data/day26_water.npz + day26_areas.json.gz + day26_water_meta.json
"""
import os, json, gzip, time, hashlib
import numpy as np

WORK = r"D:\GIS30Days"
SRC = os.path.join(WORK, "cache", "day26")
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
    meta = json.load(open(os.path.join(SRC, "water_meta.json"), encoding="utf-8"))
    zn = np.load(os.path.join(SRC, "water_nodes.npz"))
    ze = np.load(os.path.join(SRC, "water_edges.npz"))
    areas = json.load(gzip.open(os.path.join(SRC, "water_areas.json.gz"), "rt", encoding="utf-8"))

    lat_i = np.rint(zn["lat"] * 1e7).astype(np.int32)
    lon_i = np.rint(zn["lon"] * 1e7).astype(np.int32)
    assert np.abs(lat_i / 1e7 - zn["lat"]).max() <= 1e-7
    assert np.abs(lon_i / 1e7 - zn["lon"]).max() <= 1e-7
    assert lat_i.min() > 0 and lon_i.max() < 2**31 - 1
    out_npz = os.path.join(DST, "day26_water.npz")
    np.savez_compressed(out_npz,
                        site=zn["site"].astype(np.int16), nid=zn["nid"].astype(np.int64),
                        lat=lat_i, lon=lon_i,
                        esite=ze["site"].astype(np.int16), u=ze["u"].astype(np.int64),
                        v=ze["v"].astype(np.int64), cls=ze["cls"].astype(np.int16))
    out_areas = os.path.join(DST, "day26_areas.json.gz")
    with gzip.open(out_areas, "wt", encoding="utf-8") as f:
        json.dump(areas, f, ensure_ascii=False)

    meta_out = dict(meta)
    meta_out["exported_from"] = "cache/day26 (tools/collect_day26_water.py)"
    meta_out["encoding"] = dict(latlon="int32 = องศา x 1e7 (~1.1 ซม.)",
                                rings="ละติจูด/ลองจิจูด 6 ตำแหน่งใน day26_areas.json.gz")
    meta_out["areas_in_file"] = len(areas)
    meta_out["sha256_npz"] = sha(out_npz)
    meta_out["bytes_npz"] = int(os.path.getsize(out_npz))
    meta_out["bytes_areas"] = int(os.path.getsize(out_areas))
    meta_out["made_utc"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    json.dump(meta_out, open(os.path.join(DST, "day26_water_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    assert meta_out["pbf_md5"] == PBF_MD5_EXPECT, "PBF ต้นทางไม่ตรงกับที่ซีรีส์นี้ใช้"
    assert meta_out["node_slots"] == len(zn["nid"]) and meta_out["edges"] == len(ze["u"])
    print(f"เขียน {out_npz} {meta_out['bytes_npz']/1e6:.2f} MB · "
          f"{out_areas} {meta_out['bytes_areas']/1e6:.2f} MB · แปลงน้ำ {len(areas):,} ชิ้น")


if __name__ == "__main__":
    main()
