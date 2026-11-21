# -*- coding: utf-8 -*-
"""
guards.py — ชุดตรวจกลางของซีรีส์ #30DayMapChallenge 2026

เหตุผลที่อยู่: assert 14 ตัวใน day05–day09 กัน *บั๊กการคำนวณ* ได้ดี
แต่บั๊กที่ผมพลาดจริงสองรอบเป็นคนละประเภท — (1) อ่านชื่อ/อันดับจากภาพที่เรนเดอร์แทนไฟล์ข้อมูล
(2) asset ที่ commit ไว้แต่ไม่ถูก ลิงก์ เลย 404 บนเว็บ — และบน Day 10 (Prompting only)
which-ever-model เขียนโค้ดให้ จะไม่มีอะไรรับประกันว่า "ตัวเลขในโพสต์" มาจากข้อมูลจริง

ทุกฟังก์ชันในไฟล์นี้จึงออกแบบให้ (ก) ใช้ได้ฟรีทั้ง local/Colab และ (ข) ทดสอบตัวเองได้
ด้วย `python guards.py --selftest` ซึ่ง *ใส่ความผิดพลาดจงใจ* แล้วต้อง检测到 มันถึงมีค่า

การใช้:
    from guards import hav_km, guard_crs, rings, guard_labels, lint_thai, guard_assets, guard_claims
    guard_claims(post_text, facts, whitelist={...})
"""
import os, re, sys, json, unicodedata

# ---------------------------------------------------------------- of errors
class GuardError(AssertionError):
    """ข้อผิดพลาดที่ guard จับได้ — แยกจาก assert ปกติเพื่อรายงานชัดเจน"""

def _fail(name, msg):
    raise GuardError(f"[{name}] {msg}")

# ---------------------------------------------------------------- 1. ระยะทาง
def hav_km(a, b):
    """haversine ที่ถูกสูตร + ตรวจตัวเอง: a,b เป็น (N,2) ของ (lon,lat) -> (N,M) กม.
    บั๊กที่เจอจริง: พิมพ์ (lo2-la1) เป็น (lo1) แล้วได้ 10,000 กม. เงียบ ๆ ไม่มี error"""
    import numpy as np
    a = np.asarray(a, float); b = np.asarray(b, float)
    if not (a.shape[1] == b.shape[1] == 2):
        _fail("hav_km", "input must be (N,2) of (lon,lat)")
    if np.nanmax(np.abs(a[:, 1])) > 90 or np.nanmax(np.abs(a[:, 0])) > 180:
        _fail("hav_km", "พิกัดไม่ใช่ degree (น่าจะเป็นเมตร) — reproject ก่อนใช้ หรือเรียก guard_crs")
    la1, lo1 = np.radians(a[:, 1][:, None]), np.radians(a[:, 0][:, None])
    la2, lo2 = np.radians(b[:, 1][None, :]), np.radians(b[:, 0][None, :])
    d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(d, 0, 1)))

def guard_distances(D, name, max_km=2500.0):
    import numpy as np
    D = np.asarray(D, float)
    if not np.isfinite(D).all():
        _fail("distances", f"{name}: มี NaN/Inf ในเมทริกซ์ระยะทาง")
    if D.min() < 0:
        _fail("distances", f"{name}: ระยะทางติดลบ = สูตรผิด")
    if D.max() > max_km:
        _fail("distances", f"{name}: ระยะสูงสุด {D.max():,.0f} กม. เกิน {max_km:,.0f} "
                           f"ที่เป็นไปได้ในพื้นที่ศึกษา — check haversine หรือหน่วยพิกัด")
    return True

# ---------------------------------------------------------------- 2. หน่วย/CRS
def guard_crs(bounds, name, expect="degrees"):
    """bounds = (xmin,ymin,xmax,ymax) — กันบั๊กที่เอาเมตรไปเทียบกับ degree (เจอจริงใน Day 7)"""
    lo = min(bounds); hi = max(bounds)
    if expect == "degrees":
        if abs(hi) > 180 or abs(lo) > 180:
            _fail("crs", f"{name}: ค่า {hi:,.0f} ไม่ใช่ degree — น่าจะเป็น projection หน่วยเมตร")
    elif expect == "metres":
        if abs(hi) < 1000:
            _fail("crs", f"{name}: ค่า {hi} เล็กเกินกว่าจะเป็นเมตร (น่าจะเป็น degree)")
    return True

# ---------------------------------------------------------------- 3. geometry
def rings(geom):
    """คืน ring ทุกวงของ Polygon/MultiPolygon — กัน .exterior พังกับ MultiPolygon (เจอจริง 2 ครั้ง)"""
    polys = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
    for p in polys:
        if p.geom_type != "Polygon":
            _fail("rings", f"พบ geometry ชนิด {p.geom_type} ที่ไม่รองรับ")
        yield p.exterior

# ---------------------------------------------------------------- 4. label ↔ ข้อมูล
def guard_labels(labeled, xy_of, ref_of, name="labels", tol=1e-6):
    """labeled: list ของ key ที่ถูกวาง label · xy_of: key->(x,y) ที่วาดจริง · ref_of: key->(lat,lon) ในไฟล์
    ป้องกันบั๊ก zip(list, sorted(list)) ที่ทำให้ชื่อไปติดผิดจุด (เจอจริง Day 6)"""
    for k in labeled:
        if k not in xy_of:
            _fail(name, f"ไม่มีพิกัดของ {k} ใน lookup — ห้ามผูก label ด้วย zip กับ list ที่ sort คนละลำดับ")
        if k not in ref_of:
            _fail(name, f"{k} ไม่มีในไฟล์อ้างอิง (ชื่ออาจถูกถอดจากภาพ ไม่ใช่จากข้อมูล)")
    return True

# ---------------------------------------------------------------- 5. ภาษาไทย
COMBINING = {0x0E31, 0x0E33, 0x0E34, 0x0E35, 0x0E36, 0x0E37, 0x0E38, 0x0E39, 0x0E3A,
             0x0E47, 0x0E48, 0x0E49, 0x0E4A, 0x0E4B, 0x0E4C, 0x0E4D, 0x0E4E}
CJK = r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]"
FORBIDDEN = ["เส้นขน hachure", "เส้นชั้นสั่นมือ", "rose เส้นขอบฟ้า", "cNN", "ก.ม."]
def lint_thai(text, name="text", allow_cjk=False, allow=()):
    issues = []
    if not allow_cjk:
        cjk = set(re.findall(CJK, text))
        if cjk:
            issues.append(f"CJK หลุด: {sorted(cjk)}")
    protected = [a for a in allow if a in text]
    masked = text
    for a in protected:
        masked = masked.replace(a, "" * len(a))
    sp = [masked[max(0, i - 8):i + 8] for i, c in enumerate(masked)
          if c == " " and i + 1 < len(masked) and ord(masked[i + 1]) in COMBINING]
    if sp:
        issues.append(f"ช่องว่างตามด้วยสระ/วรรณยุกต์: {sp[:3]}")
    # คำต้องห้าม: ไม่นับที่ปรากฏใน code/inline code เพราะนั่นคือการ "อ้างถึงคำ" ไม่ใช่การใช้คำ
    naked = re.sub(r"```.*?```", " ", text, flags=re.S)
    naked = re.sub("`[^`" + chr(10) + "]+`", " ", naked)
    # บรรทัดที่เป็นภาษาไทยแต่มีคำละตินมีสระเกิน (เช่น tâm, café) = หลุดจากการพิมพ์/ถอดความ
    thai_line = re.compile(r"[฀-๿]")
    dia = []
    for line in text.splitlines():
        if thai_line.search(line):
            dia += [w for w in re.findall(r"[A-Za-zÀ-ſ]+", line)
                    if not w.isascii() and re.search(r"[A-Za-z]", w)]   # ข้ามสัญลักษณ์อย่าง × ÷ °
    if dia:
        issues.append(f"ละตินมีสระเกินในข้อความไทย: {sorted(set(dia))}")
    bad = [t for t in FORBIDDEN if t in naked and not any(t in a for a in protected)]
    if bad:
        issues.append(f"คำต้องห้าม (ศัพท์แต่งเอง/ผิดราชบัณฑิต): {bad}")
    if issues:
        _fail(name, " | ".join(issues))
    return True

def thainum(n):
    """เลขไทยจาก codepoint — ห้ามพิมพ์มือ (เคยทำหล่น ๓ จน maketrans เพี้ยนเงียบ)"""
    return str(n).translate(str.maketrans("0123456789", "".join(chr(0x0E50 + i) for i in range(10))))

# ---------------------------------------------------------------- 6. asset / publish
def guard_assets(post_path, must_exist=True):
    """ทุกไฟล์ในโฟลเดอร์โพสต์ต้อง *ถูก ลิงก์* (Quarto จะไม่ publish ไฟล์ที่ไม่ถูกอ้าง = 404)
    และทุกไฟล์ที่ถูกอ้างต้องมีจริง — ตรวจสองทาง"""
    d = os.path.dirname(post_path)
    s = open(post_path, encoding="utf-8").read()
    refs = set(re.findall(r"!\[\]\(([^)?]+)", s)) | set(re.findall(r"\]\((day\d\d[^)?]+)", s))
    m = re.search(r"^image:\s*(\S+)", s, re.M)
    if m: refs.add(m.group(1))
    missing = sorted(r for r in refs if must_exist and not os.path.exists(os.path.join(d, r)))
    on_disk = {f for f in os.listdir(d) if f != "index.qmd"}
    orphan = sorted(on_disk - refs)
    if missing:
        _fail("assets", f"อ้างถึงไฟล์ที่ไม่มี: {missing}")
    if orphan:
        _fail("assets", f"ไฟล์ในโฟลเดอร์แต่ไม่ถูกลิงก์ (จะ 404 บนเว็บ): {orphan}")
    return sorted(refs)

# ---------------------------------------------------------------- 7. ตัวเลขในโพสต์ ↔ ข้อมูล
NUMRE = re.compile(r"(?<![\w./])[\d][\d,]*\.?\d*%?")
def guard_claims(post_text, facts, whitelist=(), name="claims", percent_tolerance=0.06):
    """facts = dict ของค่าที่ "ถูกต้อง" (จาก stats json) · whitelist = ตัวเลขโครงสร้างที่ยอมรับได้
    คืนรายการตัวเลขในโพสต์ที่หาที่มาใน facts ไม่ได้ — บน Day 10 ที่โมเดลเขียนโค้ดให้
    รายการนี้ต้องว่าง หรือต้องอธิบายได้ทีละตัว"""
    ok_vals = set()
    def add(v):
        try: v = float(v)
        except Exception: return
        import math
        if not math.isfinite(v): return      # NaN/Inf ไม่ใช่ข้อเท็จจริง (json เขียนลงมาได้)
        for x in (v, round(v), int(math.floor(v + 0.5)), int(math.ceil(v - 0.5)),
                  round(v, 1), round(v, 2), round(v, 6),
                  round(v * 100, 1), round(v * 100, 2),          # percent form
                  round(v / 1e6, 2), round(v / 1e6, 1), round(v / 1e3, 1)):   # ล้าน / พัน
            ok_vals.add(x)
    def walk(o):
        if isinstance(o, dict):
            for k, x in o.items():
                walk(x)
                try: add(float(str(k).replace(",", "")))   # คีย์ที่เป็นตัวเลข (เช่น "500000") ก็คือข้อเท็จจริง
                except Exception: pass
        elif isinstance(o, (list, tuple)):
            for x in o: walk(x)
        else: add(o)
    walk(facts)
    for w in whitelist: add(w)
    stripped = re.sub("^---.*?---", " ", post_text, count=1, flags=re.S)  # frontmatter ไม่ใช่ข้ออ้าง
    stripped = re.sub("```.*?```", " ", stripped, flags=re.S)        # โค้ดไม่ถือเป็นข้ออ้าง
    stripped = re.sub("`[^`]*`", " ", stripped)                       # inline code
    unknown = []
    toks = set()
    for m in NUMRE.finditer(stripped):
        prev = stripped[:m.start()]
        if prev.endswith(",") and len(prev) >= 2 and prev[-2].isdigit():
            continue                      # หางของตัวเลขที่มีจุลภาค เช่น 1,738/1,738 -> "738" ปลอม
        toks.add(m.group(0))
    for tok in toks:
        raw = tok.replace(",", "").rstrip("%")
        try: v = float(raw)
        except ValueError: continue
        if v in ok_vals: continue
        is_int = "." not in tok and not tok.endswith("%")
        if is_int:
            if round(v) in ok_vals or v in ok_vals:
                continue
        else:
            if any(abs(v - k) <= max(0.05, abs(k) * percent_tolerance)
                   for k in ok_vals if isinstance(k, (int, float))):
                continue
        unknown.append(tok)
    if unknown:
        _fail(name, f"ตัวเลขที่หาที่มาในข้อมูลไม่ได้: {sorted(unknown, key=lambda x: -len(x))[:14]}")
    return True

# ---------------------------------------------------------------- 8. live vs local
def guard_live(pairs, bust=True):
    """pairs = [(url, local_path)] — เทียบ md5 (ยอมต่างแค่ CRLF เพราะ repo ตั้ง core.autocrlf=true)"""
    import urllib.request, hashlib
    bad = []
    for url, path in pairs:
        try:
            live = urllib.request.urlopen(url + ("?r=1" if bust else ""), timeout=45).read()
        except Exception as e:                       # deploy ยังไม่เสร็จ หรือ 404 ก็ต้องรายงานเป็นชื่อไฟล์
            bad.append(f"{url.rsplit('/', 1)[-1]} (HTTP {getattr(e, 'code', type(e).__name__)})")
            continue
        loc = open(path, "rb").read()
        if hashlib.md5(live).hexdigest() != hashlib.md5(loc).hexdigest() and \
           live.replace(b"\r\n", b"\n") != loc.replace(b"\r\n", b"\n"):
            bad.append(url.rsplit("/", 1)[-1])
    if bad:
        _fail("live", f"เว็บยังเป็นไฟล์เก่า: {bad}")
    return True

# ---------------------------------------------------------------- selftest (mutation)
def guard_bbox(points, lon_min, lon_max, lat_min, lat_max, name="bbox"):
    """พิกัดต้องอยู่ในกรอบพื้นที่ศึกษา — กันบั๊กสลับ (lat,lon) ซึ่งค่าทั้งสองยัง 'ถูก degree' ทุกตัว"""
    import numpy as np
    a = np.asarray(points, float)
    bad_lon = int(((a[:, 0] < lon_min) | (a[:, 0] > lon_max)).sum())
    bad_lat = int(((a[:, 1] < lat_min) | (a[:, 1] > lat_max)).sum())
    if bad_lon or bad_lat:
        _fail(name, f"มี {bad_lon} จุดนอกช่วงลองจิจูด [{lon_min},{lon_max}] และ {bad_lat} จุดนอกช่วง"
                    f"ละติจูด [{lat_min},{lat_max}] — ถ้าสลับกันพอดี = ใส่ (lat,lon) ผิดลำดับ")
    return True

SUPERLATIVE = ["ไกลที่สุด", "ใกล้ที่สุด", "ไกลฟ้ามืดที่สุด", "สูงสุด", "ต่ำสุด", "มากที่สุด", "น้อยที่สุด"]
# วลีที่เป็น "ชื่อนิยาม" ของสถิติ (nearest = min เหนือเซตที่ระบุ) นับว่าติดป้ายแล้ว
DEF_PHRASES = ["จุดที่ใกล้ที่สุด", "จุดทะเบียนที่ใกล้ที่สุด", "แหล่งที่ใกล้ที่สุด", "ที่ใกล้ที่สุด"]
STAT_WORDS  = ["min", "max", "mean", "median", "avg", "เฉลี่ย", "มัธยฐาน", "ถ่วงน้ำหนัก", "จำนวน",
               "สูงสุด", "ต่ำสุด", "น้อยสุด", "มากสุด"]
def guard_stat_labels(text, name="statlabels", ignore_lines=()):
    """ประโยคที่พูดว่า 'ไกล/ใกล้ที่สุด' ต้องบอกว่าใช้สถิติตัวไหน (min/mean/median/max) ภายในบรรทัดเดียวกัน
    กันกรณี Day 9-10 ที่คำว่า 'ใกล้ที่สุด' หมายถึง min ต่อเมือง ในวันหนึ่ง และ mean ต่อจังหวัดในอีกวัน"""
    text = re.sub("^---.*?---", " ", text, count=1, flags=re.S)   # frontmatter ไม่ใช่ body claim
    bad = []
    for i, line in enumerate(text.splitlines(), 1):
        if any(ig in line for ig in ignore_lines):
            continue
        if "ไหม" in line or "หรือไม่" in line:
            continue   # คำถามไม่ได้อ้างการจัดอันดับ จึงไม่ต้องติดป้ายสถิติ
        if not any(w in line for w in SUPERLATIVE):
            continue
        # นับเฉพาะ "การอ้างการจัดอันดับ" = มีตัวเลข/หน่วยวัด หรือเป็นแถวตาราง
        # (นิยามอย่าง "จุดที่ใกล้ที่สุดของแต่ละเมือง" ไม่มีตัวเลข จึงไม่ใช่ข้ออ้างที่ต้องติดป้าย)
        has_number = re.search(r"\d[\d,.]*\s*(กม|%|บล็อก|เซลล|ล้าน|ตร\.|คน)", line) or line.strip().startswith("|")
        if not has_number:
            continue
        low = line.lower()
        if any(w in line for w in DEF_PHRASES):
            continue
        if not any(w in low for w in STAT_WORDS):
            bad.append((i, line.strip()[:78]))
    if bad:
        _fail(name, " || ".join(f"บรรทัด {i}: {t}" for i, t in bad[:8]))
    return True

def guard_variance(values, name="values", min_spread_km=1.0):
    """ค่าที่ควรกระจาย (เช่นระยะทาง) ต้องมี spread จริง — จับบั๊กเงียบที่ทุกค่าออกมาเท่ากัน
    เช่นลืม reproject จาก degree เป็นเมตร แล้วได้ 13 กม. ทุกเซลล์ ซึ่ง 'ดูสมเหตุสมผล' จนไม่มีใครสังเกต"""
    import numpy as np
    v = np.asarray(values, float)
    if not np.isfinite(v).all():
        _fail("variance", f"{name}: มี NaN/Inf")
    spread = float(v.max() - v.min())
    if spread < min_spread_km:
        _fail("variance", f"{name}: ค่ากระจายเพียง {spread:.3f} (< {min_spread_km}) "
                          f"min={v.min():.2f} max={v.max():.2f} — ทุกค่าเท่ากันผิดปกติ "
                          f"สงสัยหน่วย/CRS หรือวัดกับ geometry ที่ว่างเปล่า")
    return True

def guard_area(value_km2, reference_km2, name="area", tol=0.002):
    """ผลรวมพื้นที่ต้องไม่เกินพื้นที่อ้างอิงของพื้นที่ศึกษา
    บั๊กจริงที่ guard ตัวนี้จับได้: sum(hex.area_km2) = 515,024 > เนื้อที่ไทย 512,593 ตร.กม.
    เพราะ hex ที่ขอบประเทศถูก clip แต่ attribute ยังเก็บพื้นที่เต็มของเซลล์ (sum != union)"""
    if value_km2 is None or reference_km2 is None:
        _fail(name, "ไม่ได้ส่งค่าพื้นที่มาให้เทียบ")
    if value_km2 > reference_km2 * (1 + tol):
        _fail(name, f"ผลรวมพื้นที่ {value_km2:,.0f} ตร.กม. > พื้นที่อ้างอิง {reference_km2:,.0f} "
                    f"อยู่ {value_km2-reference_km2:,.0f} ({(value_km2/reference_km2-1)*100:.2f}%) "
                    f"— น่าจะนับพื้นที่ซ้ำ/ใช้ nominal area ของเซลล์ที่ถูก clip")
    return True

def selftest():
    """ใส่ความผิดพลาดจงใจ แล้วต้องจับได้ทุกตัว — assert ที่ไม่เด้งคือของแต่ง"""
    import numpy as np
    checks = []
    OK, BAD = "✓", "✗"
    def must_fail(label, fn):
        try:
            fn(); checks.append((label, "NOT CAUGHT " + BAD))
        except GuardError as e:
            checks.append((label, "caught " + OK + "  (" + str(e)[:70] + ")"))
        except Exception as e:
            checks.append((label, "caught (other) " + type(e).__name__))
    def must_pass(label, fn):
        try:
            fn(); checks.append((label, "passed " + OK))
        except Exception as e:
            checks.append((label, "FALSE ALARM " + BAD + " " + type(e).__name__ + ": " + str(e)[:60]))
    good = np.array([[100.5, 13.7], [101.2, 14.1], [99.9, 8.4]])
    must_pass("สูตรถูกต้อง: ระยะ 3 จุดต้องสมเหตุสมผล", lambda: guard_distances(hav_km(good, good), "ok", 2500))
    def bad_km(a):
        la1, lo1 = np.radians(a[:, 1][:, None]), np.radians(a[:, 0][:, None])
        la2 = np.radians(a[:, 1][None, :])
        d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(lo1 / 2) ** 2
        return 2 * 6371 * np.arcsin(np.sqrt(d))
    must_fail("haversine พิมพ์ผิด -> ระยะ 10,000 กม.", lambda: guard_distances(bad_km(good), "bad", 2500))
    must_fail("ป้อนพิกัดเมตรให้ hav_km", lambda: hav_km([[500000.0, 1400000.0]], [[101.0, 14.0]]))
    must_fail("degree/metre สับสนใน guard_crs", lambda: guard_crs((99.3, 13.2, 102.4, 14.9), "box", expect="metres"))
    must_fail("label ชี้ไป key ที่ไม่มีใน lookup", lambda: guard_labels(["A", "B"], {"A": (0, 0)}, {"A": (1, 1)}))
    must_fail("สลับ lat/lon ตกนอกกรอบพื้นที่ศึกษา", lambda: guard_bbox([[19.9112, 20.2900]], 97.0, 106.5, 5.5, 20.6, "th"))
    must_pass("พิกัดไทย (lon, lat) ต้องผ่าน bbox", lambda: guard_bbox([[99.8265, 19.9119], [100.5, 13.7]], 97.0, 106.5, 5.5, 20.6, "th"))
    must_fail("ระยะทางทุกค่าเท่ากัน (ลืม reproject)", lambda: guard_variance(np.full(1738, 13.0), "noise_km", min_spread_km=3.0))
    must_pass("ระยะทางที่กระจายปกติต้องผ่าน", lambda: guard_variance(np.array([0.4, 5.0, 13.0, 48.2, 120.0]), "noise_km", min_spread_km=3.0))
    must_fail("ผลรวมพื้นที่มากกว่าเนื้อที่ประเทศ", lambda: guard_area(515023.8, 512593.0, "day10"))
    must_pass("ผลรวมพื้นที่ที่สมเหตุสมผลต้องผ่าน", lambda: guard_area(496867.0, 512593.0, "day10"))
    must_fail("CJK หลุดในข้อความไทย", lambda: lint_thai("สิ่งที่แผนที่นี้บอกคือ " + chr(0x653E) + chr(0x5BBD)))
    must_fail("ช่องว่าง+สระ/วรรณยุกต์", lambda: lint_thai("ข้อมูล" + chr(0x0E49) + " " + chr(0x0E39) + "มูล"))
    must_fail("ละตินมีสระเกินหลุดในไทย", lambda: lint_thai("ไม่ตรงกับ t" + chr(0x00E2) + "m ของเซลล์"))
    must_pass("คำต่างประเทศที่ตั้งใจใช้ต้องไม่โดนแบน", lambda: lint_thai("ใช้ H3 hexagon และ buffer ตามปกติ"))
    must_fail("ศัพท์แต่งเอง (cNN)", lambda: lint_thai("เลขกลุ่ม cNN"))
    must_fail("superlative ไม่มีป้ายสถิติ", lambda: guard_stat_labels("| 3 จังหวัดไกลที่สุด | สุราษฎร์ธานี 94.5 กม. |"))
    must_pass("superlative ที่มีป้ายสถิติต้องผ่าน", lambda: guard_stat_labels("จังหวัดที่ mean ไกลที่สุด (catchment) = สุราษฎร์ธานี 94.5"))
    must_pass("คำถามไม่ต้องติดป้ายสถิติ", lambda: guard_stat_labels("ที่ที่มืดที่สุด คือที่ที่เงียบที่สุดไหม"))
    facts = {"a": 0.1521, "b": 0.8968, "c": {"d": 42, "e": 11814}, "lst": [0.3, 148.0, 39.2], "500000": 10}
    must_pass("ตัวเลขที่มาจาก facts ต้องผ่าน", lambda: guard_claims("ครอบคลุม 15.2% ถึง 89.7% · 42 บล็อก · 11,814 ตร.กม. · 0.3 ถึง 148.0 กม. (มัธยฐาน 39.2)", facts))
    must_fail("ตัวเลขในโพสต์ไม่มีที่มา", lambda: guard_claims("ครอบคลุม 57.3% ของประเทศ", facts))
    must_fail("เสียบเลขใกล้เคียง (42 เป็น 43)", lambda: guard_claims("มี 43 บล็อกที่มืด", facts))
    fm = "---" + chr(10) + "title: x" + chr(10) + "date: 2026-11-11" + chr(10) + "categories: [30DayMapChallenge]" + chr(10) + "---" + chr(10)
    must_pass("ตัวเลขใน frontmatter ไม่ใช่ข้ออ้าง", lambda: guard_claims(fm + "15.2%", facts))
    must_fail("ตัวเลขผิดหลัง frontmatter ยังต้องโดนจับ", lambda: guard_claims(fm + "57.3%", facts))
    must_pass("ตัวเลขใน code fence ไม่ต้องตรวจ", lambda: guard_claims("```python x = 57.3```", facts))
    print("=== guards.py selftest (mutation testing) ===")
    n_bad = 0
    for label, res in checks:
        print("  " + label.ljust(46) + res)
        if "NOT CAUGHT" in res or "FALSE ALARM" in res: n_bad += 1
    print(chr(10) + "เคสทั้งหมด " + str(len(checks)) + " · ตรวจไม่พบ/แจ้งผิด: " + str(n_bad))
    return n_bad

if __name__ == "__main__":
    sys.exit(1 if selftest() else 0)
