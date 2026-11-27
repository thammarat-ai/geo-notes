# -*- coding: utf-8 -*-
"""
check_thai.py — หา (1) อักษรที่ไม่ใช่ไทย/ascii ที่หลุดเข้ามาในไฟล์ (CJK, ฮิรางานะ ฯลฯ)
และ (2) คำไทยที่ไม่เคยปรากฏในคลังอ้างอิง (โพสต์ที่เผยแพร่แล้ว) = ว่าที่คำผิด

ใช้:
  python tools/check_thai.py tools/day25_fingerprint.py
  python tools/check_thai.py --fixmap scratch/day25_fixmap.txt tools/day25_fingerprint.py

คลังอ้างอิง = geo-notes/posts/*/index.qmd + geo-notes/tools/*.py (คนที่ผ่านการตรวจมาแล้ว)
หมายเหตุ: คำที่ถูกต้องแต่เป็นคำใหม่ของเราจริง ๆ จะถูก flag ด้วย — ต้องอ่านเองทุกครั้ง
"""
import re, sys, glob, unicodedata, os

THAI = r"[฀-๿]"
REF = sorted(glob.glob(os.path.join("geo-notes", "posts", "*", "index.qmd")) +
             glob.glob(os.path.join("geo-notes", "tools", "*.py")))


def build_vocab():
    v = set()
    for f in REF:
        try:
            v |= set(re.findall(THAI + "+", open(f, encoding="utf-8").read()))
        except Exception:
            pass
    return v


def stray_chars(text):
    out = {}
    for c in set(text):
        o = ord(c)
        if o < 0x20 or (0x20 <= o <= 0x7e):
            continue
        if 0xe00 <= o <= 0xe7f:
            continue
        out.setdefault(c, 0)
        out[c] += 1
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fix = None
    if "--fixmap" in sys.argv:
        fix = sys.argv[sys.argv.index("--fixmap") + 1]
        args = [a for a in args if a != fix]
    vocab = build_vocab()
    print(f"คลังอ้างอิง {len(REF)} ไฟล์ · คำไทยในคลัง {len(vocab):,} รูป")
    bad_total = 0
    for f in args:
        t = open(f, encoding="utf-8").read()
        ch = stray_chars(t)
        if ch:
            print(f"\n## {f}: อักษรต้องสงสัย")
            for c, n in sorted(ch.items(), key=lambda kv: -kv[1]):
                print(f"   {c!r} U+{ord(c):04X} {unicodedata.name(c, '?')} ×{n}")
        toks = {}
        for m in re.finditer(THAI + "+", t):
            w = m.group(0)
            if w in vocab:
                continue
            toks.setdefault(w, []).append(t[:m.start()].count("\n") + 1)
        if toks:
            print(f"\n## {f}: คำไทยที่ไม่อยู่ในคลัง ({len(toks)} รูป)")
            for w, ls in sorted(toks.items(), key=lambda kv: -len(kv[1])):
                print(f"   {w}  ×{len(ls)}  บรรทัด {ls[:6]}")
        if fix:
            mp = [l.split("\t") for l in open(fix, encoding="utf-8").read().splitlines() if l.strip()]
            for pair in mp:
                if len(pair) >= 2 and pair[0] in t:
                    print(f"   แก้ {pair[0]} → {pair[1]}")
                    t = t.replace(pair[0], pair[1])
            open(f, "w", encoding="utf-8").write(t)
        bad_total += len(ch) + len(toks)
    print(f"\nรวมที่ต้องตรวจ {bad_total} รายการ")


if __name__ == "__main__":
    main()
