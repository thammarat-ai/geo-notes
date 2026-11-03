"""#30DayMapChallenge 2026 — Day 1: Points
83 Dark Sky Thailand locations (NARIT) on a night-sky themed map."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
from matplotlib.font_manager import FontProperties

FP_TH = FontProperties(fname="/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf")
FP_TH_R = FontProperties(fname="/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf")
FP_LA = FontProperties(family="DejaVu Sans", weight="bold")
FP_LA_R = FontProperties(family="DejaVu Sans")

# ---------- data ----------
pts = pd.read_csv("/home/hatch/workspace/darksky_thailand/darksky_thailand.csv")
with open("/home/hatch/workspace/30daymapchallenge/day01_points/thailand_boundary.geojson") as f:
    gj = json.load(f)
geom = gj["features"][0]["geometry"] if gj["type"] == "FeatureCollection" else gj["geometry"]
polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]

REG = [  # key, thai label, count, color
    ("northeast", "ภาคตะวันออกเฉียงเหนือ", "#ffd166"),
    ("north", "ภาคเหนือ", "#5ee7ff"),
    ("center", "ภาคกลาง", "#ff6bd6"),
    ("eastern", "ภาคตะวันออก", "#a3e635"),
    ("south", "ภาคใต้", "#c4b5fd"),
    ("western", "ภาคตะวันตก", "#ff9f43"),
]

# ---------- figure ----------
W, H = 1600, 2000
fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
fig.patch.set_facecolor("#070b18")
ax = fig.add_axes([0.04, 0.14, 0.92, 0.68])
ax.set_facecolor("#070b18")

rng = np.random.default_rng(30)
sx = rng.uniform(96.5, 106.5, 700)
sy = rng.uniform(4.5, 21.5, 700)
ax.scatter(sx, sy, s=rng.uniform(2, 14, 700), c="white",
           alpha=rng.uniform(0.04, 0.25, 700), linewidths=0, zorder=1)

# thailand boundary (GADM, includes islands)
for poly in polys:
    ring = np.asarray(poly[0])
    ax.add_patch(MplPolygon(ring, closed=True, facecolor="#101a3a",
                            edgecolor="#8ea2ff", lw=0.7, alpha=0.9, zorder=2))

for key, th_label, color in REG:
    g = pts[pts["region"] == key]
    x, y = g["longitude"].values, g["latitude"].values
    ax.scatter(x, y, s=260, c=color, alpha=0.16, linewidths=0, zorder=3)
    ax.scatter(x, y, s=46, c=color, edgecolors="white", linewidths=0.6,
               alpha=0.98, zorder=4)

ax.set_xlim(96.8, 106.2)
ax.set_ylim(4.8, 21.3)
ax.set_aspect("equal")
ax.axis("off")

# ---------- header ----------
fig.text(0.5, 0.972, "DAY 1  -  POINTS", ha="center", va="center",
         fontproperties=FP_LA, fontsize=18, color="#8ea2ff")
fig.text(0.5, 0.910, "83", ha="center", va="center",
         fontproperties=FP_LA, fontsize=46, color="white")
fig.text(0.5, 0.870, "เขตอนุรักษ์ท้องฟ้ามืดของไทย", ha="center", va="center",
         fontproperties=FP_TH, fontsize=30, color="white")
fig.text(0.5, 0.842, "จุดดูดาวทั่วประเทศ จำแนกตามภาค",
         ha="center", va="center", fontproperties=FP_TH_R,
         fontsize=14, color="#aeb8e8")

# ---------- hand-rolled legend (2 rows x 3), two-pass for count placement ----------
ly = [0.092, 0.058]
lx = [0.14, 0.47, 0.80]
_thai_txts = []
for i, (key, th_label, color) in enumerate(REG):
    cx, cy = lx[i % 3], ly[i // 3]
    fig.patches.extend([Circle((cx - 0.055, cy), 0.011, color=color,
                               ec="white", lw=0.8,
                               transform=fig.transFigure, figure=fig, zorder=5)])
    t = fig.text(cx - 0.038, cy, th_label, ha="left", va="center",
                 fontproperties=FP_TH_R, fontsize=13.5, color="#cfd6ff")
    _thai_txts.append((t, int((pts["region"] == key).sum())))

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for t, n in _thai_txts:
    bb = t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    fig.text(bb.x1 + 0.012, (bb.y0 + bb.y1) / 2, f"{n}",
             ha="left", va="center", fontproperties=FP_LA_R,
             fontsize=13.5, color="#8b96d8")

fig.text(0.5, 0.022, "Data: darksky.narit.or.th (NARIT)",
         ha="center", va="center", fontproperties=FP_LA_R,
         fontsize=11.5, color="#6f7ab8")
fig.text(0.5, 0.006, "#30DayMapChallenge 2026",
         ha="center", va="center", fontproperties=FP_LA_R,
         fontsize=11.5, color="#6f7ab8")

out = "/home/hatch/workspace/30daymapchallenge/day01_points/day01_points_darksky_thailand.png"
fig.savefig(out, dpi=200, facecolor="#070b18")
print("saved", out)
