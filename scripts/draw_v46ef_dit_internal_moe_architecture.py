#!/usr/bin/env python3
"""Draw a publication-style SVG architecture diagram for V4.6e/f DiT MoE."""

from pathlib import Path
import html
import math


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "V4_6e_f_DiT_internal_FFN_MoE_architecture.svg"

W, H = 2400, 1350

COLORS = {
    "ink": "#17212b",
    "muted": "#5a6773",
    "blue": "#174e9a",
    "blue2": "#e9f2ff",
    "teal": "#087a7a",
    "teal2": "#e5f7f5",
    "amber": "#b26a00",
    "amber2": "#fff5dd",
    "red": "#b73526",
    "red2": "#fff0ee",
    "green": "#237a38",
    "green2": "#ebf8ed",
    "gray": "#edf1f5",
    "line": "#263849",
}


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


class SVG:
    def __init__(self):
        self.parts = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="white", stroke="#222", sw=2, rx=12, dash=None, opacity=1.0):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr} opacity="{opacity}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=None, sw=2, dash=None, marker=True):
        stroke = stroke or COLORS["line"]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{dash_attr}{marker_attr}/>'
        )

    def path(self, d, stroke=None, sw=2, fill="none", dash=None, marker=True):
        stroke = stroke or COLORS["line"]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.add(
            f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"'
            f'{dash_attr}{marker_attr}/>'
        )

    def text(self, x, y, lines, size=24, weight=500, fill=None, anchor="middle", leading=1.25):
        fill = fill or COLORS["ink"]
        if isinstance(lines, str):
            lines = [lines]
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle">'
        )
        dy0 = -0.5 * (len(lines) - 1) * size * leading
        for i, line in enumerate(lines):
            dy = dy0 + i * size * leading
            self.add(f'<tspan x="{x}" dy="{dy if i == 0 else size * leading}">{esc(line)}</tspan>')
        self.add("</text>")

    def label_box(self, x, y, w, h, title, body=None, fill="white", stroke=None, title_color=None, fs=22):
        stroke = stroke or COLORS["line"]
        title_color = title_color or stroke
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=2.2, rx=14)
        if body:
            self.text(x + w / 2, y + 28, title, size=fs, weight=800, fill=title_color)
            self.text(x + w / 2, y + h / 2 + 18, body, size=max(17, fs - 3), weight=500, fill=COLORS["ink"])
        else:
            self.text(x + w / 2, y + h / 2, title, size=fs, weight=700, fill=title_color)


def header(svg, x, y, w, label, color):
    svg.rect(x, y, w, 58, fill=color, stroke=color, sw=0, rx=10)
    svg.text(x + w / 2, y + 29, label, size=24, weight=800, fill="white")


def small_tokens(svg, x, y, n=8, color="#0a7a88"):
    for i in range(n):
        svg.rect(x + i * 26, y, 17, 17, fill=color, stroke="#063c45", sw=1, rx=2)
    svg.text(x + n * 26 + 22, y + 9, "...", size=26, weight=700, fill=COLORS["muted"])


def contour(svg, x, y, scale=1.0, color="#1f5fae"):
    pts = []
    for i in range(34):
        a = 2 * math.pi * i / 34
        r = 1.0 + 0.18 * math.sin(3 * a) + 0.11 * math.cos(5 * a)
        pts.append((x + scale * r * 58 * math.cos(a), y + scale * r * 34 * math.sin(a)))
    d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts) + " Z"
    svg.path(d, stroke=color, sw=3, marker=False)
    for px, py in pts[::2]:
        svg.add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{3.5 * scale:.1f}" fill="{color}"/>')


def draw():
    svg = SVG()
    svg.add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    svg.add(
        """
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="8" refX="11" refY="4" orient="auto">
    <path d="M0,0 L12,4 L0,8 Z" fill="#263849"/>
  </marker>
  <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.13"/>
  </filter>
  <style>
    text { font-family: DejaVu Sans, Arial, Helvetica, sans-serif; letter-spacing: 0; }
    .card { filter: url(#shadow); }
  </style>
</defs>
"""
    )
    svg.rect(0, 0, W, H, fill="#ffffff", stroke="#ffffff", sw=0, rx=0)
    svg.text(W / 2, 48, "DiffusionSnake V4.6e/f: DiT-Internal FFN-MoE Architecture", size=42, weight=900)
    svg.text(
        W / 2,
        92,
        "V4.6e adds DeepSeek-style routed experts inside each DiT FFN; V4.6f keeps the same network and changes the state curriculum/refinement schedule.",
        size=20,
        weight=500,
        fill=COLORS["muted"],
    )

    # Column headers.
    header(svg, 45, 125, 330, "1. Inputs", COLORS["blue"])
    header(svg, 425, 125, 345, "2. Embedding / Context", COLORS["teal"])
    header(svg, 820, 125, 310, "3. DiT Core x 6", COLORS["blue"])
    header(svg, 1180, 125, 555, "4. One DiTBlock V3 with FFN-MoE", COLORS["blue"])
    header(svg, 1785, 125, 300, "5. MoE Final Head", "#a76500")
    header(svg, 2140, 125, 220, "6. Output", COLORS["green"])

    # Inputs.
    x0, y0 = 50, 205
    svg.rect(x0, y0, 320, 650, fill="#f8fbff", stroke=COLORS["blue"], sw=2.5, rx=18)
    svg.label_box(x0 + 22, y0 + 28, 276, 130, ["x_t contour tokens", "(N x 128 x 2)"], fill="white", stroke=COLORS["blue"], title_color=COLORS["blue"], fs=21)
    contour(svg, x0 + 160, y0 + 105, 1.0)
    svg.label_box(x0 + 22, y0 + 184, 276, 120, ["sampled point features", "(N x C x 128)"], fill="white", stroke=COLORS["blue"], title_color=COLORS["blue"], fs=21)
    for i in range(28):
        cx = x0 + 70 + (i % 7) * 31
        cy = y0 + 248 + (i // 7) * 18
        svg.add(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#1f5fae"/>')
    svg.label_box(x0 + 22, y0 + 330, 276, 120, ["CNN/P2 + detail features", "(global + local grids)"], fill="white", stroke=COLORS["blue"], title_color=COLORS["blue"], fs=20)
    for i in range(6):
        svg.rect(x0 + 82 + i * 21, y0 + 392 - i * 6, 45, 48, fill="#ccdcf5", stroke="#5070a0", sw=1.5, rx=3)
    svg.label_box(x0 + 22, y0 + 474, 276, 100, ["time t", "continuous FM time"], fill="white", stroke=COLORS["blue"], title_color=COLORS["blue"], fs=22)
    svg.line(x0 + 80, y0 + 535, x0 + 245, y0 + 535, sw=3, marker=False)
    svg.add(f'<circle cx="{x0 + 177}" cy="{y0 + 535}" r="11" fill="{COLORS["blue"]}"/>')

    # Embedding/context.
    x1 = 425
    svg.rect(x1, y0, 345, 650, fill="#f7fffe", stroke=COLORS["teal"], sw=2.5, rx=18)
    svg.label_box(x1 + 25, y0 + 28, 295, 110, ["time embedding", "Sinusoidal + MLP, dim=256"], fill="white", stroke=COLORS["teal"], title_color=COLORS["teal"], fs=21)
    small_tokens(svg, x1 + 87, y0 + 104, 7, "#0b8791")
    svg.label_box(x1 + 25, y0 + 168, 295, 110, ["point embedding", "x_t + sampled features"], fill="white", stroke=COLORS["teal"], title_color=COLORS["teal"], fs=21)
    small_tokens(svg, x1 + 87, y0 + 244, 8, "#0b8791")
    svg.label_box(x1 + 25, y0 + 308, 295, 135, ["global Perceiver context", "256 query tokens"], fill="white", stroke=COLORS["teal"], title_color=COLORS["teal"], fs=21)
    for i in range(4):
        for j in range(3):
            svg.add(f'<circle cx="{x1 + 95 + i * 42}" cy="{y0 + 385 + j * 22}" r="7" fill="#c6efef" stroke="{COLORS["teal"]}" stroke-width="2"/>')
    for i in range(3):
        svg.line(x1 + 95 + i * 42, y0 + 385, x1 + 95 + (i + 1) * 42, y0 + 429, stroke=COLORS["teal"], sw=1.4, marker=False)
    svg.label_box(x1 + 25, y0 + 480, 295, 120, ["local/detail point context", "per-point, dim=256"], fill="white", stroke=COLORS["teal"], title_color=COLORS["teal"], fs=21)
    for i in range(14):
        a = 2 * math.pi * i / 14
        svg.add(f'<circle cx="{x1 + 172 + 58 * math.cos(a):.1f}" cy="{y0 + 545 + 32 * math.sin(a):.1f}" r="5" fill="{COLORS["teal"]}"/>')
    svg.rect(x1 + 140, y0 + 520, 72, 50, fill="none", stroke=COLORS["teal"], sw=2, rx=6, dash="7 5")

    # Input to context arrows.
    for yy in [y0 + 90, y0 + 244, y0 + 390, y0 + 535]:
        svg.line(x0 + 320, yy, x1 - 15, yy, sw=2.8)

    # DiT stack.
    x2 = 820
    svg.rect(x2, y0, 310, 650, fill="#f8fbff", stroke=COLORS["blue"], sw=2.5, rx=18)
    svg.text(x2 + 155, y0 + 35, ["DiTBlock V3 x 6", "dim=256, heads=8"], size=24, weight=800, fill=COLORS["blue"])
    block_h = 75
    bys = []
    for i in range(6):
        by = y0 + 78 + i * 88
        bys.append(by)
        fill = "#f0f6ff" if i % 2 == 0 else "#eefbf9"
        stroke = COLORS["blue"] if i % 2 == 0 else COLORS["teal"]
        svg.rect(x2 + 45, by, 220, block_h, fill=fill, stroke=stroke, sw=2, rx=12)
        svg.rect(x2 + 60, by + 20, 36, 36, fill=COLORS["blue"], stroke=COLORS["blue"], sw=0, rx=6)
        svg.text(x2 + 78, by + 38, str(i + 1), size=24, weight=900, fill="white")
        ctx = "global context" if i % 2 == 0 else "local/detail context"
        svg.text(x2 + 172, by + 28, "DiTBlock V3", size=22, weight=800)
        svg.text(x2 + 172, by + 55, f"({ctx})", size=17, weight=600, fill=stroke)
    svg.text(
        x2 + 155,
        y0 + 620,
        ["Odd blocks 1/3/5: global context", "Even blocks 2/4/6: local/detail context"],
        size=18,
        weight=600,
        fill=COLORS["muted"],
    )
    for yy in [y0 + 83, y0 + 223, y0 + 390, y0 + 544]:
        svg.line(x1 + 345, yy, x2 - 15, yy, sw=2.8)

    # Enlarged DiT block inset.
    x3 = 1180
    svg.rect(x3, y0, 555, 650, fill="#fbfdff", stroke=COLORS["blue"], sw=2.5, rx=18)
    svg.text(x3 + 278, y0 + 38, "Inside one DiTBlock V3", size=25, weight=900, fill=COLORS["blue"])
    rows = [
        (y0 + 78, "adaLN-Zero from time embedding", COLORS["blue"], COLORS["blue2"]),
        (y0 + 142, "Self-Attn + QK-Norm + Cyclic RoPE", COLORS["blue"], COLORS["blue2"]),
        (y0 + 206, "Cross-Attn to global or local/detail context", COLORS["teal"], COLORS["teal2"]),
    ]
    for yy, label, stroke, fill in rows:
        svg.rect(x3 + 130, yy, 295, 44, fill=fill, stroke=stroke, sw=2, rx=9)
        svg.text(x3 + 278, yy + 22, label, size=18, weight=750, fill=COLORS["ink"])
    svg.line(x3 + 278, y0 + 122, x3 + 278, y0 + 142, sw=2.4)
    svg.line(x3 + 278, y0 + 186, x3 + 278, y0 + 206, sw=2.4)
    svg.line(x3 + 278, y0 + 250, x3 + 278, y0 + 278, sw=2.4)
    svg.rect(x3 + 45, y0 + 278, 465, 295, fill="#ffffff", stroke=COLORS["blue"], sw=2.2, rx=14)
    svg.text(x3 + 278, y0 + 306, "FFN branch", size=22, weight=900, fill=COLORS["blue"])
    svg.rect(x3 + 70, y0 + 343, 120, 160, fill="#f4f7fb", stroke="#74859a", sw=2, rx=10)
    svg.text(x3 + 130, y0 + 423, ["dense SwiGLU", "FFN", "= shared path"], size=18, weight=750)
    svg.rect(x3 + 215, y0 + 328, 270, 202, fill=COLORS["red2"], stroke=COLORS["red"], sw=2.2, rx=12)
    svg.text(x3 + 350, y0 + 354, "Routed FFN-MoE: 4 experts, top-2", size=18, weight=900, fill=COLORS["red"])
    svg.rect(x3 + 248, y0 + 378, 204, 52, fill=COLORS["amber2"], stroke=COLORS["amber"], sw=1.8, rx=8)
    svg.text(x3 + 350, y0 + 404, ["Router", "RMSNorm + time + point + cyclic conv"], size=15, weight=700, fill=COLORS["ink"])
    svg.rect(x3 + 270, y0 + 448, 160, 42, fill=COLORS["amber2"], stroke=COLORS["amber"], sw=1.8, rx=8)
    svg.text(x3 + 350, y0 + 469, "softmax / top-k (k=2)", size=16, weight=700)
    for i in range(4):
        ex = x3 + 236 + i * 61
        svg.rect(ex, y0 + 506, 52, 52, fill="#fff7f6", stroke=COLORS["red"], sw=1.8, rx=7)
        svg.text(ex + 26, y0 + 532, [f"E{i+1}", "SwiGLU"], size=13, weight=750, fill=COLORS["red"])
    svg.rect(x3 + 510, y0 + 404, 70, 55, fill="#fffaf0", stroke=COLORS["amber"], sw=1.7, rx=8, dash="6 4")
    svg.text(x3 + 545, y0 + 431, ["aux", "balance", "loss"], size=13, weight=750, fill=COLORS["amber"])
    svg.line(x3 + 350, y0 + 430, x3 + 350, y0 + 448, stroke=COLORS["amber"], sw=2)
    svg.line(x3 + 350, y0 + 490, x3 + 350, y0 + 506, stroke=COLORS["amber"], sw=2)
    svg.path(f"M {x3 + 452} {y0 + 405} C {x3 + 495} {y0 + 385}, {x3 + 520} {y0 + 398}, {x3 + 510} {y0 + 426}", stroke=COLORS["amber"], sw=2, dash="6 4")
    svg.add(f'<circle cx="{x3 + 278}" cy="{y0 + 606}" r="18" fill="white" stroke="{COLORS["line"]}" stroke-width="2"/>')
    svg.text(x3 + 278, y0 + 606, "+", size=26, weight=900)
    svg.line(x3 + 130, y0 + 503, x3 + 250, y0 + 590, sw=2.2)
    svg.line(x3 + 350, y0 + 558, x3 + 292, y0 + 590, sw=2.2)
    svg.rect(x3 + 110, y0 + 616, 335, 48, fill="#f3f7ff", stroke=COLORS["blue"], sw=2, rx=9)
    svg.text(x3 + 278, y0 + 640, "FFN_out = dense FFN + routed experts", size=20, weight=900, fill=COLORS["blue"])
    svg.line(x3 + 278, y0 + 624, x3 + 278, y0 + 616, sw=2.2, marker=False)
    svg.path(f"M {x2 + 265} {bys[0] + 35} C {x2 + 430} {bys[0] + 35}, {x3 - 40} {y0 + 35}, {x3} {y0 + 35}", stroke=COLORS["blue"], sw=2, dash="8 6")

    # Final head.
    x4 = 1785
    svg.rect(x4, y0, 300, 650, fill="#fffdf8", stroke=COLORS["amber"], sw=2.5, rx=18)
    svg.text(x4 + 150, y0 + 36, ["MoE final head", "8 MLP experts, top-2"], size=22, weight=900, fill=COLORS["red"])
    svg.rect(x4 + 55, y0 + 88, 190, 72, fill=COLORS["amber2"], stroke=COLORS["amber"], sw=2, rx=10)
    svg.text(x4 + 150, y0 + 124, ["Router", "time + point + cyclic"], size=18, weight=750)
    svg.rect(x4 + 72, y0 + 188, 156, 48, fill=COLORS["amber2"], stroke=COLORS["amber"], sw=2, rx=8)
    svg.text(x4 + 150, y0 + 212, "softmax / top-k", size=18, weight=750)
    for i, (dx, dy) in enumerate([(45, 275), (170, 275), (45, 365), (170, 365)]):
        svg.rect(x4 + dx, y0 + dy, 86, 58, fill=COLORS["red2"], stroke=COLORS["red"], sw=2, rx=8)
        label = f"E{i+1}" if i < 2 else ("..." if i == 2 else "E8")
        svg.text(x4 + dx + 43, y0 + dy + 29, [label, "MLP"], size=16, weight=800, fill=COLORS["red"])
    svg.add(f'<circle cx="{x4 + 150}" cy="{y0 + 485}" r="21" fill="white" stroke="{COLORS["line"]}" stroke-width="2"/>')
    svg.text(x4 + 150, y0 + 485, "+", size=30, weight=900)
    svg.rect(x4 + 62, y0 + 532, 176, 60, fill="#fffaf0", stroke=COLORS["amber"], sw=2, rx=9)
    svg.text(x4 + 150, y0 + 562, ["shared base head", "+ routed residual"], size=17, weight=800)
    svg.rect(x4 + 45, y0 + 618, 210, 64, fill=COLORS["green2"], stroke=COLORS["green"], sw=2, rx=10)
    svg.text(x4 + 150, y0 + 650, ["velocity / residual", "v_theta (N x 128 x 2)"], size=18, weight=850, fill=COLORS["green"])
    svg.line(x4 + 150, y0 + 160, x4 + 150, y0 + 188, stroke=COLORS["amber"], sw=2.4)
    svg.line(x4 + 150, y0 + 236, x4 + 150, y0 + 268, stroke=COLORS["amber"], sw=2.4)
    for sx in [88, 213]:
        svg.line(x4 + sx, y0 + 423, x4 + 140, y0 + 465, sw=2, marker=False)
    svg.line(x4 + 150, y0 + 506, x4 + 150, y0 + 532, sw=2.2)
    svg.line(x4 + 150, y0 + 592, x4 + 150, y0 + 618, sw=2.2)
    svg.line(x3 + 555, y0 + 470, x4 - 15, y0 + 470, sw=2.8)

    # Output/refinement.
    x5 = 2140
    svg.rect(x5, y0, 220, 650, fill="#fbfffb", stroke=COLORS["green"], sw=2.5, rx=18)
    svg.rect(x5 + 24, y0 + 48, 172, 106, fill=COLORS["green2"], stroke=COLORS["green"], sw=2, rx=10)
    svg.text(x5 + 110, y0 + 101, ["Flow-matching ODE", "dx/dt = v_theta(x,t)"], size=17, weight=850)
    svg.rect(x5 + 24, y0 + 188, 172, 92, fill=COLORS["green2"], stroke=COLORS["green"], sw=2, rx=10)
    svg.text(x5 + 110, y0 + 234, ["iterative contour", "refinement"], size=18, weight=850)
    for i, (cy, sc, col, label) in enumerate([(370, 0.66, "#9aa0a6", "coarse"), (462, 0.82, "#1f5fae", "mid"), (560, 1.0, "#1f5fae", "final")]):
        contour(svg, x5 + 110, y0 + cy, sc, col)
        svg.text(x5 + 34, y0 + cy, label, size=16, weight=700, fill=COLORS["muted"], anchor="start")
        if i < 2:
            svg.line(x5 + 110, y0 + cy + 40, x5 + 110, y0 + cy + 70, stroke=COLORS["green"], sw=2.2)
    svg.line(x4 + 300, y0 + 650, x5 - 15, y0 + 650, sw=2.8)
    svg.path(f"M {x4 + 150} {y0 + 682} C {x4 + 260} {y0 + 730}, {x5 - 70} {y0 + 185}, {x5 + 24} {y0 + 234}", stroke=COLORS["green"], sw=2.2, dash="8 6")

    # V4.6e/f note.
    svg.rect(45, 900, 610, 180, fill="#ffffff", stroke="#8a99a8", sw=2, rx=15)
    svg.text(70, 928, "Config-grounded notes", size=22, weight=900, fill=COLORS["ink"], anchor="start")
    notes = [
        "V4.6e: v4_10_use_dit_ffn_moe=true, 4 routed FFN experts, top-2.",
        "V4.6f: same DiT-internal MoE; adds near-zero and exact-zero state training.",
        "Final head remains V4.6 MoE: 8 MLP experts, top-2, shared base + routed residual.",
        "Aux regularization sums final-head MoE balance loss and each DiTBlock FFN-MoE balance loss.",
    ]
    for i, note in enumerate(notes):
        svg.text(80, 970 + i * 28, "- " + note, size=18, weight=550, fill=COLORS["ink"], anchor="start")

    # Legend.
    svg.rect(720, 900, 650, 150, fill="#ffffff", stroke="#8a99a8", sw=2, rx=15)
    legend = [
        (745, 930, COLORS["blue2"], COLORS["blue"], "DiT core / attention"),
        (745, 970, COLORS["teal2"], COLORS["teal"], "context features"),
        (745, 1010, COLORS["amber2"], COLORS["amber"], "router / gate"),
        (1040, 930, COLORS["red2"], COLORS["red"], "routed experts"),
        (1040, 970, COLORS["green2"], COLORS["green"], "flow/refinement output"),
    ]
    for x, y, fill, stroke, label in legend:
        svg.rect(x, y, 34, 22, fill=fill, stroke=stroke, sw=2, rx=4)
        svg.text(x + 45, y + 11, label, size=18, weight=650, fill=COLORS["ink"], anchor="start")

    # Source callouts.
    svg.rect(1435, 900, 925, 150, fill="#ffffff", stroke="#8a99a8", sw=2, rx=15)
    svg.text(1460, 930, "Implementation anchors", size=22, weight=900, fill=COLORS["ink"], anchor="start")
    impl = [
        "lib/networks/diffusion/dit_blocks_v3.py: RoutedFFNMoE and DiTBlockV3",
        "lib/networks/diffusion/dit_denoiser_v4_1.py: V4.1/V4.6 forward path and reg_loss",
        "configs/btcv_select_v4_6e_deepseek_dit_moe_gpu3.yaml and v4_6f_* configs",
    ]
    for i, note in enumerate(impl):
        svg.text(1470, 970 + i * 28, "- " + note, size=18, weight=550, fill=COLORS["ink"], anchor="start")

    svg.add("</svg>\n")
    OUT.write_text("\n".join(svg.parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    draw()
