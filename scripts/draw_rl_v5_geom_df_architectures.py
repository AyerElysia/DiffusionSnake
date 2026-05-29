#!/usr/bin/env python3
"""Draw research-style architecture diagrams for RL V5 geom and V5-DF."""

from pathlib import Path
import html
import math


ROOT = Path(__file__).resolve().parents[1]
W, H = 2400, 1350

COL = {
    "ink": "#17212b",
    "muted": "#5d6b78",
    "blue": "#174e9a",
    "blue2": "#eaf3ff",
    "teal": "#0b7d7b",
    "teal2": "#e6f8f6",
    "purple": "#6750a4",
    "purple2": "#f2edff",
    "amber": "#b46a00",
    "amber2": "#fff4dc",
    "red": "#b93a2d",
    "red2": "#fff0ee",
    "green": "#217a3a",
    "green2": "#ebf8ee",
    "gray": "#eef2f6",
    "line": "#263849",
}


def esc(s):
    return html.escape(str(s), quote=True)


class SVG:
    def __init__(self, title, subtitle):
        self.parts = []
        self.add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
        self.add(
            """
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="8" refX="11" refY="4" orient="auto">
    <path d="M0,0 L12,4 L0,8 Z" fill="#263849"/>
  </marker>
  <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.12"/>
  </filter>
  <style>
    text { font-family: DejaVu Sans, Arial, Helvetica, sans-serif; letter-spacing: 0; }
  </style>
</defs>
"""
        )
        self.rect(0, 0, W, H, fill="white", stroke="white", sw=0, rx=0)
        self.text(W / 2, 48, title, size=42, weight=900)
        self.text(W / 2, 90, subtitle, size=20, weight=520, fill=COL["muted"])

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="white", stroke="#222", sw=2, rx=12, dash=None, opacity=1.0):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr} opacity="{opacity}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=None, sw=2, dash=None, marker=True):
        stroke = stroke or COL["line"]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{dash_attr}{marker_attr}/>'
        )

    def path(self, d, stroke=None, sw=2, fill="none", dash=None, marker=True):
        stroke = stroke or COL["line"]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{dash_attr}{marker_attr}/>')

    def text(self, x, y, lines, size=22, weight=600, fill=None, anchor="middle", leading=1.25):
        fill = fill or COL["ink"]
        if isinstance(lines, str):
            lines = [lines]
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle">'
        )
        dy0 = -0.5 * (len(lines) - 1) * size * leading
        for i, line in enumerate(lines):
            dy = dy0 if i == 0 else size * leading
            self.add(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
        self.add("</text>")

    def header(self, x, y, w, label, color):
        self.rect(x, y, w, 58, fill=color, stroke=color, sw=0, rx=10)
        self.text(x + w / 2, y + 29, label, size=23, weight=850, fill="white")

    def box(self, x, y, w, h, title, body=None, fill="white", stroke=None, title_color=None, fs=20, dash=None):
        stroke = stroke or COL["line"]
        title_color = title_color or stroke
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=2.1, rx=13, dash=dash)
        if body:
            self.text(x + w / 2, y + 27, title, size=fs, weight=850, fill=title_color)
            self.text(x + w / 2, y + h / 2 + 17, body, size=max(15, fs - 3), weight=560, fill=COL["ink"])
        else:
            self.text(x + w / 2, y + h / 2, title, size=fs, weight=780, fill=title_color)

    def save(self, path: Path):
        self.add("</svg>\n")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def contour(svg, x, y, scale=1.0, color="#1f5fae", fill="none"):
    pts = []
    for i in range(38):
        a = 2 * math.pi * i / 38
        r = 1.0 + 0.18 * math.sin(3 * a) + 0.10 * math.cos(5 * a)
        pts.append((x + scale * r * 64 * math.cos(a), y + scale * r * 38 * math.sin(a)))
    d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts) + " Z"
    svg.path(d, stroke=color, sw=max(2, 3 * scale), fill=fill, marker=False)
    for px, py in pts[::3]:
        svg.add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{3.2 * scale:.1f}" fill="{color}"/>')


def normal_arrows(svg, x, y):
    for i in range(7):
        px = x + i * 34
        py = y + (i % 2) * 18
        svg.line(px, py, px, py - 36, stroke=COL["red"], sw=2.2)


def mini_tokens(svg, x, y, n=8, color="#0b7d7b"):
    for i in range(n):
        svg.rect(x + i * 24, y, 16, 16, fill=color, stroke="#064044", sw=1, rx=2)
    svg.text(x + n * 24 + 18, y + 8, "...", size=24, weight=800, fill=COL["muted"])


def add_common_footer(svg, source_cfg, policy):
    svg.rect(45, 1098, 710, 190, fill="white", stroke="#95a3b2", sw=2, rx=15)
    svg.text(70, 1128, "Config-grounded hyperparameters", size=22, weight=900, anchor="start")
    for i, line in enumerate(source_cfg):
        svg.text(78, 1170 + i * 29, "- " + line, size=18, weight=560, anchor="start")

    svg.rect(800, 1098, 700, 150, fill="white", stroke="#95a3b2", sw=2, rx=15)
    legend = [
        (830, 1132, COL["blue2"], COL["blue"], "V4.6c FM refiner / trainable flow"),
        (830, 1172, COL["teal2"], COL["teal"], "state and image context"),
        (830, 1212, COL["purple2"], COL["purple"], "policy action space"),
        (1160, 1132, COL["amber2"], COL["amber"], "reward / gate"),
        (1160, 1172, COL["red2"], COL["red"], "PPO / KL update"),
        (1160, 1212, COL["green2"], COL["green"], "deterministic eval / output"),
    ]
    for x, y, fill, stroke, label in legend:
        svg.rect(x, y, 34, 22, fill=fill, stroke=stroke, sw=2, rx=4)
        svg.text(x + 45, y + 11, label, size=17, weight=650, anchor="start")

    svg.rect(1540, 1098, 815, 150, fill="white", stroke="#95a3b2", sw=2, rx=15)
    svg.text(1565, 1128, "Implementation anchors", size=22, weight=900, anchor="start")
    anchors = [
        "grpo_train_v5_geom_action.py: rollout, reward, PPO ratio, KL, logging",
        "lib/train/rewards/region_reward.py and curvature_detail_reward.py",
        policy,
    ]
    for i, line in enumerate(anchors):
        svg.text(1575, 1170 + i * 29, "- " + line, size=18, weight=560, anchor="start")


def draw_geom():
    svg = SVG(
        "RL V5 Geom: Low-Frequency Normal-Direction Action Policy",
        "Three outer contour refinements; PPO acts in a reversible low-dimensional geometric subspace.",
    )
    y = 130
    svg.header(45, y, 300, "1. Base", COL["blue"])
    svg.header(390, y, 300, "2. State", COL["teal"])
    svg.header(735, y, 330, "3. FM Mean", COL["blue"])
    svg.header(1110, y, 435, "4. Geom Policy", COL["purple"])
    svg.header(1590, y, 330, "5. Reward and Gate", COL["amber"])
    svg.header(1965, y, 390, "6. PPO / Eval", COL["red"])

    top = 205
    svg.rect(45, top, 300, 845, fill="#f8fbff", stroke=COL["blue"], sw=2.5, rx=18)
    svg.box(70, top + 30, 250, 115, "V4.6c checkpoint", ["DiT-FM + MoE head", "resume_path: latest.pt"], fill="white", stroke=COL["blue"], title_color=COL["blue"])
    svg.box(70, top + 180, 250, 110, "freeze detector", ["YOLO / feature proj", "BN running stats fixed"], fill=COL["blue2"], stroke=COL["blue"], title_color=COL["blue"])
    svg.box(70, top + 330, 250, 125, "trainable module", ["inner.gcn / flow", "optimizer lr = 5e-8"], fill="#eef5ff", stroke=COL["blue"], title_color=COL["blue"])
    svg.box(70, top + 505, 250, 120, "frozen reference flow", ["ref_flow snapshot", "KL anchor"], fill="white", stroke="#5c7085", title_color="#34495e")
    contour(svg, 195, top + 735, 0.85, COL["blue"])
    svg.text(195, top + 805, "base contour refiner", size=19, weight=750, fill=COL["blue"])

    svg.rect(390, top, 300, 845, fill="#f8fffd", stroke=COL["teal"], sw=2.5, rx=18)
    svg.box(420, top + 34, 240, 112, "manual context", ["cnn_feature", "P2 + P3/detail"], fill="white", stroke=COL["teal"], title_color=COL["teal"])
    mini_tokens(svg, 453, top + 118, 7, COL["teal"])
    svg.box(420, top + 190, 240, 120, "contour state", ["i_it_py, c_it_py", "py_ind"], fill=COL["teal2"], stroke=COL["teal"], title_color=COL["teal"])
    contour(svg, 540, top + 270, 0.48, COL["teal"])
    svg.box(420, top + 350, 240, 120, "aligned target", ["i_gt_py", "orientation + roll"], fill="white", stroke=COL["teal"], title_color=COL["teal"])
    contour(svg, 540, top + 430, 0.48, "#246f9e")
    svg.box(420, top + 510, 240, 112, "fixed eval batches", ["8 validation batches", "deterministic smoke eval"], fill="white", stroke="#5c7085", title_color="#34495e")
    svg.text(540, top + 720, ["state s_j =", "current contour + context"], size=24, weight=900, fill=COL["teal"])

    svg.rect(735, top, 330, 845, fill="#f8fbff", stroke=COL["blue"], sw=2.5, rx=18)
    svg.text(900, top + 42, "Outer refinement mean", size=24, weight=900, fill=COL["blue"])
    fracs = ["0.3333", "0.5", "1.0"]
    for i, frac in enumerate(fracs):
        yy = top + 110 + i * 190
        svg.box(770, yy, 260, 92, f"outer step {i+1}", [f"frac = {frac}", "ODE steps = 20"], fill="white", stroke=COL["blue"], title_color=COL["blue"])
        svg.line(900, yy + 92, 900, yy + 140, stroke=COL["blue"], sw=2.4)
    svg.box(770, top + 690, 260, 100, "policy mean", ["mu_j = FM_disp(s_j)", "action baseline"], fill=COL["blue2"], stroke=COL["blue"], title_color=COL["blue"])
    svg.line(690, top + 420, 735, top + 420, sw=2.8)

    svg.rect(1110, top, 435, 845, fill="#fcfbff", stroke=COL["purple"], sw=2.5, rx=18)
    svg.text(1328, top + 38, "Low-frequency geometric action", size=24, weight=900, fill=COL["purple"])
    svg.box(1140, top + 85, 160, 85, "sample z", ["z ~ N(0, I)", "shape: B x 8"], fill="white", stroke=COL["purple"], title_color=COL["purple"])
    svg.box(1340, top + 85, 170, 85, "Fourier basis", ["constant + cos/sin", "8 low modes"], fill="white", stroke=COL["purple"], title_color=COL["purple"])
    svg.line(1300, top + 128, 1340, top + 128, stroke=COL["purple"], sw=2.4)
    svg.box(1140, top + 220, 370, 100, "normal-direction projection", ["delta = normal * lowfreq(z)", "sigma_px = [1.2, 0.8, 0.5]"], fill=COL["purple2"], stroke=COL["purple"], title_color=COL["purple"], fs=20)
    normal_arrows(svg, 1200, top + 390)
    contour(svg, 1328, top + 404, 0.70, COL["purple"])
    svg.box(1140, top + 505, 370, 100, "sampled action", ["a_j = mu_j + delta_j", "current <- current + a_j"], fill="white", stroke=COL["purple"], title_color=COL["purple"])
    svg.box(1140, top + 660, 370, 105, "exact log-prob", ["project(a_j - mu_j) -> z_hat", "old_log = log N(z_hat; 0, I)"], fill="white", stroke="#5c4e91", title_color=COL["purple"])
    svg.line(1065, top + 740, 1110, top + 740, sw=2.8)

    svg.rect(1590, top, 330, 845, fill="#fffdf8", stroke=COL["amber"], sw=2.5, rx=18)
    svg.text(1755, top + 38, "V5 reward", size=24, weight=900, fill=COL["amber"])
    reward_lines = ["0.30 mBoundF", "0.10 Dice", "0.25 IoU", "0.35 Dist"]
    for i, line in enumerate(reward_lines):
        svg.box(1625, top + 85 + i * 74, 260, 50, line, fill=COL["amber2"], stroke=COL["amber"], title_color=COL["amber"], fs=18)
    svg.box(1625, top + 410, 260, 86, "burr penalty", ["Laplacian spike excess", "weight = 0.06"], fill=COL["red2"], stroke=COL["red"], title_color=COL["red"])
    svg.box(1625, top + 545, 260, 92, "quality", ["Q_k = reward_k - baseline", "baseline = deterministic 3-step"], fill="white", stroke=COL["amber"], title_color=COL["amber"])
    svg.box(1625, top + 690, 260, 82, "group gate", ["max_k Q_k > 0.0", "adv = clip(norm(Q)) * gate"], fill=COL["amber2"], stroke=COL["amber"], title_color=COL["amber"])
    svg.line(1545, top + 555, 1590, top + 555, sw=2.8)

    svg.rect(1965, top, 390, 845, fill="#fffafa", stroke=COL["red"], sw=2.5, rx=18)
    svg.text(2160, top + 38, "PPO / GRPO-style update", size=24, weight=900, fill=COL["red"])
    svg.box(2000, top + 92, 320, 95, "recompute log-prob", ["lp_cur via lowfreq projection", "ratio = exp(lp_cur - old_log)"], fill="white", stroke=COL["red"], title_color=COL["red"])
    svg.box(2000, top + 235, 320, 90, "clipped policy loss", ["clip = 0.05", "inner epochs = 2"], fill=COL["red2"], stroke=COL["red"], title_color=COL["red"])
    svg.box(2000, top + 370, 320, 96, "reference KL", ["KL in z-space against ref_flow", "beta = 0.01; target = 0.002"], fill="white", stroke=COL["red"], title_color=COL["red"])
    svg.box(2000, top + 520, 320, 92, "optimizer step", ["train GCN / flow only", "grad clip = 0.3"], fill=COL["red2"], stroke=COL["red"], title_color=COL["red"])
    svg.box(2000, top + 680, 320, 105, "deterministic eval", ["3-step FM mean only", "best_iou / latest checkpoints"], fill=COL["green2"], stroke=COL["green"], title_color=COL["green"])
    svg.line(1920, top + 710, 1965, top + 710, sw=2.8)

    for x1, x2, yy in [(345, 390, top + 420), (690, 735, top + 420), (1065, 1110, top + 420), (1545, 1590, top + 420), (1920, 1965, top + 420)]:
        svg.line(x1, yy, x2, yy, sw=2.8)

    add_common_footer(
        svg,
        [
            "k=8 rollouts; outer_steps=3; fractions=[0.3333, 0.5, 1.0]",
            "geom_lowfreq_modes=8; geom_sigma_px=[1.2, 0.8, 0.5]",
            "gate_margin=0.0; ppo_clip=0.05; kl_beta=0.01; lr=5e-8",
            "reward = region/dice/iou/dist minus gt-relative burr penalty",
        ],
        "configs/btcv_select_v4_6c_rl_v5_geom_action_gpu3.yaml",
    )
    out = ROOT / "RL_V5_geom_action_architecture.svg"
    svg.save(out)
    print(out)


def draw_df():
    svg = SVG(
        "RL V5-DF: Inner Diffusion/Flow-Step Action Policy",
        "Same V5 reward and PPO loop, but each stored PPO action is an inner step_with_logprob transition.",
    )
    y = 130
    svg.header(45, y, 300, "1. V5 Setup", COL["blue"])
    svg.header(390, y, 320, "2. Outer Loop", COL["teal"])
    svg.header(755, y, 470, "3. Inner-Step Action", COL["purple"])
    svg.header(1270, y, 360, "4. Reconstruct", COL["blue"])
    svg.header(1675, y, 300, "5. Reward / Gate", COL["amber"])
    svg.header(2020, y, 335, "6. Inner-Step PPO", COL["red"])

    top = 205
    svg.rect(45, top, 300, 845, fill="#f8fbff", stroke=COL["blue"], sw=2.5, rx=18)
    svg.box(70, top + 34, 250, 120, "V4.6c base", ["DiT-FM + MoE head", "same resume_path"], fill="white", stroke=COL["blue"], title_color=COL["blue"])
    svg.box(70, top + 200, 250, 105, "frozen parts", ["YOLO / feature projection", "BN stats fixed"], fill=COL["blue2"], stroke=COL["blue"], title_color=COL["blue"])
    svg.box(70, top + 350, 250, 120, "train target", ["inner.gcn / flow", "ref_flow for KL"], fill="white", stroke=COL["blue"], title_color=COL["blue"])
    svg.box(70, top + 520, 250, 115, "action_policy", ["df_inner_step", "not geom"], fill=COL["purple2"], stroke=COL["purple"], title_color=COL["purple"])
    contour(svg, 195, top + 745, 0.82, COL["blue"])
    svg.text(195, top + 812, "same V5 batch context", size=18, weight=750, fill=COL["blue"])

    svg.rect(390, top, 320, 845, fill="#f8fffd", stroke=COL["teal"], sw=2.5, rx=18)
    svg.text(550, top + 40, "3 outer contour stages", size=24, weight=900, fill=COL["teal"])
    fracs = ["frac=0.3333", "frac=0.5", "frac=1.0"]
    for i, frac in enumerate(fracs):
        yy = top + 105 + i * 190
        svg.box(430, yy, 240, 88, f"outer stage {i+1}", [frac, "prepare context"], fill="white", stroke=COL["teal"], title_color=COL["teal"])
        svg.line(550, yy + 88, 550, yy + 137, stroke=COL["teal"], sw=2.4)
    svg.box(430, top + 680, 240, 95, "apply final inner latent", ["disp = denorm(x_T)", "current += frac * clamp(disp)"], fill=COL["teal2"], stroke=COL["teal"], title_color=COL["teal"])
    svg.line(345, top + 420, 390, top + 420, sw=2.8)

    svg.rect(755, top, 470, 845, fill="#fcfbff", stroke=COL["purple"], sw=2.5, rx=18)
    svg.text(990, top + 38, "Stored PPO action is x_prev", size=24, weight=900, fill=COL["purple"])
    svg.box(790, top + 90, 180, 88, "latent start", ["x_0 = zeros_like", "current contour state"], fill="white", stroke=COL["purple"], title_color=COL["purple"])
    svg.box(1010, top + 90, 180, 88, "sampling context", ["sampled_feat", "detail_feat, scale"], fill="white", stroke=COL["teal"], title_color=COL["teal"])
    svg.line(970, top + 134, 1010, top + 134, stroke=COL["purple"], sw=2.4)
    svg.rect(790, top + 230, 400, 300, fill=COL["purple2"], stroke=COL["purple"], sw=2.2, rx=15)
    svg.text(990, top + 260, "inner loop: idx = 0..9", size=22, weight=900, fill=COL["purple"])
    for i in range(4):
        x = 835 + i * 88
        svg.rect(x, top + 310, 66, 58, fill="white", stroke=COL["purple"], sw=1.8, rx=8)
        label = f"step {i}" if i < 3 else "..."
        svg.text(x + 33, top + 339, [label, "x_t -> x_prev"], size=13, weight=750, fill=COL["purple"])
        if i < 3:
            svg.line(x + 66, top + 339, x + 88, top + 339, stroke=COL["purple"], sw=2)
    svg.box(835, top + 405, 310, 82, "gcn.step_with_logprob(...)", ["step_mode='gaussian', action_std=0.001", "noise_level=0.0, sde_type='sde'"], fill="white", stroke=COL["purple"], title_color=COL["purple"], fs=18)
    svg.box(790, top + 585, 400, 110, "trajectory cache", ["x_t, x_prev, old_log, t, idx", "features, scale, self_cond"], fill="white", stroke=COL["purple"], title_color=COL["purple"])
    svg.box(790, top + 745, 400, 70, "old_log from exact step transition", ["not low-frequency z projection"], fill="white", stroke="#5c4e91", title_color=COL["purple"])
    svg.line(710, top + 420, 755, top + 420, sw=2.8)

    svg.rect(1270, top, 360, 845, fill="#f8fbff", stroke=COL["blue"], sw=2.5, rx=18)
    svg.text(1450, top + 38, "Reconstruct rollout", size=24, weight=900, fill=COL["blue"])
    svg.box(1305, top + 90, 290, 100, "inner latent path", ["x <- x_prev at each step", "10 log-prob actions per outer stage"], fill="white", stroke=COL["blue"], title_color=COL["blue"])
    svg.box(1305, top + 250, 290, 96, "convert to contour move", ["denormalize_pred_disp(x)", "clamp_pred_disp(disp, current)"], fill=COL["blue2"], stroke=COL["blue"], title_color=COL["blue"])
    svg.box(1305, top + 405, 290, 90, "outer update", ["applied = frac * disp", "total_disp += applied"], fill="white", stroke=COL["blue"], title_color=COL["blue"])
    contour(svg, 1450, top + 635, 0.58, "#9aa4ad")
    contour(svg, 1450, top + 710, 0.74, COL["blue"])
    contour(svg, 1450, top + 805, 0.90, COL["green"])
    svg.line(1450, top + 670, 1450, top + 690, stroke=COL["blue"], sw=2.3)
    svg.line(1450, top + 745, 1450, top + 778, stroke=COL["blue"], sw=2.3)
    svg.line(1225, top + 420, 1270, top + 420, sw=2.8)

    svg.rect(1675, top, 300, 845, fill="#fffdf8", stroke=COL["amber"], sw=2.5, rx=18)
    svg.text(1825, top + 38, "Same V5 quality", size=24, weight=900, fill=COL["amber"])
    svg.box(1705, top + 92, 240, 98, "baseline", ["deterministic 3-step", "no inner stochastic action"], fill="white", stroke=COL["amber"], title_color=COL["amber"])
    svg.box(1705, top + 240, 240, 120, "reward", ["0.30 mBoundF + 0.10 Dice", "+ 0.25 IoU + 0.35 Dist", "- 0.06 burr"], fill=COL["amber2"], stroke=COL["amber"], title_color=COL["amber"])
    svg.box(1705, top + 420, 240, 100, "quality", ["Q_k = reward_k - baseline", "k = 8 rollouts"], fill="white", stroke=COL["amber"], title_color=COL["amber"])
    svg.box(1705, top + 590, 240, 100, "gate", ["max_k Q_k > -0.002", "adv clipped to +/-2"], fill=COL["amber2"], stroke=COL["amber"], title_color=COL["amber"])
    svg.line(1630, top + 545, 1675, top + 545, sw=2.8)

    svg.rect(2020, top, 335, 845, fill="#fffafa", stroke=COL["red"], sw=2.5, rx=18)
    svg.text(2188, top + 38, "PPO over inner steps", size=24, weight=900, fill=COL["red"])
    svg.box(2048, top + 88, 280, 108, "replay transition", ["prev_sample = cached x_prev", "lp_cur = step_with_logprob"], fill="white", stroke=COL["red"], title_color=COL["red"])
    svg.box(2048, top + 245, 280, 95, "ratio objective", ["ratio = exp(lp_cur - old_log)", "clip = 0.05"], fill=COL["red2"], stroke=COL["red"], title_color=COL["red"])
    svg.box(2048, top + 388, 280, 105, "reference KL", ["mean_cur vs mean_ref", "variance from std_cur^2"], fill="white", stroke=COL["red"], title_color=COL["red"])
    svg.box(2048, top + 548, 280, 105, "aggregation", ["~30 actions per rollout", "3 outer stages x 10 inner steps"], fill=COL["red2"], stroke=COL["red"], title_color=COL["red"])
    svg.box(2048, top + 705, 280, 92, "checkpoint / eval", ["latest.pt, best_iou.pt", "fixed 8-batch eval"], fill=COL["green2"], stroke=COL["green"], title_color=COL["green"])
    svg.line(1975, top + 545, 2020, top + 545, sw=2.8)

    for x1, x2, yy in [(345, 390, top + 420), (710, 755, top + 420), (1225, 1270, top + 420), (1630, 1675, top + 420), (1975, 2020, top + 420)]:
        svg.line(x1, yy, x2, yy, sw=2.8)

    add_common_footer(
        svg,
        [
            "action_policy='df_inner_step'; df_step_mode='gaussian'",
            "ode_steps=10; df_action_std=0.001; df_noise_level=0.0",
            "k=8 rollouts; outer_steps=3; gate_margin=-0.002",
            "same V5 reward, burr penalty, PPO clip, KL beta, and eval machinery",
        ],
        "configs/btcv_select_v4_6c_rl_v5_df_inner_step_gpu5.yaml",
    )
    out = ROOT / "RL_V5_DF_inner_step_architecture.svg"
    svg.save(out)
    print(out)


def main():
    draw_geom()
    draw_df()


if __name__ == "__main__":
    main()
