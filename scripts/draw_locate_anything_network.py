#!/usr/bin/env python3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("report/RL/locate_anything_network_shapes.png")
W, H = 2400, 1500


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    if bold:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ] + candidates
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_SUB = font(23)
F_SEC = font(28, True)
F_BOX = font(23, True)
F_TXT = font(19)
F_SMALL = font(16)
F_CODE = font(18)


COL = {
    "bg": "#f7f8fb",
    "ink": "#172033",
    "muted": "#586579",
    "panel": "#ffffff",
    "panel_stroke": "#d8dee9",
    "input": "#eef7ff",
    "vision": "#f0fbf4",
    "bridge": "#f4efff",
    "llm": "#fff4e6",
    "out": "#fff0f0",
    "e4": "#edf8f7",
    "note": "#fff8db",
    "blue": "#4d81b7",
    "green": "#4d9360",
    "purple": "#8467c7",
    "orange": "#c58430",
    "red": "#b76565",
    "gray": "#536172",
}


img = Image.new("RGB", (W, H), COL["bg"])
d = ImageDraw.Draw(img)


def round_rect(xy, fill, outline=COL["panel_stroke"], width=2, r=18):
    d.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def text(x, y, s, f=F_TXT, fill=COL["ink"]):
    d.text((x, y), s, font=f, fill=fill)


def multiline(x, y, lines, f=F_TXT, fill=COL["ink"], gap=8):
    for line in lines:
        d.text((x, y), line, font=f, fill=fill)
        y += f.size + gap


def box(x, y, w, h, title, lines, fill, outline, title_fill=COL["ink"]):
    round_rect((x, y, x + w, y + h), fill=fill, outline=outline, width=2, r=16)
    text(x + 22, y + 18, title, F_BOX, title_fill)
    yy = y + 55
    for line in lines:
        text(x + 22, yy, line, F_TXT if not line.startswith("[") else F_CODE, COL["muted"] if not line.startswith("[") else "#263247")
        yy += 28


def arrow(x1, y1, x2, y2, color=COL["gray"], width=4):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    l = 16
    a = 0.55
    p1 = (x2 - l * math.cos(ang - a), y2 - l * math.sin(ang - a))
    p2 = (x2 - l * math.cos(ang + a), y2 - l * math.sin(ang + a))
    d.polygon([(x2, y2), p1, p2], fill=color)


def elbow(points, color=COL["gray"], width=4):
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        d.line((x1, y1, x2, y2), fill=color, width=width)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    arrow(x1, y1, x2, y2, color=color, width=width)


text(70, 58, "LocateAnything 网络结构与数据形状", F_TITLE)
text(70, 106, "本图按本地 checkpoint-3000 的代码/config 绘制：MoonViT vision tower + MLP connector + Qwen2 causal LM。", F_SUB, COL["muted"])

round_rect((50, 155, 1490, 1345), COL["panel"], COL["panel_stroke"])
round_rect((1540, 155, 2345, 1345), COL["panel"], COL["panel_stroke"])

text(85, 200, "A. LocateAnything 主流程", F_SEC)
text(1575, 200, "B. 和我们 E4 的关系", F_SEC)

box(95, 255, 350, 155, "输入图像 + 文本提示", [
    "Image: [H0,W0,3]",
    "Prompt: Locate all ...",
    "special image tokens"
], COL["input"], "#87b8e8")

box(535, 240, 380, 185, "Image Processor", [
    "rescale by token limit",
    "pad/resize to multiples of 28",
    "normalize mean=0.5 std=0.5",
    "patchify with patch=14"
], COL["input"], "#87b8e8")

box(1020, 240, 360, 165, "Patch Tensor", [
    "[L, 3, 14, 14]",
    "grid_hws = [B,2]",
    "L = sum(Hg*Wg)"
], COL["input"], "#87b8e8")

arrow(445, 321, 535, 321, COL["blue"])
arrow(915, 321, 1020, 321, COL["blue"])

box(95, 505, 390, 185, "MoonViT Patch Embed", [
    "Conv2d kernel=14 stride=14",
    "pos emb interpolation",
    "[L,3,14,14] -> [L,1152]"
], COL["vision"], "#80bd7b")

box(560, 505, 390, 185, "MoonViT Encoder", [
    "27 transformer layers",
    "hidden=1152 heads=16",
    "MLP intermediate=4304",
    "2D RoPE"
], COL["vision"], "#80bd7b")

box(1020, 505, 360, 185, "Patch Merger 2x2", [
    "[Hg,Wg,1152]",
    "2x2 group concat",
    "tokens: Hg/2 * Wg/2",
    "channel: 1152*4=4608"
], COL["vision"], "#80bd7b")

elbow([(1200, 405), (1200, 465), (290, 465), (290, 505)], COL["blue"])
arrow(485, 597, 560, 597, COL["green"])
arrow(950, 597, 1020, 597, COL["green"])

box(95, 775, 390, 185, "MLP Connector", [
    "LayerNorm(4608)",
    "Linear 4608 -> 2048",
    "GELU",
    "Linear 2048 -> 2048"
], COL["bridge"], "#9d83d9")

box(560, 775, 390, 175, "视觉 token embedding", [
    "[Nv,2048]",
    "Nv = sum(Hg/2 * Wg/2)",
    "replace <IMG_CONTEXT>"
], COL["bridge"], "#9d83d9")

box(1020, 775, 360, 175, "Text Embedding", [
    "input_ids -> [B,N,2048]",
    "image_token_index=151667",
    "visual tokens inserted"
], COL["bridge"], "#9d83d9")

elbow([(1200, 690), (1200, 735), (290, 735), (290, 775)], COL["green"])
arrow(485, 858, 560, 858, COL["purple"])
arrow(950, 858, 1020, 858, COL["purple"])

box(190, 1035, 430, 185, "Qwen2 Causal LM", [
    "36 decoder layers",
    "hidden=2048 heads=16",
    "vocab=152681",
    "max_position=32768"
], COL["llm"], "#dfa648")

box(800, 1035, 430, 185, "自回归输出", [
    "generated text tokens",
    "<ref> organ name",
    "<box> x1,y1,x2,y2",
    "coords parsed to boxes"
], COL["out"], "#d67d7d")

elbow([(1200, 950), (1200, 990), (405, 990), (405, 1035)], COL["purple"])
arrow(620, 1120, 800, 1120, COL["orange"])

box(1575, 260, 675, 165, "LocateAnything 原始输出", [
    "主要输出是文本序列",
    "通过生成坐标 token 得到 bbox",
    "不是直接输出 segmentation mask",
    "bbox 可用于 E2/V10.2 box-octagon"
], COL["out"], "#d67d7d")

box(1575, 500, 675, 190, "E4 抽的不是最终 bbox", [
    "E4 用 MoonViT 中间视觉特征",
    "脚本: e4_locate_extract_features.py",
    "抓 layer 9 + layer 18",
    "[1152,Hg,Wg] concat -> [2304,Hg,Wg]"
], COL["e4"], "#59aaa3")

box(1575, 765, 675, 190, "E4 后续融合", [
    "LocateFeatAdapter: 2304 -> 64",
    "grid_sample align to Snake feature",
    "[B,64,H/4,W/4]",
    "Residual Add with ResNet feature"
], COL["e4"], "#59aaa3")

box(1575, 1030, 675, 175, "关键区别", [
    "LocateAnything: vision tokens -> LLM -> bbox text",
    "E4: MoonViT mid features -> DiT/Snake condition",
    "所以 E4 没有使用 Locate 的语言端"
], COL["note"], "#ddb744")

arrow(1910, 425, 1910, 500, COL["red"])
arrow(1910, 690, 1910, 765, COL["green"])
arrow(1910, 955, 1910, 1030, COL["orange"])

round_rect((85, 1245, 1430, 1325), COL["note"], "#ddb744", width=2, r=14)
text(110, 1267, "Shape example: 若图像处理成 448x448，则 Hg=Wg=32；MoonViT patch tokens=1024；patch_merger 后视觉 token=16x16=256。", F_TXT, "#5d430b")
text(110, 1296, "MLP 后每个视觉 token 为 2048 维；processor 会按 token limit 和 28 的倍数动态 rescale，所以不同图像 Hg/Wg 可不同。", F_TXT, "#5d430b")

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT.resolve())
