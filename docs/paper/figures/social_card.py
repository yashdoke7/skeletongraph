#!/usr/bin/env python3
"""SkeletonGraph social cards (LinkedIn 4:5 portrait, 1080x1350).

Not a paper figure. Optimised for a phone-sized feed thumbnail:
dark ground, few elements, nothing smaller than ~22px in logical space.

  card_a  "what it is"  -> the files-vs-function contrast + the three signals
  card_b  "what it did" -> the headline numbers

Usage:  python social_card.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

S = 2                       # supersample factor
W, H = 1080, 1350           # logical canvas (LinkedIn 4:5)
HERE = os.path.dirname(os.path.abspath(__file__))

FD = "C:/Windows/Fonts/"
_FONTS = {
    "reg":   FD + "segoeui.ttf",
    "bold":  FD + "segoeuib.ttf",
    "light": FD + "segoeuisl.ttf",
    "mono":  FD + "consola.ttf",
    "monob": FD + "consolab.ttf",
}
_cache = {}


def font(kind, size):
    key = (kind, int(size * S))
    if key not in _cache:
        _cache[key] = ImageFont.truetype(_FONTS[kind], int(size * S))
    return _cache[key]


# palette -- deep violet ground so the card cuts against a white feed
BG_TOP = (8, 6, 16)
BG_BOT = (24, 15, 42)
CARD = (35, 22, 58)
CARD_H = (42, 27, 68)          # slightly lifted card
EDGE = (60, 44, 90)
INK = (248, 245, 252)
MUT = (172, 160, 198)
FAINT = (110, 98, 140)
GREEN = (190, 242, 100)        # structural -- lime
BLUE = (34, 211, 238)          # lexical    -- electric cyan
PURPLE = (232, 90, 192)        # semantic   -- hot magenta
ORANGE = (251, 191, 36)        # gold, for the headline/cost accent
RED = (248, 113, 113)


def p(v):
    return int(round(v * S))


def mix(c1, c2, t):
    return tuple(int(round(c1[i] + (c2[i] - c1[i]) * t)) for i in range(3))


def new_canvas():
    img = Image.new("RGB", (p(W), p(H)), BG_TOP)
    d = ImageDraw.Draw(img)
    for j in range(p(H)):
        d.line([(0, j), (p(W), j)], fill=mix(BG_TOP, BG_BOT, j / p(H)))
    return img, ImageDraw.Draw(img)


def text(d, xy, s, kind="reg", size=24, fill=INK, anchor="la", spacing=8):
    d.text((p(xy[0]), p(xy[1])), s, font=font(kind, size), fill=fill,
           anchor=anchor, spacing=p(spacing))


def tw(s, kind, size):
    """Text width in logical px."""
    return font(kind, size).getlength(s) / S


def card(d, box, fill=CARD, edge=EDGE, r=18, width=1):
    x0, y0, x1, y1 = box
    d.rounded_rectangle([p(x0), p(y0), p(x1), p(y1)], radius=p(r),
                        fill=fill, outline=edge, width=max(1, p(width)))


def chip(d, x, y, label, kind="mono", size=22, fg=MUT, bg=(31, 42, 66), padx=12, h=34):
    w = tw(label, kind, size) + padx * 2
    d.rounded_rectangle([p(x), p(y), p(x + w), p(y + h)], radius=p(8), fill=bg)
    text(d, (x + padx, y + h / 2), label, kind, size, fg, anchor="lm")
    return w


def accent_bar(d, x, y, h, color, w=5):
    d.rounded_rectangle([p(x), p(y), p(x + w), p(y + h)], radius=p(3), fill=color)


def save(img, name):
    out = img.resize((W, H), Image.LANCZOS)
    path = os.path.join(HERE, name)
    out.save(path, optimize=True)
    print(f"{name}  {os.path.getsize(path)/1024:.0f} KB  {out.size}")


# ---------------------------------------------------------------- card A
def card_a():
    img, d = new_canvas()
    M = 64

    # header ------------------------------------------------------
    text(d, (M, 62), "SkeletonGraph", "bold", 58, INK)
    text(d, (M, 136), "zero-LLM structural retrieval for coding agents  ·  MCP",
         "reg", 24, FAINT)
    d.line([p(M), p(186), p(W - M), p(186)], fill=EDGE, width=p(1))

    # hook --------------------------------------------------------
    text(d, (M, 214), "the exact function,", "bold", 44, INK)
    text(d, (M, 268), "not a pile of files.", "bold", 44, GREEN)

    # the contrast ------------------------------------------------
    top, bot = 348, 700
    gap = 26
    cw = (W - 2 * M - gap) / 2
    lx, rx = M, M + cw + gap

    # left: what the agent does on its own
    card(d, (lx, top, lx + cw, bot))
    accent_bar(d, lx + 22, top + 24, 26, RED)
    text(d, (lx + 40, top + 26), "AGENT'S OWN SEARCH", "bold", 20, RED)
    text(d, (lx + 22, top + 78), 'grep "alter_field"', "mono", 23, MUT)
    files = ["schema.py", "operations.py", "models.py", "base.py"]
    yy = top + 124
    for f in files:
        d.rounded_rectangle([p(lx + 22), p(yy), p(lx + cw - 22), p(yy + 32)],
                            radius=p(7), fill=(31, 42, 66))
        text(d, (lx + 34, yy + 16), f, "mono", 21, MUT, anchor="lm")
        yy += 40
    text(d, (lx + 22, yy + 10), "+ 8 more files to open and read", "reg", 21, FAINT)

    # right: what SkeletonGraph returns
    card(d, (rx, top, rx + cw, bot), fill=CARD_H, edge=(40, 78, 70))
    accent_bar(d, rx + 22, top + 24, 26, GREEN)
    text(d, (rx + 40, top + 26), "SKELETONGRAPH", "bold", 20, GREEN)
    text(d, (rx + 22, top + 78), "db/backends/schema.py", "mono", 21, FAINT)
    d.rounded_rectangle([p(rx + 22), p(top + 118), p(rx + cw - 22), p(top + 186)],
                        radius=p(10), fill=(18, 48, 42), outline=(45, 90, 78), width=p(1))
    text(d, (rx + 40, top + 152), "_alter_field()", "monob", 28, GREEN, anchor="lm")
    text(d, (rx + 22, top + 206), "one function  ·  rank 1", "reg", 22, MUT)
    text(d, (rx + 22, top + 244), "line 412 – 468", "mono", 21, FAINT)
    text(d, (rx + 22, bot - 46), "→ handed to the agent over MCP", "reg", 21, GREEN)

    # how it works -------------------------------------------------
    hy = 736
    text(d, (M, hy), "THREE SIGNALS, FUSED", "bold", 20, FAINT)
    sy = hy + 42
    sigs = [
        ("BM25", "lexical", BLUE),
        ("jina-code vectors", "semantic", PURPLE),
        ("call graph + PageRank", "structural", GREEN),
    ]
    sh = 62
    for name, kind, col in sigs:
        card(d, (M, sy, W - M, sy + sh), fill=CARD, edge=EDGE, r=12)
        accent_bar(d, M + 18, sy + 16, sh - 32, col, w=4)
        text(d, (M + 38, sy + sh / 2), name, "bold", 25, INK, anchor="lm")
        text(d, (W - M - 24, sy + sh / 2), kind, "reg", 22, FAINT, anchor="rm")
        sy += sh + 12

    text(d, (W / 2, sy + 16), "reciprocal-rank fusion  →  one ranked function",
         "reg", 23, MUT, anchor="ma")
    text(d, (W / 2, sy + 54), "indexed once per repo  ·  zero LLM calls  ·  10 languages",
         "reg", 21, FAINT, anchor="ma")

    # measured -----------------------------------------------------
    sty, sth = 1098, 116
    card(d, (M, sty, W - M, sty + sth), fill=CARD, edge=EDGE, r=16)
    cols = [
        ("66→86%", "first-search recall", GREEN),
        ("0→80%", "function-level", BLUE),
        ("−42%", "cost at p95", ORANGE),
    ]
    colw = (W - 2 * M) / 3
    for i, (big, lab, col) in enumerate(cols):
        cx = M + colw * (i + 0.5)
        if i:
            d.line([p(M + colw * i), p(sty + 26), p(M + colw * i), p(sty + sth - 26)],
                   fill=EDGE, width=p(1))
        text(d, (cx, sty + 26), big, "bold", 36, col, anchor="ma")
        text(d, (cx, sty + 76), lab, "reg", 21, MUT, anchor="ma")

    # footer ------------------------------------------------------
    text(d, (M, 1262), "100 SWE-bench Verified tasks  ·  every fix Docker-verified",
         "reg", 22, FAINT)
    save(img, "sg_card_what.png")


# ---------------------------------------------------------------- card B
def card_b():
    img, d = new_canvas()
    M = 64

    text(d, (M, 62), "SkeletonGraph", "bold", 44, INK)
    text(d, (M, 122), "what better retrieval actually bought", "reg", 24, FAINT)
    d.line([p(M), p(172), p(W - M), p(172)], fill=EDGE, width=p(1))

    text(d, (M, 202), "Retrieval got much better.", "bold", 40, INK)
    text(d, (M, 252), "The typical task got no cheaper.", "bold", 40, ORANGE)

    stats = [
        ("86%", "first-search file recall", "up from 66% with the agent's own tools", GREEN),
        ("~80%", "function-level localization", "up from 0% — lexical search cannot", BLUE),
        ("−42%", "cost at the 95th percentile", "the median task barely moved", ORANGE),
    ]
    y, ch = 336, 150
    for big, label, sub, col in stats:
        card(d, (M, y, W - M, y + ch), fill=CARD, edge=EDGE, r=16)
        accent_bar(d, M, y + 24, ch - 48, col, w=6)
        mid = y + ch / 2
        text(d, (M + 40, mid), big, "bold", 58, col, anchor="lm")
        text(d, (M + 300, mid - 20), label, "bold", 26, INK, anchor="lm")
        text(d, (M + 300, mid + 20), sub, "reg", 21, FAINT, anchor="lm")
        y += ch + 22

    # why -----------------------------------------------------------
    yb = y + 14
    card(d, (M, yb, W - M, yb + 196), fill=CARD_H, edge=EDGE, r=16)
    text(d, (M + 36, yb + 28), "WHY", "bold", 20, FAINT)
    text(d, (M + 36, yb + 68),
         "Cost is set by how many turns an agent takes\n"
         "before it is confident enough to commit —\n"
         "not by how good its first search was.",
         "reg", 25, MUT, spacing=12)

    # trust ---------------------------------------------------------
    yt = yb + 218
    text(d, (M, yt), "HOW IT WAS MEASURED", "bold", 20, FAINT)
    items = [
        "every fix verified by running the repo's own tests in Docker",
        "re-run on a decontaminated benchmark of unmemorized repos",
        "bootstrap confidence intervals + paired significance tests",
    ]
    iy = yt + 40
    for it in items:
        d.ellipse([p(M + 4), p(iy + 9), p(M + 14), p(iy + 19)], fill=GREEN)
        text(d, (M + 30, iy), it, "reg", 22, MUT)
        iy += 38

    text(d, (M, 1272), "preprint on Research Square  ·  code + per-task results on GitHub",
         "reg", 22, FAINT)
    save(img, "sg_card_findings.png")


if __name__ == "__main__":
    card_a()
    card_b()
