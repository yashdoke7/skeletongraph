#!/usr/bin/env python3
"""SkeletonGraph hero visuals -- explainer animation + README banner.

Five beats, in the order the system actually runs:

  1 INDEX   tree-sitter parses the repo into symbols + a call graph, no LLM
  2 QUERY   the agent asks in natural language, over MCP
  3 RANK    three independent signals rank THE SAME symbols, differently
  4 FUSE    reciprocal-rank fusion merges the three orderings
  5 RESULT  what the accuracy bought, charted, with baselines attached

The worked example is one real task (django/django). The point of beat 4 is
that the answer is rank 2 / 3 / 2 -- top of nothing -- and still wins the
fusion. That is the argument for fusing, and it is true.

Every number shown is measured and carries its baseline. The cost chart is
deliberately two bars, because the honest headline is that the median did
not move and the tail collapsed.

Outputs
  sg_hero.gif        1120x630  16:9, README (variable frame holds)
  sg_hero.mp4        1080x1350 4:5, LinkedIn (core framed by title/stat bands)
  sg_hero_still.png  1120x630  the fuse frame
  sg_banner.png      1600x460  README header

Usage:  python sg_hero.py
"""
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
S = 2
FD = "C:/Windows/Fonts/"
_F = {"reg": FD + "segoeui.ttf", "bold": FD + "segoeuib.ttf",
      "mono": FD + "consola.ttf", "monob": FD + "consolab.ttf"}
_fc = {}


def font(kind, size):
    k = (kind, int(size * S))
    if k not in _fc:
        _fc[k] = ImageFont.truetype(_F[kind], int(size * S))
    return _fc[k]


def p(v):
    return int(round(v * S))


BG0 = (6, 10, 20)
BG1 = (14, 21, 38)
PANEL = (21, 30, 48)
INK = (237, 243, 250)
MUT = (140, 156, 178)
DIM = (86, 102, 128)
EDGE = (40, 55, 82)
STEEL = (140, 162, 194)
CODE_C = (86, 106, 136)
BLUE = (96, 165, 250)
PURPLE = (167, 139, 250)
GREEN = (52, 211, 153)
AMBER = (251, 191, 36)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def mix(a, b, t):
    t = clamp(t)
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def fade(c, t, ground=BG1):
    return mix(ground, c, t)


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def seg(t, a, b):
    return 0.0 if b <= a else clamp((t - a) / (b - a))


def band(t, a, b, r=0.03):
    return smooth(seg(t, a, a + r)) * (1 - smooth(seg(t, b - r, b)))


# ---------------------------------------------------------------- content
# one real task: django/django -- the schema editor drops the wrong index.
# positions are normalised 0..1 and mapped into whatever region we draw in.
SYMS = [
    ("alter_field",       0.02, 0.06),
    ("execute",           0.55, 0.00),
    ("_alter_field",      0.29, 0.34),   # index 2 -- the answer
    ("_delete_index_sql", 0.85, 0.32),
    ("index_sql",         0.00, 0.62),
    ("_remake_table",     0.34, 0.71),
    ("column_sql",        0.85, 0.70),
    ("add_field",         0.14, 1.00),
    ("_constraint_names", 0.62, 1.00),
]
ANSWER = 2
CALLS = [(0, 2), (2, 3), (2, 5), (2, 1), (3, 1), (5, 7), (7, 6),
         (4, 8), (4, 2), (6, 3), (8, 6)]

QUERY = '"_alter_field drops the wrong index on SQLite"'

RANKERS = [
    ("BM25", "lexical", BLUE,
     ["alter_field", "_alter_field", "add_field", "index_sql"]),
    ("jina-code", "semantic", PURPLE,
     ["_delete_index_sql", "index_sql", "_alter_field", "_constraint_names"]),
    ("call graph", "structural", GREEN,
     ["_remake_table", "_alter_field", "execute", "column_sql"]),
]
FUSED = ["_alter_field", "index_sql", "_delete_index_sql", "alter_field"]

CODE_COLS = [[
    "def _alter_field(self, model, old_field, new_field):",
    "    if not self._field_should_be_altered(old, new):",
    "        return self.execute(self._delete_index_sql())",
    "    old_type = old_field.db_parameters(connection)",
    "    self._remake_table(model, alter_field=(old, new))",
    "class BaseDatabaseSchemaEditor:",
    "    sql_create_index = 'CREATE INDEX %(name)s ON ..'",
    "    def _delete_composed_index(self, model, fields):",
    "        names = self._constraint_names(model, cols)",
    "    def add_field(self, model, field):",
    "        definition = self.column_sql(model, field)",
    "    def index_sql(self, model, fields, suffix=''):",
    "        space = self._get_index_tablespace(model)",
    "    def execute(self, sql, params=()):",
    "        cursor.execute(sql, params or None)",
], [
    "def column_sql(self, model, field, include_default):",
    "    db_params = field.db_parameters(self.connection)",
    "    sql, params = db_params['type'], []",
    "    if field.null and not features.implied_column:",
    "        null = 'NULL' if field.empty_strings else ''",
    "def _constraint_names(self, model, columns=None):",
    "    with self.connection.cursor() as cursor:",
    "        cons = introspection.get_constraints(cursor)",
    "    return [n for n, info in cons.items() if match]",
    "def _remake_table(self, model, create_field=None):",
    "    self.effective_default(create_field)",
    "    mapping = {f.column: self.quote_name(f.column)}",
    "    self.alter_db_table(new_model, old_name, tmp)",
    "    for index in model._meta.indexes:",
    "        self.execute(index.create_sql(model, self))",
], [
    "class Symbol: fqn: str; file: str; line: int",
    "    def walk(self, tree): yield from self._descend()",
    "    symbols[fqn] = Symbol(kind, start, end)",
    "    graph.add_edge(caller_fqn, callee_fqn)",
    "def build_index(root: Path) -> SkeletonIndex:",
    "    for path in iter_source_files(root):",
    "        tree = PARSERS[lang].parse(path.read_bytes())",
    "        for sym in extract_symbols(tree, path):",
    "            bm25.add(sym.fqn, tokenize(sym.body))",
    "    rank = pagerank(call_graph, alpha=0.85)",
    "def search(q: str, k: int = 10) -> list[Hit]:",
    "    lists = [bm25(q), dense(q), structural(q)]",
    "    for lst in lists:",
    "        for i, fqn in enumerate(lst):",
    "            scores[fqn] += 1.0 / (60 + i)",
]]


# ---------------------------------------------------------------- drawing
def canvas(w, h):
    img = Image.new("RGB", (p(w), p(h)), BG0)
    d = ImageDraw.Draw(img)
    for j in range(p(h)):
        d.line([(0, j), (p(w), j)], fill=mix(BG0, BG1, (j / p(h)) * 0.9))
    return img


def text(d, xy, s, kind="reg", size=24, fill=INK, anchor="la"):
    d.text((p(xy[0]), p(xy[1])), s, font=font(kind, size), fill=fill, anchor=anchor)


def tw(s, kind, size):
    return font(kind, size).getlength(s) / S


def rrect(d, box, r, fill=None, outline=None, w=1):
    d.rounded_rectangle([p(box[0]), p(box[1]), p(box[2]), p(box[3])],
                        radius=p(r), fill=fill, outline=outline, width=max(1, p(w)))


def glow_pass(img, fn, blur=10):
    gl = Image.new("RGB", img.size, (0, 0, 0))
    fn(ImageDraw.Draw(gl))
    return ImageChops.screen(img, gl.filter(ImageFilter.GaussianBlur(p(blur))))


def vignette(img, w, h, strength=0.40):
    m = Image.new("L", (p(w), p(h)), 0)
    ImageDraw.Draw(m).ellipse([p(-w * .25), p(-h * .3), p(w * 1.25), p(h * 1.3)], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(p(80)))
    return Image.composite(img, Image.blend(img, Image.new("RGB", img.size, BG0),
                                            strength), m)


def sym_chip(d, cx, cy, name, a, col=None, size=16, strong=False):
    """A node is a named function, not an anonymous dot."""
    if a <= 0.02:
        return
    w = tw(name + "()", "mono", size) + 22
    h = size + 14
    body = mix(BG1, (46, 38, 14) if strong else (30, 42, 66), a)
    rrect(d, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 7, fill=body,
          outline=fade(col or EDGE, a * (0.95 if strong else 0.5)),
          w=1.6 if strong else 1)
    text(d, (cx, cy), name + "()", "monob" if strong else "mono", size,
         fade(col or STEEL, a), anchor="mm")


def bar(d, x, y, w, h, frac, col, a, track=True):
    if track:
        rrect(d, (x, y, x + w, y + h), h / 2, fill=fade(EDGE, a * 0.55))
    if frac > 0.005:
        rrect(d, (x, y, x + max(h, w * frac), y + h), h / 2, fill=fade(col, a))


# ---------------------------------------------------------------- frame
W, H = 1120, 630


def frame(t):
    img = canvas(W, H)
    d = ImageDraw.Draw(img)
    M = 46

    a_index = band(t, 0.00, 0.36)
    # the query bar takes the space the graph vacates -- they cross-fade
    a_query = smooth(seg(t, 0.35, 0.40)) * (1 - smooth(seg(t, 0.86, 0.90)))
    a_rank = band(t, 0.40, 0.86) * (1 - 0.55 * smooth(seg(t, 0.68, 0.78)))
    a_fuse = smooth(seg(t, 0.66, 0.74)) * (1 - smooth(seg(t, 0.86, 0.90)))
    a_buys = smooth(seg(t, 0.88, 0.94))
    graph_out = smooth(seg(t, 0.34, 0.40))

    # ---- header
    text(d, (M, 26), "SkeletonGraph", "bold", 32, INK)
    text(d, (M, 68), "structural code retrieval for coding agents  ·  MCP server",
         "reg", 16, MUT)
    text(d, (W - M, 32), "example:  django/django", "mono", 14, DIM, anchor="ra")
    d.line([p(M), p(98), p(W - M), p(98)], fill=EDGE, width=p(1))

    beats = [(0.00, "1  INDEX", "tree-sitter → symbols + call graph · once per repo, no LLM"),
             (0.35, "2  QUERY", "the agent asks in natural language, over MCP"),
             (0.40, "3  RANK", "three independent signals rank the same symbols"),
             (0.66, "4  FUSE", "reciprocal-rank fusion merges the three orderings"),
             (0.88, "5  RESULT", "what that accuracy actually bought, measured")]
    for i, (st, tag, sub) in enumerate(beats):
        en = beats[i + 1][0] if i + 1 < len(beats) else 1.01
        if st <= t < en:
            a = band(t, st, en, 0.02)
            text(d, (M, 112), tag, "bold", 18, fade(AMBER, a))
            text(d, (M + tw(tag, "bold", 18) + 18, 113), sub, "reg", 17, fade(MUT, a))

    # ---- beat 1: the wall of code dissolving into a labelled call graph
    if a_index > 0.01 and graph_out < 1:
        sweep = 150 + 480 * smooth(seg(t, 0.10, 0.30)) if t >= 0.10 else None
        code_a = a_index * (1 - smooth(seg(t, 0.12, 0.30)))
        if code_a > 0.02:
            for ci, lines in enumerate(CODE_COLS):
                for i, ln in enumerate(lines):
                    y = 152 + i * 31
                    a = code_a * (clamp((y - sweep) / 50 + 0.5) if sweep else 1)
                    if a > 0.02:
                        d.text((p(M + ci * 348), p(y)), ln, font=font("mono", 12),
                               fill=fade(CODE_C, a))

        ga = smooth(seg(t, 0.12, 0.32)) * (1 - graph_out)
        if ga > 0.01:
            gx0, gx1, gy0, gy1 = 118, 1002, 172, 552

            def P(i):
                return gx0 + SYMS[i][1] * (gx1 - gx0), gy0 + SYMS[i][2] * (gy1 - gy0)

            for a_i, b_i in CALLS:
                ax, ay = P(a_i)
                bx, by = P(b_i)
                v = ga * (clamp((sweep - max(ay, by)) / 60 + 0.5) if sweep else 1)
                if v > 0.02:
                    d.line([p(ax), p(ay), p(bx), p(by)], fill=mix(BG1, EDGE, v * 1.7),
                           width=p(1.3))
            for i, (name, _, _) in enumerate(SYMS):
                cx, cy = P(i)
                v = ga * (clamp((sweep - cy) / 60 + 0.5) if sweep else 1)
                sym_chip(d, cx, cy, name, v, col=AMBER if i == ANSWER else None,
                         strong=(i == ANSWER and t > 0.26))
            if sweep and t < 0.30:
                img = glow_pass(img, lambda g: g.line(
                    [p(M), p(sweep), p(W - M), p(sweep)],
                    fill=mix((0, 0, 0), GREEN, 0.85), width=p(2)), blur=8)
                d = ImageDraw.Draw(img)
                d.line([p(M), p(sweep), p(W - M), p(sweep)], fill=mix(BG1, GREEN, 0.6),
                       width=p(1))
            # only once the code has cleared, so it never sits on top of it
            text(d, (W / 2, 594), "a node is a function  ·  an edge is a call",
                 "reg", 17, fade(DIM, ga * (1 - clamp(code_a * 2.5))), anchor="ma")

    # ---- the MCP query, persistent from beat 2
    if a_query > 0.01:
        qa = a_query
        rrect(d, (M, 142, W - M, 198), 10, fill=mix(BG1, PANEL, qa),
              outline=fade(EDGE, qa), w=1)
        text(d, (M + 20, 152), "sg_search(", "mono", 17, fade(DIM, qa))
        text(d, (M + 20 + tw("sg_search(", "mono", 17), 152), QUERY, "mono", 17,
             fade(INK, qa))
        text(d, (W - M - 20, 153), "MCP", "bold", 14, fade(GREEN, qa * .9), anchor="ra")
        text(d, (M + 20, 175), "natural language in  ·  no symbol names required",
             "reg", 14, fade(DIM, qa))

    # ---- beat 3: three rankers over the same symbols
    if a_rank > 0.01:
        cw, gap = 336, 24
        x0 = (W - (cw * 3 + gap * 2)) / 2
        for ci, (nm, kind, col, lst) in enumerate(RANKERS):
            x = x0 + ci * (cw + gap)
            ca = a_rank * smooth(seg(t, 0.42 + ci * .03, 0.50 + ci * .03))
            if ca <= 0.02:
                continue
            rrect(d, (x, 218, x + cw, 424), 11, fill=mix(BG1, PANEL, ca * .85),
                  outline=fade(EDGE, ca), w=1)
            d.ellipse([p(x + 16), p(230), p(x + 27), p(241)], fill=fade(col, ca))
            text(d, (x + 36, 226), nm, "bold", 18, fade(INK, ca))
            text(d, (x + 36, 248), kind, "reg", 14, fade(DIM, ca))
            for ri, sym in enumerate(lst):
                ry = 278 + ri * 35
                hit = sym == "_alter_field"
                ra = ca * smooth(seg(t, 0.46 + ci * .03 + ri * .012,
                                     0.54 + ci * .03 + ri * .012))
                if ra <= 0.02:
                    continue
                if hit:
                    rrect(d, (x + 10, ry - 5, x + cw - 10, ry + 26), 6,
                          fill=mix(BG1, mix(PANEL, col, .24), ra),
                          outline=fade(col, ra * .7), w=1)
                text(d, (x + 22, ry), str(ri + 1), "monob", 15,
                     fade(col if hit else DIM, ra))
                text(d, (x + 44, ry), sym + "()", "mono", 15,
                     fade(INK if hit else MUT, ra))
        ca = a_rank * smooth(seg(t, 0.56, 0.62)) * (1 - smooth(seg(t, 0.66, 0.71)))
        text(d, (W / 2, 438), "the answer is rank 2, 3, 2  —  top of none of them",
             "reg", 17, fade(MUT, ca), anchor="ma")

    # ---- beat 4: fusion into one ordering
    if a_fuse > 0.01:
        fa = a_fuse
        cw, gap = 336, 24
        x0 = (W - (cw * 3 + gap * 2)) / 2
        for ci in range(3):
            sx = x0 + ci * (cw + gap) + cw / 2
            d.line([p(sx), p(430), p(W / 2), p(462)], fill=fade(RANKERS[ci][2], fa * .5),
                   width=p(1))
        text(d, (W / 2, 440), "reciprocal-rank fusion   k = 60", "mono", 15,
             fade(MUT, fa), anchor="ma")
        rrect(d, (M, 472, W - M, 620), 11, fill=mix(BG1, PANEL, fa),
              outline=fade(EDGE, fa), w=1)
        text(d, (M + 20, 481), "FUSED RANKING", "bold", 14, fade(DIM, fa))
        for ri, sym in enumerate(FUSED):
            ry = 506 + ri * 28
            top = ri == 0
            if top:
                rrect(d, (M + 12, ry - 4, W - M - 12, ry + 24), 6,
                      fill=mix(BG1, (46, 38, 14), fa), outline=fade(AMBER, fa * .8), w=1)
            text(d, (M + 26, ry), str(ri + 1), "monob", 15, fade(AMBER if top else DIM, fa))
            text(d, (M + 50, ry), sym + "()", "monob" if top else "mono", 15,
                 fade(AMBER if top else MUT, fa))
            if top:
                text(d, (W - M - 26, ry + 1), "db/backends/schema.py:412", "mono", 14,
                     fade(MUT, fa), anchor="ra")

    # ---- beat 5: charted results, baselines attached
    if a_buys > 0.01:
        ba = a_buys
        text(d, (M, 146), "MEASURED ON 100 SWE-bench VERIFIED TASKS  ·  "
                          "EVERY FIX RUN AGAINST THE REPO'S OWN TESTS", "bold", 14,
             fade(DIM, ba))

        # left: retrieval accuracy, baseline vs SkeletonGraph
        text(d, (M, 186), "retrieval accuracy", "bold", 19, fade(INK, ba))
        bw = 300
        for gi, (lab, base_v, sg_v, col) in enumerate(
                [("first-search file recall", .663, .862, GREEN),
                 ("function-level localization", .0, .80, BLUE)]):
            gy = 228 + gi * 108
            text(d, (M, gy), lab, "reg", 16, fade(MUT, ba))
            for bi, (v, c) in enumerate(((base_v, STEEL), (sg_v, col))):
                by = gy + 26 + bi * 30
                bar(d, M, by, bw, 20, v * smooth(seg(t, 0.90, 0.99)), c, ba)
                text(d, (M + bw + 14, by + 1), f"{int(round(v*100))}%", "bold", 16,
                     fade(c, ba))
        lx = M
        for sw, lab in ((STEEL, "agent's own tools"), (GREEN, "+ SkeletonGraph")):
            rrect(d, (lx, 454, lx + 14, 466), 3, fill=fade(sw, ba))
            text(d, (lx + 22, 452), lab, "reg", 14, fade(MUT, ba))
            lx += tw(lab, "reg", 14) + 56

        # right: the honest cost story -- two bars, one axis
        rx = 640
        text(d, (rx, 186), "cost per task vs. the agent's own tools", "bold", 19,
             fade(INK, ba))
        ax = 930
        d.line([p(ax), p(224), p(ax), p(400)], fill=fade(EDGE, ba), width=p(1))
        text(d, (ax, 404), "0", "mono", 13, fade(DIM, ba), anchor="ma")
        scale = 190 / 45.0
        prog = smooth(seg(t, 0.90, 0.99))
        for bi, (lab, pct, col) in enumerate(
                [("median task  (p50)", +1.9, MUT), ("worst 5%  (p95)", -42.5, GREEN)]):
            by = 246 + bi * 78
            text(d, (rx, by - 22), lab, "reg", 16, fade(MUT, ba))
            wpx = abs(pct) * scale * prog
            if pct >= 0:
                rrect(d, (ax, by, ax + max(3, wpx), by + 26), 5, fill=fade(col, ba * .8))
            else:
                rrect(d, (ax - wpx, by, ax, by + 26), 5, fill=fade(col, ba))
            text(d, (ax + 16 if pct >= 0 else ax - wpx - 16, by + 3),
                 f"{pct:+.1f}%", "bold", 18, fade(col, ba),
                 anchor="la" if pct >= 0 else "ra")
        text(d, (rx, 424), "the median task did not get cheaper.", "reg", 17,
             fade(INK, ba))
        text(d, (rx, 448), "the expensive tail did — that is the whole finding.",
             "reg", 17, fade(AMBER, ba * .9))

        text(d, (M, 500), "accuracy is not the product. fewer turns before the agent "
                          "commits is.", "reg", 16, fade(DIM, ba))
        text(d, (M, 560), "github.com/yashdoke/skeletongraph", "mono", 16,
             fade(MUT, ba))
        text(d, (W - M, 560), "preprint  ·  10.21203/rs.3.rs-10749266/v1", "mono", 15,
             fade(DIM, ba), anchor="ra")

    return vignette(img, W, H).resize((W, H), Image.LANCZOS)


# ---------------------------------------------------------------- timing
def timeline():
    """(t, milliseconds) -- motion is quick, finished beats hold long enough to read."""
    out = []

    def move(a, b, n, ms=70):
        for i in range(n):
            out.append((a + (b - a) * i / n, ms))

    def hold(t, ms):
        out.append((t, ms))

    move(0.00, 0.12, 7)
    move(0.12, 0.30, 20)
    hold(0.315, 1500)          # read the call graph
    move(0.315, 0.40, 8)
    hold(0.398, 1700)          # read the query
    move(0.398, 0.62, 20)
    hold(0.635, 2600)          # read three rankings + the punchline
    move(0.635, 0.78, 12)
    hold(0.82, 2600)           # read the fused result
    move(0.82, 0.99, 12)
    hold(0.995, 3600)          # read the charts
    return out


# ---------------------------------------------------------------- outputs
def render_all():
    tl = timeline()
    print(f"  {len(tl)} frames, {sum(ms for _, ms in tl)/1000:.1f}s")
    frames = []
    for i, (t, _) in enumerate(tl):
        frames.append(frame(t))
        if i % 15 == 0:
            print(f"  frame {i}/{len(tl)}")
    return tl, frames


def save_gif(tl, frames, w=960):
    h = int(round(w * H / W))
    seq = [f.resize((w, h), Image.LANCZOS).convert(
        "P", palette=Image.ADAPTIVE, colors=96) for f in frames]
    path = os.path.join(HERE, "sg_hero.gif")
    seq[0].save(path, save_all=True, append_images=seq[1:],
                duration=[ms for _, ms in tl], loop=0, optimize=True, disposal=2)
    print(f"sg_hero.gif  {os.path.getsize(path)/1024/1024:.1f} MB  {w}x{h}")


def save_mp4(tl, frames, fps=25):
    """4:5 for the feed: title band, the 16:9 core, then the numbers."""
    import cv2

    VW, VH = 1080, 1350
    core_w = 1080
    core_h = int(round(core_w * H / W))
    top = 300
    plate = canvas(VW, VH)
    d = ImageDraw.Draw(plate)
    text(d, (60, 74), "SkeletonGraph", "bold", 58, INK)
    text(d, (62, 158), "the exact function, not a pile of files", "reg", 28, GREEN)
    text(d, (62, 202), "an MCP server that indexes your repo with tree-sitter, then",
         "reg", 19, MUT)
    text(d, (62, 230), "ranks symbols by BM25 + embeddings + call graph, fused by RRF",
         "reg", 19, MUT)
    by = top + core_h + 34
    d.line([p(60), p(by - 18), p(VW - 60), p(by - 18)], fill=EDGE, width=p(1))
    rows = [("first-search file recall", "66%", "86%", GREEN),
            ("function-level localization", "0%", "~80%", BLUE),
            ("cost at the 95th percentile", "", "−42%", AMBER)]
    for i, (lab, was, now, col) in enumerate(rows):
        ry = by + 14 + i * 52
        text(d, (62, ry), lab, "reg", 22, INK)
        if was:
            text(d, (600, ry), was, "bold", 22, DIM)
            text(d, (672, ry), "→", "reg", 21, DIM)
        text(d, (716, ry - 3), now, "bold", 27, col)
    text(d, (62, by + 184), "100 SWE-bench Verified tasks · Docker-verified · the median "
                            "task did not move, the tail did", "reg", 16, DIM)
    plate = plate.resize((VW, VH), Image.LANCZOS)

    path = os.path.join(HERE, "sg_hero.mp4")
    # mp4v first on purpose: OpenCV's bundled H.264 writer ignores bitrate hints
    # and emits ~50x larger files for flat UI frames like these.
    vw = None
    for cc in ("mp4v", "avc1"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (VW, VH))
        if vw.isOpened():
            print(f"  codec {cc}")
            break
        vw.release()
    for (t, ms), f in zip(tl, frames):
        page = plate.copy()
        page.paste(f.resize((core_w, core_h), Image.LANCZOS), (0, top))
        arr = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        for _ in range(max(1, int(round(ms / 1000 * fps)))):
            vw.write(arr)
    vw.release()
    print(f"sg_hero.mp4  {os.path.getsize(path)/1024/1024:.1f} MB  {VW}x{VH}")


BW, BH = 1600, 460


def banner():
    img = canvas(BW, BH)
    d = ImageDraw.Draw(img)
    gx0, gx1, gy0, gy1 = 900, 1520, 96, 388

    def P(i):
        return gx0 + SYMS[i][1] * (gx1 - gx0), gy0 + SYMS[i][2] * (gy1 - gy0)

    for a_i, b_i in CALLS:
        ax, ay = P(a_i)
        bx, by = P(b_i)
        d.line([p(ax), p(ay), p(bx), p(by)], fill=EDGE, width=p(1.4))
    for i, (name, _, _) in enumerate(SYMS):
        cx, cy = P(i)
        sym_chip(d, cx, cy, name, 1.0, col=AMBER if i == ANSWER else None,
                 size=14, strong=(i == ANSWER))
    ax, ay = P(ANSWER)
    img = glow_pass(img, lambda g: g.ellipse(
        [p(ax - 58), p(ay - 32), p(ax + 58), p(ay + 32)],
        fill=mix((0, 0, 0), AMBER, 0.30)), blur=15)
    d = ImageDraw.Draw(img)

    text(d, (72, 78), "SkeletonGraph", "bold", 70, INK)
    text(d, (76, 168), "the exact function, not a pile of files", "reg", 29, GREEN)
    text(d, (76, 216), "an MCP server that indexes your repo with tree-sitter, then ranks",
         "reg", 20, MUT)
    text(d, (76, 244), "symbols by BM25 + embeddings + call graph, fused with RRF",
         "reg", 20, MUT)
    for i, (lab, val, col) in enumerate(
            [("first-search file recall", "66% → 86%", GREEN),
             ("function-level localization", "0% → ~80%", BLUE),
             ("cost at the 95th percentile", "−42%", AMBER)]):
        y = 300 + i * 40
        text(d, (76, y), lab, "reg", 19, MUT)
        text(d, (420, y - 2), val, "bold", 21, col)
    text(d, (76, 424), "100 SWE-bench Verified tasks · Docker-verified · median cost "
                       "unchanged, the tail is where it pays", "reg", 16, DIM)

    out = vignette(img, BW, BH, 0.38).resize((BW, BH), Image.LANCZOS)
    path = os.path.join(HERE, "sg_banner.png")
    out.save(path, optimize=True)
    print(f"sg_banner.png  {os.path.getsize(path)/1024:.0f} KB  {out.size}")


if __name__ == "__main__":
    banner()
    tl, frames = render_all()
    save_gif(tl, frames)
    save_mp4(tl, frames)
    sp = os.path.join(HERE, "sg_hero_still.png")
    frame(0.82).save(sp, optimize=True)
    print(f"sg_hero_still.png  {os.path.getsize(sp)/1024:.0f} KB")
