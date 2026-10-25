"""
Build a print-ready PDF of the paper from paper/paper.md, with no LaTeX.

Equations are rendered with matplotlib's mathtext to PNG and placed inline /
as display blocks; tables are rendered as images; figures are embedded; text is
laid out with PyMuPDF. Output: paper/CLDB.pdf
"""
import hashlib
import os
import re

import fitz  # PyMuPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "paper.md")
OUT = os.path.join(HERE, "CLDB.pdf")
CACHE = os.path.join(HERE, ".mathcache")
os.makedirs(CACHE, exist_ok=True)

# ---- page geometry (A4, points) ----
PW, PH = 595.276, 841.89
M = 54.0
TW = PW - 2 * M
SIZE = 10.0
LEAD = 13.4

REG = fm.findfont("DejaVu Serif")
BOLD = fm.findfont(fm.FontProperties(family="DejaVu Serif", weight="bold"))
ITAL = fm.findfont(fm.FontProperties(family="DejaVu Serif", style="italic"))
MONO = fm.findfont("DejaVu Sans Mono")
SANS = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
import shutil
import tempfile

_SRC = {"reg": REG, "bold": BOLD, "ital": ITAL, "mono": MONO, "sans": SANS}
# PyMuPDF rejects font paths containing spaces; copy them to a space-free dir.
FDIR = os.path.join(tempfile.gettempdir(), "cldb_pdf_fonts")
os.makedirs(FDIR, exist_ok=True)
FONTS = {}
for _k, _s in _SRC.items():
    _d = os.path.join(FDIR, _k + ".ttf")
    if not os.path.exists(_d):
        shutil.copyfile(_s, _d)
    FONTS[_k] = _d
FOBJ = {k: fitz.Font(fontfile=v) for k, v in FONTS.items()}


# ============================================================ math rendering
def _strip_balanced(tex, cmd):
    out = []
    i = 0
    while i < len(tex):
        if tex.startswith(cmd, i):
            j = i + len(cmd)
            if j < len(tex) and tex[j] == "{":
                depth = 0
                k = j
                while k < len(tex):
                    if tex[k] == "{":
                        depth += 1
                    elif tex[k] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                out.append(tex[j + 1:k])
                i = k + 1
                if tex.startswith("_{", i):
                    depth = 0
                    k = i + 1
                    while k < len(tex):
                        if tex[k] == "{":
                            depth += 1
                        elif tex[k] == "}":
                            depth -= 1
                            if depth == 0:
                                break
                        k += 1
                    i = k + 1
                continue
        out.append(tex[i])
        i += 1
    return "".join(out)


def sanitize(tex):
    tex = _strip_balanced(tex, "\\underbrace")
    tex = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", tex)
    tex = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", tex)
    for bad in ["\\bigg", "\\Bigg", "\\big", "\\Big"]:
        tex = tex.replace(bad, "")
    tex = tex.replace("\\;", "\\,").replace("\\!", "")
    for bad in ["\\textstyle", "\\displaystyle", "\\limits", "\\nolimits", "\\left.", "\\right."]:
        tex = tex.replace(bad, "")
    tex = tex.replace("\\left", "").replace("\\right", "")
    tex = re.sub(r"\\operatorname\{([^{}]*)\}", r"\\mathrm{\1}", tex)
    tex = tex.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    tex = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", tex)
    tex = tex.replace("\\mid", "|").replace("\\varepsilon", "\\epsilon")
    tex = tex.replace("\\sqsubseteq", "\\subseteq")
    tex = tex.replace("\\dots", "...").replace("\\ldots", "...")
    return tex


def _fig_to_png(fig, key):
    path = os.path.join(CACHE, key + ".png")
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    w, h = Image.open(path).size
    return path, w / 300.0 * 72.0, h / 300.0 * 72.0


def render_math(tex, size=SIZE + 0.5):
    key = hashlib.md5((str(size) + "|" + tex).encode()).hexdigest()
    png = os.path.join(CACHE, key + ".png")
    if os.path.exists(png):
        w, h = Image.open(png).size
        return png, w / 300.0 * 72.0, h / 300.0 * 72.0
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, "$" + sanitize(tex) + "$", fontsize=size)
    return _fig_to_png(fig, key)


def render_cases(head, rows, size=SIZE + 0.5):
    key = hashlib.md5(("cases|" + head + "|".join(rows)).encode()).hexdigest()
    png = os.path.join(CACHE, key + ".png")
    if os.path.exists(png):
        w, h = Image.open(png).size
        return png, w / 300.0 * 72.0, h / 300.0 * 72.0
    n = len(rows)
    fig = plt.figure(figsize=(4.6, 0.30 * n + 0.2))
    fig.text(0.02, 0.5, "$" + sanitize(head) + "$", fontsize=size, va="center")
    fig.text(0.40, 0.5, "$\\{$", fontsize=size * (1.3 + 0.5 * n), va="center")
    y = 0.5 + 0.18 * (n - 1)
    for r in rows:
        fig.text(0.47, y, "$" + sanitize(r) + "$", fontsize=size, va="center")
        y -= 0.36
    return _fig_to_png(fig, key)


def render_table(rows):
    key = hashlib.md5(("tbl|" + "|".join("~".join(r) for r in rows)).encode()).hexdigest()
    png = os.path.join(CACHE, key + ".png")
    if os.path.exists(png):
        w, h = Image.open(png).size
        return png, w / 200.0 * 72.0, h / 200.0 * 72.0
    ncol = len(rows[0])
    fig_w = min(9.0, 1.1 + 0.95 * ncol)
    fig_h = 0.34 * len(rows) + 0.3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0)
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.visible_edges = "B"
            cell.set_edgecolor("#333333")
            cell.set_linewidth(0.9)
    fig.savefig(png, dpi=200, transparent=False, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    w, h = Image.open(png).size
    return png, w / 200.0 * 72.0, h / 200.0 * 72.0


# ============================================================ md -> blocks
def parse_inline(text):
    runs = []
    buf = ""
    i, n = 0, len(text)
    while i < n:
        if text[i] == "$":
            j = text.find("$", i + 1)
            if j != -1:
                if buf:
                    runs.append(("text", buf)); buf = ""
                runs.append(("math", text[i + 1:j]))
                i = j + 1
                continue
        if text.startswith("**", i):
            j = text.find("**", i + 2)
            if j != -1:
                if buf:
                    runs.append(("text", buf)); buf = ""
                runs.append(("bold", text[i + 2:j]))
                i = j + 2
                continue
        if text[i] == "`":
            j = text.find("`", i + 1)
            if j != -1:
                if buf:
                    runs.append(("text", buf)); buf = ""
                runs.append(("code", text[i + 1:j]))
                i = j + 1
                continue
        if text[i] == "*":
            j = text.find("*", i + 1)
            if j != -1 and "*" not in text[i + 1:j]:
                if buf:
                    runs.append(("text", buf)); buf = ""
                runs.append(("ital", text[i + 1:j]))
                i = j + 1
                continue
        buf += text[i]
        i += 1
    if buf:
        runs.append(("text", buf))

    def _dash(s):
        return s.replace("---", "\u2014").replace("--", "\u2013")

    return [(k, _dash(p) if k in ("text", "bold", "ital") else p) for k, p in runs]


def parse_blocks(md):
    lines = md.split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "":
            i += 1; continue
        if ln.strip() == "---":
            blocks.append(("hr", None)); i += 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            blocks.append(("h", (len(m.group(1)), m.group(2).strip()))); i += 1; continue
        if ln.lstrip().startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            blocks.append(("code", "\n".join(code))); continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", ln)
        if m:
            blocks.append(("img", m.group(2))); i += 1; continue
        if re.match(r"^\*(Figure|Table)[^*]*\*$", ln.strip()):
            blocks.append(("caption", ln.strip().strip("*"))); i += 1; continue
        if ln.lstrip().startswith(">"):
            q = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                q.append(lines[i].lstrip()[1:].strip()); i += 1
            blocks.append(("quote", " ".join(q))); continue
        if re.match(r"^\s*[-*]\s+", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i])); i += 1
            blocks.append(("ul", items)); continue
        if re.match(r"^\s*\d+\.\s+", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i])); i += 1
            blocks.append(("ol", items)); continue
        if ln.strip().startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows)); continue
        m = re.match(r"^\$\$(.+)\$\$\s*$", ln.strip())
        if m:
            blocks.append(("math", m.group(1))); i += 1; continue
        para = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#|\||>|\s*[-*]\s|\s*\d+\.\s|```|\$\$|---)", lines[i]):
            para.append(lines[i]); i += 1
        blocks.append(("p", " ".join(para)))
    return blocks


# ============================================================ layout engine
FONT_ALIAS = {"text": "reg", "bold": "bold", "ital": "ital", "code": "mono"}


class Builder:
    def __init__(self):
        self.doc = fitz.open()
        self.page = None
        self.y = 0
        self.new_page()

    def new_page(self):
        self.page = self.doc.new_page(width=PW, height=PH)
        for alias in FONTS:
            try:
                self.page.insert_font(fontname=alias, fontfile=FONTS[alias])
            except Exception:
                pass
        self.y = M

    def space(self, pts):
        self.y += pts
        if self.y > PH - M:
            self.new_page()

    def text_len(self, s, alias, size):
        return FOBJ[alias].text_length(s, fontsize=size)

    def draw_runs(self, runs, x0, size, font_alias="reg"):
        atoms = []
        for kind, payload in runs:
            if kind == "math":
                p, w, h = render_math(payload, size + 0.5)
                atoms.append(("math", p, w, h))
            else:
                al = FONT_ALIAS[kind]
                fs = size - 0.8 if kind == "code" else size
                for w in re.split(r"(\s+)", payload):
                    if w == "":
                        continue
                    atoms.append(("txt", w, self.text_len(w, al, fs), al, fs))
        x = x0
        baseline = self.y
        for a in atoms:
            if a[0] == "math":
                _, p, w, h = a
                rect = fitz.Rect(x, baseline - h * 0.76, x + w, baseline + h * 0.24)
                self.page.insert_image(rect, filename=p)
                x += w
            else:
                _, s, w, al, fs = a
                if s.strip():
                    self.page.insert_text((x, baseline), s, fontname=al, fontfile=FONTS[al], fontsize=fs)
                x += w
        return x

    def wrap_runs(self, runs, size, width):
        lines, cur, curw = [], [], 0.0
        for kind, payload in runs:
            if kind == "math":
                p, w, h = render_math(payload, size + 0.5)
                if curw + w > width and cur:
                    lines.append(cur); cur = []; curw = 0.0
                cur.append((kind, payload)); curw += w
                continue
            al = FONT_ALIAS[kind]
            fs = size - 0.8 if kind == "code" else size
            for tok in re.split(r"(\s+)", payload):
                if tok == "":
                    continue
                w = self.text_len(tok, al, fs)
                if curw + w > width and cur and tok.strip():
                    lines.append(cur); cur = []; curw = 0.0
                cur.append((kind, tok)); curw += w
        if cur:
            lines.append(cur)
        return lines

    def para(self, text, size=SIZE, width=None, italic=False):
        width = width or TW
        runs = parse_inline(text)
        if italic:
            runs = [("ital", p) if k == "text" else (k, p) for k, p in runs]
        lines = self.wrap_runs(runs, size, width)
        for line in lines:
            if self.y + LEAD > PH - M:
                self.new_page()
            self.draw_runs(line, M, size)
            self.y += LEAD

    def title(self, text):
        font = "sans"
        for line in self._wrap_plain(text, font, 17.5, TW):
            w = self.text_len(line, font, 17.5)
            self.page.insert_text(((PW - w) / 2, self.y), line, fontname=font, fontfile=FONTS[font], fontsize=17.5)
            self.y += 21
        self.y += 4

    def _wrap_plain(self, text, alias, size, width):
        words, line, out = text.split(), "", []
        for w in words:
            trial = (line + " " + w).strip()
            if self.text_len(trial, alias, size) > width and line:
                out.append(line); line = w
            else:
                line = trial
        if line:
            out.append(line)
        return out

    def heading(self, text, level):
        size = {1: 17.5, 2: 12.6, 3: 11.0, 4: 10.3}.get(level, 10)
        self.space(LEAD if level == 2 else 0.6 * LEAD)
        alias = "sans" if level <= 2 else "bold"
        for line in self._wrap_plain(text, alias, size, TW):
            if self.y + size * 1.6 > PH - M:
                self.new_page()
            self.page.insert_text((M, self.y), line, fontname=alias, fontfile=FONTS[alias], fontsize=size)
            self.y += size * 1.4
        self.space(2)

    def centered_math(self, tex):
        m = re.search(r"\\tag\{(\d+)\}", tex)
        tag = m.group(1) if m else None
        tex = re.sub(r"\\tag\{\d+\}", "", tex)
        if "\\begin{cases}" in tex:
            head, _, rest = tex.partition("\\begin{cases}")
            rest = rest.replace("\\end{cases}", "")
            rows = [r.replace("&", "\\quad \\mathrm{if}\\ ").strip()
                    for r in rest.split("\\\\") if r.strip()]
            p, w, h = render_cases(head.strip().rstrip("=").strip() + " =", rows)
        else:
            p, w, h = render_math(tex)
        if w > TW * 0.94:
            s = TW * 0.94 / w; w, h = w * s, h * s
        if self.y + h + 12 > PH - M:
            self.new_page()
        self.y += 5
        x = M + (TW - w) / 2
        self.page.insert_image(fitz.Rect(x, self.y, x + w, self.y + h), filename=p)
        if tag:
            self.page.insert_text((M + TW - 30, self.y + h * 0.55), f"({tag})", fontname="reg", fontfile=FONTS["reg"], fontsize=SIZE)
        self.y += h + 7

    def block_image(self, path):
        full = os.path.join(HERE, path)
        if not os.path.exists(full):
            print("  missing image:", path); return
        w, h = Image.open(full).size
        # images are ~dpi; scale to fit column width
        dw = min(TW * 0.9, w * 0.5)
        dh = h * dw / w
        if dh > PH - 2 * M - 40:
            dh = PH - 2 * M - 40; dw = w * dh / h
        if self.y + dh + 14 > PH - M:
            self.new_page()
        x = M + (TW - dw) / 2
        self.page.insert_image(fitz.Rect(x, self.y, x + dw, self.y + dh), filename=full)
        self.y += dh + 4

    def caption(self, text):
        size = 8.7
        runs = [("ital", p) for k, p in parse_inline(text)]
        for line in self.wrap_runs(runs, size, TW - 24):
            if self.y + 11 > PH - M:
                self.new_page()
            self.draw_runs(line, M + 12, size)
            self.y += 11
        self.y += 5

    def table_block(self, rows):
        p, w, h = render_table(rows)
        s = min(1.0, (TW * 0.98) / w)
        w, h = w * s, h * s
        if self.y + h + 10 > PH - M:
            self.new_page()
        x = M + (TW - w) / 2
        self.page.insert_image(fitz.Rect(x, self.y, x + w, self.y + h), filename=p)
        self.y += h + 7

    def code_block(self, code):
        size = 8.2
        maxw = TW - 16
        wrapped = []
        for ln in code.split("\n"):
            cur = ""
            for ch in ln:
                if self.text_len(cur + ch, "mono", size) > maxw:
                    wrapped.append(cur); cur = ""
                cur += ch
            wrapped.append(cur)
        lines = wrapped
        if self.y + len(lines) * (size + 2) + 12 > PH - M:
            self.new_page()
        y0 = self.y
        for ln in lines:
            self.page.insert_text((M + 6, self.y + size), ln, fontname="mono", fontfile=FONTS["mono"], fontsize=size)
            self.y += size + 2.4
        self.page.draw_rect(fitz.Rect(M, y0 - 4, M + TW, self.y + 1), color=(0.82, 0.82, 0.82), width=0.6)
        self.y += 8

    def quote(self, text):
        runs = [("ital", p) for k, p in parse_inline(text)]
        lines = self.wrap_runs(runs, SIZE, TW - 24)
        if self.y + len(lines) * LEAD + 8 > PH - M:
            self.new_page()
        y0 = self.y
        for line in lines:
            self.draw_runs(line, M + 18, SIZE)
            self.y += LEAD
        self.page.draw_line(fitz.Point(M + 8, y0 - 9), fitz.Point(M + 8, self.y - 5),
                            color=(0.5, 0.5, 0.5), width=1.2)
        self.y += 6

    def list_block(self, items, ordered=False):
        for k, it in enumerate(items):
            runs = parse_inline(it)
            lines = self.wrap_runs(runs, SIZE, TW - 24)
            if self.y + LEAD > PH - M:
                self.new_page()
            marker = f"{k+1}." if ordered else "\u2022"
            self.page.insert_text((M + 6, self.y), marker, fontname="reg", fontfile=FONTS["reg"], fontsize=SIZE)
            for line in lines:
                if self.y + LEAD > PH - M:
                    self.new_page()
                self.draw_runs(line, M + 22, SIZE)
                self.y += LEAD

    def hrule(self):
        self.space(4)
        self.page.draw_line(fitz.Point(M, self.y), fitz.Point(M + TW, self.y),
                            color=(0.65, 0.65, 0.65), width=0.6)
        self.space(9)


def build():
    with open(MD, encoding="utf-8") as f:
        md = f.read()
    blocks = parse_blocks(md)
    b = Builder()
    first_h = True
    for kind, payload in blocks:
        if kind == "h":
            lvl, text = payload
            if lvl == 1 and first_h:
                first_h = False
                b.space(2)
                b.title(text)
                continue
            b.heading(text, lvl)
        elif kind == "hr":
            b.hrule()
        elif kind == "p":
            b.para(payload); b.space(4)
        elif kind == "ul":
            b.list_block(payload, False); b.space(4)
        elif kind == "ol":
            b.list_block(payload, True); b.space(4)
        elif kind == "quote":
            b.quote(payload); b.space(5)
        elif kind == "code":
            b.code_block(payload)
        elif kind == "table":
            b.table_block(payload)
        elif kind == "math":
            b.centered_math(payload)
        elif kind == "img":
            b.block_image(payload)
        elif kind == "caption":
            b.caption(payload)
    n = b.doc.page_count
    for idx, pg in enumerate(b.doc):
        pg.insert_text((PW - M - 44, PH - 26), f"{idx+1} / {n}", fontname="reg", fontfile=FONTS["reg"], fontsize=8.5)
    b.doc.save(OUT, deflate=True)
    print(f"wrote {OUT} ({n} pages)")


if __name__ == "__main__":
    build()
