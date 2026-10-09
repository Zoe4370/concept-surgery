#!/usr/bin/env python3
"""Typeset paper/submission_draft.md as a research-paper PDF.

    python scripts/build_paper_pdf.py --output paper/submission_draft.pdf

Supported Markdown: a front-matter block (author, date, status, keywords)
ended by ---, '#' title, '##' subtitle, '###' sections, '####' subsections,
'![Figure N. caption](path){width=NN}' figures (width is a percent of the
text block), 'Table N. caption' followed by a pipe table, '$$ latex $$ (n)'
numbered equations (matplotlib mathtext), and **bold**, *italic* inline text.
Requires reportlab and matplotlib.
"""
from __future__ import annotations

import argparse
import html
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
LIB = Path("/usr/share/fonts/truetype/liberation")
INK = colors.HexColor("#000000")
RULE = colors.HexColor("#8a93a3")
ACCENT = colors.HexColor("#000000")
PAGE_W, PAGE_H = A4
MARGIN = 24 * mm
TEXT_W = PAGE_W - 2 * MARGIN


def register_fonts() -> str:
    files = {"LibSerif": "LiberationSerif-Regular.ttf", "LibSerif-Bold": "LiberationSerif-Bold.ttf",
             "LibSerif-Italic": "LiberationSerif-Italic.ttf",
             "LibSerif-BoldItalic": "LiberationSerif-BoldItalic.ttf"}
    if not all((LIB / f).exists() for f in files.values()):
        return "Times-Roman"
    for name, f in files.items():
        pdfmetrics.registerFont(TTFont(name, str(LIB / f)))
    pdfmetrics.registerFontFamily("LibSerif", normal="LibSerif", bold="LibSerif-Bold",
                                  italic="LibSerif-Italic", boldItalic="LibSerif-BoldItalic")
    return "LibSerif"


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    # allow the few reportlab tags used in the source
    text = text.replace("&lt;super&gt;", "<super>").replace("&lt;/super&gt;", "</super>")
    text = text.replace("&lt;sub&gt;", "<sub>").replace("&lt;/sub&gt;", "</sub>")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<i>\1</i>", text)
    return text


def render_equation(latex: str, directory: Path, index: int) -> tuple[Path, float, float]:
    plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${latex}$", fontsize=11.5, color="#000000")
    out = directory / f"eq{index}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
    w, h = PILImage.open(out).size
    return out, w / 300 * 72, h / 300 * 72


def build_table(caption: str, rows: list[list[str]], font: str, styles: dict) -> KeepTogether:
    size = 8.4
    cell = ParagraphStyle("cell", fontName=font, fontSize=size, leading=size + 2.6, textColor=INK)
    cell_c = ParagraphStyle("cellc", parent=cell, alignment=TA_CENTER)
    numeric = re.compile(r"^[\d.,\s]+$|^pending$|^n/a$", re.I)
    data = []
    for r, row in enumerate(rows):
        out = []
        for c, text in enumerate(row):
            st = cell_c if (c > 0 and (numeric.match(text.strip()) or r == 0)) else cell
            markup = inline(text)
            out.append(Paragraph(f"<b>{markup}</b>" if r == 0 else markup, st))
        data.append(out)
    nat = []
    for c in range(len(rows[0])):
        w = max(pdfmetrics.stringWidth(re.sub(r"[*]", "", row[c]), font, size) for row in rows)
        nat.append(w + 14)
    total = sum(nat)
    avail = TEXT_W
    if total > avail:
        nat = [w * avail / total for w in nat]
    elif total < avail * 0.8:
        pass
    table = Table(data, colWidths=nat, hAlign="CENTER", repeatRows=1)
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    label, rest = re.match(r"(Table [A-Z]?\d+\.)\s*(.*)", caption).groups()
    cap = Paragraph(f"<b>{label}</b> {inline(rest)}", styles["caption_above"])
    return KeepTogether([cap, table, Spacer(1, 9)])


def build_figure(caption: str, path: Path, pct: int, styles: dict) -> KeepTogether:
    w_px, h_px = PILImage.open(path).size
    width = TEXT_W * pct / 100
    img = Image(str(path), width=width, height=width * h_px / w_px)
    img.hAlign = "CENTER"
    label, rest = re.match(r"(Figure [A-Z]?\d+\.)\s*(.*)", caption).groups()
    cap = Paragraph(f"<b>{label}</b> {inline(rest)}", styles["caption"])
    return KeepTogether([Spacer(1, 3), img, Spacer(1, 3), cap, Spacer(1, 6)])


def make_styles(font: str) -> dict:
    b, i = font + "-Bold", font + "-Italic"
    if font == "Times-Roman":
        b, i = "Times-Bold", "Times-Italic"
    return {
        "title": ParagraphStyle("title", fontName=b, fontSize=18.5, leading=23, alignment=TA_CENTER,
                                textColor=INK, spaceAfter=5),
        "subtitle": ParagraphStyle("subtitle", fontName=i, fontSize=11.5, leading=15, alignment=TA_CENTER,
                                   textColor=colors.HexColor("#000000"), spaceAfter=10),
        "author": ParagraphStyle("author", fontName=b, fontSize=11.5, leading=15, alignment=TA_CENTER,
                                 textColor=INK),
        "date": ParagraphStyle("date", fontName=font, fontSize=9.5, leading=13, alignment=TA_CENTER,
                               textColor=colors.HexColor("#000000"), spaceAfter=8),
        "status": ParagraphStyle("status", fontName=i, fontSize=8.6, leading=11.8, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#5b4a1c"), backColor=colors.HexColor("#fbf5e6"),
                                 borderColor=colors.HexColor("#e2cf98"), borderWidth=0.5, borderPadding=6,
                                 leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=16),
        "abs_head": ParagraphStyle("abs_head", fontName=b, fontSize=10, leading=13, alignment=TA_CENTER,
                                   textColor=INK, spaceBefore=2, spaceAfter=3),
        "abstract": ParagraphStyle("abstract", fontName=font, fontSize=9.2, leading=12.4, alignment=TA_JUSTIFY,
                                   leftIndent=11 * mm, rightIndent=11 * mm, textColor=INK, spaceAfter=4),
        "keywords": ParagraphStyle("keywords", fontName=font, fontSize=9, leading=12, leftIndent=11 * mm,
                                   rightIndent=11 * mm, textColor=INK, spaceAfter=10),
        "h1": ParagraphStyle("h1", fontName=b, fontSize=12, leading=15, textColor=ACCENT, spaceBefore=13,
                             spaceAfter=5, keepWithNext=True),
        "h2": ParagraphStyle("h2", fontName=b, fontSize=10.3, leading=13, textColor=INK, spaceBefore=8,
                             spaceAfter=3, keepWithNext=True),
        "body": ParagraphStyle("body", fontName=font, fontSize=10, leading=13.6, alignment=TA_JUSTIFY,
                               textColor=INK, spaceAfter=6),
        "ref": ParagraphStyle("ref", fontName=font, fontSize=8.8, leading=11.6, leftIndent=15,
                              firstLineIndent=-15, textColor=INK, spaceAfter=3.5),
        "caption": ParagraphStyle("caption", fontName=font, fontSize=8.8, leading=11.6, alignment=TA_JUSTIFY,
                                  leftIndent=8 * mm, rightIndent=8 * mm, textColor=colors.HexColor("#000000")),
        "caption_above": ParagraphStyle("caption_above", fontName=font, fontSize=8.8, leading=11.6,
                                        alignment=TA_JUSTIFY, textColor=colors.HexColor("#000000"),
                                        spaceBefore=4, spaceAfter=4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "paper" / "submission_draft.md"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src, target = Path(args.input), Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    font = register_fonts()
    styles = make_styles(font)
    tmp = Path(tempfile.mkdtemp())

    lines = src.read_text(encoding="utf-8").splitlines()
    story: list = []
    front: dict[str, str] = {}
    title_done = in_front = False
    mode, para, eq_n = "body", [], 0
    i = 0

    def flush() -> None:
        nonlocal para
        if not para:
            return
        text = " ".join(x.strip() for x in para)
        style = {"abstract": "abstract", "refs": "ref"}.get(mode, "body")
        story.append(Paragraph(inline(text), styles[style]))
        para = []

    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        i += 1
        if re.match(r"^(author|date|status|keywords):", s) and mode == "body" and not in_front:
            in_front = True
        if in_front:
            m = re.match(r"^(\w+):\s*(.*)", s)
            if m:
                front[m.group(1)] = m.group(2)
                continue
            if s == "---":
                in_front = False
                story.append(Paragraph(inline(front.get("author", "")), styles["author"]))
                story.append(Paragraph(inline(front.get("date", "")), styles["date"]))
                if front.get("status"):
                    story.append(Paragraph(inline(front["status"]), styles["status"]))
                continue
            if not s:
                continue
        if not s or s == "---":
            flush()
            continue
        if s.startswith("# "):
            flush(); story.append(Paragraph(inline(s[2:]), styles["title"])); title_done = True
        elif s.startswith("## ") and title_done and not front:
            flush(); story.append(Paragraph(inline(s[3:]), styles["subtitle"]))
        elif s.startswith("### "):
            flush()
            head = s[4:]
            if head.lower() == "abstract":
                mode = "abstract"; story.append(Paragraph("Abstract", styles["abs_head"]))
            else:
                if head.startswith("Appendix A"):
                    story.append(PageBreak())
                if mode == "abstract" and front.get("keywords"):
                    story.append(Paragraph(f"<b>Keywords:</b> {inline(front['keywords'])}", styles["keywords"]))
                mode = "refs" if head.lower() == "references" else "body"
                story.append(Paragraph(inline(head), styles["h1"]))
        elif s.startswith("#### "):
            flush(); story.append(Paragraph(inline(s[5:]), styles["h2"]))
        elif s.startswith("!["):
            flush()
            m = re.match(r"!\[(.*?)\]\((.*?)\)(?:\{width=(\d+)\})?", s)
            story.append(build_figure(m.group(1), ROOT / m.group(2), int(m.group(3) or 100), styles))
        elif re.match(r"^Table [A-Z]?\d+\.", s):
            flush()
            rows = []
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-+:?$", c) for c in cells):
                    rows.append(cells)
                i += 1
            story.append(build_table(s, rows, font, styles))
        elif s.startswith("$$"):
            flush()
            m = re.match(r"^\$\$\s*(.*?)\s*\$\$\s*(\(\d+\))?$", s)
            eq_n += 1
            path, w, h = render_equation(m.group(1), tmp, eq_n)
            img = Image(str(path), width=w, height=h)
            number = Paragraph(m.group(2) or "", ParagraphStyle("eqn", fontName=font, fontSize=10,
                                                                 alignment=2, textColor=INK))
            t = Table([[img, number]], colWidths=[TEXT_W - 14 * mm, 14 * mm])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (0, 0), "CENTER"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story.append(t)
        else:
            para.append(s)
    flush()

    doc = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=22 * mm, bottomMargin=24 * mm,
                            title="When Does Concept Erasure Survive the Forward Pass?",
                            author=front.get("author", ""),
                            subject="Activation-space editing and a multi-model audit protocol",
                            keywords=front.get("keywords", ""))
    doc.build(story)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
