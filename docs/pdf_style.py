"""Shared fonts, colours, styles and layout helpers for the EduTune AI PDF documents."""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, Spacer, Table,
                                TableStyle)
from reportlab.platypus.doctemplate import BaseDocTemplate, PageTemplate
from reportlab.platypus.frames import Frame
from reportlab.platypus.tableofcontents import TableOfContents

import config

# ---------------------------------------------------------------- fonts / colours
F = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("Arial", f"{F}\\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", f"{F}\\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", f"{F}\\ariali.ttf"))
pdfmetrics.registerFont(TTFont("Consolas", f"{F}\\consola.ttf"))
pdfmetrics.registerFont(TTFont("Consolas-Bold", f"{F}\\consolab.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold")

INDIGO = colors.HexColor("#4338ca")
VIOLET = colors.HexColor("#7c3aed")
INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#475569")
LINE = colors.HexColor("#cbd5e1")
SOFT = colors.HexColor("#eef2ff")
GREEN = colors.HexColor("#15803d")
GREEN_BG = colors.HexColor("#dcfce7")
AMBER_BG = colors.HexColor("#fef3c7")
AMBER = colors.HexColor("#b45309")
GREY_BG = colors.HexColor("#f1f5f9")
PINK_BG = colors.HexColor("#fae8ff")
BASE_C = colors.HexColor("#94a3b8")

# ---------------------------------------------------------------- styles
S = {
    "body": ParagraphStyle("body", fontName="Arial", fontSize=10, leading=14.5, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Arial", fontSize=8.5, leading=11.5, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.8, leading=11.8, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.8, leading=11.8, textColor=colors.white),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=19, leading=24, textColor=INDIGO,
                         spaceBefore=4, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=13.5, leading=18, textColor=INK,
                         spaceBefore=12, spaceAfter=6, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="Arial-Bold", fontSize=11, leading=15, textColor=VIOLET,
                         spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "bullet": ParagraphStyle("bullet", fontName="Arial", fontSize=10, leading=14, textColor=INK,
                             leftIndent=14, bulletIndent=3, spaceAfter=2.5),
    "code": ParagraphStyle("code", fontName="Consolas", fontSize=7.9, leading=10.2, textColor=INK),
    "caption": ParagraphStyle("caption", fontName="Arial-Italic", fontSize=8.5, leading=11, textColor=MUTED,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "toc1": ParagraphStyle("toc1", fontName="Arial-Bold", fontSize=10.5, leading=16, leftIndent=0),
    "toc2": ParagraphStyle("toc2", fontName="Arial", fontSize=9.5, leading=13, leftIndent=16, textColor=MUTED),
}


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md(t: str) -> str:
    """Tiny markdown -> ReportLab markup (bold, headings, line breaks)."""
    t = esc(t)
    t = re.sub(r"^#{1,4}\s*(.+)$", r"<b>\1</b>", t, flags=re.MULTILINE)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t.replace("\n", "<br/>")


def P(t, style="body"):
    return Paragraph(t, S[style])


def H1(t):
    p = Paragraph(t, S["h1"]); p.toc_level = 0; return p


def H2(t):
    p = Paragraph(t, S["h2"]); p.toc_level = 1; return p


def H3(t):
    return Paragraph(t, S["h3"])


def bullets(items, style="bullet"):
    return [Paragraph(i, S[style], bulletText="•") for i in items]


def numbered(items):
    return [Paragraph(i, S["bullet"], bulletText=f"{n}.") for n, i in enumerate(items, 1)]


def table(rows, widths, header=True, zebra=True, font_size=None):
    data = []
    for r_i, row in enumerate(rows):
        style = "cellb" if header and r_i == 0 else "cell"
        data.append([c if not isinstance(c, str) else Paragraph(c, S[style]) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    ts = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.4, LINE),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        ts.append(("BACKGROUND", (0, 0), (-1, 0), INDIGO))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                ts.append(("BACKGROUND", (0, i), (-1, i), GREY_BG))
    t.setStyle(TableStyle(ts))
    return t


def callout(title, body, bg=SOFT, edge=INDIGO):
    content = [Paragraph(f"<b>{title}</b>", ParagraphStyle("ct", parent=S["body"], textColor=edge, spaceAfter=3))]
    if isinstance(body, str):
        content.append(Paragraph(body, S["body"]))
    else:
        content.extend(body)
    t = Table([[content]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return KeepTogether([t, Spacer(1, 8)])


def say(text):
    return callout("What to say in the presentation", text, bg=GREEN_BG, edge=GREEN)


def code(text):
    import textwrap
    text = "\n".join(textwrap.fill(l, 104, subsequent_indent="    ") if len(l) > 104 else l
                     for l in text.split("\n"))
    t = Table([[Preformatted(text, S["code"])]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), GREY_BG), ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return KeepTogether([t, Spacer(1, 8)])


