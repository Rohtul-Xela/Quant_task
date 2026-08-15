"""
Render reports/report.md to reports/report.pdf.

Not part of run_all.py — the numbers in report.md are hand-curated
analysis (not machine-regenerated from a template), so this is a
one-off rendering step, run after the report text is finalized, not a
pipeline phase.

Uses reportlab's Platypus (Table + Paragraph flowables) directly
rather than an HTML-to-PDF converter: reportlab's Table/Paragraph
combination handles wrapping text inside table cells correctly and
grows row height to fit, which HTML-to-PDF tools tried for this
(xhtml2pdf) got wrong — column content overlapped instead of wrapping.
WeasyPrint would likely render the HTML/CSS correctly but requires a
GTK runtime that isn't available on this machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MD_PATH = PROJECT_ROOT / "reports" / "report.md"
PDF_PATH = PROJECT_ROOT / "reports" / "report.pdf"

NAVY = colors.HexColor("#16324f")
DARK = colors.HexColor("#1a1a1a")
LIGHT_ROW = colors.HexColor("#f4f6f8")

_styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "TitleX", parent=_styles["Title"], fontSize=16, leading=19, spaceAfter=8,
    textColor=colors.HexColor("#111111"),
)
H2_STYLE = ParagraphStyle(
    "H2", parent=_styles["Heading2"], fontSize=11.5, leading=14, spaceBefore=9,
    spaceAfter=4, textColor=NAVY,
)
H3_STYLE = ParagraphStyle(
    "H3", parent=H2_STYLE, fontSize=9.5, spaceBefore=6, spaceAfter=3,
)
BODY_STYLE = ParagraphStyle(
    "BodyX", parent=_styles["Normal"], fontSize=8.6, leading=11.3, spaceAfter=3,
    textColor=DARK, alignment=4,  # justify
)
LI_STYLE = ParagraphStyle("LiX", parent=BODY_STYLE, spaceAfter=3)
CELL_STYLE = ParagraphStyle(
    "Cell", parent=_styles["Normal"], fontSize=6.8, leading=8.4, textColor=DARK,
)
CELL_ID_STYLE = ParagraphStyle(
    "CellId", parent=CELL_STYLE, fontName="Courier", fontSize=6.1, leading=7.6,
)
HEADER_STYLE = ParagraphStyle(
    "Header", parent=CELL_STYLE, fontSize=6.9, leading=8.6, textColor=colors.white,
)

TABLE_WIDTH = 17.4 * cm


def inline_markup(node) -> str:
    """Convert bs4 inline content into reportlab Paragraph markup."""
    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = (
                str(child)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            parts.append(text)
        elif isinstance(child, Tag):
            inner = inline_markup(child)
            if child.name == "strong":
                parts.append(f"<b>{inner}</b>")
            elif child.name == "em":
                parts.append(f"<i>{inner}</i>")
            elif child.name == "code":
                parts.append(f'<font face="Courier" size="7.6">{inner}</font>')
            else:
                parts.append(inner)
    return "".join(parts)


def build_table(table_tag) -> Table:
    rows = table_tag.find_all("tr")
    header_cells = rows[0].find_all(["th", "td"])

    data = [[Paragraph(inline_markup(c), HEADER_STYLE) for c in header_cells]]
    col_max_len = [len(c.get_text()) for c in header_cells]

    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        out_row = []
        for i, c in enumerate(cells):
            style = CELL_ID_STYLE if i == 0 else CELL_STYLE
            out_row.append(Paragraph(inline_markup(c), style))
            col_max_len[i] = max(col_max_len[i], len(c.get_text()))
        data.append(out_row)

    # Column widths proportional to observed max content length, with a
    # floor so no column collapses to unreadable width.
    weights = [max(w, 6) for w in col_max_len]
    col_widths = [TABLE_WIDTH * w / sum(weights) for w in weights]
    min_w = TABLE_WIDTH * 0.09
    col_widths = [max(w, min_w) for w in col_widths]
    scale = TABLE_WIDTH / sum(col_widths)
    col_widths = [w * scale for w in col_widths]

    t = Table(data, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, r), (-1, r), LIGHT_ROW))
    t.setStyle(TableStyle(style_cmds))

    return t


def build_story(soup: BeautifulSoup) -> list:
    story = []
    body = soup.body if soup.body else soup

    for el in body.find_all(recursive=False):
        if el.name == "h1":
            story.append(Paragraph(inline_markup(el), TITLE_STYLE))
        elif el.name == "h2":
            story.append(Paragraph(inline_markup(el), H2_STYLE))
        elif el.name == "h3":
            story.append(Paragraph(inline_markup(el), H3_STYLE))
        elif el.name == "p":
            story.append(Paragraph(inline_markup(el), BODY_STYLE))
        elif el.name in ("ul", "ol"):
            items = [
                ListItem(Paragraph(inline_markup(li), LI_STYLE), leftIndent=10)
                for li in el.find_all("li", recursive=False)
            ]
            story.append(
                ListFlowable(
                    items, bulletType="bullet", start="•", leftIndent=12, spaceAfter=4
                )
            )
        elif el.name == "table":
            story.append(build_table(el))
            story.append(Spacer(1, 6))
        elif el.name == "hr":
            story.append(Spacer(1, 4))
            story.append(
                HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"))
            )
            story.append(Spacer(1, 4))
        elif el.name == "blockquote":
            for child in el.find_all("p", recursive=False):
                story.append(Paragraph(inline_markup(child), BODY_STYLE))

    return story


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")

    html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    soup = BeautifulSoup(html, "html.parser")

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.6 * cm,
        title="Stockhunt Test Task Report",
    )
    doc.build(build_story(soup))

    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
