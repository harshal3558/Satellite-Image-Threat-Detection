"""
Convert HLD.md and LLD.md to styled PDF files.
Uses: Markdown (already installed), fpdf2 (just installed)
"""

import re
import markdown
from fpdf import FPDF

# ── Styling constants ────────────────────────────────────────────────────────
MARGIN       = 15
PAGE_W       = 210   # A4 width mm
CONTENT_W    = PAGE_W - 2 * MARGIN

# Colours
COLOR_BG        = (15,  23,  42)   # dark navy background
COLOR_HEADER_BG = (30,  41,  59)   # header strip
COLOR_ACCENT    = (56, 189, 248)   # sky-blue accent (headings)
COLOR_TEXT      = (226, 232, 240)  # light text
COLOR_MUTED     = (148, 163, 184)  # secondary text
COLOR_CODE_BG   = (30,  41,  59)   # code block background
COLOR_CODE_TEXT = (134, 239, 172)  # green code text
COLOR_TABLE_HDR = (51,  65,  85)   # table header
COLOR_TABLE_ROW = (30,  41,  59)   # table row
COLOR_TABLE_ALT = (22,  33,  55)   # alternating table row
COLOR_DIVIDER   = (56, 189, 248)   # horizontal rule


class DesignPDF(FPDF):
    def __init__(self, title):
        super().__init__()
        self.doc_title = title
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        # Dark header bar
        self.set_fill_color(*COLOR_HEADER_BG)
        self.rect(0, 0, 210, 12, 'F')
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(*COLOR_ACCENT)
        self.set_xy(MARGIN, 3)
        self.cell(0, 6, self.doc_title, align='L')
        self.set_text_color(*COLOR_MUTED)
        self.set_xy(-50, 3)
        self.cell(35, 6, f'Page {self.page_no()}', align='R')

    def footer(self):
        self.set_y(-12)
        self.set_fill_color(*COLOR_HEADER_BG)
        self.rect(0, self.get_y(), 210, 15, 'F')
        self.set_font('Helvetica', '', 7)
        self.set_text_color(*COLOR_MUTED)
        self.set_x(MARGIN)
        self.cell(0, 8, 'Satellite Image Threat Detection (SITP)', align='C')

    def cover_page(self, title, subtitle):
        self.add_page()
        # Full dark background
        self.set_fill_color(*COLOR_BG)
        self.rect(0, 0, 210, 297, 'F')

        # Top accent bar
        self.set_fill_color(*COLOR_ACCENT)
        self.rect(0, 40, 210, 3, 'F')

        # Decorative side bar
        self.set_fill_color(*COLOR_ACCENT)
        self.rect(MARGIN, 55, 2, 100, 'F')

        # Title
        self.set_font('Helvetica', 'B', 32)
        self.set_text_color(*COLOR_ACCENT)
        self.set_xy(MARGIN + 8, 60)
        self.multi_cell(CONTENT_W - 8, 14, title, align='L')

        # Subtitle
        self.set_font('Helvetica', '', 14)
        self.set_text_color(*COLOR_TEXT)
        self.set_xy(MARGIN + 8, self.get_y() + 6)
        self.multi_cell(CONTENT_W - 8, 8, subtitle, align='L')

        # Divider
        self.set_fill_color(*COLOR_ACCENT)
        self.rect(MARGIN + 8, self.get_y() + 8, CONTENT_W - 8, 1, 'F')

        # Footer info on cover
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*COLOR_MUTED)
        self.set_xy(MARGIN + 8, 220)
        self.cell(0, 8, 'Project: Satellite Image Threat Detection (SITP)', align='L')
        self.set_xy(MARGIN + 8, 230)
        self.cell(0, 8, 'GitHub: github.com/harshal3558/Satellite-Image-Threat-Detection', align='L')

        # Bottom accent bar
        self.set_fill_color(*COLOR_ACCENT)
        self.rect(0, 254, 210, 3, 'F')


def render_md_to_pdf(md_path: str, pdf_path: str, doc_title: str, subtitle: str):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pdf = DesignPDF(doc_title)
    pdf.cover_page(doc_title, subtitle)
    pdf.add_page()

    # Dark background for content pages
    # (We'll draw it per cell since fpdf doesn't support full-page background per se)

    lines = content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]

        # ── H1 ──────────────────────────────────────────────────────────────
        if line.startswith('# ') and not line.startswith('## '):
            text = line[2:].strip()
            pdf.ln(4)
            pdf.set_fill_color(*COLOR_ACCENT)
            pdf.set_font('Helvetica', 'B', 16)
            pdf.set_text_color(*COLOR_BG)
            pdf.set_x(MARGIN)
            pdf.cell(CONTENT_W, 10, text, fill=True, ln=True)
            pdf.ln(3)
            i += 1

        # ── H2 ──────────────────────────────────────────────────────────────
        elif line.startswith('## '):
            text = line[3:].strip()
            pdf.ln(5)
            pdf.set_fill_color(*COLOR_HEADER_BG)
            pdf.set_font('Helvetica', 'B', 13)
            pdf.set_text_color(*COLOR_ACCENT)
            pdf.set_x(MARGIN)
            # Accent left border
            pdf.set_fill_color(*COLOR_ACCENT)
            pdf.rect(MARGIN, pdf.get_y(), 2, 8, 'F')
            pdf.set_xy(MARGIN + 4, pdf.get_y())
            pdf.cell(CONTENT_W - 4, 8, text, ln=True)
            pdf.ln(2)
            i += 1

        # ── H3 ──────────────────────────────────────────────────────────────
        elif line.startswith('### '):
            text = line[4:].strip()
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(*COLOR_TEXT)
            pdf.set_x(MARGIN)
            pdf.cell(CONTENT_W, 7, text, ln=True)
            # Underline
            pdf.set_draw_color(*COLOR_ACCENT)
            pdf.set_line_width(0.3)
            pdf.line(MARGIN, pdf.get_y(), MARGIN + CONTENT_W, pdf.get_y())
            pdf.ln(2)
            i += 1

        # ── H4 ──────────────────────────────────────────────────────────────
        elif line.startswith('#### '):
            text = line[5:].strip()
            pdf.ln(2)
            pdf.set_font('Helvetica', 'BI', 10)
            pdf.set_text_color(*COLOR_MUTED)
            pdf.set_x(MARGIN)
            pdf.cell(CONTENT_W, 6, text, ln=True)
            pdf.ln(1)
            i += 1

        # ── Horizontal Rule ─────────────────────────────────────────────────
        elif line.strip() in ('---', '***', '___'):
            pdf.ln(3)
            pdf.set_draw_color(*COLOR_DIVIDER)
            pdf.set_line_width(0.5)
            pdf.line(MARGIN, pdf.get_y(), MARGIN + CONTENT_W, pdf.get_y())
            pdf.ln(4)
            i += 1

        # ── Code Block ──────────────────────────────────────────────────────
        elif line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```

            if code_lines:
                pdf.ln(2)
                block_h = len(code_lines) * 4.5 + 6
                pdf.set_fill_color(*COLOR_CODE_BG)
                # Left accent bar for code
                pdf.set_fill_color(*COLOR_ACCENT)
                start_y = pdf.get_y()
                pdf.rect(MARGIN, start_y, 2, min(block_h, 200), 'F')
                pdf.set_fill_color(*COLOR_CODE_BG)
                pdf.rect(MARGIN + 2, start_y, CONTENT_W - 2, min(block_h, 200), 'F')

                pdf.set_font('Courier', '', 7.5)
                pdf.set_text_color(*COLOR_CODE_TEXT)
                pdf.set_xy(MARGIN + 5, start_y + 3)

                for cl in code_lines:
                    if pdf.get_y() > 270:
                        pdf.add_page()
                        pdf.set_font('Courier', '', 7.5)
                        pdf.set_text_color(*COLOR_CODE_TEXT)
                    # Clean up the line for display
                    cl_clean = cl.replace('──', '--').replace('│', '|').replace('├', '+').replace('└', '\\').replace('┌', '+').replace('┐', '+').replace('┘', '+').replace('┤', '+').replace('┬', '+').replace('┴', '+').replace('┼', '+').replace('─', '-').replace('▶', '>').replace('◀', '<')
                    pdf.set_x(MARGIN + 5)
                    pdf.cell(CONTENT_W - 7, 4.5, cl_clean[:110], ln=True)

                pdf.ln(3)

        # ── Table ───────────────────────────────────────────────────────────
        elif line.startswith('|') and '|' in line[1:]:
            table_lines = []
            while i < len(lines) and lines[i].startswith('|'):
                table_lines.append(lines[i])
                i += 1

            if len(table_lines) >= 2:
                pdf.ln(3)
                # Parse header
                header_cols = [c.strip() for c in table_lines[0].split('|') if c.strip()]
                n_cols = len(header_cols)
                if n_cols == 0:
                    continue
                col_w = CONTENT_W / n_cols

                # Header row
                pdf.set_fill_color(*COLOR_TABLE_HDR)
                pdf.set_font('Helvetica', 'B', 8)
                pdf.set_text_color(*COLOR_ACCENT)
                pdf.set_x(MARGIN)
                for col in header_cols:
                    pdf.cell(col_w, 6, col[:30], border=0, fill=True, align='C')
                pdf.ln()

                # Data rows (skip separator row index 1)
                row_fill = [COLOR_TABLE_ROW, COLOR_TABLE_ALT]
                data_rows = [r for r in table_lines[2:] if '---' not in r]
                for ridx, row in enumerate(data_rows):
                    cols = [c.strip() for c in row.split('|') if c.strip()]
                    pdf.set_fill_color(*row_fill[ridx % 2])
                    pdf.set_font('Helvetica', '', 7.5)
                    pdf.set_text_color(*COLOR_TEXT)
                    pdf.set_x(MARGIN)
                    for ci in range(n_cols):
                        cell_text = cols[ci] if ci < len(cols) else ''
                        # Strip markdown bold/code from cells
                        cell_text = re.sub(r'\*\*(.+?)\*\*', r'\1', cell_text)
                        cell_text = re.sub(r'`(.+?)`', r'\1', cell_text)
                        pdf.cell(col_w, 5.5, cell_text[:35], border=0, fill=True, align='L')
                    pdf.ln()
                pdf.ln(3)

        # ── Bullet point ────────────────────────────────────────────────────
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            indent = len(line) - len(line.lstrip())
            text = line.strip()[2:].strip()
            # Strip markdown formatting
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'`(.+?)`', r'\1', text)
            text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(*COLOR_TEXT)
            bullet_x = MARGIN + (indent // 2) * 4
            pdf.set_x(bullet_x)
            # Bullet dot
            pdf.set_fill_color(*COLOR_ACCENT)
            pdf.circle(bullet_x + 1.5, pdf.get_y() + 2.5, 0.8, 'F')
            pdf.set_xy(bullet_x + 5, pdf.get_y())
            pdf.multi_cell(CONTENT_W - (bullet_x - MARGIN) - 5, 5, text)
            i += 1

        # ── Numbered list ───────────────────────────────────────────────────
        elif re.match(r'^\d+\. ', line.strip()):
            text = re.sub(r'^\d+\. ', '', line.strip())
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'`(.+?)`', r'\1', text)
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(*COLOR_TEXT)
            pdf.set_x(MARGIN + 4)
            num = re.match(r'^(\d+)\.', line.strip()).group(1)
            pdf.set_x(MARGIN)
            pdf.cell(8, 5, f'{num}.', align='R')
            pdf.set_x(MARGIN + 10)
            pdf.multi_cell(CONTENT_W - 10, 5, text)
            i += 1

        # ── Regular paragraph ───────────────────────────────────────────────
        elif line.strip():
            text = line.strip()
            # Strip markdown formatting
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            text = re.sub(r'`(.+?)`', r'\1', text)
            text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
            text = re.sub(r'^#{1,6}\s+', '', text)
            if text:
                pdf.set_font('Helvetica', '', 9)
                pdf.set_text_color(*COLOR_TEXT)
                pdf.set_x(MARGIN)
                pdf.multi_cell(CONTENT_W, 5, text)
                pdf.ln(1)
            i += 1

        else:
            i += 1

    pdf.output(pdf_path)
    print(f"✅ PDF saved: {pdf_path}")


if __name__ == '__main__':
    base = r"c:\Users\harsh\OneDrive\Desktop\Satellite-Image-Threat-Detection"

    render_md_to_pdf(
        md_path  = f"{base}\\HLD.md",
        pdf_path = f"{base}\\HLD.pdf",
        doc_title = "High Level Design (HLD)",
        subtitle  = "Satellite Image Threat Detection — System Architecture & Design"
    )

    render_md_to_pdf(
        md_path  = f"{base}\\LLD.md",
        pdf_path = f"{base}\\LLD.pdf",
        doc_title = "Low Level Design (LLD)",
        subtitle  = "Satellite Image Threat Detection — Detailed Module & Class Design"
    )
