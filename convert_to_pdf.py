"""
Publication-Quality Markdown to PDF Converter for Medical/Scientific Papers.

Features:
1. Pre-renders all LaTeX math (inline $...$ and display $$...$$) to native MathML.
   - Works natively in Chromium/Edge without any client-side JavaScript execution.
2. Automatically parses monospaced ASCII tables into clean, responsive HTML <table> elements
   with journal-grade styling (JAMA/Lancet/NEJM clean-rule aesthetic).
3. Formats ASCII pipeline and workflow diagrams with precise monospace alignment.
4. Generates PDF via headless Edge/Chrome using --no-pdf-header-footer to eliminate file path footers.
5. Implements academic typography, pagination controls (widows, orphans, page-break rules).
"""

import subprocess
import sys
import os
import re
import html

def install_if_missing(pkg, imp=None):
    try:
        __import__(imp or pkg)
    except ImportError:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

install_if_missing("markdown")
install_if_missing("pymdown-extensions", "pymdownx")
install_if_missing("latex2mathml")

import markdown
import latex2mathml.converter


# ─── Math Pre-Rendering to MathML ─────────────────────────────────────────────

def render_latex_to_mathml(latex_str, display=False):
    """Convert LaTeX math expression to native MathML."""
    latex_str = latex_str.strip()
    if not latex_str:
        return ""
    
    # Normalize common abbreviations and cleanups
    # If there are any stray double-escapes, normalize
    cleaned = latex_str.replace(r'\\', r'\temp_bs_break')
    cleaned = cleaned.replace(r'\%', '%')
    cleaned = cleaned.replace(r'\temp_bs_break', r'\\')
    
    try:
        mathml = latex2mathml.converter.convert(cleaned)
        if display:
            mathml = mathml.replace('<math', '<math display="block"', 1)
            return f'<div class="equation-block">{mathml}</div>'
        else:
            return mathml
    except Exception:
        # Fallback to nicely styled span
        esc = html.escape(latex_str)
        if display:
            return f'<div class="equation-block math-fallback"><em>{esc}</em></div>'
        return f'<span class="math-fallback"><em>{esc}</em></span>'


def prerender_math(md_text):
    """Replace all LaTeX math delimiters with MathML."""
    # 1. Display math $$...$$
    def replace_display(m):
        return render_latex_to_mathml(m.group(1), display=True)
    md_text = re.sub(r'\$\$(.+?)\$\$', replace_display, md_text, flags=re.DOTALL)

    # 2. Display math \[ ... \]
    def replace_bracket(m):
        return render_latex_to_mathml(m.group(1), display=True)
    md_text = re.sub(r'\\\[(.+?)\\\]', replace_bracket, md_text, flags=re.DOTALL)

    # 3. Inline math \( ... \)
    def replace_paren(m):
        return render_latex_to_mathml(m.group(1), display=False)
    md_text = re.sub(r'\\\((.+?)\\\)', replace_paren, md_text)

    # 4. Inline math $...$
    def replace_inline(m):
        content = m.group(1).strip()
        return render_latex_to_mathml(content, display=False)
    md_text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', replace_inline, md_text)

    return md_text


# ─── ASCII Table to Clean HTML Table Converter ───────────────────────────────

def parse_ascii_table(block):
    """
    Transform ASCII tables (delimited with === and ---) into HTML <table> structures.
    """
    raw_lines = [l.rstrip() for l in block.strip().split('\n')]
    if len(raw_lines) < 3:
        return None

    separator_pattern = re.compile(r'^[\s=\-+|]{5,}$')
    
    title = ""
    data_lines = []
    note_lines = []
    in_note = False

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if separator_pattern.match(stripped):
            continue
        if re.match(r'^TABLE\s+\d+[:\.]?', stripped, re.IGNORECASE):
            title = stripped
            continue
        if stripped.startswith('*') or stripped.startswith('†') or stripped.lower().startswith('note'):
            in_note = True
        if in_note:
            note_lines.append(stripped)
        else:
            data_lines.append(stripped)

    if len(data_lines) < 2:
        return None

    def split_columns(line):
        parts = re.split(r'  {2,}', line.strip())
        return [p.strip() for p in parts if p.strip()]

    header = split_columns(data_lines[0])
    if len(header) < 2:
        return None

    rows = []
    for line in data_lines[1:]:
        cols = split_columns(line)
        if len(cols) == 1 and not any(c.isdigit() for c in cols[0][:5]):
            # Section header inside table
            rows.append(("section", cols[0]))
        else:
            rows.append(("data", cols))

    # Build clean HTML table
    out = []
    if title:
        out.append(f'<div class="table-caption"><strong>{html.escape(title)}</strong></div>')
    
    out.append('<div class="table-wrapper"><table class="academic-table">')
    out.append('<thead><tr>')
    for h in header:
        out.append(f'<th>{html.escape(h)}</th>')
    out.append('</tr></thead><tbody>')

    for rtype, rdata in rows:
        if rtype == "section":
            out.append(f'<tr class="section-heading"><td colspan="{len(header)}"><strong>{html.escape(rdata)}</strong></td></tr>')
        else:
            out.append('<tr>')
            for i, cell in enumerate(rdata):
                out.append(f'<td>{html.escape(cell)}</td>')
            for _ in range(len(header) - len(rdata)):
                out.append('<td></td>')
            out.append('</tr>')

    out.append('</tbody></table></div>')

    if note_lines:
        note_text = ' '.join(note_lines)
        out.append(f'<div class="table-footnotes">{html.escape(note_text)}</div>')

    return '\n'.join(out)


def is_table_block(content):
    lines = content.strip().split('\n')
    sep_pattern = re.compile(r'^[\s=\-+|]{10,}$')
    has_table_marker = any(re.match(r'^\s*TABLE\s+\d+', l, re.IGNORECASE) for l in lines)
    has_separators = sum(1 for l in lines if sep_pattern.match(l.strip())) >= 2
    return has_table_marker and has_separators


def is_ascii_diagram(content):
    has_box_chars = bool(re.search(r'[┌┐└┘├┤│─▼▶►]', content))
    has_box_border = bool(re.search(r'^\s*\+[-=]+\+', content, re.MULTILINE))
    return has_box_chars or has_box_border


def convert_code_blocks(md_text):
    """Convert code blocks: tables to HTML tables, diagrams to styled pre boxes."""
    def process_block(m):
        lang = (m.group(1) or '').strip().lower()
        content = m.group(2)

        if lang == 'mermaid':
            return f'\n<div class="diagram-container"><div class="diagram-badge">Conceptual Architecture / Flowchart</div><pre class="diagram-pre">{html.escape(content)}</pre></div>\n'

        if not lang and is_table_block(content):
            tbl = parse_ascii_table(content)
            if tbl:
                return f'\n{tbl}\n'

        if not lang and is_ascii_diagram(content):
            return f'\n<div class="diagram-container"><pre class="diagram-pre">{html.escape(content)}</pre></div>\n'

        # Regular code
        if lang:
            return f'\n```{lang}\n{content}\n```\n'
        return f'\n```\n{content}\n```\n'

    pattern = re.compile(r'```(\w*)\s*\n(.*?)\n```', re.DOTALL)
    return pattern.sub(process_block, md_text)


# ─── PDF Generation ───────────────────────────────────────────────────────────

def convert_md_to_pdf(md_path, pdf_path):
    print("Reading PAPER.md...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Step 1: Pre-render math to MathML
    print("Pre-rendering LaTeX math to MathML...")
    md_content = prerender_math(md_content)

    # Step 2: Convert ASCII tables
    print("Transforming ASCII tables to journal-grade HTML tables...")
    md_content = convert_code_blocks(md_content)

    # Step 3: Markdown to HTML
    print("Converting Markdown to HTML structure...")
    extensions = [
        'tables', 'fenced_code', 'toc', 'md_in_html', 'attr_list',
        'def_list', 'footnotes', 'pymdownx.superfences'
    ]
    md = markdown.Markdown(extensions=extensions, extension_configs={'toc': {'permalink': False}})
    html_body = md.convert(md_content)

    # Step 4: Academic Publication CSS
    css = """
    @page {
        size: letter;
        margin: 1in;
    }
    * { box-sizing: border-box; }

    body {
        font-family: 'Times New Roman', 'Cambria', 'Georgia', serif;
        font-size: 14pt;
        line-height: 1.42;
        color: #1a1a1a;
        max-width: 100%;
        text-align: justify;
        hyphens: auto;
        -webkit-hyphens: auto;
    }

    /* ── Typography & Headings ── */
    h1 {
        font-size: 22pt;
        font-weight: 700;
        text-align: center;
        margin: 0 0 16pt 0;
        line-height: 1.3;
        color: #0b2545;
    }
    h2 {
        font-size: 16pt;
        font-weight: 700;
        margin-top: 24pt;
        margin-bottom: 10pt;
        color: #0b2545;
        border-bottom: 1.5pt solid #0b2545;
        padding-bottom: 4pt;
        page-break-after: avoid;
    }
    h3 {
        font-size: 16pt;
        font-weight: 700;
        margin-top: 18pt;
        margin-bottom: 8pt;
        color: #133c55;
        page-break-after: avoid;
    }
    h4 {
        font-size: 15pt;
        font-weight: 700;
        font-style: italic;
        margin-top: 14pt;
        margin-bottom: 6pt;
        color: #1b3a4b;
        page-break-after: avoid;
    }

    p {
        margin-top: 0;
        margin-bottom: 7.5pt;
        orphans: 3;
        widows: 3;
    }

    blockquote {
        margin: 12pt 18pt;
        padding: 10pt 16pt;
        border-left: 3.5pt solid #133c55;
        background: #f4f7fa;
        font-style: italic;
        font-size: 13.5pt;
        color: #2c3e50;
        page-break-inside: avoid;
    }

    /* ── Journal-Grade Tables ── */
    .table-caption {
        margin-top: 18pt;
        margin-bottom: 6pt;
        font-size: 11pt;
        font-weight: 700;
        color: #0b2545;
        page-break-after: avoid !important;
        break-after: avoid !important;
    }
    .table-wrapper {
        width: 100%;
        overflow-x: auto;
        margin-top: 10pt;
        margin-bottom: 14pt;
        page-break-inside: avoid !important;
        break-inside: avoid !important;
    }
    .academic-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 8.5pt;
        line-height: 1.32;
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        border-top: 1.5pt solid #0b2545;
        border-bottom: 1.5pt solid #0b2545;
    }
    .academic-table thead th {
        background-color: #0b2545;
        color: #ffffff;
        font-weight: 700;
        padding: 5pt 6pt;
        text-align: left;
        border-bottom: 1pt solid #0b2545;
        font-size: 8.5pt;
        white-space: nowrap;
    }
    .academic-table tbody td {
        padding: 4pt 6pt;
        border-bottom: 0.5pt solid #e0e0e0;
        vertical-align: top;
        font-size: 8.5pt;
    }
    .academic-table tbody tr {
        page-break-inside: avoid !important;
        break-inside: avoid !important;
    }
    .academic-table tbody tr:nth-child(even) {
        background-color: #f8fafc;
    }
    .academic-table .section-heading td {
        background-color: #e2e8f0;
        font-weight: 700;
        font-size: 9pt;
        color: #0b2545;
        padding: 4.5pt 6pt;
        border-top: 1pt solid #cbd5e1;
        border-bottom: 1pt solid #cbd5e1;
        page-break-after: avoid !important;
        break-after: avoid !important;
    }
    .table-footnotes {
        font-size: 7.5pt;
        color: #475569;
        margin: 4pt 0 12pt 0;
        font-style: italic;
        line-height: 1.3;
        page-break-before: avoid !important;
        break-before: avoid !important;
    }

    /* ── Diagrams & Architecture Boxes ── */
    .diagram-container {
        margin: 14pt 0;
        padding: 10pt 12pt;
        background: #f8fafc;
        border: 0.8pt solid #cbd5e1;
        border-radius: 3pt;
        page-break-inside: avoid;
    }
    .diagram-badge {
        font-size: 8pt;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5pt;
        color: #64748b;
        margin-bottom: 5pt;
    }
    .diagram-pre {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 7.5pt;
        line-height: 1.28;
        white-space: pre;
        margin: 0;
        padding: 0;
        background: transparent;
        border: none;
        color: #1e293b;
        overflow-x: auto;
    }

    /* ── Math & Equations ── */
    math {
        font-family: 'Cambria Math', 'Times New Roman', serif;
        font-size: 14pt;
    }
    .equation-block {
        text-align: center;
        margin: 12pt 0;
        padding: 6pt 0;
        page-break-inside: avoid;
    }
    .equation-block math {
        font-size: 15pt;
    }
    .math-fallback {
        font-family: 'Cambria Math', 'Times New Roman', serif;
        font-style: italic;
        font-size: 14pt;
    }

    /* ── Standard Code & Lists ── */
    pre:not(.diagram-pre) {
        background: #f1f5f9;
        border: 0.5pt solid #cbd5e1;
        border-radius: 2pt;
        padding: 6pt 8pt;
        font-size: 8pt;
        line-height: 1.3;
        font-family: 'Consolas', monospace;
        white-space: pre-wrap;
        page-break-inside: avoid;
    }
    code {
        font-family: 'Consolas', monospace;
        font-size: 8.5pt;
        background: #f1f5f9;
        padding: 1pt 3pt;
        border-radius: 2pt;
        color: #0f172a;
    }
    ul, ol {
        margin: 3pt 0 6pt 0;
        padding-left: 20pt;
    }
    li {
        margin-bottom: 2pt;
    }
    hr {
        border: none;
        border-top: 0.8pt solid #cbd5e1;
        margin: 14pt 0;
    }
    a {
        color: #133c55;
        text-decoration: none;
    }

    @media print {
        body {
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }
        .academic-table thead th {
            background-color: #0b2545 !important;
            color: #ffffff !important;
        }
        .academic-table tbody tr:nth-child(even) {
            background-color: #f8fafc !important;
        }
        .academic-table .section-heading td {
            background-color: #e2e8f0 !important;
        }
    }
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Predicting Ultrasound-Detected Gallstones from Non-Imaging Clinical Data</title>
<style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    html_path = md_path.replace(".md", "_pub.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"  HTML saved: {html_path}")

    # Also save standard PAPER.html
    std_html_path = os.path.join(os.path.dirname(md_path), "PAPER.html")
    with open(std_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"  Standard HTML saved: {std_html_path}")

    # Step 5: Render PDF via Headless Edge/Chrome
    browsers = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    browser = next((p for p in browsers if os.path.exists(p)), None)

    if not browser:
        print("\n  ERROR: No browser found for PDF rendering.")
        return False

    print(f"  Rendering PDF with {os.path.basename(browser)}...")
    html_url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
    pdf_abs = os.path.abspath(pdf_path)

    result = subprocess.run([
        browser,
        "--headless=new",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=5000",
        f"--print-to-pdf={pdf_abs}",
        "--no-pdf-header-footer",
        html_url
    ], capture_output=True, text=True, timeout=90)

    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        kb = os.path.getsize(pdf_path) / 1024
        print(f"\n  [SUCCESS] PDF saved: {pdf_path} ({kb:.0f} KB)")
        print(f"     * No local file path footers.")
        print(f"     * LaTeX math pre-rendered to native MathML.")
        print(f"     * Tables typeset with professional academic rules.")
        return True
    else:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        print(f"\n  ERROR: PDF generation failed: {stderr}")
        return False


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    md = os.path.join(base, "PAPER.md")
    pdf = os.path.join(base, "PAPER.pdf")

    if not os.path.exists(md):
        print(f"Error: {md} not found!")
        sys.exit(1)

    print("=" * 64)
    print("  PAPER.md -> Publication PDF Converter (v3)")
    print("=" * 64)
    print(f"  Source: {md}")
    print(f"  Output: {pdf}")
    print("-" * 64)

    ok = convert_md_to_pdf(md, pdf)
    sys.exit(0 if ok else 1)
