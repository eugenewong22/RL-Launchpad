"""Render docs/writeup.md to a self-contained, print-ready HTML page and
estimate its printed page count against the challenge's 4-page limit.

Usage:  uv run --with markdown python scripts/build_writeup.py

Deliberately run via `--with` rather than added to pyproject: the judge's
repro path is pinned by uv.lock, and a doc-build dependency has nothing to
do with reproducing the results. It must not land in the lock.

Writes docs/writeup.html. Open it and print to PDF with "Default" margins
(the embedded @page rule supplies 0.65in) and background graphics ON, to
produce the submission PDF. Images are inlined as data URIs so the file
stands alone.

Geometry: Letter at 0.65in margins is a 7.2in x 9.7in text box = 691 x 931
CSS px at 96dpi. The body is laid out at exactly 691px wide, so height/931
is a direct page estimate; the page logs it to the browser console on load.
Last measured: 3.92 pages against the 4-page limit.
"""

import argparse
import base64
import mimetypes
import re
from pathlib import Path

import markdown

# 0.65in margins on US Letter, at the CSS reference 96dpi. Tighter than a
# report default because the challenge caps the write-up at 4 pages and the
# content is dense with tables; 9.5pt is normal for a two-column paper and
# perfectly legible in print.
MARGIN_IN = 0.65
PAGE_W_PX = round((8.5 - 2 * MARGIN_IN) * 96)   # 691
PAGE_H_PX = round((11.0 - 2 * MARGIN_IN) * 96)  # 931

CSS = f"""
@page {{ size: Letter; margin: {MARGIN_IN}in; }}
html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{
  width: {PAGE_W_PX}px; margin: 0 auto; padding: 0;
  font: 9.5pt/1.30 "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: #111; background: #fff;
}}
#page-ruler {{ position: absolute; left: 0; top: 0; width: 0; visibility: hidden; }}
h1 {{ font-size: 16pt; margin: 0 0 2pt; line-height: 1.2; }}
h2 {{ font-size: 12pt; margin: 10pt 0 3pt; border-bottom: 1px solid #ccc;
      padding-bottom: 2pt; page-break-after: avoid; }}
h3 {{ font-size: 10.5pt; margin: 8pt 0 3pt; page-break-after: avoid; }}
p {{ margin: 0 0 4pt; text-align: justify; }}
ul, ol {{ margin: 0 0 4pt; padding-left: 16pt; }}
li {{ margin-bottom: 2pt; }}
em {{ color: #444; }}
code {{ font: 8.7pt/1.3 "SF Mono", Menlo, Consolas, monospace; background: #f2f3f5;
        padding: 0 2px; border-radius: 2px; }}
pre {{ background: #f7f8fa; border: 1px solid #e2e4e8; border-radius: 3px;
       padding: 5pt 7pt; overflow-x: auto; page-break-inside: avoid;
       margin: 0 0 6pt; }}
pre code {{ background: none; padding: 0; font-size: 8.2pt; line-height: 1.28; }}
/* Rows stay intact; tables may break *between* rows. Table-level avoid
   would shove a 380px table whole onto the next page and waste the
   remainder of this one. */
table {{ border-collapse: collapse; width: 100%; margin: 0 0 6pt;
         font-size: 8.3pt; }}
tr {{ page-break-inside: avoid; }}
thead {{ display: table-header-group; }}
th, td {{ border: 1px solid #d6d9de; padding: 2pt 3.5pt; text-align: left;
          vertical-align: top; }}
th {{ background: #f2f3f5; font-weight: 600; }}
img {{ max-width: 100%; display: block; margin: 4pt auto 6pt;
       page-break-inside: avoid; }}
blockquote {{ margin: 0 0 5pt; padding-left: 8pt; border-left: 2px solid #ccc;
              color: #444; }}
hr {{ border: 0; border-top: 1px solid #ccc; margin: 8pt 0; }}
"""


def inline_images(html: str, base: Path) -> str:
    """Replace <img src="relative"> with a base64 data URI."""

    def repl(m):
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = (base / src).resolve()
        if not path.exists():
            print(f"  WARNING: image not found, left as-is: {src}")
            return m.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f'<img src="data:{mime};base64,{b64}"'

    return re.sub(r'<img src="([^"]+)"', repl, html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="docs/writeup.md")
    parser.add_argument("--out", default="docs/writeup.html")
    parser.add_argument("--limit", type=int, default=4, help="page limit to check against")
    args = parser.parse_args()

    src = Path(args.src)
    body = markdown.markdown(
        src.read_text(),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body = inline_images(body, src.parent)

    title = "From-Scratch TD3 + HER on Sparse-Reward Fetch Manipulation"
    html = (
        f"<!doctype html>\n<html><head><meta charset='utf-8'>\n"
        f"<title>{title}</title>\n<style>{CSS}</style></head>\n"
        f"<body>\n<div id='page-ruler'></div>\n{body}\n"
        # Reports the measured page count in the page itself, so opening the
        # file answers the question without a separate tool.
        "<script>\n"
        f"window.addEventListener('load', () => {{\n"
        f"  const h = document.body.scrollHeight;\n"
        f"  const pages = h / {PAGE_H_PX};\n"
        f"  console.log('content height ' + h + 'px = ' + pages.toFixed(2) + ' pages');\n"
        f"  window.__pages = pages;\n"
        f"}});\n</script>\n</body></html>\n"
    )

    out = Path(args.out)
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KB, images inlined)")
    print(f'print to PDF: Letter, "Default" margins (@page supplies '
          f'{MARGIN_IN}in), background graphics ON')
    print(f"page count is logged to the browser console; limit is {args.limit}")


if __name__ == "__main__":
    main()
