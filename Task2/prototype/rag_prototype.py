import fitz
import re
import json
from pathlib import Path

DOCS_DIR = Path("docs")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDFs")

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text.strip()

HEADING_PATTERN = re.compile(r"^(\d+(\.\d+)*\s+[A-Z].{3,80})$", re.MULTILINE)
SCAN_THRESHOLD = 20

all_pages = []
for pdf_path in pdf_files:
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        headings = [h[0].strip() for h in HEADING_PATTERN.findall(raw_text)]
        all_pages.append({
            "document": pdf_path.name,
            "page": page_num,
            "n_chars": len(raw_text),
            "likely_scanned": len(raw_text) < SCAN_THRESHOLD,
            "text": raw_text,
            "clean_text": clean_text(raw_text),
            "section_headings": headings,
        })
    doc.close()

print(f"Extracted {len(all_pages)} pages total")
n_scanned = sum(p["likely_scanned"] for p in all_pages)
print(f"{n_scanned} pages flagged as likely scanned")

with open(OUT_DIR / "extracted_pages.json", "w") as f:
    json.dump(all_pages, f, indent=2)

# quality report
import csv
quality_rows = []
for pdf_path in pdf_files:
    doc_pages = [p for p in all_pages if p["document"] == pdf_path.name]
    quality_rows.append({
        "document": pdf_path.name,
        "n_pages": len(doc_pages),
        "n_scanned_pages": sum(p["likely_scanned"] for p in doc_pages),
        "avg_chars_per_page": round(sum(p["n_chars"] for p in doc_pages) / max(len(doc_pages), 1)),
        "total_chars": sum(p["n_chars"] for p in doc_pages),
        "n_headings_detected": sum(len(p["section_headings"]) for p in doc_pages),
    })

with open(OUT_DIR / "document_quality_report.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=quality_rows[0].keys())
    writer.writeheader()
    writer.writerows(quality_rows)

for row in quality_rows:
    print(row)