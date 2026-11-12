
import fitz
import re
import json
import csv
import tiktoken
import faiss
import numpy as np
import pickle

from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"
OUT_DIR = BASE_DIR / "outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SCAN_THRESHOLD = 20

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

BLANK_PAGE_PATTERN = re.compile(
    r"intentionally\s+left\s+blank",
    re.IGNORECASE
)

ENCODING_NAME = "cl100k_base"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Normalize whitespace and repair hyphenated line breaks."""

    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# SECTION HEADING DETECTION
# ============================================================

HEADING_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)*\s+[A-Z].{3,80})\s*$",
    re.MULTILINE
)


def extract_headings(text):
    """Extract numbered section headings from page text."""

    return [
        heading.strip()
        for heading in HEADING_PATTERN.findall(text)
    ]


# ============================================================
# PHASE 7: PDF DOCUMENT EXTRACTION
# ============================================================

def extract_documents():

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in {DOCS_DIR}"
        )

    print(f"Found {len(pdf_files)} PDFs\n")

    all_pages = []

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        try:

            with fitz.open(pdf_path) as doc:

                for page_num, page in enumerate(doc, start=1):

                    raw_text = page.get_text("text")

                    headings = extract_headings(raw_text)

                    all_pages.append({

                        "document": pdf_path.name,

                        "page": page_num,

                        "n_chars": len(raw_text),

                        "likely_scanned": (
                            len(raw_text.strip()) < SCAN_THRESHOLD
                        ),

                        "text": raw_text,

                        "clean_text": clean_text(raw_text),

                        "section_headings": headings
                    })

        except Exception as exc:

            print(
                f"ERROR processing {pdf_path.name}: {exc}"
            )

    if not all_pages:
        raise RuntimeError(
            "No pages could be extracted from the PDF documents."
        )

    print(f"\nExtracted {len(all_pages)} pages total")

    scanned_count = sum(
        page["likely_scanned"] for page in all_pages
    )

    print(
        f"{scanned_count} pages flagged as likely scanned"
    )

    output_file = OUT_DIR / "extracted_pages.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_pages,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Saved extracted pages: {output_file}")

    return all_pages, pdf_files


# ============================================================
# PHASE 7: DOCUMENT QUALITY REPORT
# ============================================================

def generate_quality_report(all_pages, pdf_files):

    quality_rows = []

    for pdf_path in pdf_files:

        doc_pages = [
            page for page in all_pages
            if page["document"] == pdf_path.name
        ]

        total_pages = len(doc_pages)

        quality_rows.append({

            "document": pdf_path.name,

            "n_pages": total_pages,

            "n_scanned_pages": sum(
                page["likely_scanned"] for page in doc_pages
            ),

            "avg_chars_per_page": round(
                sum(page["n_chars"] for page in doc_pages)
                / max(total_pages, 1)
            ),

            "total_chars": sum(
                page["n_chars"] for page in doc_pages
            ),

            "n_headings_detected": sum(
                len(page["section_headings"])
                for page in doc_pages
            )
        })

    output_file = OUT_DIR / "document_quality_report.csv"

    fieldnames = [
        "document",
        "n_pages",
        "n_scanned_pages",
        "avg_chars_per_page",
        "total_chars",
        "n_headings_detected"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(quality_rows)

    print("\nDOCUMENT QUALITY REPORT")

    for row in quality_rows:
        print(row)

    print(
        f"\nSaved quality report: {output_file}"
    )


# ============================================================
# PHASE 8: TOKEN-BASED TEXT CHUNKING
# ============================================================

def chunk_text(text, encoding):

    tokens = encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + CHUNK_SIZE

        chunk_tokens = tokens[start:end]

        chunk = encoding.decode(chunk_tokens)

        if len(chunk.strip()) >= 30:
            chunks.append(chunk)

        if end >= len(tokens):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# PHASE 8: DOCUMENT CHUNK GENERATION
# ============================================================


def generate_chunks(all_pages):

    encoding = tiktoken.get_encoding(
        ENCODING_NAME
    )

    all_chunks = []

    for page in all_pages:

        text = page["clean_text"]

        # Skip empty or very short pages
        if not text or len(text) < 30:
            continue

        # Skip pages containing blank-page notices
        if BLANK_PAGE_PATTERN.search(text):
            continue

        text_chunks = chunk_text(
            text,
            encoding
        )

        for chunk_index, chunk in enumerate(text_chunks):

            # Skip short or insignificant chunks
            if len(chunk.strip()) < 30:
                continue

            all_chunks.append({

                "chunk_id": (
                    f"{page['document']}::"
                    f"p{page['page']}::"
                    f"c{chunk_index}"
                ),

                "document": page["document"],

                "page": page["page"],

                "section_headings": page["section_headings"],

                "text": chunk,

                "token_count": len(
                    encoding.encode(chunk)
                )
            })

    print(
        f"\nTotal chunks generated: {len(all_chunks)}"
    )

    return all_chunks



# ============================================================
# PHASE 8: SAVE CHUNK OUTPUT
# ============================================================

def save_chunks(all_chunks):

    output_file = OUT_DIR / "document_chunks.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved document chunks: {output_file}"
    )


# ============================================================
# PHASE 8: BUILD EMBEDDINGS, FAISS AND BM25 INDEXES
# ============================================================

def build_indexes(all_chunks):

    if not all_chunks:
        raise ValueError(
            "No chunks available for indexing."
        )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    texts = [
        chunk["text"] for chunk in all_chunks
    ]

    # Generate dense embeddings

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Build FAISS index

    print("\nBuilding FAISS index...")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        str(OUT_DIR / "faiss_index.bin")
    )

    # Build BM25 keyword index

    print("\nBuilding BM25 index...")

    tokenized = [
        text.lower().split()
        for text in texts
    ]

    bm25 = BM25Okapi(tokenized)

    with open(
        OUT_DIR / "bm25_index.pkl",
        "wb"
    ) as f:

        pickle.dump(
            {
                "bm25": bm25,
                "tokenized_corpus": tokenized
            },
            f
        )

    # Save chunk metadata

    with open(
        OUT_DIR / "chunk_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nBuilt FAISS index: "
        f"{embeddings.shape[0]} vectors, "
        f"dim={embeddings.shape[1]}"
    )

    print(
        f"Built BM25 index: {len(tokenized)} documents"
    )

    return index, bm25, model


# ============================================================
# PHASE 8: DENSE RETRIEVAL SANITY TEST
# ============================================================

def test_retrieval(index, model, all_chunks, query, top_k=3):

    query_vec = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_vec,
        min(top_k, len(all_chunks))
    )

    print(f"\nQuery: {query!r}")

    for rank, (idx, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1
    ):

        if idx < 0:
            continue

        chunk = all_chunks[idx]

        snippet = (
            chunk["text"][:200]
            .replace("\n", " ")
        )

        print(
            f"  #{rank} (score={score:.3f}) "
            f"[{chunk['document']} p.{chunk['page']}]"
        )

        print(f"      {snippet}...")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)

    print(
        "CYCLONE TECHNICAL DOCUMENT INGESTION "
        "AND INDEXING PIPELINE"
    )

    print("=" * 65)

    # Phase 7: PDF extraction

    all_pages, pdf_files = extract_documents()

    # Phase 7: Document quality assessment

    generate_quality_report(
        all_pages,
        pdf_files
    )

    # Phase 8: Text chunking

    all_chunks = generate_chunks(
        all_pages
    )

    # Phase 8: Save chunks

    save_chunks(
        all_chunks
    )

    # Phase 8: Embeddings and indexing

    index, bm25, model = build_indexes(
        all_chunks
    )

    # Final summary

    print("\n" + "=" * 65)

    print(
        "DOCUMENT INGESTION AND INDEXING COMPLETED"
    )

    print("=" * 65)

    print(
        f"PDF documents processed: {len(pdf_files)}"
    )

    print(
        f"Pages extracted: {len(all_pages)}"
    )

    print(
        "Pages flagged as likely scanned:",
        sum(
            page["likely_scanned"]
            for page in all_pages
        )
    )

    print(
        f"Total chunks generated: {len(all_chunks)}"
    )

    print(
        f"FAISS vectors indexed: {index.ntotal}"
    )

    print(
        f"Embedding dimensions: {index.d}"
    )

    print(
        f"BM25 documents indexed: {len(all_chunks)}"
    )

    print(
        "\nOutput directory:", OUT_DIR
    )

    # ========================================================
    # DENSE RETRIEVAL SANITY TESTS
    # ========================================================

    print("\n" + "=" * 65)

    print("DENSE RETRIEVAL SANITY TEST")

    print("=" * 65)

    test_retrieval(
        index,
        model,
        all_chunks,
        "What does a sudden draft drop indicate?"
    )

    test_retrieval(
        index,
        model,
        all_chunks,
        "How do I start up the cyclone separator?"
    )

    test_retrieval(
        index,
        model,
        all_chunks,
        "What personal protective equipment is required?"
    )


if __name__ == "__main__":
    main()
