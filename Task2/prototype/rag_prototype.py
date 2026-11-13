
import fitz
import re
import json
import csv
import tiktoken
import faiss
import numpy as np
import pickle

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
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

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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


def bm25_search(bm25, tokenized_corpus, query, top_k=10):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = np.argsort(scores)[::-1][:top_k]

    return [(idx, scores[idx]) for idx in ranked]


def dense_search(index, model, query, top_k=10):
    query_vec = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_vec, top_k)

    return [
        (idx, score)
        for idx, score in zip(indices[0], scores[0])
        if idx >= 0
    ]


def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    fused_scores = {}

    for rank, (idx, _) in enumerate(dense_results):
        fused_scores[idx] = (
            fused_scores.get(idx, 0) + 1 / (k + rank + 1)
        )

    for rank, (idx, _) in enumerate(bm25_results):
        fused_scores[idx] = (
            fused_scores.get(idx, 0) + 1 / (k + rank + 1)
        )

    return sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

def hybrid_search(
    index,
    bm25,
    tokenized_corpus,
    model,
    reranker,
    all_chunks,
    query,
    top_k=10,
    rerank_top_n=5
):
    # Step 1: Retrieve using dense search
    dense_results = dense_search(
        index, model, query, top_k=top_k
    )

    # Step 2: Retrieve using BM25
    bm25_results = bm25_search(
        bm25, tokenized_corpus, query, top_k=top_k
    )

    # Step 3: Combine rankings using RRF
    fused = reciprocal_rank_fusion(
        dense_results, bm25_results
    )[:top_k]

    # Step 4: Prepare candidate chunks
    candidates = [
        all_chunks[idx]
        for idx, _ in fused
    ]

    # Step 5: Cross-Encoder reranking
    pairs = [
        [query, chunk["text"]]
        for chunk in candidates
    ]

    rerank_scores = reranker.predict(pairs)

    # Step 6: Sort by reranking score
    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return reranked[:rerank_top_n]

def test_hybrid_retrieval(
    index,
    bm25,
    tokenized_corpus,
    model,
    reranker,
    all_chunks,
    query
):
    results = hybrid_search(
        index,
        bm25,
        tokenized_corpus,
        model,
        reranker,
        all_chunks,
        query
    )

    print(f"\nHybrid + Reranked Query: {query!r}")

    for rank, (chunk, score) in enumerate(results, start=1):
        snippet = chunk["text"][:200].replace("\n", " ")

        print(
            f"  #{rank} (rerank_score={score:.3f}) "
            f"[{chunk['document']} p.{chunk['page']}]"
        )
        print(f"      {snippet}...")

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

    # Phase 9: Prepare BM25 corpus

    tokenized_corpus = [
        chunk["text"].lower().split()
        for chunk in all_chunks
    ]

    # Phase 9: Load Cross-Encoder reranker

    print("\nLoading Cross-Encoder reranking model...")

    reranker = CrossEncoder(
        RERANKER_MODEL_NAME
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
    # PHASE 8: DENSE RETRIEVAL SANITY TESTS
    # ========================================================

    print("\n" + "=" * 65)

    print("PHASE 8: DENSE RETRIEVAL SANITY TEST")

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

    # ========================================================
    # PHASE 9: HYBRID RETRIEVAL + RERANKING TESTS
    # ========================================================

    print("\n" + "=" * 65)

    print("PHASE 9: HYBRID RETRIEVAL + RERANKING")

    print("=" * 65)

    test_hybrid_retrieval(
        index,
        bm25,
        tokenized_corpus,
        model,
        reranker,
        all_chunks,
        "What does a sudden draft drop indicate?"
    )

    test_hybrid_retrieval(
        index,
        bm25,
        tokenized_corpus,
        model,
        reranker,
        all_chunks,
        "How do I start up the cyclone separator?"
    )

    test_hybrid_retrieval(
        index,
        bm25,
        tokenized_corpus,
        model,
        reranker,
        all_chunks,
        "What personal protective equipment is required?"
    )
    # ========================================================
    # PHASE 9: RETRIEVAL EVALUATION
    # ========================================================

    print("\n" + "=" * 65)
    print("PHASE 9: RETRIEVAL EVALUATION")
    print("=" * 65)

    run_evaluation(
        index,
        bm25,
        tokenized_corpus,
        model,
        reranker,
        all_chunks
    )

# ============================================================
# PHASE 9: RETRIEVAL EVALUATION DATASET
# ============================================================

EVAL_SET = [
    {
        "query": "How do I start up the cyclone separator?",
        "relevant": [
            ("Cyclone_Emission_Control.pdf", 11),
            ("Cyclone_Emission_Control.pdf", 10),
            ("Parker_DustHog_C_Series_Cyclone_Dust_Collector_User_Guide.pdf", 23),
        ],
    },
    {
        "query": "What personal protective equipment is required?",
        "relevant": [
            ("654b45bb8292f586313470.pdf", 12),
            ("654b45bb8292f586313470.pdf", 13),
        ],
    },
    {
        "query": "What does a sudden draft drop indicate?",
        "relevant": [],
    },
    {
        "query": "What should an operator inspect when the cyclone has reduced suction?",
        "relevant": [
            ("654b45bb8292f586313470.pdf", 17),
        ],
    },
    {
        "query": "How frequently should the cyclone dust container be checked and emptied?",
        "relevant": [
            ("654b45bb8292f586313470.pdf", 17),
        ],
    },
]

# ============================================================
# PHASE 9: RETRIEVAL EVALUATION
# ============================================================

def precision_recall_at_k(results, relevant_set, k=5):
    # Evaluate unique document-page pairs, not duplicate chunks.
    top_k = list(dict.fromkeys(
        (chunk["document"], chunk["page"])
        for chunk, _ in results
    ))[:k]

    hits = [page for page in top_k if page in relevant_set]

    precision = len(hits) / k
    recall = (
        len(set(hits)) / len(relevant_set)
        if relevant_set else None
    )

    return precision, recall


def run_evaluation(
    index,
    bm25,
    tokenized_corpus,
    model,
    reranker,
    all_chunks
):

    eval_rows = []

    for case in EVAL_SET:

        results = hybrid_search(
            index,
            bm25,
            tokenized_corpus,
            model,
            reranker,
            all_chunks,
            case["query"],
            top_k=10,
            rerank_top_n=5
        )

        relevant_set = set(case["relevant"])

        precision, recall = precision_recall_at_k(
            results,
            relevant_set,
            k=5
        )

        top_score = (
            float(results[0][1])
            if results else None
        )

        eval_rows.append({
            "query": case["query"],
            "precision_at_5": precision,
            "recall_at_5": recall,
            "top_rerank_score": top_score,
            "n_relevant_labeled": len(relevant_set)
        })

        print(f"\nQuery: {case['query']}")
        print(f"Precision@5: {precision:.3f}")
        print(
            f"Recall@5: {recall:.3f}"
            if recall is not None
            else "Recall@5: N/A (no relevant pages)"
        )
        print(f"Top rerank score: {top_score}")

    # Save evaluation results

    evaluation_path = OUT_DIR / "evaluation.csv"

    with open(
        evaluation_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=eval_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(eval_rows)

    print(f"\nEvaluation results saved to: {evaluation_path}")

if __name__ == "__main__":
    main()
