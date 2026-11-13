# Task 2 — RAG-Based Technical Knowledge System

## Objective

Build a Retrieval-Augmented Generation (RAG) prototype that answers questions from technical PDF documents with relevant citations.

## Technology Stack

- PDF extraction: PyMuPDF
- Embeddings: Sentence Transformers
- Vector search: FAISS
- Keyword search: BM25
- Hybrid retrieval: Reciprocal Rank Fusion (RRF)
- Reranking: Cross-Encoder
- Local LLM: Ollama
- Backend: FastAPI
- Frontend: Streamlit

## Implementation Workflow

1. PDF ingestion and text extraction
2. Document cleaning and metadata extraction
3. Page-based document chunking with heading metadata
4. Embedding generation and indexing
5. Hybrid retrieval using FAISS and BM25
6. Cross-encoder reranking
7. LLM-based answer generation
8. Citation validation and guardrails
9. FastAPI and Streamlit integration
10. Retrieval and answer-quality evaluation

## Environment Setup

```bash
source Task2/env2/bin/activate
pip install -r Task2/requirements.txt
pip install fastapi uvicorn streamlit
```

Deactivate:

```bash
deactivate
```

## Directory Structure

```
Task2/
├── architecture_diagram.pptx   — System architecture diagram
├── requirements.txt
└── prototype/
    ├── docs/                   — Sample technical PDF documents
    ├── outputs/                — Generated indexes, extracted content, evaluation outputs
    ├── rag_prototype.py        — Core pipeline: ingestion -> indexing -> retrieval -> generation -> guardrails
    ├── api.py                  — FastAPI backend
    ├── streamlit_app.py        — Streamlit frontend
    ├── notes.md                — Design trade-offs, retrieval strategy, guardrails, scalability
    └── README.md               — Exact setup and run instructions
```

See `architecture_diagram.pptx` for the visual pipeline, and `prototype/notes.md` for the full written design discussion.

## Retrieval Evaluation

The hybrid retrieval and reranking pipeline was evaluated using five manually labelled queries.

| Query | Precision@5 | Recall@5 |
|---|---:|---:|
| Cyclone start-up | 0.60 | 1.00 |
| Personal protective equipment | 0.40 | 1.00 |
| Sudden draft drop | 0.00 | N/A |
| Reduced suction inspection | 0.20 | 1.00 |
| Dust container inspection | 0.20 | 1.00 |

The sudden draft drop query had no relevant labelled pages in the retrieved evaluation set — a deliberate test case for the no-answer guardrail, not a retrieval failure. These results are preliminary and based on a small manually labelled query set.

## Guardrails

The prototype integrates a locally hosted Ollama language model to generate answers grounded in retrieved document excerpts, with four guardrails: grounded generation (LLM instructed to use only retrieved excerpts), citation validation (generated citations checked against retrieved document-page pairs), a retrieval-score no-answer gate, and sensitive-query filtering before retrieval and generation. Testing found real cases of each guardrail firing correctly, including the citation validator catching an unsupported page reference and a citation whose source was real but whose claim wasn't supported by it.

**Full test-by-test findings, known limitations, and design trade-offs are documented in `prototype/notes.md` — this is the authoritative source, not a summary here.**

## Scalability Considerations

Brief summary (full discussion with specifics in `prototype/notes.md`): the current exact-search FAISS/BM25 setup holds up to roughly 10x this corpus's size before needing approximate-nearest-neighbor indexing and a proper search engine; 100+ concurrent users would require multiple API replicas and a batching LLM-inference server rather than the single local Ollama instance used here; cost-conscious cloud deployment favors CPU-only retrieval plus either a scale-to-zero serverless GPU endpoint or staying with a small quantized model, given bursty internal-tool traffic.

## Running the Prototype

See `prototype/README.md` for exact setup and run steps — both the standalone script and the FastAPI + Streamlit app.
