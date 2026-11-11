# Task 2 — RAG-Based Technical Knowledge System

## Objective
Build a Retrieval-Augmented Generation (RAG) prototype that answers questions from technical PDF documents with relevant citations.

## Technology stack
- PDF extraction: PyMuPDF
- Embeddings: Sentence Transformers
- Vector search: FAISS
- Keyword search: BM25
- Hybrid retrieval: Reciprocal Rank Fusion (RRF)
- Reranking: Cross-Encoder
- Local LLM: Ollama
- Backend: FastAPI
- Frontend: Streamlit

## Implementation workflow
1. PDF ingestion and text extraction
2. Document cleaning and metadata extraction
3. Structure-aware document chunking
4. Embedding generation and indexing
5. Hybrid retrieval using FAISS and BM25
6. Cross-encoder reranking
7. LLM-based answer generation
8. Citation handling and guardrails
9. FastAPI and Streamlit integration
10. Retrieval and answer-quality evaluation

## Environment setup

Activate the environment:

    source Task2/env2/bin/activate

Install dependencies:

    pip install -r Task2/requirements.txt

Deactivate:

    deactivate

## Directory structure
- `prototype/docs/` — Sample technical PDFs
- `prototype/outputs/` — Generated indexes and outputs
