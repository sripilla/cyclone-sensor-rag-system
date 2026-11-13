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

Activate the environment:

```bash
source Task2/env2/bin/activate
```

Install dependencies:

```bash
pip install -r Task2/requirements.txt
```

Deactivate:

```bash
deactivate
```

## Directory Structure

- `prototype/docs/` — Technical PDF documents
- `prototype/outputs/` — Generated indexes, extracted content and evaluation outputs
- `prototype/rag_prototype.py` — RAG prototype implementation

## Retrieval Evaluation

The hybrid retrieval and reranking pipeline was evaluated using five manually labelled queries.

| Query | Precision@5 | Recall@5 |
|---|---:|---:|
| Cyclone start-up | 0.60 | 1.00 |
| Personal protective equipment | 0.40 | 1.00 |
| Sudden draft drop | 0.00 | N/A |
| Reduced suction inspection | 0.20 | 1.00 |
| Dust container inspection | 0.20 | 1.00 |

The sudden draft drop query had no relevant labelled pages in the retrieved evaluation set.

These results are preliminary and based on a small manually labelled query set.

## LLM Integration and Guardrails

The prototype integrates a locally hosted Ollama language model to generate answers using retrieved document excerpts.

The following guardrails are implemented:

- **Grounded generation:** The LLM is instructed to use only the retrieved document excerpts.
- **Citation validation:** Generated citations are checked against the retrieved document-page pairs.
- **No-answer fallback:** Queries without sufficient retrieval support receive a predefined no-answer response.
- **Sensitive-query filtering:** Queries matching configured sensitive patterns are blocked before retrieval and generation.
- **Strict answer rejection:** The complete answer is rejected if citation validation detects unsupported or incorrectly formatted citations.

## Observed Guardrail Behaviour

Testing demonstrated the following behaviours:

- **Unsupported page references:** In the PPE test, the LLM cited page 14 of `654b45bb8292f586313470.pdf`, although page 14 was not among the retrieved pages. Citation validation rejected the answer.
- **Citation-format issues:** The start-up test produced shortened citation formats, resulting in answer rejection.
- **Unsupported interpretation:** For the sudden draft drop query, the LLM generated a speculative explanation citing `D13.pdf` page 4. Although the page was retrieved, the excerpt did not explicitly establish the claimed relationship. The retrieval-score gate rejected the query before production answer generation.
- **Sensitive-query filtering:** A request for a colleague's home address was blocked before retrieval and generation.

## Known Limitations and Design Trade-offs

- Citation validation confirms whether a cited document-page pair belongs to the retrieved context. It does not establish whether the cited passage semantically supports the associated claim.
- The reranker threshold is a heuristic retrieval gate, not a calibrated measure of factual confidence.
- The system rejects the entire answer when even one citation is unsupported. This is a deliberate conservative choice that prioritizes reliability over answer completeness.
- Claim-level evidence verification and automatic correction of unsupported statements are not implemented.
- Retrieval evaluation is preliminary and uses only five manually labelled queries.

## Scalability Considerations

The prototype uses a modular retrieval and generation pipeline. The architecture can be extended to support a larger document collection and concurrent users through separate ingestion and query-processing components, scalable indexing, and independently deployable API services.

These are proposed architectural extensions, not capabilities demonstrated by the current local prototype.