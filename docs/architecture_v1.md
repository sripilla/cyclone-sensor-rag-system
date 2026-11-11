# Architecture Baseline v1.0

## Project
Cyclone Sensor Analytics and RAG-Based Technical Knowledge System

The project consists of two independent but complementary modules.

---

## Task 1 — Cyclone Sensor Analytics

### Processing pipeline

Raw Excel Dataset
    ↓
Data Cleaning and Validation
    ↓
5-Minute Timestamp Indexing
    ↓
Shutdown and Idle Detection
    ↓
Feature Engineering (Active Operating Periods)
    ↓
KMeans Operational-State Clustering
    ↓
Cluster-Conditional Contextual Anomaly Detection
    ↓
Anomaly Investigation and Root-Cause Hypotheses
    ↓
One-Hour-Ahead Temperature Forecasting
    ↓
Operational Insights and Recommendations

### Key design decisions
- Use a modular batch-processing pipeline.
- Preserve the original raw dataset.
- Convert sensor columns to numeric values during data preparation.
- Validate timestamps and handle missing intervals.
- Exclude shutdown periods from active-state clustering.
- Use contextual anomaly detection based on operating states.
- Compare forecasting models against a persistence baseline.
- Use chronological training and testing splits.

---

## Task 2 — RAG-Based Technical Knowledge System

### Offline ingestion pipeline

Technical PDF Documents
    ↓
PDF Text Extraction using PyMuPDF
    ↓
Text Cleaning and Metadata Extraction
    ↓
Structure-Aware Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Index + BM25 Keyword Index

### Online query pipeline

User Query
    ↓
Hybrid Retrieval
    ↓
Reciprocal Rank Fusion (RRF)
    ↓
Cross-Encoder Reranking
    ↓
Local LLM using Ollama
    ↓
Answer with Document and Page Citations
    ↓
FastAPI Backend
    ↓
Streamlit Interface

### Key design decisions
- Use open-source components wherever practical.
- Separate document ingestion from query processing.
- Combine semantic and keyword retrieval.
- Apply reranking to improve retrieval relevance.
- Include citations and answer guardrails.
- Build a minimal runnable local prototype.
- Evaluate retrieval quality and answer faithfulness.

---

## Overall architecture principles

1. Task 1 and Task 2 remain independent modules.
2. Both modules share the goal of industrial intelligence.
3. Use modular, maintainable and reproducible implementations.
4. Prioritize a working local prototype.
5. Document scalability considerations as future architecture, not as deployed functionality.
6. Avoid unnecessary scope expansion.

---

## Version
Architecture Baseline v1.0

This baseline remains fixed unless explicitly revised.
