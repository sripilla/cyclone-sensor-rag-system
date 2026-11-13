# Cyclone Sensor Analytics and RAG-Based Technical Knowledge System

Data Science take-home assignment — two independent deliverables.

- **`Task1/`** — Cyclone machine sensor data analytics: data cleaning, shutdown detection, operational-state clustering, contextual anomaly detection, short-horizon forecasting, and operational insights.
- **`Task2/`** — A Retrieval-Augmented Generation (RAG) prototype for querying technical PDF documentation, with hybrid retrieval, reranking, local LLM generation, and citation guardrails.
- **`Final_Presentation.pptx`** — 6-9 slide summary covering both tasks.
- **`docs/`** — The original assignment document and the architecture baseline written at project start.

Each task has its own README, Python environment, and requirements file — see `Task1/README.md` and `Task2/README.md` for setup and run instructions.

## Task 1 — Key Results

- **377,719** raw sensor records cleaned and reindexed onto a strict 5-minute grid; 6 distinct SCADA fault-code strings identified and handled; 3 full-day structural data outages found and deliberately left unfilled rather than fabricated.
- **135 shutdown events** detected, totaling **6,722 hours (280 days)** of downtime over 3.6 years — the two longest align with the COVID-19 lockdown and an annual maintenance turnaround.
- **6 operational states** identified via KMeans clustering (k=6, silhouette 0.22): High Load, Moderate Load, Reduced Draft, Degraded Heat Recovery, Material Feed Anomaly, and Startup/Shutdown Transition.
- **73% of shutdowns** (99 of 135) were immediately preceded by the Reduced Draft state, despite it covering only 2.3% of active time — the strongest early-warning signal found in the analysis.
- **2,118 contextual anomalies** detected (per-cluster Isolation Forest); 5 investigated in depth with evidence-based root-cause hypotheses.
- **Forecasting**: a Random Forest model beat a persistence baseline overall (RMSE 19.47 vs 22.67) and by 19% in steady-state conditions, but underperformed persistence during shutdown-transition windows — a documented, real limitation rather than an overstated result.

## Task 2 — Key Results

- A working hybrid retrieval pipeline (FAISS dense search + BM25 sparse search, fused with Reciprocal Rank Fusion, reordered by cross-encoder reranking) over a 10-document, 189-chunk technical corpus.
- **Recall@5 = 1.0** on every evaluation query with a genuinely answerable topic; the reranker's confidence score cleanly separates answerable queries (scores +3.3 to +8.2) from an unanswerable one (-8.5), which became the basis for the system's no-answer guardrail.
- **Guardrails validated against real failures, not hypotheticals**: citation validation caught an actual LLM-fabricated page citation and a technically-real-but-unsupported citation during testing, correctly rejecting both.
- Full design write-up — retrieval strategy, design trade-offs, guardrails, and a scalability plan for 10x documents and 100+ concurrent users — in `Task2/prototype/notes.md`.

## Repository Structure

```
.
├── README.md                     (this file)
├── Final_Presentation.pptx
├── docs/
│   ├── assignment.docx
│   └── architecture_v1.md
├── Task1/
│   ├── README.md
│   ├── requirements.txt
│   ├── data/raw/                 (data.xlsx not tracked — supply locally, see Task1/README.md)
│   ├── notebooks/task1_analysis.ipynb
│   ├── outputs/                  (cleaned_data.csv, shutdown_periods.csv, clusters_summary.csv, anomalous_periods.csv, forecasts.csv, forecast_metrics.csv, ...)
│   └── plots/                    (13 PNGs)
└── Task2/
    ├── README.md
    ├── requirements.txt
    ├── architecture_diagram.pptx
    └── prototype/
        ├── README.md
        ├── notes.md
        ├── rag_prototype.py
        ├── api.py
        ├── streamlit_app.py
        ├── docs/                 (10 sample PDFs)
        └── outputs/              (extracted pages, chunks, FAISS/BM25 indexes, evaluation.csv)
```
