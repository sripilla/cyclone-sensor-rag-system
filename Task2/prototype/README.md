# RAG Prototype — Run Instructions

See `../README.md` for the overall Task 2 overview and `notes.md` for design trade-offs, retrieval strategy, guardrails, and the scalability plan. See `../architecture_diagram.pptx` for the visual pipeline.

## Setup

    cd Task2
    python3 -m venv env2
    source env2/bin/activate
    pip install -r requirements.txt
    pip install fastapi uvicorn streamlit

## Required local model downloads

- Embedding model: `all-MiniLM-L6-v2` (auto-downloaded by sentence-transformers on first run)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (auto-downloaded on first run)
- LLM: Ollama with `llama3.2:1b` — install Ollama (https://ollama.com), then:

      ollama serve &
      ollama pull llama3.2:1b

## Run the full pipeline standalone (ingestion -> indexing -> retrieval -> generation -> guardrail tests)

    cd prototype
    python3 rag_prototype.py

## Run as an app (API + UI)

    # terminal 1
    cd prototype
    uvicorn api:app --reload --port 8000

    # terminal 2
    cd prototype
    streamlit run streamlit_app.py

Then open the Streamlit URL it prints and ask a question.

## Outputs

All generated artifacts land in `prototype/outputs/`: extracted page text, document chunks,
FAISS/BM25 indexes, the document quality report, and the retrieval evaluation CSV.
