from fastapi import FastAPI
from pydantic import BaseModel
import rag_prototype as rp

app = FastAPI(title="Cyclone Technical Q&A API")
_state = {}

@app.on_event("startup")
def startup():
    all_pages, pdf_files = rp.extract_documents()
    rp.generate_quality_report(all_pages, pdf_files)
    all_chunks = rp.generate_chunks(all_pages)
    rp.save_chunks(all_chunks)
    index, bm25, model = rp.build_indexes(all_chunks)
    tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
    reranker = rp.CrossEncoder(rp.RERANKER_MODEL_NAME)
    _state.update(
        index=index, bm25=bm25, tokenized_corpus=tokenized_corpus,
        model=model, reranker=reranker, all_chunks=all_chunks,
    )
    print(f"API ready: {len(all_chunks)} chunks indexed")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QueryRequest):
    return rp.answer_question(
        req.question, _state["index"], _state["bm25"], _state["tokenized_corpus"],
        _state["model"], _state["reranker"], _state["all_chunks"],
    )

@app.get("/health")
def health():
    return {"status": "ok", "n_chunks": len(_state.get("all_chunks", []))}