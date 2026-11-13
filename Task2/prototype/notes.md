# Task 2 — Design Notes

**Project:** Cyclone Sensor Analytics and RAG-Based Technical Knowledge System
**Component:** Task 2 — RAG-Based Technical Knowledge System
**Status:** Prototype implementation and observed behaviour documented

---

## 1. Retrieval Strategy

**Chunking.** Page-bound, token-based chunking at ~500 tokens with 100-token overlap
(tiktoken `cl100k_base`), producing 189 chunks from 170 pages across 10 documents. Page-bound
chunking (never spanning a page boundary) was chosen over naive fixed-length chunking across
the whole document so every chunk keeps a clean, unambiguous page citation — most pages in this
corpus run under 500 tokens, so most chunks are a single page; only longer pages split into 2+
chunks.

**Embedding model.** `all-MiniLM-L6-v2` (384-dim, Sentence-Transformers) — chosen for being
small, free, and fast enough to embed the whole corpus in under a second, which matters for a
prototype rebuilt from scratch on every run. Validated empirically in Phase 9: recall@5 was 1.0
on every query with a genuinely answerable topic.

**Retrieval method: hybrid, not dense-only.** Dense embedding search alone missed real content —
the "draft drop" query's actual troubleshooting-guide chunk only surfaced once BM25's exact
keyword match was fused in via Reciprocal Rank Fusion (k=60). This was an observed failure of
pure dense retrieval on this corpus, not a hypothetical justification.

**Reranking.** A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reorders the fused top-10 down to the
top 5. Its score also became the basis for the no-answer confidence gate (see Guardrails): on
our 5-query eval set, every answerable query scored positive (+3.3 to +8.2) and the one
unanswerable query scored strongly negative (-8.5).

**Relevance and faithfulness.** Reranking (above), mandatory per-sentence citations enforced at
generation time, and post-generation citation validation against the actually-retrieved
document/page pairs (see Guardrails) — not just returning raw snippets.

## 2. Design Trade-offs

- **Hybrid over dense-only retrieval**: added BM25+RRF complexity specifically to fix an
  observed retrieval miss, not speculatively.
- **Small local embedding/reranker/LLM models over larger hosted ones**: keeps the prototype
  fully free/offline-runnable per the assignment's constraint, at the cost of weaker language
  generation quality than a larger hosted model would give.
- **Whole-answer rejection over partial repair** (detailed in Section 3.6 below): prioritizes
  never serving an unsupported citation over maximizing how often the system gives *an* answer.
- **Rebuilding the index from scratch on every run over persisting it**: simpler and safer for
  a small prototype corpus (189 chunks embeds in ~1 second); would not scale to the 10x/100+
  user scenarios without the persistent-index changes described in Section 4 below.

## 3. Guardrails & Failure Modes (Phase 10)

### 3.1 Purpose

Phase 10 integrates a locally hosted large language model (LLM) into the
RAG pipeline and adds guardrails intended to reduce unsupported answers.

The system is designed to answer questions using retrieved excerpts from
the technical PDF corpus. It should provide document-and-page citations,
refuse questions when retrieval evidence is insufficient, and block
configured sensitive queries.

The local LLM used for this phase is **Ollama with `llama3.2:1b`**.

### 3.2 Implemented Components

| Component | Implementation / intended behaviour |
|---|---|
| Local LLM integration | Ollama runs locally and generates answers from retrieved excerpts. |
| Prompt construction | The prompt instructs the model to use only supplied excerpts and not rely on outside knowledge. |
| Citation format | Citations are required in the exact form `[exact_document_filename.pdf p.X]`. |
| Citation validation | Generated document-page citations are checked against the document-page pairs returned by retrieval. |
| Retrieval-score gate | If the top reranker score is below the configured threshold, the system returns the no-answer response without generating an answer in the production path. |
| No-answer fallback | The configured response is: `I don't have reliable information in the provided documents to answer this question.` |
| Sensitive-query filtering | Queries matching configured sensitive patterns are blocked before retrieval and generation. |

**Important implementation details**

- The configured retrieval threshold is `NO_ANSWER_THRESHOLD = 0.0`.
- This threshold is a heuristic applied to the Cross-Encoder reranker
  score. It is **not** a calibrated probability or a guarantee of
  factual correctness.
- Citation validation checks whether a cited document-page pair
  belongs to the retrieved context. It does not prove that the cited
  passage semantically entails the claim.
- The sensitive-query test demonstrated blocking for the specific
  tested request. It does not establish comprehensive detection of
  every possible sensitive query.

### 3.3 End-to-End Test Queries and Observations

The following observations come from the Phase 10 terminal runs.

**Test 1 — Cyclone start-up**

**Query:** `How do I start up the cyclone separator?`

**Observed behaviour:**

- The hybrid retrieval and reranking stage found relevant start-up
  material.
- The top reranker score in the Phase 9 retrieval evaluation was
  approximately `8.163`.
- The LLM generated relevant start-up content.
- Some generated references used shortened formats such as
  `[citation: 11]` and `[citation: 10]`, rather than the required full
  filename-and-page format.
- The answer was rejected by the citation guardrail and the production
  response was `no_answer`.

**Observation:** This test demonstrated that relevant retrieved content
does not guarantee that the LLM will follow the required citation
format. The validator rejected the response rather than returning an
answer that did not meet the citation rules.

**Caution:** The available terminal output shows shortened citation
formats. It does not, by itself, establish that the model invented a
particular additional page number in this start-up response.

**Test 2 — Personal protective equipment (PPE)**

**Query:** `What personal protective equipment is required?`

**Retrieved document-page pairs included:**

- `654b45bb8292f586313470.pdf` page 12
- `654b45bb8292f586313470.pdf` page 13
- `654b45bb8292f586313470.pdf` page 15
- `654b45bb8292f586313470.pdf` page 13
- `654b45bb8292f586313470.pdf` page 11

**Observed behaviour:**

- The LLM generated a list of PPE items.
- In an earlier run, the answer passed citation validation, but
  inspection of the generated content raised concerns that some
  descriptions or associations might not be supported by the cited
  excerpts.
- In a subsequent run, the LLM cited `654b45bb8292f586313470.pdf` page
  14 several times.
- Page 14 was not among the retrieved document-page pairs for that
  query.
- Citation validation rejected the answer and the production response
  was `no_answer`.

**Observation:** The page-14 reference is an observed unsupported
citation: the cited page was not part of the retrieved context. The
validator caught this and rejected the whole answer.

This is evidence that the citation validator can detect fabricated or
unsupported document-page references. It does not prove that every other
statement in the generated answer was correct.

**Test 3 — Sudden draft drop**

**Query:** `What does a sudden draft drop indicate?`

**Observed retrieval behaviour:**

- The top reranker score was approximately `-8.483`.
- The retrieved excerpts did not provide a reliable direct answer to
  the question.
- The production retrieval-score gate rejected the query before answer
  generation.
- The production response was the exact no-answer fallback.

**Diagnostic observation:**

A separate raw LLM diagnostic generation, run to inspect model
behaviour, produced a confident explanation linking a sudden draft drop
to pressure differential and cited `D13.pdf` page 4.

That page was among the retrieved pages, so the document-page citation
could pass a simple retrieved-source membership check. However, the
excerpt was a theoretical abstract about cyclone performance and
pressure differential; it did not explicitly establish what a sudden
draft drop indicates.

**Observation:** A citation can refer to a genuinely retrieved page
while the claim attached to it is still unsupported. Citation membership
validation alone cannot establish claim-level factual support.

The retrieval-score gate prevented this diagnostic unsupported
interpretation from becoming the production answer for this query.

**Test 4 — Sensitive personal information**

**Query:** `What is my colleague's home address?`

**Observed behaviour:**

- The query matched the configured sensitive-query pattern for
  `home address`.
- The system returned status `blocked_sensitive`.
- The sensitive-query check short-circuited the normal retrieval and
  generation path.

**Observation:** Sensitive-query filtering worked as intended for this
specific test.

### 3.4 Summary of End-to-End Test Results

| Test query | Observed production status | Interpretation |
|---|---|---|
| Cyclone start-up | `no_answer` | Relevant material was retrieved, but generated citation formatting did not satisfy the validator. |
| PPE requirements | `no_answer` in the later run | An unsupported page-14 citation was detected and the answer was rejected. |
| Sudden draft drop | `no_answer` | Low reranker score caused the production gate to reject the query before generation. |
| Colleague's home address | `blocked_sensitive` | Configured sensitive-query filter blocked the request. |

These are observations from a small set of end-to-end test queries. They
should not be presented as proof that the system is free from
hallucinations or that all possible sensitive requests will be detected.

### 3.5 What the Guardrails Demonstrated

**Citation validation caught an unsupported page reference.** The PPE
test demonstrated that the LLM can cite a page that was not supplied in
the retrieved context. The validator detected the mismatch between the
cited page and the retrieved document-page pairs and rejected the
answer.

**Retrieval-score gating and citation validation serve different
purposes.** The retrieval-score gate evaluates whether the retrieved
result is sufficiently relevant according to the configured heuristic
threshold. Citation validation checks whether cited document-page pairs
belong to the retrieved context. Neither mechanism alone establishes
that every generated claim is entailed by its cited excerpt. The
sudden-draft diagnostic demonstrated this limitation: the cited page was
retrieved, but the generated interpretation was not explicitly supported
by the excerpt.

**Sensitive-query filtering operates before generation.** The
home-address test demonstrated that a query matching a configured
sensitive pattern is blocked before the system proceeds to retrieval and
LLM generation.

### 3.6 Deliberate Design Trade-off: Safety Over Answer Completeness

The prototype rejects the **entire generated answer** when citation
validation finds an unsupported or incorrectly formatted citation.

This can result in rejecting an answer even when some of its statements
are relevant and correctly cited. The design is intentionally
conservative: it prioritizes avoiding unsupported answers over retaining
partially useful content.

This is a deliberate trade-off, not an accidental behaviour.

The current prototype does not attempt to remove only the unsupported
sentence, repair the citation, or regenerate a partial answer.

### 3.7 Known Limitations

1. **Citation membership is not claim verification.** The validator
   checks whether a cited document-page pair was retrieved. It does not
   determine whether the cited passage logically or semantically
   supports the associated statement.
2. **Reranker scores are heuristic.** The configured threshold is not a
   calibrated probability of answer correctness or factual confidence.
3. **Whole-answer rejection can reduce helpfulness.** One invalid
   citation can cause the complete answer to be rejected, even if other
   parts are supported.
4. **No claim-level evidence verification.** The prototype does not
   independently verify every factual sentence against its cited
   excerpt.
5. **No automatic answer repair.** The prototype does not automatically
   correct invalid citations or regenerate only unsupported portions.
6. **Limited test coverage.** The observed end-to-end behaviour is
   based on a small set of manually selected test queries.
7. **Sensitive-pattern coverage is limited.** The demonstrated test
   confirms the configured home-address pattern works for that query;
   it does not establish comprehensive sensitive-query detection.

### 3.8 Why Further Prompt Tuning Was Stopped

Further prompt changes could potentially improve the model's compliance
with citation formatting. However, the observed tests showed that the
guardrails were identifying real problems:

- A citation to a page that was not retrieved.
- Citation-format violations.
- A generated interpretation that was not explicitly supported by the
  retrieved excerpt, despite using a real retrieved citation.

The decision was to retain conservative rejection behaviour rather than
weaken validation simply to increase the number of answers marked as
successful.

For this assignment prototype, the observed guardrail behaviour and its
limitations are documented rather than claiming that the LLM always
produces correct answers.

### 3.9 Final Phase 10 Assessment

Phase 10 demonstrates a local LLM integrated with a retrieval pipeline,
strict citation instructions, retrieved-source citation validation, a
heuristic retrieval-score gate, a no-answer fallback, and a configured
sensitive-query filter.

The most important outcome is not that every test question received an
answer. It is that the system demonstrated several ways of refusing to
return an answer when the evidence or citation requirements were not
satisfied.

The prototype remains limited by the absence of claim-level entailment
verification. This limitation is explicitly acknowledged rather than
overstating the reliability of citation validation.

## 4. Scalability Considerations

### 10x increase in documents

The current prototype uses `faiss.IndexFlatIP` (exact brute-force search) and `rank_bm25`'s
pure-Python BM25 — both are fine at 189 chunks, and still fine at ~2,000 (our current scale
×10). At the assignment's real target (50+ documents, likely 1,000+ chunks) this remains
workable; at a much larger scale (10,000+ chunks) we would switch to an approximate-nearest-
neighbor FAISS index (IVF+PQ or HNSW) for sub-linear query time, and move BM25 to a proper
search engine (OpenSearch/Elasticsearch) rather than an in-memory Python structure. Ingestion
(PDF extraction, chunking, embedding) is embarrassingly parallel across documents and would
move to a batch worker pool. Indexes would move to a persistent vector DB (Qdrant/Weaviate/
pgvector) supporting incremental updates, rather than rebuilding from scratch on every startup
as this prototype does.

### 100+ concurrent users

This prototype holds one model instance per process and rebuilds the index at startup — fine
for a single demo user. For concurrent load: run multiple API worker replicas behind a load
balancer, each with its own copy of the (small, ~90MB) embedding and reranker models in memory.
The real bottleneck is LLM generation (Ollama, single local instance) — production would need
either a batching inference server (vLLM/TGI) or a hosted LLM API with autoscaling, plus a
cache for frequently repeated questions to avoid redundant generation calls.

### Cost-conscious cloud deployment

Retrieval (FAISS + BM25 + cross-encoder reranking) is cheap and runs fine on CPU — deploy that
layer on a small always-on instance. LLM generation is the expensive part: for bursty internal-
tool traffic (not constant), a serverless GPU endpoint that scales to zero when idle is more
cost-effective than a dedicated always-on GPU; alternatively, stay with a small quantized model
(1-3B params, as used here) that runs acceptably on CPU and avoids GPU cost entirely if latency
requirements allow it. The API layer itself (FastAPI) is a good fit for serverless containers
(Cloud Run/Fargate) that scale to zero between usage bursts.

---

**Documentation note:** The observations in this file describe the
recorded prototype runs. They should be updated if later code changes or
additional evaluation runs produce different results.