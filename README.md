# DocuMind

Production RAG (Retrieval-Augmented Generation) system for domain-specific document Q&A. Features hybrid retrieval (BM25 + vector search), cross-encoder reranking, citation enforcement, and a CI-gated evaluation pipeline.

Powered by **Google Gemini** for generation and evaluation.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│         Hybrid Retrieval            │
│  ┌──────────┐    ┌──────────────┐   │
│  │  BM25L   │    │ Vector Store │   │
│  │ (sparse) │    │ (dense, via  │   │
│  │          │    │  ChromaDB)   │   │
│  └────┬─────┘    └──────┬───────┘   │
│       │                 │           │
│       └───────┬─────────┘           │
│               ▼                     │
│     Reciprocal Rank Fusion          │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│     Cross-Encoder Reranking         │
│  (ms-marco-MiniLM-L-6-v2)          │
│  Re-scores top candidates by        │
│  query-document relevance           │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│    Citation-Enforced Generation     │
│  Gemini generates answer with [N]   │
│  inline citations. Post-processing  │
│  validates all references exist.    │
└───────────────┬─────────────────────┘
                │
                ▼
        Answer + Citations
```

## How It Works

### 1. Document Ingestion
Documents (Markdown, PDF, plain text) are loaded from a directory, split into overlapping chunks using a recursive text splitter that respects paragraph and sentence boundaries, and indexed into both a BM25 sparse index and a ChromaDB dense vector store.

### 2. Hybrid Retrieval
When a query arrives, it's run against both indexes in parallel:
- **BM25L** finds chunks with strong keyword overlap (good for exact terms, names, acronyms)
- **Vector search** finds semantically similar chunks (good for paraphrases, conceptual matches)

Results are merged using **Reciprocal Rank Fusion (RRF)**, which combines rankings without needing to normalize scores across different retrieval methods.

### 3. Cross-Encoder Reranking
The fused candidate set (up to 50 chunks) is re-scored by a cross-encoder model (`ms-marco-MiniLM-L-6-v2`). Unlike bi-encoders, cross-encoders see the query and document together, producing much more accurate relevance scores. The top-k (default 5) chunks survive.

### 4. Citation-Enforced Generation
The top chunks are passed to Gemini with a system prompt requiring `[N]` inline citations for every claim. After generation:
- Citation IDs are extracted from the answer
- Invalid references (citing non-existent chunks) are stripped
- If the model failed to cite anything, all sources are attached as a fallback

### 5. Evaluation Pipeline
A golden dataset (question + expected answer + expected sources) is evaluated with four metrics:
- **Faithfulness**: Is the answer grounded in the retrieved chunks? (LLM-judged)
- **Relevance**: Does the answer address the question? (LLM-judged)
- **Citation accuracy**: Are `[N]` references valid and covering all chunks?
- **Source recall**: Did retrieval find the expected source documents?

The CI pipeline gates on configurable thresholds — PRs that degrade quality are blocked.

## Quick Start

### Prerequisites
- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/apikey)

### Installation

```bash
git clone <repo-url> && cd documind
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` and set your Gemini API key:

```
GEMINI_API_KEY=your-key-here
```

### Add Your Documents

Place your documentation files in the `docs/` directory. Supported formats:
- Markdown (`.md`)
- PDF (`.pdf`)
- Plain text (`.txt`, `.rst`)

The project ships with sample docs in `docs/` for demonstration.

### Ingest Documents

```bash
python scripts/ingest.py
```

Or ingest from a custom directory:

```bash
python scripts/ingest.py /path/to/your/docs
```

### Start the Server

```bash
uvicorn src.rag.api:app --reload
```

The API starts at `http://localhost:8000`. The server automatically ingests docs on startup if no existing index is found.

### Query Your Documents

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does authentication work?"}'
```

**Example response:**

```json
{
  "answer": "The API uses JWT-based authentication with refresh tokens [1]. All requests must include a valid Bearer token in the Authorization header [1]. Access tokens are valid for 15 minutes, and refresh tokens for 7 days [1].",
  "citations": [
    {
      "ref_id": 1,
      "source": "docs/api-reference.md",
      "title": "Authentication",
      "quote": "The API uses JWT-based authentication with refresh tokens..."
    }
  ],
  "chunks_used": [
    {
      "chunk": {
        "chunk_id": "a1b2c3d4",
        "text": "The API uses JWT-based authentication...",
        "source": "docs/api-reference.md",
        "title": "Authentication"
      },
      "score": 0.95,
      "origin": "reranker"
    }
  ],
  "query": "How does authentication work?"
}
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with index status |
| `POST` | `/query` | Ask a question (returns answer + citations) |
| `POST` | `/ingest` | Re-ingest documents from the docs directory |
| `POST` | `/upload` | Upload a single document file |

### POST /query

```json
{
  "query": "What database migrations are supported?",
  "top_k": 5
}
```

- `query` (required): Your question, 1-2000 characters
- `top_k` (optional): Number of chunks to use, 1-20, default 5

### POST /ingest

```json
{
  "docs_dir": ""
}
```

Leave `docs_dir` empty to use the default directory from config. Paths are restricted to the configured docs directory for security.

### POST /upload

Multipart form upload. Max file size: 50 MB.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my-document.pdf"
```

## Running the Evaluation Pipeline

### 1. Define your golden dataset

Edit `eval/golden.jsonl` with question/answer pairs:

```jsonl
{"question": "What auth mechanism is used?", "expected_answer": "JWT-based authentication with refresh tokens.", "expected_sources": ["docs/api-reference.md"], "tags": ["auth"]}
{"question": "How is rate limiting configured?", "expected_answer": "Via the RATE_LIMIT_PER_MINUTE environment variable.", "expected_sources": ["docs/configuration.md"], "tags": ["config"]}
```

### 2. Run evaluation locally

```bash
python scripts/evaluate.py
```

Output:

```json
{
  "num_examples": 3,
  "avg_faithfulness": 0.85,
  "avg_relevance": 0.90,
  "avg_citation_accuracy": 0.93,
  "avg_source_recall": 1.0,
  "thresholds": {
    "faithfulness": 0.7,
    "relevance": 0.7,
    "citation": 0.9
  },
  "passed": true
}
```

### 3. CI gating

The included GitHub Actions workflow (`.github/workflows/eval.yml`) runs on every PR:

1. **Lint** — Ruff checks formatting and style
2. **Test** — Pytest runs all unit tests
3. **Evaluate** — Ingests docs, runs the golden dataset, fails the build if scores drop below thresholds

Set `GEMINI_API_KEY` as a repository secret in GitHub.

## Configuration Reference

All settings can be configured via environment variables (prefixed with `RAG_`) or in a `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `RAG_LLM_MODEL` | `gemini-2.0-flash` | Gemini model for generation |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for embeddings |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `RAG_CHUNK_SIZE` | `512` | Max characters per chunk |
| `RAG_CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `RAG_BM25_TOP_K` | `25` | BM25 candidates to retrieve |
| `RAG_VECTOR_TOP_K` | `25` | Vector search candidates to retrieve |
| `RAG_RERANK_TOP_K` | `5` | Final chunks after reranking |
| `RAG_RRF_K` | `60` | RRF fusion constant (higher = more uniform weighting) |
| `RAG_EVAL_FAITHFULNESS_THRESHOLD` | `0.7` | Minimum faithfulness score to pass CI |
| `RAG_EVAL_RELEVANCE_THRESHOLD` | `0.7` | Minimum relevance score to pass CI |
| `RAG_EVAL_CITATION_THRESHOLD` | `0.9` | Minimum citation accuracy to pass CI |

## Project Structure

```
documind/
├── src/rag/
│   ├── config.py              # Pydantic settings (env-driven)
│   ├── models.py              # Chunk, ScoredChunk, Citation, request/response models
│   ├── chunker.py             # Recursive text splitter with overlap
│   ├── ingest.py              # Markdown, PDF, text file loader
│   ├── embeddings.py          # Sentence-transformers dense embeddings
│   ├── bm25_index.py          # BM25L sparse retrieval with persistence
│   ├── vector_store.py        # ChromaDB dense vector store
│   ├── hybrid_retriever.py    # RRF fusion of BM25 + vector results
│   ├── reranker.py            # Cross-encoder reranking
│   ├── citations.py           # Citation extraction, validation, enforcement
│   ├── generator.py           # Gemini generation with citation prompting
│   ├── pipeline.py            # End-to-end RAG orchestration
│   └── api.py                 # FastAPI server
├── eval/
│   ├── dataset.py             # Golden set loader (JSONL format)
│   ├── metrics.py             # Faithfulness, relevance, citation, recall metrics
│   ├── runner.py              # Evaluation runner with pass/fail thresholds
│   └── golden.jsonl           # Evaluation dataset
├── tests/                     # 31 unit tests
├── docs/                      # Your documents go here
├── scripts/
│   ├── ingest.py              # CLI: ingest documents
│   └── evaluate.py            # CLI: run evaluation pipeline
├── .github/workflows/eval.yml # CI pipeline
├── pyproject.toml             # Dependencies and tool config
└── .env.example               # Configuration template
```

## Running Tests

```bash
pytest tests/ -v
```

All 31 tests run without requiring a Gemini API key — LLM-dependent components are mocked in tests.

## Why These Design Choices?

**BM25L over BM25Okapi**: BM25Okapi's IDF formula produces zero scores when a term appears in exactly half the corpus, which breaks small document sets. BM25L's formula avoids this edge case.

**RRF over score normalization**: Sparse (BM25) and dense (vector) scores are on incomparable scales. RRF merges ranked lists using only rank positions, making it robust without tuning.

**Cross-encoder reranking**: Bi-encoder retrieval is fast but approximate. A cross-encoder sees query and document together, dramatically improving precision for the final top-k.

**Citation enforcement as post-processing**: Rather than hoping the LLM cites correctly, we validate citations after generation — stripping invalid ones and flagging uncited answers. This makes citation quality measurable and enforceable in CI.
