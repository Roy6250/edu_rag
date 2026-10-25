# EduRAG — Knowledge Retrieval-Based Question & Answer Generation for Education

A simple, end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that turns a set of textbook / study-material PDFs into a searchable knowledge base, then uses an LLM to:

1. **Answer** student questions grounded in the retrieved material, and
2. **Generate** exam-style questions (MCQ or short-answer) with answers, at a chosen difficulty level,

and evaluates the system with the **RAGAS** metrics (faithfulness, answer relevancy, context recall, context precision).

This code is a minimal implementation of the paper:

> **Knowledge Retrieval-Based Intelligent Question and Answer Generation Framework for Education**
> Anay Pardasani, Karen Maleski, Sayantan Roy — *Journal of Student Research*, Volume 14, Issue 1 (2025).

📖 **Want the story behind the project?** Read [PROJECT_JOURNEY.md](PROJECT_JOURNEY.md). It covers the problem we started with, why we chose RAG, how the system was built and evaluated, the results, and how those findings became the paper.

---

## Table of contents

- [Project journey (write-up)](PROJECT_JOURNEY.md)
- [Architecture](#architecture)
- [How it maps to the paper](#how-it-maps-to-the-paper)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Ingest PDFs](#1-ingest-pdfs)
  - [2. Ask a question](#2-ask-a-question)
  - [3. Generate exam questions](#3-generate-exam-questions)
  - [4. Evaluate with RAGAS](#4-evaluate-with-ragas)
- [Using it from Python](#using-it-from-python)
- [Pipeline details](#pipeline-details)
- [MongoDB vector index](#mongodb-vector-index)
- [Evaluation metrics](#evaluation-metrics)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Architecture

```
                         ┌──────────────────────── INGESTION ────────────────────────┐
                         │                                                           │
  PDFs ──► pypdf text ──►│ clean (regex) ──► SentenceSplitter ──► multilingual       │
                         │                   (512 tok, 50 overlap)  embeddings (768d)│
                         │                                               │           │
                         └───────────────────────────────────────────────┼───────────┘
                                                                         ▼
                                                     MongoDB Atlas  (nodes: text + embedding + metadata)
                                                     + Atlas Vector Search index (cosine)
                                                                         ▲
                         ┌──────────────────────── QUERY ────────────────┼───────────┐
                         │                                               │           │
  Question / Topic ─────►│ embed query ──► cosine top-k search ──────────┘           │
                         │                        │                                  │
                         │                        ▼                                  │
                         │            prompt template + retrieved context            │
                         │                        │                                  │
                         │                        ▼                                  │
                         │                 OpenAI LLM (gpt-4o-mini)                  │
                         │                        │                                  │
                         └────────────────────────┼──────────────────────────────────┘
                                                  ▼
                              Answer + source nodes   |   Generated Q&A (JSON)
                                                  │
                                                  ▼
                                 RAGAS evaluation (4 metrics)
```

---

## How it maps to the paper

| Paper step | Paper uses | This implementation |
|---|---|---|
| Data extraction | OCR + LlamaParse | `pypdf` text extraction (digital PDFs) |
| Cleaning | Basic NLP preprocessing | Regex: de-hyphenation, whitespace collapse, page-number removal |
| Chunking | LlamaIndex sentence splitter, fixed size with overlap | `SentenceSplitter(chunk_size=512, chunk_overlap=50)` |
| Vectorization | Multilingual BERT (768-d) | `paraphrase-multilingual-mpnet-base-v2` (multilingual BERT-family, 768-d, tuned for semantic similarity) |
| Node storage | MongoDB nodes with metadata | MongoDB Atlas via `MongoDBAtlasVectorSearch`; metadata = `subject`, `source`, `page` |
| Retrieval | Cosine similarity, top-k = 1 | Atlas Vector Search (cosine), `TOP_K = 1` by default |
| Prompt | Paper's QA template | Same template, verbatim |
| Generation | GPT-4 / GPT-4o-mini / Gemini 1.5 | OpenAI models (default `gpt-4o-mini`, switchable with `--llm`) |
| Question generation | Framework goal | Topic-based retrieval + JSON question/answer generation prompt |
| Evaluation | RAGAS: faithfulness, answer relevancy, context recall, context precision | Same four metrics via `ragas` |

---

## Project structure

```
edu_rag/
├── app.py            # CLI entry point: ingest | ask | generate | eval
├── config.py         # credentials (placeholders), model names, chunking, prompts
├── common.py         # shared setup: embedder, LLM, MongoDB vector store, index loading
├── ingest.py         # PDF -> clean -> chunk -> embed -> MongoDB Atlas (+ creates vector index)
├── rag.py            # EduRAG class: answer() and generate_questions()
├── evaluate.py       # RAGAS evaluation over a CSV of questions + ground truths
├── eval_set.csv      # tiny sample evaluation set
├── PROJECT_JOURNEY.md # write-up: problem → why RAG → experiments → results → paper
├── data/             # put your PDFs here
└── requirements.txt
```

---

## Requirements

- Python **3.10+**
- An **OpenAI API key**
- A **MongoDB Atlas** cluster (the free M0 tier supports Vector Search)
- ~2 GB disk for the embedding model + PyTorch (downloaded on first run)

---

## Setup

```bash
git clone https://github.com/Roy6250/edu_rag.git
cd edu_rag

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure credentials

Open `config.py` and replace the placeholder strings:

```python
OPENAI_API_KEY = "sk-your-openai-api-key"
MONGODB_URI = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
```

> ⚠️ Do **not** commit real keys. Consider moving these to environment variables before sharing your fork.

### MongoDB Atlas checklist

1. Create a cluster (M0 free tier is fine).
2. **Database Access** → create a database user.
3. **Network Access** → add your IP address (or `0.0.0.0/0` for testing).
4. Copy the connection string into `MONGODB_URI`.

You do **not** need to create the vector index manually — `ingest` creates it automatically. See [MongoDB vector index](#mongodb-vector-index) for the exact definition, sizes, and how to create it by hand.

---

## Usage

### 1. Ingest PDFs

Put your PDFs in `data/` and run:

```bash
python app.py ingest --pdf_dir data --subject Physics
```

```
3 PDFs -> 214 pages -> 482 chunks
Generating embeddings: 100%|██████████| 482/482
Created vector index 'vector_index' (takes ~1 min to become queryable).
Ingestion complete.
```

Run once per subject (e.g. `--subject Biology` with a different folder) — the `subject` is stored as node metadata.

> The Atlas index takes about a minute to build after the first ingest. Queries run before it is ready return no context.

### 2. Ask a question

```bash
python app.py ask "How can acceleration be determined using a velocity-time graph?"
```

```
Answer: The acceleration is the slope of the velocity-time graph. ...

Sources:
  - {'score': 0.8731, 'subject': 'Physics', 'source': 'kinematics.pdf', 'page': 12}
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--llm` | `gpt-4o-mini` | Any OpenAI chat model, e.g. `gpt-4`, `gpt-4o` |
| `--top_k` | `1` | Number of chunks to retrieve as context |

### 3. Generate exam questions

```bash
python app.py generate --topic "kinematics" --n 3 --difficulty medium --type mcq
```

```json
[
  {
    "question": "What does the slope of a velocity-time graph represent?",
    "options": ["A) Displacement", "B) Acceleration", "C) Speed", "D) Jerk"],
    "answer": "B",
    "explanation": "The slope of a v-t graph is the rate of change of velocity, i.e. acceleration."
  }
]
```

| Flag | Default | Values |
|---|---|---|
| `--topic` | *(required)* | Any topic text; used as the retrieval query |
| `--n` | `5` | Number of questions |
| `--difficulty` | `medium` | `easy`, `medium`, `hard` |
| `--type` | `mcq` | `mcq`, `short` |
| `--llm` | `gpt-4o-mini` | OpenAI model |

Question generation retrieves the top 3 chunks for the topic and instructs the LLM to write questions answerable **only** from that context.

### 4. Evaluate with RAGAS

Prepare a CSV with two columns:

```csv
question,ground_truth
How can acceleration be determined using a velocity-time graph?,The slope of the velocity-time graph gives acceleration and the area under the curve gives displacement.
```

The paper used 100 questions created from a **separate** set of PDFs. Then:

```bash
python app.py eval --csv eval_set.csv --llm gpt-4o-mini --out scores.csv
```

```
Metric scores for gpt-4o-mini
  faithfulness                             0.xxxx
  answer_relevancy                         0.xxxx
  context_recall                           0.xxxx
  llm_context_precision_with_reference     0.xxxx
```

To reproduce the paper's model comparison, run it once per model:

```bash
python app.py eval --csv eval_set.csv --llm gpt-4
python app.py eval --csv eval_set.csv --llm gpt-4o-mini
```

`--out` saves per-question scores for error analysis. The RAGAS judge model is set separately via `EVAL_LLM_MODEL` in `config.py`.

---

## Using it from Python

```python
from rag import EduRAG

rag = EduRAG(llm_model="gpt-4o-mini", top_k=1)

result = rag.answer("What are decomposers?")
print(result["answer"])
print(result["sources"])        # [{'score': ..., 'subject': ..., 'source': ..., 'page': ...}]

questions = rag.generate_questions("photosynthesis", n=5, difficulty="easy", qtype="short")
for q in questions:
    print(q["question"], "->", q["answer"])
```

---

## Pipeline details

### Data preparation (`ingest.py`)
- Each PDF page is extracted with `pypdf` and cleaned: hyphenated line breaks re-joined, repeated whitespace / blank lines collapsed, standalone page numbers dropped.
- Every page becomes a LlamaIndex `Document` with metadata `{subject, source, page}`.

### Chunking
- `SentenceSplitter` with fixed-size chunks (512 tokens) and 50-token overlap. The overlap preserves context across chunk boundaries, improving retrieval.

### Vectorization & storage
- Chunks are embedded with a multilingual 768-d sentence encoder, so English and non-English material share the same vector space (the paper's bilingual setting).
- Each chunk is stored in MongoDB as a **node**, as described in the paper:

```json
{
  "_id": "…",
  "text": "Organisms that convert decaying matter into nutrients …",
  "embedding": [0.23, 0.89, -0.10, 0.54, "…"],
  "metadata": { "subject": "Biology", "source": "ecology.pdf", "page": 4 }
}
```

### Retrieval & answering (`rag.py`)
- The query is embedded with the same model, and Atlas Vector Search returns the top-k nodes by **cosine similarity**.
- The retrieved context fills the paper's prompt template:

```
You are an assistant for question-answering tasks. Use the following pieces of
retrieved context to answer the question. If you don't know the answer, just say
that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
```

### Question generation (`rag.py`)
- The topic is used as a retrieval query (top 3 chunks); the LLM is asked to return a JSON list of questions with answers (and options + explanation for MCQs), constrained to the given context and difficulty.

---

## MongoDB vector index

### Sizes at a glance

| Parameter | Value | Where it is set |
|---|---|---|
| Embedding dimensions | **768** | `EMBED_DIM` (must match `EMBED_MODEL` output) |
| Similarity function | **cosine** | Index definition in `ingest.py` |
| Chunk size | **512 tokens** (~350–400 English words) | `CHUNK_SIZE` |
| Chunk overlap | **50 tokens** (~10%) | `CHUNK_OVERLAP` |
| Retrieved chunks per question (`top_k`) | **1** (as in the paper) | `TOP_K` / `--top_k` |
| Retrieved chunks for question generation | **3** | `generate_questions(k=3)` |
| Index name | `vector_index` | `VECTOR_INDEX_NAME` |
| Database / collection | `edu_rag` / `nodes` | `DB_NAME` / `COLLECTION_NAME` |
| Approx. storage per node | **~9–10 KB** (768 × 8-byte BSON doubles ≈ 6 KB for the vector, plus BSON array overhead, chunk text, and metadata) | — |

Rough capacity: 1,000 chunks ≈ 10 MB, so the free M0 tier (512 MB) holds about 50,000 chunks. That's a few hundred textbooks.

### Node document schema

Each chunk is stored as one document. LlamaIndex's `MongoDBAtlasVectorSearch` uses these default field names:

```json
{
  "_id": "<node id>",
  "text": "chunk text ...",
  "embedding": [0.0123, -0.0456, "... 768 floats ..."],
  "metadata": {
    "subject": "Physics",
    "source": "kinematics.pdf",
    "page": 12,
    "...": "LlamaIndex bookkeeping fields (ref_doc_id, node content, etc.)"
  }
}
```

### Index definition

The index is an **Atlas Vector Search** index (`type: "vectorSearch"`). It is separate from regular MongoDB indexes and from Atlas full-text search indexes.

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "metadata.subject"
    }
  ]
}
```

- `path: "embedding"` is the field holding the vector.
- `numDimensions: 768` must equal the embedding model's output size. If it doesn't, documents aren't indexed and queries return nothing.
- `similarity: "cosine"` matches the paper's retrieval method. `dotProduct` gives the same ranking if your vectors are normalized, and `euclidean` is also available.
- The `filter` field on `metadata.subject` makes it possible to restrict a search to one subject, for example only Physics nodes.

### How it is created (automatic)

`ingest.py → ensure_vector_index()` runs after the nodes are inserted. If an index with this name doesn't exist yet, it creates one with `pymongo`:

```python
from pymongo.operations import SearchIndexModel

model = SearchIndexModel(
    name="vector_index",
    type="vectorSearch",
    definition={"fields": [
        {"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine"},
        {"type": "filter", "path": "metadata.subject"},
    ]},
)
collection.create_search_index(model)
```

Atlas builds the index in the background, which takes about a minute for small collections. Until its status is **READY**, queries return no results.

### How to create it manually

**Option A – Atlas UI**
1. Atlas → your cluster → **Atlas Search** → **Create Search Index**.
2. Choose **Atlas Vector Search → JSON Editor**.
3. Select database `edu_rag` and collection `nodes`, and name the index `vector_index`.
4. Paste the JSON definition above → **Create**.

**Option B – mongosh**

```javascript
use edu_rag
db.nodes.createSearchIndex(
  "vector_index",
  "vectorSearch",
  {
    fields: [
      { type: "vector", path: "embedding", numDimensions: 768, similarity: "cosine" },
      { type: "filter", path: "metadata.subject" }
    ]
  }
)

// check status (wait for "READY")
db.nodes.getSearchIndexes()
```

### How it is queried

At query time, LlamaIndex embeds the question with the same 768‑d model and runs a `$vectorSearch` aggregation. It is roughly equivalent to:

```javascript
db.nodes.aggregate([
  {
    $vectorSearch: {
      index: "vector_index",
      path: "embedding",
      queryVector: [/* 768 floats */],
      numCandidates: 10,   // ANN candidates considered (LlamaIndex default: top_k × oversampling factor)
      limit: 1             // top_k
    }
  },
  { $project: { text: 1, metadata: 1, score: { $meta: "vectorSearchScore" } } }
])
```

Atlas reports the score for cosine similarity normalized to the range **[0, 1]** (`(1 + cosine) / 2`). Higher means more similar.

### Changing the embedding model

If you switch `EMBED_MODEL` to one with a different output size (for example a 1536‑d OpenAI embedding, the other size mentioned in the paper):
1. Update `EMBED_DIM`.
2. Drop the `nodes` collection and `vector_index` in Atlas.
3. Re-run `python app.py ingest ...`.

---

## Evaluation metrics

| Metric | Stage | What it measures |
|---|---|---|
| **Faithfulness** | Generation | Fraction of claims in the answer that can be inferred from the retrieved context. `|supported claims| / |claims|` |
| **Answer relevancy** | Generation | LLM generates questions from the answer; mean cosine similarity with the original question. |
| **Context recall** | Retrieval | Fraction of ground-truth sentences attributable to the retrieved context. |
| **Context precision** | Retrieval | Mean precision@k over retrieved chunks — are the relevant chunks ranked at the top? |

Results reported in the paper (top-k = 1):

| Metric | GPT-4 | GPT-4o mini | Gemini 1.5 |
|---|---|---|---|
| Faithfulness | 0.6516 | 0.7044 | 0.4583 |
| Answer relevancy | 0.9779 | 0.9838 | 0.1201 |
| Context recall | 0.6154 | 0.6575 | 0.6875 |
| Context precision | 0.8756 | 0.7676 | 0.5000 |

---

## Configuration reference

All settings live in `config.py`:

| Setting | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | placeholder | Required |
| `MONGODB_URI` | placeholder | Required |
| `DB_NAME` / `COLLECTION_NAME` | `edu_rag` / `nodes` | MongoDB location of the nodes |
| `VECTOR_INDEX_NAME` | `vector_index` | Atlas Vector Search index name |
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Any HuggingFace sentence encoder; update `EMBED_DIM` to match |
| `EMBED_DIM` | `768` | Must match the embedding model and the Atlas index |
| `LLM_MODEL` | `gpt-4o-mini` | Generator model |
| `EVAL_LLM_MODEL` | `gpt-4o-mini` | RAGAS judge model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `50` | Tokens |
| `TOP_K` | `1` | Retrieved chunks per question |
| `QA_PROMPT`, `QGEN_PROMPT` | see file | Prompt templates |

> If you change `EMBED_MODEL` / `EMBED_DIM`, drop the collection and its search index in Atlas and re-ingest.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Empty answer / "I don't know" right after first ingest | Wait ~1 minute for the Atlas vector index to finish building. |
| `ServerSelectionTimeoutError` | Your IP is not in Atlas **Network Access**, or the URI is wrong. |
| `OperationFailure: ... search index` | Cluster tier doesn't support Atlas Search, or the index already exists with another definition — delete it in the Atlas UI and re-run ingest. |
| `AuthenticationError` from OpenAI | Set a valid `OPENAI_API_KEY` in `config.py`. |
| Scanned PDFs produce no text | `pypdf` can't OCR images; run OCR first (e.g. `ocrmypdf in.pdf out.pdf`) or plug in LlamaParse. |
| `ValueError: LLM did not return valid JSON` | Retry, lower `--n`, or use a stronger model with `--llm gpt-4o`. |

---

## Limitations

As discussed in the paper:

- **Hallucination** — LLMs can still produce incorrect content despite relevant context.
- **Retrieval dependence** — answer quality is bounded by what is retrieved; `top_k = 1` can miss information spread across chunks.
- **Cost / resources** — LLM calls and RAGAS evaluation consume API credits.
- **No OCR / layout parsing** — tables and figures are not extracted (the paper used LlamaParse for this).

---

## Citation

```bibtex
@article{pardasani2025edurag,
  title   = {Knowledge Retrieval-Based Intelligent Question and Answer Generation Framework for Education},
  author  = {Pardasani, Anay and Maleski, Karen and Roy, Sayantan},
  journal = {Journal of Student Research},
  volume  = {14},
  number  = {1},
  year    = {2025}
}
```
