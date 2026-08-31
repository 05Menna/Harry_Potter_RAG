# Harry Potter RAG Project

A Retrieval-Augmented Generation (RAG) system for answering questions about the **Harry Potter books** using semantic retrieval, vector search, query routing, and grounded answer generation.

## Live Frontend

The deployed frontend is available at:

**(https://harry-potter-rag.vercel.app/)**

The frontend communicates with the FastAPI backend through the `/query` endpoint.

---

## Project Architecture

The overall workflow is:

```text
                         ┌──────────────────────┐
                         │   Harry Potter PDF   │
                         │    harrypotter.pdf   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Text Extraction    │
                         │      + Cleaning      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Semantic Chunking   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Chunk JSON Payloads │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   E5 Embeddings     │
                         │ multilingual-e5-large│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Qdrant Cloud     │
                         │    Vector Database   │
                         └──────────┬───────────┘
                                    │
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
              ▼                                           ▼
      ┌────────────────┐                         ┌────────────────┐
      │ Vercel Frontend│                         │  FastAPI API   │
      │                │                         │    /query      │
      └────────────────┘                         └───────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │   Groq Router   │
                                                └────────┬────────┘
                                                         │
                              ┌──────────────────────────┼─────────────────────────┐
                              │                          │                         │
                              ▼                          ▼                         ▼
                         retrieve                   chitchat                  off-topic
                              │                          │                         │
                              ▼                          ▼                         ▼
                       E5 Embedding                Groq Response          Fixed Response
                              │
                              ▼
                         Qdrant Search
                              │
                              ▼
                         Top-K Chunks
                              │
                              ▼
                       Gemini Generation
                              │
                              ▼
                      Grounded Answer
                              │
                              ▼
                       Source Metadata
```

---

## RAG Workflow

For each user question, the backend follows these steps:

### 1. Query Routing

The user's question is first sent to a Groq-powered router.

The router classifies the message into exactly one of:

```text
retrieve
chitchat
off-topic
```

### `retrieve`

Used for questions about the Harry Potter books, including:

* Characters
* Relationships
* Events
* Plot details
* Locations
* Creatures
* Objects
* Spells
* Chapters
* Other information contained in the books

### `chitchat`

Used for casual interaction such as:

```text
Hello
Hi
Thanks
How are you?
```

These requests receive a short conversational response.

### `off-topic`

Used for questions unrelated to the Harry Potter books.

Examples include:

```text
Programming questions
Mathematics
Sports
News
Unrelated factual questions
```

These requests are rejected without performing retrieval.

---

## 2. Query Embedding

For `retrieve` queries, the backend generates an embedding using Sentence Transformers.

Current model:

```text
intfloat/multilingual-e5-large
```

The query is formatted using the E5 query prefix:

```text
query: <user question>
```

Embeddings are normalized before retrieval.

---

## 3. Vector Search

The generated query vector is sent to Qdrant Cloud.

The configured collection is:

```text
harry_potter
```

The system retrieves the top-K most relevant semantic chunks.

Default:

```text
TOP_K=3
```

Each retrieved chunk contains metadata such as:

* Chunk ID
* Book name
* Page start
* Page end
* Chunk text
* Similarity score

---

## 4. Context Construction

The retrieved chunks are formatted into a context passed to the generation model.

The context includes the relevant book text and metadata, for example:

```text
[Source 1]

Chunk ID: ...
Book: ...
Pages: ...

Text:
...
```

---

## 5. Answer Generation

The retrieved context and the user's question are sent to Gemini.

The generation model is instructed to:

1. Use only the retrieved context.
2. Avoid outside knowledge.
3. Avoid inventing facts.
4. Return a concise answer.
5. State that the information is unavailable when the retrieved context is insufficient.
6. Keep source information separate from the generated answer.

The API returns the answer together with the retrieved source metadata.

---

## Dataset

The Harry Potter books were processed from the source PDF:

```text
data/harrypotter.pdf
```

The processed dataset contains approximately:

```text
Total pages:       3,623
Total characters:  6,273,981
Total words:       1,121,939
Empty pages:       19
Low-text pages:    17
```

The processing pipeline is:

```text
PDF
 ↓
Text Extraction
 ↓
Cleaning
 ↓
Semantic Chunking
 ↓
Chunk JSON
 ↓
Embeddings
 ↓
Qdrant
```

The chunk payloads are stored under:

```text
dataset_chunks/
```

---

## Project Structure

```text
Harry_Potter_RAG/
│
├── backend/
│   ├── rag_api.py
│   └── requirements.txt
│
├── data/
│   ├── harrypotter.pdf
│   └── output.md
│
├── dataset_chunks/
│   └── precomputed chunk JSON files
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── notebooks/
│   └── RAG_Final_Project.ipynb
│
└── README.md
```

---

## Required Environment Variables

Create a `.env` file for local development.

The backend requires:

```text
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION

EMBEDDING_MODEL

GEMINI_MODEL
GEMINI_API_KEY

GROQ_MODEL
GROQ_API_KEY

TOP_K
```

Example configuration:

```env
QDRANT_URL=<your-qdrant-url>
QDRANT_API_KEY=<your-qdrant-api-key>
QDRANT_COLLECTION=harry_potter

EMBEDDING_MODEL=intfloat/multilingual-e5-large

GEMINI_MODEL=<your-gemini-model>
GEMINI_API_KEY=<your-gemini-api-key>

GROQ_MODEL=<your-groq-model>
GROQ_API_KEY=<your-groq-api-key>

TOP_K=3
```

Do not commit real API keys to GitHub.

---

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the API Locally

Start the FastAPI application with Uvicorn:

```bash
uvicorn rag_api:app --reload --host 0.0.0.0 --port 8000
```

The API exposes:

```text
GET  /
GET  /health
POST /query
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## API Endpoints

### `GET /`

Returns basic API information and status.

Example:

```json
{
  "name": "Harry Potter RAG API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs"
}
```

### `GET /health`

Checks API health and Qdrant connectivity.

### `POST /query`

Main RAG endpoint.

Request:

```json
{
  "query": "Who gave Harry the Marauder's Map?"
}
```

The endpoint returns:

```json
{
  "query": "Who gave Harry the Marauder's Map?",
  "route": "retrieve",
  "answer": "...",
  "sources": [
    {
      "chunk_id": "...",
      "book_name": "...",
      "page_start": 0,
      "page_end": 0,
      "score": 0.0
    }
  ]
}
```

---

## Example Using cURL

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Who gave Harry the Marauder's Map?\"}"
```

---

## Development / Notebook Workflow

The development notebook is:

```text
notebooks/RAG_Final_Project.ipynb
```

The main workflow is:

1. Load and inspect the Harry Potter PDF.
2. Extract and clean the text.
3. Perform semantic chunking.
4. Create chunk payloads.
5. Generate embeddings.
6. Upload vectors and metadata to Qdrant.
7. Test retrieval.
8. Test answer generation.
9. Evaluate sample questions.
10. Persist and reuse the vector database.

The API then uses the already-created Qdrant collection for real-time retrieval.

---

## Frontend Workflow

The frontend is deployed separately using Vercel.

Live application:

**https://harry-potter-h907p4he4-menna24.vercel.app/**

The frontend sends user questions to the FastAPI backend through:

```text
POST /query
```

The backend performs:

```text
User Question
     ↓
Groq Router
     ↓
Route
     │
     ├── chitchat → Groq
     │
     ├── off-topic → Fixed response
     │
     └── retrieve
            ↓
         E5 Embedding
            ↓
          Qdrant
            ↓
        Top-K Chunks
            ↓
          Gemini
            ↓
         Answer
```

---

## Security

Do not commit `.env` files or API keys.

Add `.env` to `.gitignore`:

```text
.env
.venv/
__pycache__/
```

API keys should be configured through environment variables or the deployment platform's secret/environment-variable system.

The original Harry Potter PDF contains copyrighted material. Avoid committing or distributing the complete copyrighted book text unnecessarily.

---

## Troubleshooting

### Embedding Model Fails to Load

Check:

```text
EMBEDDING_MODEL
```

and make sure the deployment environment can download the model.

The current model is:

```text
intfloat/multilingual-e5-large
```

For deployment environments with limited RAM, a smaller E5 model may be considered.

### Qdrant Connection Fails

Verify:

```text
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
```

Also make sure that the configured collection exists.

### Gemini Generation Fails

Check:

```text
GEMINI_MODEL
GEMINI_API_KEY
```

If the selected Gemini model is unavailable or temporarily unavailable, configure another supported model.

### Groq Routing Fails

Check:

```text
GROQ_MODEL
GROQ_API_KEY
```

The Groq model is responsible for routing queries and handling chitchat.

---

## Current Technology Stack

| Component           | Technology                       |
| ------------------- | -------------------------------- |
| Language            | Python                           |
| API                 | FastAPI                          |
| API Server          | Uvicorn                          |
| Query Router        | Groq                             |
| Generation          | Google Gemini                    |
| Embeddings          | Sentence Transformers            |
| Embedding Model     | `intfloat/multilingual-e5-large` |
| Vector Database     | Qdrant Cloud                     |
| Frontend            | HTML / CSS / JavaScript          |
| Frontend Deployment | Vercel                           |
| Development         | VS Code / Jupyter Notebook       |

##
