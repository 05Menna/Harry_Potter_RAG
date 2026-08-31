# Harry Potter RAG Project

A Retrieval-Augmented Generation (RAG) system for answering questions about the Harry Potter books.

**Contents**
- `rag_api.py` — FastAPI app providing a `/query` endpoint that routes, retrieves, and generates grounded answers.
- `data/` — source files and `output.md` (extracted text)
- `dataset_chunks/` — precomputed chunk JSON files (vector payloads)
- `frontend/` — minimal demo UI (`index.html`, `script.js`, `style.css`)
- `notebooks/` — Jupyter notebook used during development (`RAG_Final_Project.ipynb`)

## Required environment variables
Create a `.env` file in the project root with values for:

- `QDRANT_URL` — Qdrant host URL (e.g. `http://localhost:6333` or cloud endpoint)
- `QDRANT_API_KEY` — Qdrant API key (if required)
- `QDRANT_COLLECTION` — collection name used to store/retrieve vectors
- `GEMINI_MODEL` — Gemini model name used by `langchain_google_genai` adapter
- `GEMINI_API_KEY` — API key for Gemini (Google GenAI)
- `GROQ_MODEL` — Groq model name used by `langchain_groq` adapter
- `GROQ_API_KEY` — API key for Groq
- `EMBEDDING_MODEL` — sentence-transformers model id (default: `intfloat/multilingual-e5-large`)
- `TOP_K` — number of chunks to retrieve per query (default: `3`)

## Install

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Activate the venv (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Running the API

Start the FastAPI app with Uvicorn:

```bash
uvicorn rag_api:app --reload --host 0.0.0.0 --port 8000
```

The API will expose:
- `GET /` — basic info
- `GET /health` — health and connectivity
- `POST /query` — main RAG endpoint (JSON body: `{ "query": "your question" }`)

Open interactive docs at `http://localhost:8000/docs`.

## Notes & Security
- Do not commit the `.env` file — it contains secrets. Use `.gitignore` to exclude it.
- The project distributes or references the original Harry Potter PDF in `data/`. Avoid committing full copyrighted text.
- If you encounter provider import errors, install the corresponding SDK and ensure the adapter package versions are compatible with the installed `langchain-*` packages.

## Development/Notebook workflow
1. Use `notebooks/RAG_Final_Project.ipynb` to preprocess the PDF (`data/harrypotter.pdf`) into `data/output.md` and to create chunk payloads.
2. Upload or insert chunk vectors into Qdrant under the configured `QDRANT_COLLECTION`.
3. Start the API and query via the frontend or curl.

## Example query via curl

```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query":"Who gave Harry the Marauder\'s Map?"}'
```

## Troubleshooting
- If the embedding model fails to load, check `EMBEDDING_MODEL` in `.env` and ensure the model files are available or that you have internet access to download them.

- If Qdrant connectivity fails, confirm `QDRANT_URL` and `QDRANT_API_KEY` and that the collection exists.

- If Gemini API fails to response, try another model.

---