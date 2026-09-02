import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


# ============================================================
# Configuration
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("harry-potter-rag")


def get_required_env(name: str) -> str:
    """
    Load a required environment variable.
    Raise a clear error if it is missing.
    """

    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


QDRANT_URL = get_required_env("QDRANT_URL")
QDRANT_API_KEY = get_required_env("QDRANT_API_KEY")
QDRANT_COLLECTION = get_required_env("QDRANT_COLLECTION")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

GEMINI_MODEL = get_required_env("GEMINI_MODEL")
GEMINI_API_KEY = get_required_env("GEMINI_API_KEY")

GROQ_MODEL = get_required_env("GROQ_MODEL")
GROQ_API_KEY = get_required_env("GROQ_API_KEY")

# TOP_K: default to 3 if not provided or invalid
try:
    TOP_K = int(os.getenv("TOP_K", "3"))
except ValueError:
    TOP_K = 3


# ============================================================
# Prompts
# ============================================================

ROUTER_SYSTEM_PROMPT = """
You are the query router for a Harry Potter book
question-answering system.

Classify the user's message into exactly ONE of these labels:

retrieve
- Questions about the Harry Potter books.
- This includes characters, relationships, events, plot details,
  locations, creatures, objects, spells, chapters, and other
  information that can be answered from the books.

chitchat
- Greetings, thanks, or casual conversation.
- Examples: "hello", "hi", "thanks", "how are you?"

off-topic
- Anything unrelated to the Harry Potter books.
- Examples: programming, mathematics, sports, news, or unrelated
  factual questions.

Return exactly ONE label and nothing else:

retrieve
chitchat
off-topic

Do not explain your decision.
Do not answer the user's question.
"""


CHITCHAT_SYSTEM_PROMPT = """
You are a friendly assistant for a Harry Potter book
question-answering system.

Respond naturally to simple greetings, thanks, and casual
conversation.

Keep the response short and friendly.

Do not answer unrelated factual questions.
"""


RAG_SYSTEM_PROMPT = """
You are the answer-generation model for a Harry Potter
book question-answering system.

Your task is to answer the user's question using ONLY the
retrieved context provided to you.

Rules:

1. Use only the provided retrieved context.
2. Do not rely on outside knowledge.
3. Do not invent facts, events, characters, or sources.
4. If the context does not contain enough information to
   answer the question, say:

   "I do not know based on the retrieved book context."

5. Give a clear and concise answer.
6. DO NOT include any citations or sources in your answer.
7. The sources will be displayed separately.
8. Just provide the answer text without any source references.

The retrieved context contains information extracted from
the original Harry Potter books.
"""
# ============================================================
# Global objects
# ============================================================

embedding_model = None
qdrant_client = None
gemini_llm = None
groq_llm = None


# ============================================================
# Startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global embedding_model
    global qdrant_client
    global gemini_llm
    global groq_llm

    logger.info("Starting Harry Potter RAG API...")

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load embedding model")
        raise RuntimeError(
            f"Unable to load embedding model '{EMBEDDING_MODEL}': {e}\n"
            "Install the model or set a different EMBEDDING_MODEL in .env"
        )

    # --------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------

    logger.info(
        "Connecting to Qdrant..."
    )

    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    # Verify collection exists
    if not qdrant_client.collection_exists(
        QDRANT_COLLECTION
    ):
        raise RuntimeError(
            f"Qdrant collection "
            f"'{QDRANT_COLLECTION}' does not exist."
        )

    logger.info(
        f"Connected to Qdrant collection: "
        f"{QDRANT_COLLECTION}"
    )

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    gemini_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        max_retries=2,
    )

    logger.info(
        f"Gemini model initialized: {GEMINI_MODEL}"
    )

    # --------------------------------------------------------
    # Groq
    # --------------------------------------------------------

    groq_llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    logger.info(
        f"Groq router initialized: {GROQ_MODEL}"
    )

    logger.info(
        "Harry Potter RAG API is ready."
    )

    yield

    logger.info(
        "Shutting down Harry Potter RAG API..."
    )


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Harry Potter RAG API",
    description=(
        "A Retrieval-Augmented Generation API for "
        "answering questions from the Harry Potter books."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Schemas
# ============================================================

class QueryRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question",
    )


class Source(BaseModel):

    chunk_id: str

    book_name: str

    page_start: int

    page_end: int

    score: float


class QueryResponse(BaseModel):

    query: str

    route: str

    answer: str

    sources: list[Source]


# ============================================================
# Helper Functions
# ============================================================

def llm_response_to_text(response) -> str:
    """
    Normalize different LLM response shapes into a plain string.

    Some adapters return `response.content` as a string, others
    return a list of message dicts like `[{"type":"text","text": "..."}]`.
    This helper extracts and concatenates text parts safely.
    """

    if response is None:
        return ""

    content = getattr(response, "content", None)

    # Already a string
    if isinstance(content, str):
        return content

    # Some adapters return a list of message dicts
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # common keys: 'text' or 'content'
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))

        return "".join(parts)

    # Fallback to str()
    return str(content)


def clean_route(route: str) -> str:
    """
    Clean and validate the router output.
    """

    route = (
        route
        .strip()
        .lower()
        .splitlines()[0]
        .strip(" `.,:")
    )

    if route not in {
        "retrieve",
        "chitchat",
        "off-topic",
    }:
        return "off-topic"

    return route


def route_query(query: str) -> str:
    """
    Use Groq to classify the user query.
    """

    response = groq_llm.invoke([
        SystemMessage(
            content=ROUTER_SYSTEM_PROMPT
        ),

        HumanMessage(
            content=query
        ),
    ])

    text = llm_response_to_text(response)

    return clean_route(text)


def retrieve_chunks(
    query: str,
    top_k: int = TOP_K,
):
    """
    Generate an E5 query embedding and retrieve
    the most relevant semantic chunks from Qdrant.
    """

    # E5 uses "query:" for queries.
    query_text = f"query: {query}"

    query_vector = embedding_model.encode(
        query_text,
        normalize_embeddings=True,
    ).tolist()

    results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,

        query=query_vector,

        limit=top_k,

        with_payload=True,
    ).points

    return results


def format_context(results) -> str:
    """
    Format retrieved semantic chunks for Gemini.
    """

    context_parts = []

    for index, result in enumerate(results, start=1):

        payload = result.payload

        chunk_id = payload.get(
            "chunk_id",
            f"chunk_{index}",
        )

        book = payload.get(
            "book",
            "Unknown Book",
        )

        page_start = payload.get(
            "page_start",
            "Unknown",
        )

        page_end = payload.get(
            "page_end",
            page_start,
        )

        text = payload.get(
            "text",
            "",
        )

        if page_start == page_end:
            page_reference = f"page {page_start}"
        else:
            page_reference = f"pages {page_start}-{page_end}"

        context_parts.append(
            f"""
[Source {index}]
Chunk ID: {chunk_id}
Book: {book}
Pages: {page_reference}

Text:
{text}
"""
        )

    return "\n".join(context_parts)

def generate_answer(
    query: str,
    results,
) -> str:
    """
    Generate a grounded answer using Gemini.
    """

    context = format_context(
        results
    )

    user_prompt = f"""
Retrieved context:

{context}

User question:

{query}

Answer the question using ONLY the retrieved context.
If the context is insufficient, say:
"I do not know based on the retrieved book context."

DO NOT include any citations, sources, or references in your answer.
Just provide the factual answer based on the context.
"""

    response = gemini_llm.invoke([
        SystemMessage(
            content=RAG_SYSTEM_PROMPT
        ),

        HumanMessage(
            content=user_prompt
        ),
    ])

    return llm_response_to_text(response)


def generate_chitchat_response(
    query: str,
) -> str:
    """
    Generate a short response for casual conversation.
    """

    response = groq_llm.invoke([
        SystemMessage(
            content=CHITCHAT_SYSTEM_PROMPT
        ),

        HumanMessage(
            content=query
        ),
    ])

    return llm_response_to_text(response)


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():

    return {
        "name": "Harry Potter RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():

    try:

        collection_exists = (
            qdrant_client.collection_exists(
                QDRANT_COLLECTION
            )
        )

        return {
            "status": "ok",
            "qdrant": "connected",
            "collection": QDRANT_COLLECTION,
            "collection_exists": collection_exists,
            "embedding_model": EMBEDDING_MODEL,
        }

    except Exception as e:

        return {
            "status": "degraded",
            "error": str(e),
        }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    query = request.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    # ========================================================
    # 1. ROUTING
    # ========================================================

    try:

        route = route_query(
            query
        )

        logger.info(
            f"Query route: {route}"
        )

    except Exception as e:

        logger.exception(
            "Router error"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "The query routing service "
                "is temporarily unavailable."
            ),
        )


    # ========================================================
    # 2. CHITCHAT
    # ========================================================

    if route == "chitchat":

        try:

            answer = (
                generate_chitchat_response(
                    query
                )
            )

            return QueryResponse(
                query=query,
                route=route,
                answer=answer,
                sources=[],
            )

        except Exception as e:

            logger.exception(
                "Chitchat generation error"
            )

            raise HTTPException(
                status_code=502,
                detail=(
                    "The conversational model "
                    "is temporarily unavailable."
                ),
            )


    # ========================================================
    # 3. OFF-TOPIC
    # ========================================================

    if route == "off-topic":

        return QueryResponse(
            query=query,
            route=route,
            answer=(
                "I'm sorry, I don't know the answer to that question. "
                "I can only answer questions about the Harry Potter books."
            ),
            sources=[],
        )


    # ========================================================
    # 4. RETRIEVAL
    # ========================================================

    try:

        results = retrieve_chunks(
            query,
            top_k=TOP_K,
        )

        logger.info(
            f"Retrieved {len(results)} chunks."
        )

    except Exception as e:

        logger.exception(
            "Qdrant retrieval error"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "The retrieval service "
                "is temporarily unavailable."
            ),
        )


    # ========================================================
    # 5. NO RETRIEVED RESULTS
    # ========================================================

    if not results:

        return QueryResponse(
            query=query,
            route=route,
            answer=(
                "I do not know based on "
                "the retrieved book context."
            ),
            sources=[],
        )


    # ========================================================
    # 6. GENERATION
    # ========================================================

    try:

        answer = generate_answer(
            query,
            results,
        )

    except Exception as e:

        logger.exception(
            "Gemini generation error"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The answer generation service "
                "is temporarily unavailable. "
                "Please try again shortly."
            ),
        )


    # ========================================================
    # 7. SOURCES
    # ========================================================

    sources = []

    for result in results:

        payload = result.payload

        sources.append(
            Source(

                chunk_id=str(
                    payload.get(
                        "chunk_id",
                        "unknown",
                    )
                ),

                book_name=str(
                    payload.get(
                        "book",
                        "Unknown",
                    )
                ),

                page_start=int(
                    payload.get(
                        "page_start",
                        0,
                    )
                ),

                page_end=int(
                    payload.get(
                        "page_end",
                        payload.get(
                            "page_start",
                            0,
                        ),
                    )
                ),

                score=float(
                    result.score
                ),
            )
        )


    # ========================================================
    # 8. FINAL RESPONSE
    # ========================================================

    return QueryResponse(
        query=query,
        route=route,
        answer=answer,
        sources=sources,
    )