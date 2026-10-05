import os
from dotenv import load_dotenv

load_dotenv()

NUGEN_API_KEY = os.getenv("NUGEN_API_KEY")
NUGEN_EMBEDDING_MODEL = os.getenv("NUGEN_EMBEDDING_MODEL")
NUGEN_LLM_MODEL = os.getenv("NUGEN_LLM_MODEL")
#NUGEN_RERANKER_MODEL = os.getenv("NUGEN_RERANKER_MODEL")

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "rag_documents"
)

if not NUGEN_API_KEY:
    raise ValueError("NUGEN_API_KEY is missing in .env")

if not NUGEN_EMBEDDING_MODEL:
    raise ValueError("NUGEN_EMBEDDING_MODEL is missing in .env")

if not NUGEN_LLM_MODEL:
    raise ValueError("NUGEN_LLM_MODEL is missing in .env")

