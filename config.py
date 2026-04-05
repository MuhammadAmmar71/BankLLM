import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (next to this config file)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

MODE = "dev"

# LLM via OpenRouter (OpenAI-compatible API). Set OPENROUTER_API_KEY in `.env`.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
# Optional OpenRouter attribution (https://openrouter.ai/docs)
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "Bank LLM Assistant")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# For downloading embedding models from Hugging Face Hub (not used for OpenRouter LLM).
HF_TOKEN = os.getenv("HF_TOKEN", "")

DATASET_PATH = "dataset.xlsx"

# FAQs added via UI or tooling (merged on rebuild; appended live to Chroma).
SUPPLEMENTAL_FAQS_PATH = "./data/supplemental_faqs.jsonl"

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "bank_faqs"

TOP_K = 3

# Guardrails: max input size; minimum best cosine similarity (see embedder.retrieve score).
MAX_QUERY_CHARS        = 4000
MIN_RETRIEVAL_SCORE    = 0.28
# Uploaded single-document Q&A uses shorter chunks; threshold can be lower.
SESSION_DOC_MIN_SCORE  = 0.12

MAX_NEW_TOKENS = 300

SYSTEM_PROMPT = """You are a helpful and professional customer service assistant for a bank.
Answer the customer's question based ONLY on the provided context from the bank's knowledge base.
If the context does not contain enough information to answer, politely say you don't have that
information and suggest the customer contact the bank directly.
Do not make up information. Keep answers clear, concise, and professional."""
