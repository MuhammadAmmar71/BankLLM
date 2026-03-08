import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (next to this config file)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

MODE = "dev"   

MODELS = {
    "dev":  "Qwen/Qwen2.5-3B-Instruct",
    "prod": "meta-llama/Llama-3.2-3B-Instruct"
}
LLM_MODEL   = MODELS[MODE]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HF_TOKEN = os.getenv("HF_TOKEN", "")

DATASET_PATH = "dataset.xlsx"   

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "bank_faqs"

TOP_K = 3   

MAX_NEW_TOKENS     = 300
REPETITION_PENALTY = 1.1

SYSTEM_PROMPT = """You are a helpful and professional customer service assistant for a bank.
Answer the customer's question based ONLY on the provided context from the bank's knowledge base.
If the context does not contain enough information to answer, politely say you don't have that
information and suggest the customer contact the bank directly.
Do not make up information. Keep answers clear, concise, and professional."""
