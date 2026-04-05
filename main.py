import os

from config import (
    TOP_K,
    MODE,
    OPENROUTER_MODEL,
    MAX_QUERY_CHARS,
    MIN_RETRIEVAL_SCORE,
)
from data_loader import load_faq_file
from embedder import get_embed_model
from ingest import ingest_faq_records
from indexing import setup, reindex
from llm_handler import load_llm
from orchestrator import handle_customer_query


def print_banner():
    print("=" * 60)
    print("  CS416 · Bank LLM Prototype")
    print(f"  Mode : {MODE.upper()}  |  OpenRouter: {OPENROUTER_MODEL}")
    print("=" * 60)


def print_sources(retrieved: list[dict]):
    print("\n  --- Retrieved Sources ---")
    for i, r in enumerate(retrieved, 1):
        print(f"  [{i}] Sheet: {r['metadata']['sheet']}  |  "
              f"Similarity: {r['score']:.2f}")
        print(f"       Q: {r['metadata']['question'][:75]}...")
    print("  -------------------------")


def main():
    print_banner()
    embed_model = get_embed_model()
    collection = setup(embed_model)
    llm_client = load_llm()

    print("\n[main] System ready!")
    print("=" * 60)
    print("  Commands:  'reindex' | 'ingest <file.csv|xlsx>' | 'quit'")
    print("=" * 60)

    while True:
        print()
        query = input("Customer: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if query.lower() == "reindex":
            collection = reindex(embed_model)
            continue

        if query.lower().startswith("ingest "):
            path = query[7:].strip().strip('"').strip("'")
            if not path or not os.path.isfile(path):
                print("Usage: ingest <path/to/file.csv or .xlsx>")
                continue
            base = os.path.basename(path)
            raw = load_faq_file(path, upload_label=f"File:{base}")
            if not raw:
                print("[ingest] No records loaded (check format and columns).")
                continue
            n, msg = ingest_faq_records(
                raw,
                collection,
                embed_model,
                default_sheet=f"File:{base}",
            )
            print(f"[ingest] {msg.replace('**', '')}")
            continue

        result = handle_customer_query(
            query,
            collection,
            embed_model,
            llm_client,
            top_k=TOP_K,
            max_query_length=MAX_QUERY_CHARS,
            min_retrieval_score=MIN_RETRIEVAL_SCORE,
        )
        print("\nAssistant: ", end="", flush=True)
        if result["blocked"]:
            print(result["user_message"])
        else:
            print(result["answer"])
        if result["retrieved"]:
            print_sources(result["retrieved"])


if __name__ == "__main__":
    main()
