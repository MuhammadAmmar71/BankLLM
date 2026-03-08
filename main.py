import os
from config      import CHROMA_DIR, TOP_K, MODE, LLM_MODEL
from data_loader import load_dataset
from preprocessor import preprocess
from embedder    import get_embed_model, build_index, load_index, retrieve
from llm_handler import load_llm, generate_answer


def print_banner():
    print("=" * 60)
    print("  CS416 · Bank LLM Prototype")
    print(f"  Mode : {MODE.upper()}  |  Model: {LLM_MODEL.split('/')[-1]}")
    print("=" * 60)


def print_sources(retrieved: list[dict]):
    print("\n  --- Retrieved Sources ---")
    for i, r in enumerate(retrieved, 1):
        print(f"  [{i}] Sheet: {r['metadata']['sheet']}  |  "
              f"Similarity: {r['score']:.2f}")
        print(f"       Q: {r['metadata']['question'][:75]}...")
    print("  -------------------------")


def setup(embed_model):
    index_exists = os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR)
    if index_exists:
        print("\n[main] Existing index found — skipping re-embedding.")
        collection = load_index(embed_model)
    else:
        print("\n[main] No index found — building from scratch...")
        records    = load_dataset()
        records    = preprocess(records)
        collection = build_index(records, embed_model)
    return collection


def reindex(embed_model):
    print("\n[main] Re-indexing dataset...")
    records    = load_dataset()
    records    = preprocess(records)
    collection = build_index(records, embed_model)
    print("[main] Re-indexing complete.\n")
    return collection


def main():
    print_banner()
    embed_model  = get_embed_model()
    collection   = setup(embed_model)
    llm_pipeline = load_llm()

    print("\n[main] System ready!")
    print("=" * 60)
    print("  Commands:  'reindex' | 'quit'")
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

        retrieved = retrieve(query, collection, embed_model, top_k=TOP_K)
        print("\nAssistant: ", end="", flush=True)
        answer = generate_answer(query, retrieved, llm_pipeline)
        print(answer)
        print_sources(retrieved)


if __name__ == "__main__":
    main()
