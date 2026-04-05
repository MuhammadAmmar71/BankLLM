import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME


def persistent_chroma_client():
    """
    Local on-disk Chroma client. If you see 'default_tenant', the chroma_db folder
    is usually from another Chroma version or corrupt — delete it and rebuild the index.
    """
    path = str(Path(CHROMA_DIR).resolve())
    try:
        return chromadb.PersistentClient(path=path)
    except Exception as e:
        err = str(e).lower()
        if "tenant" in err or "default_tenant" in str(e):
            raise RuntimeError(
                "ChromaDB could not open its local database (tenant error). The folder "
                f"{CHROMA_DIR!r} was likely built with a different ChromaDB version or is corrupt.\n\n"
                "Fix: stop the app, delete that folder, then start again so the index rebuilds "
                "(e.g. rm -rf chroma_db). With the CLI, run python main.py and type reindex if needed."
            ) from e
        raise


def get_embed_model() -> SentenceTransformer:
    """Load the sentence embedding model."""
    print(f"[embedder] Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def build_index(records: list[dict], embed_model: SentenceTransformer) -> chromadb.Collection:
    """
    Embeds all records and stores them in a fresh ChromaDB collection.
    Called on first run or when reindexing is requested.
    """
    print(f"[embedder] Building index for {len(records)} records...")

    client = persistent_chroma_client()

    # Drop existing collection so we start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    documents  = [r["document"] for r in records]
    metadatas  = [{"question": r["question"], "answer": r["answer"], "sheet": r["sheet"]}
                  for r in records]
    ids        = [f"doc_{i}" for i in range(len(records))]

    # Embed in batches to avoid memory spikes
    batch_size    = 64
    all_embeddings = []

    for i in range(0, len(documents), batch_size):
        batch      = documents[i : i + batch_size]
        embeddings = embed_model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(embeddings)

    collection.add(
        documents  = documents,
        embeddings = all_embeddings,
        metadatas  = metadatas,
        ids        = ids
    )

    print(f"[embedder] Indexed {len(records)} documents into ChromaDB at '{CHROMA_DIR}'")
    return collection


def append_to_collection(
    collection: chromadb.Collection,
    records: list[dict],
    embed_model: SentenceTransformer,
    id_prefix: str = "sup",
) -> int:
    """
    Embed and add new FAQ rows without rebuilding the index.
    Each record must have document, question, answer, sheet (as returned by preprocess).
    """
    if not records:
        return 0
    documents = [r["document"] for r in records]
    metadatas = [
        {"question": r["question"], "answer": r["answer"], "sheet": r["sheet"]}
        for r in records
    ]
    ids = [f"{id_prefix}_{uuid.uuid4().hex}" for _ in records]
    batch_size = 64
    all_embeddings: list = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        all_embeddings.extend(
            embed_model.encode(batch, show_progress_bar=False).tolist()
        )
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )
    print(f"[embedder] Appended {len(records)} documents to collection '{COLLECTION_NAME}'")
    return len(records)


def load_index(embed_model: SentenceTransformer) -> chromadb.Collection:
    """
    Loads an existing ChromaDB collection without rebuilding.
    """
    client = persistent_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)
    print(f"[embedder] Loaded existing index — {collection.count()} documents")
    return collection


def retrieve(
    query:       str,
    collection:  chromadb.Collection,
    embed_model: SentenceTransformer,
    top_k:       int
) -> list[dict]:
    """
    Embeds the user query and returns the top_k most similar records.
    Each result contains the document text, metadata, and similarity score.
    """
    query_embedding = embed_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = top_k
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score":    1 - results["distances"][0][i]   # cosine similarity
        })

    return retrieved
