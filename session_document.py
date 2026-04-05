"""
Optional session-only RAG over a user-uploaded document (Streamlit).
"""
import io
import re
from pathlib import Path

import numpy as np


def read_document_bytes(data: bytes, filename: str) -> str:
    """Extract plain text from supported uploads."""
    suf = Path(filename).suffix.lower()
    if suf in (".txt", ".md", ".csv"):
        return data.decode("utf-8", errors="replace")

    if suf == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        return "\n\n".join(parts)

    if suf == ".docx":
        import docx

        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    raise ValueError(f"Unsupported type {suf!r}. Use PDF, Word (.docx), TXT, MD, or CSV.")


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = max(i + max_chars - overlap, i + 1)
    return chunks


def build_session_rag_records(
    text: str,
    source_name: str,
    embed_model,
    max_chars: int = 900,
    overlap: int = 120,
) -> list[dict]:
    """
    Chunk document, embed each chunk. Each item:
      document, metadata {source, chunk_index}, embedding (list[float])
    """
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    if not chunks:
        return []
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    if hasattr(embeddings, "tolist"):
        emb_list = embeddings.tolist()
    else:
        emb_list = [e.tolist() for e in embeddings]
    out: list[dict] = []
    for i, (chunk, emb) in enumerate(zip(chunks, emb_list)):
        out.append(
            {
                "document": chunk,
                "metadata": {
                    "source": source_name,
                    "chunk_index": i,
                    "sheet": source_name,
                    "question": f"Section {i + 1}",
                },
                "embedding": emb,
            }
        )
    return out


def retrieve_from_session_rag(
    query: str,
    embed_model,
    records: list[dict],
    top_k: int,
) -> list[dict]:
    """Same shape as embedder.retrieve: document, metadata, score (cosine similarity)."""
    if not records:
        return []
    q_emb = np.array(embed_model.encode([query], show_progress_bar=False)[0], dtype=np.float64)
    qn = np.linalg.norm(q_emb)
    if qn == 0:
        qn = 1.0

    scored: list[tuple[float, dict]] = []
    for r in records:
        emb = np.array(r["embedding"], dtype=np.float64)
        en = np.linalg.norm(emb)
        if en == 0:
            continue
        sim = float(np.dot(q_emb, emb) / (qn * en))
        scored.append((sim, r))

    scored.sort(key=lambda x: -x[0])
    top = scored[:top_k]
    return [
        {
            "document": r["document"],
            "metadata": dict(r["metadata"]),
            "score": s,
        }
        for s, r in top
    ]
