import re


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Strip leading/trailing whitespace
    - Collapse multiple spaces/newlines into one
    - Remove special characters (keep punctuation useful for NLP)
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\?\!\-\(\)]", "", text)
    return text


def preprocess(records: list[dict]) -> list[dict]:
    """
    Cleans all Q&A text and builds a combined 'document' string
    per record that will be used for embedding.

    Skips records where question or answer is empty after cleaning.
    """
    cleaned = []

    for r in records:
        q = clean_text(r["question"])
        a = clean_text(r["answer"])

        if not q or not a:
            continue

        cleaned.append({
            "question": q,
            "answer":   a,
            "sheet":    r["sheet"],
            # Combined text fed into the embedding model
            "document": f"Question: {q}\nAnswer: {a}"
        })

    print(f"[preprocessor] {len(cleaned)} records after cleaning "
          f"({len(records) - len(cleaned)} dropped)")
    return cleaned
