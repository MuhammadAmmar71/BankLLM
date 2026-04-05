"""
Persist FAQ rows added outside the core Excel workbook (manual entry, uploads).
Stored as JSONL of cleaned question / answer / sheet so reindex can merge without re-preprocessing.
"""
import json
from pathlib import Path

from config import SUPPLEMENTAL_FAQS_PATH


def _path() -> Path:
    p = Path(SUPPLEMENTAL_FAQS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_supplemental_embed_records() -> list[dict]:
    """Rows ready to merge with preprocessed core records (includes 'document')."""
    path = _path()
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()
            sheet = row.get("sheet", "Supplemental").strip() or "Supplemental"
            if not q or not a:
                continue
            out.append(
                {
                    "question": q,
                    "answer": a,
                    "sheet": sheet,
                    "document": f"Question: {q}\nAnswer: {a}",
                }
            )
    return out


def count_supplemental() -> int:
    return len(load_supplemental_embed_records())


def append_cleaned_records(records: list[dict]) -> int:
    """
    Append records that already went through preprocess (question, answer, sheet).
    Returns number of lines written.
    """
    path = _path()
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            q = r.get("question", "").strip()
            a = r.get("answer", "").strip()
            sheet = (r.get("sheet") or "Supplemental").strip() or "Supplemental"
            if not q or not a:
                continue
            f.write(
                json.dumps({"question": q, "answer": a, "sheet": sheet}, ensure_ascii=False)
                + "\n"
            )
            n += 1
    return n
