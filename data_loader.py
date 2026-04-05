import re
from pathlib import Path

import pandas as pd
from config import DATASET_PATH

# Skipping Sheets that don't contain Q&A data 
SKIP_SHEETS = {"Main", "Rate Sheet July 1 2024", "Sheet1"}


def is_question(text: str) -> bool:
    """
    Detect whether a cell is a question.
    Handles formats like:
      - "What is the eligibility criteria?"
      - "1. What are the benefits?"
      - "2. Who can apply?"
      - "I would like to open an account..."  (conversational)
    Key rule: questions never start with bullet points (o , -, •, ·)
    and are not continuation/detail lines.
    """
    if not text:
        return False

    # Skip lines that are clearly answer detail/bullet points
    if re.match(r"^[o•·\-\*]\s", text):
        return False

    # Skip lines that look like table headers or values (short numbers/%)
    if re.match(r"^[\d\.]+\s*%?$", text):
        return False

    # Numbered question: "1. question text" or "1) question text"
    if re.match(r"^\d+[\.\)]\s+\S", text):
        return True

    # Ends with a question mark
    if text.strip().endswith("?"):
        return True

    # Conversational openers (common in this dataset)
    conversational = (
        "i would", "i want", "i need", "do you", "does your",
        "can i", "can you", "is there", "are there", "how do",
        "how can", "what if", "tell me", "please"
    )
    lower = text.lower()
    if any(lower.startswith(p) for p in conversational):
        return True

    return False


def get_main_col(df: pd.DataFrame) -> int:
    """
    Find the column with the most meaningful text content.
    Ignores columns that only contain short numeric/rate values.
    """
    best_col = 0
    best_score = 0
    for col in df.columns:
        values = df[col].dropna().astype(str)
        score = sum(1 for v in values if len(v.strip()) > 10)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def parse_qa_from_sheet(df: pd.DataFrame, sheet_name: str) -> list[dict]:
    """
    Parses Q&A pairs from a sheet.
    - Finds the main text column automatically
    - Treats alternating non-empty rows as Q then A
    - Appends extra detail columns to the answer where present
    - Multi-row answers (bullet points) are concatenated
    """
    records = []
    main_col = get_main_col(df)

    # Extra columns that carry supplementary data (e.g. profit rates)
    extra_cols = [c for c in df.columns if c != main_col and df[c].notna().any()]

    current_q = None
    current_a = None

    for _, row in df.iterrows():
        cell = str(row[main_col]).strip() if pd.notna(row[main_col]) else ""

        # Skip empty cells and navigation links like "Main"
        if not cell or cell.lower() in ("nan", "none", "", "main"):
            continue

        # Skip sheet title row (first non-empty row if it's all caps and short)
        if len(cell) < 80 and cell.replace(" ", "").replace("(", "").replace(")", "").isupper() and current_q is None:
            continue

        if is_question(cell):
            # Save previous Q&A pair before starting a new one
            if current_q and current_a:
                records.append({
                    "question": current_q.strip(),
                    "answer":   current_a.strip(),
                    "sheet":    sheet_name
                })

            # Strip leading number prefix "1. " or "2) "
            current_q = re.sub(r"^\d+[\.\)]\s+", "", cell).strip()
            current_a = None

        else:
            # This row is answer content
            if current_q is not None:
                # Append extra detail columns for this row
                extras = []
                for ec in extra_cols:
                    val = str(row[ec]).strip() if pd.notna(row[ec]) else ""
                    if val and val.lower() not in ("nan", "none", "", "main"):
                        extras.append(val)

                line = cell
                if extras:
                    line += " (" + ", ".join(extras) + ")"

                if current_a is None:
                    current_a = line
                else:
                    current_a += " " + line

    # Don't forget the last pair
    if current_q and current_a:
        records.append({
            "question": current_q.strip(),
            "answer":   current_a.strip(),
            "sheet":    sheet_name
        })

    return records


def load_faq_csv(path: str, sheet_label: str = "CSV FAQ") -> list[dict]:
    """
    Load Q&A rows from a CSV. Recognizes columns named question/q/faq/prompt and
    answer/a/response/reply (case-insensitive). If missing, uses the first two columns.
    """
    df = pd.read_csv(path)
    col_map = {str(c).lower().strip(): c for c in df.columns}
    q_col = None
    a_col = None
    for key in ("question", "q", "faq", "prompt"):
        if key in col_map:
            q_col = col_map[key]
            break
    for key in ("answer", "a", "response", "reply"):
        if key in col_map:
            a_col = col_map[key]
            break
    if q_col is None or a_col is None:
        if len(df.columns) >= 2:
            q_col, a_col = df.columns[0], df.columns[1]
        else:
            return []

    records: list[dict] = []
    for _, row in df.iterrows():
        q = str(row[q_col]).strip() if pd.notna(row[q_col]) else ""
        a = str(row[a_col]).strip() if pd.notna(row[a_col]) else ""
        if not q or not a or q.lower() == "nan" or a.lower() == "nan":
            continue
        records.append({"question": q, "answer": a, "sheet": sheet_label})
    return records


def load_faq_file(path: str, upload_label: str = "Uploaded FAQ") -> list[dict]:
    """Load FAQs from .csv or Excel (.xlsx / .xls) using the same rules as the core workbook."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_faq_csv(path, sheet_label=upload_label)
    if suffix in (".xlsx", ".xls"):
        return load_dataset(path)
    return []


def load_dataset(path: str = DATASET_PATH) -> list[dict]:
    """
    Reads all sheets from the Excel workbook and returns
    a flat list of Q&A records. Skips non-Q&A sheets.
    """
    print(f"\n[data_loader] Loading dataset: {path}")
    xl = pd.ExcelFile(path)
    all_records = []

    for sheet in xl.sheet_names:
        if sheet in SKIP_SHEETS:
            print(f"             Sheet '{sheet}': skipped (non-QA sheet)")
            continue

        df = xl.parse(sheet, header=None)

        # Skip completely empty sheets
        if df.empty or df.shape == (0, 0):
            print(f"             Sheet '{sheet}': skipped (empty)")
            continue

        records = parse_qa_from_sheet(df, sheet_name=sheet)
        print(f"             Sheet '{sheet}': {len(records)} Q&A pairs")
        all_records.extend(records)

    print(f"\n[data_loader] Total: {len(all_records)} Q&A pairs loaded")
    return all_records