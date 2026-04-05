"""
Shared RAG flow: guardrails → retrieve → guardrails → generate.
Used by the CLI (main.py) and the Streamlit UI (app.py).
"""
from embedder import retrieve
from guardrails import GuardrailResult, validate_input, validate_retrieval
from llm_handler import generate_answer
from session_document import retrieve_from_session_rag


def handle_customer_query(
    query: str,
    collection,
    embed_model,
    llm_client,
    top_k: int,
    max_query_length: int,
    min_retrieval_score: float,
    session_rag: list[dict] | None = None,
    session_min_score: float = 0.12,
) -> dict:
    """
    llm_client: OpenRouter client from load_llm().
    If session_rag is set (uploaded document chunks), retrieval uses only that list;
    otherwise the Chroma collection (bank KB) is used.

    Returns a dict:
      - blocked: bool — if True, do not show model output; use user_message
      - user_message: str — either error/fallback text or empty when not blocked
      - answer: str — model answer when not blocked
      - retrieved: list[dict] — retrieval results (may be empty)
    """
    inp = validate_input(query, max_length=max_query_length)
    if not inp.ok:
        return {
            "blocked": True,
            "user_message": inp.user_message,
            "answer": "",
            "retrieved": [],
        }

    if session_rag:
        retrieved = retrieve_from_session_rag(
            query.strip(), embed_model, session_rag, top_k=top_k
        )
        ret_min = session_min_score
    else:
        retrieved = retrieve(query.strip(), collection, embed_model, top_k=top_k)
        ret_min = min_retrieval_score

    ret_check: GuardrailResult = validate_retrieval(retrieved, min_best_score=ret_min)
    if not ret_check.ok:
        return {
            "blocked": True,
            "user_message": ret_check.user_message,
            "answer": "",
            "retrieved": retrieved,
        }

    answer = generate_answer(query.strip(), retrieved, llm_client)
    return {
        "blocked": False,
        "user_message": "",
        "answer": answer,
        "retrieved": retrieved,
    }
