import os

from openai import OpenAI

from config import (
    MAX_NEW_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_TITLE,
    OPENROUTER_BASE_URL,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_MODEL,
    SYSTEM_PROMPT,
)


def _resolve_api_key() -> str:
    return (OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "") or "").strip()


def build_chat_messages(query: str, retrieved: list[dict]) -> list[dict[str, str]]:
    """OpenRouter / chat-completions style messages with RAG context in the user turn."""
    context = "\n\n".join(
        [f"[Source: {r['metadata']['sheet']}]\n{r['document']}" for r in retrieved]
    )
    user_content = (
        f"Context from bank knowledge base:\n{context}\n\nCustomer question: {query}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def load_llm() -> OpenAI:
    """
    Returns an OpenAI-compatible client pointed at OpenRouter (no local LLM weights).
    """
    key = _resolve_api_key()
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to your environment or `.env` file."
        )

    headers: dict[str, str] = {}
    ref = (OPENROUTER_HTTP_REFERER or os.environ.get("OPENROUTER_HTTP_REFERER", "")).strip()
    if ref:
        headers["HTTP-Referer"] = ref
    title = (
        OPENROUTER_APP_TITLE
        or os.environ.get("OPENROUTER_APP_TITLE", "")
        or "Bank LLM Assistant"
    ).strip()
    if title:
        headers["X-Title"] = title

    print(f"\n[llm_handler] OpenRouter · model `{OPENROUTER_MODEL}`")
    print(f"              Base URL: {OPENROUTER_BASE_URL}\n")

    client_kw: dict = {
        "api_key": key,
        "base_url": OPENROUTER_BASE_URL.rstrip("/"),
    }
    if headers:
        client_kw["default_headers"] = headers
    return OpenAI(**client_kw)


def generate_answer(query: str, retrieved: list[dict], llm_client: OpenAI) -> str:
    """
    Calls OpenRouter chat completions and returns the assistant message text.
    """
    messages = build_chat_messages(query, retrieved)
    try:
        completion = llm_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0,
        )
    except Exception as e:
        print(f"[llm_handler] OpenRouter error: {e!r}")
        return (
            "I'm sorry, the language service is unavailable right now. "
            "Please try again in a moment or contact the bank directly."
        )

    choice = completion.choices[0].message
    text = (choice.content or "").strip()
    if not text:
        return (
            "I could not generate a reply. Please try rephrasing your question "
            "or contact the bank directly."
        )
    return text
