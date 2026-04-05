"""
Streamlit UI: optional user document Q&A, or bank knowledge base when no document is loaded.
Run: streamlit run app.py
"""
import streamlit as st

from config import (
    TOP_K,
    MODE,
    OPENROUTER_MODEL,
    EMBED_MODEL,
    MAX_QUERY_CHARS,
    MIN_RETRIEVAL_SCORE,
    SESSION_DOC_MIN_SCORE,
)
from embedder import get_embed_model
from indexing import setup
from llm_handler import load_llm
from orchestrator import handle_customer_query
from session_document import build_session_rag_records, read_document_bytes


@st.cache_resource(show_spinner="Loading embedding model…")
def cached_embed_model():
    return get_embed_model()


@st.cache_resource(show_spinner="Connecting to OpenRouter…")
def cached_llm():
    return load_llm()


def init_session_collection():
    if "collection" not in st.session_state:
        st.session_state.collection = setup(cached_embed_model())


def _session_rag():
    return st.session_state.get("session_rag") or None


def _doc_label():
    return st.session_state.get("session_doc_name") or ""


def render_sources(retrieved: list[dict]) -> None:
    if not retrieved:
        return
    with st.expander("Sources used", expanded=False):
        for i, r in enumerate(retrieved, 1):
            meta = r.get("metadata") or {}
            src = meta.get("source") or meta.get("sheet", "?")
            if "chunk_index" in meta:
                st.markdown(
                    f"**[{i}]** `{src}` · section **{meta['chunk_index'] + 1}** · "
                    f"similarity **{r.get('score', 0):.2f}**  \n"
                    f"*{r.get('document', '')[:280]}…*"
                )
            else:
                q = meta.get("question", "")[:200]
                st.markdown(
                    f"**[{i}]** `{src}` · similarity **{r.get('score', 0):.2f}**  \n"
                    f"*Q:* {q}…"
                )


def main_ui():
    st.set_page_config(
        page_title="Bank Assistant",
        page_icon="🏦",
        layout="centered",
    )

    st.title("Bank assistant")
    st.caption(
        f"Mode **{MODE}** · LLM **OpenRouter** `{OPENROUTER_MODEL}` · "
        f"Embeddings (local) `{EMBED_MODEL.split('/')[-1]}`"
    )

    init_session_collection()
    llm_client = cached_llm()

    with st.expander("Add your own document (optional)", expanded=not _session_rag()):
        st.markdown(
            "Upload a **PDF**, **Word** (.docx), **text**, or **Markdown** file. "
            "While a document is loaded, answers use **only that file**. "
            "Clear it anytime to go back to the **bank knowledge base**."
        )
        up = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt", "md", "csv"],
            label_visibility="collapsed",
        )
        c1, c2 = st.columns(2)
        with c1:
            load_clicked = st.button("Load document for questions", type="primary")
        with c2:
            if st.button("Clear document", disabled=not _session_rag()):
                st.session_state.pop("session_rag", None)
                st.session_state.pop("session_doc_name", None)
                st.rerun()

        if load_clicked:
            if up is None:
                st.warning("Choose a file first.")
            else:
                try:
                    raw = up.getvalue()
                    text = read_document_bytes(raw, up.name)
                    if not text.strip():
                        st.error("No text could be read from this file.")
                    else:
                        with st.spinner("Indexing your document…"):
                            recs = build_session_rag_records(
                                text,
                                source_name=up.name,
                                embed_model=cached_embed_model(),
                            )
                        if not recs:
                            st.error("Document is empty after processing.")
                        else:
                            st.session_state.session_rag = recs
                            st.session_state.session_doc_name = up.name
                            st.success(
                                f"Loaded **{up.name}** ({len(recs)} sections). "
                                "Ask questions below."
                            )
                            st.rerun()
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Could not read file: {e!s}")

    if _session_rag():
        st.info(f"**Active document:** `{_doc_label()}` — answers use this file only.")
    else:
        st.info("No personal document loaded — using the **bank FAQ knowledge base**.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    hint = (
        f"Ask about your document ({_doc_label()})…"
        if _session_rag()
        else "Ask about bank products and services…"
    )
    prompt = st.chat_input(hint)
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = handle_customer_query(
                prompt,
                st.session_state.collection,
                cached_embed_model(),
                llm_client,
                top_k=TOP_K,
                max_query_length=MAX_QUERY_CHARS,
                min_retrieval_score=MIN_RETRIEVAL_SCORE,
                session_rag=_session_rag(),
                session_min_score=SESSION_DOC_MIN_SCORE,
            )
        if result["blocked"]:
            text = result["user_message"]
            st.warning(text)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "sources": result["retrieved"]}
            )
        else:
            st.markdown(result["answer"])
            render_sources(result["retrieved"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["retrieved"],
                }
            )


if __name__ == "__main__":
    main_ui()
