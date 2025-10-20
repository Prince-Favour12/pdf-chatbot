from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st

def clean_text(text: str) -> str:
    """Remove invalid Unicode surrogates to prevent UTF-8 errors."""
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

def is_streamlit_running() -> bool:
    try:
        import streamlit.runtime.scriptrunner.script_runner as sr
        return sr.get_script_run_ctx() is not None
    except Exception:
        return False

def safe_log(message: str, level: str = "info"):
    if is_streamlit_running():
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.info(message)
    else:
        print(message)

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for embedding."""
    if not docs:
        safe_log("⚠️ No documents provided for chunking.", "warning")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    chunks = []
    for doc in docs:
        cleaned = clean_text(doc.page_content)
        doc_chunks = text_splitter.split_text(cleaned)
        chunks.extend(doc_chunks)

    doc_objects = [Document(page_content=clean_text(chunk)) for chunk in chunks]
    safe_log(f"✅ Created {len(doc_objects)} total chunks.")
    return doc_objects
