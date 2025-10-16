import os
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader
)
import streamlit as st


def is_streamlit_running() -> bool:
    """Detect if running under Streamlit."""
    try:
        import streamlit.runtime.scriptrunner.script_runner as sr
        return sr.get_script_run_ctx() is not None
    except Exception:
        return False


def safe_log(message: str, level: str = "info"):
    """Safely print or display messages based on environment."""
    if is_streamlit_running():
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.info(message)
    else:
        print(message)


def load_single_document(file_path):
    """Load a single document based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    loader = None

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            safe_log(f"⚠️ Unsupported file type: {ext}", "warning")
            return []

        docs = loader.load()
        return docs

    except Exception as e:
        safe_log(f"⚠️ Error loading {file_path}: {e}", "error")
        return []


def load_batch_documents(folder_path):
    """Load all supported documents from a folder."""
    all_docs = []

    if not os.path.exists(folder_path):
        safe_log(f"❌ Folder not found: {folder_path}", "error")
        return []

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if os.path.isfile(path):
            all_docs.extend(load_single_document(path))

    return all_docs
