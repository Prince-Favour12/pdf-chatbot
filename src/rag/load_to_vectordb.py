import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from src.rag.load_docs import load_batch_documents
from src.rag.chunk import chunk_documents

load_dotenv()  # ensure OPENAI_API_KEY is loaded

def clean_text(text: str) -> str:
    """Remove or replace problematic Unicode characters (like unpaired surrogates)."""
    if not isinstance(text, str):
        return ""
    # Encode to UTF-8 and ignore invalid chars
    cleaned = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    # Optionally strip invisible characters or multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_to_db():
    folder_path = "C:/Users/DELL/Documents/Liora"
    persist_directory = "./chroma_db"

    print("📥 Loading documents...")
    documents = load_batch_documents(folder_path)
    if not documents:
        print("⚠️ No documents found.")
        return None

    print("✂️ Chunking documents...")
    chunk_docs = chunk_documents(documents)

    print("🧹 Cleaning chunk text...")
    for doc in chunk_docs:
        if hasattr(doc, "page_content"):
            doc.page_content = clean_text(doc.page_content)

    cleaned_chunks = [
        doc for doc in chunk_docs
        if isinstance(doc.page_content, str) and doc.page_content.strip()
    ]

    print(f"✅ {len(cleaned_chunks)} valid chunks ready for embedding.")

    print("🧠 Creating OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("💾 Storing embeddings in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=cleaned_chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()

    print(f"✅ Stored {len(cleaned_chunks)} chunks successfully in {persist_directory}")
    return vectorstore


if __name__ == "__main__":
    vs = load_to_db()
    if vs:
        print("🎉 Vector DB ready for retrieval!")
