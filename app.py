import streamlit as st
import os
import sys
import io
from dotenv import load_dotenv
from src.rag.load_docs import load_single_document, load_batch_documents
from src.rag.chunk import chunk_documents
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.summarizer.summarize_docs import summarize_text

# 🔐 Load .env
load_dotenv()

# 🧠 Initialize Embeddings + LLM
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.4)

# 🏠 App Title
st.set_page_config(page_title="Mini RAG & Summarizer", page_icon="📚", layout="centered")
st.title("📚 Mini RAG & Document Summarizer")

# Sidebar Navigation
mode = st.sidebar.radio("Choose Mode", ["💬 Chat with Documents", "🧾 Summarize Documents"])

# ------------------------------------------------------
# 🧠 CHAT WITH PDF MODE
# ------------------------------------------------------
if mode == "💬 Chat with Documents":
    st.header("💬 Chat with your documents")

    uploaded_files = st.file_uploader(
        "Upload one or multiple documents (PDF, DOCX, TXT, PPTX)",
        type=["pdf", "docx", "txt", "pptx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        temp_dir = "uploaded_docs"
        os.makedirs(temp_dir, exist_ok=True)

        saved_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)

        st.success(f"✅ Uploaded {len(saved_paths)} documents.")

        # Load and chunk
        all_docs = []
        for path in saved_paths:
            all_docs.extend(load_single_document(path))
        chunked_docs = chunk_documents(all_docs)

        # Store in ChromaDB (ephemeral)
        vectorstore = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

        st.info("Documents embedded successfully! You can now ask questions.")
        query = st.text_input("Ask a question about your documents:")

        if query:
            from langchain.chains import RetrievalQA

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )

            result = qa_chain.invoke({"query": query})
            st.write("### 🤖 Answer:")
            st.write(result["result"])

# ------------------------------------------------------
# 🧾 SUMMARIZE DOCUMENTS MODE
# ------------------------------------------------------
elif mode == "🧾 Summarize Documents":
    st.header("🧾 Summarize Single or Multiple Documents")

    summarize_type = st.radio("Choose Summarization Type:", ["Single Document", "Batch (Folder)"])

    if summarize_type == "Single Document":
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "pptx"])
        if uploaded_file:
            temp_path = os.path.join("uploaded_docs", uploaded_file.name)
            os.makedirs("uploaded_docs", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.info(f"Summarizing {uploaded_file.name}...")
            docs = load_single_document(temp_path)
            full_text = " ".join(doc.page_content for doc in docs)
            summary = summarize_text(full_text)
            st.subheader("📄 Summary:")
            st.write(summary)

    else:  # Batch
        folder_path = st.text_input("Enter the path to your folder containing documents:")
        if folder_path and os.path.isdir(folder_path):
            st.info(f"Summarizing all documents in {folder_path}...")
            docs = load_batch_documents(folder_path)
            if docs:
                combined_text = " ".join(doc.page_content for doc in docs)
                summary = summarize_text(combined_text)
                st.subheader("📁 Batch Summary:")
                st.write(summary)
            else:
                st.warning("⚠️ No documents found in the specified folder.")
        elif folder_path:
            st.error("❌ Invalid folder path.")
