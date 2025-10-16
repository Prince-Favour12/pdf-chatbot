import streamlit as st
import os
from dotenv import load_dotenv
from src.rag.load_docs import load_single_document
from src.rag.chunk import chunk_documents
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.summarizer.summarize_docs import summarize_text

# 🔐 Load environment variables
load_dotenv()

# 🧠 Initialize embeddings and LLM
API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=API_KEY)
llm = ChatOpenAI(api_key=API_KEY, model="gpt-4o-mini", temperature=0.4)

# 🏠 Streamlit page config
st.set_page_config(page_title="PDF RAG & Summarizer", page_icon="📚", layout="centered")
st.title("📚 PDF Chat & Summarizer")

# Sidebar navigation
mode = st.sidebar.radio("Choose Mode", ["💬 Chat with PDFs", "🧾 Summarize PDFs"])

# ------------------------------------------------------
# 💬 Chat with PDFs
# ------------------------------------------------------
if mode == "💬 Chat with PDFs":
    st.header("💬 Chat with your PDFs")
    
    uploaded_files = st.file_uploader(
        "Upload one or multiple PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Create temp folder for session files
        temp_dir = os.path.join("uploaded_docs", st.session_state.get("session_id", "default"))
        os.makedirs(temp_dir, exist_ok=True)

        saved_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)

        st.success(f"✅ Uploaded {len(saved_paths)} PDF(s).")

        # Load and chunk documents
        all_docs = []
        for path in saved_paths:
            all_docs.extend(load_single_document(path))
        chunked_docs = chunk_documents(all_docs)

        # Store in ChromaDB (session-isolated)
        vectorstore = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory=os.path.join("chroma_db", st.session_state.get("session_id", "default"))
        )

        st.info("Documents embedded successfully! You can now ask questions.")
        query = st.text_input("Ask a question about your PDFs:")

        if query:
            from langchain.chains import RetrievalQA
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )
            try:
                result = qa_chain.invoke({"query": query})
                st.subheader("🤖 Answer:")
                st.write(result["result"])
            except Exception as e:
                st.error(f"⚠️ Error during QA: {e}")

# ------------------------------------------------------
# 🧾 Summarize PDFs
# ------------------------------------------------------
elif mode == "🧾 Summarize PDFs":
    st.header("🧾 Summarize PDFs")
    
    uploaded_files = st.file_uploader(
        "Upload one or multiple PDF documents for summarization",
        type=["pdf"],
        accept_multiple_files=True,
        key="summarize_uploader"
    )

    if uploaded_files:
        temp_dir = os.path.join("uploaded_docs", st.session_state.get("session_id", "default"))
        os.makedirs(temp_dir, exist_ok=True)

        all_docs = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            docs = load_single_document(file_path)
            all_docs.extend(docs)

        if all_docs:
            try:
                combined_text = " ".join(doc.page_content for doc in all_docs)
                summary = summarize_text(combined_text)
                st.subheader("📄 Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"⚠️ Error during summarization: {e}")
        else:
            st.warning("⚠️ No text found in the uploaded PDFs.")
