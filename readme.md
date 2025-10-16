# 📘 PDF Chatbot with JSON Support (RAG System)

An interactive **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit**, **LangChain**, and **OpenAI embeddings**.  
This application allows users to **upload PDFs or JSON files**, automatically chunk their content, embed it into a **Chroma vector store**, and interact with the data using natural language.  

It also includes a **document summarization feature**, powered by OpenAI or Hugging Face models.

---

## 🚀 Features

- ✅ Multi-format document ingestion — supports both **PDF** and **JSON** files  
- ✅ Automatic text chunking for better embedding quality  
- ✅ Vector-based semantic search using **ChromaDB** and **OpenAI embeddings**  
- ✅ Conversational question-answering over uploaded documents  
- ✅ Summarization mode with emoji-safe text cleaning  
- ✅ Persistent vector storage using `chroma_db`  
- ✅ Clean, modern **Streamlit UI** for easy interaction

---

## 🧰 Tech Stack

- **Python 3.10+**  
- **Streamlit** — interactive web app framework  
- **LangChain** — document retrieval & RAG pipeline  
- **ChromaDB** — vector database for embeddings  
- **OpenAI** — embeddings and LLM completions  
- **PyMuPDF (fitz)** — PDF parsing and text extraction  
- **JSON loader** — for custom question-answering data

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
```

### 2. Create virtual environment
```bash
python -m venv .venv
# Activate on Windows
.venv\Scripts\activate
# Activate on macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add environmental variable
add a .env file with this config inside:
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_ID=gpt-4o-mini
```

### 5. Run app
```bash
streamlit run app.py
```

## How to Use

1. Upload a PDF or JSON file through the Streamlit interface.
2. The application automatically:
   - Extracts and cleans the text.
   - Splits the content into smaller chunks.
   - Embeds each chunk using OpenAI embeddings.
   - Stores the vector data in a Chroma vector database.
3. After processing, you can:
   - Ask natural language questions about the uploaded documents.
   - Summarize an entire document or specific sections.
4. The summarizer safely handles all characters and removes invalid Unicode automatically.


## Summarization

In summarization mode, the system:
- Cleans the input text to remove unsupported characters.
- Sends the cleaned text to the OpenAI or Hugging Face model.
- Returns a concise summary in readable text.
- Handles large text inputs safely by chunking them before summarization.


## Common Issues and Fixes

| Issue | Cause | Solution |
|-------|--------|-----------|
| UnicodeEncodeError: surrogates not allowed | Invalid or unsupported Unicode characters in text | The system now cleans the text before summarization or embedding. |
| openai.APIConnectionError | Network timeout or unstable internet connection | Check your internet connection and retry. |
| ValueError: I/O operation on closed file | File stream closed prematurely by Streamlit or Python | The file handling logic has been fixed in the latest version. |
| 422 Unprocessable Entity | Oversized or malformed text input | Reduce chunk size or validate the file format. |
| AttributeError: 'str' object has no attribute 'page_content' | Non-LangChain Document object passed to vectorstore | Ensure chunking returns Document objects, not plain strings. |


## Developer Notes

- Modify chunk size and overlap values in `src/rag/chunk.py` to control how documents are divided.
- To switch to a different embedding model, update the embedding setup in the app.
- Delete the `chroma_db` folder to reset and rebuild the vector database.
- The summarizer supports both OpenAI and Hugging Face models.
- The project is modular, so you can extend it with other loaders or LLMs as needed.


## Deployment

This application can be deployed on multiple platforms such as:
- Streamlit Cloud
- Hostinger Cloud
- Render
- Vercel
- Docker or any containerized environment

Example Docker deployment:

```bash
docker build -t pdf-chatbot .
docker run -p 8501:8501 pdf-chatbot
