import os
import sys
import io
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.rag.load_docs import load_single_document, load_batch_documents


# 🔐 Load API key from .env
load_dotenv()

from openai import OpenAI
import re

client = OpenAI()

def clean_text(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r'[\ud800-\udfff]', '', text)
    return text

def summarize_text(text):
    text = clean_text(text)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the document briefly."},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during summarization: {e}"



def summarize_single(file_path: str):
    """Summarize a single document (PDF, DOCX, TXT, etc.)"""
    docs = load_single_document(file_path)
    if not docs:
        print(f"⚠️ No content found in {file_path}")
        return

    full_text = " ".join(doc.page_content for doc in docs)
    summary = summarize_text(full_text)

    print(f"\n📄 Summary for {os.path.basename(file_path)}:\n")
    print(summary)


def summarize_batch(folder_path: str):
    """Summarize all documents inside a folder."""
    docs = load_batch_documents(folder_path)
    if not docs:
        print(f"⚠️ No documents found in {folder_path}")
        return

    combined_text = " ".join(doc.page_content for doc in docs)
    summary = summarize_text(combined_text)

    print(f"\n📁 Summary for all documents in folder '{folder_path}':\n")
    print(summary)


if __name__ == "__main__":
    # This only runs when you execute the file directly (not via Streamlit)
    mode = input("Choose mode (single/batch): ").strip().lower()

    if mode == "single":
        file_path = input("Enter path to the file: ").strip()
        summarize_single(file_path)

    elif mode == "batch":
        folder_path = input("Enter folder path: ").strip()
        summarize_batch(folder_path)

    else:
        print("❌ Invalid mode. Choose 'single' or 'batch'.")
