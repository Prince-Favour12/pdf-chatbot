import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from .retrieval import retrieve  # fixed import
from dotenv import load_dotenv

load_dotenv()

def create_llm():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.3
    )

def create_qa_chain():
    llm = create_llm()
    retriever = retrieve()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

if __name__ == "__main__":
    qa_chain = create_qa_chain()

    query = "Why is Liora different from Google Classroom?"
    result = qa_chain.invoke({"query": query})

    print("\n🤖 Answer:\n", result["result"])

    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\n📘 Source {i}:\n{doc.page_content[:300]}...")
