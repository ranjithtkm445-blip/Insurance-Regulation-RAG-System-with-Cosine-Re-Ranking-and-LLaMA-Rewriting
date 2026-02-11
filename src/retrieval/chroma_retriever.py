from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# --------------------------------------------------
# PATH CONFIG
# --------------------------------------------------
BASE_DIR = Path("D:/company_policy")
CHROMA_DIR = BASE_DIR / "chroma_db"
# --------------------------------------------------


def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )


def retrieve(query: str, k: int = 5) -> List[Document]:
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


if __name__ == "__main__":
    docs = retrieve(
        "What are the duties of insurers towards policyholders?",
        k=3
    )

    print(f"Retrieved {len(docs)} chunks\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Chunk {i} ---")
        print(doc.page_content[:400])
        print()
