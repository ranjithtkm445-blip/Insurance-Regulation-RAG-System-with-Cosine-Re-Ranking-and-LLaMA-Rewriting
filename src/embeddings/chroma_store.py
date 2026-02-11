import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# PATH CONFIG
# --------------------------------------------------
BASE_DIR = Path("D:/company_policy")
INPUT_JSON = BASE_DIR / "data" / "processed" / "cleaned_pages.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
# --------------------------------------------------


def load_cleaned_pages():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"{INPUT_JSON} not found")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def create_chunks(pages) -> List[Document]:
    """
    SAME simple chunking logic used earlier.
    Do NOT change this.
    """
    full_text = "\n".join(page["content"] for page in pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    texts = splitter.split_text(full_text)

    return [
        Document(
            page_content=text.strip(),
            metadata={"source": "IRDAI_Policyholders_Regulations_2024"}
        )
        for text in texts
    ]


def store_in_chroma(documents: List[Document]):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    print("✅ ChromaDB populated successfully")
    print(f"📁 Stored at: {CHROMA_DIR}")


def main():
    pages = load_cleaned_pages()
    documents = create_chunks(pages)
    store_in_chroma(documents)


if __name__ == "__main__":
    main()
