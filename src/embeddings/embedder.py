import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# PATH CONFIG
# --------------------------------------------------
BASE_DIR = Path("D:/company_policy")
INPUT_JSON = BASE_DIR / "data" / "processed" / "cleaned_pages.json"
# --------------------------------------------------


def load_cleaned_pages():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def create_chunks(pages) -> List[Document]:
    full_text = "\n".join(page["content"] for page in pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    texts = splitter.split_text(full_text)

    return [
        Document(
            page_content=text,
            metadata={"source": "IRDAI_Policyholders_Regulations_2024"}
        )
        for text in texts
    ]


def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def main():
    pages = load_cleaned_pages()
    documents = create_chunks(pages)
    embedding_model = load_embedding_model()

    # Test embedding generation
    vector = embedding_model.embed_query(documents[0].page_content)
    print("✅ Embedding model working")
    print(f"📐 Embedding vector length: {len(vector)}")


if __name__ == "__main__":
    main()
