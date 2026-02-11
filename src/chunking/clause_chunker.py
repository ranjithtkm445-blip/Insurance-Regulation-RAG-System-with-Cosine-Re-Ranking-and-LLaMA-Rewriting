import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# PATH CONFIG (UNCHANGED)
# --------------------------------------------------
BASE_DIR = Path("D:/company_policy")
INPUT_JSON = BASE_DIR / "data" / "processed" / "cleaned_pages.json"
# --------------------------------------------------


def load_cleaned_pages() -> List[dict]:
    """
    Load cleaned PDF pages from JSON.
    """
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"File not found: {INPUT_JSON}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_chunking(pages: List[dict]) -> List[Document]:
    """
    Simple recursive chunking (NO clause / regulation awareness).
    """

    # Combine all pages into one continuous text
    full_text = "\n".join(page["content"] for page in pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    texts = splitter.split_text(full_text)

    documents = [
        Document(
            page_content=text.strip(),
            metadata={
                "source": "IRDAI_Policyholders_Regulations_2024"
            }
        )
        for text in texts
    ]

    return documents


def main():
    pages = load_cleaned_pages()
    documents = simple_chunking(pages)

    print("✅ Chunking completed (simple mode)")
    print(f"📦 Total chunks created: {len(documents)}")
    print("🔍 Sample metadata:")
    print(documents[0].metadata)


if __name__ == "__main__":
    main()
