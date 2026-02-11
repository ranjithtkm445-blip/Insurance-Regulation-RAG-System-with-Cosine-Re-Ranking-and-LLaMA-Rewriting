import json
import re
from pathlib import Path

# ✅ Correct LangChain imports
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


# --------------------------------------------------
# CONFIG (YOUR EXACT PATH)
# --------------------------------------------------
BASE_DIR = Path("D:/company_policy")

RAW_DIR = BASE_DIR / "data" / "policies"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PDF_FILE_NAME = "IRDAI_Policyholders_Regulations_2024.pdf"
OUTPUT_JSON = PROCESSED_DIR / "cleaned_pages.json"
# --------------------------------------------------


def clean_text(text: str) -> str:
    """
    Cleans PDF text without changing legal meaning.
    """
    # Remove page numbers like "Page 12 of 31"
    text = re.sub(
        r'Page\s+\d+\s+of\s+\d+',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def load_pdf() -> list[Document]:
    """
    Load PDF using LangChain PyPDFLoader.
    """
    pdf_path = RAW_DIR / PDF_FILE_NAME

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()   # returns List[Document]

    return documents


def process_and_save(documents: list[Document]) -> None:
    """
    Clean page-wise documents and save to JSON.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_pages = []

    for doc in documents:
        cleaned_pages.append({
            "page": doc.metadata.get("page"),
            "content": clean_text(doc.page_content),
            "source": PDF_FILE_NAME
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned_pages, f, indent=2, ensure_ascii=False)

    print("✅ PDF loaded successfully using LangChain")
    print(f"📄 Pages extracted: {len(cleaned_pages)}")
    print(f"💾 Output saved to: {OUTPUT_JSON}")


def main():
    documents = load_pdf()
    process_and_save(documents)


if __name__ == "__main__":
    main()
