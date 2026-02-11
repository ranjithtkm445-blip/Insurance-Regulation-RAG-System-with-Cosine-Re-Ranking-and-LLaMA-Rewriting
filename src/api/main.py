from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re


# -----------------------------
# Request / Response Models
# -----------------------------

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: List[str]


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(
    title="Insurance Policy RAG API",
    description="Ask insurance regulation questions and get simple answers.",
    version="1.0"
)


# -----------------------------
# Enable CORS (Frontend Access)
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Health Endpoint
# -----------------------------

@app.get("/")
def health():
    return {"status": "API running successfully"}


# -----------------------------
# Main RAG Endpoint
# -----------------------------

@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    try:
        # Import inside function to avoid circular imports
        from src.retrieval.chroma_retriever import retrieve
        from src.retrieval.cosine_search import rerank_by_cosine
        from src.llm.llama_rewriter import (
            deduplicate_documents,
            rewrite_with_llama
        )

        query_text = payload.question.strip()

        # Improve weak single-word queries
        if len(query_text.split()) == 1:
            query_text = f"What does {query_text} mean under this regulation?"

        # -----------------------------
        # Stage 1: Vector Retrieval
        # -----------------------------
        docs = retrieve(query_text, k=8)

        # -----------------------------
        # Stage 2: Remove Duplicates
        # -----------------------------
        docs = deduplicate_documents(docs)

        # -----------------------------
        # Stage 3: Cosine Re-Ranking
        # -----------------------------
        docs = rerank_by_cosine(query_text, docs, top_k=3)

        # -----------------------------
        # Stage 4: LLaMA Rewriting
        # -----------------------------
        final_answer = rewrite_with_llama(docs, payload.question)

        # -----------------------------
        # Clean and Format Bullets
        # -----------------------------
        bullets = []

        for line in final_answer.split("\n"):
            clean_line = line.strip()

            # Remove leading bullet symbols and extra characters
            clean_line = re.sub(r"^[\-\*\•\s]+", "", clean_line)

            if clean_line:
                bullets.append(clean_line)

        if not bullets:
            bullets = ["The document does not clearly say this."]

        return {
            "question": payload.question,
            "answer": bullets
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
