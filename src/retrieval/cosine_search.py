import numpy as np
from typing import List

from src.embeddings.embedder import load_embedding_model


# ----------------------------------
# Load embedding model ONCE
# ----------------------------------
_embedding_model = load_embedding_model()


# ----------------------------------
# Cosine Similarity Function
# ----------------------------------
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# ----------------------------------
# Re-rank Documents Using Cosine
# ----------------------------------
def rerank_by_cosine(query: str, documents: List, top_k: int = 3):
    """
    Re-ranks retrieved documents using cosine similarity
    between query embedding and document embeddings.
    """

    if not documents:
        return []

    # Embed query once
    query_embedding = _embedding_model.embed_query(query)

    scored_docs = []

    for doc in documents:
        try:
            doc_text = doc.page_content
            doc_embedding = _embedding_model.embed_query(doc_text)

            score = cosine_similarity(query_embedding, doc_embedding)
            scored_docs.append((score, doc))

        except Exception:
            continue  # Skip problematic docs safely

    # Sort by highest similarity score
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Return top_k documents
    return [doc for _, doc in scored_docs[:top_k]]
