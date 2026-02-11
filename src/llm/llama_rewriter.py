from ollama import chat


# -----------------------------
# Remove Duplicate Chunks
# -----------------------------
def deduplicate_documents(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        content = getattr(doc, "page_content", None)
        if content and content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    return unique_docs


# -----------------------------
# LLaMA Layman Rewriter
# -----------------------------
def rewrite_with_llama(documents, query):
    """
    Sends retrieved regulation text to LLaMA
    and asks it to rewrite in simple bullet points.
    """

    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""
You are explaining Indian insurance regulations to a common person.

Rules:
- Use very simple English.
- Use short sentences.
- Do not use legal words.
- Convert into bullet points.
- Only use information from the context.
- If answer is not clearly available, say:
  "The document does not clearly say this."

Question:
{query}

Context:
{context}
"""

    response = chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
