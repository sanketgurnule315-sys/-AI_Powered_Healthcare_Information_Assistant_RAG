import requests

from app.config import (
    NUGEN_API_KEY,
    NUGEN_LLM_MODEL
)


NUGEN_CHAT_URL = (
    "https://api.nugen.in/api/v3/inference/chat/completions"
)


def enhance_query(question):

    prompt = f"""
You are a search query enhancer for a document-based RAG system.

The document is:
HEALTHCARE INFORMATION

Rewrite the user's question into a concise search query
that will help retrieve the most relevant information
from the document.

Rules:

1. Keep the original meaning.
2. Keep important keywords.
3. Do not answer the question.
4. Do not add facts that are not present in the question.
5. Do not remove important names or locations.
6. If the question is already clear, keep it almost unchanged.
7. Return ONLY the improved search query.
8. Do not add explanations.
9. Do not write "Search query:" before the result.

User question:
{question}

Improved search query:
"""

    headers = {
        "Authorization": f"Bearer {NUGEN_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": NUGEN_LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 100,
        "temperature": 0,
        "stream": False
    }

    response = requests.post(
        NUGEN_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    if "choices" in data:

        choice = data["choices"][0]

        if "message" in choice:

            content = choice["message"]["content"]

            return content.strip()

        if "text" in choice:

            return choice["text"].strip()

    raise ValueError(
        f"Unexpected Nugen response: {data}"
    )