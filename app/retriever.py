from app.embeddings import get_embedding
from app.vector_store import search_vectors


def retrieve_top_10(question):

    question_embedding = get_embedding(
        question
    )

    results = search_vectors(
        question_embedding,
        limit=10
    )

    retrieved_chunks = []

    for result in results:

        retrieved_chunks.append({
            "text": result.payload["text"],
            "source": result.payload["source"],
            "page": result.payload["page"],
            "chunk_id": result.payload["chunk_id"],
            "score": result.score
        })

    return retrieved_chunks