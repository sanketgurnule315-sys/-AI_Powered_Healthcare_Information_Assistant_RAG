from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def rerank_documents(
    question,
    chunks,
    top_n=3
):

    if not chunks:
        return []

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

  
    query_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    document_embeddings = model.encode(
        documents,
        convert_to_tensor=True
    )

 
    relevance_scores = cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    candidates = []

    for index, chunk in enumerate(chunks):

        semantic_score = float(
            relevance_scores[index]
        )

        retrieval_score = float(
            chunk["score"]
        )

        # Normalize the Qdrant score approximately
        # so it can be combined with semantic score.
        combined_score = (
            0.65 * semantic_score
            + 0.35 * retrieval_score
        )

        candidates.append({
            "index": index,
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "retrieval_score": retrieval_score,
            "rerank_score": semantic_score,
            "combined_score": combined_score
        })

 
    selected = []

    first_chunk = max(
        candidates,
        key=lambda x: x["combined_score"]
    )

    selected.append(first_chunk)

    remaining = [
        item
        for item in candidates
        if item["index"] != first_chunk["index"]
    ]

 
    while (
        len(selected) < top_n
        and remaining
    ):

        best_candidate = None
        best_score = None

        for candidate in remaining:

            candidate_embedding = (
                document_embeddings[
                    candidate["index"]
                ]
            )

            similarities = []

            for selected_chunk in selected:

                selected_embedding = (
                    document_embeddings[
                        selected_chunk["index"]
                    ]
                )

                similarity = cos_sim(
                    candidate_embedding,
                    selected_embedding
                ).item()

                similarities.append(
                    similarity
                )

            max_similarity = max(
                similarities
            )

           
            diversity_score = (
                1.0 - max_similarity
            )

            final_score = (
                0.55 * candidate["combined_score"]
                + 0.45 * diversity_score
            )

            if (
                best_score is None
                or final_score > best_score
            ):
                best_score = final_score
                best_candidate = candidate

        selected.append(
            best_candidate
        )

        remaining = [
            item
            for item in remaining
            if item["index"]
            != best_candidate["index"]
        ]

   
    final_chunks = []

    for chunk in selected:

        final_chunks.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "retrieval_score": chunk[
                "retrieval_score"
            ],
            "rerank_score": chunk[
                "rerank_score"
            ]
        })

    return final_chunks