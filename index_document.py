from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from app.pdf_processor import (
    extract_text_from_pdf,
    create_chunks
)

from app.embeddings import get_embeddings

from app.config import (
    QDRANT_URL,
    QDRANT_COLLECTION
)


PDF_PATH = "data/healthcare_RAG_document.pdf"


def main():

    print("=" * 70)
    print("INDEXING HEALTHCARE_RAG_DOCUMENT.PDF")
    print("=" * 70)

    # ------------------------------------------------
    # 1. Extract PDF text
    # ------------------------------------------------

    print()
    print("1. Reading PDF...")

    pages = extract_text_from_pdf(
        PDF_PATH
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    # ------------------------------------------------
    # 2. Create improved chunks
    # ------------------------------------------------

    print()
    print("2. Creating chunks...")

    chunks = create_chunks(
        pages,
        chunk_size=1600,
        overlap=400
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # ------------------------------------------------
    # 3. Add chunk IDs
    # ------------------------------------------------

    for index, chunk in enumerate(chunks):

        chunk["chunk_id"] = index

    # ------------------------------------------------
    # 4. Basic healthcare content check
    # ------------------------------------------------

    print()
    print("3. Checking healthcare content...")

    healthcare_keywords = [
        "diabetes", "hypertension", "asthma", "nutrition",
        "first aid", "vaccination", "medicine safety",
        "emergency", "child health", "laboratory tests"
    ]

    matches = 0
    for chunk in chunks:
        if any(k in chunk["text"].lower() for k in healthcare_keywords):
            matches += 1
    print(f"Healthcare-relevant chunks found: {matches}")

    # ------------------------------------------------
    # 5. Create embeddings
    # ------------------------------------------------

    print()
    print("4. Creating embeddings...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = get_embeddings(
        texts
    )

    print(
        f"Embeddings created: {len(embeddings)}"
    )

    # ------------------------------------------------
    # 6. Connect to Qdrant
    # ------------------------------------------------

    print()
    print("5. Connecting to Qdrant...")

    client = QdrantClient(
        url=QDRANT_URL
    )

    # ------------------------------------------------
    # 7. Get embedding dimension
    # ------------------------------------------------

    vector_size = len(
        embeddings[0]
    )

    print(
        f"Embedding dimension: {vector_size}"
    )

    # ------------------------------------------------
    # 8. Delete old collection
    # ------------------------------------------------

    print()
    print(
        "6. Removing old Qdrant collection..."
    )

    try:

        client.delete_collection(
            collection_name=QDRANT_COLLECTION
        )

        print(
            "Old collection deleted."
        )

    except Exception:

        print(
            "Old collection did not exist."
        )

    # ------------------------------------------------
    # 9. Create new collection
    # ------------------------------------------------

    print()
    print(
        "7. Creating new Qdrant collection..."
    )

    client.create_collection(

        collection_name=QDRANT_COLLECTION,

        vectors_config=VectorParams(

            size=vector_size,

            distance=Distance.COSINE
        )
    )

    # ------------------------------------------------
    # 10. Create Qdrant points
    # ------------------------------------------------

    print()
    print(
        "8. Preparing Qdrant points..."
    )

    points = []

    for index, chunk in enumerate(chunks):

        points.append(

            PointStruct(

                id=index,

                vector=embeddings[index],

                payload={

                    "text": chunk["text"],

                    "source": chunk["source"],

                    "page": chunk["page"],

                    "chunk_id": chunk["chunk_id"]
                }
            )
        )

    # ------------------------------------------------
    # 11. Upload points
    # ------------------------------------------------

    print()
    print(
        "9. Uploading points to Qdrant..."
    )

    client.upsert(

        collection_name=QDRANT_COLLECTION,

        points=points
    )

    print()
    print("=" * 70)
    print("INDEXING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(
        f"Total chunks indexed: {len(chunks)}"
    )


if __name__ == "__main__":

    main()