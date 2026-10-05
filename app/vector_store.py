from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams,PointStruct
from app.config import QDRANT_URL,QDRANT_COLLECTION


client = QdrantClient(url=QDRANT_URL)


def create_collection(vector_size):

    collections = client.get_collections()

    collection_names = [
        collection.name
        for collection in collections.collections
    ]

    if QDRANT_COLLECTION in collection_names:

        client.delete_collection(
            collection_name=QDRANT_COLLECTION
        )

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )


def insert_chunks(chunks, embeddings):

    points = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=index,
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "chunk_id": index
                }
            )
        )

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )


def search_vectors(query_embedding,limit=10):
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=limit,
        with_payload=True
    )

    return results.points