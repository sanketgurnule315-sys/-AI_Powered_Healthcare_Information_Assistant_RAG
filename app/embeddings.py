import requests

from app.config import NUGEN_API_KEY, NUGEN_EMBEDDING_MODEL


NUGEN_EMBEDDING_URL = (
    "https://api.nugen.in/api/v3/inference/embeddings"
)


def get_embeddings(texts):

    headers = {
        "Authorization": f"Bearer {NUGEN_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": NUGEN_EMBEDDING_MODEL,
        "input": texts
    }

    response = requests.post(
        NUGEN_EMBEDDING_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        detail = response.text.strip()
        if detail:
            raise requests.HTTPError(
                f"{error}\nNugen response: {detail}",
                response=response,
            ) from error
        raise

    data = response.json()

    embeddings = []

    for item in data["data"]:
        embeddings.append(
            item["embedding"]
        )

    return embeddings


def get_embedding(text):

    embeddings = get_embeddings([text])

    return embeddings[0]


