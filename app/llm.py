import json
import requests

from app.config import (
    NUGEN_API_KEY,
    NUGEN_LLM_MODEL
)


NUGEN_CHAT_URL = (
    "https://api.nugen.in/api/v3/inference/chat/completions"
)


def get_headers():

    return {
        "Authorization": f"Bearer {NUGEN_API_KEY}",
        "Content-Type": "application/json"
    }


def extract_answer(data):

    if "choices" in data:

        choice = data["choices"][0]

        if "message" in choice:

            return choice["message"]["content"].strip()

        if "text" in choice:

            return choice["text"].strip()

    raise ValueError(
        f"Unexpected Nugen response: {data}"
    )


def generate_answer(prompt):

    payload = {
        "model": NUGEN_LLM_MODEL,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "max_tokens": 80,

        "temperature": 0.2,

        "stream": False
    }

    response = requests.post(
        NUGEN_CHAT_URL,
        headers=get_headers(),
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return extract_answer(data)


def stream_answer(prompt):

    payload = {
        "model": NUGEN_LLM_MODEL,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "max_tokens": 80,

        "temperature": 0.2,

        "stream": True
    }

    response = requests.post(
        NUGEN_CHAT_URL,
        headers=get_headers(),
        json=payload,
        stream=True,
        timeout=120
    )

    response.raise_for_status()

    for line in response.iter_lines(
        decode_unicode=True
    ):

        if not line:
            continue

        if line.startswith("data:"):

            data_text = line[
                len("data:"):
            ].strip()

            if data_text == "[DONE]":
                break

            try:

                data = json.loads(
                    data_text
                )

            except json.JSONDecodeError:

                continue

            if "choices" not in data:
                continue

            choice = data["choices"][0]

            if "delta" in choice:

                delta = choice["delta"]

                content = delta.get(
                    "content"
                )

                if content:
                    yield content

            elif "text" in choice:

                content = choice["text"]

                if content:
                    yield content