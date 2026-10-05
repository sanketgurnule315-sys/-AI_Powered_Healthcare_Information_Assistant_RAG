import pymupdf


PDF_PATH: str = "data/healthcare_RAG_document.pdf"
CHUNK_SIZE: int = 256
CHUNK_OVERLAP: int = 64


# --------------------------------------------------
# 1. Open PDF and extract pages
# --------------------------------------------------

def extract_text_from_pdf(pdf_path):

    document = pymupdf.open(pdf_path)

    print("PDF type:", type(document))

    print(
        "Total pages:",
        len(document)
    )

    pages = []

    for page_number in range(
        len(document)
    ):

        page = document[page_number]

        text = page.get_text()

        if text.strip():

            pages.append({

                "source": pdf_path,

                "page": page_number + 1,

                "text": text.strip()

            })

    document.close()

    return pages


# --------------------------------------------------
# 2. Split text into paragraphs
# --------------------------------------------------

def split_into_paragraphs(text):

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    lines = text.split("\n")

    paragraphs = []

    current_paragraph = []

    for line in lines:

        line = line.strip()

        if not line:

            if current_paragraph:

                paragraph = " ".join(
                    current_paragraph
                )

                paragraphs.append(
                    paragraph
                )

                current_paragraph = []

        else:

            current_paragraph.append(
                line
            )

    if current_paragraph:

        paragraph = " ".join(
            current_paragraph
        )

        paragraphs.append(
            paragraph
        )

    return paragraphs


# --------------------------------------------------
# 3. Create intelligent chunks
# --------------------------------------------------

def create_chunks(
    pages,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    chunk_id = 1

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        paragraphs = split_into_paragraphs(
            text
        )

        current_chunk = ""

        for paragraph in paragraphs:

            # --------------------------------------
            # Large paragraph
            # --------------------------------------

            if len(paragraph) > chunk_size:

                if current_chunk.strip():

                    chunks.append({

                        "chunk_id": chunk_id,

                        "page": page_number,

                        "source": page_data["source"],

                        "text": current_chunk.strip()

                    })

                    chunk_id += 1

                    current_chunk = ""


                # Split large paragraph with overlap

                start = 0

                while start < len(paragraph):

                    end = start + chunk_size

                    piece = paragraph[
                        start:end
                    ].strip()

                    if piece:

                        chunks.append({

                            "chunk_id": chunk_id,

                            "page": page_number,

                            "source": page_data["source"],

                            "text": piece

                        })

                        chunk_id += 1


                    start += (
                        chunk_size - overlap
                    )

                continue


            # --------------------------------------
            # First paragraph
            # --------------------------------------

            if not current_chunk:

                current_chunk = paragraph

                continue


            # --------------------------------------
            # Try adding paragraph
            # --------------------------------------

            candidate = (
                current_chunk
                + "\n\n"
                + paragraph
            )


            if len(candidate) <= chunk_size:

                current_chunk = candidate

                continue


            # --------------------------------------
            # Save current chunk
            # --------------------------------------

            chunks.append({

                "chunk_id": chunk_id,

                "page": page_number,

                "source": page_data["source"],

                "text": current_chunk.strip()

            })

            chunk_id += 1


            # --------------------------------------
            # Create 300-character overlap
            # --------------------------------------

            words = current_chunk.split()

            overlap_words = []

            length = 0


            for word in reversed(words):

                word_length = len(word) + 1

                if (
                    length + word_length
                    <= overlap
                ):

                    overlap_words.insert(
                        0,
                        word
                    )

                    length += word_length

                else:

                    break


            overlap_text = " ".join(
                overlap_words
            )


            # --------------------------------------
            # Start next chunk with overlap
            # --------------------------------------

            if overlap_text:

                current_chunk = (
                    overlap_text
                    + "\n\n"
                    + paragraph
                )

            else:

                current_chunk = paragraph


        # ------------------------------------------
        # Save final chunk
        # ------------------------------------------

        if current_chunk.strip():

            chunks.append({

                "chunk_id": chunk_id,

                "page": page_number,

                "source": page_data["source"],

                "text": current_chunk.strip()

            })

            chunk_id += 1


    return chunks


# --------------------------------------------------
# 4. Check healthcare content

def check_healthcare_chunks(chunks):

    print("\n")
    print("=" * 80)
    print("HEALTHCARE CONTENT CHECK")
    print("=" * 80)

    keywords = [
        "healthcare", "diabetes", "hypertension", "asthma",
        "nutrition", "first aid", "vaccination", "medicine safety",
        "emergency", "when to seek medical help", "menstrual health",
        "child health", "laboratory tests"
    ]

    found = False
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        matches = [k for k in keywords if k in text_lower]
        if matches:
            found = True
            print("\nChunk ID:", chunk["chunk_id"])
            print("Page:", chunk["page"])
            print("Found:", ", ".join(matches))
            print("\nText:")
            print(chunk["text"])
            print("-" * 80)
    if not found:
        print("No healthcare content found.")


# 5. Main
# --------------------------------------------------

def main():

    print("=" * 80)

    print(
        "HEALTHCARE PDF CHUNKING"
    )

    print("=" * 80)


    pages = extract_text_from_pdf(
        PDF_PATH
    )


    print(
        "\nPages extracted:",
        len(pages)
    )


    chunks = create_chunks(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )


    print(
        "Total chunks:",
        len(chunks)
    )


    print(
        "Chunk size:",
        CHUNK_SIZE
    )


    print(
        "Chunk overlap:",
        CHUNK_OVERLAP
    )


    check_healthcare_chunks(
        chunks
    )


    print("\n")

    print("=" * 80)

    print("FIRST 5 CHUNKS")

    print("=" * 80)


    for chunk in chunks[:5]:

        print("\n")

        print(
            "Chunk ID:",
            chunk["chunk_id"]
        )

        print(
            "Page:",
            chunk["page"]
        )

        print(
            "Characters:",
            len(chunk["text"])
        )

        print(
            chunk["text"]
        )

        print(
            "-" * 80
        )


if __name__ == "__main__":

    main()