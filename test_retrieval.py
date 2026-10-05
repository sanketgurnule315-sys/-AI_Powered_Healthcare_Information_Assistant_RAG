from app.retriever import retrieve_top_10


question = "What is the attendance requirement?"


results = retrieve_top_10(
    question
)


print("================================")
print("TOP 10 RETRIEVED CHUNKS")
print("================================")


for index, result in enumerate(results):

    print("\n----------------------------")

    print(
        "Rank:",
        index + 1
    )

    print(
        "Page:",
        result["page"]
    )

    print(
        "Score:",
        result["score"]
    )

    print(
        "Text:",
        result["text"][:500]
    )