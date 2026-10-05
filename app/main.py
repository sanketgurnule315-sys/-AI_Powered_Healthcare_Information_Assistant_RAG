from fastapi import FastAPI
from pydantic import BaseModel

from app.query_enhancer import enhance_query
from app.retriever import retrieve_top_10
from app.reranker import rerank_documents
from app.prompts import build_prompt
from app.llm import generate_answer



# FASTAPI APPLICATION


app = FastAPI(
    title="AI Healthcare Information Assistant using RAG",
    description=(
        "AI-powered Healthcare Information Assistant using "
        "Retrieval-Augmented Generation (RAG)."
    ),
    version="1.0.0"
)






class QuestionRequest(BaseModel):
    question: str



# TOP 3 CHUNK MODEL


class RetrievedChunk(BaseModel):
    rank: int
    page: int
    source: str
    score: float
    text: str





class QueryResponse(BaseModel):
    question: str
    answer: str

    # Number of embeddings retrieved before reranking
    number_of_embeddings: int

    # Number of final reranked chunks
    number_of_reranked_chunks: int

    # Top 3 retrieved/reranked chunks
    reranked_chunks: list[RetrievedChunk]




@app.get(
    "/",
    tags=["General"],
    summary="API Information"
)
def root():

    return {
        "message": (
            "AI Healthcare Information Assistant "
            "using RAG API is running"
        ),
        "status": "active"
    }



# HEALTH CHECK


@app.get(
    "/health",
    tags=["Health"],
    summary="Healthcare RAG System Health Check"
)
def health_check():

    return {
        "status": "healthy",
        "service": (
            "AI Healthcare Information Assistant "
            "using RAG"
        )
    }




@app.post(
    "/ask",
    response_model=QueryResponse,
    tags=["Healthcare"],
    summary="Ask a Healthcare Question"
)
def ask_question(request: QuestionRequest):

    

    question = request.question


    

    enhanced_query = enhance_query(
        question
    )


    

    top_10 = retrieve_top_10(
        enhanced_query
    )


   

    top_3 = rerank_documents(
        enhanced_query,
        top_10,
        top_n=3
    )




    prompt = build_prompt(
        question,
        top_3
    )



    answer = generate_answer(
        prompt
    )



    final_chunks = []

    for index, chunk in enumerate(top_3):

        final_chunks.append(
            RetrievedChunk(
                rank=index + 1,
                page=chunk["page"],
                source=chunk["source"],
                score=chunk["retrieval_score"],
                text=chunk["text"]
            )
        )


     # STEP 8: FINAL RESPONSE
    

    return QueryResponse(
        question=question,
        answer=answer,

        # Top 10 embeddings retrieved
        number_of_embeddings=len(top_10),

        # Final 3 chunks after reranking
        number_of_reranked_chunks=len(top_3),

        # Top 3 chunks
        reranked_chunks=final_chunks
    )