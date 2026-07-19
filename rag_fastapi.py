import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


app = FastAPI(title="Hawk Cluster Pure RAG")

# 1. INITIALIZE INTERNAL SERVICES
# Uses K8s core DNS network routing paths
# Accepts either QDRANT_HOST or QDRANT_INTERNAL_URL for the Qdrant hostname
QDRANT_HOST = os.getenv("QDRANT_HOST") or os.getenv("QDRANT_INTERNAL_URL", "qdrant-service.hawk.svc.cluster.local")
_vllm_raw = os.getenv("VLLM_INTERNAL_URL", "")
# Normalize: ensure scheme and /v1 suffix required by the OpenAI client
if not _vllm_raw.startswith("http"):
    _vllm_raw = f"http://{_vllm_raw}"
VLLM_URL = _vllm_raw.rstrip("/") + "/v1"
MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# Native programmatic clients instead of manual requests hooks
db_client = QdrantClient(host=QDRANT_HOST, port=6333)
llm_client = OpenAI(base_url=VLLM_URL, api_key="not-needed")
embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

# Collection names are fixed — change_chunks holds commit/diff hunks,
# code_chunks holds source/doc snapshots used as reference context.
CHANGE_COLLECTION = os.getenv("CHANGE_COLLECTION", "change_chunks")
CODE_COLLECTION   = os.getenv("CODE_COLLECTION",   "code_chunks")


class RAGRequest(BaseModel):
    query_text: str
    repo_id: str
    type: str | None = None
    limit: int = 5


def _query_collection(collection: str, vector: list, repo_id: str, limit: int) -> list[str]:
    """Return text chunks from a single Qdrant collection filtered by repo_id."""
    resp = db_client.query_points(
        collection_name=collection,
        query=vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="repo_id",
                    match=models.MatchValue(value=repo_id),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    )
    return [
        hit.payload["text"]
        for hit in resp.points
        if hit.payload and "text" in hit.payload
    ]


@app.post("/api/v1/rag")
async def execute_rag_pipeline(request: RAGRequest):
    try:
        # A. Embed query once; reuse vector for both collections
        vector = embed_model.encode(
            f"query: {request.query_text}", normalize_embeddings=True
        ).tolist()

        # B. Pull diff/commit hunks (what changed)
        change_chunks = _query_collection(
            CHANGE_COLLECTION, vector, request.repo_id, request.limit
        )

        # C. Pull source/doc snapshots (reference context for the changes)
        code_chunks = _query_collection(
            CODE_COLLECTION, vector, request.repo_id, request.limit
        )

        change_context = "\n---\n".join(change_chunks) if change_chunks else "No change data found."
        code_context   = "\n---\n".join(code_chunks)   if code_chunks   else "No source context found."

        # D. Prompt strategy:
        # - type=standard -> structured release-note output via system prompt.
        # - otherwise -> direct Q&A from retrieved context only.
        is_standard_request = (request.type or "").strip().lower() == "standard"

        standard_system_prompt = (
            "You are a senior engineer producing release-note summaries. "
            "Use only the provided context. Structure your answer with these sections:\n"
            "1. **What Changed** - describe the commits/diffs concisely.\n"
            "2. **User Impact** - explain what end-users will notice or need to act on.\n"
            "3. **Security & Performance** - flag any security fixes or performance optimizations; "
            "write 'None identified' if absent."
        )
        standard_user_prompt = (
            f"## Diff / Change Hunks\n{change_context}\n\n"
            f"## Source / Doc Reference\n{code_context}\n\n"
            f"## Request\n{request.query_text}"
        )
        direct_user_prompt = (
            "Answer the user question using only the context below. "
            "Be concise and factual. If asked whether a feature is supported, answer with 'Yes' or 'No' "
            "and include when it first appears in the provided context if available; "
            "otherwise say 'Unknown based on provided context'.\n\n"
            f"## Diff / Change Hunks\n{change_context}\n\n"
            f"## Source / Doc Reference\n{code_context}\n\n"
            f"## Question\n{request.query_text}"
        )

        if is_standard_request:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": standard_system_prompt},
                    {"role": "user", "content": standard_user_prompt},
                ],
                temperature=0.3,
            )
        else:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": direct_user_prompt}],
                temperature=0.1,
            )
        return {
            "answer": response.choices[0].message.content,
            "sources": {
                "change_chunks_retrieved": len(change_chunks),
                "code_chunks_retrieved":   len(code_chunks),
            },
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline exception triggered: {str(e)}")
