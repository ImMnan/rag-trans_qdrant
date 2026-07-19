import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Hawk RAG Core")

# Pull clean internal service addresses from K8s env variables
QDRANT_URL = os.getenv("QDRANT_INTERNAL_URL")
VLLM_URL = os.getenv("VLLM_INTERNAL_URL")
MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")  # Default to Qwen 2.5 if not set

# Internal LLM client directly targets the clusterIP service
llm_client = OpenAI(base_url=VLLM_URL, api_key="not-needed")

class RAGRequest(BaseModel):
    query_text: str
    query_vector: list[float]
    collection_name: str
    repo_id: str
    limit: int = 3

@app.post("/api/v1/rag")
async def execute_rag(request: RAGRequest):
    # 1. Internal Qdrant Fetch
    # Points directly to http://cluster.local...
    search_url = f"{QDRANT_URL}/collections/{request.collection_name}/points/search"
    
    payload = {
        "vector": request.query_vector,
        "limit": request.limit,
        "filter": {"must": [{"key": "repo_id", "match": {"value": request.repo_id}}]},
        "with_payload": True
    }
    
    try:
        res = requests.post(search_url, json=payload, timeout=5)
        points = res.json().get("result", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Qdrant internal error: {str(e)}")

    # 2. Process chunks
    context = "\n---\n".join([p["payload"]["text"] for p in points if "text" in p.get("payload", {})])
    if not context: 
        context = "No context available."

    # 3. Internal vLLM Generation
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Answer using only the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {request.query_text}"}
            ],
            temperature=0.3
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vLLM internal error: {str(e)}")

@app.get("/healthz")
async def health():
    return {"status": "ok"}
