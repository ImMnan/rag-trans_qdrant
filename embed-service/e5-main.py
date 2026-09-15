import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

# Load model once at startup
embed_model = None
# Reranker is a separate cross-encoder model, independent of the embedding model above.
reranker_model = None


def resolve_model_path(model_name: str, hub_cache: Optional[str]) -> str:
    if not hub_cache:
        return model_name

    hub_root = Path(hub_cache)
    repo_cache_dir = hub_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    refs_main = repo_cache_dir / "refs" / "main"

    if refs_main.is_file():
        snapshot_id = refs_main.read_text().strip()
        snapshot_path = snapshots_dir / snapshot_id
        if snapshot_path.is_dir():
            return str(snapshot_path)

    if snapshots_dir.is_dir():
        snapshot_dirs = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshot_dirs:
            return str(snapshot_dirs[0])

    return model_name

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embed_model, reranker_model
    model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
    hf_home = os.getenv("HF_HOME", "/models")
    hub_cache = os.getenv("HF_HUB_CACHE")
    cache_folder = os.getenv("TRANSFORMERS_CACHE", hf_home)
    model_path = resolve_model_path(model_name, hub_cache)
    print(
        f"Loading embedding model: {model_name} using model path: {model_path} "
        f"and writable cache: {cache_folder}"
    )
    embed_model = SentenceTransformer(
        model_path,
        cache_folder=cache_folder,
        local_files_only=True,
    )

    reranker_name = os.getenv("RERANKER_MODEL", "")
    if reranker_name:
        reranker_path = resolve_model_path(reranker_name, hub_cache)
        print(f"Loading reranker model: {reranker_name} using model path: {reranker_path}")
        reranker_model = CrossEncoder(reranker_path, trust_remote_code=True, local_files_only=True)
    else:
        print("RERANKER_MODEL not set, /rerank endpoint will return 503")

    yield
    # Shutdown (optional cleanup)
    print("Embedding service shutting down")

app = FastAPI(title="Embedding Service", lifespan=lifespan)

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    vector: list[float]

@app.post("/embed")
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Encode text into embeddings vector."""
    if embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded yet")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    
    try:
        # Match the prefix from original RAG pipeline
        vector = embed_model.encode(
            f"query: {request.text}",
            normalize_embeddings=True
        ).tolist()
        return EmbedResponse(vector=vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

class RerankRequest(BaseModel):
    query: str
    documents: list[str]

class RerankResponse(BaseModel):
    scores: list[float]

@app.post("/rerank")
async def rerank(request: RerankRequest) -> RerankResponse:
    """Score each document's relevance to the query with a cross-encoder."""
    if reranker_model is None:
        raise HTTPException(status_code=503, detail="Reranker model not loaded; set RERANKER_MODEL")

    if not request.documents:
        return RerankResponse(scores=[])

    try:
        pairs = [[request.query, doc] for doc in request.documents]
        scores = reranker_model.predict(pairs).tolist()
        return RerankResponse(scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rerank failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "embedding-service"}
