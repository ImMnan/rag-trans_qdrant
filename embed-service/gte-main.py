import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer
from starlette.concurrency import run_in_threadpool

# Load model once at startup
embed_model = None
# Reranker is a separate cross-encoder model, independent of the embedding model above.
reranker_model = None
query_template = "{text}"
document_template = "{text}"


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
    global embed_model, reranker_model, query_template, document_template
    model_name = os.getenv("EMBED_MODEL", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    hf_home = os.getenv("HF_HOME", "/models")
    hub_cache = os.getenv("HF_HUB_CACHE")
    cache_folder = os.getenv("TRANSFORMERS_CACHE", hf_home)
    query_template = os.getenv("EMBED_QUERY_TEMPLATE", "{text}")
    document_template = os.getenv("EMBED_DOCUMENT_TEMPLATE", "{text}")
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
    input_type: str = "query"

class EmbedResponse(BaseModel):
    vector: list[float]


def format_embedding_input(text: str, input_type: str) -> str:
    normalized_type = input_type.strip().lower()
    if normalized_type == "query":
        template = query_template
    elif normalized_type == "document":
        template = document_template
    else:
        raise HTTPException(status_code=400, detail="input_type must be 'query' or 'document'")

    if "{text}" in template:
        return template.replace("{text}", text)
    return f"{template}{text}"

@app.post("/embed")
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Encode text into embeddings vector."""
    if embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded yet")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    
    try:
        text_for_embedding = format_embedding_input(request.text, request.input_type)
        vector = await run_in_threadpool(
            embed_model.encode, text_for_embedding, normalize_embeddings=True
        )
        return EmbedResponse(vector=vector.tolist())
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
        scores = await run_in_threadpool(reranker_model.predict, pairs)
        return RerankResponse(scores=scores.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rerank failed: {str(e)}")

class OpenAIEmbeddingsRequest(BaseModel):
    model: str = ""
    input: str | list[str]

class OpenAIEmbeddingDatum(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]

class OpenAIEmbeddingsResponse(BaseModel):
    object: str = "list"
    data: list[OpenAIEmbeddingDatum]
    model: str

@app.post("/v1/embeddings")
async def openai_embeddings(request: OpenAIEmbeddingsRequest) -> OpenAIEmbeddingsResponse:
    """OpenAI-compatible batch embeddings endpoint, so ingestion scripts written against a
    vLLM/OpenAI embeddings server (e.g. ingest_repo_vllm.py) can point at this service
    unmodified. Always embeds as documents — this endpoint has no query-time caller."""
    if embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded yet")

    texts = [request.input] if isinstance(request.input, str) else request.input
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="input cannot be empty")

    try:
        formatted = [format_embedding_input(t, "document") for t in texts]
        vectors = await run_in_threadpool(embed_model.encode, formatted, normalize_embeddings=True)
        data = [
            OpenAIEmbeddingDatum(index=i, embedding=vector.tolist())
            for i, vector in enumerate(vectors)
        ]
        return OpenAIEmbeddingsResponse(data=data, model=request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "embedding-service"}
