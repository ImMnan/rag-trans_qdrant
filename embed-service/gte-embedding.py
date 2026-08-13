import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load model once at startup
embed_model = None
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
    global embed_model, query_template, document_template
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
        vector = embed_model.encode(
            text_for_embedding,
            normalize_embeddings=True
        ).tolist()
        return EmbedResponse(vector=vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "embedding-service"}
