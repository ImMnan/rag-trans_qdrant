import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load model once at startup
embed_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embed_model
    model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
    # HF_HUB_CACHE is the hub blob directory (models--org--name/); fall back to HF_HOME/hub
    hf_home = os.getenv("HF_HOME", "/models")
    cache_folder = os.getenv("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    print(f"Loading embedding model: {model_name} from cache: {cache_folder}")
    embed_model = SentenceTransformer(model_name, cache_folder=cache_folder)
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

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "embedding-service"}
