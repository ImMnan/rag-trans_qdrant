import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import os

# CONFIGURATION
# 1. Qdrant Connection Info
RAW_IP_ADDRESS = os.environ.get("GATEWAY_IP")  # Your Qdrant pod/ingress node IP
PATH_PREFIX = "q1"                       # Ingress target prefix
HOST_HEADER = "infer.hawk-llm.ai"        # Your Ingress target host domain
QDRANT_BASE = f"http://{RAW_IP_ADDRESS}:80/{PATH_PREFIX}/collections"
MAX_TOKEN = 2048
# 2. vLLM Qwen 2.5 Connection Info
# VLLM_API_URL = "http://localhost:8000/v1" 
VLLM_API_URL = "http://infer.hawk-llm.ai/v1"  # Update to your vLLM Service IP/Port
QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"    # The exact name used in your vLLM launch script

# CLIENT INITIALIZATION
print("Initializing E5-Large locally for query encoding...")
embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

print("Connecting to vLLM OpenAI-compatible endpoint...")
llm_client = OpenAI(base_url=VLLM_API_URL, api_key="not-needed-for-vllm")

# RETRIEVAL LOGIC
def query_qdrant_collection(collection_name: str, query_vector: list, repo_id: str, repo_name: str, limit: int = 3):
    url = f"{QDRANT_BASE}/{collection_name}/points/search"
    headers = {"Host": HOST_HEADER, "Content-Type": "application/json"}
    
    # The filter blocks out all other repos at the indexing layer
    payload = {
        "vector": query_vector,
        "limit": limit,
        "filter": {
            "must": [
                {
                    "key": "repo_id",
                    "match": {"value": repo_id}
                }
            ]
        },
        "with_payload": True
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    return response.json().get("result", [])


def run_rag_pipeline(user_query: str):
    print(f"\n[User Query]: {user_query}")
    
    # CRITICAL E5-LARGE RULE: Queries MUST be prefixed with 'query: '
    e5_query_input = f"query: {user_query}"
    query_vector = embed_model.encode(e5_query_input).tolist()
    
    # 1. Search Change Chunks (What happened in commits/sprints recently?)
    print("Searching Git change history...")
    change_results = query_qdrant_collection("change_chunks", query_vector, repo_id="github.com/Blazemeter/helm-crane", repo_name="helm-crane", limit=3)
    
    # 2. Search Code Chunks (What did the structural repository layout look like?)
    print("Searching base code snapshots...")
    code_results = query_qdrant_collection("code_chunks", query_vector, repo_id="github.com/Blazemeter/helm-crane", repo_name="helm-crane", limit=3)
    
    # 3. Format Context beautifully for Qwen 2.5 (No 'query:' or 'passage:' tags here!)
    context_str = ""
    
    if change_results:
        context_str += "--- RECENT CHANGELOG & COMMITS DELTAS ---\n"
        for match in change_results:
            p = match["payload"]
            context_str += f"Commit: {p.get('commit_sha', 'N/A')} | Author: {p.get('author', 'N/A')} | Message: {p.get('commit_message', 'N/A')}\n"
            context_str += f"File: {p.get('file_path', 'N/A')}\nDiff Context:\n{p.get('text', '')}\n\n"
            
    if code_results:
        context_str += "--- STATIC BASELINE SOURCE CODE CHUNKS ---\n"
        for match in code_results:
            p = match["payload"]
            context_str += f"File: {p.get('file_path', 'N/A')} (Chunk #{p.get('chunk_index', 0)})\n"
            context_str += f"Code Snippet:\n{p.get('text', '')}\n\n"

    # 4. Construct the Final Prompt
    system_prompt = (
        "You are an expert software engineering assistant. Use the provided repository "
        "context (which includes baseline source code and historical git commit diffs) "
        "to answer the question accurately. If you don't know the answer based on the context, say so."
    )
    
    user_prompt = f"""Review the following context and answer the user query.

[REPOSITORY CONTEXT]
{context_str}

[USER QUERY]
{user_query}c

Answer:"""

    # 5. Stream or generate response from vLLM Qwen 2.5
    print("Sending contextual payload to Qwen 2.5 on vLLM...")
    response = llm_client.chat.completions.create(
        model=QWEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2, # Low temperature for accurate, non-hallucinated code responses
        max_tokens=MAX_TOKEN
    )
    
    return response.choices[0].message.content

# EXECUTION TEST RUN
if __name__ == "__main__":
    # Test query that touches both structural code and updates
    test_query = "<Your question>"
    
    qwen_response = run_rag_pipeline(test_query)
    
    print("\n[QWEN 2.5 RESPONSE]:")
    print(qwen_response)
