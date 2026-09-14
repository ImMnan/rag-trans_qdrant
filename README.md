## rag-trans_qdrant

### Download the model from HuggingFace Hub & Uploading to GCP bucket

```bash
hf download Alibaba-NLP/gte-Qwen2-1.5B-instruct --local-dir /Desktop/gte/

gcloud storage cp -r /Desktop/gte/* gs://rag-model-weights-hawk/gte-model/
```

### Build and publish Orca

The `VERSION` argument is required. It is used as both the Docker image tag and
the Orca version reported in the application startup log.

Build the container image locally:

```bash
make build-image VERSION=1.2.3
```

Push an image that has already been built to Docker Hub:

```bash
docker login
make push-image VERSION=1.2.3
```

Build and push the image in one command:

```bash
docker login
make release VERSION=1.2.3
```

These commands publish the image as `immnan/orca-rag:1.2.3`. Replace `1.2.3`
with the tag you want to release.

To build only the Linux Go binary with the same embedded version:

```bash
make build-binary VERSION=1.2.3
```

List all available targets or remove local binary artifacts:

```bash
make help
make clean
```


```sh
curl -X POST "http://orca-infer.ai/api/v1/rag-go/generate-doc" \
     -H "Content-Type: application/json" \
     -d '{ "query_text": "How to run selenium test with specific version of chromedriver", "repo_id": "github.com/Blazemeter/taurus", "type":"direct", "limit": 15}'
```


