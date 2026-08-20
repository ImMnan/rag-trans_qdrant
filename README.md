## rag-trans_qdrant

### Download the model from HuggingFace Hub & Uploading to GCP bucket

```bash
hf download Alibaba-NLP/gte-Qwen2-1.5B-instruct --local-dir /Desktop/gte/

gcloud storage cp -r /Desktop/gte/* gs://rag-model-weights-hawk/gte-model/
```


