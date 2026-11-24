from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(host="localhost", port=6333)

q = "Can typed decisions rank single requests as well as LLM rankers when the graded attributes are with held?"
prefix = "Represent this sentence for searching relevant passages: "
vec = model.encode(prefix + q, normalize_embeddings=True).tolist()

for r in client.query_points("docs", query=vec, limit=4).points:
    print(r.payload["source"], "p.", r.payload["page"], round(r.score, 3))
    print(r.payload["text"] , "\n---")