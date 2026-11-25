import json, time, uuid
from pathlib import Path
import pymupdf
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from llm import llm

DOCS_DIR = Path("docs")
SUMMARIES = "doc_summaries"
MANIFEST = Path("summaries.json")

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(host="localhost", port=6333)
if not client.collection_exists(SUMMARIES):
    client.create_collection(
        SUMMARIES, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

done = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

def first_text(path, max_chars=4000):
    text = ""
    with pymupdf.open(path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
            if len(text) >= max_chars:
                break
    return text[:max_chars]

for pdf in sorted(DOCS_DIR.glob("*.pdf")):
    if pdf.name in done:
        continue
    try:
        text = first_text(pdf)
        out = llm.invoke(
            "Below is the beginning of a research paper. Write the paper's title on the "
            "first line, then a 3-sentence summary covering the problem, the method and "
            "the main result. Use only what the text says.\n\n" + text
        ).content.strip()
        vec = embedder.encode(out, normalize_embeddings=True).tolist()
        client.upsert(SUMMARIES, points=[PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, pdf.name)),
            vector=vec,
            payload={"source": pdf.name, "summary": out},
        )])
        done[pdf.name] = True
        MANIFEST.write_text(json.dumps(done))
        print("summarized:", pdf.name)
        time.sleep(6)   # stay under free-tier token limits
    except Exception as e:
        print("FAILED:", pdf.name, e)
        time.sleep(20)