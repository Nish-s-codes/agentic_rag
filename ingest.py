import time; start = time.time()
import hashlib, json, uuid
from pathlib import Path
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

DOCS_DIR = Path("docs")
MANIFEST = Path("manifest.json")
COLLECTION = "docs"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(host="localhost", port=6333)
if not client.collection_exists(COLLECTION):
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
done = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()

def parse(path):
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                yield i + 1, text

pdfs = sorted(DOCS_DIR.glob("*.pdf"))
for n, pdf in enumerate(pdfs, 1):
    h = file_hash(pdf)
    if done.get(pdf.name) == h:
        continue
    try:
        texts, metas = [], []
        for page_no, text in parse(pdf):
            for chunk in splitter.split_text(text):
                texts.append(chunk)
                metas.append({"source": pdf.name, "page": page_no})
        if not texts:
            print("No text (scanned?):", pdf.name)
            continue

        vectors = model.encode(texts, batch_size=64, normalize_embeddings=True)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf.name}:{i}")),
                vector=v.tolist(),
                payload={**m, "text": t},
            )
            for i, (v, m, t) in enumerate(zip(vectors, metas, texts))
        ]
        for j in range(0, len(points), 256):
            client.upsert(COLLECTION, points=points[j : j + 256])

        done[pdf.name] = h
        MANIFEST.write_text(json.dumps(done))
        print(f"[{n}/{len(pdfs)}] {pdf.name}: {len(points)} chunks")
    except Exception as e:
        print("FAILED:", pdf.name, e)

print("Total minutes:", (time.time() - start) / 60)