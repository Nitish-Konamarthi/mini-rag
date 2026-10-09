import pathlib
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE, OVERLAP = 800, 100

embedder = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="./db").get_or_create_collection(
    "docs", metadata={"hnsw:space": "cosine"}
)

def load(path):
    if path.suffix == ".pdf":
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")

def chunk(text):
    step = CHUNK_SIZE - OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]

for path in pathlib.Path("data").glob("*"):
    if path.suffix not in (".txt", ".pdf"):
        continue
    chunks = [c for c in chunk(load(path)) if c.strip()]
    collection.upsert(
        ids=[f"{path.name}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embedder.encode(chunks).tolist(),
        metadatas=[{"source": path.name}] * len(chunks),
    )
    print(f"Indexed {path.name}: {len(chunks)} chunks")