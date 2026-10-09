import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads GROQ_API_KEY from .env

MODEL = "openai/gpt-oss-20b"
TOP_K = 4
MAX_DISTANCE = 0.65   # tune this by watching the printed distances

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not found. Check your .env file.")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="./db").get_collection("docs")
llm = Groq(api_key=api_key)

def ask(question):
    q_vec = embedder.encode([question]).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=TOP_K)

    hits = [
        (c, m, d)
        for c, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        if d < MAX_DISTANCE
    ]

    print("\n--- Retrieved chunks ---")
    for c, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        kept = "USED" if d < MAX_DISTANCE else "skipped"
        print(f"[{m['source']}] distance={d:.3f} ({kept})\n{c[:150]}...\n")

    if not hits:
        return "I don't know based on the documents."

    context = "\n\n".join(c for c, _, _ in hits)
    prompt = (
        "Answer the question using ONLY the context below. "
        "Only include items that directly answer the question; ignore unrelated context. "
        "If the answer is not in the context, say \"I don't know based on the documents.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    r = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000,
    )
    return r.choices[0].message.content

while True:
    q = input("\nAsk (or 'quit'): ")
    if q.strip().lower() == "quit":
        break
    print("\nAnswer:", ask(q))