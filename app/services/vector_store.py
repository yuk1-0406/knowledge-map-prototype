import chromadb
from chromadb.config import Settings
from typing import List, Dict
from .embeddings import get_embedding

_client = None
_collection = None

def init_store(persist_dir: str = "app/data/chroma"):
    global _client, _collection
    _client = chromadb.Client(Settings(persist_directory=persist_dir))
    _collection = _client.get_or_create_collection(name="notes")

def upsert_texts(items: List[Dict]):
    embeddings = [get_embedding(x["text"]) for x in items]
    _collection.upsert(
        ids=[x["id"] for x in items],
        documents=[x["text"] for x in items],
        metadatas=[x.get("meta", {}) for x in items],
        embeddings=embeddings
    )

def search(query: str, top_k: int = 10) -> List[Dict]:
    qemb = get_embedding(query)
    res = _collection.query(query_embeddings=[qemb], n_results=top_k)
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0] if "distances" in res else [None]*len(ids)
    for i in range(len(ids)):
        out.append({
            "id": ids[i],
            "text": docs[i],
            "score": float(distances[i]) if distances[i] is not None else None,
            "meta": metas[i]
        })
    return out
