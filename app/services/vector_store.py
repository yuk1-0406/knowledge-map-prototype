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

# --- Helpers for inspecting DB state ---

def get_count() -> int:
    """Return number of items in the 'notes' collection."""
    if _collection is None:
        return 0
    try:
        return int(_collection.count())
    except Exception:
        return 0


def list_items(limit: int = 50, offset: int = 0) -> List[Dict]:
    """Return up to `limit` items with id, text, meta for inspection."""
    if _collection is None:
        return []
    limit = max(0, int(limit))
    offset = max(0, int(offset))
    # Chroma 0.5.x: include accepts only embeddings, documents, metadatas, uris, data
    res = None
    try:
        res = _collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])
    except TypeError:
        try:
            res = _collection.get(limit=limit, offset=offset)
        except TypeError:
            res = _collection.get(limit=limit)
    if not isinstance(res, dict):
        return []
    ids = res.get("ids", []) if isinstance(res.get("ids", []), list) else []
    docs = res.get("documents", []) if isinstance(res.get("documents", []), list) else []
    metas = res.get("metadatas", []) if isinstance(res.get("metadatas", []), list) else []
    items: List[Dict] = []
    for i, _id in enumerate(ids):
        doc = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        items.append({"id": _id, "text": doc, "meta": meta})
    return items

# --- Deletion helpers ---

def delete_by_ids(ids: List[str]) -> int:
    """Delete items by their IDs. Returns the number of requested deletions."""
    if _collection is None or not ids:
        return 0
    try:
        # Best-effort: Chroma delete doesn't return count
        _collection.delete(ids=list(set(ids)))
        return len(set(ids))
    except Exception:
        return 0


def delete_all() -> int:
    """Delete all items in the collection. Returns number of items attempted to delete."""
    if _collection is None:
        return 0
    try:
        total = int(_collection.count())
        if total == 0:
            return 0
        res = _collection.get(limit=total)
        all_ids = res.get("ids", []) if isinstance(res.get("ids", []), list) else []
        if not all_ids:
            return 0
        _collection.delete(ids=all_ids)
        return len(all_ids)
    except Exception:
        return 0
