from typing import List
import chromadb
from app.core.config import settings

_client: chromadb.HttpClient | None = None
_collection = None


def get_client() -> chromadb.HttpClient:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(settings.CHROMA_COLLECTION)
    return _collection


def upsert_texts(ids: List[str], texts: List[str], metadatas: List[dict] | None = None):
    col = get_collection()
    col.upsert(ids=ids, documents=texts, metadatas=metadatas)


def query_text(q: str, n: int = 3):
    col = get_collection()
    return col.query(query_texts=[q], n_results=n)
