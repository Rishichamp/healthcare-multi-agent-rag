"""
RAG pipeline for the Healthcare Management System.
Retrieves relevant medical information from the knowledge base
to ground the LLM's responses in documented medical guidelines.
"""
import re, json, logging, hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)
_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _embed(texts: list[str]) -> list[list[float]]:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.embeddings.create(
        input=[t if t.strip() else "empty" for t in texts],
        model=settings.EMBEDDING_MODEL
    )
    return [d.embedding for d in resp.data]


def _chunk(text: str) -> list[str]:
    """Recursive chunking with overlap."""
    if len(text) <= settings.CHUNK_SIZE * 4:
        return [text.strip()] if text.strip() else []
    parts = re.split(r'\n\n|\n', text)
    chunks, current = [], ""
    for part in parts:
        if len(current) + len(part) + 2 <= settings.CHUNK_SIZE * 4:
            current = current + "\n\n" + part if current else part
        else:
            if current:
                chunks.append(current.strip())
            current = part
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


def ingest_document(text: str, title: str, category: str = "general") -> int:
    collection = get_collection()
    chunks = _chunk(text)
    if not chunks:
        return 0
    embeddings = _embed(chunks)
    doc_hash = hashlib.md5(title.encode()).hexdigest()[:8]
    ids = [f"{doc_hash}_c{i}" for i in range(len(chunks))]
    metas = [{"title": title, "category": category, "chunk_idx": i} for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)
    logger.info(f"Ingested '{title}': {len(chunks)} chunks")
    return len(chunks)


def ingest_directory() -> dict:
    results = {"success": [], "failed": []}
    for path in Path(settings.DOCS_PATH).rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
            n = ingest_document(text, title=path.stem.replace("_"," ").title(), category="medical")
            results["success"].append({"file": path.name, "chunks": n})
        except Exception as e:
            results["failed"].append({"file": path.name, "error": str(e)})
    return results


def retrieve(query: str, k: int = None) -> list[dict]:
    k = k or settings.TOP_K
    collection = get_collection()
    if collection.count() == 0:
        return []
    q_emb = _embed([query])[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(k, collection.count()),
        include=["documents", "distances", "metadatas"]
    )
    docs = []
    for i in range(len(results["ids"][0])):
        score = round(1.0 - results["distances"][0][i], 4)
        if score >= settings.SIMILARITY_THRESHOLD:
            docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": score,
            })
    return sorted(docs, key=lambda d: d["score"], reverse=True)


def get_medical_context(query: str) -> str:
    """Format retrieved docs as context string for LLM prompts."""
    docs = retrieve(query)
    if not docs:
        return ""
    parts = []
    for doc in docs:
        title = doc["metadata"].get("title", "Medical Reference")
        parts.append(f"[{title}]\n{doc['text']}")
    return "\n\n---\n\n".join(parts)


def get_collection_stats() -> dict:
    return {"total_chunks": get_collection().count(), "docs_path": settings.DOCS_PATH}
