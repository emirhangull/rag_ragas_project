from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


class QdrantIndexer:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        vector_size: int,
        path: str | None = None,
    ) -> None:
        if path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {item.name for item in self.client.get_collections().collections}
        if self.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def is_ready(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def upsert_chunks(
        self,
        *,
        doc_id: str,
        chunks: list[str],
        vectors: list[list[float]],
        source_name: str,
    ) -> int:
        points: list[PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload: dict[str, Any] = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{idx}",
                "chunk_index": idx,
                "text": chunk,
                "source_name": source_name,
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, payload["chunk_id"])),
                    vector=vector,
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def retrieve(
        self,
        query_vector: list[float],
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = None
        if doc_id:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "score": float(point.score),
                    "doc_id": payload.get("doc_id", ""),
                    "chunk_id": payload.get("chunk_id", ""),
                    "chunk_index": payload.get("chunk_index", -1),
                    "source_name": payload.get("source_name", ""),
                    "text": payload.get("text", ""),
                }
            )
        return results

    def list_chunks(self, doc_id: str, limit: int = 200) -> list[dict[str, Any]]:
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results: list[dict[str, Any]] = []
        for point in points:
            payload = point.payload or {}
            results.append(
                {
                    "doc_id": payload.get("doc_id", ""),
                    "chunk_id": payload.get("chunk_id", ""),
                    "chunk_index": payload.get("chunk_index", -1),
                    "source_name": payload.get("source_name", ""),
                    "text": payload.get("text", ""),
                }
            )
        results.sort(key=lambda row: row.get("chunk_index", -1))
        return results

    def scroll_all(self, limit: int = 2000) -> list[dict[str, Any]]:
        """Return up to *limit* chunks from the entire collection (no doc_id filter).

        Used for global lexical search across all documents.
        """
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results: list[dict[str, Any]] = []
        for point in points:
            payload = point.payload or {}
            results.append(
                {
                    "doc_id": payload.get("doc_id", ""),
                    "chunk_id": payload.get("chunk_id", ""),
                    "chunk_index": payload.get("chunk_index", -1),
                    "source_name": payload.get("source_name", ""),
                    "text": payload.get("text", ""),
                }
            )
        return results

    def list_all_docs(self) -> list[dict[str, Any]]:
        """Return one entry per unique doc_id: {doc_id, source_name, chunk_count}.

        Iterates the whole collection via scroll (cursor-based, memory-safe).
        """
        doc_info: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id", "")
                if not doc_id:
                    continue
                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        "doc_id": doc_id,
                        "source_name": payload.get("source_name", ""),
                        "chunk_count": 0,
                    }
                doc_info[doc_id]["chunk_count"] += 1
            if next_offset is None:
                break
            offset = next_offset

        return sorted(doc_info.values(), key=lambda d: d["source_name"])

    def delete_doc(self, doc_id: str) -> int:
        """Delete all chunks belonging to *doc_id*. Returns number of deleted points."""
        # First count how many we're deleting
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=10_000,
            with_payload=False,
            with_vectors=False,
        )
        count = len(points)
        if count == 0:
            return 0
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=query_filter,
        )
        return count
