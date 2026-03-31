from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .embedder import SentenceTransformerEmbedder
from .qdrant_index import QdrantIndexer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    rerank_score: float
    doc_id: str
    chunk_id: str
    source_name: str


class Retriever:
    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        indexer: QdrantIndexer,
        top_k: int = 5,
        fetch_k: int = 20,
        rerank_model: str = "",
        use_mmr: bool = False,
        mmr_lambda: float = 0.6,
        lexical_global_limit: int = 2000,
    ) -> None:
        self.embedder = embedder
        self.indexer = indexer
        self.top_k = top_k
        self.fetch_k = max(fetch_k, top_k)
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.lexical_global_limit = lexical_global_limit

        # Optional cross-encoder reranker
        self._cross_encoder = None
        if rerank_model:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(rerank_model)
                logger.info("Cross-encoder loaded: %s", rerank_model)
            except Exception as exc:
                logger.warning("CrossEncoder load failed (%s) – falling back to hybrid rerank", exc)

    # ------------------------------------------------------------------
    # Dynamic limit tuning
    # ------------------------------------------------------------------

    def _dynamic_limits(self, question: str) -> tuple[int, int]:
        tokens = self._tokenize(question)
        token_count = len(tokens)
        lowered = question.lower()
        ambiguity_markers = {"nedir", "neler", "hangi", "kaç", "kac"}
        has_ambiguous_pattern = any(marker in lowered for marker in ambiguity_markers)

        top_k = self.top_k
        fetch_k = self.fetch_k
        if token_count <= 6 or has_ambiguous_pattern:
            top_k = max(self.top_k, 10)
            fetch_k = max(self.fetch_k, 40)
        return top_k, fetch_k

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def expand_query(self, question: str) -> list[str]:
        """Generate query variants for broader retrieval coverage."""
        base = " ".join(question.split())
        variants: list[str] = [base]
        variants.append(f"{base} Cevabı açıkça yaz.")
        variants.append(f"{base} Maddeler halinde listele.")

        keyword_query = self._strip_question_words(base)
        if keyword_query and keyword_query.lower() != base.lower():
            variants.append(keyword_query)

        unique_variants: list[str] = []
        seen: set[str] = set()
        for item in variants:
            normalized = item.strip()
            if not normalized:
                continue
            lowered_item = normalized.lower()
            if lowered_item in seen:
                continue
            seen.add(lowered_item)
            unique_variants.append(normalized)
        return unique_variants

    @staticmethod
    def _strip_question_words(question: str) -> str:
        """Remove common Turkish question particles to create a keyword-focused query."""
        question_words = {
            "nedir", "nelerdir", "neler", "hangi", "hangileri", "hangisi",
            "kaç", "kac", "nasıl", "nasil", "neden", "niye", "kim",
            "söylenen", "soylenen", "geçen", "gecen", "kullanılan", "kullanilan",
            "raporda", "belgede", "dokümanda", "dokumanda",
        }
        tokens = question.split()
        filtered = [t for t in tokens if t.lower().rstrip("?.,!") not in question_words]
        return " ".join(filtered).strip()

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        lowered = value.lower().replace("ı", "i")
        tokens = re.findall(r"[a-zçğıöşü0-9]+", lowered)
        stopwords = {
            "ve", "ile", "bu", "bir", "için", "icin", "olan", "olarak",
            "ne", "nedir", "hangi", "kaç", "kac", "kullanilan", "kullanılan",
        }
        return {token for token in tokens if len(token) > 1 and token not in stopwords}

    # ------------------------------------------------------------------
    # Hybrid rerank (dense + lexical overlap)
    # ------------------------------------------------------------------

    def _rerank(self, question: str, rows: list[dict], top_k: int | None = None) -> list[dict]:
        """Re-rank candidates using cross-encoder (if available) or dense+lexical hybrid."""
        effective_top_k = top_k if top_k is not None else self.top_k

        # --- Cross-encoder path ---
        if self._cross_encoder is not None and rows:
            try:
                pairs = [(question, row.get("text", "")) for row in rows]
                ce_scores = self._cross_encoder.predict(pairs)
                for row, score in zip(rows, ce_scores):
                    row["rerank_score"] = float(score)
                rows.sort(key=lambda r: r["rerank_score"], reverse=True)
                return rows[:effective_top_k]
            except Exception as exc:
                logger.warning("CrossEncoder predict failed: %s – using hybrid fallback", exc)

        # --- Hybrid (dense + lexical overlap) path ---
        query_tokens = self._tokenize(question)
        if not query_tokens:
            for row in rows:
                row["rerank_score"] = float(row.get("score", 0.0))
            return rows[:effective_top_k]

        ranked: list[tuple[float, dict]] = []
        for row in rows:
            text = row.get("text", "")
            text_tokens = self._tokenize(text)
            overlap = len(query_tokens & text_tokens)
            overlap_ratio = overlap / max(1, len(query_tokens))
            dense_score = float(row.get("score", 0.0))
            hybrid_score = (dense_score * 0.75) + (overlap_ratio * 0.25)
            row["rerank_score"] = hybrid_score
            ranked.append((hybrid_score, row))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:effective_top_k]]

    # ------------------------------------------------------------------
    # MMR – Maximal Marginal Relevance
    # ------------------------------------------------------------------

    def _mmr_select(
        self,
        candidates: list[dict],
        query_vector: list[float],
        top_k: int,
        lambda_param: float = 0.6,
    ) -> list[dict]:
        """Select *top_k* chunks using MMR to balance relevance and diversity.

        Args:
            candidates: Reranked candidate rows (must have 'text' and 'rerank_score').
            query_vector: Embedded query vector (used as relevance proxy).
            top_k: Number of results to return.
            lambda_param: 1.0 = pure relevance, 0.0 = pure diversity.
        """
        if len(candidates) <= top_k:
            return candidates

        # Embed candidate texts
        texts = [row.get("text", "") for row in candidates]
        try:
            doc_vectors = self.embedder.embed_texts(texts)
        except Exception as exc:
            logger.warning("MMR embedding failed: %s – skipping MMR", exc)
            return candidates[:top_k]

        q_vec = np.array(query_vector, dtype=np.float32)
        d_vecs = np.array(doc_vectors, dtype=np.float32)

        # Relevance scores: cosine similarity with query (vectors already normalised)
        relevance = d_vecs @ q_vec  # shape (N,)

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break
            if not selected_indices:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: relevance[i])
            else:
                sel_vecs = d_vecs[selected_indices]  # (k, dim)
                # Max similarity to any already-selected doc
                sim_to_sel = (d_vecs[remaining] @ sel_vecs.T).max(axis=1)  # (rem,)
                rem_arr = np.array(remaining)
                mmr_scores = (
                    lambda_param * relevance[rem_arr]
                    - (1 - lambda_param) * sim_to_sel
                )
                best_local = int(np.argmax(mmr_scores))
                best = remaining[best_local]

            selected_indices.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected_indices]

    # ------------------------------------------------------------------
    # Primary search (dense vector → rerank → optional MMR)
    # ------------------------------------------------------------------

    def search(
        self,
        question: str,
        doc_id: str | None = None,
        top_k: int | None = None,
        fetch_k: int | None = None,
        use_mmr: bool | None = None,
    ) -> list[RetrievedChunk]:
        auto_top_k, auto_fetch_k = self._dynamic_limits(question)
        effective_top_k = top_k if top_k is not None else auto_top_k
        effective_fetch_k = fetch_k if fetch_k is not None else max(auto_fetch_k, effective_top_k)
        apply_mmr = use_mmr if use_mmr is not None else self.use_mmr

        query_vector = self.embedder.embed_query(question)
        rows = self.indexer.retrieve(query_vector, top_k=effective_fetch_k, doc_id=doc_id)
        rows = self._rerank(question, rows, top_k=effective_fetch_k)

        if apply_mmr and len(rows) > effective_top_k:
            rows = self._mmr_select(rows, query_vector, top_k=effective_top_k, lambda_param=self.mmr_lambda)
        else:
            rows = rows[:effective_top_k]

        logger.debug(
            "search: question=%s top_k=%d fetch_k=%d mmr=%s results=%d",
            question[:60], effective_top_k, effective_fetch_k, apply_mmr, len(rows),
        )

        return [
            RetrievedChunk(
                text=row["text"],
                score=row["score"],
                rerank_score=float(row.get("rerank_score", row.get("score", 0.0))),
                doc_id=row["doc_id"],
                chunk_id=row["chunk_id"],
                source_name=row["source_name"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Lexical search (global multi-doc or per-doc)
    # ------------------------------------------------------------------

    def search_lexical(
        self,
        question: str,
        doc_id: str | None,
        top_k: int = 12,
        limit: int = 400,
    ) -> list[RetrievedChunk]:
        """BM25-style lexical search.

        When *doc_id* is None, searches the entire collection (multi-doc mode)
        using ``indexer.scroll_all()``.  When *doc_id* is given, searches only
        that document's chunks.
        """
        if doc_id:
            rows = self.indexer.list_chunks(doc_id=doc_id, limit=limit)
        else:
            # Global search across all documents
            rows = self.indexer.scroll_all(limit=self.lexical_global_limit)

        query_tokens = self._tokenize(question)
        if not query_tokens or not rows:
            return []

        ranked: list[tuple[float, dict]] = []
        for row in rows:
            text = row.get("text", "")
            text_tokens = self._tokenize(text)
            overlap = len(query_tokens & text_tokens)
            if overlap == 0:
                continue

            overlap_ratio = overlap / max(1, len(query_tokens))
            lexical_density = overlap / max(1, len(text_tokens))

            toc_penalty = 0.0
            if text.count(". . .") >= 3:
                toc_penalty += 0.15
            if text.count("Tablo") >= 2 and text.count(".") >= 20:
                toc_penalty += 0.05

            lexical_score = (overlap_ratio * 0.75) + (lexical_density * 0.25) - toc_penalty
            row["score"] = float(lexical_score)
            row["rerank_score"] = float(lexical_score)
            ranked.append((lexical_score, row))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [item[1] for item in ranked[:top_k]]

        logger.debug("search_lexical: question=%s doc_id=%s results=%d", question[:60], doc_id, len(selected))

        return [
            RetrievedChunk(
                text=row["text"],
                score=float(row.get("score", 0.0)),
                rerank_score=float(row.get("rerank_score", 0.0)),
                doc_id=row.get("doc_id", ""),
                chunk_id=row.get("chunk_id", ""),
                source_name=row.get("source_name", ""),
            )
            for row in selected
        ]
