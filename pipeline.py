"""
Arşivciye der ki: \"Şirket karı ile ilgili olan 3-5 tane metin parçasını depodan getir.\"

Yazar'a (vLLM) der ki: \"Bak, kütüphaneden şu bilgileri buldum. Bu bilgilere dayanarak kullanıcının 'Şirket karı ne kadar?' sorusuna cevap ver.\"

Sana da sonucu paketleyip gönderir.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from .chunker import RecursiveTextChunker
from .config import Settings, settings
from .embedder import SentenceTransformerEmbedder
from .qdrant_index import QdrantIndexer
from .retriever import Retriever
from .vllm_client import NO_ANSWER_TEXT, VllmClient

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    doc_id: str
    source_name: str
    chunk_count: int


class RagPipeline:
    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self.chunker = RecursiveTextChunker(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        self.embedder = SentenceTransformerEmbedder(cfg.embedding_model_name)
        self.indexer = QdrantIndexer(
            host=cfg.qdrant_host,
            port=cfg.qdrant_port,
            collection_name=cfg.qdrant_collection,
            vector_size=self.embedder.dimension,
            path=cfg.qdrant_path or None,
        )
        self.retriever = Retriever(
            embedder=self.embedder,
            indexer=self.indexer,
            top_k=cfg.retrieval_top_k,
            fetch_k=cfg.retrieval_fetch_k,
            rerank_model=cfg.rerank_model,
            use_mmr=cfg.use_mmr,
            mmr_lambda=cfg.mmr_lambda,
            lexical_global_limit=cfg.lexical_global_limit,
        )
        self.vllm = VllmClient(
            base_url=cfg.vllm_base_url,
            model=cfg.vllm_model,
            temperature=cfg.vllm_temperature,
            max_tokens=cfg.vllm_max_tokens,
            timeout_s=cfg.vllm_timeout_s,
        )
        logger.info(
            "RagPipeline created: embed=%s, qdrant=%s, vllm=%s, mmr=%s, rerank_model=%r",
            cfg.embedding_model_name, cfg.qdrant_collection, cfg.vllm_base_url,
            cfg.use_mmr, cfg.rerank_model or "disabled",
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_text(self, text: str, source_name: str) -> IngestResult:
        doc_id = str(uuid.uuid4())
        chunk_objs = self.chunker.split_text(text)
        chunk_texts = [item.text for item in chunk_objs]
        logger.info("Chunked '%s': %d chunks from %d chars", source_name, len(chunk_texts), len(text))

        vectors = self.embedder.embed_texts(chunk_texts)
        count = self.indexer.upsert_chunks(
            doc_id=doc_id,
            chunks=chunk_texts,
            vectors=vectors,
            source_name=source_name,
        )
        logger.info("Indexed '%s': doc_id=%s, %d chunks upserted", source_name, doc_id, count)
        return IngestResult(doc_id=doc_id, source_name=source_name, chunk_count=count)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def ask(self, question: str, doc_id: str | None = None) -> dict:
        scope = f"doc_id={doc_id}" if doc_id else "GLOBAL_POOL"
        logger.info("Pipeline sorgusu başlatıldı. Kapsam: %s", scope)

        hits = self.retriever.search(question, doc_id=doc_id)

        context_chunks = [f"[Kaynak: {item.source_name}] {item.text}" for item in hits]

        answer = self.vllm.answer_with_context(question, context_chunks)
        if answer == NO_ANSWER_TEXT:
            logger.info("First pass returned no-answer, starting fallback retrieval...")
            expanded_questions = self.retriever.expand_query(question)
            merged_hits = list(hits)
            seen_chunk_ids = {item.chunk_id for item in merged_hits}

            fallback_top_k = max(self.cfg.retrieval_top_k, 12)
            fallback_fetch_k = max(self.cfg.retrieval_fetch_k, 48)

            for rewritten_question in expanded_questions[1:]:
                extra_hits = self.retriever.search(
                    rewritten_question,
                    doc_id=doc_id,
                    top_k=fallback_top_k,
                    fetch_k=fallback_fetch_k,
                )
                for item in extra_hits:
                    if item.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(item.chunk_id)
                    merged_hits.append(item)

            lexical_hits = self.retriever.search_lexical(
                question,
                doc_id=doc_id,
                top_k=fallback_top_k,
                limit=400,
            )
            for item in lexical_hits:
                if item.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(item.chunk_id)
                merged_hits.append(item)

            merged_hits.sort(key=lambda item: item.rerank_score, reverse=True)
            merged_hits = merged_hits[:fallback_top_k]

            logger.info(
                "Fallback retrieval: %d expanded queries, %d merged chunks",
                len(expanded_questions) - 1, len(merged_hits),
            )

            retry_context = [item.text for item in merged_hits]
            retry_answer = self.vllm.answer_with_context(question, retry_context)
            if retry_answer != NO_ANSWER_TEXT:
                answer = retry_answer
                hits = merged_hits
                logger.info("Fallback succeeded – answer found on retry.")
            else:
                logger.warning("Fallback also returned no-answer for: %s", question[:80])

        return {
            "answer": answer,
            "retrieved": [
                {
                    "score": item.score,
                    "rerank_score": item.rerank_score,
                    "doc_id": item.doc_id,
                    "chunk_id": item.chunk_id,
                    "source_name": item.source_name,
                    "text": item.text,
                }
                for item in hits
            ],
        }

    # ------------------------------------------------------------------
    # Ragas Testset Üretimi
    # ------------------------------------------------------------------

    def build_ragas_testset(
        self,
        testset_size: int = 10,
        doc_id: str | None = None,
        save_path: str = "testset_cache.json",
    ) -> list[dict]:
        """
        İndekslenmiş dokümanlardan otomatik soru-cevap testset'i üretir.

        Parameters
        ----------
        testset_size : Üretilecek soru sayısı
        doc_id       : Belirli bir dokümanla sınırla (None = tüm havuz)
        save_path    : Üretilen testset'in kaydedileceği JSON dosyası

        Returns
        -------
        list[dict] : [{"question": ..., "ground_truth": ...}, ...]
        """
        from .ragas_evaluator import build_testset

        # Qdrant'tan chunk'ları çek
        if doc_id:
            rows = self.indexer.list_chunks(doc_id=doc_id, limit=500)
        else:
            rows = self.indexer.scroll_all(limit=2000)

        if not rows:
            raise ValueError("Testset üretmek için önce doküman yükleyin.")

        texts = [row["text"] for row in rows]
        source_names = [row["source_name"] for row in rows]

        logger.info(
            "Testset üretimi: %d chunk, %d soru hedefleniyor",
            len(texts), testset_size,
        )

        return build_testset(
            texts=texts,
            source_names=source_names,
            testset_size=testset_size,
            save_path=save_path,
        )

    # ------------------------------------------------------------------
    # Ragas Değerlendirmesi
    # ------------------------------------------------------------------

    def evaluate_with_ragas(
        self,
        testset_path: str = "testset_cache.json",
        save_path: str = "eval_results.csv",
        doc_id: str | None = None,
    ):
        """
        Kaydedilmiş testset üzerinde pipeline'ı değerlendirir.

        Parameters
        ----------
        testset_path : build_ragas_testset() ile üretilen JSON
        save_path    : Sonuçların kaydedileceği CSV
        doc_id       : Belirli bir dokümanla sınırla (None = tüm havuz)

        Returns
        -------
        pandas.DataFrame : Metrik sonuçları
        """
        from .ragas_evaluator import evaluate_pipeline

        return evaluate_pipeline(
            pipeline=self,
            testset_path=testset_path,
            save_path=save_path,
            doc_id=doc_id,
        )

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def list_doc_chunks(self, doc_id: str, limit: int = 200) -> list[dict]:
        return self.indexer.list_chunks(doc_id=doc_id, limit=limit)

    def list_documents(self) -> list[dict]:
        """Return all indexed documents: [{doc_id, source_name, chunk_count}]."""
        return self.indexer.list_all_docs()

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks of a document. Returns number of deleted chunks."""
        deleted = self.indexer.delete_doc(doc_id)
        logger.info("Deleted doc_id=%s (%d chunks)", doc_id, deleted)
        return deleted

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_status(self) -> dict:
        return {
            "qdrant_ready": self.indexer.is_ready(),
            "vllm_ready": self.vllm.is_ready(),
            "embedding_dim": self.embedder.dimension,
        }
