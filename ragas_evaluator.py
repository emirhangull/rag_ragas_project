"""
ragas_evaluator.py
──────────────────
İki ana işlev:

1. build_testset(docs, size)
   - Qdrant'tan veya doğrudan metin listesinden LangChain Document'a dönüştürür
   - TestsetGenerator ile soru-cevap çiftleri üretir
   - Sonucu JSON olarak kaydeder (testset_cache.json)

2. evaluate_pipeline(pipeline, testset_path)
   - Kaydedilmiş testset'i okur
   - Her soru için pipeline.ask() çağırır
   - Ragas metrikleri hesaplar: faithfulness, answer_relevancy, context_precision, context_recall
   - Sonuçları DataFrame olarak döner ve CSV'ye kaydeder
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ── Ragas ───────────────────────────────────────────────────────────────────
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms.default import default_transforms
from ragas import evaluate
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument

if TYPE_CHECKING:
    from .pipeline import RagPipeline

# ── Config ──────────────────────────────────────────────────────────────────
from .config import settings

DEFAULT_TESTSET_PATH = Path("testset_cache.json")
DEFAULT_EVAL_PATH = Path("eval_results.csv")


# ── LLM / Embedding kurulumu ─────────────────────────────────────────────────

def _build_ragas_llm():
    client = OpenAI(
        api_key="EMPTY",
        base_url=settings.vllm_base_url,
    )
    return llm_factory(settings.vllm_model, client=client)


def _build_ragas_embeddings():
    lc_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name
    )
    return LangchainEmbeddingsWrapper(lc_embeddings)


# ── Testset üretimi ──────────────────────────────────────────────────────────

def build_testset(
    texts: list[str],
    source_names: list[str] | None = None,
    testset_size: int = 10,
    save_path: Path | str = DEFAULT_TESTSET_PATH,
) -> list[dict]:
    """
    Verilen metin listesinden soru-cevap çiftleri üretir.

    Parameters
    ----------
    texts       : Ham metin listesi (her eleman bir doküman)
    source_names: Her metne karşılık kaynak adı (isteğe bağlı)
    testset_size: Üretilecek soru sayısı
    save_path   : Testset'in kaydedileceği JSON dosyası

    Returns
    -------
    list[dict]  : [{"question": ..., "ground_truth": ...}, ...]
    """
    save_path = Path(save_path)

    # LangChain Document formatına dönüştür
    documents: list[LCDocument] = []
    for idx, text in enumerate(texts):
        source = source_names[idx] if source_names and idx < len(source_names) else f"doc_{idx}"
        documents.append(LCDocument(page_content=text, metadata={"source": source}))

    logger.info("TestsetGenerator başlatılıyor: %d doküman, %d soru", len(documents), testset_size)

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    # KnowledgeGraph oluştur — önce dokümanları node olarak ekle
    from ragas.testset.graph import Node, NodeType
    kg = KnowledgeGraph()
    for doc in documents:
        kg.add(Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        ))

    logger.info("KnowledgeGraph'a %d node eklendi", len(kg.nodes))

    transforms = default_transforms(
        documents=documents,
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
    )

    from ragas.testset.transforms.engine import apply_transforms
    apply_transforms(kg, transforms)

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
        knowledge_graph=kg,
    )

    testset = generator.generate(testset_size=testset_size)
    testset_df = testset.to_pandas()

    logger.info("Testset üretildi: %d örnek", len(testset_df))

    # Kaydet
    records = []
    for _, row in testset_df.iterrows():
        records.append({
            "question": str(row.get("user_input", row.get("question", ""))),
            "ground_truth": str(row.get("reference", row.get("ground_truth", ""))),
        })

    save_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Testset kaydedildi: %s", save_path)

    return records


# ── Pipeline değerlendirmesi ─────────────────────────────────────────────────

def evaluate_pipeline(
    pipeline: "RagPipeline",
    testset_path: Path | str = DEFAULT_TESTSET_PATH,
    save_path: Path | str = DEFAULT_EVAL_PATH,
    doc_id: str | None = None,
) -> "import pandas; pandas.DataFrame":
    """
    Kaydedilmiş testset üzerinde pipeline'ı çalıştırır ve ragas metrikleri hesaplar.

    Parameters
    ----------
    pipeline    : Çalışır durumdaki RagPipeline instance'ı
    testset_path: build_testset() ile üretilen JSON dosyası
    save_path   : Sonuçların kaydedileceği CSV dosyası
    doc_id      : Belirli bir dokümanla sınırlamak istersen (None = tüm havuz)

    Returns
    -------
    pandas.DataFrame : Her satır bir soru, sütunlar metrik skorları
    """
    import pandas as pd

    testset_path = Path(testset_path)
    save_path = Path(save_path)

    if not testset_path.exists():
        raise FileNotFoundError(
            f"Testset bulunamadı: {testset_path}. "
            "Önce build_testset() çalıştırın."
        )

    records = json.loads(testset_path.read_text(encoding="utf-8"))
    logger.info("Testset yüklendi: %d soru", len(records))

    # Pipeline'ı her soru için çalıştır
    questions, answers, contexts, ground_truths = [], [], [], []

    for idx, record in enumerate(records):
        question = record["question"]
        ground_truth = record["ground_truth"]

        logger.info("[%d/%d] Soru: %s", idx + 1, len(records), question[:60])

        try:
            result = pipeline.ask(question, doc_id=doc_id)
            answer = result["answer"]
            context_texts = [chunk["text"] for chunk in result.get("retrieved", [])]
        except Exception as exc:
            logger.warning("Pipeline hatası soru %d için: %s", idx, exc)
            answer = ""
            context_texts = []

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

    # Ragas dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # Metrikleri ayarla
    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm

    logger.info("Ragas değerlendirmesi başlıyor...")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    df = result.to_pandas()

    # CSV'ye kaydet
    df.to_csv(save_path, index=False, encoding="utf-8")
    logger.info("Değerlendirme sonuçları kaydedildi: %s", save_path)

    # Özet yazdır
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available_cols = [c for c in metric_cols if c in df.columns]
    print("\n── Değerlendirme Özeti ──────────────────────")
    print(df[available_cols].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
    print("─────────────────────────────────────────────\n")

    return df