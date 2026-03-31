from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from openai import OpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


def _build_ragas_llm(
    *,
    base_url: str,
    model: str,
    temperature: float,
    timeout_s: int,
) -> Any:
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
        timeout=timeout_s,
    )
    return llm_factory(
        model=model,
        client=client,
        temperature=temperature,
    )


def _build_ragas_embeddings(embedding_model_name: str) -> Any:
    return HuggingFaceEmbeddings(model=embedding_model_name)


def _create_generator(llm: Any, emb: Any) -> Any:
    from ragas.testset import TestsetGenerator
    try:
        return TestsetGenerator(llm=llm, embedding_model=emb)
    except Exception as exc:
        raise RuntimeError(
            "Ragas TestsetGenerator initialize edilemedi. Kurulu ragas surumu ile API uyumunu kontrol edin."
        ) from exc


def _generate_testset(generator: Any, docs: list[Any], test_size: int) -> Any:
    # Throttle parallelism to avoid overwhelming vLLM.
    run_config = RunConfig(max_workers=2, timeout=180)
    common_kwargs = {"raise_exceptions": False}
    last_exc: Exception | None = None
    methods = [
        (
            "generate_with_langchain_docs",
            {
                "documents": docs,
                "testset_size": test_size,
                "run_config": run_config,
                **common_kwargs,
            },
        ),
        (
            "generate",
            {
                "testset_size": test_size,
                "run_config": run_config,
                **common_kwargs,
            },
        ),
    ]

    for method_name, kwargs in methods:
        if not hasattr(generator, method_name):
            continue
        try:
            method = getattr(generator, method_name)
            return method(**kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning("Testset generation attempt failed (%s): %s", method_name, exc)

    if last_exc:
        raise RuntimeError(
            "Testset uretimi basarisiz oldu. Ragas API degisikligi olabilir."
        ) from last_exc
    raise RuntimeError("Testset uretimi basarisiz oldu. Ragas API degisikligi olabilir.")


def _extract_rows(testset_obj: Any) -> list[dict[str, Any]]:
    if hasattr(testset_obj, "to_pandas"):
        frame = testset_obj.to_pandas()
        rows = frame.to_dict(orient="records")
    elif isinstance(testset_obj, list):
        rows = testset_obj
    else:
        raise RuntimeError("Testset sonucu pandas'a cevrilemedi.")

    normalized: list[dict[str, Any]] = []
    for row in rows:
        question = row.get("question") or row.get("user_input") or ""
        ground_truth = row.get("ground_truth") or row.get("reference") or ""
        contexts = (
            row.get("contexts")
            or row.get("reference_contexts")
            or row.get("retrieved_contexts")
            or []
        )
        if isinstance(contexts, str):
            contexts = [contexts]

        normalized.append(
            {
                "question": str(question),
                "ground_truth": str(ground_truth),
                "contexts": [str(item) for item in contexts],
                "source": row.get("source") or row.get("document") or "",
            }
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF havuzundan Ragas sentetik testset uretir.")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("pdfs"),
        help="PDF klasoru",
    )
    parser.add_argument("--glob", type=str, default="**/*.pdf", help="PDF glob pattern")
    parser.add_argument("--test-size", type=int, default=60, help="Uretilecek soru sayisi")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ragas_testset.jsonl"),
        help="JSONL cikti yolu",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Ragas ic operasyonlarinda kullanilacak lokal embedding modeli",
    )
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8108/v1")
    parser.add_argument("--vllm-model", type=str, default="/model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.pdf_dir.exists():
        fallback_dir = Path("pdfs")
        if fallback_dir.exists():
            logger.warning(
                "PDF klasoru bulunamadi: %s. Fallback kullaniliyor: %s",
                args.pdf_dir,
                fallback_dir,
            )
            args.pdf_dir = fallback_dir
        else:
            raise FileNotFoundError(f"PDF klasoru bulunamadi: {args.pdf_dir}")

    loader = DirectoryLoader(
        str(args.pdf_dir),
        glob=args.glob,
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"PDF bulunamadi. Klasor={args.pdf_dir}, glob={args.glob}")

    generator_llm = _build_ragas_llm(
        base_url=args.vllm_base_url,
        model=args.vllm_model,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )
    embeddings = _build_ragas_embeddings(args.embedding_model)
    generator = _create_generator(generator_llm, embeddings)

    logger.info("Generating synthetic testset from %d docs (test_size=%d)", len(docs), args.test_size)
    testset = _generate_testset(generator, docs, args.test_size)
    rows = _extract_rows(testset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Testset kaydedildi: %s (row=%d)", args.output, len(rows))


if __name__ == "__main__":
    main()
