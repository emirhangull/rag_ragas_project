from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset
import pandas as pd
from openai import OpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory

try:
    from rag_mvp import RagPipeline
except Exception:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    workspace_parent = repo_root.parent
    if str(workspace_parent) not in sys.path:
        sys.path.insert(0, str(workspace_parent))
    from rag_mvp import RagPipeline

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_ragas_clients(
    *,
    base_url: str,
    model: str,
    timeout_s: int,
    temperature: float,
    embedding_model_name: str,
) -> tuple[Any, Any]:
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
        timeout=timeout_s,
    )
    llm = llm_factory(
        model=model,
        client=client,
        temperature=temperature,
    )
    embeddings = HuggingFaceEmbeddings(model=embedding_model_name)
    return llm, embeddings


def _resolve_metrics() -> list[Any]:
    try:
        from ragas.metrics import collections as ragas_metrics
    except Exception:
        from ragas import metrics as ragas_metrics

    names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_precision_with_reference",
        "llm_context_precision_with_reference",
    ]

    selected: list[Any] = []
    for name in names:
        metric = getattr(ragas_metrics, name, None)
        if metric is not None:
            selected.append(metric)

    # Keep unique by metric name while preserving order.
    uniq: list[Any] = []
    seen: set[str] = set()
    for metric in selected:
        metric_name = getattr(metric, "name", None) or getattr(metric, "__name__", str(metric))
        if metric_name in seen:
            continue
        seen.add(metric_name)
        uniq.append(metric)

    if not uniq:
        raise RuntimeError("Ragas metric import basarisiz. Kurulu ragas surumunu kontrol edin.")
    return uniq


def _attach_runtime(metric: Any, llm_wrapper: Any, emb_wrapper: Any) -> Any:
    try:
        metric_llm = getattr(metric, "llm", None)
        if metric_llm is None and hasattr(metric, "llm"):
            metric.llm = llm_wrapper
    except Exception:
        pass

    try:
        metric_emb = getattr(metric, "embeddings", None)
        if metric_emb is None and hasattr(metric, "embeddings"):
            metric.embeddings = emb_wrapper
    except Exception:
        pass

    return metric


def _evaluate_dataset(dataset: Dataset, metrics: list[Any], llm_wrapper: Any, emb_wrapper: Any) -> dict[str, Any]:
    from ragas import evaluate

    prepared_metrics = [_attach_runtime(metric, llm_wrapper, emb_wrapper) for metric in metrics]

    result = evaluate(
        dataset=dataset,
        metrics=prepared_metrics,
        llm=llm_wrapper,
        embeddings=emb_wrapper,
    )

    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        averages = {}
        for col in frame.columns:
            if col in {
                "question",
                "answer",
                "contexts",
                "ground_truth",
                "user_input",
                "response",
                "retrieved_contexts",
                "reference",
            }:
                continue
            try:
                averages[col] = float(frame[col].mean())
            except Exception:
                continue
        return {"averages": averages, "rows": frame.to_dict(orient="records")}

    if hasattr(result, "__dict__"):
        return result.__dict__

    return {"result": str(result)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RagPipeline ciktilari icin Ragas degerlendirme")
    parser.add_argument(
        "--testset",
        type=Path,
        required=True,
        help="generate_testset.py tarafindan uretilen JSONL testset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation_results.json"),
        help="Degerlendirme sonucu JSON",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Ragas metric embedding modeli",
    )
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8108/v1")
    parser.add_argument("--vllm-model", type=str, default="/model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-samples", type=int, default=0, help="0 ise tum satirlar")
    parser.add_argument("--doc-id", type=str, default="", help="Opsiyonel sabit doc_id scope")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.testset.exists():
        raise FileNotFoundError(f"Testset dosyasi bulunamadi: {args.testset}")

    rows = _read_jsonl(args.testset)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("Bos testset")

    rag = RagPipeline()

    user_inputs: list[str] = []
    responses: list[str] = []
    retrieved_contexts_all: list[list[str]] = []
    references: list[str] = []

    static_doc_id = args.doc_id.strip() or None

    for idx, row in enumerate(rows, start=1):
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        row_doc_id = str(row.get("doc_id", "")).strip() or None
        effective_doc_id = static_doc_id or row_doc_id
        response = rag.ask(question, doc_id=effective_doc_id)

        answer = str(response.get("answer", ""))
        retrieved = response.get("retrieved", [])
        retrieved_contexts = [str(item.get("text", "")) for item in retrieved if item.get("text")]

        reference = str(row.get("ground_truth") or row.get("reference") or "")

        user_inputs.append(question)
        responses.append(answer)
        retrieved_contexts_all.append(retrieved_contexts)
        references.append(reference)

        if idx % 5 == 0:
            logger.info("Processed %d/%d questions", idx, len(rows))

    eval_dataset = Dataset.from_dict(
        {
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": retrieved_contexts_all,
            "reference": references,
            # Legacy aliases for older ragas/table readability.
            "question": user_inputs,
            "answer": responses,
            "contexts": retrieved_contexts_all,
            "ground_truth": references,
        }
    )

    llm_wrapper, emb_wrapper = _build_ragas_clients(
        base_url=args.vllm_base_url,
        model=args.vllm_model,
        timeout_s=args.timeout_s,
        temperature=args.temperature,
        embedding_model_name=args.embedding_model,
    )
    metrics = _resolve_metrics()

    logger.info("Evaluating %d rows with %d metrics", len(user_inputs), len(metrics))
    result_payload = _evaluate_dataset(eval_dataset, metrics, llm_wrapper, emb_wrapper)

    rows = result_payload.get("rows") if isinstance(result_payload, dict) else None
    if isinstance(rows, list) and rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame([result_payload.get("averages", {})])
    print(df.to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    logger.info("Evaluation yazildi: %s", args.output)


if __name__ == "__main__":
    main()
