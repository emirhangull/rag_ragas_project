"""
run_evaluation.py
─────────────────
Ragas değerlendirmesini çalıştırmak için örnek script.

Kullanım:
    # Tüm havuz üzerinde testset üret ve değerlendir
    python3 run_evaluation.py

    # Belirli bir doküman üzerinde
    python3 run_evaluation.py --doc_id <doc_id>

    # Sadece testset üret (değerlendirme yapma)
    python3 run_evaluation.py --only-build

    # Mevcut testset ile sadece değerlendir (tekrar üretme)
    python3 run_evaluation.py --only-eval
"""

import argparse
import logging
from rag_mvp.pipeline import RagPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_id", default=None, help="Belirli bir dokümanla sınırla")
    parser.add_argument("--size", type=int, default=10, help="Testset soru sayısı")
    parser.add_argument("--testset", default="testset_cache.json", help="Testset JSON yolu")
    parser.add_argument("--output", default="eval_results.csv", help="Sonuç CSV yolu")
    parser.add_argument("--only-build", action="store_true", help="Sadece testset üret")
    parser.add_argument("--only-eval", action="store_true", help="Sadece değerlendirme yap")
    args = parser.parse_args()

    pipeline = RagPipeline()

    # ── Testset üretimi ──────────────────────────────────────────────
    if not args.only_eval:
        print(f"\n[1/2] Testset üretiliyor ({args.size} soru)...")
        records = pipeline.build_ragas_testset(
            testset_size=args.size,
            doc_id=args.doc_id,
            save_path=args.testset,
        )
        print(f"      {len(records)} soru üretildi → {args.testset}")
        for i, r in enumerate(records[:3]):
            print(f"      Örnek {i+1}: {r['question'][:80]}")

    if args.only_build:
        print("\nTamam. Değerlendirme için: python3 run_evaluation.py --only-eval")
        return

    # ── Değerlendirme ────────────────────────────────────────────────
    print(f"\n[2/2] Ragas değerlendirmesi çalışıyor...")
    df = pipeline.evaluate_with_ragas(
        testset_path=args.testset,
        save_path=args.output,
        doc_id=args.doc_id,
    )

    print(f"\nSonuçlar → {args.output}")
    print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].to_string())


if __name__ == "__main__":
    main()
