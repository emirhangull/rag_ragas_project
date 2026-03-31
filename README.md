# RAG MVP (Upload + Qdrant + vLLM)

Bu klasör, uzun dosya yükleyip soru-cevap yapan minimal ve modüler bir RAG pipeline içerir.

## Modüller

- `file_loader.py`: dosya yükleme ve metin çıkarma
- `chunker.py`: recursive chunking + overlap
- `embedder.py`: embedding üretimi
- `qdrant_index.py`: Qdrant indexleme + retrieval
- `retriever.py`: retrieval orchestration
- `vllm_client.py`: localhost:8108 üzerindeki vLLM `/model` çağrısı
- `pipeline.py`: tüm akışı birleştiren servis
- `app.py`: FastAPI endpointleri (`/upload`, `/ask`)

## Kurulum

```bash
cd /path/to/rag_mvp
pip install -r requirements.txt
```

Qdrant ayakta olmalı (`localhost:6333`).

Docker ile Qdrant başlatmak için (önerilen):

```bash
cd /path/to/rag_mvp
docker compose up -d qdrant
```

Eğer `unauthenticated pull rate limit` hatası alırsan önce Docker Hub login yap:

```bash
docker login
docker compose up -d qdrant
```

Durum kontrolü:

```bash
curl -s http://localhost:6333/healthz
```

Servisi durdurmak için:

```bash
cd /path/to/rag_mvp
docker compose stop qdrant
```

Docker/servis izni yoksa local embedded mod kullanabilirsin:

```bash
export QDRANT_PATH="$PWD/.qdrant_local"
```

Bu modda ayrı Qdrant servisi gerekmez.

vLLM OpenAI-compatible endpoint ayakta olmalı (`http://localhost:8108/v1`).

## Çalıştırma

```bash
cd /path/to/rag_mvp
uvicorn rag_mvp.app:app --host 0.0.0.0 --port 8099 --reload
```

Local embedded Qdrant ile çalıştırmak için:

```bash
cd /path/to/rag_mvp
export QDRANT_PATH="$PWD/.qdrant_local"
uvicorn rag_mvp.app:app --host 0.0.0.0 --port 8099 --reload
```

## Kullanım

Dosya yükleme:
```bash
curl -X POST "http://localhost:8099/upload" \
  -F "file=@/path/to/long_document.txt"
```

Soru sorma:
```bash
curl -X POST "http://localhost:8099/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Dokümanda ana öneri nedir?"}'
```

## Ortam Değişkenleri
- `QDRANT_HOST` (default: `localhost`)
- `QDRANT_PORT` (default: `6333`)
- `QDRANT_COLLECTION` (default: `rag_mvp_chunks`)
- `QDRANT_PATH` (default: `""` – eğer tanımlıysa local embedded mod kullanılır)
- `EMBED_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `CHUNK_SIZE` (default: `900`)
- `CHUNK_OVERLAP` (default: `150`)
- `RETRIEVAL_TOP_K` (default: `8`)
- `VLLM_BASE_URL` (default: `http://localhost:8108/v1`)
- `VLLM_MODEL` (default: `/model`)
- `VLLM_TEMPERATURE` (default: `0.0`)
- `VLLM_MAX_TOKENS` (default: `4096`)
- `VLLM_TIMEOUT_S` (default: `120`)

## Son Güncelleme (P1)

Bu sürümde no-answer durumlarını azaltmak için retrieval ve cevap üretim tarafında üç iyileştirme eklendi:

- Dinamik retrieval limitleri: kısa/belirsiz sorularda daha geniş aday havuzu (`top_k`/`fetch_k`) kullanılır.
- Query rewrite fallback: ilk cevap no-answer olursa soru otomatik varyantlarla yeniden aranır.
- vLLM ikinci deneme: ilk cevap no-answer olursa, yine sadece bağlamı kullanan daha esnek bir ikinci prompt ile tekrar denenir.

Örnek olarak daha önce zorlanan şu tarz soru artık daha tutarlı cevaplanır:

```text
"raporda kullanıldığı söylenen 3 model nedir"
```

Beklenen cevap örüntüsü:

```text
Longformer, BigBird, ModernBERT
```

## Sorun Giderme

- `/health` içinde `qdrant_ready=false` görüyorsan local mode için `QDRANT_PATH` tanımla.
- `vllm_ready=false` ise `VLLM_BASE_URL` endpointinin ayakta olduğunu kontrol et (`/models` yanıt vermeli).
- Yanıt sürekli no-answer dönüyorsa soruyu daha açık yaz (örn. "raporda geçen model adlarını listele") veya `doc_id` göndererek belgeyi sabitle.

Sağlık kontrolü:

```bash
curl -s "http://localhost:8099/health"
```

Yanıtta `dependencies.qdrant_ready` ve `dependencies.vllm_ready` alanları görünür.

## Testler

```bash
cd /path/to/rag_mvp
pip install pytest
python -m pytest tests/ -v
```

Testler dış bağımlılık (Qdrant, vLLM) gerektirmez; yalnızca pure logic testleri içerir.


## Ragas ile Otomatik Değerlendirme

Bu repoda iki yeni script ile sentetik testset uretimi ve otomatik skor hesaplama akisi var:

- `ragas/generate_testset.py`: PDF havuzundan Ragas sentetik soru/ground-truth/context seti uretir.
- `ragas/evaluate_rag.py`: `RagPipeline.ask()` ciktilarini toplayip Ragas metriklerini hesaplar.

### 1) Testset Uret

```bash
cd /path/to/rag_mvp
python ragas/generate_testset.py \
  --pdf-dir pdfs \
  --glob "**/*.pdf" \
  --test-size 80 \
  --output outputs/ragas_testset.jsonl \
  --vllm-base-url http://localhost:8108/v1 \
  --vllm-model /model
```

Ornek JSONL satiri:

```json
{"question":"...","ground_truth":"...","contexts":["..."],"source":"..."}
```

### 2) Pipeline Uzerinden Degerlendir

```bash
cd /path/to/rag_mvp
python ragas/evaluate_rag.py \
  --testset outputs/ragas_testset.jsonl \
  --output outputs/ragas_eval.json \
  --vllm-base-url http://localhost:8108/v1 \
  --vllm-model /model
```

Hesaplanan metrikler (ragas surumune gore mevcut olanlar):

- Faithfulness
- Answer Relevancy
- Context Precision (veya context_precision_with_reference varyanti)

### Notlar

- vLLM modeli hem generator/critic hem de metric LLM ihtiyacinda kullanilir.
- Embedding islemleri lokal HuggingFace modeli ile yapilir (`--embedding-model`).
- `ragas/evaluate_rag.py`, dogrudan `RagPipeline.ask()` donusundeki `retrieved[].text` alanlarini context olarak kullanir; boylece chunking/rerank zayifliklari skora yansir.


## vllm call:
ssh -N -L 8108:127.0.0.1:8108 stj@10.150.96.44