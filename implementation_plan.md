# Modüler RAGAS API Test Aracı

Mevcut RAGAS test yapısı ([NisanRagasTest/deneme.py](file:///home/emirhan/Desktop/rag_mvp/NisanRagasTest/deneme.py)) doğrudan yerel [RagPipeline](file:///home/emirhan/Desktop/rag_mvp/pipeline.py#32-277)'a bağlı. Bu değişiklikle, **herhangi bir RAG API endpoint'ini** test edebilen bağımsız bir araç haline getiriyoruz.

## Mimari

```
NisanRagasTest/
├── config.yaml          # API endpoint, LLM judge ayarları, test dosya yolu
├── test_questions.json  # Soru + ground_truth çiftleri (harici dosya)
├── api_client.py        # Herhangi bir RAG API'sine HTTP isteği atar
├── ragas_runner.py      # API yanıtlarını RAGAS formatına çevirir ve değerlendirir
└── main.py              # CLI giriş noktası: config oku → API'yi çağır → RAGAS puanla
```

## Akış

```
config.yaml → main.py → api_client.py → Harici RAG API
                 ↓
         ragas_runner.py → RAGAS puanları → CSV/terminal çıktısı
```

## Proposed Changes

### NisanRagasTest

---

#### [NEW] [config.yaml](file:///home/emirhan/Desktop/rag_mvp/NisanRagasTest/config.yaml)

Tüm ayarlar tek yerde:

```yaml
# ── Hedef RAG API ──────────────────────────────
api:
  base_url: "http://localhost:8000"    # Test edilecek RAG servisinin adresi
  ask_endpoint: "/ask"                 # Soru sorma endpoint'i
  method: "POST"
  # Request body şablonu: {question} placeholder'ı yerine soru yazılır
  request_template:
    question: "{question}"
  # Response'dan answer ve context'leri çıkarmak için JSON path'leri
  response_mapping:
    answer_field: "answer"                        # Cevap alanı
    contexts_field: "retrieved"                   # Context listesi alanı
    context_text_field: "text"                    # Her context içindeki metin alanı

# ── Ragas Judge LLM ───────────────────────────
judge_llm:
  base_url: "http://localhost:8108/v1"
  model: "/model"
  api_key: "EMPTY"

# ── Test Soruları ──────────────────────────────
test_questions_file: "test_questions.json"

# ── Çıktı ──────────────────────────────────────
output:
  csv_path: "ragas_results.csv"
  json_path: "ragas_results.json"
```

---

#### [NEW] [test_questions.json](file:///home/emirhan/Desktop/rag_mvp/NisanRagasTest/test_questions.json)

Soru ve beklenen cevap çiftleri (config'den bağımsız dosya):

```json
[
  {
    "question": "Feride neden İstanbul'dan kaçtı?",
    "ground_truth": "Nişanlısı Kamran'ın onu aldattığını öğrendiği için İstanbul'dan kaçtı."
  }
]
```

---

#### [NEW] [api_client.py](file:///home/emirhan/Desktop/rag_mvp/NisanRagasTest/api_client.py)

- `RagApiClient` sınıfı: config.yaml'daki ayarlarla herhangi bir RAG API'sine istek atar
- [ask(question: str) → dict](file:///home/emirhan/Desktop/rag_mvp/pipeline.py#94-168) metodu: `{"answer": ..., "contexts": [...]}`  döner
- Response mapping ile farklı API formatlarını destekler
- Timeout, retry, hata yönetimi

---

#### [NEW] [ragas_runner.py](file:///home/emirhan/Desktop/rag_mvp/NisanRagasTest/ragas_runner.py)

- API yanıtlarını RAGAS Dataset formatına çevirir
- Judge LLM'i config'den oluşturur
- `faithfulness`, `context_precision`, `context_recall` metriklerini hesaplar
- Sonuçları CSV ve JSON'a kaydeder

---

#### [NEW] [main.py](file:///home/emirhan/Desktop/rag_mvp/NisanRagasTest/main.py)

CLI giriş noktası:

```bash
# Varsayılan config ile çalıştır
python -m NisanRagasTest.main

# Farklı config dosyası
python -m NisanRagasTest.main --config my_config.yaml

# Farklı soru dosyası (config'i override eder)
python -m NisanRagasTest.main --questions custom_questions.json
```

---

#### [MODIFY] [deneme.py](file:///home/emirhan/Desktop/rag_mvp/NisanRagasTest/deneme.py)

Dokunulmayacak — eski çalışan script olarak kalacak, referans için.

---

## Verification Plan

### Otomatik Test

Mevcut test altyapısında RAGAS'a özel test yok. Yeni bir test dosyası yazabiliriz ama RAGAS testleri gerçek LLM ve API gerektirdiği için birim testi sınırlı olur.

`api_client.py` için mock-tabanlı bir test yazılabilir:

```bash
cd /home/emirhan/Desktop/rag_mvp
python -m pytest tests/test_api_client.py -v
```

### Manuel Doğrulama

1. **Kendi RAG API'nizi test edin**: Projenin FastAPI sunucusunu başlatın ve yeni aracı kendi API'nize yönlendirin:
   ```bash
   # Terminal 1: RAG API'yi başlat
   cd /home/emirhan/Desktop/rag_mvp
   uvicorn rag_mvp.app:app --host 0.0.0.0 --port 8000

   # Terminal 2: RAGAS test aracını çalıştır
   cd /home/emirhan/Desktop/rag_mvp
   python -m NisanRagasTest.main
   ```
2. Sonuçların `NisanRagasTest/ragas_results.csv` dosyasına yazıldığını kontrol edin
3. Terminal'de metrik özetlerinin görüntülendiğini doğrulayın

> [!IMPORTANT]
> Manuel test için vLLM sunucusunun (port 8108) çalışıyor ve RAG API'nin veri yüklenmiş durumda olması gerekir.
