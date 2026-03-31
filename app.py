"""Arayüz Sağlar: Tarayıcıda açılan, dosya yükleme butonu ve soru kutusu olan basit bir web sayfası sunar.

Dosya İşler (/upload): Yüklediğin PDF veya metin dosyasını okur, küçük parçalara böler ve yapay zekanın arama yapabileceği bir "akıllı hafızaya" kaydeder.

Soru Cevaplar (/ask): Sorduğun soruyla ilgili bilgileri o hafızadan bulur ve ana yapay zeka modeline (vLLM) göndererek sana dökümana dayalı, doğru bir cevap üretir."""
from __future__ import annotations

import logging
import threading
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .file_loader import load_text_from_upload
from .pipeline import RagPipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG MVP", version="0.2.0")
_pipeline: RagPipeline | None = None
_last_doc_id: str | None = None
_lock = threading.Lock()

INDEX_HTML = """
<!doctype html>
<html lang="tr">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>RAG MVP – Multi-Doc</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #090b10;
            --bg-soft: #11141c;
            --card: rgba(19, 24, 33, 0.82);
            --border: rgba(255, 255, 255, 0.12);
            --text: #edf2fb;
            --muted: #9aa6bd;
            --accent: #39d98a;
            --accent-strong: #2fc077;
            --danger: #f56262;
            --pre-bg: #0d1118;
            --shadow: 0 16px 36px rgba(0, 0, 0, 0.38);
            --radius: 16px;
        }

        * { box-sizing: border-box; }

        body {
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
            color: var(--text);
            margin: 0;
            min-height: 100vh;
            background:
                radial-gradient(1000px 500px at 15% -10%, rgba(57, 217, 138, 0.16), transparent 65%),
                radial-gradient(900px 420px at 85% 10%, rgba(64, 173, 255, 0.14), transparent 72%),
                linear-gradient(160deg, #05070b 0%, #0b0f16 45%, #0f121b 100%);
            padding: 34px 16px;
        }

        .page { max-width: 1020px; margin: 0 auto; position: relative; z-index: 1; }

        .hero { margin-bottom: 18px; padding: 18px 2px; animation: fadeUp 0.5s ease-out; }
        h1 { margin: 0 0 8px; font-size: clamp(28px, 4.2vw, 42px); font-weight: 700; line-height: 1.1; }
        .muted { color: var(--muted); margin: 0; font-size: 15px; }

        .cards { display: grid; grid-template-columns: 1fr; gap: 14px; }
        .card {
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--card);
            backdrop-filter: blur(8px);
            padding: 18px;
            box-shadow: var(--shadow);
            animation: fadeUp 0.52s ease-out both;
        }
        .card:nth-child(2) { animation-delay: 0.08s; }
        .card:nth-child(3) { animation-delay: 0.14s; }
        .card:nth-child(4) { animation-delay: 0.20s; }

        h3 { margin: 0 0 12px; font-size: 18px; color: #f4f8ff; }

        input[type='file'], input[type='text'] {
            width: 100%; border-radius: 12px; border: 1px solid var(--border);
            background: var(--bg-soft); color: var(--text); padding: 11px 12px;
            margin: 8px 0 10px; font-size: 15px; font-family: inherit; outline: none;
            transition: border-color 140ms ease, box-shadow 140ms ease;
        }
        input[type='file']:focus, input[type='text']:focus {
            border-color: rgba(57,217,138,0.7);
            box-shadow: 0 0 0 3px rgba(57,217,138,0.15);
        }

        .btn-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
        button {
            border: 0; border-radius: 12px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
            color: #04140c; font-weight: 700; letter-spacing: 0.2px;
            padding: 10px 18px; cursor: pointer;
            transition: transform 130ms ease, filter 130ms ease;
        }
        button:hover { transform: translateY(-1px); filter: brightness(1.06); }
        button:active { transform: translateY(0); }
        .btn-danger {
            background: linear-gradient(135deg, #f56262 0%, #c94e4e 100%);
            color: #fff;
        }
        .btn-outline {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--muted);
        }
        .btn-outline:hover { border-color: var(--accent); color: var(--accent); }

        pre {
            margin-top: 12px; background: var(--pre-bg);
            border: 1px solid rgba(255,255,255,0.09); border-radius: 12px;
            padding: 12px; white-space: pre-wrap; color: #dce7fa;
            font-family: 'IBM Plex Mono', Consolas, monospace; font-size: 13px;
            line-height: 1.5; max-height: 280px; overflow: auto;
        }

        /* Doc list */
        .doc-list { display: grid; gap: 8px; margin-top: 10px; }
        .doc-item {
            display: flex; align-items: center; justify-content: space-between;
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px; padding: 10px 14px; gap: 12px;
        }
        .doc-item-info { flex: 1; min-width: 0; }
        .doc-item-name { font-weight: 600; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .doc-item-meta { color: var(--muted); font-size: 12px; margin-top: 2px; }
        .doc-empty { color: var(--muted); font-style: italic; font-size: 14px; margin-top: 8px; }

        /* Ask result */
        .ask-result { margin-top: 12px; display: grid; gap: 10px; }
        .result-panel {
            background: var(--pre-bg); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px; padding: 12px;
        }
        .result-title { margin: 0 0 8px; font-weight: 700; font-size: 13px; letter-spacing: 0.4px; color: #b7c5de; text-transform: uppercase; }
        .result-answer { margin: 0; color: #ecf3ff; white-space: pre-wrap; line-height: 1.55; }
        .chunks-grid { display: grid; gap: 8px; }
        .chunk-item { border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; background: rgba(255,255,255,0.02); overflow: hidden; }
        .chunk-item summary { cursor: pointer; list-style: none; padding: 10px 12px; font-weight: 600; font-size: 13px; color: #c9d8f0; border-bottom: 1px solid rgba(255,255,255,0.06); }
        .chunk-item summary::-webkit-details-marker { display: none; }
        .chunk-body { padding: 10px 12px 12px; font-family: 'IBM Plex Mono', Consolas, monospace; font-size: 12px; color: #dce7fa; line-height: 1.55; }
        .chunk-meta { color: #9cb0cf; margin-bottom: 8px; }
        .chunk-text { margin: 0; white-space: pre-wrap; word-break: break-word; }
        .result-empty { color: #9cb0cf; font-style: italic; margin: 0; }

        /* Search scope radio */
        .scope-row { display: flex; gap: 14px; margin-bottom: 10px; flex-wrap: wrap; }
        .scope-row label { display: flex; align-items: center; gap: 6px; font-size: 14px; color: var(--muted); cursor: pointer; }
        .scope-row input[type="radio"] { width: auto; margin: 0; accent-color: var(--accent); }

        @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

        @media (max-width: 640px) {
            body { padding: 18px 12px; }
            .card { padding: 14px; }
            button { width: 100%; }
        }
    </style>
</head>
<body>
<main class="page">
    <header class="hero">
        <h1>RAG MVP <span style="font-size:18px;opacity:.55;font-weight:400">v0.2 · Multi-Doc</span></h1>
        <p class="muted">Birden fazla dosya yükle, tüm havuzda ya da tek belgede soru sor.</p>
    </header>

    <section class="cards">
        <!-- 1: Upload -->
        <div class="card">
            <h3>1) Dosya Yükle <span style="font-size:13px;font-weight:400;color:var(--muted)">(çoklu seçim desteklenir)</span></h3>
            <input id="fileInput" type="file" multiple />
            <div class="btn-row">
                <button id="uploadBtn" type="button">Yükle ve İndeksle</button>
            </div>
            <pre id="uploadResult">Henüz yükleme yapılmadı.</pre>
        </div>

        <!-- 2: Document Library -->
        <div class="card">
            <h3>2) Doküman Havuzu</h3>
            <div class="btn-row">
                <button id="refreshDocsBtn" type="button" class="btn-outline">↻ Yenile</button>
            </div>
            <div id="docList" class="doc-list">
                <p class="doc-empty">Henüz doküman yüklenmedi.</p>
            </div>
        </div>

        <!-- 3: Ask -->
        <div class="card">
            <h3>3) Soru Sor</h3>
            <div class="scope-row">
                <label><input type="radio" name="searchScope" value="all" checked> Tüm Havuzda Ara</label>
                <label><input type="radio" name="searchScope" value="last"> Son Yüklenen Belgede</label>
                <label><input type="radio" name="searchScope" value="selected" id="radioSelected"> Seçili Belgede</label>
            </div>
            <div id="selectedDocRow" style="display:none; margin-bottom:8px;">
                <select id="docSelect" style="width:100%;background:var(--bg-soft);border:1px solid var(--border);color:var(--text);padding:9px 12px;border-radius:12px;font-family:inherit;font-size:14px;outline:none;">
                    <option value="">— belge seç —</option>
                </select>
            </div>
            <input id="questionInput" type="text" placeholder="Örn: Proje detayları nelerdir?" />
            <div class="btn-row">
                <button id="askBtn" type="button">Soru Sor</button>
            </div>
            <div id="askResult" class="ask-result"></div>
        </div>

        <!-- 4: Chunk Explorer -->
        <div class="card">
            <h3>4) Chunk Gezgini</h3>
            <div id="chunkDocRow" style="margin-bottom:8px;">
                <select id="chunkDocSelect" style="width:100%;background:var(--bg-soft);border:1px solid var(--border);color:var(--text);padding:9px 12px;border-radius:12px;font-family:inherit;font-size:14px;outline:none;">
                    <option value="">— belge seç —</option>
                </select>
            </div>
            <div class="btn-row">
                <button id="chunksBtn" type="button">Chunk'ları Getir</button>
            </div>
            <pre id="chunksResult">Henüz chunk listesi istenmedi.</pre>
        </div>
    </section>
</main>

<script>
    /* ---- Utilities ---- */
    function escapeHtml(v) {
        return String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
    }
    function normChunk(v) {
        const raw = String(v||'');
        const n = raw.replaceAll('\\r\\n','\\n').replace(/\\n{3,}/g,'\\n\\n').replace(/[ \\t]+\\n/g,'\\n').replace(/\\n[ \\t]+/g,'\\n');
        return n.split(/\\n{2,}/).map(p=>p.replace(/\\n+/g,' ').replace(/\\s{2,}/g,' ').trim()).filter(Boolean).join('\\n\\n');
    }

    /* ---- Doc list state ---- */
    let _docs = [];   // [{doc_id, source_name, chunk_count}]

    function populateDocSelects() {
        const selects = [document.getElementById('docSelect'), document.getElementById('chunkDocSelect')];
        selects.forEach(sel => {
            const prev = sel.value;
            while (sel.options.length > 1) sel.remove(1);
            _docs.forEach(d => {
                const opt = new Option(`${escapeHtml(d.source_name)}  (${d.chunk_count} chunk)`, d.doc_id);
                sel.add(opt);
            });
            if (prev) sel.value = prev;
        });
    }

    async function refreshDocs(silent=false) {
        try {
            const r = await fetch('/documents');
            const data = await r.json();
            _docs = data.documents || [];
            const listEl = document.getElementById('docList');
            if (_docs.length === 0) {
                listEl.innerHTML = '<p class="doc-empty">Havuz boş – dosya yükleyin.</p>';
            } else {
                listEl.innerHTML = _docs.map(d => `
                    <div class="doc-item" id="doc-${d.doc_id}">
                        <div class="doc-item-info">
                            <div class="doc-item-name" title="${escapeHtml(d.doc_id)}">${escapeHtml(d.source_name)}</div>
                            <div class="doc-item-meta">${d.chunk_count} chunk &nbsp;|&nbsp; id: ${escapeHtml(d.doc_id.slice(0,8))}…</div>
                        </div>
                        <button class="btn-danger" onclick="deleteDoc('${escapeHtml(d.doc_id)}','${escapeHtml(d.source_name)}')" title="Sil">🗑 Sil</button>
                    </div>`).join('');
            }
            populateDocSelects();
        } catch(e) {
            if (!silent) console.error('refreshDocs error', e);
        }
    }

    async function deleteDoc(docId, sourceName) {
        if (!confirm(`"${sourceName}" silinsin mi?`)) return;
        try {
            const r = await fetch(`/documents/${encodeURIComponent(docId)}`, {method:'DELETE'});
            const data = await r.json();
            if (!r.ok) { alert('Silinemedi: ' + (data.detail||r.status)); return; }
            await refreshDocs();
        } catch(e) { alert('Silinemedi: ' + e); }
    }

    /* ---- Upload (multi-file) ---- */
    async function uploadFiles() {
        const input = document.getElementById('fileInput');
        const out = document.getElementById('uploadResult');
        if (!input.files || input.files.length === 0) { out.textContent = 'Lütfen bir veya daha fazla dosya seç.'; return; }
        out.textContent = `Yükleniyor… (${input.files.length} dosya)`;
        const results = [];
        for (const file of input.files) {
            const fd = new FormData();
            fd.append('file', file);
            try {
                const r = await fetch('/upload', {method:'POST', body:fd});
                const ct = r.headers.get('content-type')||'';
                const d = ct.includes('application/json') ? await r.json() : {detail: await r.text()};
                if (!r.ok) { results.push(`✗ ${file.name}: ${d.detail||r.status}`); } 
                else { results.push(`✓ ${file.name} → ${d.chunk_count} chunk (${d.doc_id.slice(0,8)}…)`); }
            } catch(e) { results.push(`✗ ${file.name}: ${e.message||e}`); }
        }
        out.textContent = results.join('\\n');
        await refreshDocs();
    }

    /* ---- Ask ---- */
    function renderAskResult(answerText, retrieved) {
        const safe = escapeHtml(answerText||'Cevap alınamadı.');
        const has = Array.isArray(retrieved) && retrieved.length > 0;
        const chunksHtml = has ? retrieved.map((item,idx)=>{
            const dense = Number(item.score), rerank = Number(item.rerank_score);
            return `<details class="chunk-item" ${idx<2?'open':''}>
                <summary>#${idx+1} | dense=${Number.isFinite(dense)?dense.toFixed(4):'n/a'} | rerank=${Number.isFinite(rerank)?rerank.toFixed(4):'n/a'}</summary>
                <div class="chunk-body">
                    <div class="chunk-meta">chunk_id=${escapeHtml(item.chunk_id||'?')} | source=${escapeHtml(item.source_name||'?')}</div>
                    <p class="chunk-text">${escapeHtml(normChunk(item.text||''))}</p>
                </div></details>`;
        }).join('') : '<p class="result-empty">Getirilen chunk yok.</p>';
        return `<div class="result-panel"><p class="result-title">Cevap</p><p class="result-answer">${safe}</p></div>
                <div class="result-panel"><p class="result-title">Getirilen Chunklar (${has?retrieved.length:0})</p><div class="chunks-grid">${chunksHtml}</div></div>`;
    }

    async function askQuestion() {
        const question = document.getElementById('questionInput').value.trim();
        const scope = document.querySelector('input[name="searchScope"]:checked')?.value || 'all';
        const out = document.getElementById('askResult');
        if (!question) return;
        out.innerHTML = '<div class="result-panel"><p class="result-empty">Cevap üretiliyor…</p></div>';
        let docId = null;
        if (scope === 'selected') {
            docId = document.getElementById('docSelect').value || null;
        } else if (scope === 'last') {
            // send without doc_id = server uses last uploaded
            docId = undefined;
        }
        // scope==='all' → docId = null (explicit null = global pool)
        const body = {question};
        if (docId !== undefined) body.doc_id = docId;
        try {
            const r = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
            const ct = r.headers.get('content-type')||'';
            const data = ct.includes('json') ? await r.json() : {detail: await r.text()};
            if (!r.ok) { out.innerHTML = `<div class="result-panel"><p class="result-answer" style="color:var(--danger)">Hata (${r.status}): ${escapeHtml(data.detail||'')}</p></div>`; return; }
            out.innerHTML = renderAskResult(data.answer, data.retrieved);
        } catch(e) { out.innerHTML = `<div class="result-panel"><p class="result-answer" style="color:var(--danger)">İstek hatası: ${escapeHtml(String(e))}</p></div>`; }
    }

    /* ---- Chunks ---- */
    async function loadChunks() {
        const docId = document.getElementById('chunkDocSelect').value;
        const out = document.getElementById('chunksResult');
        if (!docId) { out.textContent = 'Lütfen belge seçin.'; return; }
        out.textContent = 'Yükleniyor…';
        try {
            const r = await fetch(`/chunks?doc_id=${encodeURIComponent(docId)}&limit=300`);
            const data = await r.json();
            if (!data.chunks || data.chunks.length === 0) { out.textContent = 'Chunk bulunamadı.'; return; }
            const lines = [`doc_id=${data.doc_id}`, `chunk_count=${data.chunk_count}`, ''];
            data.chunks.forEach(item => {
                lines.push(`#${item.chunk_index} chunk_id=${item.chunk_id}`);
                lines.push(`  source=${item.source_name}`);
                lines.push(`  text=${item.text.slice(0,200)}${item.text.length>200?'…':''}`);
                lines.push('');
            });
            out.textContent = lines.join('\\n');
        } catch(e) { out.textContent = 'Hata: ' + e; }
    }

    /* ---- Scope radio toggle ---- */
    document.querySelectorAll('input[name="searchScope"]').forEach(r => {
        r.addEventListener('change', () => {
            document.getElementById('selectedDocRow').style.display =
                document.getElementById('radioSelected').checked ? 'block' : 'none';
        });
    });

    /* ---- Init ---- */
    window.addEventListener('DOMContentLoaded', () => {
        document.getElementById('uploadBtn').addEventListener('click', uploadFiles);
        document.getElementById('askBtn').addEventListener('click', askQuestion);
        document.getElementById('chunksBtn').addEventListener('click', loadChunks);
        document.getElementById('refreshDocsBtn').addEventListener('click', () => refreshDocs(false));
        refreshDocs(true);
    });
</script>
</body>
</html>
"""


class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                logger.info("Initializing RagPipeline (first request)...")
                _pipeline = RagPipeline()
                logger.info("RagPipeline initialized successfully.")
    return _pipeline


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        content=INDEX_HTML,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health")
def health() -> dict:
    pipeline_ready = True
    dependency_status = {
        "qdrant_ready": False,
        "vllm_ready": False,
        "embedding_dim": 0,
    }
    try:
        dependency_status = get_pipeline().health_status()
    except Exception:
        pipeline_ready = False
    return {
        "status": "ok",
        "pipeline_ready": pipeline_ready,
        "dependencies": dependency_status,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Upload and index a single file. Call multiple times (or use the batch helper) for multi-doc."""
    global _last_doc_id
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")
    try:
        pipeline = get_pipeline()
        text = await load_text_from_upload(file)
        logger.info("Ingesting file: %s (%d chars)", file.filename, len(text))
        result = pipeline.ingest_text(text=text, source_name=file.filename)
        with _lock:
            _last_doc_id = result.doc_id
        logger.info("File ingested: doc_id=%s, chunks=%d", result.doc_id, result.chunk_count)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload failed for file: %s", file.filename)
        raise HTTPException(status_code=503, detail=f"Pipeline is not ready: {exc}") from exc
    return {
        "message": "File indexed successfully.",
        "doc_id": result.doc_id,
        "source_name": result.source_name,
        "chunk_count": result.chunk_count,
    }


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz")

    try:
        pipeline = get_pipeline()

        # Scoping logic:
        # 1. Explicit doc_id provided → use it (single-doc search)
        # 2. doc_id explicitly null in JSON → None (global pool search)
        # 3. doc_id field absent from JSON (payload default None) → fallback to last uploaded
        target_doc_id = payload.doc_id
        if target_doc_id is None and payload.model_fields_set and "doc_id" not in payload.model_fields_set:
            with _lock:
                target_doc_id = _last_doc_id

        logger.info(
            "Arama başlatılıyor: %s (Kapsam: %s)",
            payload.question[:50],
            target_doc_id if target_doc_id else "TÜM HAVUZ",
        )
        return pipeline.ask(payload.question, doc_id=target_doc_id)
    except Exception as exc:
        logger.exception("Arama hatası")
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/documents")
def list_documents() -> dict:
    """Return all indexed documents with chunk counts."""
    try:
        pipeline = get_pipeline()
        docs = pipeline.list_documents()
        return {"documents": docs, "total": len(docs)}
    except Exception as exc:
        logger.exception("list_documents failed")
        raise HTTPException(status_code=503, detail=str(exc))


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    """Delete all chunks belonging to the given doc_id."""
    global _last_doc_id
    if not doc_id.strip():
        raise HTTPException(status_code=400, detail="doc_id is required")
    try:
        pipeline = get_pipeline()
        deleted = pipeline.delete_document(doc_id)
        with _lock:
            if _last_doc_id == doc_id:
                _last_doc_id = None
        return {"message": "Document deleted.", "doc_id": doc_id, "deleted_chunks": deleted}
    except Exception as exc:
        logger.exception("delete_document failed for doc_id: %s", doc_id)
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/chunks")
def chunks(
    doc_id: str | None = Query(default=None),
    limit: int = Query(default=300, ge=1, le=2000),
) -> dict:
    with _lock:
        target_doc_id = doc_id or _last_doc_id
    if not target_doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required or upload a file first")
    try:
        pipeline = get_pipeline()
        chunk_rows = pipeline.list_doc_chunks(doc_id=target_doc_id, limit=limit)
        return {
            "doc_id": target_doc_id,
            "chunk_count": len(chunk_rows),
            "chunks": chunk_rows,
        }
    except Exception as exc:
        logger.exception("Chunks retrieval failed for doc_id: %s", target_doc_id)
        raise HTTPException(status_code=503, detail=f"Pipeline is not ready: {exc}") from exc
