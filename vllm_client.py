from __future__ import annotations

import logging
import re
import time

import requests

logger = logging.getLogger(__name__)

NO_ANSWER_TEXT = "Bu sorunun cevabı verilen bağlamda yok."
SYSTEM_PROMPT = (
    "Sen bir RAG asistanısın.\n"
    "Sadece verilen bağlama göre cevap ver.\n"
    "Bağlam dışında bilgi kullanma.\n"
    "Cevabı kısa, net ve doğrudan ver.\n"
    "Asla düşünce süreci, analiz adımları, reasoning veya açıklama üretme.\n"
    "Eğer cevap bağlamda yoksa aynen şu cümleyi yaz:\n"
    '"Bu sorunun cevabı verilen bağlamda yok."'
)

RETRY_SYSTEM_PROMPT = (
    "Sen bir RAG asistanısın.\n"
    "Yalnızca verilen bağlamdan cevap üret.\n"
    "Bağlamda açıkça geçen isim, sayı ve terimleri öncelikle çıkar.\n"
    "Soru model/isim listesi soruyorsa bağlamdaki adayları madde madde yaz.\n"
    "Bağlamda gerçekten bilgi yoksa şu cümleyi aynen yaz:\n"
    '"Bu sorunun cevabı verilen bağlamda yok."'
)


class VllmClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def _clean_answer(self, raw_answer: str) -> str:
        answer = (raw_answer or "").strip()
        answer = re.sub(r"(?is)<think>.*?</think>", "", answer).strip()

        if not answer:
            return NO_ANSWER_TEXT

        final_match = re.search(r"(?is)(?:final answer|cevap)\s*[:\-]\s*(.+)$", answer)
        if final_match:
            answer = final_match.group(1).strip()

        lowered = answer.lower()
        if any(token in lowered for token in ["thinking process", "reasoning", "analiz adımları"]):
            return NO_ANSWER_TEXT

        answer = answer.strip().strip('"').strip("'")
        if answer == NO_ANSWER_TEXT.strip('"'):
            return NO_ANSWER_TEXT

        compact_answer = " ".join(line.strip() for line in answer.splitlines() if line.strip())
        if not compact_answer:
            return NO_ANSWER_TEXT

        if len(compact_answer) > 700:
            sentence_split = re.split(r"(?<=[.!?])\s+", compact_answer)
            trimmed: list[str] = []
            total_len = 0
            for sentence in sentence_split:
                sentence = sentence.strip()
                if not sentence:
                    continue
                projected = total_len + len(sentence) + (1 if trimmed else 0)
                if projected > 700:
                    break
                trimmed.append(sentence)
                total_len = projected
            compact_answer = " ".join(trimmed).strip() or compact_answer[:700].strip()

        return compact_answer

    def is_ready(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=min(5, self.timeout_s),
            )
            response.raise_for_status()
            body = response.json()
            return isinstance(body, dict) and "data" in body
        except Exception:
            return False

    def _request_completion(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        start_time = time.monotonic()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            elapsed = time.monotonic() - start_time
            logger.error("vLLM request timed out after %.1fs (limit: %ds)", elapsed, self.timeout_s)
            raise RuntimeError(
                f"LLM isteği zaman aşımına uğradı ({elapsed:.0f}s). "
                f"vLLM sunucusu yanıt vermiyor olabilir. URL: {self.base_url}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            logger.error("vLLM connection failed: %s", exc)
            raise RuntimeError(
                f"LLM sunucusuna bağlanılamadı. "
                f"vLLM servisinin çalıştığından emin olun. URL: {self.base_url}"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            logger.error("vLLM HTTP error: %s %s", response.status_code, response.text[:200])
            raise RuntimeError(
                f"LLM sunucusu hata döndü (HTTP {response.status_code}). "
                f"Detay: {response.text[:200]}"
            ) from exc

        elapsed = time.monotonic() - start_time
        body = response.json()
        raw_answer = body["choices"][0]["message"].get("content", "")
        cleaned = self._clean_answer(raw_answer)

        logger.info(
            "vLLM completion: %.1fs, prompt_len=%d, answer_len=%d",
            elapsed, len(user_prompt), len(cleaned),
        )
        return cleaned

    def answer_with_context(self, question: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return NO_ANSWER_TEXT

        context = "\n\n".join(
            [f"[{idx + 1}] {chunk}" for idx, chunk in enumerate(context_chunks)]
        )
        user_prompt = (
            "Bağlam:\n"
            f"{context}\n\n"
            "Soru:\n"
            f"{question}\n\n"
            "Sadece nihai cevabı yaz."
        )

        first_answer = self._request_completion(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        if first_answer != NO_ANSWER_TEXT:
            return first_answer

        logger.info("First answer was no-answer, retrying with relaxed prompt...")

        retry_user_prompt = (
            "Bağlamı dikkatle tara ve soruyla doğrudan ilişkili isimleri/terimleri çıkar.\n"
            "Eğer soru bir liste istiyorsa, bağlamda geçen adayları listele.\n\n"
            f"{user_prompt}"
        )
        return self._request_completion(
            system_prompt=RETRY_SYSTEM_PROMPT,
            user_prompt=retry_user_prompt,
        )
