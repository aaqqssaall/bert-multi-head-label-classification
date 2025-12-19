"""
gemini_client.py

Abstraksi kecil untuk berinteraksi dengan Gemini API (Developer API)
menggunakan Google Gen AI Python SDK.

Dependensi:
    pip install --upgrade google-genai

Konfigurasi:
    - Set environment variable GEMINI_API_KEY dengan API key Gemini Developer.
    - Opsional:
        GEMINI_MODEL_NAME (default: "gemini-2.5-flash")

API utama yang akan dipakai di aplikasi:
    - get_gemini_client() -> genai.Client
    - suggest_labels_for_review(...) -> LLMLabelSuggestion
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import Client, types
from google.genai import errors as genai_errors
from google.genai.errors import ServerError



# ------------- Konfigurasi dasar -------------


def _get_default_model_name() -> str:
    """
    Mengambil nama model default dari env var GEMINI_MODEL_NAME,
    jika tidak ada pakai gemini-2.5-flash.
    """
    return os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")


_GEMINI_CLIENT: Optional[genai.Client] = None


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Mengembalikan singleton genai.Client.

    Jika api_key tidak diberikan, library akan otomatis membaca
    environment variable GEMINI_API_KEY.
    """
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT

    if api_key:
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    else:
        _GEMINI_CLIENT = genai.Client()

    return _GEMINI_CLIENT


# ------------- Tipe data untuk rekomendasi label -------------


@dataclass
class LLMLabelSuggestion:
    """
    Representasi hasil rekomendasi label dari Gemini untuk satu review.
    Struktur ini yang nantinya akan ditulis ke llm_recommendation.jsonl.
    """

    id: str
    aspects: List[str]
    comment_types: List[str]
    reason: str

    llm_model: str
    created_at_ts: float  # epoch timestamp detik
    input_version: Dict[str, Any]

    raw_response: Dict[str, Any]


# ------------- Prompt & helper -------------


def _build_prompt_for_labeling(
    text: str,
    aspect_labels: List[str],
    comment_type_labels: List[str],
    *,
    language: str = "id",
    nli_aspects: Optional[List[str]] = None,
    nli_comment_types: Optional[List[str]] = None,
    model_aspects: Optional[List[str]] = None,
    model_comment_types: Optional[List[str]] = None,
) -> str:
    """
    Menyusun prompt teks yang hemat token untuk meminta saran aspek dan
    comment_type dari Gemini.

    Prinsip:
        - Instruksi singkat, satu blok.
        - Daftar label hanya disebut sekali.
        - Format output tidak perlu dijelaskan karena sudah diatur via
          response_schema di sisi kode.
        - Info NLI / IndoBERT hanya dimasukkan jika tersedia.
    """
    lines: List[str] = []

    # Instruksi inti, singkat
    lines.append("Anda menganalisis ulasan pelanggan berbahasa Indonesia.")
    lines.append(
        "Tugas Anda: dari teks ulasan, pilih nol atau lebih aspek dan comment_type "
        "BERDASARKAN daftar label berikut, lalu berikan alasan singkat."
    )
    lines.append(
        f"- aspek (multi-label): {aspect_labels}"
    )
    lines.append(
        f"- comment_type (multi-label): {comment_type_labels}"
    )
    lines.append(
        "Jika tidak ada aspek atau comment_type yang cocok, gunakan list kosong."
    )
    lines.append(
        "Alasan harus singkat dalam bahasa Indonesia dan, jika relevan, "
        "sebutkan bagian teks yang mendukung."
    )
    lines.append(f"Bahasa teks ulasan: {language}.")
    lines.append("")

    # Info model lain, hanya jika ada. Tetap singkat.
    if nli_aspects or nli_comment_types or model_aspects or model_comment_types:
        lines.append("Info model lain (opsional untuk dipertimbangkan):")
        if nli_aspects or nli_comment_types:
            lines.append(f"- aspek (NLI): {nli_aspects or []}")
            lines.append(f"- comment_type (NLI): {nli_comment_types or []}")
        if model_aspects or model_comment_types:
            lines.append(f"- aspek (IndoBERT): {model_aspects or []}")
            lines.append(f"- comment_type (IndoBERT): {model_comment_types or []}")
        lines.append(
            "Jika Anda tidak setuju dengan salah satu model, boleh dijelaskan singkat "
            "di alasan."
        )
        lines.append("")

    # Teks ulasan
    lines.append("Teks ulasan:")
    lines.append(text.strip())

    # Tidak perlu menjelaskan format JSON di sini karena sudah dijamin
    # oleh response_schema di konfigurasi generate_content.

    return "\n".join(lines)


# ------------- Fungsi utama untuk rekomendasi label -------------


def suggest_labels_for_review(
    *,
    review_id: str,
    text: str,
    aspect_labels: List[str],
    comment_type_labels: List[str],
    language: str = "id",
    text_hash: Optional[str] = None,
    nli_aspects: Optional[List[str]] = None,
    nli_comment_types: Optional[List[str]] = None,
    model_aspects: Optional[List[str]] = None,
    model_comment_types: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    extra_input_version: Optional[Dict[str, Any]] = None,
) -> LLMLabelSuggestion:
    """
    Meminta rekomendasi aspek & comment_type ke Gemini untuk satu review.

    Fokus pada efisiensi token:
        - Prompt singkat, tanpa penjelasan format JSON.
        - Format output diatur lewat response_schema.

    Parameter lain sama seperti versi sebelumnya.
    """
    client = get_gemini_client(api_key=api_key)
    model = model_name or _get_default_model_name()

    prompt = _build_prompt_for_labeling(
        text=text,
        aspect_labels=aspect_labels,
        comment_type_labels=comment_type_labels,
        language=language,
        nli_aspects=nli_aspects,
        nli_comment_types=nli_comment_types,
        model_aspects=model_aspects,
        model_comment_types=model_comment_types,
    )

    # Schema terstruktur agar output langsung JSON valid
    class LabelSuggestionSchema(types.TypedDict):
        aspects: List[str]
        comment_types: List[str]
        reason: str

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=LabelSuggestionSchema,
    )

    max_retries = 3
    backoff_seconds = 0.8
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            break
        except ServerError as e:
            # fokus ke kasus 503 UNAVAILABLE (model overloaded)
            msg = str(e)
            if "UNAVAILABLE" in msg or "overloaded" in msg:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(backoff_seconds * (attempt + 1))
                    continue
            # error server lain langsung lempar
            raise RuntimeError(f"Gemini API error while labeling review {review_id}: {e}")
        except genai_errors.ClientError as e:
            # error client (quota, auth, dsb.) langsung lempar
            raise RuntimeError(f"Gemini API error while labeling review {review_id}: {e}")
    else:
        # semua percobaan gagal karena 503
        raise RuntimeError(
            f"Gemini API overloaded while labeling review {review_id}: {last_err}"
        )


    parsed: Any = getattr(response, "parsed", None)
    if parsed is None:
        text_out = (getattr(response, "text", None) or "").strip()
        raise RuntimeError(
            f"Gemini response for review {review_id} does not contain structured JSON. "
            f"Raw text: {text_out[:400]}"
        )

    aspects = parsed.get("aspects") or []
    comment_types = parsed.get("comment_types") or []
    reason = parsed.get("reason") or ""

    # Normalisasi: hanya terima label yang memang ada di daftar
    aspects = [a for a in aspects if a in aspect_labels]
    comment_types = [c for c in comment_types if c in comment_type_labels]

    created_at_ts = time.time()
    input_version: Dict[str, Any] = {
        "text_hash": text_hash,
        "source": "low_score",
        "language": language,
        "nli_aspects": nli_aspects or [],
        "nli_comment_types": nli_comment_types or [],
        "model_aspects": model_aspects or [],
        "model_comment_types": model_comment_types or [],
    }
    if extra_input_version:
        input_version.update(extra_input_version)

    suggestion = LLMLabelSuggestion(
        id=review_id,
        aspects=aspects,
        comment_types=comment_types,
        reason=reason,
        llm_model=model,
        created_at_ts=created_at_ts,
        input_version=input_version,
        raw_response={
            "parsed": parsed,
            "raw_text": getattr(response, "text", None),
        },
    )

    return suggestion


# ------------- Utility opsional untuk testing manual -------------


def _demo() -> None:
    """
    Demo singkat untuk menjalankan modul ini secara langsung:

        python -m gemini_client

    Pastikan GEMINI_API_KEY sudah di-set.
    """
    example_text = "rasanya cukup enak dan segar, tapi masih lama pelayanannya"
    aspect_labels = ["food", "service", "place_ambience", "price"]
    comment_type_labels = ["praise", "complaint", "suggestion", "question"]

    sugg = suggest_labels_for_review(
        review_id="demo_001",
        text=example_text,
        aspect_labels=aspect_labels,
        comment_type_labels=comment_type_labels,
        language="id",
    )
    print("LLM suggestion:")
    print("id        :", sugg.id)
    print("aspects   :", sugg.aspects)
    print("types     :", sugg.comment_types)
    print("reason    :", sugg.reason)
    print("model     :", sugg.llm_model)
    print("timestamp :", sugg.created_at_ts)


if __name__ == "__main__":
    _demo()
