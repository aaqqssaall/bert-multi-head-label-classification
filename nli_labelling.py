# nli_labelling.py
#
# Zero-shot NLI labelling untuk aspek & comment_type (multi-label).
# Fokus: akurat + label_quality lebih representatif (lebih banyak high bila evidence jelas),
# tanpa mengubah format output (keys/struktur tetap sama).
#
# Konfigurasi threshold via data/nli_params.json:
#   {
#     "kw_threshold": 0.2,
#     "strict_threshold": 0.55,
#     "comment_threshold": 0.35
#   }

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

id2label = model.config.id2label

ASPECT_LABELS = ["food", "price", "service", "place_ambience", "misc"]
COMMENT_TYPE_LABELS = ["complaint", "praise", "suggestion"]
ASPECT_LABELS_NLI = ["food", "price", "service", "place_ambience"]

# Hypothesis dibuat lebih natural dan konsisten (Indonesia), untuk meningkatkan entailment.
# Aspek = topik, bukan sentimen.
ASPECT_HYPOTHESES: Dict[str, List[str]] = {
    "food": [
        "Ada komentar tentang makanan atau minuman.",
        "Ulasan ini membahas rasa, porsi, atau kualitas makanan/minuman.",
    ],
    "price": [
        "Ada komentar tentang harga atau biaya.",
        "Ulasan ini membahas murah/mahal atau nilai dari uang yang dibayar.",
    ],
    "service": [
        "Ada komentar tentang pelayanan atau staf.",
        "Ulasan ini membahas kecepatan layanan atau sikap pelayan.",
    ],
    "place_ambience": [
        "Ada komentar tentang tempat atau suasana.",
        "Ulasan ini membahas kebersihan, kenyamanan, atau ambience.",
    ],
}

ASPECT_KEYWORDS = {
    "food": [
        "makan", "makanan", "minum", "minuman", "menu",
        "rasa", "enak", "lezat", "porsi", "kopi", "susu", "teh"
    ],
    "price": ["harga", "mahal", "murah", "worth it", "worth", "terjangkau", "diskon", "promo"],
    "service": [
        "pelayan", "pelayanan", "staf", "karyawan", "server", "waiter", "waitress",
        "antri", "antre", "lama", "lambat", "cepat"
    ],
    "place_ambience": ["tempat", "suasana", "ambience", "ramai", "sepi", "nyaman", "bersih", "kotor", "interior", "parkir", "toilet"],
}

COMMENT_HYPOTHESES: Dict[str, List[str]] = {
    "complaint": [
        "Ulasan ini berisi keluhan atau ketidakpuasan.",
        "Ulasan ini menyebut masalah atau hal yang mengecewakan.",
    ],
    "praise": [
        "Ulasan ini memuji atau bernada positif.",
        "Ulasan ini menyatakan kepuasan.",
    ],
    "suggestion": [
        "Ulasan ini berisi saran atau usulan perbaikan.",
        "Ulasan ini meminta perbaikan atau memberi rekomendasi perubahan.",
    ],
}

# Lexicon cues (backoff & segment selection), tetap lightweight.
NEGATIVE_HINTS = [
    "buruk", "jelek", "parah", "kecewa", "mengecewakan",
    "tidak puas", "kurang puas", "ga puas", "gak puas", "nggak puas", "enggak puas",
    "lambat", "lama", "telat", "dingin", "asin", "pahit", "basi", "kotor",
]
POSITIVE_HINTS = [
    "enak", "lezat", "mantap", "puas", "sangat puas",
    "recommended", "rekomen", "top", "suka", "favorit",
    "ramah", "cepat", "bersih", "nyaman", "segar",
]
SUGGESTION_HINTS = [
    "sebaiknya", "saran", "mungkin bisa", "akan lebih baik",
    "alangkah baiknya", "harap", "tolong", "mohon", "please",
]

NEGATORS = {"tidak", "tak", "ga", "gak", "nggak", "enggak", "bukan", "jangan", "tanpa", "belum"}
ENCLITICS = ("nya", "ku", "mu", "lah", "kah", "pun")

# discourse markers; sering keluhan muncul setelah ini
CONTRAST_MARKERS = ("tapi", "namun", "cuma", "hanya", "sayang", "meski", "walau", "walaupun")
MAX_SEGMENTS = 6  # tetap ringan


@dataclass
class NLIParams:
    kw_threshold: float = 0.2
    strict_threshold: float = 0.55
    comment_threshold: float = 0.35


def _normalize_token(tok: str) -> str:
    t = (tok or "").lower().strip()
    for suf in ENCLITICS:
        if len(t) > len(suf) + 2 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    return t


def _simple_tokens(text: str) -> List[str]:
    raw = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
    return [_normalize_token(t) for t in raw if t]


def _split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?:[\n\r]+|[.!?]+|;)+", t)
    sents = [p.strip() for p in parts if p and p.strip()]
    return sents if sents else [t]


def _split_contrast_clauses(sentence: str) -> List[str]:
    s = (sentence or "").strip()
    if not s:
        return []
    sl = s.lower()
    # split on first contrast marker occurrence to capture "after but" clause
    for m in CONTRAST_MARKERS:
        idx = sl.find(f" {m} ")
        if idx >= 0:
            left = s[:idx].strip()
            right = s[idx + 1 :].strip()  # keep marker in right roughly
            out = []
            if left:
                out.append(left)
            if right:
                out.append(right)
            return out if out else [s]
    return [s]


def _contains_kw(text_lower: str, tokens: List[str], kw: str) -> bool:
    kw = (kw or "").strip()
    if not kw:
        return False
    if " " in kw:
        return kw.lower() in text_lower
    return _normalize_token(kw) in set(tokens)


def _find_phrase_positions(tokens: List[str], phrase_tokens: List[str]) -> List[int]:
    if not phrase_tokens:
        return []
    out: List[int] = []
    n = len(phrase_tokens)
    for i in range(0, max(0, len(tokens) - n + 1)):
        if tokens[i:i + n] == phrase_tokens:
            out.append(i)
    return out


def _is_negated(tokens: List[str], start_idx: int, window: int = 3) -> bool:
    left = max(0, start_idx - window)
    ctx = tokens[left:start_idx]
    return any(t in NEGATORS for t in ctx)


def _detect_cues(text: str) -> Dict[str, bool]:
    text_lower = (text or "").lower()
    tokens = _simple_tokens(text)

    has_neg = False
    has_pos = False
    has_sugg = False

    # Frasa eksplisit: "tidak ada keluhan" biasanya bernada puas/netral-positif.
    if re.search(r"\b(tidak|ga|gak|nggak|enggak)\s+ada\s+keluhan\b", text_lower):
        has_pos = True

    # suggestion cues
    for hint in SUGGESTION_HINTS:
        if hint.lower() in text_lower:
            has_sugg = True
            break

    def apply_polarity(hint: str, is_positive_hint: bool) -> None:
        nonlocal has_neg, has_pos
        h = hint.lower().strip()
        if not h:
            return

        if " " in h:
            phrase_tokens = [_normalize_token(x) for x in re.findall(r"[a-zA-Z0-9_]+", h)]
            for pos in _find_phrase_positions(tokens, phrase_tokens):
                if _is_negated(tokens, pos):
                    if is_positive_hint:
                        has_neg = True
                    else:
                        has_pos = True
                else:
                    if is_positive_hint:
                        has_pos = True
                    else:
                        has_neg = True
        else:
            ht = _normalize_token(h)
            positions = [i for i, t in enumerate(tokens) if t == ht]
            for pos in positions:
                if _is_negated(tokens, pos):
                    if is_positive_hint:
                        has_neg = True
                    else:
                        has_pos = True
                else:
                    if is_positive_hint:
                        has_pos = True
                    else:
                        has_neg = True

    for hint in POSITIVE_HINTS:
        apply_polarity(hint, is_positive_hint=True)
    for hint in NEGATIVE_HINTS:
        apply_polarity(hint, is_positive_hint=False)

    return {"has_neg": has_neg, "has_pos": has_pos, "has_sugg": has_sugg}


def _get_entail_contrad_indices() -> Tuple[int, int, Optional[int]]:
    entail_idx: Optional[int] = None
    contr_idx: Optional[int] = None
    neutral_idx: Optional[int] = None
    for i, name in id2label.items():
        name_low = str(name).lower()
        if "entail" in name_low:
            entail_idx = int(i)
        elif "contrad" in name_low:
            contr_idx = int(i)
        elif "neutral" in name_low:
            neutral_idx = int(i)
    if entail_idx is None or contr_idx is None:
        raise RuntimeError(f"Cannot locate entailment/contradiction labels in model.config.id2label: {id2label}")
    return entail_idx, contr_idx, neutral_idx


ENTAIL_IDX, CONTR_IDX, NEUTRAL_IDX = _get_entail_contrad_indices()


def _nli_logits(premise: str, hypothesis: str) -> torch.Tensor:
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0]
    return logits


def nli_entailment_prob(premise: str, hypothesis: str) -> float:
    """
    Output p(entailment) 0..1 seperti versi sebelumnya.
    Dipakai untuk debug (comment_raw_entailment) dan aspect_confidence.
    """
    logits = _nli_logits(premise, hypothesis)
    if NEUTRAL_IDX is not None:
        probs = torch.softmax(logits, dim=-1)
        return float(probs[ENTAIL_IDX].detach().cpu().numpy())

    pair = torch.stack([logits[CONTR_IDX], logits[ENTAIL_IDX]])
    probs2 = torch.softmax(pair, dim=-1)
    return float(probs2[1].detach().cpu().numpy())


def _nli_margin_score(premise: str, hypothesis: str) -> float:
    """
    Skor lebih stabil untuk keputusan: entailment harus mengalahkan neutral/contradiction.
    Output 0..1 via sigmoid(margin).
    """
    logits = _nli_logits(premise, hypothesis)
    other = logits[CONTR_IDX]
    if NEUTRAL_IDX is not None:
        other = torch.maximum(other, logits[NEUTRAL_IDX])
    margin = float((logits[ENTAIL_IDX] - other).detach().cpu().numpy())
    return float(1.0 / (1.0 + np.exp(-margin)))


def _select_segments(text: str) -> List[str]:
    """
    Segmentasi: kalimat -> pecah klausa kontras (tapi/namun/...) -> pilih segmen relevan.
    Tujuan: jangan kehilangan evidence yang sering muncul setelah "tapi".
    """
    sents = _split_sentences(text)
    if not sents:
        return [text]

    clauses: List[str] = []
    for s in sents:
        clauses.extend(_split_contrast_clauses(s))

    if len(clauses) <= MAX_SEGMENTS:
        return clauses

    # score segmen berdasarkan keyword hits agar segmen informatif diprioritaskan
    all_kw: List[str] = []
    for kws in ASPECT_KEYWORDS.values():
        all_kw.extend(kws)
    all_kw.extend(NEGATIVE_HINTS)
    all_kw.extend(POSITIVE_HINTS)
    all_kw.extend(SUGGESTION_HINTS)
    all_kw.extend(list(CONTRAST_MARKERS))

    scored: List[Tuple[int, int, str]] = []
    for s in clauses:
        sl = s.lower()
        toks = _simple_tokens(s)
        hit = 0
        for kw in all_kw:
            if _contains_kw(sl, toks, kw):
                hit += 1
        scored.append((hit, len(sl), s))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = [s for _, _, s in scored[:MAX_SEGMENTS]]
    return top if top else clauses[:MAX_SEGMENTS]


def _segments_with_label_keywords(text: str, label_keywords: List[str]) -> List[str]:
    segs = _select_segments(text)
    if not label_keywords:
        return segs
    out = []
    for s in segs:
        sl = s.lower()
        toks = _simple_tokens(s)
        if any(_contains_kw(sl, toks, kw) for kw in label_keywords):
            out.append(s)
    return out if out else segs


def _segments_for_comment_label(text: str, label: str) -> List[str]:
    """
    Pilih segmen yang lebih relevan untuk masing-masing label comment type.
    Complaint: segmen yang mengandung negative cues atau segmen setelah marker kontras.
    Praise: segmen yang mengandung positive cues atau frasa "tidak ada keluhan".
    Suggestion: segmen yang mengandung suggestion cues.
    """
    segs = _select_segments(text)
    tl = (text or "").lower()

    if label == "suggestion":
        out = []
        for s in segs:
            sl = s.lower()
            if any(h in sl for h in SUGGESTION_HINTS):
                out.append(s)
        return out if out else segs

    if label == "complaint":
        out = []
        for s in segs:
            sl = s.lower()
            # prioritaskan segmen kontras/keluhan
            if any(f" {m} " in sl for m in CONTRAST_MARKERS):
                out.append(s)
                continue
            if any(h in sl for h in NEGATIVE_HINTS):
                out.append(s)
                continue
            # juga ambil segmen berisi kata service slow umum
            if ("lama" in sl) or ("lambat" in sl) or ("antri" in sl) or ("antre" in sl):
                out.append(s)
        return out if out else segs

    if label == "praise":
        out = []
        for s in segs:
            sl = s.lower()
            if re.search(r"\b(tidak|ga|gak|nggak|enggak)\s+ada\s+keluhan\b", sl):
                out.append(s)
                continue
            if any(h in sl for h in POSITIVE_HINTS):
                out.append(s)
        # bila mixed, praise sering ada di awal sebelum "tapi"; ambil segmen pertama juga
        if not out and segs:
            out = [segs[0]]
        return out if out else segs

    return segs


def _best_entailment_over_segments(text: str, hyps: List[str], prefer_segments: Optional[List[str]] = None) -> float:
    segments = prefer_segments if prefer_segments else _select_segments(text)
    best = 0.0
    for seg in segments:
        for hyp in hyps:
            p = nli_entailment_prob(seg, hyp)
            if p > best:
                best = p
    return float(best)


def _best_margin_over_segments(text: str, hyps: List[str], prefer_segments: Optional[List[str]] = None) -> float:
    segments = prefer_segments if prefer_segments else _select_segments(text)
    best = 0.0
    for seg in segments:
        for hyp in hyps:
            sc = _nli_margin_score(seg, hyp)
            if sc > best:
                best = sc
    return float(best)


def _load_nli_params_from_file() -> NLIParams:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    param_path = os.path.join(data_dir, "nli_params.json")

    params = NLIParams()
    if os.path.exists(param_path):
        try:
            with open(param_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "kw_threshold" in cfg:
                params.kw_threshold = float(cfg["kw_threshold"])
            if "strict_threshold" in cfg:
                params.strict_threshold = float(cfg["strict_threshold"])
            if "comment_threshold" in cfg:
                params.comment_threshold = float(cfg["comment_threshold"])
        except Exception:
            pass
    return params


DEFAULT_NLI_PARAMS = _load_nli_params_from_file()


def classify_review_zero_shot(text: str, params: Optional[NLIParams] = None) -> Dict[str, Any]:
    if params is None:
        params = DEFAULT_NLI_PARAMS

    kw_threshold = params.kw_threshold
    strict_threshold = params.strict_threshold
    comment_threshold = params.comment_threshold

    text_clean = (text or "").strip()
    text_lower = text_clean.lower()
    tokens = _simple_tokens(text_clean)

    aspect_scores: Dict[str, float] = {}
    aspects: List[str] = []
    aspect_selected_by_kw: Dict[str, bool] = {}

    for label in ASPECT_LABELS_NLI:
        hyps = ASPECT_HYPOTHESES[label]
        keywords = ASPECT_KEYWORDS.get(label, [])

        segs = _segments_with_label_keywords(text_clean, keywords)

        # debug confidence tetap entailment (format output sama)
        p_yes = _best_entailment_over_segments(text_clean, hyps, prefer_segments=segs)
        aspect_scores[label] = float(p_yes)

        has_kw = any(_contains_kw(text_lower, tokens, kw) for kw in keywords) if keywords else False
        aspect_selected_by_kw[label] = bool(has_kw)

        # Perbaikan: keyword -> aspek hadir; NLI strict threshold hanya fallback.
        if has_kw:
            aspects.append(label)
        elif p_yes >= strict_threshold:
            aspects.append(label)

        _ = kw_threshold  # menjaga variabel param tetap ada (tidak mengubah output)

    if not aspects:
        aspects = []

    # misc tidak dilatih sebagai kelas; confidence misc diturunkan dari "none-of-the-above"
    # (semakin rendah semua skor aspek non-misc, semakin tinggi misc).
    if aspect_scores:
        aspect_scores["misc"] = float(1.0 - max(aspect_scores.values()))
    else:
        aspect_scores["misc"] = 0.0

    # Comment type: output debug raw tetap entailment; tetapi confidence/decision pakai margin score.
    comment_raw: Dict[str, float] = {}
    comment_margin: Dict[str, float] = {}

    for label in COMMENT_TYPE_LABELS:
        hyps = COMMENT_HYPOTHESES[label]
        segs = _segments_for_comment_label(text_clean, label)

        # debug raw entailment (format sama)
        p_entail = _best_entailment_over_segments(text_clean, hyps, prefer_segments=segs)
        comment_raw[label] = float(p_entail)

        # score keputusan yang lebih stabil
        p_margin = _best_margin_over_segments(text_clean, hyps, prefer_segments=segs)
        comment_margin[label] = float(p_margin)

    # comment_confidence tetap berupa dict float dan normalisasi (sum ~ 1) seperti sebelumnya
    total_margin = float(sum(comment_margin.values()) + 1e-8)
    comment_conf = {l: float(v) / total_margin for l, v in comment_margin.items()}

    cues = _detect_cues(text_clean)
    has_neg = cues["has_neg"]
    has_pos = cues["has_pos"]
    has_sugg_kw = cues["has_sugg"]

    # Seleksi label comment types: utamakan margin score absolut, lalu lengkapi via cues (mixed review common).
    comment_types: List[str] = []
    for lab in COMMENT_TYPE_LABELS:
        if comment_margin.get(lab, 0.0) >= comment_threshold:
            comment_types.append(lab)

    if has_neg and "complaint" not in comment_types:
        comment_types.append("complaint")
    if has_pos and "praise" not in comment_types:
        comment_types.append("praise")
    if has_sugg_kw and "suggestion" not in comment_types:
        comment_types.append("suggestion")

    if not comment_types:
        comment_types = [max(comment_margin.items(), key=lambda x: x[1])[0]]

    # Primary: gunakan margin tertinggi di antara candidate.
    primary = max(comment_types, key=lambda lab: comment_margin.get(lab, 0.0))

    # tie-break mixed: complaint cenderung lebih actionable, terutama bila ada marker kontras / cues negatif
    if "complaint" in comment_types and "praise" in comment_types:
        if has_neg and (comment_margin.get("complaint", 0.0) >= (comment_margin.get("praise", 0.0) - 0.03)):
            primary = "complaint"

    # Quality: selaraskan dengan sumber keputusan.
    # Aspek yang terdeteksi via keyword diberi "evidence floor" internal agar quality tidak jatuh hanya karena NLI entailment rendah.
    if aspects == ["misc"] or not aspects:
        # Untuk no-aspect, evidence merepresentasikan keyakinan bahwa tidak ada aspek non-misc.
        # Ini dibuat sebagai komplemen dari skor aspek non-misc tertinggi.
        if aspect_scores:
            non_misc_max = float(max(aspect_scores.get(a, 0.0) for a in ASPECT_LABELS_NLI))
            max_asp_evidence = float(1.0 - non_misc_max)
        else:
            max_asp_evidence = 0.0
    else:
        evidences: List[float] = []
        for a in aspects:
            if a == "misc":
                continue
            base = float(aspect_scores.get(a, 0.0))
            if aspect_selected_by_kw.get(a, False):
                # floor moderat: evidence topik hadir
                base = max(base, 0.70)
            evidences.append(base)
        max_asp_evidence = max(evidences) if evidences else (max(aspect_scores.values()) if aspect_scores else 0.0)

    # comment evidence pakai margin score; boost kecil bila cue konsisten, penalti kecil bila konflik keras
    primary_comment_evidence = float(comment_margin.get(primary, 0.0))
    if primary == "complaint" and has_neg:
        primary_comment_evidence = min(1.0, primary_comment_evidence + 0.05)
    elif primary == "praise" and has_pos:
        primary_comment_evidence = min(1.0, primary_comment_evidence + 0.05)
    elif primary == "suggestion" and has_sugg_kw:
        primary_comment_evidence = min(1.0, primary_comment_evidence + 0.04)

    # konflik kuat: cue negatif + positif sekaligus, turunkan sedikit agar high tidak terlalu mudah
    if has_neg and has_pos and primary_comment_evidence < 0.85:
        primary_comment_evidence = max(0.0, primary_comment_evidence - 0.03)

    nli_quality = float(min(max_asp_evidence, primary_comment_evidence))

    # Threshold quality dibuat lebih selaras dengan evidence (bukan hanya entailment mentah).
    if nli_quality >= 0.62:
        label_quality = "high"
    elif nli_quality >= 0.48:
        label_quality = "medium"
    else:
        label_quality = "low"

    return {
        "aspects": aspects,
        "comment_type": primary,
        "comment_types": comment_types,
        "label_quality": label_quality,
        "nli_quality_score": nli_quality,
        "debug": {
            "aspect_confidence": aspect_scores,
            "comment_confidence": comment_conf,
            "comment_raw_entailment": comment_raw,
        },
    }


def label_batch_with_nli(texts: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for t in texts:
        t_clean = (t or "").strip()
        if not t_clean:
            results.append({
                "text": t,
                "aspects": [],
                "comment_type": None,
                "comment_types": [],
                "label_quality": "low",
                "nli_quality_score": 0.0,
                "nli_debug": {},
            })
            continue

        lab = classify_review_zero_shot(t_clean, params=DEFAULT_NLI_PARAMS)
        results.append({
            "text": t,
            "aspects": lab["aspects"],
            "comment_type": lab["comment_type"],
            "comment_types": lab["comment_types"],
            "label_quality": lab["label_quality"],
            "nli_quality_score": lab["nli_quality_score"],
            "nli_debug": lab["debug"],
        })
    return results


if __name__ == "__main__":
    sample = "rasanya cukup enak dan segar, tapi masih lama pelayanannya. tidak ada keluhan soal tempat."
    out = classify_review_zero_shot(sample)
    print(json.dumps(out, ensure_ascii=False, indent=2))