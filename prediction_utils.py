import os
import json
import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

from training_core import read_jsonl
from config import OUTPUT_ROOT, PREDICTIONS_DIR, LATEST_PRED_PATH


def get_latest_model_run_dir() -> Optional[str]:
    log_file = os.path.join(OUTPUT_ROOT, "training_runs.jsonl")
    if not os.path.exists(log_file):
        return None

    lines: List[str] = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        run_dir = rec.get("run_dir")
        if run_dir and os.path.exists(run_dir):
            return run_dir
    return None


def log_prediction_run(pred_path: str, model_run_dir: str, num_records: int) -> None:
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    log_file = os.path.join(PREDICTIONS_DIR, "prediction_runs.jsonl")
    rec = {
        "ts": datetime.datetime.now().isoformat(),
        "pred_path": pred_path,
        "model_run_dir": model_run_dir,
        "num_records": num_records,
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_latest_predictions_path() -> Optional[str]:
    if os.path.exists(LATEST_PRED_PATH):
        return LATEST_PRED_PATH

    log_file = os.path.join(PREDICTIONS_DIR, "prediction_runs.jsonl")
    if not os.path.exists(log_file):
        return None

    lines: List[str] = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        path = rec.get("pred_path")
        if path and os.path.exists(path):
            return path
    return None

STOPWORDS_ID = {
    # fungsi umum
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "ada", "tidak", "tak",
    "atau", "untuk", "pada", "dengan", "karena", "jadi", "sebagai", "dalam",
    "oleh", "saat", "ketika", "agar", "supaya", "namun", "tapi", "tetapi",
    "juga", "lagi", "sudah", "udah", "belum", "baru", "bisa", "dapat", "harus",
    "kalau", "kalo", "jika", "bila", "misal", "misalnya",

    # pronomina / partikel
    "saya", "aku", "kami", "kita", "dia", "mereka", "kamu", "anda", "nya", "nih",
    "sih", "aja", "kok", "lah", "dong", "deh", "pun", "ya", "yah",

    # variasi negasi informal (biasanya noise untuk wordcloud; bisa kamu whitelist kalau perlu)
    "ga", "gak", "nggak", "enggak",

    # filler / ekspresi
    "banget", "bgt", "gitu", "begitu", "kayak", "kaya", "apaan", "aja", "jg",
}

def aggregate_predictions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    aspect_counts: Counter = Counter()
    type_counts: Counter = Counter()
    date_counts: Counter = Counter()
    co_counts: Dict[str, Counter] = {}

    for rec in records:
        ctype = rec.get("model_pred_comment_type")
        if not ctype:
            types = rec.get("model_pred_comment_types") or []
            if types:
                ctype = types[0]

        aspects = rec.get("model_pred_aspects") or []

        for asp in aspects:
            aspect_counts[asp] += 1
            if ctype:
                co_counts.setdefault(asp, Counter())[ctype] += 1

        if ctype:
            type_counts[ctype] += 1

        created = rec.get("created_at")
        date_str = "unknown"
        if created:
            try:
                dt = datetime.datetime.fromisoformat(created)
                date_str = dt.date().isoformat()
            except Exception:
                date_str = "unknown"
        date_counts[date_str] += 1

    return {
        "aspect_counts": dict(aspect_counts),
        "type_counts": dict(type_counts),
        "date_counts": dict(date_counts),
        "co_counts": {asp: dict(c) for asp, c in co_counts.items()},
    }


def get_training_history(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Membaca riwayat training dari OUTPUT_ROOT/training_runs.jsonl
    dan mengembalikan list run paling baru terlebih dahulu.
    """
    path = os.path.join(OUTPUT_ROOT, "training_runs.jsonl")
    if not os.path.exists(path):
        return []

    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    history: List[Dict[str, Any]] = []
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except Exception:
            continue

        # normalisasi key supaya aman di template
        rec.setdefault("run_dir", "")
        rec.setdefault("created_at", "")
        rec.setdefault("total_records", 0)
        rec.setdefault("best_hp_run_dir", None)
        rec.setdefault("best_hp_val_score", None)
        rec.setdefault("grid_search_summary_path", None)
        rec.setdefault("final_run_dir", None)

        history.append(rec)
        if len(history) >= limit:
            break

    return history

def get_prediction_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Membaca riwayat prediction dari data/predictions/prediction_runs.jsonl.
    Mengembalikan list run terbaru terlebih dahulu.
    """
    log_file = os.path.join(PREDICTIONS_DIR, "prediction_runs.jsonl")
    if not os.path.exists(log_file):
        return []

    lines: List[str] = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    history: List[Dict[str, Any]] = []
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except Exception:
            continue

        rec.setdefault("ts", rec.get("ts") or rec.get("time") or "")
        rec.setdefault("pred_path", rec.get("pred_path") or "")
        rec.setdefault("model_run_dir", rec.get("model_run_dir") or "")
        rec.setdefault("num_records", rec.get("num_records") or 0)
        history.append(rec)
        if len(history) >= limit:
            break

    return history
