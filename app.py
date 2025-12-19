import os
import json
import datetime
import shutil
from typing import List, Dict, Any, Optional
import glob, re
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from collections import Counter, defaultdict

from gemini_client import suggest_labels_for_review

from training_core import (
    load_model_for_inference,
    compute_model_confidences,
    read_jsonl,
    DEFAULT_ENCODER_NAME,
    DEFAULT_MAX_SEQ_LEN,
)

from config import (
    TEMPLATES_DIR,
    STATIC_DIR,
    ASPECT_LABELS,
    COMMENT_TYPE_LABELS,
    MIN_TRAIN_RECORDS_DEFAULT,
    CONFIDENCE_THRESHOLD_DEFAULT,
    BASE_TRAIN_PATH,
    ADDITIONAL_DATA_PATH,
    LOW_SCORE_PATH,
    PREDICTIONS_DIR,
    DELETED_REVIEW_PATH,
    LATEST_PRED_PATH,
    OUTPUT_ROOT,
    LLM_RECOMMENDATION_PATH,
    RELABELLED_LOG_PATH,
    DELETED_REVIEW_PATH,
    PRED_GLOB,
)
from io_utils import parse_input_to_records, append_jsonl

from jobs import JOBS, job_log
from training_pipeline import background_run_pipeline
from prediction_utils import (
    get_latest_model_run_dir,
    log_prediction_run,
    get_latest_predictions_path,
    aggregate_predictions,
    get_training_history,
    get_prediction_history,
)
from kbs_rules import apply_business_rules


from nli_labelling import (
    label_batch_with_nli,
    ASPECT_LABELS,
    COMMENT_TYPE_LABELS,
)

def _jaccard(a, b):
  sa = set(a or [])
  sb = set(b or [])
  if not sa and not sb:
      return 1.0
  if not sa or not sb:
      return 0.0
  return len(sa & sb) / float(len(sa | sb))

def append_jsonl_line(path: str, rec: Dict[str, Any]):
    """
    Append satu baris JSONL.
    """
    if not rec:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_jsonl(path: str, records: List[Dict[str, Any]]):
    """
    Overwrite file JSONL dengan daftar records baru.
    """
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_latest_llm_recommendations() -> Dict[str, Dict[str, Any]]:
    """
    Membaca llm_recommendation.jsonl dan mengembalikan dict:
        id -> entry terbaru (berdasarkan created_at_ts atau revision).
    """
    recs = read_jsonl(LLM_RECOMMENDATION_PATH)
    latest: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        rid = r.get("id")
        if not rid:
            continue
        prev = latest.get(rid)
        if not prev:
            latest[rid] = r
            continue
        prev_rev = prev.get("revision", 0)
        cur_rev = r.get("revision", 0)
        prev_ts = prev.get("created_at_ts", 0.0)
        cur_ts = r.get("created_at_ts", 0.0)
        # utamakan revision, lalu timestamp
        if cur_rev > prev_rev or (cur_rev == prev_rev and cur_ts >= prev_ts):
            latest[rid] = r
    return latest

def _load_low_score_records() -> List[Dict[str, Any]]:
    return read_jsonl(LOW_SCORE_PATH)


def _save_low_score_records(records: List[Dict[str, Any]]):
    save_jsonl(LOW_SCORE_PATH, records)

def _move_record_to_additional_data(
    rec_id: str,
    *,
    new_aspects: Optional[List[str]] = None,
    new_comment_types: Optional[List[str]] = None,
    move_source: str,
    from_llm: bool,
    llm_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Update label untuk satu record di low_score, lalu pindahkan ke additional_data.jsonl
    dan hapus dari low_score.jsonl. Juga tulis event ke relabelled.jsonl.
    Mengembalikan record yang sudah dipindahkan.
    """
    records = _load_low_score_records()
    idx = None
    for i, r in enumerate(records):
        if r.get("id") == rec_id:
            idx = i
            break
    if idx is None:
        raise KeyError(f"Record id {rec_id} not found in low_score.")

    rec = records[idx]

    prev_aspects = rec.get("aspects", []) or []
    prev_comment_types = rec.get("comment_types", []) or []

    if new_aspects is not None:
        rec["aspects"] = new_aspects
    if new_comment_types is not None:
        rec["comment_types"] = new_comment_types
        rec["comment_type"] = new_comment_types[0] if new_comment_types else None

    now = datetime.datetime.utcnow().isoformat()

    rec["relabel_last_saved_at"] = now
    rec["relabel_last_source"] = move_source
    rec["relabel_from_llm"] = bool(from_llm)
    rec["relabel_llm_revision"] = llm_revision
    rec["relabel_moved_to"] = "additional_data"

    # hapus dari low_score
    del records[idx]
    _save_low_score_records(records)

    # append ke additional_data.jsonl
    append_jsonl(ADDITIONAL_DATA_PATH, [rec])

    # event log ke relabelled.jsonl
    event = {
        "id": rec_id,
        "created_at": now,
        "source": move_source,
        "from_llm": bool(from_llm),
        "llm_revision": llm_revision,
        "moved_to": "additional_data",
        "prev_labels": {
            "aspects": prev_aspects,
            "comment_types": prev_comment_types,
        },
        "new_labels": {
            "aspects": rec.get("aspects", []),
            "comment_types": rec.get("comment_types", []),
        },
    }
    append_jsonl(RELABELLED_LOG_PATH, [event])

    return rec


def _update_low_score_record(
    rec_id: str,
    *,
    new_aspects: Optional[List[str]] = None,
    new_comment_types: Optional[List[str]] = None,
    relabel_source: str,
    from_llm: bool,
    llm_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Update satu record di low_score dan tulis event ke relabelled.jsonl.

    Mengembalikan record terbaru (setelah update).
    """
    records = _load_low_score_records()
    idx = None
    for i, r in enumerate(records):
        if r.get("id") == rec_id:
            idx = i
            break
    if idx is None:
        raise KeyError(f"Record id {rec_id} not found in low_score.")

    rec = records[idx]

    prev_aspects = rec.get("aspects", []) or []
    prev_comment_types = rec.get("comment_types", []) or []

    if new_aspects is not None:
        rec["aspects"] = new_aspects
    if new_comment_types is not None:
        rec["comment_types"] = new_comment_types
        # primary comment_type (satu label utama)
        rec["comment_type"] = new_comment_types[0] if new_comment_types else None

    now = datetime.datetime.utcnow().isoformat()
    rec["relabel_last_saved_at"] = now
    rec["relabel_last_source"] = relabel_source
    rec["relabel_from_llm"] = bool(from_llm)
    rec["relabel_llm_revision"] = llm_revision

    records[idx] = rec
    _save_low_score_records(records)

    # event log ke relabelled.jsonl
    event = {
        "id": rec_id,
        "created_at": now,
        "source": relabel_source,
        "from_llm": bool(from_llm),
        "llm_revision": llm_revision,
        "prev_labels": {
            "aspects": prev_aspects,
            "comment_types": prev_comment_types,
        },
        "new_labels": {
            "aspects": rec.get("aspects", []),
            "comment_types": rec.get("comment_types", []),
        },
    }
    append_jsonl_line(RELABELLED_LOG_PATH, event)

    return rec

def list_prediction_runs() -> list[dict]:
    """
    Return list of available prediction runs:
    [{ "id": "20251211214153", "path": ".../pred_20251211214153.jsonl", "ts": "2025-12-11T21:41:53" }, ...]
    """
    runs = []
    for p in sorted(glob.glob(PRED_GLOB)):
        base = os.path.basename(p)
        m = re.match(r"pred_(\d{14})\.jsonl$", base)
        if not m:
            continue
        run_id = m.group(1)
        try:
            ts = datetime.strptime(run_id, "%Y%m%d%H%M%S").isoformat()
        except Exception:
            ts = run_id
        runs.append({"id": run_id, "path": p, "ts": ts})
    # newest first
    runs = sorted(runs, key=lambda x: x["id"], reverse=True)
    return runs


def _get_created_date(rec: dict) -> str:
    # prefer created_at (ISO), fallback to created_date, fallback empty
    v = rec.get("created_at") or rec.get("created_date") or ""
    if isinstance(v, str) and len(v) >= 10:
        return v[:10]
    return ""


def _get_confidence(rec: dict) -> float | None:
    mc = rec.get("model_confidence") or {}
    v = mc.get("overall")
    try:
        if v is None:
            return None
        if isinstance(v, str) and v.startswith("overall="):
            v = v.split("=", 1)[1]
        return float(v)
    except Exception:
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

def _parse_overall_conf(rec: dict) -> Optional[float]:
    """
    Extract overall confidence from rec["model_confidence"]["overall"].
    Handles numeric, "overall=0.123" string, or missing.
    """
    mc = rec.get("model_confidence") or {}
    v = mc.get("overall")
    if v is None:
        return None
    try:
        if isinstance(v, str):
            s = v.strip()
            if s.lower().startswith("overall="):
                s = s.split("=", 1)[1].strip()
            return float(s)
        return float(v)
    except Exception:
        return None

def _get_aspects_and_types(rec: dict) -> tuple[list[str], list[str], str]:
    """
    Pull aspects and comment types from common fields.
    Returns: (aspects, comment_types, primary_type)
    """
    aspects = (
        rec.get("model_pred_aspects")
        or rec.get("nli_aspects")
        or rec.get("aspects")
        or []
    )
    if aspects is None:
        aspects = []
    if not isinstance(aspects, list):
        aspects = [str(aspects)]


    # display fallback: jika benar-benar tidak ada aspek, tampilkan misc untuk kebutuhan UI
    if not aspects:
        aspects = ["misc"]

    primary_type = (
        rec.get("model_pred_comment_type")
        or rec.get("nli_comment_type")
        or rec.get("comment_type")
        or ""
    )

    comment_types = (
        rec.get("model_pred_comment_types")
        or rec.get("nli_comment_types")
        or rec.get("comment_types")
        or []
    )
    if comment_types is None:
        comment_types = []
    if not isinstance(comment_types, list):
        comment_types = [str(comment_types)]

    # fallback: jika list kosong tapi primary ada
    if not comment_types and primary_type:
        comment_types = [primary_type]

    return aspects, comment_types, primary_type

def _extract_tokens_for_wordcloud(rec: dict) -> list[str]:
    """
    Prefer preproc.tokens; fallback regex tokenizer on text_normalized/text.
    """
    preproc = rec.get("preproc") or {}
    toks: list[str] = []
    if isinstance(preproc, dict):
        t = preproc.get("tokens")
        if isinstance(t, list) and t:
            toks = [str(x) for x in t]

    if not toks:
        txt = (rec.get("text_normalized") or rec.get("text") or "").lower()
        toks = re.findall(r"[a-zA-Z0-9_]+", txt)

    return toks

def aggregate_predictions(
    records: list[dict],
    *,
    min_conf: Optional[float] = None,
    filter_aspect: Optional[str] = None,
    filter_type: Optional[str] = None,
    keyword: Optional[str] = None,
    max_tokens_for_wc: int = 180,
) -> dict[str, Any]:
    """
    Aggregate predictions for analytics dashboard.

    Filters:
      - min_conf: keep record if overall_conf is None or >= min_conf (configurable; see below)
      - filter_aspect: keep record that contains aspect
      - filter_type: keep record that contains type (in list or primary)
      - keyword: substring match across text + labels

    Output:
      {
        "type_counts": {type: count},
        "aspect_counts": {aspect: count},
        "date_counts": {YYYY-MM-DD: count} sorted asc,
        "conf_stats": {"avg":..., "min":..., "max":..., "n":...},
        "top_tokens": [{"word":..., "count":...}, ...]   # for wordcloud
      }
    """
    type_counts = Counter()
    aspect_counts = Counter()
    date_counts = Counter()
    conf_values: list[float] = []
    token_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    heatmap = {}  # {aspect: {type: count}}


    q = (keyword or "").strip().lower()
    fa = (filter_aspect or "").strip()
    ft = (filter_type or "").strip()

    for r in records:
        if not isinstance(r, dict):
            continue

        text = (r.get("text") or r.get("text_normalized") or "").strip()
        aspects, comment_types, primary_type = _get_aspects_and_types(r)
        overall_conf = _parse_overall_conf(r)

        # --- filtering ---
        # min_conf policy:
        # - if overall_conf is None, we KEEP it (so you can still analyze legacy/no-conf runs)
        # - if you want strict, change condition to: if overall_conf is None or overall_conf < min_conf: continue
        if min_conf is not None and overall_conf is not None and overall_conf < float(min_conf):
            continue

        if fa and fa not in aspects:
            continue

        if ft and (ft not in comment_types) and (ft != primary_type):
            continue

        if q:
            hay = (text + " " + " ".join(aspects) + " " + " ".join(comment_types)).lower()
            if q not in hay:
                continue

        # --- counts ---
        d = _get_created_date(r)
        if d:
            date_counts[d] += 1

        for a in aspects:
            if a:
                aspect_counts[str(a)] += 1

        # prefer primary type for distribution; fallback first of list
        if primary_type:
            type_counts[str(primary_type)] += 1
        elif comment_types:
            type_counts[str(comment_types[0])] += 1

        if overall_conf is not None:
            conf_values.append(float(overall_conf))

        # --- token counts for wordcloud ---
        toks = _extract_tokens_for_wordcloud(r)
        for t in toks:
            t = str(t).lower().strip()
            if not t:
                continue
            # drop very short tokens
            if len(t) < 3:
                continue
            # drop pure digits
            if t.isdigit():
                continue
            # drop stopwords
            if t in STOPWORDS_ID:
                continue
            # drop tokens that are basically punctuation-ish underscores
            if re.fullmatch(r"_+", t):
                continue
            token_counts[t] += 1

                # heatmap aspect x type (pakai primary_type sebagai axis type)
        t_for_heat = primary_type or (comment_types[0] if comment_types else "")
        if t_for_heat:
            for a in aspects:
                if not a:
                    continue
                if a not in heatmap:
                    heatmap[a] = {}
                heatmap[a][t_for_heat] = int(heatmap[a].get(t_for_heat, 0)) + 1

        # bigrams / trigrams (prefer preproc)
        preproc = r.get("preproc") or {}
        if isinstance(preproc, dict):
            bgs = preproc.get("bigrams") or []
            tgs = preproc.get("trigrams") or []
            if isinstance(bgs, list):
                for bg in bgs:
                    bg = str(bg).lower().strip()
                    if len(bg) >= 3 and bg not in STOPWORDS_ID and not bg.isdigit():
                        bigram_counts[bg] += 1
            if isinstance(tgs, list):
                for tg in tgs:
                    tg = str(tg).lower().strip()
                    if len(tg) >= 3 and tg not in STOPWORDS_ID and not tg.isdigit():
                        trigram_counts[tg] += 1


    # sort date counts ascending
    date_counts_sorted = dict(sorted(date_counts.items(), key=lambda kv: kv[0]))

    # confidence stats
    conf_avg = (sum(conf_values) / len(conf_values)) if conf_values else None
    conf_min = min(conf_values) if conf_values else None
    conf_max = max(conf_values) if conf_values else None

    top_tokens = [{"word": w, "count": int(c)} for (w, c) in token_counts.most_common(max_tokens_for_wc)]

    return {
        "type_counts": dict(type_counts),
        "aspect_counts": dict(aspect_counts),
        "date_counts": date_counts_sorted,
        "conf_stats": {"avg": conf_avg, "min": conf_min, "max": conf_max, "n": len(conf_values)},
        "top_tokens": top_tokens,
        "top_bigrams": [{"word": w, "count": int(c)} for (w, c) in bigram_counts.most_common(max_tokens_for_wc)],
        "top_trigrams": [{"word": w, "count": int(c)} for (w, c) in trigram_counts.most_common(max_tokens_for_wc)],
        "heatmap": heatmap,

    }

app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    history = get_training_history(limit=20)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "aspect_labels": ASPECT_LABELS,
            "comment_type_labels": COMMENT_TYPE_LABELS,
            "min_train_records_default": MIN_TRAIN_RECORDS_DEFAULT,
            "confidence_threshold_default": CONFIDENCE_THRESHOLD_DEFAULT,
            "training_history": history,
        },
    )


@app.post("/run_pipeline", response_class=JSONResponse)
async def run_pipeline(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    csv_file: Optional[UploadFile] = File(None),
    text_input: Optional[str] = Form(None),
    raw_text: Optional[str] = Form(None),
    # parameter lama dipertahankan untuk kompatibilitas, tetapi diabaikan
    run_training: Optional[bool] = Form(True),
    confidence_threshold: Optional[float] = Form(None),
    min_train_records: Optional[int] = Form(None),
    num_epochs: Optional[int] = Form(None),
    batch_size: Optional[int] = Form(None),
    lr: Optional[float] = Form(None),
    warmup_ratio: Optional[float] = Form(None),
    dropout: Optional[float] = Form(None),
):
    ts_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    job_id = ts_id
    JOBS[job_id] = {"status": "pending", "logs": [], "result": None}
    job_log(job_id, f"Job {job_id} created.")

    file_bytes = None
    if file is not None:
        file_bytes = await file.read()
    elif csv_file is not None:
        file_bytes = await csv_file.read()

    effective_text = text_input or raw_text

    records = parse_input_to_records(file_bytes, effective_text, source="upload", ts_id=ts_id)
    job_log(job_id, f"Parsed {len(records)} input records.")

    if not records:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["result"] = {"error": "No valid input records."}
        job_log(job_id, "No valid input records.")
        return {"job_id": job_id, "status": "error"}

    # hitung total kandidat training (base + additional + batch) SEBELUM NLI
    base_recs = read_jsonl(BASE_TRAIN_PATH) if os.path.exists(BASE_TRAIN_PATH) else []
    add_recs = read_jsonl(ADDITIONAL_DATA_PATH) if os.path.exists(ADDITIONAL_DATA_PATH) else []
    existing_count = len(base_recs) + len(add_recs)
    total_candidate = existing_count + len(records)

    job_log(job_id, f"Existing training records (base+additional): {existing_count}")
    job_log(job_id, f"New batch records: {len(records)}")
    job_log(job_id, f"Total candidate training records: {total_candidate}")

    if total_candidate < MIN_TRAIN_RECORDS_DEFAULT:
        # batalkan SEBELUM NLI
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["result"] = {
            "error": (
                f"Total training records {total_candidate} < minimal "
                f"{MIN_TRAIN_RECORDS_DEFAULT}. Job dibatalkan sebelum NLI."
            ),
            "total_candidate_records": total_candidate,
            "existing_training_records": existing_count,
            "new_batch_records": len(records),
        }
        job_log(
            job_id,
            f"Training cancelled: total training records {total_candidate} "
            f"< minimum {MIN_TRAIN_RECORDS_DEFAULT}.",
        )
        return {"job_id": job_id, "status": "error"}

    # tulis batch mentah, lanjut ke background pipeline
    input_jsonl_path = os.path.join("data", "input", f"train_{ts_id}.jsonl")
    os.makedirs(os.path.dirname(input_jsonl_path), exist_ok=True)
    with open(input_jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # override: training selalu dijalankan, min record dan confidence default dari config
    train_cfg_overrides = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "dropout": dropout,
    }

    background_tasks.add_task(
        background_run_pipeline,
        job_id=job_id,
        input_jsonl_path=input_jsonl_path,
        confidence_threshold=float(CONFIDENCE_THRESHOLD_DEFAULT),
        run_training_flag=True,
        min_train_records=int(MIN_TRAIN_RECORDS_DEFAULT),
        train_cfg_overrides=train_cfg_overrides,
    )

    return {"job_id": job_id, "status": "started"}


@app.get("/job_status/{job_id}", response_class=JSONResponse)
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"status": "not_found", "logs": [], "result": None}
    return {"status": job["status"], "logs": job["logs"], "result": job["result"]}


@app.get("/relabel_low_score", response_class=HTMLResponse)
async def relabel_low_score(
    request: Request,
    show: int = 5,
    order_by: str = "created_at",
    order_dir: str = "desc",
    label_quality: Optional[str] = None,
    aspect: Optional[str] = None,
    comment_type: Optional[str] = None,
    msg: Optional[str] = None,
):
    records = read_jsonl(LOW_SCORE_PATH) if os.path.exists(LOW_SCORE_PATH) else []

    # current_aspects / current_comment_types + overall_conf
    for rec in records:
        current_aspects = (
            rec.get("nli_aspects")
            or rec.get("model_pred_aspects")
            or rec.get("aspects")
            or []
        )
        if current_aspects is None:
            current_aspects = []
        rec["current_aspects"] = current_aspects

        cts = (
            rec.get("nli_comment_types")
            or rec.get("model_pred_comment_types")
            or rec.get("comment_types")
            or []
        )
        if not cts:
            primary = rec.get("nli_comment_type")
            if primary:
                cts = [primary]
        rec["current_comment_types"] = cts or []

        mc = rec.get("model_confidence") or {}
        overall = mc.get("overall")
        try:
            rec["overall_conf"] = float(overall) if overall is not None else None
        except Exception:
            rec["overall_conf"] = None

    # filter
    if label_quality:
        records = [
            r for r in records
            if (r.get("label_quality") or "") == label_quality
        ]

    if aspect:
        records = [
            r for r in records
            if aspect in (r.get("current_aspects") or [])
        ]

    if comment_type:
        records = [
            r for r in records
            if comment_type in (r.get("current_comment_types") or [])
        ]

    # sort
    def sort_key(rec):
        if order_by == "confidence":
            return (
                rec.get("overall_conf")
                if rec.get("overall_conf") is not None
                else -1e9
            )
        if order_by == "label":
            return rec.get("label_quality") or ""
        created = rec.get("created_at")
        return created or ""

    reverse = (order_dir == "desc")
    records_sorted = sorted(records, key=sort_key, reverse=reverse)

    # limit
    if show is None or show <= 0:
        show = 50
    records_limited = records_sorted[:show]

    # rekomendasi LLM terbaru per id
    llm_latest = load_latest_llm_recommendations()

    return templates.TemplateResponse(
        "relabel_low_score.html",
        {
            "request": request,
            "records": records_limited,
            "aspect_labels": ASPECT_LABELS,
            "comment_type_labels": COMMENT_TYPE_LABELS,
            "show": show,
            "order_by": order_by,
            "order_dir": order_dir,
            "filter_label_quality": label_quality or "",
            "filter_aspect": aspect or "",
            "filter_comment_type": comment_type or "",
            "message": msg,
            "llm_map": llm_latest,
        },
    )

# ---------------------------------------------------------
# API: LLM recommendation dan relabel actions
# ---------------------------------------------------------

@app.post("/api/relabel/llm_refresh", response_class=JSONResponse)
async def relabel_llm_refresh(request: Request):
    """
    Meminta rekomendasi LLM (Gemini) untuk satu id di low_score.
    """
    payload = await request.json()
    rec_id = payload.get("id")
    if not rec_id:
        return JSONResponse({"status": "error", "error": "Missing id"}, status_code=400)

    records = _load_low_score_records()
    rec = next((r for r in records if r.get("id") == rec_id), None)
    if not rec:
        return JSONResponse(
            {"status": "error", "error": f"Record id {rec_id} not found"},
            status_code=404,
        )

    text = (rec.get("text") or rec.get("text_normalized") or rec.get("review") or "").strip()
    if not text:
        return JSONResponse(
            {"status": "error", "error": "Record has empty text"},
            status_code=400,
        )

    nli_aspects = rec.get("aspects") or rec.get("nli_aspects") or []
    nli_comment_types = rec.get("comment_types") or rec.get("nli_comment_types") or []
    nli_ct_single = rec.get("comment_type") or rec.get("nli_comment_type")
    if nli_ct_single and nli_ct_single not in nli_comment_types:
        nli_comment_types = nli_comment_types + [nli_ct_single]

    model_aspects = rec.get("model_pred_aspects") or []
    model_comment_types = rec.get("model_pred_comment_types") or []

    try:
        suggestion = suggest_labels_for_review(
            review_id=rec_id,
            text=text,
            aspect_labels=ASPECT_LABELS,
            comment_type_labels=COMMENT_TYPE_LABELS,
            language="id",
            text_hash=None,
            nli_aspects=nli_aspects,
            nli_comment_types=nli_comment_types,
            model_aspects=model_aspects,
            model_comment_types=model_comment_types,
        )
    except Exception as e:
        return JSONResponse(
            {
                "status": "error",
                "error": f"LLM error: {str(e)}"
            },
            status_code=503
        )


    latest_map = load_latest_llm_recommendations()
    prev = latest_map.get(rec_id)
    revision = (prev.get("revision", 0) + 1) if prev else 1

    llm_entry = {
        "id": rec_id,
        "aspects": suggestion.aspects,
        "comment_types": suggestion.comment_types,
        "reason": suggestion.reason,
        "llm_model": suggestion.llm_model,
        "created_at_ts": suggestion.created_at_ts,
        "input_version": suggestion.input_version,
        "status": "suggested",
        "revision": revision,
    }
    append_jsonl(LLM_RECOMMENDATION_PATH, [llm_entry])

    return JSONResponse({"status": "ok", "suggestion": llm_entry})


@app.post("/api/relabel/llm_batch_generate", response_class=JSONResponse)
async def relabel_llm_batch_generate(request: Request):
    """
    Batch generate rekomendasi LLM untuk subset id (row yang sedang ditampilkan).
    Body JSON: { "ids": ["id1", "id2", ...] }
    """
    payload = await request.json()
    ids = payload.get("ids") or []
    if not ids:
        return JSONResponse({"status": "error", "error": "Missing ids"}, status_code=400)
        # batasi jumlah id per batch untuk menghindari spam ke API LLM
    MAX_BATCH_IDS = 10  # bisa kamu naikkan kalau kuota aman
    if len(ids) > MAX_BATCH_IDS:
        return JSONResponse(
            {
                "status": "error",
                "error": f"Terlalu banyak row dalam satu batch (maks {MAX_BATCH_IDS}). "
                         f"Perkecil jumlah data yang ditampilkan atau gunakan batch lebih kecil."
            },
            status_code=429,
        )

    records = _load_low_score_records()
    rec_by_id = {r.get("id"): r for r in records if r.get("id")}
    latest_map = load_latest_llm_recommendations()

    results = []
    processed = 0
    skipped_not_found = 0
    skipped_empty_text = 0

    for rec_id in ids:
        rec = rec_by_id.get(rec_id)
        if not rec:
            skipped_not_found += 1
            continue

        text = (rec.get("text") or rec.get("text_normalized") or rec.get("review") or "").strip()
        if not text:
            skipped_empty_text += 1
            continue

        nli_aspects = rec.get("aspects") or rec.get("nli_aspects") or []
        nli_comment_types = rec.get("comment_types") or rec.get("nli_comment_types") or []
        nli_ct_single = rec.get("comment_type") or rec.get("nli_comment_type")
        if nli_ct_single and nli_ct_single not in nli_comment_types:
            nli_comment_types = nli_comment_types + [nli_ct_single]

        model_aspects = rec.get("model_pred_aspects") or []
        model_comment_types = rec.get("model_pred_comment_types") or []

        try:
            suggestion = suggest_labels_for_review(
                review_id=rec_id,
                text=text,
                aspect_labels=ASPECT_LABELS,
                comment_type_labels=COMMENT_TYPE_LABELS,
                language="id",
                text_hash=None,
                nli_aspects=nli_aspects,
                nli_comment_types=nli_comment_types,
                model_aspects=model_aspects,
                model_comment_types=model_comment_types,
            )
        except Exception as e:
            # skip row ini, jangan crash seluruh batch
            # errors.append(f"{rec_id}: {str(e)}")
            continue


        prev = latest_map.get(rec_id)
        revision = (prev.get("revision", 0) + 1) if prev else 1

        llm_entry = {
            "id": rec_id,
            "aspects": suggestion.aspects,
            "comment_types": suggestion.comment_types,
            "reason": suggestion.reason,
            "llm_model": suggestion.llm_model,
            "created_at_ts": suggestion.created_at_ts,
            "input_version": suggestion.input_version,
            "status": "suggested",
            "revision": revision,
        }
        append_jsonl(LLM_RECOMMENDATION_PATH, [llm_entry])
        results.append(llm_entry)
        processed += 1

    return JSONResponse(
        {
            "status": "ok",
            "processed": processed,
            "skipped_not_found": skipped_not_found,
            "skipped_empty_text": skipped_empty_text,
            "results": results,
        }
    )


@app.post("/api/relabel/row_save", response_class=JSONResponse)
async def relabel_row_save(request: Request):
    """
    Save label aspek & comment_type untuk satu row,
    lalu memindahkan record tersebut ke additional_data.jsonl
    (hilang dari low_score).
    """
    payload = await request.json()
    rec_id = payload.get("id")
    if not rec_id:
        return JSONResponse({"status": "error", "error": "Missing id"}, status_code=400)

    aspects = payload.get("aspects") or []
    comment_types = payload.get("comment_types") or []
    source = payload.get("source") or "manual_form"

    # filter label hanya yang valid
    aspects = [a for a in aspects if a in ASPECT_LABELS]
    comment_types = [c for c in comment_types if c in COMMENT_TYPE_LABELS]

    # cek apakah label sama persis dengan rekomendasi LLM
    llm_map = load_latest_llm_recommendations()
    llm = llm_map.get(rec_id)
    from_llm = False
    llm_revision = None
    if llm:
        if set(aspects) == set(llm.get("aspects", [])) and set(comment_types) == set(
            llm.get("comment_types", [])
        ):
            from_llm = True
            llm_revision = llm.get("revision")

    try:
        rec = _move_record_to_additional_data(
            rec_id,
            new_aspects=aspects,
            new_comment_types=comment_types,
            move_source=source,
            from_llm=from_llm,
            llm_revision=llm_revision,
        )
    except KeyError as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=404)

    return JSONResponse({"status": "ok", "record": rec, "moved": True})


@app.post("/api/relabel/llm_apply_row", response_class=JSONResponse)
async def relabel_llm_apply_row(request: Request):
    """
    Apply rekomendasi LLM terbaru untuk satu id, langsung commit (save).
    """
    payload = await request.json()
    rec_id = payload.get("id")
    if not rec_id:
        return JSONResponse({"status": "error", "error": "Missing id"}, status_code=400)

    llm_map = load_latest_llm_recommendations()
    llm = llm_map.get(rec_id)
    if not llm:
        return JSONResponse(
            {"status": "error", "error": f"No LLM recommendation for id {rec_id}"},
            status_code=404,
        )

    aspects = [a for a in llm.get("aspects", []) if a in ASPECT_LABELS]
    comment_types = [c for c in llm.get("comment_types", []) if c in COMMENT_TYPE_LABELS]

    try:
        rec = _update_low_score_record(
            rec_id,
            new_aspects=aspects,
            new_comment_types=comment_types,
            relabel_source="llm_apply_row",
            from_llm=True,
            llm_revision=llm.get("revision"),
        )
    except KeyError as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=404)

    return JSONResponse({"status": "ok", "record": rec, "llm": llm})


@app.post("/api/relabel/llm_batch_apply", response_class=JSONResponse)
async def relabel_llm_batch_apply(request: Request):
    """
    Batch apply LLM recommendation untuk subset id (row yang sedang ditampilkan).
    Body JSON: { "ids": ["id1", "id2", ...] }
    """
    payload = await request.json()
    ids = payload.get("ids") or []
    if not ids:
        return JSONResponse({"status": "error", "error": "Missing ids"}, status_code=400)

    llm_map = load_latest_llm_recommendations()
    applied = 0
    skipped_no_llm = 0
    errors: List[str] = []

    for rec_id in ids:
        llm = llm_map.get(rec_id)
        if not llm:
            skipped_no_llm += 1
            continue

        aspects = [a for a in llm.get("aspects", []) if a in ASPECT_LABELS]
        comment_types = [c for c in llm.get("comment_types", []) if c in COMMENT_TYPE_LABELS]

        try:
            _update_low_score_record(
                rec_id,
                new_aspects=aspects,
                new_comment_types=comment_types,
                relabel_source="llm_apply_all",
                from_llm=True,
                llm_revision=llm.get("revision"),
            )
            applied += 1
        except KeyError as e:
            errors.append(str(e))

    return JSONResponse(
        {
            "status": "ok",
            "applied": applied,
            "skipped_no_llm": skipped_no_llm,
            "errors": errors,
        }
    )


@app.post("/api/relabel/row_delete", response_class=JSONResponse)
async def relabel_row_delete(request: Request):
    """
    Menghapus satu record dari low_score (dipindah ke deleted_review.jsonl).
    """
    payload = await request.json()
    rec_id = payload.get("id")
    if not rec_id:
        return JSONResponse({"status": "error", "error": "Missing id"}, status_code=400)

    records = _load_low_score_records()
    new_records: List[Dict[str, Any]] = []
    deleted: Optional[Dict[str, Any]] = None
    for r in records:
        if r.get("id") == rec_id:
            deleted = r
        else:
            new_records.append(r)

    if not deleted:
        return JSONResponse(
            {"status": "error", "error": f"Record id {rec_id} not found"},
            status_code=404,
        )

    _save_low_score_records(new_records)

    now = datetime.datetime.utcnow().isoformat()
    deleted["deleted_at"] = now
    append_jsonl(DELETED_REVIEW_PATH, [deleted])

    # event log juga ke relabelled.jsonl
    event = {
        "id": rec_id,
        "created_at": now,
        "source": "delete_review",
        "from_llm": False,
        "llm_revision": None,
        "prev_labels": {
            "aspects": deleted.get("aspects", []),
            "comment_types": deleted.get("comment_types", []),
        },
        "new_labels": {
            "aspects": [],
            "comment_types": [],
        },
    }
    append_jsonl(RELABELLED_LOG_PATH, [event])

    return JSONResponse({"status": "ok"})


@app.post("/relabel_low_score_submit")
async def relabel_low_score_submit(request: Request):
    """
    Memproses perpindahan batch dari low_score ke additional_data
    hanya untuk row yang dicentang (selected_ids).
    """
    form = await request.form()

    selected_ids = form.getlist("selected_ids")
    if not selected_ids:
        # tidak ada yang dipilih
        url = request.url_for("relabel_low_score")
        url = str(url) + "?msg=" + "Tidak ada row yang dipilih untuk dipindah."
        return RedirectResponse(url, status_code=303)

    moved = 0
    not_found = 0
    errors: List[str] = []

    for rec_id in selected_ids:
        aspects = form.getlist(f"aspects_{rec_id}")
        comment_types = form.getlist(f"comment_types_{rec_id}")

        aspects = [a for a in aspects if a in ASPECT_LABELS]
        comment_types = [c for c in comment_types if c in COMMENT_TYPE_LABELS]

        # cek apakah sama persis dengan rekomendasi LLM
        llm_map = load_latest_llm_recommendations()
        llm = llm_map.get(rec_id)
        from_llm = False
        llm_revision = None
        if llm:
            if set(aspects) == set(llm.get("aspects", [])) and set(comment_types) == set(
                llm.get("comment_types", [])
            ):
                from_llm = True
                llm_revision = llm.get("revision")

        try:
            _move_record_to_additional_data(
                rec_id,
                new_aspects=aspects,
                new_comment_types=comment_types,
                move_source="batch_form",
                from_llm=from_llm,
                llm_revision=llm_revision,
            )
            moved += 1
        except KeyError:
            not_found += 1
        except Exception as e:
            errors.append(f"{rec_id}: {e}")

    msg_parts = [f"Moved {moved} row(s) to additional_data."]
    if not_found:
        msg_parts.append(f"{not_found} id tidak ditemukan.")
    if errors:
        msg_parts.append("Errors: " + "; ".join(errors[:3]))
    msg = " ".join(msg_parts)

    url = request.url_for("relabel_low_score")
    url = str(url) + "?msg=" + msg
    return RedirectResponse(url, status_code=303)

@app.post("/delete_review", response_class=JSONResponse)
async def delete_review(request: Request):
    data = await request.json()
    rid = data.get("id")
    if not rid:
        raise HTTPException(status_code=400, detail="Missing id")

    records = read_jsonl(LOW_SCORE_PATH) if os.path.exists(LOW_SCORE_PATH) else []
    remaining: List[Dict[str, Any]] = []
    deleted: List[Dict[str, Any]] = []

    for rec in records:
        if rec.get("id") == rid:
            rec["deleted_at"] = datetime.datetime.now().isoformat()
            deleted.append(rec)
        else:
            remaining.append(rec)

    if not deleted:
        return JSONResponse({"status": "not_found"}, status_code=404)

    append_jsonl(DELETED_REVIEW_PATH, deleted)

    with open(LOW_SCORE_PATH, "w", encoding="utf-8") as f:
        for r in remaining:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"status": "ok", "deleted_count": len(deleted)}



@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request):
    latest_run = get_latest_model_run_dir()
    model_ready = latest_run is not None
    prediction_history = get_prediction_history(limit=50)
    return templates.TemplateResponse(
        "prediction.html",
        {
            "request": request,
            "model_ready": model_ready,
            "aspect_labels": ASPECT_LABELS,
            "comment_type_labels": COMMENT_TYPE_LABELS,
            "prediction_history": prediction_history,
        },
    )


@app.post("/run_prediction", response_class=HTMLResponse)
async def run_prediction(
    request: Request,
    file: Optional[UploadFile] = File(None),
    csv_file: Optional[UploadFile] = File(None),
    text_input: Optional[str] = Form(None),
    raw_text: Optional[str] = Form(None),
):
    ts_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # 1) Ambil input (file atau textarea)
    if file is not None:
        file_bytes = await file.read()
    elif csv_file is not None:
        file_bytes = await csv_file.read()
    else:
        file_bytes = b""

    effective_text = text_input or raw_text
    records = parse_input_to_records(file_bytes, effective_text, source="predict", ts_id=ts_id)

    # Deteksi apakah ini request AJAX (JS di prediction.html)
    is_ajax = request.headers.get("x-requested-with") == "XMLHttpRequest"

    if not records:
        if is_ajax:
            return JSONResponse(
                {
                    "status": "error",
                    "error": "Tidak ada record valid di input.",
                    "total": 0,
                    "records": [],
                    "agg": None,
                    "pred_path": None,
                },
                status_code=400,
            )
        return templates.TemplateResponse(
            "prediction_result.html",
            {
                "request": request,
                "error": "Tidak ada record valid di input.",
                "total": 0,
                "records": [],
                "agg": None,
                "pred_path": None,
            },
        )

    latest_run_dir = get_latest_model_run_dir()
    if not latest_run_dir:
        if is_ajax:
            return JSONResponse(
                {
                    "status": "error",
                    "error": "Belum ada model terlatih. Jalankan training terlebih dahulu.",
                    "total": len(records),
                    "records": [],
                    "agg": None,
                    "pred_path": None,
                },
                status_code=400,
            )
        return templates.TemplateResponse(
            "prediction_result.html",
            {
                "request": request,
                "error": "Belum ada model terlatih. Jalankan training terlebih dahulu.",
                "total": len(records),
                "records": [],
                "agg": None,
                "pred_path": None,
            },
        )

    # 2) NLI untuk prediction (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)
    texts_for_nli = [
        (r.get("text_normalized") or r.get("text") or "")
        for r in records
    ]
    nli_results = label_batch_with_nli(texts_for_nli)

    for rec, nli in zip(records, nli_results):
        rec["nli_aspects"] = nli.get("aspects", [])
        rec["nli_comment_type"] = nli.get("comment_type")
        rec["nli_comment_types"] = nli.get("comment_types", [])
        rec["label_quality"] = nli.get("label_quality", "unknown")
        rec["nli_quality_score"] = nli.get("nli_quality_score", 0.0)
        rec["nli_debug"] = nli.get("nli_debug") or nli.get("debug", {})

    model, tokenizer, cfg_model, thresholds, type_thresholds = load_model_for_inference(latest_run_dir)
    run_id = os.path.basename(latest_run_dir)
    encoder_name = cfg_model.get("encoder_name", DEFAULT_ENCODER_NAME)
    max_len = int(cfg_model.get("max_len", DEFAULT_MAX_SEQ_LEN))

    # 4) IndoBERT multi-task + confidence detail
    scored_records = compute_model_confidences(
        records=records,
        model=model,
        tokenizer=tokenizer,
        aspect_labels=cfg_model.get("aspect_labels", ASPECT_LABELS),
        comment_type_labels=COMMENT_TYPE_LABELS,
        run_id=run_id,
        encoder_name=encoder_name,
        max_len=max_len,
        aspect_thresholds=thresholds,
        type_thresholds=type_thresholds,
    )

    # 5) Tambahkan kategori agreement berbasis Jaccard
    def _jaccard(a, b):
        sa, sb = set(a or []), set(b or [])
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / float(len(sa | sb))

    for rec in scored_records:
        nli_aspects = rec.get("nli_aspects") or []
        model_aspects = rec.get("model_pred_aspects") or []

        nli_cts = rec.get("nli_comment_types") or []
        if not nli_cts and rec.get("nli_comment_type"):
            nli_cts = [rec["nli_comment_type"]]

        model_cts = rec.get("model_pred_comment_types") or []
        if not model_cts and rec.get("model_pred_comment_type"):
            model_cts = [rec["model_pred_comment_type"]]

        mc = rec.get("model_confidence") or {}

        # jika belum ada Jaccard di dalam model_confidence, hitung sekarang
        aspect_j = mc.get("aspect_jaccard")
        if aspect_j is None:
            aspect_j = _jaccard(nli_aspects, model_aspects)
        type_j = mc.get("comment_type_jaccard")
        if type_j is None:
            type_j = _jaccard(nli_cts, model_cts)

        # kategori agreement
        def _agree_label(j):
            if j is None:
                return "unknown"
            if j >= 0.999:
                return "full"
            if j <= 0.001:
                return "none"
            return "partial"

        aspect_agree = _agree_label(aspect_j)
        type_agree = _agree_label(type_j)

        if aspect_agree == "full" and type_agree == "full":
            overall_agree = "full"
        elif aspect_agree == "none" and type_agree == "none":
            overall_agree = "none"
        else:
            overall_agree = "partial"

        mc["aspect_jaccard"] = float(aspect_j)
        mc["comment_type_jaccard"] = float(type_j)

        rec["model_confidence"] = mc
        rec["agreement"] = {
            "aspect": aspect_agree,
            "comment_type": type_agree,
            "overall": overall_agree,
        }

    # 6) Simpan hasil prediksi ke file seperti sebelumnya
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    pred_path = os.path.join(PREDICTIONS_DIR, f"pred_{ts_id}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for rec in scored_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    shutil.copy(pred_path, LATEST_PRED_PATH)
    log_prediction_run(pred_path, latest_run_dir, num_records=len(scored_records))

    agg = aggregate_predictions(scored_records)
    preview_records = scored_records[:100]

    # 7) Response: JSON untuk AJAX, HTML untuk fallback lama
    if is_ajax:
        return JSONResponse(
            {
                "status": "ok",
                "error": None,
                "total": len(scored_records),
                "records": scored_records,
                "agg": agg,
                "pred_path": pred_path,
                "model_run_dir": latest_run_dir,
            }
        )

    # fallback: halaman lama prediction_result.html
    return templates.TemplateResponse(
        "prediction_result.html",
        {
            "request": request,
            "error": None,
            "total": len(scored_records),
            "records": preview_records,
            "agg": agg,
            "pred_path": pred_path,
        },
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    run: str = "all",
    min_conf: float | None = None,
    aspect: str | None = None,
    comment_type: str | None = None,
    q: str | None = None,
    compare_a: str | None = None,
    compare_b: str | None = None,
):
    # ------------------------------------------------------------
    # 1) Discover prediction runs
    # ------------------------------------------------------------
    runs = []
    if os.path.exists(PREDICTIONS_DIR):
        for fn in sorted(os.listdir(PREDICTIONS_DIR)):
            if not fn.startswith("pred_") or not fn.endswith(".jsonl"):
                continue
            ts = fn.replace("pred_", "").replace(".jsonl", "")
            runs.append({
                "id": ts,
                "ts": ts,
                "path": os.path.join(PREDICTIONS_DIR, fn),
            })

    has_data = len(runs) > 0

    # ------------------------------------------------------------
    # 2) Load records for selected run(s)
    # ------------------------------------------------------------
    records: list[dict] = []
    selected_run = run

    if has_data:
        if run == "all":
            for r in runs:
                if os.path.exists(r["path"]):
                    records.extend(read_jsonl(r["path"]))
        else:
            matched = next((r for r in runs if r["id"] == run), None)
            if matched and os.path.exists(matched["path"]):
                records = read_jsonl(matched["path"])
            else:
                records = []

    records_total = len(records)

    # ------------------------------------------------------------
    # 3) Aggregate for main dashboard
    # ------------------------------------------------------------
    agg = aggregate_predictions(
        records,
        min_conf=min_conf,
        filter_aspect=aspect,
        filter_type=comment_type,
        keyword=q,
    ) if has_data else {
        "type_counts": {},
        "aspect_counts": {},
        "date_counts": {},
        "conf_stats": {"avg": None, "min": None, "max": None, "n": 0},
        "top_tokens": [],
        "top_bigrams": [],
        "top_trigrams": [],
        "heatmap": {},
    }

    # ------------------------------------------------------------
    # 4) Available filter values
    # ------------------------------------------------------------
    available_aspects = sorted(agg["aspect_counts"].keys())
    available_types = sorted(agg["type_counts"].keys())

    # ------------------------------------------------------------
    # 5) Sample records (for table exploration)
    # ------------------------------------------------------------
    sample_records = records[:200]

    # ------------------------------------------------------------
    # 6) Compare run A vs B
    # ------------------------------------------------------------
    compare = None
    if compare_a and compare_b and compare_a != compare_b:
        run_map = {r["id"]: r["path"] for r in runs}
        pa = run_map.get(compare_a)
        pb = run_map.get(compare_b)

        if pa and pb and os.path.exists(pa) and os.path.exists(pb):
            ra = read_jsonl(pa)
            rb = read_jsonl(pb)

            agg_a = aggregate_predictions(
                ra,
                min_conf=min_conf,
                filter_aspect=aspect,
                filter_type=comment_type,
                keyword=q,
            )
            agg_b = aggregate_predictions(
                rb,
                min_conf=min_conf,
                filter_aspect=aspect,
                filter_type=comment_type,
                keyword=q,
            )

            def delta_counts(a: dict, b: dict) -> dict:
                keys = set(a.keys()) | set(b.keys())
                return {k: int(b.get(k, 0)) - int(a.get(k, 0)) for k in keys}

            compare = {
                "a": {"id": compare_a, "agg": agg_a},
                "b": {"id": compare_b, "agg": agg_b},
                "delta_aspects": delta_counts(
                    agg_a["aspect_counts"], agg_b["aspect_counts"]
                ),
                "delta_types": delta_counts(
                    agg_a["type_counts"], agg_b["type_counts"]
                ),
            }

    # ------------------------------------------------------------
    # 7) Render template
    # ------------------------------------------------------------
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "has_data": has_data,
            "runs": runs,
            "selected_run": selected_run,
            "records_total": records_total,

            "agg": agg,
            "sample_records": sample_records,

            "available_aspects": available_aspects,
            "available_types": available_types,

            "filter_min_conf": min_conf if min_conf is not None else "",
            "filter_aspect": aspect or "",
            "filter_type": comment_type or "",
            "filter_q": q or "",

            "compare": compare,
            "compare_a_selected": compare_a or "",
            "compare_b_selected": compare_b or "",
        },
    )



@app.get("/kbs", response_class=HTMLResponse)
async def kbs_page(
    request: Request,
    run: str = "all",
    min_conf: float | None = None,
    priority: str | None = None,
    owner_team: str | None = None,
    tag: str | None = None,
    q: str | None = None,
):
    """
    Hybrid KBS view (business-rule layer) built purely from existing prediction JSONL.
    Does not modify or overwrite existing prediction outputs.
    """
    # 1) Discover prediction runs (same file pattern as dashboard)
    def _count_jsonl_lines(path: str) -> int:
        try:
            c = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c += 1
            return c
        except Exception:
            return 0

    runs = []
    if os.path.exists(PREDICTIONS_DIR):
        for fn in sorted(os.listdir(PREDICTIONS_DIR)):
            if not fn.startswith("pred_") or not fn.endswith(".jsonl"):
                continue
            ts = fn.replace("pred_", "").replace(".jsonl", "")
            path = os.path.join(PREDICTIONS_DIR, fn)
            runs.append({
                "id": ts,
                "ts": ts,
                "path": path,
                "num_records": _count_jsonl_lines(path),
            })

    runs = sorted(runs, key=lambda r: r["id"], reverse=True)
    has_data = len(runs) > 0

    all_num_records = sum(int(r.get("num_records") or 0) for r in runs)

    # Resolve selected run
    selected_run_key = run
    selected_run = run
    selected_paths: list[tuple[str, str]] = []  # (run_id, path)
    if has_data:
        if run == "all":
            selected_run = "all"
            selected_paths = [(r["id"], r["path"]) for r in runs]
        elif run == "latest":
            selected_run = runs[0]["id"]
            selected_paths = [(runs[0]["id"], runs[0]["path"])]
        else:
            matched = next((r for r in runs if r["id"] == run), None)
            if matched:
                selected_paths = [(matched["id"], matched["path"])]

    # 2) Load records (single run or all runs). When loading all, attach a lightweight run id.
    records: list[dict] = []
    for run_id, path in selected_paths:
        if not path or not os.path.exists(path):
            continue
        batch = read_jsonl(path)
        for rec in batch:
            # Keep original payload intact; add a new field for traceability.
            rec.setdefault("_pred_run_id", run_id)
        records.extend(batch)

    records_total = len(records)

    # 3) Apply KBS rules (adds rec["kbs"])
    enriched = [apply_business_rules(r) for r in records]

    # 4) Filtering (post-rule)
    def _overall_conf(r: dict) -> float | None:
        mc = r.get("model_confidence") or {}
        v = mc.get("overall")
        if v is None:
            return None
        try:
            if isinstance(v, str) and v.strip().lower().startswith("overall="):
                v = v.split("=", 1)[1].strip()
            return float(v)
        except Exception:
            return None

    pr = (priority or "").strip().upper()
    ot = (owner_team or "").strip()
    tg = (tag or "").strip().lower()
    qq = (q or "").strip().lower()

    filtered: list[dict] = []
    for r in enriched:
        kb = r.get("kbs") or {}
        if pr and str(kb.get("priority", "")).upper() != pr:
            continue
        if ot and str(kb.get("owner_team", "")) != ot:
            continue
        if tg:
            tags = [str(x).lower() for x in (kb.get("business_tags") or [])]
            if tg not in tags:
                continue
        if min_conf is not None:
            oc = _overall_conf(r)
            if oc is not None and oc < float(min_conf):
                continue
        if qq:
            text = (r.get("text") or r.get("text_normalized") or "").lower()
            hay = text + " " + " ".join((kb.get("business_tags") or [])).lower()
            if qq not in hay:
                continue
        filtered.append(r)

    # 5) Aggregations
    from collections import Counter
    pri_counts = Counter()
    owner_counts = Counter()
    tag_counts = Counter()

    for r in filtered:
        kb = r.get("kbs") or {}
        pri_counts[str(kb.get("priority") or "")] += 1
        owner_counts[str(kb.get("owner_team") or "")] += 1
        for t in (kb.get("business_tags") or []):
            tag_counts[str(t)] += 1

    # 6) Sample rows for display (prioritize higher priority)
    def _pri_rank(p: str) -> int:
        m = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        return m.get((p or "").upper(), 9)

    sample = sorted(
        filtered,
        key=lambda r: (_pri_rank((r.get("kbs") or {}).get("priority")), -float((_overall_conf(r) or 0.0))),
    )[:200]

    available_priorities = ["P0", "P1", "P2", "P3"]
    available_owners = sorted([k for k in owner_counts.keys() if k])
    available_tags = sorted([k for k in tag_counts.keys() if k])

    return templates.TemplateResponse(
        "kbs.html",
        {
            "request": request,
            "has_data": has_data,
            "runs": runs,
            "all_num_records": all_num_records,
            "selected_run": selected_run,
            "selected_run_key": selected_run_key,
            "records_total": records_total,

            "filtered_total": len(filtered),

            "priority_counts": dict(pri_counts),
            "owner_counts": dict(owner_counts),
            "tag_counts": dict(tag_counts),

            "available_priorities": available_priorities,
            "available_owners": available_owners,
            "available_tags": available_tags,

            "filter_min_conf": min_conf if min_conf is not None else "",
            "filter_priority": priority or "",
            "filter_owner_team": owner_team or "",
            "filter_tag": tag or "",
            "filter_q": q or "",

            "sample_records": sample,
        },
    )


@app.get("/prediction_run", response_class=JSONResponse)
async def prediction_run(pred_path: str):
    """
    Memuat hasil prediction dari file JSONL tertentu (pred_path),
    lalu mengembalikan records + agregat untuk ditampilkan di Prediction Analysis.
    """
    # Normalisasi dan keamanan dasar: pastikan path berada di dalam PREDICTIONS_DIR
    abs_base = os.path.abspath(PREDICTIONS_DIR)
    abs_target = os.path.abspath(pred_path)
    if not abs_target.startswith(abs_base):
        return JSONResponse(
            {"status": "error", "error": "pred_path di luar direktori predictions."},
            status_code=400,
        )

    if not os.path.exists(abs_target):
        return JSONResponse(
            {"status": "error", "error": "File prediction tidak ditemukan."},
            status_code=404,
        )

    # Baca records
    records = read_jsonl(abs_target)

    # Optional: pastikan agreement / jaccard ada; jika tidak, isi sebagai 'unknown'
    def _jaccard(a, b):
        sa, sb = set(a or []), set(b or [])
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / float(len(sa | sb))

    for rec in records:
        mc = rec.get("model_confidence") or {}
        nli_aspects = rec.get("nli_aspects") or []
        model_aspects = rec.get("model_pred_aspects") or []

        nli_cts = rec.get("nli_comment_types") or []
        if not nli_cts and rec.get("nli_comment_type"):
            nli_cts = [rec["nli_comment_type"]]

        model_cts = rec.get("model_pred_comment_types") or []
        if not model_cts and rec.get("model_pred_comment_type"):
            model_cts = [rec["model_pred_comment_type"]]

        aspect_j = mc.get("aspect_jaccard")
        if aspect_j is None:
            aspect_j = _jaccard(nli_aspects, model_aspects)
        type_j = mc.get("comment_type_jaccard")
        if type_j is None:
            type_j = _jaccard(nli_cts, model_cts)

        def _agree_label(j):
            if j is None:
                return "unknown"
            if j >= 0.999:
                return "full"
            if j <= 0.001:
                return "none"
            return "partial"

        aspect_agree = _agree_label(aspect_j)
        type_agree = _agree_label(type_j)
        if aspect_agree == "full" and type_agree == "full":
            overall_agree = "full"
        elif aspect_agree == "none" and type_agree == "none":
            overall_agree = "none"
        else:
            overall_agree = "partial"

        mc["aspect_jaccard"] = float(aspect_j)
        mc["comment_type_jaccard"] = float(type_j)
        rec["model_confidence"] = mc

        if "agreement" not in rec:
            rec["agreement"] = {
                "aspect": aspect_agree,
                "comment_type": type_agree,
                "overall": overall_agree,
            }

    # Cari model_run_dir dari history (jika ada)
    model_run_dir = ""
    history = get_prediction_history(limit=200)
    for h in history:
        if h.get("pred_path") == abs_target:
            model_run_dir = h.get("model_run_dir") or ""
            break

    agg = aggregate_predictions(records)

    return JSONResponse(
        {
            "status": "ok",
            "error": None,
            "total": len(records),
            "records": records,
            "agg": agg,
            "pred_path": abs_target,
            "model_run_dir": model_run_dir,
        }
    )

@app.get("/training_metrics", response_class=JSONResponse)
async def training_metrics(metrics_run_dir: str):
    """
    Mengambil metrics.json dari satu run (biasanya best_hp_run_dir / final_run_dir)
    dan mengembalikan ringkasan metrik + confusion matrix per kelas.
    Fokus di test_metrics (hasil terbaik).
    """
    # pastikan path berada di bawah OUTPUT_ROOT
    abs_base = os.path.abspath(OUTPUT_ROOT)
    abs_target = os.path.abspath(metrics_run_dir)

    if not abs_target.startswith(abs_base):
        return JSONResponse(
            {"status": "error", "error": "metrics_run_dir di luar OUTPUT_ROOT."},
            status_code=400,
        )

    metrics_path = os.path.join(abs_target, "metrics.json")
    if not os.path.exists(metrics_path):
        return JSONResponse(
            {"status": "error", "error": f"metrics.json tidak ditemukan di {abs_target}."},
            status_code=404,
        )

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            met = json.load(f)
    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": f"Gagal membaca metrics.json: {e}"},
            status_code=500,
        )

    test = met.get("test_metrics", {}) or {}

    # ringkasan global
    aspect_global = {
        "f1_micro": float(test.get("aspect_f1_micro", 0.0)),
        "f1_macro": float(test.get("aspect_f1_macro", 0.0)),
        "accuracy": float(test.get("aspect_accuracy", 0.0)),
    }
    type_global = {
        "f1_micro": float(test.get("comment_type_f1_micro", 0.0)),
        "f1_macro": float(test.get("comment_type_f1_macro", 0.0)),
        "accuracy": float(test.get("comment_type_accuracy", 0.0)),
    }

    # per-kelas: F1 + confusion matrix → precision, recall, accuracy, specificity, support
    aspect_f1_per_class = test.get("aspect_f1_per_class", {}) or {}
    type_f1_per_class = test.get("comment_type_f1_per_class", {}) or {}

    aspect_conf = test.get("aspect_confusion_matrices", {}) or {}
    type_conf = test.get("type_confusion_matrices", {}) or {}

    def per_class_from_conf(f1_dict, conf_dict):
        out = {}
        for label, f1 in f1_dict.items():
            cm = conf_dict.get(label)
            if not cm or len(cm) < 2 or len(cm[0]) < 2 or len(cm[1]) < 2:
                out[label] = {
                    "f1": float(f1),
                    "precision": None,
                    "recall": None,
                    "accuracy": None,
                    "specificity": None,
                    "support_pos": None,
                    "support_neg": None,
                    "support_total": None,
                    "tn": None,
                    "fp": None,
                    "fn": None,
                    "tp": None,
                }
                continue

            tn = int(cm[0][0])
            fp = int(cm[0][1])
            fn = int(cm[1][0])
            tp = int(cm[1][1])
            total = tn + fp + fn + tp

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            acc = (tp + tn) / total if total > 0 else 0.0
            # specificity = TN / (TN + FP)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            # support (jumlah contoh positif untuk kelas ini)
            support_pos = tp + fn
            support_neg = tn + fp
            support_total = total

            out[label] = {
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "accuracy": float(acc),
                "specificity": float(spec),
                "support_pos": int(support_pos),
                "support_neg": int(support_neg),
                "support_total": int(support_total),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        return out

    aspect_per_class = per_class_from_conf(aspect_f1_per_class, aspect_conf)
    type_per_class = per_class_from_conf(type_f1_per_class, type_conf)

    return JSONResponse(
        {
            "status": "ok",
            "metrics_run_dir": abs_target,
            "aspect": {
                "global": aspect_global,
                "per_class": aspect_per_class,
            },
            "comment_type": {
                "global": type_global,
                "per_class": type_per_class,
            },
        }
    )

@app.get("/training_test_predictions", response_class=JSONResponse)
async def training_test_predictions(metrics_run_dir: str):
    """
    Mengambil test_predictions.json dari satu run (biasanya best_hp_run_dir / final_run_dir)
    dan mengembalikan daftar baris untuk ditampilkan di tabel 'Data Test Prediction Result'.
    """
    abs_base = os.path.abspath(OUTPUT_ROOT)
    abs_target = os.path.abspath(metrics_run_dir)

    if not abs_target.startswith(abs_base):
        return JSONResponse(
            {"status": "error", "error": "metrics_run_dir di luar OUTPUT_ROOT."},
            status_code=400,
        )

    test_pred_path = os.path.join(abs_target, "test_predictions.json")
    if not os.path.exists(test_pred_path):
        return JSONResponse(
            {
                "status": "error",
                "error": f"test_predictions.json tidak ditemukan di {abs_target}.",
            },
            status_code=404,
        )

    try:
        with open(test_pred_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": f"Gagal membaca test_predictions.json: {e}"},
            status_code=500,
        )

    items = payload.get("items") or []
    return JSONResponse(
        {
            "status": "ok",
            "run_dir": metrics_run_dir,
            "aspect_labels": payload.get("aspect_labels") or [],
            "comment_type_labels": payload.get("comment_type_labels") or [],
            "items": items,
        }
    )

@app.get("/training_overview_metrics", response_class=JSONResponse)
async def training_overview_metrics(limit: int = 100):
    """
    Mengembalikan ringkasan performa training per run (training history),
    termasuk F1 macro dan accuracy untuk aspect dan comment type (test set).
    Data dipakai untuk grafik 'training performance over time'.
    """
    runs = get_training_history(limit=limit)

    overview = []
    for run in runs:
        run_dir = run.get("run_dir")
        best_hp_run_dir = run.get("best_hp_run_dir")
        final_run_dir = run.get("final_run_dir") or run_dir

        metrics_dir = best_hp_run_dir or final_run_dir or run_dir
        if not metrics_dir:
            continue

        metrics_path = os.path.join(metrics_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            overview.append(
                {
                    "run_dir": run_dir,
                    "metrics_dir": metrics_dir,
                    "created_at": run.get("created_at"),
                    "total_records": run.get("total_records", 0),
                    "aspect_f1_macro": None,
                    "comment_type_f1_macro": None,
                    "aspect_accuracy": None,
                    "comment_type_accuracy": None,
                    "best_hp_val_score": run.get("best_hp_val_score"),
                }
            )
            continue

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                met = json.load(f)
        except Exception:
            overview.append(
                {
                    "run_dir": run_dir,
                    "metrics_dir": metrics_dir,
                    "created_at": run.get("created_at"),
                    "total_records": run.get("total_records", 0),
                    "aspect_f1_macro": None,
                    "comment_type_f1_macro": None,
                    "aspect_accuracy": None,
                    "comment_type_accuracy": None,
                    "best_hp_val_score": run.get("best_hp_val_score"),
                }
            )
            continue

        test = met.get("test_metrics", {}) or {}
        overview.append(
            {
                "run_dir": run_dir,
                "metrics_dir": metrics_dir,
                "created_at": run.get("created_at"),
                "total_records": run.get("total_records", 0),
                "aspect_f1_macro": float(test.get("aspect_f1_macro", 0.0)),
                "comment_type_f1_macro": float(test.get("comment_type_f1_macro", 0.0)),
                "aspect_accuracy": float(test.get("aspect_accuracy", 0.0)),
                "comment_type_accuracy": float(test.get("comment_type_accuracy", 0.0)),
                "best_hp_val_score": run.get("best_hp_val_score"),
            }
        )

    # urutkan berdasarkan created_at kalau ada
    overview_sorted = sorted(
        overview,
        key=lambda r: r.get("created_at") or "",
    )

    return JSONResponse(
        {
            "status": "ok",
            "runs": overview_sorted,
        }
    )
