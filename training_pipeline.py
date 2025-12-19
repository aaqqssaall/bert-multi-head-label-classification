import os
import json
import datetime
import re
from typing import List, Dict, Any, Optional

from training_core import (
    TrainConfig,
    train_multitask_model_gridsearch,
    load_model_for_inference,
    compute_model_confidences,
    read_jsonl,
)
from nli_labelling import label_batch_with_nli

from config import (
    BASE_TRAIN_PATH,
    ADDITIONAL_DATA_PATH,
    LOW_SCORE_PATH,
    INPUT_DIR,
    OUTPUT_ROOT,
    MODEL_ASPECT_LABELS,
    COMMENT_TYPE_LABELS,
    TRAINING_EXCLUSION_PATH,
)

from io_utils import append_jsonl, log_training_run
from jobs import job_log, JOBS


# -----------------------------------------------------------------------------
# Training exclusion logic (post-NLI)
# -----------------------------------------------------------------------------
# Catatan: File TRAINING_EXCLUSION_PATH bernama *.json namun ditulis sebagai JSONL
# (satu JSON object per baris) agar bisa di-append dengan aman.

_FALLBACK_ASPECTS = {"misc", "none", "blank", ""}


def _utc_now_z() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _append_exclusion(rec: Dict[str, Any]) -> None:
    if not rec:
        return
    os.makedirs(os.path.dirname(TRAINING_EXCLUSION_PATH), exist_ok=True)
    with open(TRAINING_EXCLUSION_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _word_count(text: str) -> int:
    toks = re.findall(r"[a-zA-Z0-9_]+", (text or ""))
    return int(len(toks))


def _extract_aspects_for_training(rec: Dict[str, Any]) -> List[str]:
    aspects = rec.get("nli_aspects")
    if aspects is None:
        aspects = rec.get("aspects")
    if aspects is None:
        aspects = []
    if not isinstance(aspects, list):
        aspects = [str(aspects)]
    # Keep only model labels (no misc).
    return [a for a in aspects if isinstance(a, str) and a in set(MODEL_ASPECT_LABELS)]


def _filter_and_log_training_exclusions(
    records: List[Dict[str, Any]],
    *,
    source: str,
    stage: str,
    min_words: int = 5,
) -> List[Dict[str, Any]]:
    """Return records eligible for training; append excluded rows to TRAINING_EXCLUSION_PATH."""
    kept: List[Dict[str, Any]] = []
    for r in records or []:
        if not isinstance(r, dict):
            continue

        text = (r.get("text_normalized") or r.get("text") or "").strip()
        aspects_clean = _extract_aspects_for_training(r)

        # 1) Exclude if no non-fallback aspect after NLI.
        if not aspects_clean:
            ex = dict(r)
            ex["removed_by_nli_misc"] = True
            ex["removed_at"] = _utc_now_z()
            ex["removed_reason"] = "nli_misc_or_no_aspect"
            ex["exclusion_source"] = source
            ex["exclusion_stage"] = stage
            _append_exclusion(ex)
            continue

        # 2) Exclude if too short after NLI + misc removal.
        if _word_count(text) <= int(min_words):
            ex = dict(r)
            ex["removed_by_nli_misc"] = False
            ex["removed_at"] = _utc_now_z()
            ex["removed_reason"] = "insufficient_word_count"
            ex["exclusion_source"] = source
            ex["exclusion_stage"] = stage
            _append_exclusion(ex)
            continue

        # Normalize aspects for training usage (avoid leaking misc/unknown labels).
        r2 = dict(r)
        r2["nli_aspects"] = aspects_clean
        kept.append(r2)

    return kept


def train_and_route_new_batch_job(
    job_id: str,
    labeled_records: List[Dict[str, Any]],
    labeled_jsonl_path: str,
    confidence_threshold: float,
    run_training_flag: bool,
    min_train_records: int,
    train_cfg_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    def log(msg: str) -> None:
        job_log(job_id, msg)

    base_count = 0
    additional_count = 0
    low_count = 0
    training_skipped = False
    best_hp_run_dir: Optional[str] = None
    final_run_dir: Optional[str] = None

    # Gunakan versi *cleaned* untuk training (base/additional/batch) tanpa mengubah file sumber.
    train_paths: List[str] = []
    total_train_records = 0

    if os.path.exists(BASE_TRAIN_PATH):
        base_recs = read_jsonl(BASE_TRAIN_PATH)
        base_clean = _filter_and_log_training_exclusions(
            base_recs,
            source="base_train",
            stage="pre_training",
        )
        base_clean_path = os.path.join(INPUT_DIR, f"base_train_clean_{job_id}.jsonl")
        with open(base_clean_path, "w", encoding="utf-8") as f:
            for r in base_clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        train_paths.append(base_clean_path)
        total_train_records += len(base_clean)
        log(f"Excluded from training (base_train): {len(base_recs) - len(base_clean)}")

    if os.path.exists(ADDITIONAL_DATA_PATH):
        add_recs = read_jsonl(ADDITIONAL_DATA_PATH)
        add_clean = _filter_and_log_training_exclusions(
            add_recs,
            source="additional_data",
            stage="pre_training",
        )
        add_clean_path = os.path.join(INPUT_DIR, f"additional_data_clean_{job_id}.jsonl")
        with open(add_clean_path, "w", encoding="utf-8") as f:
            for r in add_clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        train_paths.append(add_clean_path)
        total_train_records += len(add_clean)
        log(f"Excluded from training (additional_data): {len(add_recs) - len(add_clean)}")

    # Labeled batch: sudah berisi NLI label; tetap lakukan sanitasi aspek + word-count.
    batch_clean = _filter_and_log_training_exclusions(
        labeled_records,
        source="upload_batch",
        stage="post_nli",
    )
    if batch_clean:
        batch_clean_path = os.path.join(INPUT_DIR, f"{job_id}_batch_clean.jsonl")
        with open(batch_clean_path, "w", encoding="utf-8") as f:
            for r in batch_clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        train_paths.append(batch_clean_path)
        total_train_records += len(batch_clean)
    log(f"Excluded from training (current batch): {len(labeled_records) - len(batch_clean)}")

    # Selanjutnya, routing hanya memakai record yang eligible untuk training.
    labeled_records = batch_clean

    log(f"Total training records candidate (base + additional + batch): {total_train_records}")

    if run_training_flag and total_train_records >= min_train_records:
        log("Training enabled and enough data. Starting grid search training.")

        ts_run = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        run_root_dir = os.path.join(OUTPUT_ROOT, f"run_{ts_run}")

        base_cfg = TrainConfig()
        if "num_epochs" in train_cfg_overrides and train_cfg_overrides["num_epochs"] is not None:
            base_cfg.num_epochs = int(train_cfg_overrides["num_epochs"])
        if "batch_size" in train_cfg_overrides and train_cfg_overrides["batch_size"] is not None:
            base_cfg.batch_size = int(train_cfg_overrides["batch_size"])
        if "lr" in train_cfg_overrides and train_cfg_overrides["lr"] is not None:
            base_cfg.lr = float(train_cfg_overrides["lr"])
        if "warmup_ratio" in train_cfg_overrides and train_cfg_overrides["warmup_ratio"] is not None:
            base_cfg.warmup_ratio = float(train_cfg_overrides["warmup_ratio"])
        if "dropout" in train_cfg_overrides and train_cfg_overrides["dropout"] is not None:
            base_cfg.dropout = float(train_cfg_overrides["dropout"])

        grid_summary = train_multitask_model_gridsearch(
            train_data_paths=train_paths,
            aspect_labels=MODEL_ASPECT_LABELS,
            comment_type_labels=COMMENT_TYPE_LABELS,
            run_root_dir=run_root_dir,
            base_cfg=base_cfg,
            max_configs=4,
            log_fn=log,
        )
        best_hp_run_dir = grid_summary["best_run_dir"]

        best_cfg_dict = None
        for cfg_summary in grid_summary["configs"]:
            if cfg_summary["run_dir"] == best_hp_run_dir:
                best_cfg_dict = cfg_summary["config"]
                break
        if best_cfg_dict is None:
            raise RuntimeError("Best config tidak ditemukan di grid_summary['configs'].")

        best_cfg = TrainConfig(**best_cfg_dict)
        final_run_dir = best_hp_run_dir

        log_training_run(
            final_run_dir,
            total_train_records,
            {
                "best_hp_run_dir": best_hp_run_dir,
                "best_hp_val_score": grid_summary["best_val_score"],
                "grid_search_summary_path": os.path.join(run_root_dir, "grid_search_summary.json"),
            },
        )

        log(f"Loading best grid-search model from: {final_run_dir}")
        model, tokenizer, cfg_model, thresholds, type_thresholds = load_model_for_inference(final_run_dir)
        encoder_name = cfg_model["encoder_name"]
        run_id = os.path.basename(final_run_dir)

        log(f"Scoring {len(labeled_records)} newly labelled records...")
        scored_records = compute_model_confidences(
            records=labeled_records,
            model=model,
            tokenizer=tokenizer,
            aspect_labels=MODEL_ASPECT_LABELS,
            comment_type_labels=COMMENT_TYPE_LABELS,
            run_id=run_id,
            encoder_name=encoder_name,
            max_len=cfg_model["max_len"],
            aspect_thresholds=thresholds,
            type_thresholds=type_thresholds,
        )

        base_list: List[Dict[str, Any]] = []
        add_list: List[Dict[str, Any]] = []
        low_list: List[Dict[str, Any]] = []

        for rec in scored_records:
            quality = rec.get("label_quality", "unknown")
            mc = rec.get("model_confidence", {}) or {}
            overall = float(mc.get("overall", 0.0))

            if quality == "high":
                if overall >= confidence_threshold:
                    rec["model_disagreement"] = False
                    base_list.append(rec)
                else:
                    rec["model_disagreement"] = True
                    add_list.append(rec)
            elif quality == "medium":
                if overall >= confidence_threshold:
                    rec["model_disagreement"] = False
                    add_list.append(rec)
                else:
                    rec["model_disagreement"] = True
                    low_list.append(rec)
            else:
                rec["model_disagreement"] = True
                low_list.append(rec)

        if base_list:
            append_jsonl(BASE_TRAIN_PATH, base_list)
        if add_list:
            append_jsonl(ADDITIONAL_DATA_PATH, add_list)
        if low_list:
            append_jsonl(LOW_SCORE_PATH, low_list)

        base_count = len(base_list)
        additional_count = len(add_list)
        low_count = len(low_list)

        log(
            f"Routed to base_train: {base_count}, "
            f"additional_data: {additional_count}, low_score: {low_count}"
        )

    else:
        training_skipped = True
        if not run_training_flag:
            log("Training skipped: user unchecked 'run_training'.")
        else:
            log(f"Training skipped: total records {total_train_records} < min_train_records ({min_train_records}).")

        base_list: List[Dict[str, Any]] = []
        add_list: List[Dict[str, Any]] = []
        low_list: List[Dict[str, Any]] = []

        for rec in labeled_records:
            quality = rec.get("label_quality", "unknown")
            rec["model_name"] = None
            rec["model_run_id"] = None
            rec["model_confidence"] = None
            rec["model_disagreement"] = None
            rec["model_pred_aspects"] = None
            rec["model_pred_comment_type"] = None
            rec["model_pred_comment_types"] = None

            if quality == "high":
                add_list.append(rec)
            else:
                low_list.append(rec)

        if add_list:
            append_jsonl(ADDITIONAL_DATA_PATH, add_list)
        if low_list:
            append_jsonl(LOW_SCORE_PATH, low_list)

        base_count = 0
        additional_count = len(add_list)
        low_count = len(low_list)

        log(f"Routed (no training) to additional_data: {additional_count}, low_score: {low_count}")

    return {
        "best_hp_run_dir": best_hp_run_dir,
        "final_run_dir": final_run_dir,
        "base_count": base_count,
        "additional_count": additional_count,
        "low_score_count": low_count,
        "training_skipped": training_skipped,
        "total_train_records": total_train_records,
    }


def background_run_pipeline(
    job_id: str,
    input_jsonl_path: str,
    confidence_threshold: float,
    run_training_flag: bool,
    min_train_records: int,
    train_cfg_overrides: Dict[str, Any],
) -> None:
    try:
        # tandai job sebagai running
        job = JOBS.get(job_id)
        if job is not None:
            job["status"] = "running"

        job_log(job_id, f"Job started. Input file: {input_jsonl_path}")

        raw_records = read_jsonl(input_jsonl_path)
        job_log(job_id, f"Loaded {len(raw_records)} raw records.")

        job_log(job_id, "Starting NLI zero-shot labelling...")
        texts = [
            (r.get("text_normalized") or r.get("text") or "")
            for r in raw_records
        ]
        nli_results = label_batch_with_nli(texts)

        job_log(job_id, "NLI labelling done.")

        labeled_records: List[Dict[str, Any]] = []
        for rec, nli in zip(raw_records, nli_results):
            out = dict(rec)
            out["nli_aspects"] = nli.get("aspects", [])
            out["nli_comment_type"] = nli.get("comment_type")
            out["nli_comment_types"] = nli.get("comment_types", [])
            out["label_quality"] = nli.get("label_quality", "unknown")
            out["nli_quality_score"] = nli.get("nli_quality_score", 0.0)
            out["nli_debug"] = nli.get("nli_debug") or nli.get("debug", {})
            labeled_records.append(out)

        base_name = os.path.basename(input_jsonl_path).replace(".jsonl", "")
        labeled_jsonl_path = os.path.join(INPUT_DIR, f"{base_name}_labeled.jsonl")
        with open(labeled_jsonl_path, "w", encoding="utf-8") as f:
            for rec in labeled_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        job_log(job_id, f"NLI labelled file saved to {labeled_jsonl_path}.")

        res = train_and_route_new_batch_job(
            job_id=job_id,
            labeled_records=labeled_records,
            labeled_jsonl_path=labeled_jsonl_path,
            confidence_threshold=confidence_threshold,
            run_training_flag=run_training_flag,
            min_train_records=min_train_records,
            train_cfg_overrides=train_cfg_overrides,
        )

        job = JOBS[job_id]
        job["status"] = "finished"
        job["result"] = {
            "raw_count": len(raw_records),
            "input_jsonl_path": input_jsonl_path,
            "labeled_jsonl_path": labeled_jsonl_path,
            "best_hp_run_dir": res["best_hp_run_dir"],
            "final_run_dir": res["final_run_dir"],
            "base_count": res["base_count"],
            "additional_count": res["additional_count"],
            "low_score_count": res["low_score_count"],
            "total_train_records": res["total_train_records"],
            "training_skipped": res["training_skipped"],
            "run_training": run_training_flag,
            "min_train_records": min_train_records,
        }
        job_log(job_id, "Job finished successfully.")

    except Exception as e:
        job = JOBS[job_id]
        job["status"] = "error"
        job["result"] = {"error": str(e)}
        job_log(job_id, f"Job failed: {repr(e)}")
