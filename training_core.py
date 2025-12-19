# training_core.py
#
# Multi-task transformer encoder for:
#  - aspects (multi-label)
#  - comment_type (multi-label)
#
# Defaults to a multilingual encoder (XLM-R) to better handle Indonesian + English + code-mixed reviews.
# Output artifact/file formats are kept stable for existing UI/API dependencies:
#   - config.json
#   - metrics.json
#   - aspect_thresholds.json
#   - grid_search_summary.json
# #   - checkpoint.pt
#
from __future__ import annotations

import os
import json
import csv
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

DEFAULT_ENCODER_NAME = "xlm-roberta-base"
DEFAULT_MAX_SEQ_LEN = 128

TEST_CSV_PATH = "./data/test.csv"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _split_csv_labels(s: str) -> List[str]:
    if s is None:
        return []
    raw = str(s).strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def load_human_test_csv(
    path: str,
    aspect_labels: List[str],
    type_labels: List[str],
) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    allowed_aspects = set(aspect_labels)
    allowed_types = set(type_labels)

    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="#")
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue

            aspects = [a for a in _split_csv_labels(row.get("aspect")) if a in allowed_aspects]
            cts = [c for c in _split_csv_labels(row.get("comment_type")) if c in allowed_types]

            out.append(
                {
                    "text": text,
                    "text_normalized": text,
                    "nli_aspects": aspects,
                    "nli_comment_types": cts,
                    "reason": (row.get("reason") or "").strip(),
                    "source": "human_test",
                }
            )
    return out


def _dedup_train_against_test(train_recs: List[Dict[str, Any]], test_recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    test_keys = set()
    for r in test_recs:
        key = (r.get("text_normalized") or r.get("text") or "").strip().lower()
        if key:
            test_keys.add(key)

    kept: List[Dict[str, Any]] = []
    for r in train_recs:
        key = (r.get("text_normalized") or r.get("text") or "").strip().lower()
        if key and key in test_keys:
            continue
        kept.append(r)
    return kept


def _simple_train_val_split(
    records: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)

    if len(records) < 2:
        return list(records), []

    n_val = int(round(len(records) * float(val_ratio)))
    n_val = max(1, min(len(records) - 1, n_val))

    val_idx = set(idxs[:n_val])
    train = [records[i] for i in idxs if i not in val_idx]
    val = [records[i] for i in idxs if i in val_idx]
    return train, val


def build_splits_with_fixed_test(
    all_records: List[Dict[str, Any]],
    *,
    aspect_labels: List[str],
    comment_type_labels: List[str],
    val_ratio: float = 0.15,
    seed: int = 42,
    test_csv_path: str = TEST_CSV_PATH,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    fixed_test_recs = load_human_test_csv(test_csv_path, aspect_labels, comment_type_labels)

    pool = list(all_records)
    if fixed_test_recs:
        pool = _dedup_train_against_test(pool, fixed_test_recs)

    train_recs, val_recs = _simple_train_val_split(pool, val_ratio=val_ratio, seed=seed)
    test_recs = fixed_test_recs
    return train_recs, val_recs, test_recs


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        aspect2idx: Dict[str, int],
        type2idx: Dict[str, int],
        max_len: int,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.aspect2idx = aspect2idx
        self.type2idx = type2idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        text = rec["text"]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        aspect_vec = np.zeros(len(self.aspect2idx), dtype=np.float32)
        for asp in rec.get("nli_aspects", []) or []:
            if asp in self.aspect2idx:
                aspect_vec[self.aspect2idx[asp]] = 1.0

        type_vec = np.zeros(len(self.type2idx), dtype=np.float32)
        cts = rec.get("nli_comment_types")
        if not cts:
            primary = rec.get("nli_comment_type")
            if primary:
                cts = [primary]
        if cts:
            for t in cts:
                if t in self.type2idx:
                    type_vec[self.type2idx[t]] = 1.0

        # sample weight (optional)
        w = 1.0
        lq = (rec.get("label_quality") or "").lower()
        nq = rec.get("nli_quality_score")
        if isinstance(nq, (int, float)):
            if float(nq) < 0.35:
                w *= 0.6
            elif float(nq) < 0.45:
                w *= 0.8
        elif lq == "low":
            w *= 0.7

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels_aspects": torch.from_numpy(aspect_vec),
            "labels_type": torch.from_numpy(type_vec),
            "sample_weight": torch.tensor(w, dtype=torch.float32),
        }


class MultiTaskReviewModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_aspects: int,
        num_types: int,
        dropout: float = 0.2,
        freeze_n_layers: int = 6,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            try:
                layers = self.encoder.encoder.layer
                for layer in layers[: max(0, int(freeze_n_layers))]:
                    for param in layer.parameters():
                        param.requires_grad = False
            except Exception:
                pass

        self.dropout = nn.Dropout(dropout)
        self.shared_dense = nn.Linear(hidden_size, hidden_size)
        self.shared_act = nn.ReLU()

        self.aspect_head = nn.Linear(hidden_size, num_aspects)
        self.type_head = nn.Linear(hidden_size, num_types)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)
        masked_hidden = last_hidden * mask
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-6)
        pooled = sum_hidden / lengths

        x = self.dropout(pooled)
        x = self.shared_dense(x)
        x = self.shared_act(x)
        x = self.dropout(x)

        return {
            "logits_aspects": self.aspect_head(x),
            "logits_type": self.type_head(x),
        }


@dataclass
class TrainConfig:
    encoder_name: str = DEFAULT_ENCODER_NAME
    batch_size: int = 16
    num_epochs: int = 5
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    max_len: int = DEFAULT_MAX_SEQ_LEN
    aspect_loss_weight: float = 1.0
    type_loss_weight: float = 1.0
    freeze_layers: int = 6
    dropout: float = 0.2

    val_ratio: float = 0.15
    seed: int = 42
    early_stopping_patience: int = 2


def _compute_pos_weight_multilabel(
    records: List[Dict[str, Any]],
    labels: List[str],
    field_getter: Callable[[Dict[str, Any]], List[str]],
) -> torch.Tensor:
    n = max(1, len(records))
    counts = np.zeros(len(labels), dtype=np.float64)
    idx = {lab: i for i, lab in enumerate(labels)}

    for rec in records:
        labs = field_getter(rec) or []
        for lab in labs:
            if lab in idx:
                counts[idx[lab]] += 1.0

    pos = counts
    neg = n - counts
    pw = np.ones(len(labels), dtype=np.float32)
    for i in range(len(labels)):
        if pos[i] > 0:
            pw[i] = float(neg[i] / max(1.0, pos[i]))
        else:
            pw[i] = 1.0
    return torch.tensor(pw, dtype=torch.float32)


def _build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
    pos_weight_aspects: Optional[torch.Tensor] = None,
    pos_weight_types: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits_aspects = outputs["logits_aspects"]
    logits_type = outputs["logits_type"]

    labels_aspects = batch["labels_aspects"].to(device)
    labels_type = batch["labels_type"].to(device)

    sample_weight = batch.get("sample_weight")
    if sample_weight is None:
        sample_weight = torch.ones(labels_aspects.shape[0], dtype=torch.float32)
    sample_weight = sample_weight.to(device).view(-1)

    if pos_weight_aspects is not None:
        bce_aspect = nn.BCEWithLogitsLoss(pos_weight=pos_weight_aspects.to(device), reduction="none")
    else:
        bce_aspect = nn.BCEWithLogitsLoss(reduction="none")

    if pos_weight_types is not None:
        bce_type = nn.BCEWithLogitsLoss(pos_weight=pos_weight_types.to(device), reduction="none")
    else:
        bce_type = nn.BCEWithLogitsLoss(reduction="none")

    loss_aspects_mat = bce_aspect(logits_aspects, labels_aspects)
    loss_type_mat = bce_type(logits_type, labels_type)

    loss_aspects_per = loss_aspects_mat.mean(dim=1)
    loss_type_per = loss_type_mat.mean(dim=1)

    loss_aspects = (loss_aspects_per * sample_weight).sum() / (sample_weight.sum() + 1e-8)
    loss_type = (loss_type_per * sample_weight).sum() / (sample_weight.sum() + 1e-8)

    loss = cfg.aspect_loss_weight * loss_aspects + cfg.type_loss_weight * loss_type

    return loss, {
        "loss_aspects": float(loss_aspects.item()),
        "loss_type": float(loss_type.item()),
        "loss_total": float(loss.item()),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    log_fn: Callable[[str], None] = print,
    pos_weight_aspects: Optional[torch.Tensor] = None,
    pos_weight_types: Optional[torch.Tensor] = None,
    grad_accum_steps: int = 1,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    model.train()
    running_loss = 0.0

    if grad_accum_steps is None or grad_accum_steps <= 0:
        grad_accum_steps = 1

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        use_amp = (scaler is not None) and (device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, loss_dict = compute_loss(
                outputs,
                batch,
                cfg,
                device,
                pos_weight_aspects=pos_weight_aspects,
                pos_weight_types=pos_weight_types,
            )
            loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss_dict["loss_total"])

        if (step + 1) % 50 == 0:
            avg_loss = running_loss / (step + 1)
            log_fn(
                f"Epoch {epoch} | Step {step+1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} "
                f"(aspect={loss_dict['loss_aspects']:.4f}, "
                f"type={loss_dict['loss_type']:.4f})"
            )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    aspect_thresholds: Optional[Dict[str, float]] = None,
    aspect_labels: List[str] = None,
    type_labels: List[str] = None,
) -> Dict[str, Any]:
    if aspect_labels is None or type_labels is None:
        raise ValueError("aspect_labels and type_labels are required.")

    model.eval()

    all_true_aspects: List[np.ndarray] = []
    all_pred_aspects: List[np.ndarray] = []
    all_aspect_scores: List[np.ndarray] = []
    all_type_scores: List[np.ndarray] = []

    all_true_types: List[np.ndarray] = []
    all_pred_types: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            true_aspects = batch["labels_aspects"].cpu().numpy()
            true_types = batch["labels_type"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_aspects = outputs["logits_aspects"].cpu().numpy()
            logits_type = outputs["logits_type"].cpu().numpy()

            probs_aspects = 1.0 / (1.0 + np.exp(-logits_aspects))
            all_aspect_scores.append(probs_aspects)

            if aspect_thresholds is None:
                thr = np.array([0.5] * len(aspect_labels))[None, :]
            else:
                thr = np.array([aspect_thresholds[label] for label in aspect_labels])[None, :]

            preds_aspect = (probs_aspects >= thr).astype(int)

            probs_type = 1.0 / (1.0 + np.exp(-logits_type))
            all_type_scores.append(probs_type)
            preds_type = (probs_type >= 0.5).astype(int)

            all_true_aspects.append(true_aspects)
            all_pred_aspects.append(preds_aspect)

            all_true_types.append(true_types)
            all_pred_types.append(preds_type)

    if not all_true_aspects:
        return {
            "aspect_scores": np.zeros((0, len(aspect_labels)), dtype=np.float32),
            "type_scores": np.zeros((0, len(type_labels)), dtype=np.float32),
            "aspect_true": np.zeros((0, len(aspect_labels)), dtype=np.int32),
            "aspect_pred": np.zeros((0, len(aspect_labels)), dtype=np.int32),
            "aspect_f1_micro": 0.0,
            "aspect_f1_macro": 0.0,
            "aspect_f1_per_class": {lab: 0.0 for lab in aspect_labels},
            "aspect_accuracy": 0.0,
            "aspect_precision_micro": 0.0,
            "aspect_recall_micro": 0.0,
            "aspect_confusion_matrices": {lab: [[0, 0], [0, 0]] for lab in aspect_labels},
            "type_true": np.zeros((0, len(type_labels)), dtype=np.int32),
            "type_pred": np.zeros((0, len(type_labels)), dtype=np.int32),
            "type_f1_micro": 0.0,
            "type_f1_macro": 0.0,
            "type_f1_per_class": {lab: 0.0 for lab in type_labels},
            "type_accuracy": 0.0,
            "type_precision_micro": 0.0,
            "type_recall_micro": 0.0,
            "type_confusion_matrices": {lab: [[0, 0], [0, 0]] for lab in type_labels},
        }

    all_true_aspects = np.vstack(all_true_aspects)
    all_pred_aspects = np.vstack(all_pred_aspects)
    all_aspect_scores = np.vstack(all_aspect_scores)

    all_true_types = np.vstack(all_true_types)
    all_pred_types = np.vstack(all_pred_types)
    all_type_scores = np.vstack(all_type_scores)

    f1_micro_aspect = f1_score(all_true_aspects.flatten(), all_pred_aspects.flatten(), average="micro")
    f1_macro_aspect = f1_score(all_true_aspects.flatten(), all_pred_aspects.flatten(), average="macro")
    acc_aspect = accuracy_score(all_true_aspects.flatten(), all_pred_aspects.flatten())
    prec_micro_aspect = precision_score(all_true_aspects.flatten(), all_pred_aspects.flatten(), average="micro", zero_division=0)
    rec_micro_aspect = recall_score(all_true_aspects.flatten(), all_pred_aspects.flatten(), average="micro", zero_division=0)

    f1_per_aspect: Dict[str, float] = {}
    aspect_conf_mats: Dict[str, List[List[int]]] = {}
    for i, label in enumerate(aspect_labels):
        f1_per_aspect[label] = f1_score(all_true_aspects[:, i], all_pred_aspects[:, i], zero_division=0)
        cm = confusion_matrix(all_true_aspects[:, i], all_pred_aspects[:, i], labels=[0, 1])
        aspect_conf_mats[label] = cm.tolist()

    f1_micro_type = f1_score(all_true_types.flatten(), all_pred_types.flatten(), average="micro")
    f1_macro_type = f1_score(all_true_types.flatten(), all_pred_types.flatten(), average="macro")
    acc_type = accuracy_score(all_true_types.flatten(), all_pred_types.flatten())
    prec_micro_type = precision_score(all_true_types.flatten(), all_pred_types.flatten(), average="micro", zero_division=0)
    rec_micro_type = recall_score(all_true_types.flatten(), all_pred_types.flatten(), average="micro", zero_division=0)

    f1_per_type: Dict[str, float] = {}
    type_conf_mats: Dict[str, List[List[int]]] = {}
    for i, label in enumerate(type_labels):
        f1_per_type[label] = f1_score(all_true_types[:, i], all_pred_types[:, i], zero_division=0)
        cm = confusion_matrix(all_true_types[:, i], all_pred_types[:, i], labels=[0, 1])
        type_conf_mats[label] = cm.tolist()

    return {
        "aspect_scores": all_aspect_scores,
        "type_scores": all_type_scores,
        "aspect_true": all_true_aspects,
        "aspect_pred": all_pred_aspects,
        "aspect_f1_micro": float(f1_micro_aspect),
        "aspect_f1_macro": float(f1_macro_aspect),
        "aspect_f1_per_class": f1_per_aspect,
        "aspect_accuracy": float(acc_aspect),
        "aspect_precision_micro": float(prec_micro_aspect),
        "aspect_recall_micro": float(rec_micro_aspect),
        "aspect_confusion_matrices": aspect_conf_mats,
        "type_true": all_true_types,
        "type_pred": all_pred_types,
        "type_f1_micro": float(f1_micro_type),
        "type_f1_macro": float(f1_macro_type),
        "type_f1_per_class": f1_per_type,
        "type_accuracy": float(acc_type),
        "type_precision_micro": float(prec_micro_type),
        "type_recall_micro": float(rec_micro_type),
        "type_confusion_matrices": type_conf_mats,
    }


def tune_aspect_thresholds(
    aspect_scores: np.ndarray,
    aspect_true: np.ndarray,
    aspect_labels: List[str],
    num_steps: int = 21,
) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    candidates = np.linspace(0.1, 0.9, num_steps)

    for i, label in enumerate(aspect_labels):
        best_thr = 0.5
        best_f1 = 0.0
        y_true = aspect_true[:, i]
        scores = aspect_scores[:, i]

        for thr in candidates:
            y_pred = (scores >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        thresholds[label] = float(best_thr)

    return thresholds


def _train_single_config(
    all_records: List[Dict[str, Any]],
    aspect_labels: List[str],
    comment_type_labels: List[str],
    run_dir: str,
    cfg: TrainConfig,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)

    train_recs, val_recs, test_recs = build_splits_with_fixed_test(
        all_records,
        aspect_labels=aspect_labels,
        comment_type_labels=comment_type_labels,
        val_ratio=getattr(cfg, "val_ratio", 0.15),
        seed=getattr(cfg, "seed", 42),
    )

    run_tag = os.path.basename(run_dir)
    log_fn(f"[{run_tag}] Split: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")

    aspect2idx = {lab: i for i, lab in enumerate(aspect_labels)}
    type2idx = {lab: i for i, lab in enumerate(comment_type_labels)}

    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)

    train_ds = MultiTaskDataset(train_recs, tokenizer, aspect2idx, type2idx, max_len=cfg.max_len)
    val_ds = MultiTaskDataset(val_recs, tokenizer, aspect2idx, type2idx, max_len=cfg.max_len)
    test_ds = MultiTaskDataset(test_recs, tokenizer, aspect2idx, type2idx, max_len=cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_fn(f"[{run_tag}] Using device: {device}")

    model = MultiTaskReviewModel(
        encoder_name=cfg.encoder_name,
        num_aspects=len(aspect_labels),
        num_types=len(comment_type_labels),
        dropout=cfg.dropout,
        freeze_n_layers=cfg.freeze_layers,
    ).to(device)

    pos_weight_aspects = _compute_pos_weight_multilabel(
        train_recs,
        aspect_labels,
        field_getter=lambda r: (r.get("nli_aspects") or []),
    )
    pos_weight_types = _compute_pos_weight_multilabel(
        train_recs,
        comment_type_labels,
        field_getter=lambda r: (
            r.get("nli_comment_types")
            or ([r.get("nli_comment_type")] if r.get("nli_comment_type") else [])
            or []
        ),
    )

    optimizer = _build_optimizer(model, lr=cfg.lr, weight_decay=0.01)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    grad_accum_steps = 1

    total_steps = max(1, (len(train_loader) // grad_accum_steps) * cfg.num_epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_combined = -1.0
    best_ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")
    patience = int(getattr(cfg, "early_stopping_patience", 2))
    no_improve = 0

    for epoch in range(1, cfg.num_epochs + 1):
        log_fn(f"[{run_tag}] Epoch {epoch}/{cfg.num_epochs}")
        train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            epoch=epoch,
            log_fn=log_fn,
            pos_weight_aspects=pos_weight_aspects,
            pos_weight_types=pos_weight_types,
            grad_accum_steps=grad_accum_steps,
            scaler=scaler,
        )

        val_metrics_tmp = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            aspect_thresholds=None,
            aspect_labels=aspect_labels,
            type_labels=comment_type_labels,
        )
        combined_tmp = 0.5 * (val_metrics_tmp["aspect_f1_micro"] + val_metrics_tmp["type_f1_micro"])

        log_fn(
            f"[{run_tag}] Val F1@0.5 "
            f"(aspect micro={val_metrics_tmp['aspect_f1_micro']:.4f}, "
            f"type micro={val_metrics_tmp['type_f1_micro']:.4f}, "
            f"combined={combined_tmp:.4f})"
        )

        if combined_tmp > best_val_combined + 1e-6:
            best_val_combined = float(combined_tmp)
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            log_fn(f"[{run_tag}] Saved best checkpoint: {best_ckpt_path}")
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                log_fn(f"[{run_tag}] Early stopping: no improvement for {patience} epochs.")
                break

    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        log_fn(f"[{run_tag}] Loaded best checkpoint: {best_ckpt_path} (val_combined={best_val_combined:.4f})")

    val_metrics_raw = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        aspect_thresholds=None,
        aspect_labels=aspect_labels,
        type_labels=comment_type_labels,
    )

    aspect_thresholds = tune_aspect_thresholds(
        aspect_scores=val_metrics_raw["aspect_scores"],
        aspect_true=val_metrics_raw["aspect_true"],
        aspect_labels=aspect_labels,
    )

    val_metrics_tuned = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        aspect_thresholds=aspect_thresholds,
        aspect_labels=aspect_labels,
        type_labels=comment_type_labels,
    )

    raw_combined = 0.5 * (val_metrics_raw["aspect_f1_micro"] + val_metrics_raw["type_f1_micro"])
    tuned_combined = 0.5 * (val_metrics_tuned["aspect_f1_micro"] + val_metrics_tuned["type_f1_micro"])
    val_score = float(tuned_combined)

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        aspect_thresholds=aspect_thresholds,
        aspect_labels=aspect_labels,
        type_labels=comment_type_labels,
    )

    # ---------------------------------------------------------
    # Simpan detail prediksi pada data test ke test_predictions.json
    # ---------------------------------------------------------
    test_items_for_export: List[Dict[str, Any]] = []
    try:
        # shallow copy supaya tidak mengganggu struktur asli
        test_recs_for_conf = [dict(r) for r in test_recs]

        test_scored = compute_model_confidences(
            records=test_recs_for_conf,
            model=model,
            tokenizer=tokenizer,
            aspect_labels=aspect_labels,
            comment_type_labels=comment_type_labels,
            run_id=os.path.basename(run_dir),
            encoder_name=cfg.encoder_name,
            max_len=cfg.max_len,
        )

        for i, rec in enumerate(test_scored):
            gold_aspects = rec.get("nli_aspects") or []
            gold_types = rec.get("nli_comment_types") or []
            if not gold_types and rec.get("nli_comment_type"):
                gold_types = [rec["nli_comment_type"]]

            probs_aspects = rec.get("model_probs_aspects") or {}
            probs_types = rec.get("model_probs_comment_types") or {}
            mc = rec.get("model_confidence") or {}

            test_items_for_export.append(
                {
                    # row index sesuai urutan di test set (1-based)
                    "row": i + 1,
                    "text": rec.get("text", ""),
                    "aspect": gold_aspects,
                    "comment_type": gold_types,
                    "pred_aspect_labels": rec.get("model_pred_aspects") or [],
                    "pred_comment_type_labels": (
                        rec.get("model_pred_comment_types")
                        or (
                            [rec.get("model_pred_comment_type")]
                            if rec.get("model_pred_comment_type")
                            else []
                        )
                    ),
                    "probs_aspects": probs_aspects,
                    "probs_comment_types": probs_types,
                    "overall_aspect_model_confidence": mc.get("aspect_min"),
                    "overall_comment_type_model_confidence": mc.get(
                        "nli_comment_type_conf"
                    ),
                    "overall_model_confidence": mc.get("overall"),
                }
            )

        test_predictions_payload = {
            "aspect_labels": aspect_labels,
            "comment_type_labels": comment_type_labels,
            "items": test_items_for_export,
        }
        with open(os.path.join(run_dir, "test_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(test_predictions_payload, f, ensure_ascii=False, indent=2)

        log_fn(f"[{run_tag}] Saved test_predictions.json with {len(test_items_for_export)} rows")
    except Exception as e:
        log_fn(f"[{run_tag}] WARNING: failed to write test_predictions.json: {e}")


    model_path = os.path.join(run_dir, "checkpoint.pt")
    if os.path.exists(best_ckpt_path):
        torch.save(torch.load(best_ckpt_path, map_location="cpu"), model_path)
    else:
        torch.save(model.state_dict(), model_path)

    config = {
        "encoder_name": cfg.encoder_name,
        "aspect_labels": aspect_labels,
        "comment_type_labels": comment_type_labels,
        "max_len": cfg.max_len,
        "freeze_layers": cfg.freeze_layers,
        "dropout": cfg.dropout,
        "train_config": {
            "batch_size": cfg.batch_size,
            "num_epochs": cfg.num_epochs,
            "lr": cfg.lr,
            "warmup_ratio": cfg.warmup_ratio,
            "max_len": cfg.max_len,
            "aspect_loss_weight": cfg.aspect_loss_weight,
            "type_loss_weight": cfg.type_loss_weight,
            "freeze_layers": cfg.freeze_layers,
            "dropout": cfg.dropout,
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "aspect_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(aspect_thresholds, f, ensure_ascii=False, indent=2)

    metrics_to_save = {
        "val_metrics_raw": {
            "aspect_f1_micro": float(val_metrics_raw["aspect_f1_micro"]),
            "aspect_f1_macro": float(val_metrics_raw["aspect_f1_macro"]),
            "aspect_f1_per_class": val_metrics_raw["aspect_f1_per_class"],
            "aspect_accuracy": float(val_metrics_raw.get("aspect_accuracy", 0.0)),
            "comment_type_f1_micro": float(val_metrics_raw["type_f1_micro"]),
            "comment_type_f1_macro": float(val_metrics_raw["type_f1_macro"]),
            "comment_type_f1_per_class": val_metrics_raw["type_f1_per_class"],
            "comment_type_accuracy": float(val_metrics_raw.get("type_accuracy", 0.0)),
            "combined": float(raw_combined),
        },
        "val_metrics_tuned": {
            "aspect_f1_micro": float(val_metrics_tuned["aspect_f1_micro"]),
            "aspect_f1_macro": float(val_metrics_tuned["aspect_f1_macro"]),
            "aspect_f1_per_class": val_metrics_tuned["aspect_f1_per_class"],
            "aspect_accuracy": float(val_metrics_tuned.get("aspect_accuracy", 0.0)),
            "comment_type_f1_micro": float(val_metrics_tuned["type_f1_micro"]),
            "comment_type_f1_macro": float(val_metrics_tuned["type_f1_macro"]),
            "comment_type_f1_per_class": val_metrics_tuned["type_f1_per_class"],
            "comment_type_accuracy": float(val_metrics_tuned.get("type_accuracy", 0.0)),
            "combined": float(tuned_combined),
        },
        "test_metrics": {
            "aspect_f1_micro": float(test_metrics["aspect_f1_micro"]),
            "aspect_f1_macro": float(test_metrics["aspect_f1_macro"]),
            "aspect_f1_per_class": test_metrics["aspect_f1_per_class"],
            "aspect_accuracy": float(test_metrics.get("aspect_accuracy", 0.0)),
            "comment_type_f1_micro": float(test_metrics["type_f1_micro"]),
            "comment_type_f1_macro": float(test_metrics["type_f1_macro"]),
            "comment_type_f1_per_class": test_metrics["type_f1_per_class"],
            "comment_type_accuracy": float(test_metrics.get("type_accuracy", 0.0)),
            "type_confusion_matrices": test_metrics.get("type_confusion_matrices", {}),
            "aspect_confusion_matrices": test_metrics.get("aspect_confusion_matrices", {}),
        },
        "aspect_thresholds": aspect_thresholds,
        "val_score": float(val_score),
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)

    return {
        "run_dir": run_dir,
        "config": asdict(cfg),
        "val_score": float(val_score),
        "val_metrics_raw": metrics_to_save["val_metrics_raw"],
        "val_metrics_tuned": metrics_to_save["val_metrics_tuned"],
        "test_metrics": metrics_to_save["test_metrics"],
        "aspect_thresholds": aspect_thresholds,
    }


def train_multitask_model_gridsearch(
    train_data_paths: List[str],
    aspect_labels: List[str],
    comment_type_labels: List[str],
    run_root_dir: str,
    base_cfg: Optional[TrainConfig] = None,
    max_configs: int = 4,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    os.makedirs(run_root_dir, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    for path in train_data_paths:
        if path and os.path.exists(path):
            recs = read_jsonl(path)
            all_records.extend(recs)
            log_fn(f"[grid] Loaded {len(recs)} records from {path}")
        else:
            log_fn(f"[grid] WARNING: missing/empty path: {path}")

    log_fn(f"[grid] Total training records: {len(all_records)}")
    if not all_records:
        raise ValueError("No records for training grid search.")

    if base_cfg is None:
        base_cfg = TrainConfig()

    lr_candidates = [base_cfg.lr, 3e-5]
    batch_candidates = [base_cfg.batch_size, max(8, base_cfg.batch_size // 2)]
    dropout_candidates = [base_cfg.dropout, 0.2]

    cfg_list: List[TrainConfig] = []
    for lr in lr_candidates:
        for bs in batch_candidates:
            for dr in dropout_candidates:
                cfg_list.append(
                    TrainConfig(
                        encoder_name=base_cfg.encoder_name,
                        batch_size=bs,
                        num_epochs=base_cfg.num_epochs,
                        lr=lr,
                        warmup_ratio=base_cfg.warmup_ratio,
                        max_len=base_cfg.max_len,
                        aspect_loss_weight=base_cfg.aspect_loss_weight,
                        type_loss_weight=base_cfg.type_loss_weight,
                        freeze_layers=base_cfg.freeze_layers,
                        dropout=dr,
                        val_ratio=getattr(base_cfg, "val_ratio", 0.15),
                        seed=getattr(base_cfg, "seed", 42),
                        early_stopping_patience=getattr(base_cfg, "early_stopping_patience", 2),
                    )
                )

    if len(cfg_list) > max_configs:
        cfg_list = cfg_list[:max_configs]

    summaries: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(cfg_list):
        hp_dir = os.path.join(run_root_dir, f"hp_{idx+1}")
        log_fn(f"[grid] Training config {idx+1}/{len(cfg_list)} in {hp_dir}")
        log_fn(f"[grid]  encoder={cfg.encoder_name}, lr={cfg.lr}, batch_size={cfg.batch_size}, dropout={cfg.dropout}")

        summary = _train_single_config(
            all_records=all_records,
            aspect_labels=aspect_labels,
            comment_type_labels=comment_type_labels,
            run_dir=hp_dir,
            cfg=cfg,
            log_fn=log_fn,
        )
        summaries.append(summary)

        vr = summary["val_metrics_raw"]
        vt = summary["val_metrics_tuned"]
        thr = summary["aspect_thresholds"]

        log_fn(
            f"[grid] Result hp_{idx+1}: "
            f"raw_combined={vr['combined']:.4f}, "
            f"tuned_combined={vt['combined']:.4f} "
            f"(aspect_tuned={vt['aspect_f1_micro']:.4f}, "
            f"type_tuned={vt['comment_type_f1_micro']:.4f})"
        )
        log_fn("[grid]  thresholds " + ", ".join(f"{k}={thr[k]:.2f}" for k in aspect_labels))

    best = max(summaries, key=lambda x: x["val_score"])
    best_run_dir = best["run_dir"]

    grid_summary = {
        "train_data_paths": train_data_paths,
        "aspect_labels": aspect_labels,
        "comment_type_labels": comment_type_labels,
        "configs": summaries,
        "best_run_dir": best_run_dir,
        "best_val_score": best["val_score"],
    }
    with open(os.path.join(run_root_dir, "grid_search_summary.json"), "w", encoding="utf-8") as f:
        json.dump(grid_summary, f, ensure_ascii=False, indent=2)

    log_fn(f"[grid] Best run: {best_run_dir} with val_score={best['val_score']:.4f}")
    return grid_summary


def load_model_for_inference(run_dir: str):
    config_path = os.path.join(run_dir, "config.json")
    thresh_path = os.path.join(run_dir, "aspect_thresholds.json")
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(thresh_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    encoder_name = cfg["encoder_name"]
    aspect_labels = cfg["aspect_labels"]
    comment_type_labels = cfg["comment_type_labels"]

    max_len = cfg.get("max_len", DEFAULT_MAX_SEQ_LEN)
    cfg["max_len"] = max_len
    freeze_layers = cfg.get("freeze_layers", 6)
    cfg["freeze_layers"] = freeze_layers
    dropout = cfg.get("dropout", 0.2)
    cfg["dropout"] = dropout

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = MultiTaskReviewModel(
        encoder_name=encoder_name,
        num_aspects=len(aspect_labels),
        num_types=len(comment_type_labels),
        dropout=dropout,
        freeze_n_layers=freeze_layers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    type_thresh_path = os.path.join(run_dir, "type_thresholds.json")
    type_thresholds = None
    if os.path.exists(type_thresh_path):
        try:
            with open(type_thresh_path, "r", encoding="utf-8") as f:
                type_thresholds = json.load(f)
        except Exception:
            type_thresholds = None

    return model, tokenizer, cfg, thresholds, type_thresholds


def compute_model_confidences(
    records: List[Dict[str, Any]],
    model,
    tokenizer,
    aspect_labels: List[str],
    comment_type_labels: List[str],
    run_id: str,
    encoder_name: str,
    max_len: int,
    aspect_thresholds: Optional[Dict[str, float]] = None,
    type_thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    aspect2idx = {lab: i for i, lab in enumerate(aspect_labels)}
    type2idx = {lab: i for i, lab in enumerate(comment_type_labels)}

    out_records: List[Dict[str, Any]] = []
    for rec in records:
        text = rec.get("text") or rec.get("text_normalized") or ""
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_aspects = out["logits_aspects"].cpu().numpy()[0]
            logits_type = out["logits_type"].cpu().numpy()[0]

        # sigmoid → probabilitas
        probs_aspects = 1.0 / (1.0 + np.exp(-logits_aspects))
        probs_type = 1.0 / (1.0 + np.exp(-logits_type))

        # simpan probabilitas per label (untuk UI)
        model_probs_aspects = {
            lab: float(probs_aspects[i]) for i, lab in enumerate(aspect_labels)
        }
        # Tambahkan probabilitas misc (turunan). Jika model tidak punya dimensi misc, nilai ini adalah komplemen
        # dari probabilitas aspek non-misc tertinggi.
        if "misc" not in model_probs_aspects:
            if len(probs_aspects) > 0:
                model_probs_aspects["misc"] = float(1.0 - float(np.max(probs_aspects)))
            else:
                model_probs_aspects["misc"] = 0.0
            misc_prob_is_derived = True
        else:
            misc_prob_is_derived = False
        model_probs_comment_types = {
            lab: float(probs_type[i]) for i, lab in enumerate(comment_type_labels)
        }

        # label prediksi
        # misc diperlakukan sebagai label turunan (no-aspect), bukan kelas yang harus aktif dari head.
        thr_map = aspect_thresholds or {}
        model_pred_aspects: List[str] = []
        for i, lab in enumerate(aspect_labels):
            if lab == "misc":
                continue
            thr = float(thr_map.get(lab, 0.5))
            if float(probs_aspects[i]) >= thr:
                model_pred_aspects.append(lab)

        model_pred_aspects_display = model_pred_aspects if model_pred_aspects else ["misc"]

        type_thr_map = type_thresholds or {}
        model_pred_comment_types: List[str] = []
        for i, lab in enumerate(comment_type_labels):
            thr_t = float(type_thr_map.get(lab, 0.5))
            if float(probs_type[i]) >= thr_t:
                model_pred_comment_types.append(lab)

        primary_idx = int(np.argmax(probs_type))
        model_pred_comment_type = comment_type_labels[primary_idx]

        # overall_conf: min(conf aspek, conf comment_type). Untuk no-aspect (list kosong),
        # gunakan keyakinan "none-of-the-above" (misc turunan) alih-alih 1.0.
        nli_aspects_raw = rec.get("nli_aspects", []) or []
        nli_aspects = [a for a in nli_aspects_raw if a and a != "misc"]
        asp_conf_list: List[float] = []
        for a in nli_aspects:
            if a in aspect2idx:
                asp_conf_list.append(float(probs_aspects[aspect2idx[a]]))
        if asp_conf_list:
            aspect_conf = float(min(asp_conf_list))
        else:
            aspect_conf = float(model_probs_aspects.get("misc", 0.0))

        nli_primary_type = rec.get("nli_comment_type")
        if nli_primary_type and nli_primary_type in type2idx:
            type_conf = float(probs_type[type2idx[nli_primary_type]])
        else:
            type_conf = 1.0

        overall_conf = min(aspect_conf, type_conf)

        rec["model_name"] = encoder_name
        rec["model_run_id"] = run_id
        rec["model_confidence"] = {
            "aspect_min": aspect_conf,
            "no_aspect_conf": float(model_probs_aspects.get("misc", 0.0)),
            "misc_prob_is_derived": bool(misc_prob_is_derived),
            "nli_comment_type": nli_primary_type,
            "nli_comment_type_conf": type_conf,
            "overall": overall_conf,
        }
        rec["model_pred_aspects"] = model_pred_aspects
        rec["model_pred_aspects_display"] = model_pred_aspects_display
        rec["model_misc_prob_is_derived"] = bool(misc_prob_is_derived)
        rec["model_pred_comment_type"] = model_pred_comment_type
        rec["model_pred_comment_types"] = model_pred_comment_types
        rec["model_probs_aspects"] = model_probs_aspects
        rec["model_probs_comment_types"] = model_probs_comment_types

        out_records.append(rec)

    return out_records