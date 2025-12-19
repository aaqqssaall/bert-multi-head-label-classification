#!/usr/bin/env python3
"""
hk_checkpoint.py

Utility to reduce disk usage by pruning training artifacts under ./outputs
based on ./outputs/training_runs.jsonl.

What it does (safe defaults):
1) For each run_* directory referenced in training_runs.jsonl, keep only:
   - best_hp_run_dir (and final_run_dir if present)
   - delete other hp_* sibling directories under the same run_YYYY... folder
2) Optionally, prune checkpoints inside kept run dirs by using HuggingFace
   Trainer's trainer_state.json "best_model_checkpoint" field.

Safety:
- Dry-run by default (prints what would be deleted).
- Requires --yes to actually delete files/directories.
- Moves nothing; it deletes. Use with care.

Examples:
  python hk_checkpoint.py
  python hk_checkpoint.py --yes
  python hk_checkpoint.py --yes --prune-checkpoints
  python hk_checkpoint.py --outputs ./outputs --keep-latest 3 --yes

Notes:
- Works cross-platform (Windows/Linux). Paths in jsonl may be absolute Windows
  paths; if they don't exist on the current machine, they will be skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class RunEntry:
    run_dir: Path
    best_hp_run_dir: Optional[Path]
    final_run_dir: Optional[Path]
    created_at: Optional[str]
    best_hp_val_score: Optional[float]


def _safe_path(p: str) -> Path:
    # Accept both windows and posix style paths
    return Path(p)


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"training_runs.jsonl not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
    return rows


def _parse_entries(rows: List[dict]) -> List[RunEntry]:
    out: List[RunEntry] = []
    for r in rows:
        run_dir = _safe_path(str(r.get("run_dir", "")))
        best_hp = r.get("best_hp_run_dir")
        final = r.get("final_run_dir")
        created_at = r.get("created_at")
        score = r.get("best_hp_val_score")
        out.append(
            RunEntry(
                run_dir=run_dir,
                best_hp_run_dir=_safe_path(best_hp) if best_hp else None,
                final_run_dir=_safe_path(final) if final else None,
                created_at=created_at,
                best_hp_val_score=float(score) if score is not None else None,
            )
        )
    return out


def _infer_run_root_from_hp_dir(hp_dir: Path) -> Optional[Path]:
    # Expect outputs/run_YYYYMMDDhhmmss/hp_k
    if hp_dir is None:
        return None
    if hp_dir.name.startswith("hp_") and hp_dir.parent.name.startswith("run_"):
        return hp_dir.parent
    # Sometimes run_dir might already be the run_* directory
    if hp_dir.name.startswith("run_"):
        return hp_dir
    return None


def _list_hp_siblings(run_root: Path) -> List[Path]:
    if not run_root.exists() or not run_root.is_dir():
        return []
    return sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("hp_")])


def _human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, dirs, files in os.walk(path):
        for fn in files:
            fp = Path(root) / fn
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def _delete_path(path: Path, yes: bool) -> Tuple[bool, int]:
    """Return (deleted?, bytes_freed_estimate)."""
    if not path.exists():
        return False, 0
    size = _dir_size(path)
    if not yes:
        return False, size
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass
    return True, size


def _prune_checkpoints_in_run_dir(run_dir: Path, yes: bool) -> List[Tuple[str, Path, int]]:
    """
    Remove checkpoint-* directories except the best checkpoint declared in trainer_state.json.
    Returns a list of actions (action, path, bytes_est).
    """
    actions: List[Tuple[str, Path, int]] = []
    trainer_state = run_dir / "trainer_state.json"
    if not trainer_state.exists():
        return actions

    try:
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
    except Exception:
        return actions

    best_ckpt = state.get("best_model_checkpoint")
    if not best_ckpt:
        return actions

    best_path = Path(best_ckpt)
    # best_ckpt might be absolute or relative
    if not best_path.is_absolute():
        best_path = (run_dir / best_path).resolve()
    else:
        best_path = best_path.resolve()

    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            if p.resolve() == best_path:
                continue
            deleted, bytes_est = _delete_path(p, yes=yes)
            actions.append(("delete_checkpoint" if yes else "would_delete_checkpoint", p, bytes_est))
    return actions


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune training runs/checkpoints to save disk space.")
    ap.add_argument("--outputs", default="outputs", help="Outputs directory (default: ./outputs)")
    ap.add_argument("--runs-jsonl", default=None, help="Path to training_runs.jsonl (default: <outputs>/training_runs.jsonl)")
    ap.add_argument("--yes", action="store_true", help="Actually delete. Without this flag, it's a dry-run.")
    ap.add_argument("--prune-checkpoints", action="store_true", help="Also prune checkpoint-* within kept run dirs using trainer_state.json")
    ap.add_argument("--keep-latest", type=int, default=0, help="Additionally keep latest N run_* folders (0 disables).")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    runs_jsonl = Path(args.runs_jsonl) if args.runs_jsonl else outputs_dir / "training_runs.jsonl"

    rows = _read_jsonl(runs_jsonl)
    entries = _parse_entries(rows)

    # Determine which directories to keep
    keep_dirs: Set[Path] = set()
    run_roots: Set[Path] = set()

    for e in entries:
        for d in [e.best_hp_run_dir, e.final_run_dir, e.run_dir]:
            if d:
                keep_dirs.add(d)
                root = _infer_run_root_from_hp_dir(d)
                if root:
                    run_roots.add(root)

    # Optionally keep latest N run_* folders found under outputs_dir
    if args.keep_latest and outputs_dir.exists():
        run_folders = sorted([p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")],
                             key=lambda p: p.name, reverse=True)
        for p in run_folders[: args.keep_latest]:
            run_roots.add(p)

    # Plan deletions: delete hp_* siblings not in keep set
    planned: List[Tuple[str, Path, int]] = []
    for run_root in sorted(run_roots):
        hp_dirs = _list_hp_siblings(run_root)
        if not hp_dirs:
            continue

        # Keep hp dirs whose path matches any keep_dirs exactly (resolve for safety)
        keep_resolved = {k.resolve() for k in keep_dirs if k.exists()}
        for hp in hp_dirs:
            if hp.resolve() in keep_resolved:
                continue
            deleted, bytes_est = _delete_path(hp, yes=args.yes)
            planned.append(("delete_hp" if args.yes else "would_delete_hp", hp, bytes_est))

        if args.prune_checkpoints:
            # prune checkpoints inside kept hp dirs under this run_root
            for hp in hp_dirs:
                if hp.resolve() in keep_resolved:
                    planned.extend(_prune_checkpoints_in_run_dir(hp, yes=args.yes))

    # Print summary
    total_bytes = sum(b for _, _, b in planned)
    mode = "DELETED" if args.yes else "DRY-RUN"
    print(f"[{mode}] Planned actions: {len(planned)}; space {'freed' if args.yes else 'to free'} ~ {_human_bytes(total_bytes)}")
    for action, path, b in planned:
        print(f"- {action}: {path}  (~{_human_bytes(b)})")

    if not args.yes:
        print("\nNo files were deleted (dry-run). Re-run with --yes to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
