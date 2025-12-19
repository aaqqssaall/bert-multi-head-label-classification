import os
import io
import csv
import json
import datetime
from typing import List, Dict, Any, Optional

from config import OUTPUT_ROOT


def append_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_input_to_records(
    file_bytes: Optional[bytes],
    text_input: Optional[str],
    source: str,
    ts_id: str,
) -> List[Dict[str, Any]]:
    """
    Mengubah input file atau teks menjadi list record standar:
    {
      "id": "000000_<ts>",
      "text": "...",
      "source": source,
      "created_at": ISOString
    }
    """
    records: List[Dict[str, Any]] = []
    created = datetime.datetime.now().isoformat()

    if file_bytes:
        buf = io.StringIO(file_bytes.decode("utf-8", errors="ignore"))

        try:
            reader = csv.DictReader(buf)
            idx = 0
            for row in reader:
                if not row:
                    continue
                text = (
                    row.get("review")
                    or row.get("text")
                    or row.get("ulasan")
                    or list(row.values())[0]
                )
                text = (text or "").strip()
                if not text:
                    continue
                rec_id = f"{idx:06d}_{ts_id}"
                records.append(
                    {
                        "id": rec_id,
                        "text": text,
                        "source": source,
                        "created_at": created,
                    }
                )
                idx += 1
        except Exception:
            records = []

        if len(records) == 0:
            buf.seek(0)
            reader = csv.reader(buf)
            idx = 0
            for row in reader:
                if not row:
                    continue
                text = (row[0] or "").strip()
                if not text:
                    continue
                rec_id = f"{idx:06d}_{ts_id}"
                records.append(
                    {
                        "id": rec_id,
                        "text": text,
                        "source": source,
                        "created_at": created,
                    }
                )
                idx += 1
    else:
        if text_input:
            lines = [l.strip() for l in text_input.splitlines() if l.strip()]
            for idx, line in enumerate(lines):
                rec_id = f"{idx:06d}_{ts_id}"
                records.append(
                    {
                        "id": rec_id,
                        "text": line,
                        "source": source,
                        "created_at": created,
                    }
                )

    return records


def log_training_run(run_dir: Optional[str], total_records: int, extra: Dict[str, Any]) -> None:
    if not run_dir:
        return
    path = os.path.join(OUTPUT_ROOT, "training_runs.jsonl")
    rec = {
        "run_dir": run_dir,
        "created_at": datetime.datetime.now().isoformat(),
        "total_records": total_records,
    }
    rec.update(extra or {})
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
