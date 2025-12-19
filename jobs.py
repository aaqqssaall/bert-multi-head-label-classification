import datetime
from typing import Dict, Any

JOBS: Dict[str, Dict[str, Any]] = {}


def job_log(job_id: str, msg: str) -> None:
    now = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{now}] {msg}"
    job = JOBS.setdefault(job_id, {"status": "pending", "logs": [], "result": None})
    job["logs"].append(line)
    print(line)