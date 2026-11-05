"""Exclusive creation and durable JSONL writes for the separate LLM workflow."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4


def new_run(root):
    run = Path(root).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    return run


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def append_jsonl(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
