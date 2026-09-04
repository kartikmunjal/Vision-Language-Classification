from __future__ import annotations
import hashlib, json, platform, subprocess
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def run_provenance(inputs: dict[str, str | Path], *, args: dict | None = None) -> dict:
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: commit=None
    return {"git_commit":commit,"python":platform.python_version(),"arguments":args or {},
            "inputs":{k:{"path":str(v),"sha256":file_sha256(v)} for k,v in inputs.items()}}
