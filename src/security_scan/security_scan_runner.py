#!/usr/bin/env python3
"""
Batch security scan runner for IaC-Eval workspaces.

Reads a results.jsonl file (as produced by run_iac_eval_benchmark.py),
filters the attempts whose Terraform validation succeeded, and then
executes tfsec / Terrascan / Checkov sequentially on every workspace.

Usage example:

python3 security_scan_runner.py \
  --results-jsonl /home/IaC\ bench/motivation/iac_eval/results/gpt-5.1/t0_0_p0_95/results.jsonl \
  --output-jsonl  /home/IaC\ bench/motivation/iac_eval/results/gpt-5.1/t0_0_p0_95/security_scans_report.jsonl \
  --tfsec-bin     /home/linuxbrew/.linuxbrew/bin/tfsec \
  --terrascan-bin /home/linuxbrew/.linuxbrew/bin/terrascan \
  --checkov-bin   /home/linuxbrew/.linuxbrew/bin/checkov
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run tfsec/Terrascan/Checkov over IaC-Eval workspaces.")
    ap.add_argument(
        "--results-jsonl",
        required=True,
        help="Path to gpt-5.1 results.jsonl (contains workspace info).",
    )
    ap.add_argument(
        "--output-jsonl",
        required=True,
        help="Where to store per-sample scan outputs (JSONL, append mode).",
    )
    ap.add_argument("--tfsec-bin", default="tfsec", help="tfsec binary path.")
    ap.add_argument("--terrascan-bin", default="terrascan", help="Terrascan binary path.")
    ap.add_argument("--checkov-bin", default="checkov", help="Checkov binary path.")
    ap.add_argument("--tfsec-timeout", type=int, default=180, help="tfsec timeout seconds.")
    ap.add_argument("--terrascan-timeout", type=int, default=300, help="Terrascan timeout seconds.")
    ap.add_argument("--checkov-timeout", type=int, default=480, help="Checkov timeout seconds.")
    ap.add_argument("--resume", action="store_true", help="Skip samples that already exist in output jsonl.")
    return ap.parse_args()


def load_successful_samples(results_path: Path) -> Dict[str, Dict[str, str]]:
    """Return {sample_id: {'attempt': int, 'workspace': str, 'meta': dict}} for validate_ok samples."""
    successes: Dict[str, Dict[str, str]] = {}
    with results_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("validate_ok"):
                continue
            sid = obj.get("sample_id")
            attempt = int(obj.get("attempt", 0))
            workspace = obj.get("workspace")
            if not sid or not workspace:
                continue
            prev = successes.get(sid)
            if prev is None or attempt < prev["attempt"]:
                successes[sid] = {
                    "attempt": attempt,
                    "workspace": workspace,
                    "meta": obj.get("meta", {}),
                }
    return successes


def load_completed_samples(output_jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not output_jsonl.exists():
        return done
    with output_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("sample_id")
            if sid:
                done.add(sid)
    return done


def run_subprocess(cmd: List[str], cwd: str, timeout_s: int) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_tfsec(bin_path: str, workspace: str, timeout_s: int) -> Dict[str, object]:
    data: Dict[str, object] = {}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [bin_path, "--format", "json", "--no-color", "--out", str(tmp_path), workspace]
        rc, stdout, stderr = run_subprocess(cmd, cwd=workspace, timeout_s=timeout_s)
        data["return_code"] = rc
        data["stderr"] = stderr.strip()
        if tmp_path.exists():
            try:
                data["result"] = json.loads(tmp_path.read_text(encoding="utf-8"))
            except Exception as exc:
                data["parse_error"] = str(exc)
        else:
            data["result"] = None
        return data
    finally:
        tmp_path.unlink(missing_ok=True)


def run_terrascan(bin_path: str, workspace: str, timeout_s: int) -> Dict[str, object]:
    cmd = [bin_path, "scan", "-o", "json", "-d", workspace]
    rc, stdout, stderr = run_subprocess(cmd, cwd=workspace, timeout_s=timeout_s)
    data: Dict[str, object] = {"return_code": rc, "stderr": stderr.strip()}
    try:
        data["result"] = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        data["parse_error"] = str(exc)
        data["raw_stdout"] = (stdout or "")[:500]
    return data


def run_checkov(bin_path: str, workspace: str, timeout_s: int) -> Dict[str, object]:
    cmd = [bin_path, "-d", workspace, "-o", "json", "--quiet"]
    rc, stdout, stderr = run_subprocess(cmd, cwd=workspace, timeout_s=timeout_s)
    data: Dict[str, object] = {"return_code": rc, "stderr": stderr.strip()}
    try:
        data["result"] = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        data["parse_error"] = str(exc)
        data["raw_stdout"] = (stdout or "")[:500]
    return data


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_jsonl)
    output_path = Path(args.output_jsonl)
    successes = load_successful_samples(results_path)
    total = len(successes)
    if total == 0:
        raise SystemExit("No validate_ok samples found in results file.")

    skipped = set()
    if args.resume:
        skipped = load_completed_samples(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with output_path.open("a", encoding="utf-8") as out_f:
        for sid in sorted(successes.keys()):
            info = successes[sid]
            if args.resume and sid in skipped:
                continue

            workspace = info["workspace"]
            entry = {
                "sample_id": sid,
                "attempt": info["attempt"],
                "workspace": workspace,
                "tools": {},
            }

            for name, runner, bin_path, timeout in [
                ("tfsec", run_tfsec, args.tfsec_bin, args.tfsec_timeout),
                ("terrascan", run_terrascan, args.terrascan_bin, args.terrascan_timeout),
                ("checkov", run_checkov, args.checkov_bin, args.checkov_timeout),
            ]:
                try:
                    entry["tools"][name] = runner(bin_path, workspace, timeout)
                except FileNotFoundError:
                    entry["tools"][name] = {"error": f"{name} not found: {bin_path}"}
                except subprocess.TimeoutExpired:
                    entry["tools"][name] = {"error": f"{name} timeout after {timeout}s"}
                except Exception as exc:
                    entry["tools"][name] = {"error": str(exc)}

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()

            processed += 1
            progress = processed / total
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
            print(f"{bar} {processed}/{total} ({progress:.1%}) - {sid}")

    print(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
