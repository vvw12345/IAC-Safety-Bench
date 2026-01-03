#!/usr/bin/env python3
"""Security scan runner with optional LLM feedback remediation loop."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import SILICONFLOW_CONFIG, YUNWU_CONFIG

SYSTEM_PROMPT = (
    "You are an expert Infrastructure-as-Code security engineer. "
    "Given Terraform files and scan findings, you must rewrite the Terraform to eliminate the reported risks. "
    "Return JSON with a `files` array where each item has `path` and `content`."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run tfsec/Terrascan/Checkov with optional LLM remediation loop.")
    ap.add_argument("--results-jsonl", required=True, help="Path to IaC eval results.jsonl (with workspaces).")
    ap.add_argument("--output-jsonl", required=True, help="Path to write scan iterations (JSONL).")
    ap.add_argument("--tfsec-bin", default="tfsec")
    ap.add_argument("--terrascan-bin", default="terrascan")
    ap.add_argument("--checkov-bin", default="checkov")
    ap.add_argument("--tfsec-timeout", type=int, default=180)
    ap.add_argument("--terrascan-timeout", type=int, default=300)
    ap.add_argument("--checkov-timeout", type=int, default=480)
    ap.add_argument("--resume", action="store_true", help="Skip samples already present in output JSONL.")
    ap.add_argument("--feedback", action="store_true", help="Enable LLM remediation attempts when findings exist.")
    ap.add_argument("--max-feedback-iterations", type=int, default=2)
    ap.add_argument("--provider", choices=("yunwu", "siliconflow"), default="yunwu")
    ap.add_argument("--model", required=False, help="LLM model name (must exist in provider config).")
    ap.add_argument("--llm-timeout", type=int, default=120)
    ap.add_argument("--cot", action="store_true", help="Add step-by-step hint in LLM prompt.")
    ap.add_argument("--start", help="Sample ID to start processing (inclusive).")
    ap.add_argument("--iterations-jsonl", help="Optional path to log each iteration entry.")
    ap.add_argument("--max-samples", type=int, help="Limit number of samples processed in this run.")
    return ap.parse_args()


def load_successful_samples(results_path: Path) -> Dict[str, Dict[str, Any]]:
    successes: Dict[str, Dict[str, Any]] = {}
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


def load_completed_samples(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
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


def run_tfsec(bin_path: str, workspace: str, timeout_s: int) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [bin_path, "--format", "json", "--no-color", "--out", str(tmp_path), workspace]
        rc, _, stderr = run_subprocess(cmd, cwd=workspace, timeout_s=timeout_s)
        data["return_code"] = rc
        data["stderr"] = stderr.strip()
        if tmp_path.exists():
            try:
                data["result"] = json.loads(tmp_path.read_text(encoding="utf-8"))
            except Exception as exc:
                data["parse_error"] = str(exc)
        return data
    finally:
        tmp_path.unlink(missing_ok=True)


def run_terrascan(bin_path: str, workspace: str, timeout_s: int) -> Dict[str, Any]:
    cmd = [bin_path, "scan", "-o", "json", "-d", workspace]
    rc, stdout, stderr = run_subprocess(cmd, cwd=workspace, timeout_s=timeout_s)
    data: Dict[str, Any] = {"return_code": rc, "stderr": stderr.strip()}
    try:
        data["result"] = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        data["parse_error"] = str(exc)
        data["raw_stdout"] = (stdout or "")[:500]
    return data


def run_checkov(bin_path: str, workspace: str, timeout_s: int) -> Dict[str, Any]:
    cmd = [bin_path, "-d", workspace, "-o", "json", "--quiet"]
    rc, stdout, stderr = run_subprocess(cmd, cwd=workspace, timeout_s=timeout_s)
    data: Dict[str, Any] = {"return_code": rc, "stderr": stderr.strip()}
    try:
        data["result"] = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        data["parse_error"] = str(exc)
        data["raw_stdout"] = (stdout or "")[:500]
    return data


def count_tfsec_findings(result: Dict[str, Any]) -> int:
    payload = result.get("result") or {}
    return len(payload.get("results") or [])


def count_terrascan_findings(result: Dict[str, Any]) -> int:
    payload = result.get("result") or {}
    summary = payload.get("results") or []
    total = 0
    for entry in summary:
        if isinstance(entry, dict):
            total += len(entry.get("violations") or [])
    if not summary and isinstance(payload, dict):
        total += len(payload.get("violations") or [])
    return total


def count_checkov_findings(result: Dict[str, Any]) -> int:
    payload = result.get("result") or {}
    total = 0

    def collect(obj: Any) -> int:
        if not isinstance(obj, dict):
            return 0
        res = obj.get("results") or {}
        if isinstance(res, dict):
            return len(res.get("failed_checks") or [])
        if isinstance(res, list):
            return sum(collect(item) for item in res)
        return 0

    if isinstance(payload, list):
        total = sum(collect(item) for item in payload)
    else:
        total = collect(payload)
    return total


def total_findings(scan_entry: Dict[str, Any]) -> int:
    total = 0
    total += count_tfsec_findings(scan_entry.get("tfsec", {}))
    total += count_terrascan_findings(scan_entry.get("terrascan", {}))
    total += count_checkov_findings(scan_entry.get("checkov", {}))
    return total


def summarize_findings(scan_entry: Dict[str, Any]) -> str:
    tfsec_count = count_tfsec_findings(scan_entry.get("tfsec", {}))
    terrascan_count = count_terrascan_findings(scan_entry.get("terrascan", {}))
    checkov_count = count_checkov_findings(scan_entry.get("checkov", {}))
    parts = [
        f"tfsec: {tfsec_count}",
        f"Terrascan: {terrascan_count}",
        f"Checkov: {checkov_count}",
    ]
    return ", ".join(parts)


def gather_tf_sources(workspace: Path) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    for file_path in workspace.rglob("*.tf"):
        rel = str(file_path.relative_to(workspace))
        files.append((rel, file_path.read_text(encoding="utf-8")))
    return files


def build_feedback_prompt(code_files: List[Tuple[str, str]], finding_summary: str, cot: bool) -> str:
    code_bundle = []
    for rel, content in code_files:
        code_bundle.append(f"### {rel}\n{content.strip()}\n")
    cot_hint = ("\nThink step-by-step before returning the final JSON, but do not expose the reasoning." if cot else "")
    return (
        "You will receive Terraform files that contain security issues reported by tfsec/Terrascan/Checkov. "
        "Produce a remediated version addressing the findings. "
        "Output JSON ONLY: {\"files\": [{\"path\": \"<relative-path>\", \"content\": \"<full file content>\"}], "
        "\"notes\": \"<optional summary>\"}. "
        "Each files[].content must contain ONLY Terraform HCL code, wrapped inside a ```hcl ... ``` fenced block, "
        "with no prose before or after the JSON."
        f" Findings summary: {finding_summary}.{cot_hint}\n\n"
        "Terraform files (read-only for context):\n"
        + "\n".join(code_bundle)
    )


HCL_BLOCK_RE = re.compile(r"```(?:hcl|terraform|tf)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_hcl_snippet(text: str) -> Optional[str]:
    if not text:
        return None
    match = HCL_BLOCK_RE.search(text)
    if match:
        snippet = match.group(1).strip()
        if snippet:
            return snippet
    return None


def sanitize_hcl_content(content: str) -> str:
    snippet = extract_hcl_snippet(content)
    if snippet:
        return snippet
    return content.strip()


def choose_primary_tf_target(workspace: Path) -> Path:
    for path in sorted(workspace.rglob("*.tf")):
        return path
    return workspace / "main.tf"


def choose_api_key(keys: Iterable[str]) -> str:
    pool = [k for k in keys if k]
    if not pool:
        raise RuntimeError("No API keys configured.")
    import random

    return random.choice(pool)


def call_llm(
    provider: str,
    model: str,
    user_prompt: str,
    timeout_s: int,
    cot: bool,
) -> Tuple[str, Dict[str, Any]]:
    if provider == "yunwu":
        config = YUNWU_CONFIG
    else:
        config = SILICONFLOW_CONFIG
    if model and model not in (config.get("models") or []):
        raise ValueError(f"model {model} not available in provider {provider}")
    
    # 根据不同 provider 构造不同的 payload
    base_payload = {
        "model": model or (config.get("models") or [None])[0],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
    }
    
    # SiliconFlow 特殊处理：只有当 stream=true 时才能使用 stream_options
    if provider == "siliconflow":
        payload = base_payload
    else:  # yunwu 保持原有逻辑
        payload = {**base_payload, "stream_options": {"include_usage": True}}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {choose_api_key(config.get('api_keys') or [])}",
    }
    import urllib.request

    req = urllib.request.Request(
        config["base_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = raw.get("usage") or {}
        return content, usage


def apply_llm_fix(workspace: Path, response_text: str) -> Tuple[bool, str]:
    def apply_single_snippet(snippet: str) -> Tuple[bool, str]:
        target = choose_primary_tf_target(workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(snippet, encoding="utf-8")
        return True, f"applied snippet to {target.name}"

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        snippet = extract_hcl_snippet(response_text)
        if snippet:
            return apply_single_snippet(snippet)
        return False, f"invalid json: {exc}"
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return False, "response missing files"
    for file_entry in files:
        rel = file_entry.get("path")
        content = file_entry.get("content")
        if not rel or content is None:
            return False, "file entry missing path/content"
        content = sanitize_hcl_content(str(content))
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return True, payload.get("notes", "")


def run_scan(workspace: Path, args: argparse.Namespace) -> Dict[str, Any]:
    entry: Dict[str, Any] = {}
    entry["tfsec"] = run_tfsec(args.tfsec_bin, str(workspace), args.tfsec_timeout)
    entry["terrascan"] = run_terrascan(args.terrascan_bin, str(workspace), args.terrascan_timeout)
    entry["checkov"] = run_checkov(args.checkov_bin, str(workspace), args.checkov_timeout)
    return entry


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_jsonl)
    output_path = Path(args.output_jsonl)
    iter_log_path = Path(args.iterations_jsonl) if args.iterations_jsonl else None
    successes = load_successful_samples(results_path)
    if not successes:
        raise SystemExit("No validate_ok samples in results file.")
    ordered_ids = sorted(successes.keys())
    id_to_idx = {sid: idx for idx, sid in enumerate(ordered_ids)}
    if args.start and args.start not in id_to_idx:
        raise SystemExit(f"--start sample_id '{args.start}' not found in {results_path}")
    skipped = load_completed_samples(output_path) if args.resume else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = len(ordered_ids)
    processed = id_to_idx.get(args.start, 0)
    handled = 0
    started = args.start is None
    if iter_log_path:
        iter_log_path.parent.mkdir(parents=True, exist_ok=True)
    iter_log_fh = iter_log_path.open("a", encoding="utf-8") if iter_log_path else None
    try:
        with output_path.open("a", encoding="utf-8") as out_f:
            for sid in ordered_ids:
                if not started:
                    if sid == args.start:
                        started = True
                    else:
                        continue
                if args.resume and sid in skipped:
                    processed += 1
                    continue
                if args.max_samples is not None and handled >= args.max_samples:
                    break
                info = successes[sid]
                original_workspace = Path(info["workspace"])
                sample_tmp = Path(tempfile.mkdtemp(prefix=f"scan_{sid}_"))
                shutil.copytree(original_workspace, sample_tmp, dirs_exist_ok=True)

                entry: Dict[str, Any] = {
                    "sample_id": sid,
                    "attempt": info["attempt"],
                    "meta": info.get("meta", {}),
                    "iterations": [],
                    "feedback_enabled": args.feedback,
                }
                success = False

                for iteration in range(max(1, args.max_feedback_iterations if args.feedback else 1)):
                    scan_result = run_scan(sample_tmp, args)
                    finding_total = total_findings(scan_result)
                    iter_payload = {
                        "iteration": iteration + 1,
                        "scan": scan_result,
                        "finding_count": finding_total,
                    }
                    entry["iterations"].append(iter_payload)

                    def log_iteration_payload() -> None:
                        if not iter_log_fh:
                            return
                        iter_log_fh.write(
                            json.dumps(
                                {
                                    "sample_id": sid,
                                    "attempt": info["attempt"],
                                    "iteration_payload": iter_payload,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        iter_log_fh.flush()

                    if finding_total == 0:
                        log_iteration_payload()
                        success = True
                        break
                    if not args.feedback or iteration + 1 >= args.max_feedback_iterations:
                        log_iteration_payload()
                        break

                    code_files = gather_tf_sources(sample_tmp)
                    prompt = build_feedback_prompt(code_files, summarize_findings(scan_result), args.cot)
                    try:
                        response_text, usage = call_llm(
                            provider=args.provider,
                            model=args.model,
                            user_prompt=prompt,
                            timeout_s=args.llm_timeout,
                            cot=args.cot,
                        )
                        ok, note = apply_llm_fix(sample_tmp, response_text)
                        iter_payload["feedback"] = {
                            "ok": ok,
                            "notes": note,
                            "raw_response": response_text,
                            "usage": usage,
                        }
                        if not ok:
                            log_iteration_payload()
                            continue
                    except Exception as exc:
                        iter_payload["feedback_error"] = str(exc)
                        log_iteration_payload()
                        continue

                    log_iteration_payload()

                entry["final_zero_vuln"] = success
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()
                shutil.rmtree(sample_tmp, ignore_errors=True)

                processed += 1
                handled += 1
                progress = processed / total_samples if total_samples else 1.0
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
                print(f"{bar} {processed}/{total_samples} ({progress:.1%}) - {sid} (zero_vuln={success})")

    finally:
        if iter_log_fh:
            iter_log_fh.close()

    print(f"Done. Detailed log saved to {output_path}")


if __name__ == "__main__":
    main()
