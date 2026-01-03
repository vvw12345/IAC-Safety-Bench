#!/usr/bin/env python3
"""
对没有工作目录 / validate_ok 标记的 JSONL 数据（如 poison_pool_llm.jsonl）执行安全扫描。

用法示例：
python scan_poison_jsonl.py \
  --input-jsonl /home/IaC bench/data/sft_data/poison_pool_llm.jsonl \
  --output-jsonl /home/IaC bench/data/sft_data/poison_pool_llm_security_scans.jsonl \
  --workspace-root /home/IaC bench/tmp/poison_scan_workspaces \
  --tfsec-bin /home/linuxbrew/.linuxbrew/bin/tfsec \
  --terrascan-bin /home/linuxbrew/.linuxbrew/bin/terrascan \
  --checkov-bin /home/linuxbrew/.linuxbrew/bin/checkov
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描 JSONL 中的 Terraform 代码输出。")
    parser.add_argument("--input-jsonl", required=True, help="包含 Terraform 代码 (output 字段) 的 JSONL 文件。")
    parser.add_argument("--output-jsonl", required=True, help="扫描结果输出 JSONL。")
    parser.add_argument(
        "--workspace-root",
        default="/home/IaC bench/tmp/poison_scan_workspaces",
        help="临时写入 main.tf 的目录，将为每个样本创建子目录。",
    )
    parser.add_argument("--tfsec-bin", default="tfsec", help="tfsec 可执行文件路径。")
    parser.add_argument("--terrascan-bin", default="terrascan", help="terrascan 可执行文件路径。")
    parser.add_argument("--checkov-bin", default="checkov", help="checkov 可执行文件路径。")
    parser.add_argument("--tfsec-timeout", type=int, default=180)
    parser.add_argument("--terrascan-timeout", type=int, default=300)
    parser.add_argument("--checkov-timeout", type=int, default=480)
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="保留生成的工作目录（默认扫描完成后删除以节省空间）。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅扫描前 N 条记录，默认全部。",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return safe[:120] if len(safe) > 120 else safe


def run_tfsec(bin_path: str, workspace: Path, timeout_s: int) -> Dict[str, object]:
    data: Dict[str, object] = {}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [bin_path, "--format", "json", "--no-color", "--out", str(tmp_path), str(workspace)]
        proc = subprocess.run(
            cmd,
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
        data["return_code"] = proc.returncode
        data["stderr"] = proc.stderr.strip()
        if tmp_path.exists():
            try:
                data["result"] = json.loads(tmp_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                data["parse_error"] = str(exc)
        return data
    finally:
        tmp_path.unlink(missing_ok=True)


def run_terrascan(bin_path: str, workspace: Path, timeout_s: int) -> Dict[str, object]:
    cmd = [bin_path, "scan", "-o", "json", "-d", str(workspace)]
    proc = subprocess.run(
        cmd,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    data: Dict[str, object] = {"return_code": proc.returncode, "stderr": proc.stderr.strip()}
    try:
        data["result"] = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        data["parse_error"] = str(exc)
        data["raw_stdout"] = (proc.stdout or "")[:500]
    return data


def run_checkov(bin_path: str, workspace: Path, timeout_s: int) -> Dict[str, object]:
    cmd = [bin_path, "-d", str(workspace), "-o", "json", "--quiet"]
    proc = subprocess.run(
        cmd,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    data: Dict[str, object] = {"return_code": proc.returncode, "stderr": proc.stderr.strip()}
    try:
        data["result"] = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        data["parse_error"] = str(exc)
        data["raw_stdout"] = (proc.stdout or "")[:500]
    return data


def extract_code(entry: Dict[str, object]) -> Optional[str]:
    for key in ("output", "code", "response"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_workspace(root: Path, sample_id: str, code: str) -> Path:
    workspace = root / sample_id
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "main.tf").write_text(code.strip() + "\n", encoding="utf-8")
    return workspace


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    workspace_root = Path(args.workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    # 预先统计总行数，便于显示进度条
    total_entries = 0
    with input_path.open("r", encoding="utf-8") as src:
        for _ in src:
            total_entries += 1
    if args.limit:
        total_entries = min(total_entries, args.limit)

    processed = 0
    written = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            code = extract_code(entry)
            if not code:
                continue

            sample_id = entry.get("source_id") or entry.get("sample_id") or f"sample_{processed}"
            safe_name = sanitize_name(str(sample_id))
            workspace = build_workspace(workspace_root, safe_name, code)

            record = {
                "sample_id": sample_id,
                "workspace": str(workspace),
                "tools": {},
            }

            try:
                record["tools"]["tfsec"] = run_tfsec(args.tfsec_bin, workspace, args.tfsec_timeout)
            except FileNotFoundError:
                record["tools"]["tfsec"] = {"error": f"tfsec not found: {args.tfsec_bin}"}
            except subprocess.TimeoutExpired:
                record["tools"]["tfsec"] = {"error": f"tfsec timeout after {args.tfsec_timeout}s"}

            try:
                record["tools"]["terrascan"] = run_terrascan(args.terrascan_bin, workspace, args.terrascan_timeout)
            except FileNotFoundError:
                record["tools"]["terrascan"] = {"error": f"terrascan not found: {args.terrascan_bin}"}
            except subprocess.TimeoutExpired:
                record["tools"]["terrascan"] = {"error": f"terrascan timeout after {args.terrascan_timeout}s"}

            try:
                record["tools"]["checkov"] = run_checkov(args.checkov_bin, workspace, args.checkov_timeout)
            except FileNotFoundError:
                record["tools"]["checkov"] = {"error": f"checkov not found: {args.checkov_bin}"}
            except subprocess.TimeoutExpired:
                record["tools"]["checkov"] = {"error": f"checkov timeout after {args.checkov_timeout}s"}

            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if not args.keep_workspaces:
                shutil.rmtree(workspace, ignore_errors=True)

            processed += 1
            if args.limit and processed >= args.limit:
                break

            if total_entries:
                progress = processed / total_entries
                print(
                    f"\r[scan_poison_jsonl] {processed}/{total_entries} ({progress:.1%}) processed",
                    end="",
                    flush=True,
                )

    if total_entries:
        print()
    print(f"[scan_poison_jsonl] completed. processed={processed}, results={written}, saved to {output_path}")


if __name__ == "__main__":
    main()
