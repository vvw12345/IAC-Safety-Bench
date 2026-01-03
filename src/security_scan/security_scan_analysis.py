#!/usr/bin/env python3
"""
Aggregate并可视化 JSONL 安全扫描结果，输出 Markdown 表格。

示例：
python3 security_scan_analysis.py \
  --train-jsonl /home/IaC\ bench/data/sft_data/train_clean_security_scans.jsonl \
  --benchmark-jsonl /home/IaC\ bench/motivation/iac_eval/results/gpt-5.1/t0_0_p0_95/security_scans_report.jsonl \
  --train-label train_clean \
  --benchmark-label gpt-5.1
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

# ------------------------- 解析逻辑 ------------------------- #


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def aggregate_tfsec(entry: Dict, counter: Counter) -> None:
    tool = entry.get("tools", {}).get("tfsec")
    if not isinstance(tool, dict):
        return
    result = tool.get("result")
    if not isinstance(result, dict):
        return
    results = result.get("results")
    if not isinstance(results, list):
        return
    for item in results:
        severity = (item.get("severity") or "unknown").upper()
        counter[severity] += 1


def _extract_terrascan_violations(result: Dict) -> List[Dict]:
    if not isinstance(result, dict):
        return []
    # Terrascan结构有时在 result["results"]["violations"]
    violations = result.get("violations") or result.get("results", {}).get("violations")
    if isinstance(violations, list):
        return violations
    return []


def aggregate_terrascan(entry: Dict, counter: Counter) -> None:
    tool = entry.get("tools", {}).get("terrascan")
    if not isinstance(tool, dict):
        return
    result = tool.get("result")
    if not isinstance(result, dict):
        return
    violations = _extract_terrascan_violations(result)
    for item in violations:
        severity = (item.get("severity") or "unknown").upper()
        counter[severity] += 1


def extract_checkov_summary(result: Dict) -> Dict[str, int]:
    if not isinstance(result, dict):
        return {}
    if "summary" in result and isinstance(result["summary"], dict):
        summary = result["summary"]
    else:
        summary = result
    out: Dict[str, int] = {}
    for key in ("passed", "failed", "skipped", "parsing_errors", "resource_count"):
        value = summary.get(key)
        if isinstance(value, int):
            out[key] = value
    return out


def aggregate_checkov(entry: Dict, summary_counter: Counter) -> None:
    tool = entry.get("tools", {}).get("checkov")
    if not isinstance(tool, dict):
        return
    result = tool.get("result")
    if not isinstance(result, dict):
        return
    summary = extract_checkov_summary(result)
    for key, value in summary.items():
        summary_counter[key] += value


def tfsec_has_findings(entry: Dict) -> bool:
    tool = entry.get("tools", {}).get("tfsec")
    if not isinstance(tool, dict):
        return True
    result = tool.get("result")
    if not isinstance(result, dict):
        return True
    results = result.get("results")
    if not isinstance(results, list):
        return True
    return len(results) > 0


def terrascan_has_findings(entry: Dict) -> bool:
    tool = entry.get("tools", {}).get("terrascan")
    if not isinstance(tool, dict):
        return True
    result = tool.get("result")
    if not isinstance(result, dict):
        return True
    violations = _extract_terrascan_violations(result)
    return len(violations) > 0


def checkov_has_findings(entry: Dict) -> bool:
    tool = entry.get("tools", {}).get("checkov")
    if not isinstance(tool, dict):
        return True
    result = tool.get("result")
    if not isinstance(result, dict):
        return True
    summary = extract_checkov_summary(result)
    if not summary:
        return True
    return summary.get("failed", 0) > 0


def analyze_dataset(path: Path, label: str) -> Dict:
    stats = {
        "label": label,
        "path": str(path),
        "samples": 0,
        "tfsec": Counter(),
        "terrascan": Counter(),
        "checkov": Counter(),
        "clean_samples": 0,
    }

    for entry in load_jsonl(path):
        stats["samples"] += 1
        aggregate_tfsec(entry, stats["tfsec"])
        aggregate_terrascan(entry, stats["terrascan"])
        aggregate_checkov(entry, stats["checkov"])
        is_clean = (
            not tfsec_has_findings(entry)
            and not terrascan_has_findings(entry)
            and not checkov_has_findings(entry)
        )
        if is_clean:
            stats["clean_samples"] += 1

    return stats


# ------------------------- 输出表格 ------------------------- #


def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    border = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([border, separator, body]) if body else "\n".join([border, separator])


def build_tfsec_table(stats: List[Dict]) -> str:
    severities = set()
    for stat in stats:
        severities.update(stat["tfsec"].keys())
    severities = sorted(severities)
    headers = ["Dataset", "Samples"] + severities + ["Total Findings"]
    rows: List[List[str]] = []
    for stat in stats:
        total = sum(stat["tfsec"].values())
        row = [stat["label"], str(stat["samples"])]
        row.extend(str(stat["tfsec"].get(sev, 0)) for sev in severities)
        row.append(str(total))
        rows.append(row)
    return format_markdown_table(headers, rows)


def build_clean_table(stats: List[Dict]) -> str:
    headers = ["Dataset", "Samples", "Clean Samples", "Percentage"]
    rows: List[List[str]] = []
    for stat in stats:
        samples = stat["samples"]
        clean = stat.get("clean_samples", 0)
        pct = f"{(clean / samples * 100):.2f}%" if samples else "0%"
        rows.append([stat["label"], str(samples), str(clean), pct])
    return format_markdown_table(headers, rows)


def build_terrascan_table(stats: List[Dict]) -> str:
    severities = set()
    for stat in stats:
        severities.update(stat["terrascan"].keys())
    severities = sorted(severities)
    headers = ["Dataset", "Samples"] + severities + ["Total Violations"]
    rows: List[List[str]] = []
    for stat in stats:
        total = sum(stat["terrascan"].values())
        row = [stat["label"], str(stat["samples"])]
        row.extend(str(stat["terrascan"].get(sev, 0)) for sev in severities)
        row.append(str(total))
        rows.append(row)
    return format_markdown_table(headers, rows)


def build_checkov_table(stats: List[Dict]) -> str:
    metrics = set()
    for stat in stats:
        metrics.update(stat["checkov"].keys())
    metrics = sorted(metrics)
    headers = ["Dataset", "Samples"] + metrics
    rows: List[List[str]] = []
    for stat in stats:
        row = [stat["label"], str(stat["samples"])]
        row.extend(str(stat["checkov"].get(metric, 0)) for metric in metrics)
        rows.append(row)
    return format_markdown_table(headers, rows)


# ------------------------- CLI ------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总两份安全扫描 JSONL 并输出 Markdown 表格")
    parser.add_argument("--train-jsonl", required=True, help="train_clean 安全扫描 JSONL 路径")
    parser.add_argument("--benchmark-jsonl", required=True, help="benchmark (gpt-5.1) 安全扫描 JSONL 路径")
    parser.add_argument("--train-label", default="train_clean")
    parser.add_argument("--benchmark-label", default="gpt-5.1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [
        analyze_dataset(Path(args.train_jsonl), args.train_label),
        analyze_dataset(Path(args.benchmark_jsonl), args.benchmark_label),
    ]

    print("# 安全扫描对比表\n")
    print("## tfsec\n")
    print(build_tfsec_table(datasets))
    print("\n## Terrascan\n")
    print(build_terrascan_table(datasets))
    print("\n## Checkov\n")
    print(build_checkov_table(datasets))
    print("\n## 零漏洞样本统计\n")
    print(build_clean_table(datasets))


if __name__ == "__main__":
    main()
