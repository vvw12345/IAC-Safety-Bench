from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

CSV_PATH = Path(r"h:\\基于元学习的云网络用户个性化检测\\IAC\\IAC-Safety-Bench\\rule analysis\\checkov_rules.csv")
CLUSTER_ASSIGNMENTS_PATH = Path(r"h:\\基于元学习的云网络用户个性化检测\\IAC\\IAC-Safety-Bench\\rule analysis\\rule_cluster_assignments.csv")
VALIDATION_REPORT_PATH = Path(r"h:\\基于元学习的云网络用户个性化检测\\IAC\\IAC-Safety-Bench\\rule analysis\\category_validation.txt")

NETWORK_KEYWORDS = (
    "public",
    "internet",
    "network",
    "exposure",
    "ip",
    "cidr",
    "port",
    "endpoint",
    "firewall",
    "gateway",
    "vpn",
    "vpc",
    "ingress",
    "egress",
    "route",
)
DATA_PRIVACY_KEYWORDS = (
    "encrypt",
    "encryption",
    "encrypted",
    "kms",
    "key management",
    "kms key",
    "ssl",
    "tls",
    "certificate",
    "privacy",
    "data masking",
    "data retention",
)
HARDCODED_SECRETS_KEYWORDS = (
    "hard coded",
    "hard-coded",
    "hardcoded",
    "secret",
    "credential",
    "token",
    "apikey",
    "api key",
    "access key",
)

CATEGORY_ORDER = [
    "Network Exposure",
    "Data Privacy",
    "Hardcoded Secrets",
    "Compliance",
]


@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    dominant_category: str
    dominant_ratio: float
    keywords: str
    samples: list[str]


def choose_cluster_count(X) -> tuple[int, dict[int, float]]:
    candidates = [8, 10, 12, 15, 18, 22]
    scores: dict[int, float] = {}
    best_k = candidates[0]
    best_score = float("-inf")
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        if len(set(labels)) == 1:
            score = float("-inf")
        else:
            score = silhouette_score(X, labels)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores


def contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def assign_category(name: str) -> str:
    text = name.lower()
    if contains_keyword(text, NETWORK_KEYWORDS):
        return "Network Exposure"
    if contains_keyword(text, DATA_PRIVACY_KEYWORDS):
        return "Data Privacy"
    if contains_keyword(text, HARDCODED_SECRETS_KEYWORDS):
        return "Hardcoded Secrets"
    return "Compliance"


def summarize_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    model: KMeans,
    samples: int = 4,
) -> list[ClusterSummary]:
    feature_names = vectorizer.get_feature_names_out()
    centers = model.cluster_centers_
    summaries: list[ClusterSummary] = []

    for cluster_id in range(model.n_clusters):
        mask = labels == cluster_id
        size = int(mask.sum())
        category_counts = df.loc[mask, "paper_category"].value_counts()
        dominant_category = category_counts.idxmax()
        dominant_ratio = float(category_counts.max() / size)
        top_idx = centers[cluster_id].argsort()[::-1][:8]
        keywords = ", ".join(feature_names[i] for i in top_idx)
        cluster_samples = df.loc[mask, "name"].head(samples).tolist()
        summaries.append(
            ClusterSummary(
                cluster_id=cluster_id,
                size=size,
                dominant_category=dominant_category,
                dominant_ratio=dominant_ratio,
                keywords=keywords,
                samples=cluster_samples,
            )
        )
    return summaries


def write_report(
    df: pd.DataFrame,
    scores: dict[int, float],
    best_k: int,
    summaries: list[ClusterSummary],
    report_path: Path,
) -> None:
    category_counts = df["paper_category"].value_counts()
    avg_purity = mean(summary.dominant_ratio for summary in summaries)

    lines: list[str] = []
    lines.append(f"Total rules: {len(df)}")
    lines.append("Silhouette scores per k:")
    for k, score in scores.items():
        lines.append(f"  k={k:<2} -> {score:.3f}")
    lines.append("")
    lines.append(f"Chosen cluster count: {best_k}")
    lines.append(f"Average cluster purity vs. paper categories: {avg_purity:.2f}")
    lines.append("")
    lines.append("Paper category coverage:")
    for category in CATEGORY_ORDER:
        count = int(category_counts.get(category, 0))
        pct = count / len(df)
        lines.append(f"  - {category}: {count} rules ({pct:.1%})")
    lines.append("")
    lines.append("Cluster alignment:")
    for summary in sorted(summaries, key=lambda s: s.size, reverse=True):
        lines.append(
            f"Cluster {summary.cluster_id:02d} (n={summary.size}) -> {summary.dominant_category} "
            f"[{summary.dominant_ratio:.0%}] | keywords: {summary.keywords}"
        )
        for sample in summary.samples:
            lines.append(f"    - {sample}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["name"] = df["name"].fillna("")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    X = vectorizer.fit_transform(df["name"].str.lower())

    best_k, scores = choose_cluster_count(X)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    df["cluster"] = labels
    df["paper_category"] = df["name"].map(assign_category)
    df.to_csv(CLUSTER_ASSIGNMENTS_PATH, index=False)

    summaries = summarize_clusters(df, labels, vectorizer, model)
    write_report(df, scores, best_k, summaries, VALIDATION_REPORT_PATH)


if __name__ == "__main__":
    main()
