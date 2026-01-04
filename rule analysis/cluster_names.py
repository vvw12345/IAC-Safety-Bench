from __future__ import annotations

from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

CSV_PATH = Path(r"h:\\基于元学习的云网络用户个性化检测\\IAC\\IAC-Safety-Bench\\rule analysis\\checkov_rules.csv")


def load_names(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df["name"].fillna("")


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


def describe_clusters(names: pd.Series, labels: np.ndarray, model: KMeans, top_terms: int = 8, samples: int = 5) -> list[str]:
    descriptions: list[str] = []
    feature_names = model.vectorizer.get_feature_names_out()  # type: ignore[attr-defined]
    centers = model.cluster_centers_
    cluster_sizes = Counter(labels)

    order = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    for cluster_id, size in order:
        top_idx = centers[cluster_id].argsort()[::-1][:top_terms]
        keywords = ", ".join(feature_names[i] for i in top_idx)
        descriptions.append(f"Cluster {cluster_id} (n={size}) | keywords: {keywords}")
        cluster_names = list(islice((name for name, label in zip(names, labels) if label == cluster_id), samples))
        for name in cluster_names:
            descriptions.append(f"  - {name}")
        descriptions.append("")
    return descriptions


def main() -> None:
    names = load_names(CSV_PATH)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    X = vectorizer.fit_transform(names)

    best_k, scores = choose_cluster_count(X)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    # attach vectorizer for convenience when describing clusters
    model.vectorizer = vectorizer  # type: ignore[attr-defined]

    print(f"Total rules: {len(names)}")
    print("Silhouette scores per k:")
    for k, score in scores.items():
        print(f"  k={k:<2} -> {score:.3f}")
    print(f"\nChosen cluster count: {best_k}\n")

    descriptions = describe_clusters(names, labels, model)
    print("\n".join(descriptions))


if __name__ == "__main__":
    main()
