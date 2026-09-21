"""Xây dựng và truy vấn kho câu tương tự bằng embedding của encoder fine-tune."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from intentguard.schemas import SimilarCase


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mean-pool các wordpiece có attention mask bằng một phép tính vector hoá."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Chuẩn hoá từng vector về norm 1, bảo vệ vector zero."""
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def build_index(
    texts: list[str],
    labels: list[int],
    label_names: list[str],
    embedder: Any,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    device: Any,
    artifact_dir: str | Path,
) -> None:
    """Tạo embedding index chỉ từ split train và lưu cùng câu/intent.

    Args:
        texts: Các câu của split train.
        labels: Class index tương ứng.
        label_names: Tên nhãn theo class index.
        embedder: Encoder transformer đã fine-tune.
        tokenizer: Tokenizer tương ứng.
        max_length: Độ dài wordpiece tối đa.
        batch_size: Kích thước batch embedding.
        device: Torch device.
        artifact_dir: Thư mục artifact model.
    """
    import torch

    embedder.eval()
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = embedder(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            vectors.append(pooled.detach().cpu().numpy())
    embeddings = normalize_rows(np.concatenate(vectors, axis=0))
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "train_embeddings.npy", embeddings)
    pd.DataFrame({"text": texts, "label": labels, "intent": [label_names[label] for label in labels]}).to_parquet(
        destination / "train_cases.parquet", index=False
    )


def query_index(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    cases: pd.DataFrame,
    k: int,
) -> list[SimilarCase]:
    """Trả về k câu gần nhất theo cosine similarity.

    Args:
        query_embedding: Vector truy vấn chưa hoặc đã chuẩn hoá.
        embeddings: Matrix embedding chỉ gồm train.
        cases: DataFrame có cột text và intent.
        k: Số kết quả cần trả.

    Returns:
        Danh sách SimilarCase giảm dần theo similarity.
    """
    if len(embeddings) != len(cases):
        raise ValueError("Số embedding không khớp số câu train")
    query = query_embedding.reshape(1, -1)
    query = normalize_rows(query)[0]
    scores = embeddings @ query
    order = np.argsort(-scores)[:k]
    return [
        SimilarCase(
            text=str(cases.iloc[index]["text"]),
            intent=str(cases.iloc[index]["intent"]),
            similarity=float(scores[index]),
        )
        for index in order
    ]


def query_sparse_index(query: Any, vectors: Any, cases: pd.DataFrame, k: int) -> list[SimilarCase]:
    """Trả k câu train gần nhất theo cosine trên TF-IDF của baseline.

    Args:
        query: Vector TF-IDF của một câu hỏi.
        vectors: Ma trận TF-IDF của các câu train cùng thứ tự với cases.
        cases: Bảng câu train và intent.
        k: Số câu cần lấy.

    Returns:
        Danh sách câu tương tự giảm dần theo cosine.
    """
    if vectors.shape[0] != len(cases):
        raise ValueError("Số vector TF-IDF không khớp số câu train")
    scores = cosine_similarity(query, vectors).ravel()
    order = np.argsort(-scores)[:k]
    return [
        SimilarCase(
            text=str(cases.iloc[index]["text"]),
            intent=str(cases.iloc[index]["intent"]),
            similarity=float(scores[index]),
        )
        for index in order
    ]
