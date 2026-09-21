"""Kiểm thử retrieval chỉ trả các mẫu có trong index train."""

import numpy as np
import pandas as pd

from intentguard.retrieval import normalize_rows, query_index


def test_query_index_returns_nearest_train_cases_in_order() -> None:
    cases = pd.DataFrame({"text": ["train-a", "train-b", "train-c"], "intent": ["a", "b", "c"]})
    embeddings = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    result = query_index(np.array([0.8, 0.2]), embeddings, cases, k=2)
    assert [item.text for item in result] == ["train-a", "train-b"]
    assert all(item.text in set(cases["text"]) for item in result)
