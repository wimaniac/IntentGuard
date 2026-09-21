"""Baseline TF-IDF word/character n-gram kết hợp Logistic Regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import FeatureUnion

from intentguard.config import dump_json


@dataclass
class BaselineBundle:
    """Artifact baseline có thể dùng để dự đoán xác suất."""

    model: Any
    label_names: list[str]
    parameters: dict[str, Any]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Tính xác suất cho danh sách văn bản.

        Args:
            texts: Danh sách câu cần phân loại.

        Returns:
            Ma trận xác suất theo thứ tự label trong artifact.
        """
        return self.model.predict_proba(texts)


def _build_model(parameters: dict[str, Any], random_state: int) -> FeatureUnion:
    word_range = tuple(parameters["word_ngram_range"])
    char_range = tuple(parameters["char_ngram_range"])
    union = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=word_range,
                    min_df=parameters["min_df"],
                    max_features=parameters["max_features"],
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=char_range,
                    min_df=parameters["min_df"],
                    max_features=parameters["max_features"],
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return union


def train_baseline(
    train: pd.DataFrame,
    selection: pd.DataFrame,
    grid_config: dict[str, list[Any]],
    label_names: list[str],
    random_state: int,
    artifact_dir: str | Path,
) -> BaselineBundle:
    """Tìm cấu hình baseline trên split model_selection và fit lại trên train.

    Args:
        train: Split huấn luyện.
        selection: Split dùng chọn tham số, không dùng fit cuối.
        grid_config: Lưới tham số lấy từ YAML.
        label_names: Tên nhãn theo chỉ số classifier.
        random_state: Seed của Logistic Regression.
        artifact_dir: Thư mục lưu model và metadata.

    Returns:
        BaselineBundle đã fit trên toàn bộ train.
    """
    best_score = float("-inf")
    best_params: dict[str, Any] | None = None
    for candidate in ParameterGrid(grid_config):
        vectorizer = _build_model(candidate, random_state)
        features = vectorizer.fit_transform(train["text"])
        classifier = LogisticRegression(
            C=candidate["c"],
            class_weight=candidate.get("class_weight"),
            max_iter=300,
            random_state=random_state,
            n_jobs=None,
        )
        classifier.fit(features, train["label"])
        selection_features = vectorizer.transform(selection["text"])
        score = f1_score(selection["label"], classifier.predict(selection_features), average="macro")
        if score > best_score:
            best_score = float(score)
            best_params = dict(candidate)
    if best_params is None:
        raise ValueError("Lưới baseline không có candidate")

    vectorizer = _build_model(best_params, random_state)
    features = vectorizer.fit_transform(train["text"])
    classifier = LogisticRegression(
        C=best_params["c"],
        class_weight=best_params.get("class_weight"),
        max_iter=300,
        random_state=random_state,
    )
    classifier.fit(features, train["label"])
    pipeline = _TextProbabilityPipeline(vectorizer, classifier)
    bundle = BaselineBundle(pipeline, label_names, {"selection_macro_f1": best_score, **best_params})
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination / "baseline.joblib")
    dump_json(
        {"backend": "baseline", "label_names": label_names, "parameters": bundle.parameters},
        destination / "baseline.json",
    )
    return bundle


class _TextProbabilityPipeline:
    """Adapter giữ vectorizer và classifier trong một artifact joblib."""

    def __init__(self, vectorizer: FeatureUnion, classifier: LogisticRegression) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Tính xác suất từ raw text.

        Args:
            texts: Danh sách văn bản.

        Returns:
            Ma trận xác suất.
        """
        return self.classifier.predict_proba(self.vectorizer.transform(texts))

    def predict(self, texts: list[str]) -> np.ndarray:
        """Dự đoán class index từ raw text.

        Args:
            texts: Danh sách văn bản.

        Returns:
            Mảng chỉ số nhãn.
        """
        return self.classifier.predict(self.vectorizer.transform(texts))
