"""API suy luận thống nhất cho baseline/DL, UNKNOWN và câu tương tự."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intentguard.calibration import apply_temperature
from intentguard.retrieval import mean_pool, query_index, query_sparse_index
from intentguard.schemas import InferenceResult, SimilarCase, TopPrediction


class IntentGuardPredictor:
    """Facade suy luận được dùng bởi CLI, test và Streamlit."""

    def __init__(self, artifact_dir: str | Path, backend: str = "deep") -> None:
        """Nạp model, calibration, threshold và kho câu train.

        Args:
            artifact_dir: Thư mục artifact sau pipeline train.
            backend: ``deep`` hoặc ``baseline``.
        """
        self.artifact_dir = Path(artifact_dir)
        self.backend = backend
        calibration_path = self.artifact_dir / f"calibration_{backend}.json"
        if not calibration_path.exists():
            calibration_path = self.artifact_dir / "calibration.json"
        self.calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        model_metadata = json.loads((self.artifact_dir / f"{backend}.json").read_text(encoding="utf-8"))
        self.label_names = list(model_metadata["label_names"])
        self.top_k = int(model_metadata.get("top_k", 3))
        self.similar_k = int(model_metadata.get("similar_k", 3))
        self._deep_model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._baseline: Any = None
        self.train_cases = (
            pd.read_parquet(self.artifact_dir / "train_cases.parquet")
            if (self.artifact_dir / "train_cases.parquet").exists()
            else pd.DataFrame()
        )
        self.train_vectors: Any = None
        if backend == "deep":
            from intentguard.models.deep import load_deep_model

            self._deep_model, self._tokenizer, self._device, deep_metadata = load_deep_model(self.artifact_dir)
            self.max_length = int(deep_metadata["config"]["max_length"])
            self.batch_size = int(deep_metadata["config"]["batch_size"])
            self.train_embeddings = np.load(self.artifact_dir / "train_embeddings.npy")
        elif backend == "baseline":
            import joblib

            self._baseline = joblib.load(self.artifact_dir / "baseline.joblib")
            self.train_embeddings = None
            if not self.train_cases.empty:
                self.train_vectors = self._baseline.model.vectorizer.transform(self.train_cases["text"].astype(str))
        else:
            raise ValueError(f"Backend không hỗ trợ: {backend}")

    def _raw_probabilities(self, texts: list[str]) -> np.ndarray:
        if self.backend == "baseline":
            return self._baseline.predict_proba(texts)
        frame = pd.DataFrame({"text": texts, "label": [0] * len(texts)})
        from intentguard.models.deep import predict_probabilities

        return predict_probabilities(
            self._deep_model,
            self._tokenizer,
            frame,
            self.max_length,
            self.batch_size,
            self._device,
        )

    def _similar_cases(self, text: str) -> list[SimilarCase]:
        if self.train_cases.empty:
            return []
        if self.backend == "baseline":
            query = self._baseline.model.vectorizer.transform([text])
            return query_sparse_index(query, self.train_vectors, self.train_cases, self.similar_k)
        if self.train_embeddings is None:
            return []
        import torch

        encoded = self._tokenizer(
            [text], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self._deep_model.encoder(**encoded)
            embedding = mean_pool(output.last_hidden_state, encoded["attention_mask"]).cpu().numpy()[0]
        return query_index(embedding, self.train_embeddings, self.train_cases, self.similar_k)

    def predict(self, text: str) -> InferenceResult:
        """Phân loại một tin nhắn theo artifact đã hiệu chỉnh.

        Args:
            text: Tin nhắn tiếng Việt cần phân loại.

        Returns:
            InferenceResult gồm top-3, UNKNOWN và câu tương tự.

        Raises:
            ValueError: Khi text rỗng.
        """
        if not str(text).strip():
            raise ValueError("text không được rỗng")
        raw = self._raw_probabilities([str(text)])[0]
        probabilities = apply_temperature(raw[None, :], float(self.calibration["temperature"]))[0]
        order = np.argsort(-probabilities)[: self.top_k]
        top_3 = [TopPrediction(self.label_names[index], float(probabilities[index])) for index in order]
        confidence = float(probabilities[order[0]])
        is_unknown = confidence < float(self.calibration["threshold"])
        return InferenceResult(
            intent="UNKNOWN" if is_unknown else top_3[0].intent,
            confidence=confidence,
            top_3=top_3,
            is_unknown=is_unknown,
            similar_cases=self._similar_cases(str(text)),
        )
