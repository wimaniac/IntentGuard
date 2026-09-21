"""Các kiểu dữ liệu ổn định giữa mô hình, hiệu chỉnh và giao diện."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TopPrediction:
    """Một nhãn ứng viên và xác suất sau hiệu chỉnh."""

    intent: str
    probability: float


@dataclass(frozen=True)
class SimilarCase:
    """Một câu trong tập train gần với truy vấn theo cosine similarity."""

    text: str
    intent: str
    similarity: float


@dataclass(frozen=True)
class InferenceResult:
    """Kết quả suy luận chuẩn dùng chung cho API và Streamlit.

    Attributes:
        intent: Nhãn dự đoán hoặc chuỗi UNKNOWN theo metadata model.
        confidence: Xác suất top-1 sau temperature scaling.
        top_3: Ba nhãn có xác suất cao nhất, không khẳng định nhãn khi UNKNOWN.
        is_unknown: Cờ cho biết confidence thấp hơn threshold.
        similar_cases: Các mẫu train gần nhất để tham khảo.
    """

    intent: str
    confidence: float
    top_3: list[TopPrediction] = field(default_factory=list)
    is_unknown: bool = False
    similar_cases: list[SimilarCase] = field(default_factory=list)


@dataclass
class DatasetBundle:
    """Các split in-domain và metadata nhãn sau khi chuẩn hoá."""

    train: Any
    model_selection: Any
    calibration: Any
    threshold: Any
    test: Any
    label_names: list[str]
    metadata: dict[str, Any]
