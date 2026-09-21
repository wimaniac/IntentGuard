"""Kiểm thử các invariant của split dữ liệu và cổng chất lượng OOD."""

import json

import pandas as pd
import pytest

from intentguard.data import _stratified_partition, load_reviewed_bank_ood, normalize_text


def test_normalize_text_only_removes_whitespace() -> None:
    assert normalize_text("  a\n b ") == "a b"


def test_stratified_partition_has_no_overlap() -> None:
    frame = pd.DataFrame(
        {
            "text": [f"row-{index}" for index in range(40)],
            "label": [index % 4 for index in range(40)],
        }
    )
    parts = _stratified_partition(frame, [0.7, 0.1, 0.1, 0.1], seed=42)
    assert [len(part) for part in parts] == [28, 4, 4, 4]
    assert sum(len(part) for part in parts) == len(frame)
    assert len(set().union(*(set(part["text"]) for part in parts))) == len(frame)
    assert set(parts[0]["text"]).isdisjoint(parts[1]["text"])


def test_invalid_partition_fraction_is_rejected() -> None:
    frame = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    with pytest.raises(ValueError):
        _stratified_partition(frame, [0.6, 0.6], seed=42)


def test_bank_ood_quality_gate_blocks_invalidated_split(tmp_path) -> None:
    """Split cũ không được nạp khi nhãn OOD đang chờ audit."""
    status = tmp_path / "quality.json"
    status.write_text(json.dumps({"status": "pending_audit"}), encoding="utf-8")
    config = {"data": {"ood": {"bank_quality_file": "quality.json"}}}
    with pytest.raises(FileNotFoundError, match="kiểm tra chất lượng"):
        load_reviewed_bank_ood(config, {"root": tmp_path})
