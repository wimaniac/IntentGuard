"""Kiểm thử tách bộ FAQ theo câu nguồn để tránh rò rỉ biến thể."""

import pandas as pd
import pytest
from scripts.curate_faq_ood import split_by_seed


def _frame() -> pd.DataFrame:
    rows = []
    for topic, count in (("LOAN", 8), ("SAVING", 7), ("INTEREST_RATE", 5), ("PROMOTION", 5)):
        for number in range(count):
            seed_id = f"{topic}:{number}"
            rows.extend({"seed_id": seed_id, "topic": topic, "text": f"Câu {index}"} for index in range(4))
    return pd.DataFrame(rows)


def test_split_by_seed_keeps_all_paraphrases_together() -> None:
    """Mỗi nhóm bốn câu chỉ xuất hiện trong một split, tổng là 48/52."""
    validation, test = split_by_seed(_frame(), seed=42)
    assert (len(validation), len(test)) == (48, 52)
    assert set(validation.seed_id).isdisjoint(test.seed_id)
    assert validation.groupby("seed_id").size().eq(4).all()
    assert test.groupby("seed_id").size().eq(4).all()


def test_split_by_seed_rejects_inconsistent_topic() -> None:
    """Một câu gắn topic khác trong cùng seed phải làm bước chia dừng lại."""
    frame = _frame()
    frame.loc[0, "topic"] = "PROMOTION"
    with pytest.raises(ValueError, match="cùng topic"):
        split_by_seed(frame)
