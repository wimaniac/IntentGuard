"""Kiểm thử nguồn FAQ thật, ẩn tên ngân hàng và chia theo URL."""

import pandas as pd
import pytest
from scripts.prepare_real_faq_ood import INPUT, normalize_bank_names, prepare_faq, split_by_source_url

from intentguard.faq_audit import validate_faq_label_audit


def test_normalize_bank_names_preserves_product_name() -> None:
    """Chỉ ẩn thương hiệu, vẫn giữ tên chương trình hoặc sản phẩm thẻ."""
    assert normalize_bank_names("Thẻ VPBank UnionPay FreeWays có ưu đãi gì?") == "Thẻ UnionPay FreeWays có ưu đãi gì?"
    assert normalize_bank_names("Techcombank có hỗ trợ gửi tiết kiệm không?") == (
        "Ngân hàng có hỗ trợ gửi tiết kiệm không?"
    )


def test_real_faq_source_split_is_balanced_and_page_disjoint() -> None:
    """Toàn bộ 100 câu nguồn phải đi vào 50/50 mà không tách cùng URL."""
    raw = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    prepared = prepare_faq(raw, [])
    validation, test = split_by_source_url(prepared)
    assert len(validation) == len(test) == 50
    assert set(validation.source_url).isdisjoint(test.source_url)
    assert validation.topic.value_counts().between(12, 13).all()
    assert test.topic.value_counts().between(12, 13).all()
    assert prepared.source_url.nunique() == 45
    assert prepared.text_reviewed.ne(prepared.text_original).sum() == 2


def test_real_faq_rejects_invalid_source_url() -> None:
    """URL thiếu HTTPS không được dùng làm bằng chứng nguồn."""
    raw = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    raw.loc[0, "URL"] = "http://example.com/faq"
    with pytest.raises(ValueError, match="URL HTTPS"):
        prepare_faq(raw, [])


def test_review_rewrite_rejects_changed_source_question() -> None:
    """Không áp bản thu gọn cũ lên một câu nguồn đã đổi nội dung."""
    raw = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    raw.loc[90, "text"] = "Tiền hoàn của thẻ VPBank MWG được dùng như thế nào?"
    with pytest.raises(ValueError, match="Câu nguồn 91 đã thay đổi"):
        prepare_faq(raw, [])


def test_faq_label_audit_matches_current_data() -> None:
    """Bản rà 100 câu phải khớp nguyên văn và không tách cặp gần trùng."""
    base = INPUT.parents[1] / "processed/faq_ood_real"
    frames = [pd.read_parquet(base / f"{split}.parquet").assign(split=split) for split in ("validation", "test")]
    faq = pd.concat(frames, ignore_index=True)
    audit = pd.read_csv(INPUT.parents[2] / "reports/faq_ood_label_audit.csv", dtype=str, keep_default_na=False)
    assert validate_faq_label_audit(faq, audit) == {
        "approved_ood": 100,
        "focused_rewrites": 2,
        "semantic_duplicate_followups": 3,
    }
    audit.loc[0, "text_original"] = "Câu khác"
    with pytest.raises(ValueError, match="text_original"):
        validate_faq_label_audit(faq, audit)
