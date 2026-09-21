"""Kiểm tra 100 FAQ ngân hàng có URL và chia tập OOD theo trang nguồn."""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INPUT = ROOT / "data/raw/vietnam_banks_faq_100.csv"
OUTPUT = ROOT / "data/processed/faq_ood_real"
IN_DOMAIN = ROOT / "data/processed/in_domain"
TOPICS = {"LOAN", "SAVING", "INTEREST_RATE", "PROMOTION"}
BANK_NAMES = re.compile(r"\b(?:Techcombank|VPBank|MSB|ACB|VIB|OCB)\b", re.IGNORECASE)
BANK_PATTERN = r"(?:Techcombank|VPBank|MSB|ACB|VIB|OCB)"
# Hai câu gốc hỏi hai quyền lợi khác nhau; bản đánh giá giữ một nhu cầu đã có trong câu nguồn.
REVIEW_REWRITES = {
    91: (
        "Khách hàng dùng thẻ VPBank MWG nhận và sử dụng tiền hoàn như thế nào?",
        "Khách hàng dùng thẻ VPBank MWG sử dụng tiền hoàn như thế nào?",
    ),
    97: (
        "Điều kiện nhận thiết kế thẻ giới hạn và quà tặng T1 của VPBank GameON là gì?",
        "Điều kiện nhận quà tặng T1 của VPBank GameON là gì?",
    ),
}


def _match_key(value: object) -> str:
    """Tạo khóa Unicode để kiểm tra câu trùng sau khi bỏ dấu câu và khoảng trắng thừa."""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE).strip()


def _duplicate_groups(texts: list[str], similarity: float = 0.92) -> list[int]:
    """Nhóm câu trùng hoặc gần trùng để không tách chúng sang hai split khác nhau."""
    parent = list(range(len(texts)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left: int, right: int) -> None:
        parent[root(right)] = root(left)

    keys = [_match_key(value) for value in texts]
    seen: dict[str, int] = {}
    for index, key in enumerate(keys):
        if key in seen:
            join(index, seen[key])
        else:
            seen[key] = index
    eligible = [index for index, key in enumerate(keys) if len(key) >= 20]
    if len(eligible) >= 2:
        vectors = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform([keys[i] for i in eligible])
        scores = cosine_similarity(vectors)
        for row in range(len(eligible)):
            for column in np.flatnonzero(scores[row, row + 1 :] >= similarity) + row + 1:
                join(eligible[row], eligible[int(column)])
    return [root(index) for index in range(len(texts))]

def normalize_bank_names(value: str) -> str:
    """Ẩn tên ngân hàng trong câu hỏi nhưng giữ tên sản phẩm và nội dung.

    Args:
        value: Câu hỏi FAQ nguyên gốc.

    Returns:
        Câu hỏi đã thay tên ngân hàng bằng danh từ chung và gọn khoảng trắng.
    """
    text = " ".join(value.split())
    text = re.sub(r"\bngân hàng\s+(?:Techcombank|VPBank|MSB|ACB|VIB|OCB)\b", "ngân hàng", text, flags=re.I)
    text = re.sub(r"\b(thẻ(?: tín dụng)?)\s+(?:Techcombank|VPBank|MSB|ACB|VIB|OCB)\b", r"\1", text, flags=re.I)
    text = re.sub(rf"\b(khách hàng(?: ưu tiên)?)\s+{BANK_PATTERN}\b", r"\1 của ngân hàng", text, flags=re.I)
    text = re.sub(
        rf"\b((?:gửi )?tiết kiệm(?: online| trực tuyến)?|sổ tiết kiệm(?: online| trực tuyến)?|"
        rf"khoản vay|vay tín chấp|tiền gửi)\s+{BANK_PATTERN}\b",
        r"\1 tại ngân hàng",
        text,
        flags=re.I,
    )
    text = re.sub(rf"\b{BANK_PATTERN}\s+(?=GameON\b)", "", text, flags=re.I)
    text = BANK_NAMES.sub("ngân hàng", text)
    if text.startswith("ngân hàng"):
        text = "Ngân hàng" + text[len("ngân hàng"):]
    return " ".join(text.split())


def prepare_faq(frame: pd.DataFrame, domain_texts: list[str]) -> pd.DataFrame:
    """Kiểm tra schema, nội dung, URL và trùng lặp trước khi gắn ID nguồn.

    Args:
        frame: CSV người dùng cung cấp với `text`, `topic`, `URL`.
        domain_texts: Tất cả câu in-domain để kiểm tra trùng chính xác.

    Returns:
        FAQ có câu gốc, câu ẩn tên ngân hàng, topic, URL và nhóm trang.

    Raises:
        ValueError: Khi số lượng, nội dung, URL, topic hoặc trùng lặp không đạt.
    """
    required = {"text", "topic", "URL"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV thiếu cột: {sorted(required - set(frame.columns))}")
    if len(frame) != 100:
        raise ValueError(f"Cần đúng 100 FAQ nguồn; nhận được {len(frame)}")
    result = frame[["text", "topic", "URL"]].copy().reset_index(drop=True)
    for column in result.columns:
        result[column] = result[column].astype(str).str.strip()
    if result.isna().any().any() or result.eq("").any().any():
        raise ValueError("FAQ có ô text/topic/URL trống")
    if not set(result.topic).issubset(TOPICS) or set(result.topic) != TOPICS:
        raise ValueError("FAQ cần đủ bốn topic LOAN, SAVING, INTEREST_RATE, PROMOTION")
    if not result.groupby("topic").size().eq(25).all():
        raise ValueError("Mỗi topic cần đúng 25 câu")
    if not result.text.str.len().between(15, 300).all():
        raise ValueError("FAQ có câu ngoài giới hạn 15–300 ký tự")
    if result.text.str.contains(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{7,}\b|https?://", regex=True).any():
        raise ValueError("Nội dung câu hỏi có email, chuỗi số dài hoặc URL")
    if not result.URL.map(lambda value: urlparse(value).scheme == "https" and bool(urlparse(value).netloc)).all():
        raise ValueError("Mỗi FAQ cần URL HTTPS hợp lệ về cú pháp")
    result = result.rename(columns={"text": "text_original", "URL": "source_url"})
    result["text_reviewed"] = result.text_original
    for number, (expected_original, reviewed_text) in REVIEW_REWRITES.items():
        if result.at[number - 1, "text_original"] != expected_original:
            raise ValueError(f"Câu nguồn {number} đã thay đổi; cần rà lại bản thu gọn trước khi xuất")
        result.loc[number - 1, "text_reviewed"] = reviewed_text
    result["text"] = result.text_reviewed.map(normalize_bank_names)
    domain_keys = set(map(_match_key, domain_texts))
    for column in ("text_original", "text_reviewed", "text"):
        keys = result[column].map(_match_key)
        if keys.duplicated().any():
            raise ValueError(f"FAQ trùng chính xác theo {column}")
        if set(keys) & domain_keys:
            raise ValueError(f"FAQ trùng chính xác Banking77-VN theo {column}")
    result.insert(0, "source_case_id", [f"faq_real:{number:03d}" for number in range(1, 101)])
    url_ids = {url: f"url:{number:03d}" for number, url in enumerate(sorted(result.source_url.unique()), start=1)}
    result["source_group_id"] = result.source_url.map(url_ids)
    result["origin"] = "user_collected_bank_faq"
    result["ood_source"] = "bank_faq_real"
    return result


def split_by_source_url(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia đúng 50/50, cân bằng topic và giữ cùng URL trong một split.

    Args:
        frame: FAQ đã chuẩn bị, có `source_url` và `topic`.
        seed: Seed dùng chọn các nhóm URL.

    Returns:
        Hai DataFrame validation và test.

    Raises:
        ValueError: Khi không thể tách 50/50 với 12–13 câu mỗi topic.
    """
    topics = sorted(TOPICS)
    urls = sorted(frame.source_url.unique())
    counts = np.stack([
        frame.loc[frame.source_url.eq(url), "topic"].value_counts().reindex(topics, fill_value=0).to_numpy()
        for url in urls
    ])
    rng = np.random.default_rng(seed)
    assignments = rng.integers(0, 2, size=(100_000, len(urls)), dtype=np.int8)
    validation_counts = assignments.astype(np.int16) @ counts.astype(np.int16)
    valid = (validation_counts.sum(axis=1) == 50) & ((validation_counts >= 12) & (validation_counts <= 13)).all(axis=1)
    if not valid.any():
        raise ValueError("Không tìm thấy split 50/50 cân bằng topic theo URL; cần kiểm tra phân bố nguồn")
    selected_urls = {url for url, selected in zip(urls, assignments[np.flatnonzero(valid)[0]], strict=True) if selected}
    validation = frame.loc[frame.source_url.isin(selected_urls)].copy().reset_index(drop=True)
    test = frame.loc[~frame.source_url.isin(selected_urls)].copy().reset_index(drop=True)
    if set(validation.source_url) & set(test.source_url):
        raise ValueError("URL nguồn bị tách giữa validation và test")
    for column in ("text_original", "text"):
        groups = _duplicate_groups(frame[column].tolist())
        memberships: dict[int, set[bool]] = {}
        for group, is_validation in zip(groups, frame.source_url.isin(selected_urls), strict=True):
            memberships.setdefault(group, set()).add(bool(is_validation))
        if any(len(names) > 1 for names in memberships.values()):
            raise ValueError(f"Câu gần trùng theo {column} bị tách validation/test")
    return validation, test


def main() -> int:
    """Ghi bản FAQ đã kiểm tra cùng hai split và thống kê nguồn."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raw = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    domain_texts = [
        text
        for split in ("train", "model_selection", "calibration", "threshold", "test")
        for text in pd.read_parquet(IN_DOMAIN / f"{split}.parquet", columns=["text"]).text.astype(str)
    ]
    frame = prepare_faq(raw, domain_texts)
    validation, test = split_by_source_url(frame)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "all.csv", index=False, encoding="utf-8-sig")
    validation.to_parquet(OUTPUT / "validation.parquet", index=False)
    test.to_parquet(OUTPUT / "test.parquet", index=False)
    print(
        f"FAQ nguồn {len(frame)} câu / {frame.source_url.nunique()} URL; "
        f"validation {len(validation)} / test {len(test)}; "
        f"ẩn tên ngân hàng ở {sum(frame.text.ne(frame.text_original))} câu"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
