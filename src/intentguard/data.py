"""Tải, kiểm tra, khử trùng lặp và chia dữ liệu cho IntentGuard."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from intentguard.config import dump_json
from intentguard.review import duplicate_groups, validate_approved_rows
from intentguard.schemas import DatasetBundle


def normalize_text(value: Any) -> str:
    """Chuẩn hoá khoảng trắng để nhận diện bản trùng mà không sửa nội dung gốc.

    Args:
        value: Giá trị cần chuẩn hoá.

    Returns:
        Chuỗi Unicode đã bỏ khoảng trắng thừa và trim.
    """
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def _as_frame(dataset: Dataset, text_column: str, label_column: str, label_text_column: str) -> pd.DataFrame:
    frame = dataset.to_pandas()
    required = {text_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset thiếu cột bắt buộc: {sorted(missing)}")
    rename = {text_column: "text", label_column: "label"}
    if label_text_column in frame.columns:
        rename[label_text_column] = "label_text"
    frame = frame.rename(columns=rename)
    if "label_text" not in frame.columns:
        frame["label_text"] = frame["label"].astype(str)
    frame["text"] = frame["text"].map(normalize_text)
    if frame["text"].eq("").any():
        raise ValueError("Dataset chứa câu rỗng sau khi chuẩn hoá")
    return frame[["text", "label", "label_text"]].reset_index(drop=True)


def load_banking_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tải train/test của Banking77-VN từ Hugging Face.

    Args:
        config: Nhánh ``data.banking_dataset`` trong YAML.

    Returns:
        Tuple gồm train và test đã chuẩn hoá thành DataFrame.

    Raises:
        ValueError: Khi schema nguồn không khớp cấu hình.
    """
    kwargs = {"path": config["name"]}
    if config.get("config"):
        kwargs["name"] = config["config"]
    dataset = load_dataset(**kwargs)
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Dataset Banking77-VN phải có train/test split")
    text_column = config["text_column"]
    label_column = config["label_column"]
    label_text_column = config.get("label_text_column", "label_text")
    train = _as_frame(dataset[config["train_split"]], text_column, label_column, label_text_column)
    test = _as_frame(dataset[config["test_split"]], text_column, label_column, label_text_column)
    raw_labels = pd.concat([train["label"], test["label"]], ignore_index=True).drop_duplicates().tolist()
    raw_labels = sorted(raw_labels, key=lambda value: str(value))
    label_to_index = {value: index for index, value in enumerate(raw_labels)}
    train["label"] = train["label"].map(label_to_index).astype(int)
    test["label"] = test["label"].map(label_to_index).astype(int)
    label_text_by_index = (
        pd.concat([train, test], ignore_index=True)
        .sort_values("label")
        .drop_duplicates("label")
        .set_index("label")["label_text"]
        .astype(str)
        .to_dict()
    )
    train["label_text"] = train["label"].map(label_text_by_index)
    test["label_text"] = test["label"].map(label_text_by_index)
    return train, test


def _deduplicate_train_against_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Khử bản trùng, ưu tiên giữ nguyên test để đánh giá không bị thay đổi."""
    test_keys = set(test["text"].map(normalize_text))
    train = train.copy()
    train["_key"] = train["text"].map(normalize_text)
    duplicate_test = train["_key"].isin(test_keys)
    duplicate_train = train["_key"].duplicated(keep="first")
    removed_test_overlap = int(duplicate_test.sum())
    removed_within_train = int((~duplicate_test & duplicate_train).sum())
    train = train.loc[~duplicate_test & ~duplicate_train].drop(columns="_key")
    return train.reset_index(drop=True), {
        "train_removed_overlapping_test": removed_test_overlap,
        "train_removed_within_train": removed_within_train,
        "test_rows_kept": int(len(test)),
    }


def _stratified_partition(frame: pd.DataFrame, fractions: Iterable[float], seed: int) -> list[pd.DataFrame]:
    """Chia DataFrame theo tỷ lệ với stratify trên nhãn số."""
    fractions = list(fractions)
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Tổng tỷ lệ split phải bằng 1, nhận được {fractions}")
    raw_counts = np.asarray(fractions) * len(frame)
    counts = np.floor(raw_counts).astype(int)
    remainder = len(frame) - int(counts.sum())
    if remainder:
        fractional_order = np.argsort(-(raw_counts - counts))
        counts[fractional_order[:remainder]] += 1
    remaining = frame.copy()
    result: list[pd.DataFrame] = []
    seed_sequence = np.random.SeedSequence(seed).spawn(len(fractions) - 1)
    for target_count, child_seed in zip(counts[:-1], seed_sequence, strict=False):
        remaining_count = len(remaining)
        if target_count <= 0 or target_count >= remaining_count:
            raise ValueError("Mỗi split phải có ít nhất một mẫu và nhỏ hơn phần còn lại")
        first, remaining = train_test_split(
            remaining,
            test_size=remaining_count - int(target_count),
            random_state=int(child_seed.generate_state(1)[0]),
            stratify=remaining["label"],
        )
        result.append(first.reset_index(drop=True))
    result.append(remaining.reset_index(drop=True))
    return result


def prepare_in_domain(
    config: dict[str, Any], paths: dict[str, Path], seed: int, output_name: str = "in_domain"
) -> DatasetBundle:
    """Chuẩn bị split in-domain và ghi metadata tái lập.

    Args:
        config: Toàn bộ cấu hình project.
        paths: Đường dẫn đã chuẩn hoá bởi ``resolve_project_paths``.
        seed: Seed chia stratified.
        output_name: Tên thư mục artifact của dataset.

    Returns:
        DatasetBundle gồm bốn phần train từ train gốc và test gốc.
    """
    data_config = config["data"]
    train_raw, test = load_banking_dataset(data_config["banking_dataset"])
    train, dedup = _deduplicate_train_against_test(train_raw, test)
    split_config = data_config["split"]
    splits = _stratified_partition(
        train,
        [
            split_config["train_fraction"],
            split_config["model_selection_fraction"],
            split_config["calibration_fraction"],
            split_config["threshold_fraction"],
        ],
        seed,
    )
    train_split, model_selection, calibration, threshold = splits
    label_table = (
        train_raw[["label", "label_text"]].drop_duplicates("label").sort_values("label").reset_index(drop=True)
    )
    label_names = label_table["label_text"].astype(str).tolist()
    dataset_dir = paths["processed"] / output_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "train": train_split,
        "model_selection": model_selection,
        "calibration": calibration,
        "threshold": threshold,
        "test": test,
    }.items():
        frame.to_parquet(dataset_dir / f"{name}.parquet", index=False)
    metadata = {
        "seed": seed,
        "source": data_config["banking_dataset"],
        "counts": {
            name: int(len(frame))
            for name, frame in {
                "train": train_split,
                "model_selection": model_selection,
                "calibration": calibration,
                "threshold": threshold,
                "test": test,
            }.items()
        },
        "deduplication": dedup,
        "label_count": len(label_names),
        "label_names": label_names,
    }
    dump_json(metadata, dataset_dir / "metadata.json")
    return DatasetBundle(
        train_split,
        model_selection,
        calibration,
        threshold,
        test,
        label_names,
        metadata,
    )


def load_processed_bundle(dataset_dir: str | Path) -> DatasetBundle:
    """Nạp DatasetBundle từ artifact parquet đã chuẩn bị.

    Args:
        dataset_dir: Thư mục chứa parquet và metadata.

    Returns:
        DatasetBundle đã nạp.

    Raises:
        FileNotFoundError: Khi artifact chưa được tạo.
    """
    directory = Path(dataset_dir)
    with (directory / "metadata.json").open("r", encoding="utf-8") as file:
        import json

        metadata = json.load(file)
    frames = {
        name: pd.read_parquet(directory / f"{name}.parquet")
        for name in ["train", "model_selection", "calibration", "threshold", "test"]
    }
    return DatasetBundle(
        frames["train"],
        frames["model_selection"],
        frames["calibration"],
        frames["threshold"],
        frames["test"],
        metadata["label_names"],
        metadata,
    )


def _find_text_column(frame: pd.DataFrame, configured: str) -> str:
    if configured in frame.columns:
        return configured
    candidates = [name for name in frame.columns if frame[name].dtype == "object"]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Không tìm thấy cột văn bản {configured!r}; schema={list(frame.columns)}")


def build_massive_ood(config: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    """Tải và lọc MASSIVE vi-VN theo topics khai báo trong cấu hình.

    Args:
        config: Toàn bộ cấu hình project.
        paths: Đường dẫn project đã chuẩn hoá.

    Returns:
        DataFrame OOD có cột ``text``, ``ood_source`` và metadata nguồn.
    """
    ood_config = config["data"]["ood"]["massive"]
    scenario_names = _load_massive_scenario_names(ood_config)
    records: list[pd.DataFrame] = []
    for split in ood_config["splits"]:
        file_path = hf_hub_download(
            repo_id=ood_config["name"],
            filename=f"{ood_config['config']}/{split}/0000.parquet",
            revision=ood_config.get("revision"),
            repo_type="dataset",
        )
        frame = pd.read_parquet(file_path)
        text_column = _find_text_column(frame, ood_config["text_column"])
        topic_column = ood_config["topic_column"]
        if topic_column not in frame.columns:
            raise ValueError(f"MASSIVE thiếu topic column {topic_column!r}")
        topic_names = frame[topic_column].map(
            lambda value: scenario_names[int(value)] if str(value).isdigit() else str(value)
        )
        selected = frame[topic_names.isin(map(str, ood_config["topics"]))].copy()
        selected["text"] = selected[text_column].map(normalize_text)
        selected["ood_source"] = f"massive:{split}"
        selected["source_split"] = split
        selected["source_topic"] = topic_names.loc[selected.index].astype(str)
        records.append(selected[["text", "ood_source", "source_split", "source_topic"]])
    result = pd.concat(records, ignore_index=True).drop_duplicates("text")
    output = paths["interim"] / "ood_massive.parquet"
    result.to_parquet(output, index=False)
    return result


def _load_massive_scenario_names(config: dict[str, Any]) -> list[str]:
    """Đọc danh sách scenario từ script nguồn MASSIVE thay vì nhúng mapping."""
    script_path = hf_hub_download(
        repo_id=config["name"],
        filename=config["schema_file"],
        revision=config.get("schema_revision"),
        repo_type="dataset",
    )
    tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_SCENARIOS" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                return value
    raise ValueError("Không đọc được _SCENARIOS từ script MASSIVE nguồn")


def build_bank_ood_candidates(config: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    """Tải nguồn tham khảo UTS2017_Bank để review, không tự gán nhãn.

    Args:
        config: Toàn bộ cấu hình project.
        paths: Đường dẫn project đã chuẩn hoá.

    Returns:
        DataFrame candidate với text và metadata nguồn; không được dùng để train.
    """
    source = config["data"]["ood"]["candidate_source"]
    records: list[pd.DataFrame] = []
    for split in source["splits"]:
        if source.get("format") == "jsonl":
            file_path = hf_hub_download(
                repo_id=source["name"],
                filename=source["files"][split],
                revision=source.get("revision"),
                repo_type="dataset",
            )
            frame = pd.read_json(file_path, lines=True)
        else:
            kwargs = {"path": source["name"], "split": split}
            if source.get("config"):
                kwargs["name"] = source["config"]
            frame = load_dataset(**kwargs).to_pandas()
        text_column = _find_text_column(frame, source["text_column"])
        selected = pd.DataFrame({"text": frame[text_column].map(normalize_text)})
        selected["source_split"] = split
        selected["source_row"] = np.arange(len(selected), dtype=int)
        if source.get("topic_column") in frame.columns:
            selected["source_topic"] = frame[source["topic_column"]].astype(str).to_numpy()
        records.append(selected)
    result = pd.concat(records, ignore_index=True).drop_duplicates("text")
    output = paths["root"] / config["data"]["ood"]["bank_candidate_file"]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def load_reviewed_bank_ood(config: dict[str, Any], paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nạp 100 câu OOD ngân hàng đã review và tách validation/test theo case nguồn.

    Args:
        config: Toàn bộ cấu hình project.
        paths: Đường dẫn project đã chuẩn hoá.

    Returns:
        Tuple ``(validation, test)`` gồm các câu được đánh dấu OOD.

    Raises:
        FileNotFoundError: Khi chưa cung cấp file review và danh sách case.
        ValueError: Khi số lượng hoặc phân vùng không đúng cấu hình.
    """
    ood_config = config["data"]["ood"]
    quality_path = paths["root"] / ood_config.get("bank_quality_file", "data/ood_bank_quality_status.json")
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if quality.get("status") != "verified":
            raise FileNotFoundError("Tập OOD ngân hàng đang chờ kiểm tra chất lượng nhãn; không dùng split cũ")
    review_path = paths["root"] / ood_config["bank_review_file"]
    if not review_path.exists():
        raise FileNotFoundError(
            f"Chưa có file review OOD ngân hàng: {review_path}. Hãy duyệt trên app hoặc chạy CLI DeepSeek."
        )
    case_path = paths["root"] / ood_config["bank_validation_cases"]
    test_case_path = paths["root"] / ood_config["bank_test_cases"]
    if not case_path.exists() or not test_case_path.exists():
        raise FileNotFoundError("Chưa xuất danh sách case OOD validation/test từ giao diện review")
    frame = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    case_col = ood_config["bank_source_case_column"]
    candidate_path = paths["root"] / ood_config["bank_candidate_file"]
    candidates = pd.read_parquet(candidate_path)
    domain_dir = paths["processed"] / "in_domain"
    domain_texts = [
        text
        for name in ("train", "model_selection", "calibration", "threshold", "test")
        for text in pd.read_parquet(domain_dir / f"{name}.parquet", columns=["text"])["text"].astype(str)
    ]
    review_config = ood_config["review"]
    approved, source_texts = validate_approved_rows(
        frame,
        candidates,
        domain_texts,
        int(ood_config["expected_total"]),
        int(review_config["min_length"]),
        int(review_config["max_length"]),
    )
    approved["text"] = approved["text"].map(normalize_text)
    validation_cases = _read_case_ids(case_path)
    test_cases = _read_case_ids(test_case_path)
    if validation_cases & test_cases or len(validation_cases) != int(ood_config["expected_per_split"]):
        raise ValueError("Danh sách source_case_id validation/test phải tách biệt và đủ số lượng")
    validation = approved[approved[case_col].astype(str).isin(validation_cases)].copy()
    test = approved[approved[case_col].astype(str).isin(test_cases)].copy()
    if len(validation) != int(ood_config["expected_per_split"]) or len(test) != int(ood_config["expected_per_split"]):
        raise ValueError("Validation/test OOD ngân hàng phải đạt số lượng cấu hình")
    approved_ids = approved[case_col].astype(str).tolist()
    groups = duplicate_groups(approved["text"].tolist(), [source_texts[case_id] for case_id in approved_ids])
    split_by_group: dict[int, set[str]] = {}
    for case_id, group in zip(approved_ids, groups, strict=True):
        split_by_group.setdefault(group, set()).add("validation" if case_id in validation_cases else "test")
    if any(len(splits) > 1 for splits in split_by_group.values()):
        raise ValueError("Câu gần trùng hoặc cùng trường hợp nguồn bị tách giữa validation và test")
    for split, split_frame in [("validation", validation), ("test", test)]:
        split_frame["ood_source"] = f"bank_review:{split}"
        split_frame.to_parquet(paths["processed"] / f"ood_bank_{split}.parquet", index=False)
    return validation, test


def _read_case_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Chưa có danh sách case nguồn: {path}")
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
