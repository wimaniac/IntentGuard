"""Thử encoder MiniLM đa ngôn ngữ với classifier tuyến tính trên split chọn mô hình."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentguard.config import dump_json, load_config, resolve_project_paths, set_global_seed
from intentguard.data import load_processed_bundle
from intentguard.retrieval import mean_pool, normalize_rows

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def encode_texts(
    texts: list[str], model: torch.nn.Module, tokenizer: object, max_length: int, batch_size: int, device: torch.device
) -> np.ndarray:
    """Tạo embedding mean-pooling cho các câu mà không cập nhật encoder.

    Args:
        texts: Danh sách câu tiếng Việt.
        model: Encoder Transformer đã nạp.
        tokenizer: Tokenizer tương ứng encoder.
        max_length: Số token tối đa mỗi câu.
        batch_size: Số câu mỗi batch.
        device: CPU hoặc CUDA.

    Returns:
        Ma trận embedding float32 theo đúng thứ tự đầu vào.
    """
    vectors: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            vectors.append(pooled.float().cpu().numpy())
            if start and (start // batch_size) % 40 == 0:
                print(f"Đã mã hóa {min(start + batch_size, len(texts))}/{len(texts)} câu", flush=True)
    return np.concatenate(vectors, axis=0)


def main() -> int:
    """Huấn luyện candidate trên train, chọn C bằng model-selection và lưu artifact riêng."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/experiments/deep_multilingual_minilm")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 8:
        raise ValueError("batch-size và max-length phải dương")
    config = load_config(ROOT / "configs/default.yaml")
    paths = resolve_project_paths(config, ROOT)
    set_global_seed(int(config["project"]["seed"]))
    bundle = load_processed_bundle(paths["processed"] / "in_domain")
    destination = ROOT / args.output
    destination.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    train_vectors = encode_texts(
        bundle.train.text.astype(str).tolist(), encoder, tokenizer, args.max_length, args.batch_size, device
    )
    selection_vectors = encode_texts(
        bundle.model_selection.text.astype(str).tolist(), encoder, tokenizer, args.max_length, args.batch_size, device
    )
    train_labels = bundle.train.label.astype(int).to_numpy()
    selection_labels = bundle.model_selection.label.astype(int).to_numpy()
    history: list[dict[str, float]] = []
    best: LogisticRegression | None = None
    best_score = -1.0
    best_c = 0.0
    for c in (0.3, 3.0, 30.0):
        classifier = LogisticRegression(C=c, max_iter=500, solver="lbfgs")
        classifier.fit(train_vectors, train_labels)
        score = float(f1_score(selection_labels, classifier.predict(selection_vectors), average="macro"))
        history.append({"c": c, "selection_macro_f1": score})
        print(f"C={c}: selection_macro_f1={score:.4f}", flush=True)
        if score > best_score:
            best = classifier
            best_score = score
            best_c = c
    if best is None or not np.array_equal(best.classes_, np.arange(len(bundle.label_names))):
        raise ValueError("Classifier không bao phủ đủ 77 nhãn")
    encoder.save_pretrained(destination / "encoder")
    tokenizer.save_pretrained(destination / "tokenizer")
    torch.save(
        {
            "classifier": {
                "weight": torch.tensor(best.coef_, dtype=torch.float32),
                "bias": torch.tensor(best.intercept_, dtype=torch.float32),
            }
        },
        destination / "deep_classifier.pt",
    )
    np.save(destination / "train_embeddings.npy", normalize_rows(train_vectors))
    pd.DataFrame(
        {
            "text": bundle.train.text.astype(str).tolist(),
            "label": train_labels,
            "intent": [bundle.label_names[label] for label in train_labels],
        }
    ).to_parquet(destination / "train_cases.parquet", index=False)
    metadata = {
        "backend": "deep",
        "model_name": MODEL_NAME,
        "label_names": bundle.label_names,
        "num_labels": len(bundle.label_names),
        "training_mode": "frozen_encoder_logistic_head",
        "selection_macro_f1": best_score,
        "history": history,
        "config": {"name": MODEL_NAME, "max_length": args.max_length, "batch_size": args.batch_size, "c": best_c},
    }
    dump_json(metadata, destination / "deep.json")
    print(f"Candidate ready: selection_macro_f1={best_score:.4f}; output={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
