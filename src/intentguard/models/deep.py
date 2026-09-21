"""Mô hình transformer fine-tune cho phân loại 77 intent."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from intentguard.config import dump_json
from intentguard.retrieval import build_index, mean_pool


class TextFrameDataset(Dataset):
    """Dataset PyTorch giữ raw text và label index."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], int(self.labels[index])


class TransformerClassifier(nn.Module):
    """Encoder transformer kèm linear head có số lớp lấy từ dữ liệu."""

    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Sinh logits cho batch tokenized.

        Args:
            input_ids: Tensor token id.
            attention_mask: Tensor mask.

        Returns:
            Tensor logits shape ``(batch, num_labels)``.
        """
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        pooled = mean_pool(output.last_hidden_state, attention_mask)
        return self.classifier(pooled)


def _collate(tokenizer: Any, max_length: int):
    def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        """Tokenize một batch raw text và đính kèm label tensor."""
        texts, labels = zip(*batch, strict=True)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate


def predict_probabilities(
    model: TransformerClassifier,
    tokenizer: Any,
    frame: pd.DataFrame,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Tính xác suất softmax cho DataFrame mà không cập nhật trọng số."""
    loader = DataLoader(
        TextFrameDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate(tokenizer, max_length),
    )
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            del labels
            batch = {key: value.to(device) for key, value in batch.items()}
            batches.append(torch.softmax(model(**batch), dim=-1).cpu().numpy())
    return np.concatenate(batches, axis=0)


def train_deep(
    train: pd.DataFrame,
    selection: pd.DataFrame,
    label_names: list[str],
    model_config: dict[str, Any],
    seed: int,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Fine-tune transformer, dừng sớm theo macro-F1 và tạo retrieval index.

    Args:
        train: Split huấn luyện.
        selection: Split chọn checkpoint.
        label_names: Tên nhãn theo class index.
        model_config: Nhánh ``models.deep`` trong YAML.
        seed: Seed tái lập DataLoader/torch.
        artifact_dir: Thư mục artifact.

    Returns:
        Metadata huấn luyện và đường dẫn artifact.
    """
    torch.manual_seed(seed)
    model_name = model_config["name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_from = model_config.get("resume_from")
    if resume_from:
        checkpoint = Path(resume_from)
        previous = __import__("json").loads((checkpoint / "deep.json").read_text(encoding="utf-8"))
        if previous["label_names"] != label_names or previous["model_name"] != model_name:
            raise ValueError("Checkpoint tiếp tục không khớp backbone hoặc thứ tự nhãn")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
        model = TransformerClassifier(str(checkpoint / "encoder"), len(label_names)).to(device)
        state = torch.load(checkpoint / "deep_classifier.pt", map_location=device, weights_only=True)
        model.classifier.load_state_dict(state["classifier"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TransformerClassifier(model_name, len(label_names)).to(device)
    batch_size = int(model_config["batch_size"])
    grad_accum = int(model_config["gradient_accumulation_steps"])
    max_length = int(model_config["max_length"])
    train_loader = DataLoader(
        TextFrameDataset(train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=int(model_config.get("num_workers", 0)),
        collate_fn=_collate(tokenizer, max_length),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(model_config.get("use_mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(int(model_config["max_epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(**batch), labels) / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        probabilities = predict_probabilities(model, tokenizer, selection, max_length, batch_size, device)
        predictions = probabilities.argmax(axis=1)
        score = float(f1_score(selection["label"], predictions, average="macro"))
        history.append({"epoch": float(epoch + 1), "selection_macro_f1": score})
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(model_config["patience"]):
            break

    if best_state is None:
        raise RuntimeError("Không tạo được checkpoint DL")
    model.load_state_dict(best_state)
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    encoder_dir = destination / "encoder"
    tokenizer_dir = destination / "tokenizer"
    model.encoder.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    torch.save({"classifier": model.classifier.state_dict()}, destination / "deep_classifier.pt")
    metadata = {
        "backend": "deep",
        "model_name": model_name,
        "label_names": label_names,
        "num_labels": len(label_names),
        "device_at_training": str(device),
        "selection_macro_f1": best_score,
        "history": history,
        "resumed_from": str(resume_from) if resume_from else None,
        "config": model_config,
    }
    dump_json(metadata, destination / "deep.json")
    build_index(
        train["text"].tolist(),
        train["label"].astype(int).tolist(),
        label_names,
        model.encoder,
        tokenizer,
        max_length,
        batch_size,
        device,
        destination,
    )
    return metadata


def load_deep_model(
    artifact_dir: str | Path,
) -> tuple[TransformerClassifier, Any, torch.device, dict[str, Any]]:
    """Nạp encoder, classifier head và tokenizer từ artifact.

    Args:
        artifact_dir: Thư mục có ``deep.json`` và trọng số.

    Returns:
        Tuple model, tokenizer, device và metadata.
    """
    destination = Path(artifact_dir)
    metadata = __import__("json").loads((destination / "deep.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(str(destination / "encoder"), int(metadata["num_labels"]))
    state = torch.load(destination / "deep_classifier.pt", map_location=device, weights_only=True)
    model.classifier.load_state_dict(state["classifier"])
    tokenizer = AutoTokenizer.from_pretrained(destination / "tokenizer")
    model.to(device).eval()
    return model, tokenizer, device, metadata
