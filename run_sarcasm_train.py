#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizerFast, RobertaModel, DataCollatorWithPadding

# ---- metrics ----
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# =========================================================
#  Paths (按你给的目录写死)
# =========================================================
ROOT = "/gpfs/work/aac/jiazhenhu24/PythonProjects/Advanced-Sentiment-Classifier-RoBERTa-BiLSTM-Attention"

TRAIN_CSV = f"{ROOT}/sarcasm_train.csv"
VAL_CSV   = f"{ROOT}/sarcasm_val.csv"
TEST_CSV  = f"{ROOT}/sarcasm_test.csv"

# 你在 slurm 里用的是本地模型目录 roberta-base
BASE_MODEL = "/gpfs/work/aac/jiazhenhu24/PythonProjects/hf_models/roberta-base"

OUTPUT_DIR = "/gpfs/work/aac/jiazhenhu24/PythonProjects/checkpoints/RobertaBiLSTMAttention_sarcasm_run1"


# =========================================================
#  Training Config
# =========================================================
SEED = 42
NUM_LABELS = 2

MAX_LENGTH = 256
EPOCHS = 20
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01

# 如果你要微调 roberta（推荐 sarcasm）
FREEZE_BASE = False
LR = 2e-5

# 如果你要冻结 roberta（更快但效果可能一般）
# FREEZE_BASE = True
# LR = 2e-4

FP16 = True
GRAD_ACCUM = 1
CPU = False

HF_CACHE_DIR = os.environ.get("HF_HOME", None)  # 与 slurm 中 HF_HOME 对齐（可选）


# -----------------------------
#  Model
# -----------------------------
class RobertaBiLSTMAttention(nn.Module):
    def __init__(
        self,
        base_model: str,
        hidden_dim: int = 128,
        num_labels: int = 2,
        dropout: float = 0.3,
        freeze_base: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(base_model, cache_dir=cache_dir)

        self.freeze_base = freeze_base
        if freeze_base:
            for p in self.roberta.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.roberta.config.hidden_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        if self.freeze_base:
            with torch.no_grad():
                roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        lstm_out, _ = self.lstm(roberta_out)
        scores = self.attn(lstm_out).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)

        context = torch.sum(weights * lstm_out, dim=1)
        logits = self.fc(self.dropout(context))
        return logits


# -----------------------------
#  Save / Config
# -----------------------------
@dataclass
class ModelConfig:
    base_model: str
    hidden_dim: int = 128
    num_labels: int = 2
    dropout: float = 0.3
    freeze_base: bool = True
    max_length: int = 256
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[int, str]] = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_all(output_dir: str, model: nn.Module, tokenizer, cfg: ModelConfig):
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), od / "pytorch_model.bin")
    tokenizer.save_pretrained(od)

    with (od / "trainer_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


# -----------------------------
#  Data
# -----------------------------
def read_csv_as_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    if "text" not in df.columns or "labels" not in df.columns:
        raise ValueError(f"{path} 缺少列 text 或 labels。当前列：{list(df.columns)}")

    df["text"] = df["text"].astype(str).fillna("")
    df = df[df["text"].str.len() > 0].copy()
    df["labels"] = df["labels"].astype(int)

    return Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)


def build_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": read_csv_as_dataset(TRAIN_CSV),
            "validation": read_csv_as_dataset(VAL_CSV),
            "test": read_csv_as_dataset(TEST_CSV),
        }
    )


def tokenize_dataset(ds: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
        out["labels"] = batch["labels"]
        return out

    return ds.map(_tok, batched=True, remove_columns=["text"])


# -----------------------------
#  Metrics / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, dataloader, device, num_labels: int = 2) -> Dict[str, object]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    losses = []
    preds_all = []
    labels_all = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = ce(logits, labels)
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        preds_all.append(preds.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(preds_all) if preds_all else np.array([])
    y_true = np.concatenate(labels_all) if labels_all else np.array([])

    if y_true.size == 0:
        return {"loss": 0.0, "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "cm": [[0]*num_labels]*num_labels}

    acc = float((y_pred == y_true).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels))).tolist()

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cm": cm,
    }


# -----------------------------
#  Train
# -----------------------------
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() and not CPU else "cpu")
    print("Device:", device)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer & dataset
    tokenizer = RobertaTokenizerFast.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE_DIR)
    ds = build_dataset()
    print("Loaded sizes:", {k: len(ds[k]) for k in ds.keys()})

    ds = tokenize_dataset(ds, tokenizer, MAX_LENGTH)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(ds["validation"], batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(ds["test"], batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)

    cfg = ModelConfig(
        base_model=BASE_MODEL,
        hidden_dim=128,
        num_labels=NUM_LABELS,
        dropout=0.3,
        freeze_base=FREEZE_BASE,
        max_length=MAX_LENGTH,
    )

    model = RobertaBiLSTMAttention(
        base_model=BASE_MODEL,
        hidden_dim=128,
        num_labels=NUM_LABELS,
        dropout=0.3,
        freeze_base=FREEZE_BASE,
        cache_dir=HF_CACHE_DIR,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=FP16 and device.type == "cuda")

    best_val_f1 = -1.0

    # 保存一份训练配置
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "TRAIN_CSV": TRAIN_CSV,
                "VAL_CSV": VAL_CSV,
                "TEST_CSV": TEST_CSV,
                "BASE_MODEL": BASE_MODEL,
                "OUTPUT_DIR": OUTPUT_DIR,
                **asdict(cfg),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "fp16": FP16,
                "grad_accum": GRAD_ACCUM,
                "seed": SEED,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

        running_loss = []
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=FP16 and device.type == "cuda"):
                logits = model(input_ids, attention_mask)
                loss = ce(logits, labels) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss.append(loss.item() * GRAD_ACCUM)
            pbar.set_postfix(loss=float(np.mean(running_loss[-50:])))

        val_metrics = evaluate(model, val_loader, device, num_labels=NUM_LABELS)

        # 每个 epoch 保存
        save_all(str(out_dir / f"epoch_{epoch}"), model, tokenizer, cfg)

        # 用 val_f1 选最佳（你也可以改成 acc）
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_all(str(out_dir / "best"), model, tokenizer, cfg)

        log_line = {
            "epoch": epoch,
            "train_loss": float(np.mean(running_loss)) if running_loss else 0.0,
            **val_metrics,
            "best_val_f1": float(best_val_f1),
        }

        print(
            f"[epoch {epoch}] train_loss={log_line['train_loss']:.4f} "
            f"val_loss={log_line['loss']:.4f} acc={log_line['acc']:.4f} "
            f"p={log_line['precision']:.4f} r={log_line['recall']:.4f} f1={log_line['f1']:.4f} "
            f"best_f1={log_line['best_val_f1']:.4f}"
        )
        print("val_confusion_matrix:", log_line["cm"])

        with (out_dir / "metrics_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

    # 最终 test
    test_metrics = evaluate(model, test_loader, device, num_labels=NUM_LABELS)
    print(
        f"[test] loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f} "
        f"p={test_metrics['precision']:.4f} r={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
    )
    print("test_confusion_matrix:", test_metrics["cm"])

    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print(f"Best checkpoint: {out_dir / 'best'}")


if __name__ == "__main__":
    main()
