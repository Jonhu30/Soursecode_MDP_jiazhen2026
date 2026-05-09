#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    DataCollatorWithPadding,
)
from datasets import load_dataset, DatasetDict


# -----------------------------
#  Model
# -----------------------------
class RobertaBiLSTMAttention(nn.Module):
    def __init__(
        self,
        base_model: str = "roberta-base",
        hidden_dim: int = 128,
        num_labels: int = 2,
        dropout: float = 0.3,
        freeze_base: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.base_model_name = base_model
        self.roberta = RobertaModel.from_pretrained(base_model, cache_dir=cache_dir)

        self.freeze_base = freeze_base
        if freeze_base:
            for p in self.roberta.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.roberta.config.hidden_size,  # 768 for roberta-base
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)  # scalar attention per token
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        # attention_mask: [B, T], 1 for real tokens, 0 for padding
        if self.freeze_base:
            with torch.no_grad():
                roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        lstm_out, _ = self.lstm(roberta_out)  # [B, T, 2H]

        # token scores: [B, T]
        scores = self.attn(lstm_out).squeeze(-1)
        # mask padding before softmax
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]

        context = torch.sum(weights * lstm_out, dim=1)  # [B, 2H]
        logits = self.fc(self.dropout(context))         # [B, C]
        return logits


# -----------------------------
#  Config / Save & Load
# -----------------------------
@dataclass
class ModelConfig:
    base_model: str = "roberta-base"
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

    # 1) model weights
    torch.save(model.state_dict(), od / "pytorch_model.bin")

    # 2) tokenizer
    tokenizer.save_pretrained(od)

    # 3) config
    with (od / "trainer_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def load_all(model_dir: str, device: torch.device, cache_dir: Optional[str] = None):
    md = Path(model_dir)
    with (md / "trainer_config.json").open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = ModelConfig(**cfg_dict)

    tokenizer = RobertaTokenizerFast.from_pretrained(md)

    model = RobertaBiLSTMAttention(
        base_model=cfg.base_model,
        hidden_dim=cfg.hidden_dim,
        num_labels=cfg.num_labels,
        dropout=cfg.dropout,
        freeze_base=cfg.freeze_base,
        cache_dir=cache_dir,
    ).to(device)

    state = torch.load(md / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return cfg, tokenizer, model


# -----------------------------
#  Data
# -----------------------------
def build_dataset_from_args(args) -> DatasetDict:
    """
    支持两类数据来源：
    1) HuggingFace datasets：--dataset imdb （或其他名字）
    2) 本地文件：--train_file train.csv --valid_file valid.csv
       文件格式支持 csv/json，要求包含 text_field & label_field
    """
    if args.train_file:
        data_files = {"train": args.train_file}
        if args.valid_file:
            data_files["validation"] = args.valid_file
        ds = load_dataset(
            args.file_format,
            data_files=data_files,
            cache_dir=args.hf_cache_dir,
        )
        return ds

    # HF datasets
    ds = load_dataset(args.dataset, cache_dir=args.hf_cache_dir)
    # 尽量统一 split 名
    if "validation" not in ds and "test" in ds:
        ds = DatasetDict({"train": ds["train"], "validation": ds["test"]})
    return ds


def tokenize_dataset(ds: DatasetDict, tokenizer, text_field: str, label_field: Optional[str], max_length: int):
    def _tok(batch):
        out = tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_length,
        )
        if label_field is not None and label_field in batch:
            out["labels"] = batch[label_field]
        return out

    remove_cols = []
    for split in ds.keys():
        remove_cols = list(set(remove_cols) | set(ds[split].column_names))

    # batched=True 会更快
    ds2 = ds.map(_tok, batched=True, remove_columns=remove_cols)
    return ds2


# -----------------------------
#  Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, dataloader, device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    losses = []
    ce = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = ce(logits, labels)
        losses.append(loss.item())

        pred = torch.argmax(logits, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    acc = correct / max(total, 1)
    return float(np.mean(losses)) if losses else 0.0, acc


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tokenizer = RobertaTokenizerFast.from_pretrained(args.base_model, cache_dir=args.hf_cache_dir)

    ds = build_dataset_from_args(args)
    ds = tokenize_dataset(ds, tokenizer, args.text_field, args.label_field, args.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(ds["validation"], batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)

    cfg = ModelConfig(
        base_model=args.base_model,
        hidden_dim=args.hidden_dim,
        num_labels=args.num_labels,
        dropout=args.dropout,
        freeze_base=args.freeze_base,
        max_length=args.max_length,
    )

    model = RobertaBiLSTMAttention(
        base_model=args.base_model,
        hidden_dim=args.hidden_dim,
        num_labels=args.num_labels,
        dropout=args.dropout,
        freeze_base=args.freeze_base,
        cache_dir=args.hf_cache_dir,
    ).to(device)

    # 如果要“继续训练/加载已有权重”
    if args.init_model_dir:
        state = torch.load(Path(args.init_model_dir) / "pytorch_model.bin", map_location=device)
        model.load_state_dict(state, strict=True)

    # 只优化 requires_grad=True 的参数（freeze_base 时会自动排除 roberta）
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    best_val_acc = -1.0
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        running_loss = []
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                logits = model(input_ids, attention_mask)
                loss = ce(logits, labels) / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss.append(loss.item() * args.grad_accum)
            pbar.set_postfix(loss=float(np.mean(running_loss[-50:])))

        val_loss, val_acc = evaluate(model, val_loader, device)

        # 每个 epoch 都存一个 last
        cfg_path = Path(out_dir) / f"epoch_{epoch}"
        save_all(str(cfg_path), model, tokenizer, cfg)

        # 保存最优
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_all(str(Path(out_dir) / "best"), model, tokenizer, cfg)

        print(f"[epoch {epoch}] train_loss={np.mean(running_loss):.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

    print(f"Training finished. Best checkpoint: {Path(out_dir) / 'best'}")


# -----------------------------
#  Predict
# -----------------------------
@torch.no_grad()
def predict_texts(cfg: ModelConfig, tokenizer, model, device, texts: List[str], batch_size: int):
    model.eval()
    results = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=cfg.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        for t, p, pr in zip(chunk, preds, probs):
            results.append({
                "text": t,
                "pred": int(p),
                "prob": pr.tolist(),
            })
    return results


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg, tokenizer, model = load_all(args.model_dir, device=device, cache_dir=args.hf_cache_dir)

    texts = []
    if args.text:
        texts.extend(args.text)
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    if not texts:
        raise ValueError("No input text. Use --text or --input_file.")

    outputs = predict_texts(cfg, tokenizer, model, device, texts, batch_size=args.batch_size)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for obj in outputs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        for obj in outputs:
            print(json.dumps(obj, ensure_ascii=False))


# -----------------------------
#  Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("RoBERTa+BiLSTM+Attention (train & predict)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Download dataset and train, then save checkpoints.")
    p_train.add_argument("--dataset", type=str, default="imdb",
                         help="HF datasets name, e.g., imdb / sst2 / tweet_eval ...")
    p_train.add_argument("--train_file", type=str, default=None, help="Local train file (csv/json).")
    p_train.add_argument("--valid_file", type=str, default=None, help="Local valid file (csv/json).")
    p_train.add_argument("--file_format", type=str, default="csv", choices=["csv", "json"], help="Format for local files.")

    p_train.add_argument("--text_field", type=str, default="text", help="Text column name.")
    p_train.add_argument("--label_field", type=str, default="label", help="Label column name (int).")

    p_train.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints.")
    p_train.add_argument("--init_model_dir", type=str, default=None,
                         help="Optional: load an existing checkpoint dir to continue training.")

    p_train.add_argument("--base_model", type=str, default="roberta-base")
    p_train.add_argument("--hidden_dim", type=int, default=128)
    p_train.add_argument("--num_labels", type=int, default=2)
    p_train.add_argument("--dropout", type=float, default=0.3)
    p_train.add_argument("--freeze_base", action="store_true", help="Freeze RoBERTa (default in this repo's idea).")
    p_train.add_argument("--no_freeze_base", dest="freeze_base", action="store_false")
    p_train.set_defaults(freeze_base=True)

    p_train.add_argument("--max_length", type=int, default=256)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--eval_batch_size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--weight_decay", type=float, default=0.01)
    p_train.add_argument("--grad_accum", type=int, default=1)
    p_train.add_argument("--fp16", action="store_true")

    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--cpu", action="store_true")
    p_train.add_argument("--hf_cache_dir", type=str, default=None,
                         help="HF cache dir (recommended on clusters, e.g., /gpfs/.../hf_cache).")

    # ---- predict ----
    p_pred = sub.add_parser("predict", help="Load saved model and run inference.")
    p_pred.add_argument("--model_dir", type=str, required=True, help="Checkpoint dir produced by this script.")
    p_pred.add_argument("--text", action="append", default=None, help="Input text (can repeat).")
    p_pred.add_argument("--input_file", type=str, default=None, help="Text file, one line per sample.")
    p_pred.add_argument("--output_file", type=str, default=None, help="Write JSONL results.")
    p_pred.add_argument("--batch_size", type=int, default=32)
    p_pred.add_argument("--cpu", action="store_true")
    p_pred.add_argument("--hf_cache_dir", type=str, default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)


if __name__ == "__main__":
    main()
