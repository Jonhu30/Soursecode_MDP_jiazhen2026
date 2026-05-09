import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

LOG = Path("/gpfs/work/aac/jiazhenhu24/PythonProjects/checkpoints/RobertaBiLSTMAttention_sarcasm_run1/metrics_log.jsonl")
OUT_DIR = LOG.parent

rows = []
with LOG.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

if not rows:
    raise RuntimeError(f"No rows found in {LOG}")

df = pd.DataFrame(rows).sort_values("epoch")

# loss
plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="train_loss")
plt.plot(df["epoch"], df["loss"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
loss_png = OUT_DIR / "loss_curve.png"
plt.savefig(loss_png, dpi=200)

# metrics
plt.figure()
plt.plot(df["epoch"], df["acc"], label="val_acc")
plt.plot(df["epoch"], df["f1"], label="val_f1")
plt.plot(df["epoch"], df["precision"], label="val_precision")
plt.plot(df["epoch"], df["recall"], label="val_recall")
plt.xlabel("epoch")
plt.ylabel("score")
plt.legend()
plt.tight_layout()
metrics_png = OUT_DIR / "metrics_curve.png"
plt.savefig(metrics_png, dpi=200)

print("Saved:", loss_png)
print("Saved:", metrics_png)
