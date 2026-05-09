import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# === 1) 你的 checkpoint 输出目录（你说的这个）===
CKPT_DIR = Path("/gpfs/work/aac/jiazhenhu24/PythonProjects/checkpoints/RobertaBiLSTMAttention_run1")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# === 2) 你的训练 .out 日志文件：改成你的真实文件名 ===
# 如果你不确定，就把 OUT_LOG 改成 logs 目录下那个 .out 的绝对路径
OUT_LOG = Path("/gpfs/work/aac/jiazhenhu24/ondemand/data/sys/myjobs/projects/default/2/logs/RobertaBiLSTMAttention_1317438.out")

pat = re.compile(
    r"\[epoch\s+(?P<epoch>\d+)\]\s+train_loss=(?P<train_loss>[0-9.]+)\s+"
    r"val_loss=(?P<val_loss>[0-9.]+)\s+val_acc=(?P<val_acc>[0-9.]+)\s+best=(?P<best>[0-9.]+)"
)

rows = []
for line in OUT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
    m = pat.search(line)
    if m:
        d = m.groupdict()
        rows.append({
            "epoch": int(d["epoch"]),
            "train_loss": float(d["train_loss"]),
            "val_loss": float(d["val_loss"]),
            "val_acc": float(d["val_acc"]),
            "best": float(d["best"]),
        })

if not rows:
    raise RuntimeError(f"No epoch lines matched. Check OUT_LOG path: {OUT_LOG}")

df = pd.DataFrame(rows).sort_values("epoch")
df_csv = CKPT_DIR / "epoch_metrics_from_out.csv"
df.to_csv(df_csv, index=False)

# --- loss curve ---
plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="train_loss")
plt.plot(df["epoch"], df["val_loss"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
loss_png = CKPT_DIR / "loss_curve.png"
plt.savefig(loss_png, dpi=200)

# --- acc curve ---
plt.figure()
plt.plot(df["epoch"], df["val_acc"], label="val_acc")
best_epoch = int(df.loc[df["best"].idxmax(), "epoch"])
best_val = float(df["best"].max())
plt.scatter([best_epoch], [best_val], label=f"best={best_val:.4f} @ epoch {best_epoch}")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.tight_layout()
acc_png = CKPT_DIR / "val_acc_curve.png"
plt.savefig(acc_png, dpi=200)

print("Saved:", df_csv)
print("Saved:", loss_png)
print("Saved:", acc_png)
