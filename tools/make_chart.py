"""Render assets/rtf_comparison.png from results/npu_benchmark_results.txt."""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
text = (root / "results" / "npu_benchmark_results.txt").read_text(encoding="utf-8")
npu = float(re.search(r"NPU RTF\s*:\s*([\d.]+)", text).group(1))
cpu = float(re.search(r"CPU RTF\s*:\s*([\d.]+)", text).group(1))
speedup = float(re.search(r"Speedup\s*:\s*([\d.]+)x", text).group(1))

BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
fig, ax = plt.subplots(figsize=(8, 3.4), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
labels = ["Mobilint NPU (MXQ)", "CPU (ONNX Runtime)"]
vals = [npu, cpu]
bars = ax.barh(labels, vals, color=[BLUE, ORANGE], height=0.5)
ax.invert_yaxis()
for b, v in zip(bars, vals):
    ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"RTF {v:.4f}", va="center", color=INK, fontsize=10)
ax.set_xlim(0, cpu * 1.3)
ax.set_xlabel("Real-time factor (lower is faster; <1.0 = faster than real time)", color=MUTED, fontsize=9)
ax.set_title(f"HiFi-GAN generator: {speedup:.2f}x faster on the NPU (single-frame benchmark)",
             loc="left", color=INK, fontsize=11, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c9c8c2")
ax.tick_params(colors=MUTED, labelsize=9)
fig.tight_layout()
out = root / "assets" / "rtf_comparison.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
