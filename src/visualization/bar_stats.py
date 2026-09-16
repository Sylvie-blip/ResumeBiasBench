"""Generate a 4x4 grid of AI vs Human preference bar charts.

Edit the `PROPORTIONS` and `LABELS` lists below with your 16 values and
16 model names. Values may be proportions (0.0-1.0) or percentages (0-100).

Run:
	python /Users/sylviadong/Documents/Resume_Bias/Bias_code/Graphing/bar_stats.py

The script writes a PNG to `Bias_code/Graphing/ai_vs_human_4x4.png` by default.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Sequence


# --- EDIT THESE TWO LISTS ---
# 16 numbers (proportions AI preferred). Examples shown below.
PROPORTIONS: list[float] = [
	0.64, 0.32, 0.50, 0.78,
	0.22, 0.60, 0.40, 0.90,
	0.10, 0.55, 0.45, 0.66,
	0.34, 0.50, 0.30, 0.70,
]

# 16 labels for the models (one per subplot)
LABELS: list[str] = [
	"qwen2.4:3b",
	"qwen2.5:14b",
	"qwen2.5:32b",
	"gemma3:4b",
	"gemma3:12b",
	"mistral-small3.2:latest",
	"mistral-small3.2:24b",
	"llama3.1:8b",
	"nemotron-cascade-2:30b",
	"granite4:3b",
	"granite4:latest",
	"cogito:3b",
	"cogito:8b",
	"cogito:32b",
	"devstral:24b",
	"devstral:latest"
]
# ----------------------------


OUTPUT_PATH = Path(__file__).parent / "ai_vs_human_4x4.png"


def normalize_proportions(values: Sequence[float]) -> list[float]:
	"""Normalize list to floats in [0,1]. Accepts percentages (>1) or 0-1.

	Raises ValueError if values out of range after normalization.
	"""
	out: list[float] = []
	for v in values:
		if v is None:
			raise ValueError("None value in proportions")
		try:
			fv = float(v)
		except Exception:
			raise ValueError(f"Invalid numeric value: {v}")
		if fv > 1.0:
			fv = fv / 100.0
		if not (0.0 <= fv <= 1.0):
			raise ValueError(f"Proportion out of range 0..1: {fv}")
		out.append(fv)
	return out


def draw_grid(proportions: Sequence[float], labels: Sequence[str], out_path: Path | str) -> None:
	p = normalize_proportions(proportions)
	if len(p) != 16 or len(labels) != 16:
		raise ValueError("Expect exactly 16 proportions and 16 labels")

	fig, axes = plt.subplots(4, 4, figsize=(12, 12))
	axes = axes.flatten()

	for i, ax in enumerate(axes):
		ai = p[i]
		human = 1.0 - ai
		# place bars closer together by using smaller x separation and narrower width
		positions = [0.0, 0.6]
		bars = ax.bar(positions, [ai, human], color=["#1f77b4", "#ff7f0e"], width=0.5)
		ax.set_ylim(0, 1)
		ax.set_xticks(positions)
		ax.set_xticklabels(["AI", "Human"])
		ax.set_title(labels[i], fontsize=10)
		# annotate
		for rect in bars:
			height = rect.get_height()
			ax.annotate(f"{height*100:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
						xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
		ax.tick_params(axis="x", labelsize=9)
		ax.tick_params(axis="y", labelsize=8)

	fig.suptitle("AI vs Human Preference", fontsize=14)
	# add a single color key (legend) for AI vs Human
	legend_handles = [Patch(color="#1f77b4"), Patch(color="#ff7f0e")]
	fig.legend(legend_handles, ["AI", "Human"], loc="upper right", fontsize=10)

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.savefig(out_path, dpi=200)
	plt.close(fig)


def main() -> None:
	draw_grid(PROPORTIONS, LABELS, OUTPUT_PATH)
	print(f"Saved 4x4 bar chart to: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
