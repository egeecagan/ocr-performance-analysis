"""
Generates the engine-comparison chart for Figure fig:comparison in
Internship_Report.tex, from outputs/comparison_report.json.

Usage:
    python3 chart.py

Output:
    chart.png (project root)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPORT_PATH = Path("outputs/comparison_report.json")
OUTPUT_PATH = Path("chart.png")

# Fixed engine -> (display name, key prefix, color) mapping. The color
# assignment stays the same across all three panels below so that "blue"
# always means Tesseract, etc. — identity is encoded by color, never by
# position or rank.
ENGINES = [
    ("Tesseract", "tesseract", "#2a78d6"),  # blue
    ("EasyOCR",   "easyocr",   "#1baf7a"),  # aqua
    ("docTR",     "doctr",     "#eda100"),  # yellow
    ("RapidOCR",  "rapidocr",  "#008300"),  # green
]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def load_metrics():
    data = json.loads(REPORT_PATH.read_text())
    doc_type = next(iter(data))  # "surucubelgesi"
    results = data[doc_type]

    metrics = {"confidence": [], "cer": [], "time": []}
    labels = []
    for display_name, key_prefix, color in ENGINES:
        # match "tesseract/model_v1", "rapidocr/model_v1_lowspec", etc.
        match_key = next(
            (k for k in results if k.split("/")[0] == key_prefix), None
        )
        if match_key is None:
            continue
        entry = results[match_key]
        labels.append((display_name, color))
        metrics["confidence"].append(entry["avg_confidence"])
        metrics["cer"].append(entry["avg_cer"])
        metrics["time"].append(entry["avg_total_time_seconds"])

    return doc_type, labels, metrics


def style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.set_xticks([])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def add_value_labels(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )


def main():
    doc_type, labels, metrics = load_metrics()
    names = [name for name, _ in labels]
    colors = [color for _, color in labels]
    x = range(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), facecolor="#fcfcfb")
    fig.suptitle(
        f"OCR Engine Comparison — {doc_type} (native run, {len(x)} engines, "
        f"10 sample images)",
        fontsize=13,
        color=INK,
        y=0.99,
    )

    bars = axes[0].bar(x, metrics["confidence"], color=colors, width=0.6, zorder=3)
    style_axis(axes[0], "Avg. Confidence", "%")
    axes[0].set_ylim(0, 105)
    add_value_labels(axes[0], bars, "{:.1f}")

    bars = axes[1].bar(x, metrics["cer"], color=colors, width=0.6, zorder=3)
    style_axis(axes[1], "Avg. CER", "lower is better")
    add_value_labels(axes[1], bars, "{:.3f}")

    bars = axes[2].bar(x, metrics["time"], color=colors, width=0.6, zorder=3)
    style_axis(axes[2], "Avg. Time / Image", "seconds")
    add_value_labels(axes[2], bars, "{:.2f}")

    # One shared legend for all three panels — color always means the same
    # engine across panels, so the mapping only needs to be stated once.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color) for _, color in labels
    ]
    fig.legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(labels),
        frameon=False,
        fontsize=10,
        labelcolor=INK,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(OUTPUT_PATH, dpi=200, facecolor=fig.get_facecolor())
    print(f"Saved {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
