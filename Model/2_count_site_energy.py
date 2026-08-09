"""Descriptive statistics of the cleaned modeling dataset.

Console output (unchanged from the original version of this script): category
counts for Site and Energy. The script additionally writes the descriptive
artifacts the dissertation cites.

Input: Model/1_remove_missing_rows.csv

Outputs (under Model/2_count_site_energy/):
  - site_energy_counts.csv  (category counts for Site and Energy)
  - gpr_summary_stats.csv   (count/mean/std/min/quartiles/max per GPR target)
  - class_balance.csv       (rows per clinical class 2/1/0 under THRESHOLDS)
  - gpr_histograms.pdf      (GPR3mm/2mm/1mm distributions with the clinical
                             thresholds marked)

Requires pandas + matplotlib (the original version was stdlib-only).
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend and save plots directly.
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "1_remove_missing_rows.csv"
OUTPUT_DIR = SCRIPT_DIR / "2_count_site_energy"

GPR_TARGETS = ("GPR3mm", "GPR2mm", "GPR1mm")
# Clinical decision rule; the values mirror THRESHOLDS in 8_operating_point.py
# (class 2 = auto-pass if GPR3mm >= 99 and GPR2mm >= 97; class 0 = replan if
# GPR3mm < 98 and GPR2mm < 92; class 1 = manual review otherwise).
THRESHOLDS = {
    "class2": {"GPR3mm": 99.0, "GPR2mm": 97.0},
    "class0": {"GPR3mm": 98.0, "GPR2mm": 92.0},
}

INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_BLUE = "#2a78d6"   # same series accent as steps 7/8
CLASS2_COLOR = "#0ca30c"  # status colors, as in 8_operating_point.py
CLASS0_COLOR = "#d03b3b"


def load_data(file_path: Path) -> list[dict[str, str]]:
    if not file_path.is_file():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"Site", "Energy"}.union(GPR_TARGETS)
        missing_columns = required_columns.difference(reader.fieldnames or [])

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns: {missing}")

        return list(reader)


# Load all rows from the cleaned data file once for reuse.
DATA = load_data(DATA_FILE)


def print_category_counts(column_name: str) -> Counter:
    counts = Counter(row[column_name] for row in DATA)

    print(f"{column_name} has {len(counts)} categories")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count} rows")
    return counts


def write_site_energy_counts(counts_by_column: dict[str, Counter]) -> None:
    output_path = OUTPUT_DIR / "site_energy_counts.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["column", "category", "count"])
        for column_name, counts in counts_by_column.items():
            for category, count in sorted(counts.items()):
                writer.writerow([column_name, category, count])


def load_gpr_values() -> pd.DataFrame:
    """GPR columns as floats; a malformed cell raises ValueError."""
    return pd.DataFrame(
        {target: [float(row[target]) for row in DATA] for target in GPR_TARGETS}
    )


def write_summary_stats(gpr: pd.DataFrame) -> None:
    stats = gpr.describe().T.round(3)
    stats.index.name = "target"
    stats.to_csv(OUTPUT_DIR / "gpr_summary_stats.csv", encoding="utf-8-sig")


def write_class_balance(gpr: pd.DataFrame) -> None:
    """Count rows per clinical class, replicating load_true_labels() in step 8."""
    is2 = (gpr["GPR3mm"] >= THRESHOLDS["class2"]["GPR3mm"]) & (
        gpr["GPR2mm"] >= THRESHOLDS["class2"]["GPR2mm"]
    )
    is0 = (gpr["GPR3mm"] < THRESHOLDS["class0"]["GPR3mm"]) & (
        gpr["GPR2mm"] < THRESHOLDS["class0"]["GPR2mm"]
    )
    counts = {2: int(is2.sum()), 0: int(is0.sum())}
    counts[1] = len(gpr) - counts[2] - counts[0]

    output_path = OUTPUT_DIR / "class_balance.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", "count", "share"])
        for cls in (2, 1, 0):
            writer.writerow([cls, counts[cls], round(counts[cls] / len(gpr), 3)])
    print(
        "\nClass balance (clinical rule): "
        + ", ".join(f"{cls}->{counts[cls]}" for cls in (2, 1, 0))
    )


def style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: hairline y-grid, light baseline, muted tick labels."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor="#52514e")


def plot_histograms(gpr: pd.DataFrame) -> None:
    """One panel per GPR target; the clinical thresholds marked where defined."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, target in zip(axes, GPR_TARGETS):
        ax.hist(gpr[target], bins=30, color=SERIES_BLUE, alpha=0.85)
        if target in THRESHOLDS["class2"]:
            class2 = THRESHOLDS["class2"][target]
            class0 = THRESHOLDS["class0"][target]
            ax.axvline(class2, color=CLASS2_COLOR, linewidth=1.5, linestyle="--")
            ax.axvline(class0, color=CLASS0_COLOR, linewidth=1.5, linestyle=":")
            ax.annotate(
                f"class-2 threshold {class2:g}",
                (0.02, 0.95), xycoords="axes fraction",
                color="#52514e", fontsize=8,
            )
            ax.annotate(
                f"class-0 threshold {class0:g}",
                (0.02, 0.89), xycoords="axes fraction",
                color="#52514e", fontsize=8,
            )
        style_axes(ax)
        ax.set_title(target, color=INK)
        ax.set_xlabel(f"Measured {target} (%)", color="#52514e")
        ax.set_ylabel("rows", color="#52514e")
    fig.suptitle(
        f"Measured gamma pass rate distributions (n={len(gpr)} rows)", color=INK
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "gpr_histograms.pdf")
    plt.close(fig)


def main() -> None:
    site_counts = print_category_counts("Site")
    print()
    energy_counts = print_category_counts("Energy")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_site_energy_counts({"Site": site_counts, "Energy": energy_counts})

    gpr = load_gpr_values()
    write_summary_stats(gpr)
    write_class_balance(gpr)
    plot_histograms(gpr)

    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
