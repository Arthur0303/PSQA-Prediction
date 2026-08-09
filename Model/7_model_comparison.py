"""Compare model performance across feature counts from the step-6 results.

Inputs (under Model/6_model_training/, produced by 6_model_training.py):
  - fold_results.csv  (per outer-fold r2 / mae / rmse for every configuration)
  - summary.csv       (mean/std per target/model/feature-count)

Outputs (under Model/7_model_comparison/):
  - line_mae_vs_features.pdf / line_r2_vs_features.pdf
      (one panel per target: metric vs number of features, one line per model)
  - box_mae_{target}.pdf / box_r2_{target}.pdf
      (per-fold metric distributions, grouped by feature count)
  - comparison_table.csv
      (wide mean +/- std table: rows = metric x model, columns = target x count)

Only reads the step-6 CSVs, so plots can be restyled without retraining.
Colors follow the model identity in a fixed order across every figure; the
Dummy baseline is drawn in dashed gray to separate it from the real models.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend and save plots directly.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR / "6_model_training"
FOLD_RESULTS_FILE = TRAINING_DIR / "fold_results.csv"
SUMMARY_FILE = TRAINING_DIR / "summary.csv"
OUTPUT_DIR = SCRIPT_DIR / "7_model_comparison"

TARGETS = ["GPR3mm", "GPR2mm", "GPR1mm"]
# Fixed model order and colors (CVD-validated categorical palette); the Dummy
# baseline is gray and dashed so it never reads as a competing model.
MODEL_COLORS = {
    "Dummy": "#898781",
    "Ridge": "#2a78d6",
    "SVR": "#1baf7a",
    "RandomForest": "#eda100",
    "XGBoost": "#008300",
    "MLP": "#4a3aa7",
}
MODEL_LABELS = {
    "Dummy": "Dummy (mean)",
    "Ridge": "Ridge",
    "SVR": "SVR (RBF)",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "MLP": "MLP",
}
METRICS = {"mae": "MAE (GPR percentage points)", "r2": "R²"}
R2_AXIS_FLOOR = -1.5  # fold R2 can be hugely negative on tiny folds; clip the axis

INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
PLOT_DPI = 300


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    for path in (FOLD_RESULTS_FILE, SUMMARY_FILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"Input file not found: {path}. Run 6_model_training.py first."
            )
    fold_df = pd.read_csv(FOLD_RESULTS_FILE, encoding="utf-8-sig")
    summary_df = pd.read_csv(SUMMARY_FILE, encoding="utf-8-sig")

    needed = {"target", "model", "n_features", "r2", "mae", "rmse"}
    if missing := needed - set(fold_df.columns):
        raise ValueError(f"fold_results.csv is missing columns: {', '.join(sorted(missing))}")
    unknown_models = set(fold_df["model"]) - set(MODEL_COLORS)
    if unknown_models:
        raise ValueError(
            f"fold_results.csv contains models without a color slot: {', '.join(sorted(unknown_models))}"
        )

    feature_counts = sorted(fold_df["n_features"].unique())
    print(
        f"Loaded {len(fold_df)} fold rows: models={list(MODEL_COLORS)}, "
        f"feature counts={feature_counts}."
    )
    return fold_df, summary_df, feature_counts


def style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: hairline y-grid, light baseline, muted tick labels."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor="#52514e")


def count_tick_labels(feature_counts: list[int]) -> list[str]:
    return [
        f"{n} (all)" if n == max(feature_counts) else str(n) for n in feature_counts
    ]


def model_offsets() -> np.ndarray:
    """Horizontal dodge so error bars / boxes of the 6 models do not overlap."""
    return np.linspace(-0.28, 0.28, len(MODEL_COLORS))


def line_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [], [],
            color=MODEL_COLORS[m],
            linestyle="--" if m == "Dummy" else "-",
            marker="o",
            markersize=5,
            linewidth=2,
            label=MODEL_LABELS[m],
        )
        for m in MODEL_COLORS
    ]


def make_line_plot(summary_df: pd.DataFrame, metric: str, feature_counts: list[int]) -> None:
    """One panel per target: mean +/- SD of the metric vs number of features."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(13, 4.4))
    positions = np.arange(len(feature_counts))
    offsets = model_offsets()

    clipped = False
    for ax, target in zip(axes, TARGETS):
        sub = summary_df[summary_df["target"] == target]
        if metric == "mae":
            y_bottom, y_top = 0.0, None
        else:
            low = (sub["r2_mean"] - sub["r2_std"]).min()
            clipped = clipped or low < R2_AXIS_FLOOR
            y_bottom, y_top = max(R2_AXIS_FLOOR, low - 0.05), 1.0
        for offset, model in zip(offsets, MODEL_COLORS):
            rows = (
                sub[sub["model"] == model]
                .set_index("n_features")
                .reindex(feature_counts)
            )
            mean = rows[f"{metric}_mean"]
            std = rows[f"{metric}_std"]
            # Clamp error bars to the axis range so clipped bars do not run
            # edge-to-edge and read as gridlines (R2 std can be huge).
            err_lower = np.clip(np.minimum(std, mean - y_bottom), 0, None)
            err_upper = std if y_top is None else np.clip(np.minimum(std, y_top - mean), 0, None)
            ax.errorbar(
                positions + offset,
                mean,
                yerr=np.vstack([err_lower, err_upper]),
                color=MODEL_COLORS[model],
                linestyle="--" if model == "Dummy" else "-",
                marker="o",
                markersize=5,
                linewidth=2,
                capsize=2.5,
                elinewidth=1,
            )
        style_axes(ax)
        ax.set_xticks(positions, count_tick_labels(feature_counts))
        ax.set_xlabel("Number of features", color="#52514e")
        ax.set_title(target, color=INK)
        ax.set_ylim(bottom=y_bottom, top=y_top)
    ylabel = METRICS[metric]
    if clipped:
        ylabel += f" (bars clipped at {R2_AXIS_FLOOR})"
    axes[0].set_ylabel(ylabel, color="#52514e")

    fig.suptitle(
        f"Outer-CV {METRICS[metric].split(' ')[0]} vs number of features "
        "(mean ± SD over 25 folds)",
        color=INK,
    )
    fig.legend(
        handles=line_legend_handles(),
        loc="lower center",
        ncol=len(MODEL_COLORS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(OUTPUT_DIR / f"line_{metric}_vs_features.pdf", dpi=PLOT_DPI)
    plt.close(fig)


def make_box_plot(
    fold_df: pd.DataFrame, target: str, metric: str, feature_counts: list[int]
) -> None:
    """Per-fold metric distributions for one target, grouped by feature count."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    positions = np.arange(len(feature_counts))
    offsets = model_offsets()
    sub = fold_df[fold_df["target"] == target]

    clipped = False
    for offset, model in zip(offsets, MODEL_COLORS):
        data = [
            sub.loc[
                (sub["model"] == model) & (sub["n_features"] == n), metric
            ].to_numpy()
            for n in feature_counts
        ]
        color = MODEL_COLORS[model]
        boxes = ax.boxplot(
            data,
            positions=positions + offset,
            widths=0.085,
            patch_artist=True,
            boxprops={"facecolor": to_rgba(color, 0.25), "edgecolor": color},
            whiskerprops={"color": color, "linewidth": 1},
            capprops={"color": color, "linewidth": 1},
            medianprops={"color": INK, "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "markerfacecolor": color,
                "markeredgecolor": "none",
            },
        )
        del boxes  # styling is fully set via the props above

    style_axes(ax)
    ax.set_xticks(positions, count_tick_labels(feature_counts))
    ax.set_xlabel("Number of features", color="#52514e")
    if metric == "r2":
        low = sub["r2"].min()
        if low < R2_AXIS_FLOOR:
            clipped = True
        ax.set_ylim(max(R2_AXIS_FLOOR, low - 0.05), 1.0)
    ylabel = METRICS[metric]
    if clipped:
        ylabel += f" (axis clipped at {R2_AXIS_FLOOR})"
    ax.set_ylabel(ylabel, color="#52514e")
    ax.set_title(f"{target}: per-fold {METRICS[metric].split(' ')[0]} (25 folds)", color=INK)

    fig.legend(
        handles=[
            Patch(facecolor=to_rgba(MODEL_COLORS[m], 0.25), edgecolor=MODEL_COLORS[m], label=MODEL_LABELS[m])
            for m in MODEL_COLORS
        ],
        loc="lower center",
        ncol=len(MODEL_COLORS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUTPUT_DIR / f"box_{metric}_{target}.pdf", dpi=PLOT_DPI)
    plt.close(fig)


def make_comparison_table(summary_df: pd.DataFrame, feature_counts: list[int]) -> None:
    """Wide mean +/- std table (rows = metric x model) for the dissertation."""
    decimals = {"r2": 3, "mae": 2, "rmse": 2}
    rows = []
    for metric, nd in decimals.items():
        for model in MODEL_COLORS:
            row: dict[str, str] = {"metric": metric, "model": MODEL_LABELS[model]}
            for target in TARGETS:
                for n in feature_counts:
                    match = summary_df[
                        (summary_df["target"] == target)
                        & (summary_df["model"] == model)
                        & (summary_df["n_features"] == n)
                    ]
                    if len(match) != 1:
                        raise ValueError(
                            f"summary.csv should have exactly one row for "
                            f"({target}, {model}, n={n}); found {len(match)}."
                        )
                    mean = match[f"{metric}_mean"].iloc[0]
                    std = match[f"{metric}_std"].iloc[0]
                    row[f"{target}_n{n}"] = f"{mean:.{nd}f} ± {std:.{nd}f}"
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "comparison_table.csv", index=False, encoding="utf-8-sig"
    )


def print_best_configs(summary_df: pd.DataFrame, feature_counts: list[int]) -> None:
    """Lowest-MAE configuration per target among the deployment candidates.

    Dummy and the full feature set are excluded, mirroring the candidate rules
    of 8_operating_point.py / 10_site_error_analysis.py, so this console line
    can never contradict the deployed combo.
    """
    n_full = max(feature_counts)
    print(
        "\nBest deployment-candidate configuration per target "
        f"(lowest mean MAE; Dummy and the full n={n_full} feature set excluded):"
    )
    for target in TARGETS:
        sub = summary_df[
            (summary_df["target"] == target)
            & (summary_df["model"] != "Dummy")
            & (summary_df["n_features"] < n_full)
        ]
        best = sub.loc[sub["mae_mean"].idxmin()]
        print(
            f"  [{target}] {best['model']:<12} n_features={int(best['n_features']):>2}  "
            f"MAE={best['mae_mean']:.3f}+/-{best['mae_std']:.3f}  "
            f"R2={best['r2_mean']:.3f}+/-{best['r2_std']:.3f}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_df, summary_df, feature_counts = load_results()

    for metric in METRICS:
        make_line_plot(summary_df, metric, feature_counts)
        for target in TARGETS:
            make_box_plot(fold_df, target, metric, feature_counts)
    make_comparison_table(summary_df, feature_counts)
    print_best_configs(summary_df, feature_counts)

    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
