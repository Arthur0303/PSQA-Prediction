"""Classification analysis of the out-of-fold GPR predictions (stages 3-4).

Clinical rule (true labels, from measured GPR):
  2 = GPR3mm >= 99 and GPR2mm >= 97   -> auto-pass, no manual review
  0 = GPR3mm <  98 and GPR2mm <  92   -> replan directly
  1 = everything else                 -> manual review

Two analyses, both on the pooled row x repeat out-of-fold predictions (every
row keeps one prediction per RepeatedKFold repeat; repeated measurements of
the same plan stay separate rows, as in step 6):

Part 1 - baseline: per target, the configuration with the lowest mean outer-CV
MAE classifies with the raw clinical thresholds (no margin). Dummy and the
full-feature configurations are excluded as deployment candidates throughout
this script (study decision: the full-feature configs are not considered for
deployment). Reported: confusion matrix, overall and per-class accuracy.

Part 2 - residual-quantile margins: per-target margins are read off the
out-of-fold residual distribution e = pred - true. For quantile levels
(q2, q0) the class-2 margin is the q2-quantile of e (overprediction risk) and
the class-0 margin the q0-quantile of -e (underprediction risk):
  pred 2:  pred_GPR3mm >= 99 + m2(3mm)  and  pred_GPR2mm >= 97 + m2(2mm)
  pred 0:  pred_GPR3mm <  98 - m0(3mm)  and  pred_GPR2mm <  92 - m0(2mm)
Margins may be negative (a threshold looser than the clinical rule); a point
matching both boxes at once is ambiguous and stays in class 1. Every
GPR3mm-config x GPR2mm-config x (q2, q0) combination is scored. Feasible
solutions have ZERO class-0 <-> class-2 cross-misclassifications - the safety
principle: an uncertain plan may fall into manual review, but a replan
candidate must never auto-pass (nor the reverse). Among feasible solutions
the score F1(class 2) + F1(class 0) is maximized; 100% precision is NOT
required anymore. Ties go to higher overall accuracy, then to the more
conservative (larger) q2 + q0.

The final comparison (baseline / chosen combo without margins / chosen combo
with tuned margins) and a recommendation are printed and saved. Caveat: the
margins are tuned and evaluated on the same out-of-fold points, so the
reported metrics carry an optimism bias - state this next to the numbers in
the write-up (step 9 quantifies the analogous bias for feature selection).

Inputs (produced by 6_model_training.py / 4_feature_importance.py):
  - Model/6_model_training/predictions.csv   (out-of-fold predictions per config)
  - Model/6_model_training/fold_results.csv  (per-fold MAE -> model selection)
  - Model/4_kept_features_targets.csv        (measured GPR -> true labels)

Outputs (under Model/8_operating_point/):
  - baseline_confusion_matrix.csv / .pdf  (part 1)
  - margin_search.csv       (best feasible solution per model combo, ranked)
  - tuned_confusion_matrix.csv / .pdf     (part 2, tuned margins)
  - metrics_comparison.csv  (the three schemes side by side)
  - residuals_margins.pdf   (residual distributions + chosen margins)
  - scatter_pred_vs_true.pdf
  - decision_plane.pdf      (pooled out-of-fold predictions in the
                             (GPR3mm, GPR2mm) plane with the raw clinical and
                             tuned decision boxes)
  - operating_point.json    (recommended scheme; consumed by step 9 and
                             train_final_model.py, and through the saved model
                             metadata by predict_new_plan.py)
"""

import json
import math
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend and save plots directly.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import beta


SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTIONS_FILE = SCRIPT_DIR / "6_model_training" / "predictions.csv"
FOLD_RESULTS_FILE = SCRIPT_DIR / "6_model_training" / "fold_results.csv"
KEPT_FILE = SCRIPT_DIR / "4_kept_features_targets.csv"
OUTPUT_DIR = SCRIPT_DIR / "8_operating_point"
# Files written by the pre-2026-07 version of this script (100%-precision
# margin scan); removed on every run so the output folder never mixes the two
# generations of the analysis.
OBSOLETE_OUTPUTS = (
    "combo_ranking.csv",
    "curve_data.csv",
    "curve_class2.png",
    "curve_class0.png",
    "confusion_matrix.csv",
    "confusion_matrix.png",
    # PNG versions of the current figures, from before the 2026-07 switch to
    # PDF-only vector output.
    "baseline_confusion_matrix.png",
    "tuned_confusion_matrix.png",
    "residuals_margins.png",
    "scatter_pred_vs_true.png",
)

ALL_TARGETS = ("GPR3mm", "GPR2mm", "GPR1mm")  # target columns in the kept file
CLS_TARGETS = ("GPR3mm", "GPR2mm")  # GPR1mm is not part of the clinical rule
CLASSES = (2, 1, 0)  # row/column order of every confusion matrix
THRESHOLDS = {
    "class2": {"GPR3mm": 99.0, "GPR2mm": 97.0},
    "class0": {"GPR3mm": 98.0, "GPR2mm": 92.0},
}
EXCLUDED_MODELS = ("Dummy",)  # baseline only, never a deployment candidate
# Quantile levels scanned for both margin sides; the two extra levels give the
# search enough reach to silence a single badly mispredicted outlier point.
Q_GRID = np.round(np.append(np.arange(0.05, 0.95 + 1e-9, 0.05), [0.975, 0.99]), 3)
CI_ALPHA = 0.05  # two-sided 95% Clopper-Pearson
# None -> take the top combo of the margin search. To pin the choice, set
# e.g. (("XGBoost", 15), ("XGBoost", 15)) as ((model3, n3), (model2, n2)).
CHOSEN_COMBO: tuple[tuple[str, int], tuple[str, int]] | None = None
SEARCH_PRINT_TOP = 10
PLOT_DPI = 300

INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_BLUE = "#2a78d6"
# Label 2/1/0 is a state (auto-pass / manual review / replan), so the reserved
# status palette applies; marker shape is the secondary, color-independent cue.
LABEL_STYLE = {
    2: {"color": "#0ca30c", "marker": "o", "name": "2 auto-pass"},
    1: {"color": "#fab219", "marker": "^", "name": "1 manual review"},
    0: {"color": "#d03b3b", "marker": "X", "name": "0 replan"},
}
SEQUENTIAL_BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def load_true_labels() -> tuple[np.ndarray, pd.DataFrame, int]:
    """Return (label per row, measured GPR frame, full feature count)."""
    if not KEPT_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {KEPT_FILE}")
    df = pd.read_csv(KEPT_FILE, encoding="utf-8-sig")
    g3 = df["GPR3mm"].to_numpy(float)
    g2 = df["GPR2mm"].to_numpy(float)
    labels = np.ones(len(df), dtype=int)
    labels[(g3 >= THRESHOLDS["class2"]["GPR3mm"]) & (g2 >= THRESHOLDS["class2"]["GPR2mm"])] = 2
    labels[(g3 < THRESHOLDS["class0"]["GPR3mm"]) & (g2 < THRESHOLDS["class0"]["GPR2mm"])] = 0
    counts = pd.Series(labels).value_counts().sort_index()
    print(f"True labels from {KEPT_FILE.name}: " + ", ".join(f"{c}->{n}" for c, n in counts.items()))
    n_full_features = df.shape[1] - sum(1 for c in ALL_TARGETS if c in df.columns)
    return labels, df[list(CLS_TARGETS)], n_full_features


def load_prediction_matrices(
    measured: pd.DataFrame, n_full_features: int
) -> tuple[dict[str, dict[tuple[str, int], np.ndarray]], int]:
    """Return {target: {(model, n_features): pred matrix (n_rows x n_repeats)}}.

    Dummy and full-feature configurations are dropped here, so every later
    stage only ever sees deployment candidates. Also cross-checks that y_true
    in predictions.csv matches the measured GPR.
    """
    if not PREDICTIONS_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {PREDICTIONS_FILE}. Re-run 6_model_training.py "
            "(the current version saves out-of-fold predictions)."
        )
    df = pd.read_csv(PREDICTIONS_FILE, encoding="utf-8-sig")
    needed = {"target", "model", "n_features", "repeat", "row_index", "y_true", "y_pred"}
    if missing := needed - set(df.columns):
        raise ValueError(f"predictions.csv is missing columns: {', '.join(sorted(missing))}")

    df = df[
        df["target"].isin(CLS_TARGETS)
        & ~df["model"].isin(EXCLUDED_MODELS)
        & (df["n_features"] < n_full_features)
    ]
    n_rows = len(measured)
    n_repeats = df["repeat"].nunique()

    matrices: dict[str, dict[tuple[str, int], np.ndarray]] = {t: {} for t in CLS_TARGETS}
    for (target, model, n_features), group in df.groupby(["target", "model", "n_features"]):
        wide = group.pivot(index="row_index", columns="repeat", values="y_pred")
        if wide.shape != (n_rows, n_repeats) or wide.isna().any().any():
            raise ValueError(
                f"[{target}] {model} n={n_features}: expected one prediction per "
                f"row per repeat ({n_rows}x{n_repeats}), got shape {wide.shape}."
            )
        wide = wide.sort_index()
        if not np.array_equal(wide.index.to_numpy(), np.arange(n_rows)):
            raise ValueError(f"[{target}] {model} n={n_features}: row_index is not 0..{n_rows - 1}.")
        truth = group.groupby("row_index")["y_true"].first().sort_index().to_numpy(float)
        if not np.allclose(truth, measured[target].to_numpy(float)):
            raise ValueError(
                f"[{target}] {model} n={n_features}: y_true in predictions.csv does "
                f"not match {KEPT_FILE.name}. Re-run steps 4 and 6 from the same data."
            )
        matrices[target][(str(model), int(n_features))] = wide.to_numpy(float)

    for target in CLS_TARGETS:
        if not matrices[target]:
            raise ValueError(f"No usable configurations found for {target}.")
        print(
            f"{target}: {len(matrices[target])} candidate configurations "
            f"(Dummy and n={n_full_features} full-feature configs excluded)."
        )
    return matrices, n_repeats


def load_cv_mae() -> dict[tuple[str, str, int], float]:
    """Return {(target, model, n_features): mean outer-fold MAE}."""
    fold_df = pd.read_csv(FOLD_RESULTS_FILE, encoding="utf-8-sig")
    grouped = fold_df.groupby(["target", "model", "n_features"])["mae"].mean()
    return {key: float(value) for key, value in grouped.items()}


def lowest_mae_configs(
    matrices: dict[str, dict[tuple[str, int], np.ndarray]],
    cv_mae: dict[tuple[str, str, int], float],
) -> dict[str, tuple[str, int]]:
    """Per target, the candidate configuration with the lowest mean CV MAE."""
    return {
        target: min(matrices[target], key=lambda cfg: cv_mae[(target, *cfg)])
        for target in CLS_TARGETS
    }


def precision_ci_lower(correct: int, total: int) -> float:
    """Clopper-Pearson two-sided lower bound on precision."""
    if total == 0:
        return float("nan")
    if correct == 0:
        return 0.0
    return float(beta.ppf(CI_ALPHA / 2, correct, total - correct + 1))


def zero_margins() -> dict[str, dict[str, float]]:
    return {side: {t: 0.0 for t in CLS_TARGETS} for side in ("class2", "class0")}


def predicted_classes(
    pred3: np.ndarray, pred2: np.ndarray, margins: dict[str, dict[str, float]]
) -> np.ndarray:
    """Apply the margin rule pointwise.

    With negative margins the class-2 and class-0 boxes can overlap; a point
    satisfying both is contradictory evidence and stays in manual review.
    """
    mask2 = (pred3 >= THRESHOLDS["class2"]["GPR3mm"] + margins["class2"]["GPR3mm"]) & (
        pred2 >= THRESHOLDS["class2"]["GPR2mm"] + margins["class2"]["GPR2mm"]
    )
    mask0 = (pred3 < THRESHOLDS["class0"]["GPR3mm"] - margins["class0"]["GPR3mm"]) & (
        pred2 < THRESHOLDS["class0"]["GPR2mm"] - margins["class0"]["GPR2mm"]
    )
    predicted = np.ones(pred3.shape, dtype=int)
    predicted[mask2 & ~mask0] = 2
    predicted[mask0 & ~mask2] = 0
    return predicted


def confusion_counts(predicted: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """3x3 counts, rows = true, cols = predicted, both in CLASSES order."""
    lab = np.broadcast_to(labels[:, None], predicted.shape)
    return np.array(
        [[int(((lab == t) & (predicted == p)).sum()) for p in CLASSES] for t in CLASSES]
    )


def metrics_from_confusion(counts: np.ndarray) -> dict[str, float]:
    """Overall/per-class accuracy, precision, F1, cross errors and rates."""
    idx = {cls: i for i, cls in enumerate(CLASSES)}
    total = int(counts.sum())
    metrics: dict[str, float] = {
        "overall_accuracy": float(np.trace(counts) / total),
        "cross_true0_pred2": int(counts[idx[0], idx[2]]),
        "cross_true2_pred0": int(counts[idx[2], idx[0]]),
    }
    for cls in CLASSES:
        i = idx[cls]
        support = int(counts[i].sum())
        flagged = int(counts[:, i].sum())
        tp = int(counts[i, i])
        metrics[f"n_true_{cls}"] = support
        metrics[f"n_pred_{cls}"] = flagged
        metrics[f"recall_{cls}"] = tp / support if support else float("nan")
        metrics[f"precision_{cls}"] = tp / flagged if flagged else float("nan")
        metrics[f"f1_{cls}"] = 2 * tp / (support + flagged) if support + flagged else 0.0
    metrics["autopass_rate"] = metrics["n_pred_2"] / total
    metrics["manual_rate"] = metrics["n_pred_1"] / total
    metrics["replan_rate"] = metrics["n_pred_0"] / total
    for cls in (2, 0):
        i = idx[cls]
        metrics[f"precision{cls}_ci95_lower"] = precision_ci_lower(
            int(counts[i, i]), int(counts[:, i].sum())
        )
    metrics["score"] = metrics["f1_2"] + metrics["f1_0"]
    return metrics


def evaluate_scheme(
    matrices: dict[str, dict[tuple[str, int], np.ndarray]],
    configs: dict[str, tuple[str, int]],
    margins: dict[str, dict[str, float]],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return (predicted classes, confusion counts, metrics) for one scheme."""
    pred3 = matrices["GPR3mm"][configs["GPR3mm"]]
    pred2 = matrices["GPR2mm"][configs["GPR2mm"]]
    predicted = predicted_classes(pred3, pred2, margins)
    counts = confusion_counts(predicted, labels)
    metrics = metrics_from_confusion(counts)
    for cls, name in ((2, "autopass"), (0, "replan")):
        per_repeat = (predicted == cls).mean(axis=0)
        metrics[f"{name}_repeat_mean"] = float(per_repeat.mean())
        metrics[f"{name}_repeat_std"] = float(per_repeat.std(ddof=1))
    return predicted, counts, metrics


def build_scan_data(
    matrices: dict[str, dict[tuple[str, int], np.ndarray]], measured: pd.DataFrame
) -> dict[str, dict[tuple[str, int], dict]]:
    """Per target/config: pooled residuals, per-q margins and threshold masks.

    mask2[i] / mask0[i] hold the single-target threshold test for Q_GRID[i],
    so the combo scan only needs boolean AND operations.
    """
    data: dict[str, dict[tuple[str, int], dict]] = {t: {} for t in CLS_TARGETS}
    for target in CLS_TARGETS:
        true = measured[target].to_numpy(float)[:, None]
        upper = THRESHOLDS["class2"][target]
        lower = THRESHOLDS["class0"][target]
        for config, pred in matrices[target].items():
            residuals = (pred - true).ravel()
            margins2 = np.quantile(residuals, Q_GRID)
            margins0 = np.quantile(-residuals, Q_GRID)
            data[target][config] = {
                "residuals": residuals,
                "margins2": margins2,
                "margins0": margins0,
                "mask2": pred[None, :, :] >= upper + margins2[:, None, None],
                "mask0": pred[None, :, :] < lower - margins0[:, None, None],
            }
    return data


def search_margins(
    scan_data: dict[str, dict[tuple[str, int], dict]],
    cv_mae: dict[tuple[str, str, int], float],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Best feasible (q2, q0) per model combo, ranked by F1(2) + F1(0).

    Feasible = zero class-0 <-> class-2 cross-misclassifications on the pooled
    points. Ties within a combo go to higher overall accuracy, then larger
    (more conservative) q2 + q0; the same keys rank the combos.
    """
    lab2 = (labels == 2)[:, None]
    lab1 = (labels == 1)[:, None]
    lab0 = (labels == 0)[:, None]
    rows = []
    for (model3, n3), d3 in scan_data["GPR3mm"].items():
        for (model2, n2), d2 in scan_data["GPR2mm"].items():
            mask2_all = d3["mask2"] & d2["mask2"]  # (n_q, n_rows, n_repeats)
            mask0_all = d3["mask0"] & d2["mask0"]
            total = mask2_all[0].size
            support2 = int(lab2.sum()) * mask2_all.shape[2]
            support0 = int(lab0.sum()) * mask2_all.shape[2]

            best_key, best_idx = None, None
            for i2, q2 in enumerate(Q_GRID):
                for i0, q0 in enumerate(Q_GRID):
                    pred2_mask = mask2_all[i2] & ~mask0_all[i0]
                    pred0_mask = mask0_all[i0] & ~mask2_all[i2]
                    if (pred2_mask & lab0).any() or (pred0_mask & lab2).any():
                        continue  # cross-misclassification -> infeasible
                    tp2 = int((pred2_mask & lab2).sum())
                    tp0 = int((pred0_mask & lab0).sum())
                    flagged2 = int(pred2_mask.sum())
                    flagged0 = int(pred0_mask.sum())
                    f1_2 = 2 * tp2 / (support2 + flagged2) if support2 + flagged2 else 0.0
                    f1_0 = 2 * tp0 / (support0 + flagged0) if support0 + flagged0 else 0.0
                    correct1 = int((~pred2_mask & ~pred0_mask & lab1).sum())
                    accuracy = (tp2 + tp0 + correct1) / total
                    key = (f1_2 + f1_0, accuracy, q2 + q0)
                    if best_key is None or key > best_key:
                        best_key, best_idx = key, (i2, i0)

            row = {
                "model_3mm": model3,
                "n_features_3mm": n3,
                "model_2mm": model2,
                "n_features_2mm": n2,
                "cv_mae_3mm": round(cv_mae[("GPR3mm", model3, n3)], 4),
                "cv_mae_2mm": round(cv_mae[("GPR2mm", model2, n2)], 4),
                "feasible": best_idx is not None,
            }
            if best_idx is not None:
                i2, i0 = best_idx
                predicted = np.ones(mask2_all[0].shape, dtype=int)
                predicted[mask2_all[i2] & ~mask0_all[i0]] = 2
                predicted[mask0_all[i0] & ~mask2_all[i2]] = 0
                metrics = metrics_from_confusion(confusion_counts(predicted, labels))
                row.update(
                    q2=float(Q_GRID[i2]),
                    q0=float(Q_GRID[i0]),
                    margin2_GPR3mm=round(float(d3["margins2"][i2]), 4),
                    margin2_GPR2mm=round(float(d2["margins2"][i2]), 4),
                    margin0_GPR3mm=round(float(d3["margins0"][i0]), 4),
                    margin0_GPR2mm=round(float(d2["margins0"][i0]), 4),
                    **{
                        name: round(metrics[name], 4)
                        for name in (
                            "score", "f1_2", "f1_0", "recall_2", "precision_2",
                            "recall_0", "precision_0", "overall_accuracy",
                            "autopass_rate", "replan_rate", "manual_rate",
                        )
                    },
                )
            rows.append(row)

    search = pd.DataFrame(rows)
    search["_tie_q"] = search.get("q2", np.nan) + search.get("q0", np.nan)
    search = search.sort_values(
        ["feasible", "score", "overall_accuracy", "_tie_q"],
        ascending=False, kind="stable", na_position="last",
    ).drop(columns="_tie_q").reset_index(drop=True)
    search.insert(0, "rank", range(1, len(search) + 1))
    return search


def pick_combo(search: pd.DataFrame) -> pd.Series:
    if CHOSEN_COMBO is None:
        top = search.iloc[0]
        if not top["feasible"]:
            raise ValueError("No combo has a feasible (zero cross-error) solution.")
        return top
    (model3, n3), (model2, n2) = CHOSEN_COMBO
    match = search[
        (search["model_3mm"] == model3)
        & (search["n_features_3mm"] == n3)
        & (search["model_2mm"] == model2)
        & (search["n_features_2mm"] == n2)
    ]
    if len(match) != 1:
        raise ValueError(f"CHOSEN_COMBO {CHOSEN_COMBO} not found in the search results.")
    if not match.iloc[0]["feasible"]:
        raise ValueError(f"CHOSEN_COMBO {CHOSEN_COMBO} has no feasible solution.")
    return match.iloc[0]


def margins_from_row(row: pd.Series) -> dict[str, dict[str, float]]:
    return {
        "class2": {t: float(row[f"margin2_{t}"]) for t in CLS_TARGETS},
        "class0": {t: float(row[f"margin0_{t}"]) for t in CLS_TARGETS},
    }


def format_metrics(metrics: dict[str, float]) -> str:
    recalls = "/".join(
        f"{metrics[f'recall_{cls}']:.3f}" if math.isfinite(metrics[f"recall_{cls}"]) else "-"
        for cls in CLASSES
    )
    return (
        f"overall acc {metrics['overall_accuracy']:.3f} | recall 2/1/0 {recalls} | "
        f"F1(2)+F1(0) {metrics['score']:.3f} | "
        f"cross 0->pred2: {metrics['cross_true0_pred2']}, 2->pred0: {metrics['cross_true2_pred0']}"
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


def save_confusion(counts: np.ndarray, stem: str, title: str) -> pd.DataFrame:
    """Confusion matrix (rows = true, cols = predicted) as CSV and heatmap."""
    matrix = pd.DataFrame(
        counts,
        index=[f"true_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    )
    matrix.to_csv(OUTPUT_DIR / f"{stem}.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(counts, cmap=SEQUENTIAL_BLUES, vmin=0)
    threshold = counts.max() * 0.55
    for i in range(3):
        for j in range(3):
            share = counts[i, j] / counts[i].sum() if counts[i].sum() else 0.0
            ax.text(
                j, i, f"{counts[i, j]}\n({100 * share:.0f}%)",
                ha="center", va="center", fontsize=10,
                color="#ffffff" if counts[i, j] > threshold else INK,
            )
    names = {2: "2 auto-pass", 1: "1 manual", 0: "0 replan"}
    ax.set_xticks(range(3), [names[c] for c in CLASSES])
    ax.set_yticks(range(3), [names[c] for c in CLASSES])
    ax.set_xlabel("Predicted", color="#52514e")
    ax.set_ylabel("True", color="#52514e")
    ax.tick_params(colors=MUTED, labelcolor="#52514e")
    ax.set_title(f"{title}\n(row % of true class)", color=INK)
    fig.colorbar(image, ax=ax, shrink=0.75, label="row x repeat points")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", dpi=PLOT_DPI)
    plt.close(fig)
    return matrix


def plot_residuals(
    residuals: dict[str, np.ndarray],
    margins: dict[str, dict[str, float]],
    q2: float,
    q0: float,
) -> None:
    """Residual histograms with the chosen quantile margins marked."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, target in zip(axes, CLS_TARGETS):
        ax.hist(residuals[target], bins=40, color=SERIES_BLUE, alpha=0.85)
        m2 = margins["class2"][target]
        m0 = margins["class0"][target]
        # The class-0 margin is a quantile of -e, so it sits at -m0 on the e axis.
        ax.axvline(m2, color=LABEL_STYLE[2]["color"], linewidth=1.5, linestyle="--")
        ax.axvline(-m0, color=LABEL_STYLE[0]["color"], linewidth=1.5, linestyle=":")
        ax.annotate(
            f"class-2 margin m2 = Q{q2:g}(e) = {m2:+.2f}",
            (0.98, 0.95), xycoords="axes fraction", ha="right",
            color="#52514e", fontsize=8,
        )
        ax.annotate(
            f"class-0 margin m0 = Q{q0:g}(-e) = {m0:+.2f} (line at -m0)",
            (0.98, 0.88), xycoords="axes fraction", ha="right",
            color="#52514e", fontsize=8,
        )
        style_axes(ax)
        ax.set_title(target, color=INK)
        ax.set_xlabel(f"Out-of-fold residual e = pred - true (pp)", color="#52514e")
        ax.set_ylabel("row x repeat points", color="#52514e")
    fig.suptitle("Residual distributions and the chosen quantile margins", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "residuals_margins.pdf", dpi=PLOT_DPI)
    plt.close(fig)


def plot_scatter(
    matrices: dict[str, dict[tuple[str, int], np.ndarray]],
    configs: dict[str, tuple[str, int]],
    measured: pd.DataFrame,
    labels: np.ndarray,
    margins: dict[str, dict[str, float]],
) -> None:
    """Pred vs true per target, colored by true label, with decision thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, target in zip(axes, CLS_TARGETS):
        pred = matrices[target][configs[target]]
        true = measured[target].to_numpy(float)
        for cls, style in LABEL_STYLE.items():
            mask = labels == cls
            ax.scatter(
                np.repeat(true[mask], pred.shape[1]),
                pred[mask].ravel(),
                s=22, alpha=0.6, linewidths=0,
                color=style["color"], marker=style["marker"], label=style["name"],
            )
        ax.axvline(THRESHOLDS["class2"][target], color=BASELINE, linewidth=1)
        ax.axvline(THRESHOLDS["class0"][target], color=BASELINE, linewidth=1)
        upper = THRESHOLDS["class2"][target] + margins["class2"][target]
        lower = THRESHOLDS["class0"][target] - margins["class0"][target]
        ax.axhline(upper, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(
            f"pred 2 above {upper:.2f}", (0.02, upper), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points", color="#52514e", fontsize=8,
        )
        ax.axhline(lower, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(
            f"pred 0 below {lower:.2f}", (0.02, lower), xycoords=("axes fraction", "data"),
            xytext=(0, -11), textcoords="offset points", color="#52514e", fontsize=8,
        )
        style_axes(ax)
        model, n_features = configs[target]
        ax.set_title(f"{target}: {model} (n={n_features})", color=INK)
        ax.set_xlabel(f"Measured {target} (%)", color="#52514e")
        ax.set_ylabel(f"Out-of-fold predicted {target} (%)", color="#52514e")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.0), title=None,
    )
    fig.suptitle(
        "Out-of-fold predictions vs measurement (one point per plan row x repeat)",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "scatter_pred_vs_true.pdf", dpi=PLOT_DPI)
    plt.close(fig)


def plot_decision_plane(
    matrices: dict[str, dict[tuple[str, int], np.ndarray]],
    configs: dict[str, tuple[str, int]],
    labels: np.ndarray,
    margins: dict[str, dict[str, float]],
) -> None:
    """Pooled predictions in the (GPR3mm, GPR2mm) plane with both decision boxes.

    The class-2 box is the upper-right corner (both predictions at or above
    their thresholds), the class-0 box the lower-left corner (both below).
    The raw clinical corners (margin 0) are drawn next to the tuned ones, so
    the margin shift is visible as the gap between the two boundaries.
    """
    pred3 = matrices["GPR3mm"][configs["GPR3mm"]]
    pred2 = matrices["GPR2mm"][configs["GPR2mm"]]
    lab = np.broadcast_to(labels[:, None], pred3.shape)

    raw2 = (THRESHOLDS["class2"]["GPR3mm"], THRESHOLDS["class2"]["GPR2mm"])
    raw0 = (THRESHOLDS["class0"]["GPR3mm"], THRESHOLDS["class0"]["GPR2mm"])
    tuned2 = (
        raw2[0] + margins["class2"]["GPR3mm"],
        raw2[1] + margins["class2"]["GPR2mm"],
    )
    tuned0 = (
        raw0[0] - margins["class0"]["GPR3mm"],
        raw0[1] - margins["class0"]["GPR2mm"],
    )

    pad = 0.8
    x_lo = min(pred3.min(), raw0[0], tuned0[0]) - pad
    x_hi = max(pred3.max(), raw2[0], tuned2[0]) + pad
    y_lo = min(pred2.min(), raw0[1], tuned0[1]) - pad
    y_hi = max(pred2.max(), raw2[1], tuned2[1]) + pad

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    for cls, style in LABEL_STYLE.items():
        mask = lab == cls
        ax.scatter(
            pred3[mask], pred2[mask], s=22, alpha=0.6, linewidths=0,
            color=style["color"], marker=style["marker"], label=style["name"],
        )

    def draw_corner(x: float, y: float, upper: bool, **line_kwargs) -> None:
        """L-shaped boundary of one decision box, clipped at the axis limits."""
        if upper:  # class-2 box: right of x AND above y
            ax.plot([x, x, x_hi], [y_hi, y, y], **line_kwargs)
        else:  # class-0 box: left of x AND below y
            ax.plot([x, x, x_lo], [y_lo, y, y], **line_kwargs)

    draw_corner(*raw2, upper=True, color=BASELINE, linewidth=1.2)
    draw_corner(*raw0, upper=False, color=BASELINE, linewidth=1.2)
    draw_corner(
        *tuned2, upper=True,
        color=LABEL_STYLE[2]["color"], linewidth=1.5, linestyle="--",
    )
    draw_corner(
        *tuned0, upper=False,
        color=LABEL_STYLE[0]["color"], linewidth=1.5, linestyle=":",
    )

    style_axes(ax)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    model3, n3 = configs["GPR3mm"]
    model2, n2 = configs["GPR2mm"]
    ax.set_title(
        f"GPR3mm: {model3} (n={n3})    GPR2mm: {model2} (n={n2})",
        color=INK, fontsize=10,
    )
    ax.set_xlabel("Out-of-fold predicted GPR3mm (%)", color="#52514e")
    ax.set_ylabel("Out-of-fold predicted GPR2mm (%)", color="#52514e")

    handles, names = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color=BASELINE, linewidth=1.2),
        Line2D([], [], color=LABEL_STYLE[2]["color"], linewidth=1.5, linestyle="--"),
        Line2D([], [], color=LABEL_STYLE[0]["color"], linewidth=1.5, linestyle=":"),
    ]
    names += [
        "clinical thresholds (no margin)",
        f"tuned class-2 box ({tuned2[0]:.2f} / {tuned2[1]:.2f})",
        f"tuned class-0 box ({tuned0[0]:.2f} / {tuned0[1]:.2f})",
    ]
    fig.legend(
        handles, names, loc="lower center", ncol=2, frameon=False,
        fontsize=8.5, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Decision plane: pooled out-of-fold predictions vs the decision boxes",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "decision_plane.pdf", dpi=PLOT_DPI)
    plt.close(fig)


def jsonable(value):
    """Recursively convert numpy scalars and NaN for strict-JSON output."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(value) else round(float(value), 6)
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in OBSOLETE_OUTPUTS:
        (OUTPUT_DIR / name).unlink(missing_ok=True)

    labels, measured, n_full_features = load_true_labels()
    matrices, n_repeats = load_prediction_matrices(measured, n_full_features)
    cv_mae = load_cv_mae()
    n_points = labels.size * n_repeats

    # ---- Part 1: lowest-MAE configs, raw clinical thresholds (no margin) ----
    print("\n=== Part 1: baseline - lowest CV MAE per target, raw thresholds ===")
    base_configs = lowest_mae_configs(matrices, cv_mae)
    for target, (model, n) in base_configs.items():
        print(f"  {target}: {model} (n={n}), CV MAE {cv_mae[(target, model, n)]:.4f}")
    _, base_counts, base_metrics = evaluate_scheme(
        matrices, base_configs, zero_margins(), labels
    )
    base_matrix = save_confusion(
        base_counts, "baseline_confusion_matrix",
        "Baseline: lowest-MAE models, no margin",
    )
    print(base_matrix.to_string())
    print("  " + format_metrics(base_metrics))
    print(
        "  Per-class accuracy (recall): "
        + ", ".join(f"class {cls}: {base_metrics[f'recall_{cls}']:.3f}" for cls in CLASSES)
    )

    # ---- Part 2: residual-quantile margin search over all combos ----
    print("\n=== Part 2: residual-quantile margin search ===")
    scan_data = build_scan_data(matrices, measured)
    search = search_margins(scan_data, cv_mae, labels)
    search.to_csv(OUTPUT_DIR / "margin_search.csv", index=False, encoding="utf-8-sig")
    n_feasible = int(search["feasible"].sum())
    print(
        f"Scanned {len(search)} combos x {len(Q_GRID)}x{len(Q_GRID)} quantile pairs; "
        f"{n_feasible} combos have a feasible (zero 0<->2 cross-error) solution."
    )
    show_cols = [
        "rank", "model_3mm", "n_features_3mm", "model_2mm", "n_features_2mm",
        "q2", "q0", "score", "f1_2", "f1_0", "recall_2", "recall_0", "overall_accuracy",
    ]
    print(search[show_cols].head(SEARCH_PRINT_TOP).to_string(index=False))

    combo_row = pick_combo(search)
    tuned_configs = {
        "GPR3mm": (str(combo_row["model_3mm"]), int(combo_row["n_features_3mm"])),
        "GPR2mm": (str(combo_row["model_2mm"]), int(combo_row["n_features_2mm"])),
    }
    tuned_margins = margins_from_row(combo_row)
    q2, q0 = float(combo_row["q2"]), float(combo_row["q0"])
    origin = "top of search" if CHOSEN_COMBO is None else "pinned via CHOSEN_COMBO"
    print(
        f"\nChosen combo ({origin}): "
        f"GPR3mm={tuned_configs['GPR3mm'][0]}(n={tuned_configs['GPR3mm'][1]}), "
        f"GPR2mm={tuned_configs['GPR2mm'][0]}(n={tuned_configs['GPR2mm'][1]}); "
        f"q2={q2:g}, q0={q0:g}"
    )
    print(
        "Margins (pp): class2 "
        + ", ".join(f"{t} {tuned_margins['class2'][t]:+.2f}" for t in CLS_TARGETS)
        + " | class0 "
        + ", ".join(f"{t} {tuned_margins['class0'][t]:+.2f}" for t in CLS_TARGETS)
    )

    # ---- The three schemes, side by side ----
    schemes = [
        {
            "name": "baseline_lowest_mae_no_margin",
            "label": "A baseline (lowest-MAE combo, no margin)",
            "configs": base_configs, "margins": zero_margins(), "q2": None, "q0": None,
        },
        {
            "name": "tuned_combo_no_margin",
            "label": "B chosen combo, no margin",
            "configs": tuned_configs, "margins": zero_margins(), "q2": None, "q0": None,
        },
        {
            "name": "tuned_combo_residual_margin",
            "label": "C chosen combo, residual-quantile margins",
            "configs": tuned_configs, "margins": tuned_margins, "q2": q2, "q0": q0,
        },
    ]
    for scheme in schemes:
        predicted, counts, metrics = evaluate_scheme(
            matrices, scheme["configs"], scheme["margins"], labels
        )
        if int(counts.sum()) != n_points:
            raise AssertionError(f"{scheme['name']}: confusion total != {n_points}.")
        scheme.update(predicted=predicted, counts=counts, metrics=metrics)

    tuned_scheme = schemes[2]
    if tuned_scheme["metrics"]["cross_true0_pred2"] or tuned_scheme["metrics"]["cross_true2_pred0"]:
        raise AssertionError(
            "Tuned scheme has cross-misclassifications; the pointwise rule "
            "diverged from the margin search."
        )
    if not np.isclose(tuned_scheme["metrics"]["score"], combo_row["score"], atol=1e-3):
        raise AssertionError("Tuned scheme score does not reproduce the search result.")

    comparison_rows = []
    for scheme in schemes:
        row = {
            "scheme": scheme["name"],
            "model_3mm": scheme["configs"]["GPR3mm"][0],
            "n_features_3mm": scheme["configs"]["GPR3mm"][1],
            "model_2mm": scheme["configs"]["GPR2mm"][0],
            "n_features_2mm": scheme["configs"]["GPR2mm"][1],
            "q2": scheme["q2"],
            "q0": scheme["q0"],
            "margin2_GPR3mm": scheme["margins"]["class2"]["GPR3mm"],
            "margin2_GPR2mm": scheme["margins"]["class2"]["GPR2mm"],
            "margin0_GPR3mm": scheme["margins"]["class0"]["GPR3mm"],
            "margin0_GPR2mm": scheme["margins"]["class0"]["GPR2mm"],
            **scheme["metrics"],
        }
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).round(4)
    comparison.to_csv(OUTPUT_DIR / "metrics_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n=== Before/after margin comparison ===")
    for scheme in schemes:
        print(f"  {scheme['label']}")
        print(f"    {format_metrics(scheme['metrics'])}")

    # ---- Recommendation: safest scheme first, then classification quality ----
    def is_cross_free(scheme: dict) -> bool:
        return (
            scheme["metrics"]["cross_true0_pred2"] == 0
            and scheme["metrics"]["cross_true2_pred0"] == 0
        )

    recommended = max(
        (s for s in schemes if is_cross_free(s)),
        key=lambda s: (s["metrics"]["score"], s["metrics"]["overall_accuracy"]),
    )
    print(f"\nRecommended scheme: {recommended['label']}")
    for scheme in schemes:
        if not is_cross_free(scheme):
            print(
                f"  {scheme['label']} violates the safety principle: "
                f"{scheme['metrics']['cross_true0_pred2']} true-0 point(s) auto-passed, "
                f"{scheme['metrics']['cross_true2_pred0']} true-2 point(s) sent to replan."
            )
    if recommended is not tuned_scheme:
        print(
            "  Note: the margin-0 scheme already matches or beats the tuned margins; "
            "the residual margins add nothing on this data."
        )

    # ---- Plots for the tuned/recommended scheme ----
    save_confusion(
        tuned_scheme["counts"], "tuned_confusion_matrix",
        "Chosen combo with residual-quantile margins",
    )
    plot_residuals(
        {t: scan_data[t][tuned_configs[t]]["residuals"] for t in CLS_TARGETS},
        tuned_margins, q2, q0,
    )
    plot_scatter(matrices, recommended["configs"], measured, labels, recommended["margins"])
    plot_decision_plane(matrices, recommended["configs"], labels, recommended["margins"])

    # ---- operating_point.json: the recommended scheme (step 9 + deployment scripts) ----
    rec_margins = recommended["margins"]
    operating_point = {
        "created": date.today().isoformat(),
        "combo": {
            target: {
                "model": recommended["configs"][target][0],
                "n_features": recommended["configs"][target][1],
                "cv_mae": cv_mae[(target, *recommended["configs"][target])],
            }
            for target in CLS_TARGETS
        },
        "thresholds": THRESHOLDS,
        "margin_method": {
            "type": "residual_quantile",
            "q2": recommended["q2"],
            "q0": recommended["q0"],
            "allow_negative_margins": True,
            "feasibility": "zero class-0 <-> class-2 cross-misclassifications",
            "objective": "max F1(class 2) + F1(class 0) on pooled out-of-fold points",
        },
        "margins_pp": rec_margins,
        "decision_thresholds": {
            "class2": {
                t: THRESHOLDS["class2"][t] + rec_margins["class2"][t] for t in CLS_TARGETS
            },
            "class0": {
                t: THRESHOLDS["class0"][t] - rec_margins["class0"][t] for t in CLS_TARGETS
            },
        },
        "n_points": n_points,
        "recommended_scheme": recommended["name"],
        "classification_metrics": {
            scheme["name"]: {
                "configs": {
                    t: {"model": scheme["configs"][t][0], "n_features": scheme["configs"][t][1]}
                    for t in CLS_TARGETS
                },
                "q2": scheme["q2"],
                "q0": scheme["q0"],
                "margins_pp": scheme["margins"],
                **scheme["metrics"],
            }
            for scheme in schemes
        },
    }
    with (OUTPUT_DIR / "operating_point.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(operating_point), handle, indent=2, ensure_ascii=False)

    metrics = recommended["metrics"]
    print(
        f"\nRecommended operating point: auto-pass {100 * metrics['autopass_rate']:.1f}% "
        f"(per-repeat {100 * metrics['autopass_repeat_mean']:.1f}"
        f"+/-{100 * metrics['autopass_repeat_std']:.1f}%), "
        f"replan {100 * metrics['replan_rate']:.1f}%, "
        f"manual review {100 * metrics['manual_rate']:.1f}% of {n_points} points."
    )
    print(
        "Statistical caution: precision(2) "
        f"{metrics['precision_2']:.3f} (95% CP lower bound "
        f"{metrics['precision2_ci95_lower']:.3f}) and precision(0) "
        + (
            f"{metrics['precision_0']:.3f} (lower bound {metrics['precision0_ci95_lower']:.3f})"
            if math.isfinite(metrics["precision_0"]) else "undefined (no replan calls)"
        )
        + " rest on pooled out-of-fold points; the margins were tuned on the same "
        "points, so quote these numbers with that caveat."
    )
    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
