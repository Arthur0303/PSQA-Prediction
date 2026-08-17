"""ROC discrimination across quantile levels for the triage rule of step 8.

Step 8 evaluates the clinical triage rule at a single operating point: the
residual-quantile margins (q2, q0) are fixed and one confusion matrix is
reported. That leaves an obvious question unanswered - is the modest overall
accuracy a property of the models, or of the deliberately conservative
quantile choice? An ROC curve separates discrimination from that single choice
of (q2, q0). It does not, however, remove the optimism from selecting the model
pair or estimating the residual distributions on the same pooled out-of-fold
predictions; it also inherits the feature-selection optimism quantified by
step 9.

For discrimination analysis, the two extreme decisions are evaluated as
separate one-vs-rest detectors: "is this plan auto-pass?" and "is this plan a
replan candidate?" The deployed operating point enforces ZERO class-0 <->
class-2 cross-misclassifications, so every observed error at that point is a
one-step confusion with manual review. This one-vs-rest treatment of a
three-class PSQA triage scheme follows the same route as published three-class
QA models.

The score
---------
The triage rule is a box in the (predicted GPR3mm, predicted GPR2mm) plane, not
a scalar, so a score has to be constructed before any ROC exists. The one
already implicit in step 8 is the residual-quantile level itself. For the
linear empirical quantile Q used by numpy, define I(z) as the largest q for
which Q_q <= z (or Q_q < z for the strict replan inequality). With pooled
out-of-fold residuals e_t = pred_t - true_t:

    u = min_t I_{e_t}( pred_t - U_t )        U = (99, 97)   auto-pass score
    v = min_t I_{-e_t}( L_t - pred_t )       L = (98, 92)   replan score

Because the margins m2 = Q_q2(e) and m0 = Q_q0(-e) are monotone in q, the two
families of decision boxes are nested, and box membership at level q is exactly
u >= q2 and v > q0. Sweeping q therefore traces the ROC curve of each binary box
detector, in which each target is rescaled by its own residual distribution
rather than by an arbitrary constant. The deployed operating point lies on the
curve by construction; this is asserted, not assumed.

Uncertainty
-----------
Three views, because 620 pooled points come from only 124 measurements of 119
distinct plans:
  - pooled AUC over all row x repeat points (comparable with step 8's metrics)
  - per-repeat mean +/- SD over the 5 RepeatedKFold repeats
  - cluster bootstrap 95% percentile CI, resampling the 119 distinct plans.
    Repeated measurements of the same plan share an identical feature vector
    and differ only in GPR (see 4_feature_importance.py), so identical feature
    rows identify the clusters. Resampling plans rather than points respects
    the effective sample size: the 15 class-0 measurements come from 12 plans.

Inputs (produced by 4_feature_importance.py / 6_model_training.py / 8_operating_point.py):
  - Model/4_kept_features_targets.csv        (measured GPR -> true labels, plan clusters)
  - Model/6_model_training/predictions.csv   (out-of-fold predictions per config)
  - Model/8_operating_point/operating_point.json  (deployed combo and thresholds)

Outputs (under Model/11_roc_analysis/):
  - roc_curves.pdf   two one-vs-rest ROC curves with the deployed operating point
  - roc_auc.csv      AUC table (pooled, per-repeat, bootstrap CI) for both schemes
  - roc_points.csv   curve coordinates of the deployed scheme, so the figure is
                     reproducible without re-running this script

Nothing written by earlier steps is modified; this script only reads them.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend and save plots directly.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


SCRIPT_DIR = Path(__file__).resolve().parent
KEPT_FILE = SCRIPT_DIR / "4_kept_features_targets.csv"
PREDICTIONS_FILE = SCRIPT_DIR / "6_model_training" / "predictions.csv"
OPERATING_POINT_FILE = SCRIPT_DIR / "8_operating_point" / "operating_point.json"
OUTPUT_DIR = SCRIPT_DIR / "11_roc_analysis"

ALL_TARGETS = ("GPR3mm", "GPR2mm", "GPR1mm")
CLS_TARGETS = ("GPR3mm", "GPR2mm")  # GPR1mm is not part of the clinical rule
# Same raw clinical thresholds as 8_operating_point.py; kept here so this script
# can be read on its own, and cross-checked against the JSON at run time.
THRESHOLDS = {
    "class2": {"GPR3mm": 99.0, "GPR2mm": 97.0},
    "class0": {"GPR3mm": 98.0, "GPR2mm": 92.0},
}
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
CI_PERCENTILES = (2.5, 97.5)
PLOT_DPI = 300
# Same grid as step 8. It is repeated here so score/box equivalence can be
# asserted for every quantile level used to choose the operating point.
MARGIN_Q_GRID = np.r_[np.arange(0.05, 1.0, 0.05), 0.975, 0.99]

INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
# Same status palette as 8_operating_point.py: class 2 is a pass state, class 0
# a failure state, so the two curves keep the colors their classes have in the
# confusion matrix and decision-plane figures.
LABEL_STYLE = {
    2: {"color": "#0ca30c", "marker": "o", "name": "2 auto-pass"},
    0: {"color": "#d03b3b", "marker": "X", "name": "0 replan"},
}


def load_true_labels() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Return (label per row, measured GPR frame, plan-cluster id per row).

    The label rule is the one in 8_operating_point.py. Cluster ids group rows
    with an identical feature vector, i.e. repeated measurements of one plan.
    """
    if not KEPT_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {KEPT_FILE}")
    df = pd.read_csv(KEPT_FILE, encoding="utf-8-sig")
    g3 = df["GPR3mm"].to_numpy(float)
    g2 = df["GPR2mm"].to_numpy(float)
    labels = np.ones(len(df), dtype=int)
    labels[(g3 >= THRESHOLDS["class2"]["GPR3mm"]) & (g2 >= THRESHOLDS["class2"]["GPR2mm"])] = 2
    labels[(g3 < THRESHOLDS["class0"]["GPR3mm"]) & (g2 < THRESHOLDS["class0"]["GPR2mm"])] = 0

    features = df.drop(columns=[c for c in ALL_TARGETS if c in df.columns])
    clusters = features.groupby(list(features.columns), dropna=False).ngroup().to_numpy()
    # A plan cannot belong to two classes; if it did, the cluster bootstrap
    # would be resampling something other than plans.
    mixed = pd.DataFrame({"cluster": clusters, "label": labels}).groupby("cluster")["label"].nunique()
    if (mixed > 1).any():
        raise ValueError("Some feature-identical rows carry different labels; clusters are not plans.")

    counts = pd.Series(labels).value_counts().sort_index()
    print(f"True labels from {KEPT_FILE.name}: " + ", ".join(f"{c}->{n}" for c, n in counts.items()))
    print(f"{len(df)} rows in {len(np.unique(clusters))} distinct plans (feature-identical rows grouped).")
    return labels, df[list(CLS_TARGETS)], clusters


def load_operating_point() -> dict:
    """Return the step-8 operating point, cross-checked against THRESHOLDS."""
    if not OPERATING_POINT_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {OPERATING_POINT_FILE}. Run 8_operating_point.py first."
        )
    with OPERATING_POINT_FILE.open(encoding="utf-8") as handle:
        operating_point = json.load(handle)
    for side in ("class2", "class0"):
        for target in CLS_TARGETS:
            if operating_point["thresholds"][side][target] != THRESHOLDS[side][target]:
                raise ValueError(
                    f"THRESHOLDS[{side}][{target}] disagrees with {OPERATING_POINT_FILE.name}."
                )
    return operating_point


def prediction_matrix(
    predictions: pd.DataFrame, target: str, model: str, n_features: int, measured: pd.DataFrame
) -> np.ndarray:
    """Out-of-fold predictions of one configuration as n_rows x n_repeats."""
    group = predictions[
        (predictions["target"] == target)
        & (predictions["model"] == model)
        & (predictions["n_features"] == n_features)
    ]
    if group.empty:
        raise ValueError(f"[{target}] {model} n={n_features}: no rows in {PREDICTIONS_FILE.name}.")
    wide = group.pivot(index="row_index", columns="repeat", values="y_pred").sort_index()
    n_rows = len(measured)
    if wide.shape[0] != n_rows or wide.isna().any().any():
        raise ValueError(
            f"[{target}] {model} n={n_features}: expected one prediction per row per "
            f"repeat ({n_rows} rows), got shape {wide.shape}."
        )
    if not np.array_equal(wide.index.to_numpy(), np.arange(n_rows)):
        raise ValueError(f"[{target}] {model} n={n_features}: row_index is not 0..{n_rows - 1}.")
    truth = group.groupby("row_index")["y_true"].first().sort_index().to_numpy(float)
    if not np.allclose(truth, measured[target].to_numpy(float)):
        raise ValueError(
            f"[{target}] {model} n={n_features}: y_true in {PREDICTIONS_FILE.name} does not "
            f"match {KEPT_FILE.name}. Re-run steps 4 and 6 from the same data."
        )
    return wide.to_numpy(float)


def quantile_level(
    sample: np.ndarray, values: np.ndarray, *, strict: bool
) -> np.ndarray:
    """Invert numpy's default linear empirical quantile.

    With ``strict=False``, return the supremum of q such that
    ``np.quantile(sample, q) <= value``. With ``strict=True``, use ``<``.
    The separate boundary conventions reproduce the ``>=`` auto-pass and
    ``<`` replan inequalities in step 8, including ties and interpolation
    between adjacent order statistics.
    """
    ordered = np.sort(np.asarray(sample, dtype=float))
    values = np.asarray(values, dtype=float)
    if ordered.size < 2:
        raise ValueError("At least two residuals are required to invert a quantile.")

    levels = np.empty(values.shape, dtype=float)
    if strict:
        below = values <= ordered[0]
        above = values > ordered[-1]
        side = "left"
        levels[below] = 0.0
        levels[above] = np.nextafter(1.0, np.inf)
    else:
        below = values < ordered[0]
        above = values >= ordered[-1]
        side = "right"
        levels[below] = np.nextafter(0.0, -np.inf)
        levels[above] = 1.0

    interior = ~(below | above)
    interior_values = values[interior]
    left_index = np.searchsorted(ordered, interior_values, side=side) - 1
    left_value = ordered[left_index]
    right_value = ordered[left_index + 1]
    fraction = (interior_values - left_value) / (right_value - left_value)
    levels[interior] = (left_index + fraction) / (ordered.size - 1)
    return levels


def triage_scores(
    matrices: dict[str, np.ndarray], measured: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return (auto-pass score u, replan score v), both n_rows x n_repeats.

    u is the largest quantile level at which the point still sits inside the
    class-2 box, v the same for the class-0 box. Larger means more confidently
    that class.
    """
    residuals = {
        t: (matrices[t] - measured[t].to_numpy(float)[:, None]).ravel()
        for t in CLS_TARGETS
    }
    u = np.minimum.reduce(
        [
            quantile_level(
                residuals[t], matrices[t] - THRESHOLDS["class2"][t], strict=False
            )
            for t in CLS_TARGETS
        ]
    )
    v = np.minimum.reduce(
        [
            quantile_level(
                -residuals[t], THRESHOLDS["class0"][t] - matrices[t], strict=True
            )
            for t in CLS_TARGETS
        ]
    )
    return u, v


def check_score_equivalence(
    matrices: dict[str, np.ndarray],
    scores: tuple[np.ndarray, np.ndarray],
    measured: pd.DataFrame,
) -> None:
    """Assert that score thresholds reproduce every step-8 quantile box."""
    residuals = {
        t: (matrices[t] - measured[t].to_numpy(float)[:, None]).ravel()
        for t in CLS_TARGETS
    }
    u, v = scores
    for q in MARGIN_Q_GRID:
        box2 = np.logical_and.reduce(
            [
                matrices[t]
                >= THRESHOLDS["class2"][t]
                + np.quantile(residuals[t], q, method="linear")
                for t in CLS_TARGETS
            ]
        )
        box0 = np.logical_and.reduce(
            [
                matrices[t]
                < THRESHOLDS["class0"][t]
                - np.quantile(-residuals[t], q, method="linear")
                for t in CLS_TARGETS
            ]
        )
        if not np.array_equal(u >= q, box2):
            raise AssertionError(f"auto-pass score does not reproduce the q={q:g} box.")
        if not np.array_equal(v > q, box0):
            raise AssertionError(f"replan score does not reproduce the q={q:g} box.")


def comparison_mask(labels: np.ndarray, positive: int, negatives: tuple[int, ...]) -> np.ndarray:
    """Row mask selecting the positive class plus the chosen negative classes."""
    return np.isin(labels, (positive,) + negatives)


def bootstrap_ci(
    score: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    positive: int,
    negatives: tuple[int, ...],
) -> tuple[float, float]:
    """Cluster-bootstrap percentile CI for the AUC, resampling whole plans."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique_clusters = np.unique(clusters)
    rows_of = {c: np.flatnonzero(clusters == c) for c in unique_clusters}
    n_repeats = score.shape[1]
    values = []
    for _ in range(N_BOOTSTRAP):
        drawn = rng.choice(unique_clusters, size=unique_clusters.size, replace=True)
        rows = np.concatenate([rows_of[c] for c in drawn])
        keep = comparison_mask(labels[rows], positive, negatives)
        rows = rows[keep]
        y = (labels[rows] == positive).astype(int)
        if y.sum() == 0 or y.sum() == y.size:
            continue  # a resample without both classes carries no information
        values.append(roc_auc_score(np.repeat(y, n_repeats), score[rows].ravel()))
    if not values:
        return float("nan"), float("nan")
    lower, upper = np.percentile(values, CI_PERCENTILES)
    return float(lower), float(upper)


def auc_summary(
    score: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    positive: int,
    negatives: tuple[int, ...],
) -> dict[str, float]:
    """Pooled AUC, per-repeat mean/SD and cluster-bootstrap CI for one contrast."""
    rows = comparison_mask(labels, positive, negatives)
    y = (labels[rows] == positive).astype(int)
    selected = score[rows]
    per_repeat = [roc_auc_score(y, selected[:, r]) for r in range(selected.shape[1])]
    lower, upper = bootstrap_ci(score, labels, clusters, positive, negatives)
    return {
        "n_pos_rows": int(y.sum()),
        "n_neg_rows": int(y.size - y.sum()),
        "n_points": int(selected.size),
        "auc_pooled": float(roc_auc_score(np.repeat(y, selected.shape[1]), selected.ravel())),
        "auc_repeat_mean": float(np.mean(per_repeat)),
        "auc_repeat_std": float(np.std(per_repeat, ddof=1)),
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def pooled_curve(
    score: np.ndarray, labels: np.ndarray, positive: int, negatives: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, threshold) of the pooled row x repeat ROC curve."""
    rows = comparison_mask(labels, positive, negatives)
    y = np.repeat((labels[rows] == positive).astype(int), score.shape[1])
    return roc_curve(y, score[rows].ravel())


def deployed_rates(
    mask: np.ndarray, labels: np.ndarray, positive: int
) -> tuple[float, float]:
    """(fpr, tpr) of a fixed decision mask against the one-vs-rest labels."""
    truth = np.broadcast_to(labels[:, None], mask.shape)
    tpr = float((mask & (truth == positive)).sum() / (truth == positive).sum())
    fpr = float((mask & (truth != positive)).sum() / (truth != positive).sum())
    return fpr, tpr


def check_operating_point(
    matrices: dict[str, np.ndarray],
    scores: tuple[np.ndarray, np.ndarray],
    operating_point: dict,
    labels: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Reproduce the deployed classifier and verify it lies on the ROC curves.

    Three checks: the box rule reproduces step 8's recalls; the two boxes do not
    overlap, so each point belongs to at most one of them; and thresholding the
    scores at (q2, q0) selects exactly the same points as the box rule, which is
    what puts the marker on the curve rather than merely near it.
    """
    decision = operating_point["decision_thresholds"]
    box2 = np.logical_and.reduce([matrices[t] >= decision["class2"][t] for t in CLS_TARGETS])
    box0 = np.logical_and.reduce([matrices[t] < decision["class0"][t] for t in CLS_TARGETS])
    if (box2 & box0).any():
        raise AssertionError(
            "Some points fall inside both decision boxes; the marker would not be "
            "a single point on either curve."
        )

    metrics = operating_point["classification_metrics"][operating_point["recommended_scheme"]]
    u, v = scores
    q2 = float(operating_point["margin_method"]["q2"])
    q0 = float(operating_point["margin_method"]["q0"])
    marked = {}
    for positive, mask, score, level, strict in (
        (2, box2, u, q2, False),
        (0, box0, v, q0, True),
    ):
        recall_json = float(metrics[f"recall_{positive}"])
        fpr, tpr = deployed_rates(mask, labels, positive)
        if not np.isclose(tpr, recall_json, atol=5e-6):
            raise AssertionError(
                f"class {positive}: recomputed recall {tpr:.6f} does not match "
                f"{recall_json:.6f} in {OPERATING_POINT_FILE.name}."
            )
        by_score = (score > level) if strict else (score >= level)
        if not np.array_equal(by_score, mask):
            raise AssertionError(
                f"class {positive}: thresholding the score at q={level:g} selects "
                f"{int(by_score.sum())} points but the deployed box selects "
                f"{int(mask.sum())}; the operating point is not on the curve."
            )
        marked[positive] = {"q": level, "fpr": fpr, "tpr": tpr, "n_flagged": int(mask.sum())}
    return marked


def style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: hairline grid, light baseline, muted tick labels."""
    ax.set_axisbelow(True)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor="#52514e")


def plot_roc(
    scores: tuple[np.ndarray, np.ndarray],
    labels: np.ndarray,
    summaries: dict[int, dict[str, float]],
    marked: dict[int, dict[str, float]],
    configs: dict[str, tuple[str, int]],
) -> None:
    """The two one-vs-rest curves, with per-repeat spread and the deployed point."""
    panels = (
        (2, (1, 0), scores[0], "Auto-pass detection (class 2 vs rest)"),
        (0, (1, 2), scores[1], "Replan detection (class 0 vs rest)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (positive, negatives, score, title) in zip(axes, panels):
        color = LABEL_STYLE[positive]["color"]
        y = (labels == positive).astype(int)  # one-vs-rest: every row is used

        for repeat in range(score.shape[1]):
            fpr, tpr, _ = roc_curve(y, score[:, repeat])
            ax.plot(fpr, tpr, color=color, linewidth=0.8, alpha=0.28, zorder=1)
        fpr, tpr, _ = pooled_curve(score, labels, positive, negatives)
        ax.plot(fpr, tpr, color=color, linewidth=2.0, zorder=3)
        ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.0, linestyle="--", zorder=0)

        point = marked[positive]
        ax.scatter(
            [point["fpr"]], [point["tpr"]],
            s=150, marker="*", color=color, edgecolor="#ffffff", linewidths=0.8, zorder=4,
        )
        ax.annotate(
            f"deployed point (q = {point['q']:g})\n"
            f"TPR {point['tpr']:.3f}, FPR {point['fpr']:.3f}",
            (point["fpr"], point["tpr"]), xytext=(10, -26), textcoords="offset points",
            color="#52514e", fontsize=8.5,
        )
        summary = summaries[positive]
        ax.annotate(
            f"AUC {summary['auc_pooled']:.3f}\n"
            f"95% CI {summary['ci95_lower']:.3f}-{summary['ci95_upper']:.3f}",
            (0.97, 0.06), xycoords="axes fraction", ha="right",
            color=INK, fontsize=9.5,
        )

        style_axes(ax)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("False positive rate", color="#52514e")
        ax.set_ylabel("True positive rate", color="#52514e")

    model3, n3 = configs["GPR3mm"]
    model2, n2 = configs["GPR2mm"]
    fig.suptitle(
        f"Triage discrimination on the pooled out-of-fold points "
        f"(GPR3mm: {model3} n={n3}, GPR2mm: {model2} n={n2})",
        color=INK,
    )
    fig.text(
        0.5, 0.015,
        "Bold: pooled over 620 points. Faint: the five RepeatedKFold repeats. "
        "Dashed: chance. Star: the deployed operating point of the triage scheme.",
        ha="center", color="#52514e", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "roc_curves.pdf", dpi=PLOT_DPI)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels, measured, clusters = load_true_labels()
    operating_point = load_operating_point()
    predictions = pd.read_csv(PREDICTIONS_FILE, encoding="utf-8-sig")

    deployed_configs = {
        target: (
            str(operating_point["combo"][target]["model"]),
            int(operating_point["combo"][target]["n_features"]),
        )
        for target in CLS_TARGETS
    }
    baseline_block = operating_point["classification_metrics"]["baseline_lowest_mae_no_margin"]
    baseline_configs = {
        target: (
            str(baseline_block["configs"][target]["model"]),
            int(baseline_block["configs"][target]["n_features"]),
        )
        for target in CLS_TARGETS
    }
    schemes = [("deployed", deployed_configs), ("baseline_lowest_mae", baseline_configs)]

    # One-vs-rest is the headline; the adjacent-class contrasts are reported as
    # extra table rows because the zero cross-error structure means class 2 and
    # class 0 are only ever confused with class 1 in practice.
    contrasts = [
        ("class2_vs_rest", 2, (1, 0)),
        ("class0_vs_rest", 0, (1, 2)),
        ("class2_vs_class1", 2, (1,)),
        ("class0_vs_class1", 0, (1,)),
    ]

    rows = []
    deployed_summaries: dict[int, dict[str, float]] = {}
    deployed_marked: dict[int, dict[str, float]] = {}
    curve_frames = []

    for scheme_name, configs in schemes:
        matrices = {
            target: prediction_matrix(predictions, target, *configs[target], measured)
            for target in CLS_TARGETS
        }
        u, v = triage_scores(matrices, measured)
        check_score_equivalence(matrices, (u, v), measured)
        score_of = {2: u, 0: v}
        print(f"\n=== {scheme_name}: " + ", ".join(
            f"{t} {configs[t][0]}(n={configs[t][1]})" for t in CLS_TARGETS) + " ===")

        for name, positive, negatives in contrasts:
            summary = auc_summary(score_of[positive], labels, clusters, positive, negatives)
            rows.append({
                "scheme": scheme_name,
                "model_3mm": configs["GPR3mm"][0],
                "n_features_3mm": configs["GPR3mm"][1],
                "model_2mm": configs["GPR2mm"][0],
                "n_features_2mm": configs["GPR2mm"][1],
                "comparison": name,
                "positive_class": positive,
                "negative_classes": "+".join(str(c) for c in negatives),
                **summary,
            })
            print(
                f"  AUC {name:18s} pooled {summary['auc_pooled']:.4f} | "
                f"per-repeat {summary['auc_repeat_mean']:.4f} +/- {summary['auc_repeat_std']:.4f} | "
                f"95% CI [{summary['ci95_lower']:.3f}, {summary['ci95_upper']:.3f}]"
            )
            if scheme_name == "deployed" and name.endswith("vs_rest"):
                deployed_summaries[positive] = summary

        if scheme_name == "deployed":
            deployed_marked = check_operating_point(matrices, (u, v), operating_point, labels)
            print("  Operating point reproduced from operating_point.json and verified on curve:")
            for positive, point in deployed_marked.items():
                print(
                    f"    class {positive}: q={point['q']:g}, TPR {point['tpr']:.4f}, "
                    f"FPR {point['fpr']:.4f}, {point['n_flagged']} points flagged"
                )
            for name, positive, negatives in contrasts[:2]:
                fpr, tpr, thresholds = pooled_curve(score_of[positive], labels, positive, negatives)
                curve_frames.append(pd.DataFrame({
                    "comparison": name,
                    "fpr": fpr,
                    "tpr": tpr,
                    "threshold_q": thresholds,
                }))
            plot_roc((u, v), labels, deployed_summaries, deployed_marked, configs)

    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUTPUT_DIR / "roc_auc.csv", index=False, encoding="utf-8-sig")
    pd.concat(curve_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "roc_points.csv", index=False, encoding="utf-8-sig"
    )

    print(
        "\nThe curves do not depend on the selected q2 and q0, but the model pair and "
        "residual distributions were estimated from the same pooled out-of-fold predictions. "
        "Feature-selection optimism also applies. The bootstrap intervals rest on 119 plans; "
        "the 15 class-0 measurements come from 12 plans."
    )
    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
