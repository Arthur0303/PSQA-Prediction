"""Per-site error analysis of the out-of-fold GPR predictions (post-hoc).

The modeling pipeline (steps 4-9) deliberately excludes treatment Site from the
feature set: only plan-complexity metrics predict the gamma pass rates. This
script asks whether Site nonetheless carries predictive information the
complexity metrics miss. A model that never sees Site should have residuals
that are unrelated to Site; if instead one site's GPR is systematically over-
or under-predicted, or a site's error is much larger, that is direct evidence
of site-specific signal left on the table. No retraining happens here - the
analysis reads the step-6 out-of-fold predictions already on disk.

Raw RadCalc Site has 25 categories, most with only 1-5 rows, so they are first
merged into 10 clinical groups plus Other (SITE_GROUPS below). Only groups with
at least MIN_ROWS_FOR_ANALYSIS rows get per-site metrics, plots and the test;
every raw -> group mapping is still recorded in site_mapping.csv.

Predictions used (deployment configurations, one per target):
  - GPR3mm, GPR2mm: the combo written to 8_operating_point/operating_point.json
  - GPR1mm: the lowest mean-CV-MAE candidate (Dummy and the full-feature set
    excluded, mirroring 8_operating_point.py)
Each row's 5 RepeatedKFold out-of-fold predictions are averaged to one value, so
a row is one observation; repeated measurements of the same plan stay separate
rows, as elsewhere in the pipeline.

Inputs:
  - Model/1_remove_missing_rows.csv           (raw Site text, same row order)
  - Model/4_kept_features_targets.csv         (measured GPR -> true labels)
  - Model/6_model_training/predictions.csv    (out-of-fold predictions)
  - Model/6_model_training/fold_results.csv   (per-fold MAE -> GPR1mm config)
  - Model/8_operating_point/operating_point.json (deployed GPR3mm/GPR2mm combo)

Outputs (under Model/10_site_error_analysis/):
  - site_mapping.csv        (raw Site -> group, raw/group counts, analyzed flag)
  - site_error_metrics.csv  (per group x target: n, MAE, RMSE, bias, sd, se)
  - site_kruskal.csv        (per target: Kruskal-Wallis H, p, dof, groups, n)
  - site_bias.pdf           (signed bias per analyzed site, per target)
  - site_mae.pdf            (MAE per analyzed site, per target)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-GUI backend: save figures directly.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_SITE_FILE = SCRIPT_DIR / "1_remove_missing_rows.csv"
KEPT_FILE = SCRIPT_DIR / "4_kept_features_targets.csv"
PREDICTIONS_FILE = SCRIPT_DIR / "6_model_training" / "predictions.csv"
FOLD_RESULTS_FILE = SCRIPT_DIR / "6_model_training" / "fold_results.csv"
OPERATING_POINT_FILE = SCRIPT_DIR / "8_operating_point" / "operating_point.json"
OUTPUT_DIR = SCRIPT_DIR / "10_site_error_analysis"

TARGETS = ("GPR3mm", "GPR2mm", "GPR1mm")
DEPLOYED_TARGETS = ("GPR3mm", "GPR2mm")  # configs come from operating_point.json
EXCLUDED_MODELS = ("Dummy",)  # never a deployment candidate (same as step 8)
MIN_ROWS_FOR_ANALYSIS = 10
ALPHA = 0.05  # significance level for the Kruskal-Wallis test

# Clinical-proximity merge of the raw RadCalc Site labels into the 10 named
# groups plus Other. Row totals over the 124 cleaned rows are in the comments.
SITE_GROUPS = {
    "PROSTATE": "Prostate", "PROSNODE": "Prostate",
    "PROSBED": "Prostate", "PROSBEDNODE": "Prostate",          # Prostate   (32)
    "PELVIS": "Pelvis", "PELVICNODE": "Pelvis",
    "ENDO": "Pelvis", "VULVA": "Pelvis",                       # Pelvis     (19)
    "BREAST": "Breast", "BREASTNODE": "Breast", "BREASTCW": "Breast",
    "BREASTCWNODE": "Breast", "BREASTSCF": "Breast", "LTCWLN": "Breast",  # Breast (18)
    "HEADNECK": "HeadNeck",                                    # HeadNeck   (15)
    "RECTUM": "Rectum", "RECTUMNODE": "Rectum",               # Rectum     (10)
    "ANUSNODE": "Anus",                                       # Anus        (8)
    "BRAIN": "Brain",                                         # Brain       (7)
    "OESOPH": "Oesophagus",                                   # Oesophagus  (4)
    "LUNG": "Lung", "CHEST": "Lung",                          # Lung        (4)
    "BLADDER": "Bladder",                                     # Bladder     (3)
    "KIDNEY": "Other", "SPINE": "Other",                      # Other       (4)
}

# Figure chrome/palette, matching 8_operating_point.py for cross-figure
# consistency. Bias sign uses a diverging pair; MAE a single neutral hue.
INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
OVER_COLOR = "#2a78d6"   # bias > 0: model over-predicts GPR (optimistic)
UNDER_COLOR = "#d03b3b"  # bias < 0: model under-predicts GPR (pessimistic)
MAE_COLOR = "#457b9d"
PLOT_DPI = 300


def load_row_sites() -> tuple[list[str], list[str]]:
    """Return (raw Site per row, merged group per row), in file order."""
    if not RAW_SITE_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {RAW_SITE_FILE}")
    df = pd.read_csv(RAW_SITE_FILE, encoding="utf-8-sig")
    if "Site" not in df.columns:
        raise ValueError(f"{RAW_SITE_FILE.name} is missing the 'Site' column.")
    raw = df["Site"].astype(str).str.strip()
    unknown = sorted(set(raw) - set(SITE_GROUPS))
    if unknown:
        raise ValueError(
            f"Unmapped Site categories in {RAW_SITE_FILE.name}: {', '.join(unknown)}. "
            "Add them to SITE_GROUPS before re-running."
        )
    return raw.tolist(), raw.map(SITE_GROUPS).tolist()


def full_feature_count(kept_df: pd.DataFrame) -> int:
    """Number of feature columns in the kept file (targets excluded)."""
    return kept_df.shape[1] - sum(t in kept_df.columns for t in TARGETS)


def chosen_configs(kept_df: pd.DataFrame) -> dict[str, tuple[str, int]]:
    """Deployment config per target: GPR3mm/GPR2mm from operating_point.json,
    GPR1mm from the lowest mean-CV-MAE candidate (Dummy and full-feature set
    excluded, matching 8_operating_point.py)."""
    if not OPERATING_POINT_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {OPERATING_POINT_FILE}. Run 8_operating_point.py first."
        )
    with OPERATING_POINT_FILE.open(encoding="utf-8") as handle:
        combo = json.load(handle)["combo"]
    configs = {t: (str(combo[t]["model"]), int(combo[t]["n_features"])) for t in DEPLOYED_TARGETS}

    fold_df = pd.read_csv(FOLD_RESULTS_FILE, encoding="utf-8-sig")
    n_full = full_feature_count(kept_df)
    candidates = fold_df[
        (fold_df["target"] == "GPR1mm")
        & ~fold_df["model"].isin(EXCLUDED_MODELS)
        & (fold_df["n_features"] != n_full)
    ]
    mean_mae = candidates.groupby(["model", "n_features"])["mae"].mean()
    best_model, best_n = mean_mae.idxmin()
    configs["GPR1mm"] = (str(best_model), int(best_n))
    return configs


def per_row_predictions(
    pred_df: pd.DataFrame, target: str, model: str, n_features: int, n_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean out-of-fold prediction and true GPR per row for one config.

    Averages the RepeatedKFold repeats so each row is a single observation.
    """
    sub = pred_df[
        (pred_df["target"] == target)
        & (pred_df["model"] == model)
        & (pred_df["n_features"] == n_features)
    ]
    if sub.empty:
        raise ValueError(f"No predictions for {target} {model} n={n_features} in predictions.csv.")
    wide = sub.pivot(index="row_index", columns="repeat", values="y_pred").sort_index()
    if wide.shape[0] != n_rows or wide.isna().any().any():
        raise ValueError(
            f"[{target}] {model} n={n_features}: expected one prediction per row per "
            f"repeat over {n_rows} rows, got shape {wide.shape} with gaps."
        )
    if not np.array_equal(wide.index.to_numpy(), np.arange(n_rows)):
        raise ValueError(f"[{target}] {model} n={n_features}: row_index is not 0..{n_rows - 1}.")
    mean_pred = wide.mean(axis=1).to_numpy(float)
    truth = sub.groupby("row_index")["y_true"].first().sort_index().to_numpy(float)
    return mean_pred, truth


def site_metric_rows(target: str, groups: np.ndarray, residual: np.ndarray) -> list[dict]:
    """Per merged group: n, MAE, RMSE, signed bias and its sd/se for one target."""
    rows = []
    frame = pd.DataFrame({"site": groups, "e": residual})
    for site, gdf in frame.groupby("site"):
        e = gdf["e"].to_numpy(float)
        n = e.size
        sd = float(np.std(e, ddof=1)) if n > 1 else float("nan")
        rows.append(
            {
                "target": target,
                "site": site,
                "n_rows": n,
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "bias": float(np.mean(e)),
                "bias_sd": sd,
                "bias_se": sd / np.sqrt(n) if n > 1 else float("nan"),
                "analyzed": bool(n >= MIN_ROWS_FOR_ANALYSIS),
            }
        )
    return rows


def kruskal_row(target: str, groups: np.ndarray, residual: np.ndarray, sites: list[str]) -> dict:
    """Kruskal-Wallis test of residual location across the analyzed sites."""
    samples = [residual[groups == s] for s in sites]
    stat, p_value = kruskal(*samples)
    return {
        "target": target,
        "h_statistic": float(stat),
        "p_value": float(p_value),
        "dof": len(sites) - 1,
        "n_groups": len(sites),
        "n_rows": int(sum(s.size for s in samples)),
        "groups": ", ".join(sites),
        "significant": bool(p_value < ALPHA),
    }


def write_site_mapping(raw_sites: list[str], groups: list[str]) -> None:
    raw_counts = pd.Series(raw_sites).value_counts()
    group_counts = pd.Series(groups).value_counts()
    rows = [
        {
            "raw_site": raw,
            "group": SITE_GROUPS[raw],
            "raw_count": int(raw_counts[raw]),
            "group_count": int(group_counts[SITE_GROUPS[raw]]),
            "analyzed": bool(group_counts[SITE_GROUPS[raw]] >= MIN_ROWS_FOR_ANALYSIS),
        }
        for raw in raw_counts.index
    ]
    mapping = pd.DataFrame(rows).sort_values(
        ["group_count", "group", "raw_count"], ascending=[False, True, False]
    )
    mapping.to_csv(OUTPUT_DIR / "site_mapping.csv", index=False, encoding="utf-8-sig")


def style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: hairline y-grid, light baseline, muted tick labels."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor="#52514e")


def plot_bias(metrics: pd.DataFrame, site_order: list[str], kruskal_df: pd.DataFrame) -> None:
    """Signed bias per analyzed site (shared site order), one panel per target."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(14, 4.6), sharex=True)
    for ax, target in zip(axes, TARGETS):
        sub = metrics[(metrics["target"] == target) & metrics["analyzed"]].set_index("site")
        sub = sub.reindex(site_order)
        colors = [OVER_COLOR if b >= 0 else UNDER_COLOR for b in sub["bias"]]
        ax.bar(
            range(len(site_order)), sub["bias"], yerr=sub["bias_se"],
            color=colors, capsize=3, error_kw={"ecolor": MUTED, "elinewidth": 1},
        )
        ax.axhline(0, color=BASELINE, linewidth=1)
        style_axes(ax)
        ax.set_xticks(range(len(site_order)), site_order, rotation=30, ha="right")
        p_value = kruskal_df.loc[kruskal_df["target"] == target, "p_value"].iloc[0]
        ax.set_title(f"{target}\nKruskal-Wallis p = {p_value:.3f}", color=INK, fontsize=11)
        ax.set_ylabel("Mean bias = pred - true (pp)", color="#52514e")
    fig.suptitle(
        "Per-site prediction bias (blue = GPR over-predicted, red = under-predicted; "
        "error bars +/-1 SE)",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "site_bias.pdf", dpi=PLOT_DPI)
    plt.close(fig)


def plot_mae(metrics: pd.DataFrame) -> None:
    """MAE per analyzed site, sorted worst -> best within each target panel."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(14, 4.6))
    for ax, target in zip(axes, TARGETS):
        sub = metrics[(metrics["target"] == target) & metrics["analyzed"]]
        sub = sub.sort_values("mae", ascending=False)
        ax.bar(range(len(sub)), sub["mae"], color=MAE_COLOR)
        style_axes(ax)
        ax.set_xticks(range(len(sub)), sub["site"], rotation=30, ha="right")
        ax.set_title(target, color=INK, fontsize=11)
        ax.set_ylabel("MAE (pp)", color="#52514e")
    fig.suptitle("Per-site mean absolute error (worst to best)", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "site_mae.pdf", dpi=PLOT_DPI)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_sites, group_list = load_row_sites()
    groups = np.array(group_list)
    n_rows = len(raw_sites)

    kept_df = pd.read_csv(KEPT_FILE, encoding="utf-8-sig")
    if len(kept_df) != n_rows:
        raise ValueError(
            f"Row count mismatch: {RAW_SITE_FILE.name} has {n_rows} rows but "
            f"{KEPT_FILE.name} has {len(kept_df)}. Re-run the pipeline from step 1."
        )

    pred_df = pd.read_csv(PREDICTIONS_FILE, encoding="utf-8-sig")
    needed = {"target", "model", "n_features", "repeat", "row_index", "y_true", "y_pred"}
    if missing := needed - set(pred_df.columns):
        raise ValueError(f"predictions.csv is missing columns: {', '.join(sorted(missing))}")
    pred_rows = int(pred_df["row_index"].max()) + 1
    if pred_rows != n_rows:
        raise ValueError(
            f"predictions.csv covers {pred_rows} rows but {RAW_SITE_FILE.name} has "
            f"{n_rows}. Re-run steps 4 and 6 from the same data."
        )

    configs = chosen_configs(kept_df)
    print("Chosen deployment configurations (one per target):")
    for target in TARGETS:
        model, n_feat = configs[target]
        origin = "operating_point.json" if target in DEPLOYED_TARGETS else "lowest CV MAE (GPR1mm)"
        print(f"  {target}: {model} (n={n_feat})  [{origin}]")

    analyzed_sites = sorted(
        {g for g in group_list if group_list.count(g) >= MIN_ROWS_FOR_ANALYSIS},
        key=lambda s: -group_list.count(s),
    )
    print(
        f"\nAnalyzed sites (n >= {MIN_ROWS_FOR_ANALYSIS}): "
        + ", ".join(f"{s} ({group_list.count(s)})" for s in analyzed_sites)
    )

    write_site_mapping(raw_sites, group_list)

    metric_rows: list[dict] = []
    kruskal_rows: list[dict] = []
    for target in TARGETS:
        model, n_feat = configs[target]
        mean_pred, truth = per_row_predictions(pred_df, target, model, n_feat, n_rows)
        if not np.allclose(truth, kept_df[target].to_numpy(float)):
            raise ValueError(
                f"[{target}] y_true in predictions.csv does not match {KEPT_FILE.name}. "
                "Re-run steps 4 and 6 from the same data."
            )
        residual = mean_pred - truth  # + = model over-predicts GPR
        metric_rows.extend(site_metric_rows(target, groups, residual))
        kruskal_rows.append(kruskal_row(target, groups, residual, analyzed_sites))

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["target", "n_rows"], ascending=[True, False]
    )
    metrics.round(4).to_csv(OUTPUT_DIR / "site_error_metrics.csv", index=False, encoding="utf-8-sig")
    kruskal_df = pd.DataFrame(kruskal_rows)
    kruskal_df.round(6).to_csv(OUTPUT_DIR / "site_kruskal.csv", index=False, encoding="utf-8-sig")

    plot_bias(metrics, analyzed_sites, kruskal_df)
    plot_mae(metrics)

    # ---- Console summary per target ----
    print("\n=== Per-site error (analyzed sites only) ===")
    for target in TARGETS:
        sub = metrics[(metrics["target"] == target) & metrics["analyzed"]]
        worst_mae = sub.loc[sub["mae"].idxmax()]
        best_mae = sub.loc[sub["mae"].idxmin()]
        most_over = sub.loc[sub["bias"].idxmax()]
        most_under = sub.loc[sub["bias"].idxmin()]
        kw = kruskal_df.loc[kruskal_df["target"] == target].iloc[0]
        verdict = (
            "Site adds signal beyond the complexity metrics"
            if kw["significant"]
            else "no significant residual difference across sites"
        )
        print(f"\n{target}:")
        print(f"  Highest MAE: {worst_mae['site']} ({worst_mae['mae']:.2f} pp); "
              f"lowest: {best_mae['site']} ({best_mae['mae']:.2f} pp)")
        print(f"  Most over-predicted: {most_over['site']} (bias {most_over['bias']:+.2f} pp); "
              f"most under-predicted: {most_under['site']} (bias {most_under['bias']:+.2f} pp)")
        print(f"  Kruskal-Wallis H={kw['h_statistic']:.2f}, p={kw['p_value']:.3f} "
              f"(dof={kw['dof']}) -> {verdict}")

    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
