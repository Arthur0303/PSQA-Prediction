"""Analyze feature importance for GPR3mm / GPR2mm / GPR1mm.

Input: Model/3_one_hot_encoding.csv
Outputs:
  - Model/4_kept_features_targets.csv
  - CSV files and PDF vector figures under Model/4_feature_importance/, grouped by analysis method.

Preprocessing assumptions:
  - 0_remove_columns.py has removed unused raw columns.
  - 1_remove_missing_rows.py has removed rows with missing values.
  - 3_one_hot_encoding.py has converted Site / Energy to one-hot columns.

Analysis methods (run separately for each of the three GPR targets):
  1. Spearman correlation + p-value
  2. Feature correlation heatmap (shared once) + high-correlation pair list
  3. Random Forest + permutation importance with CV R2/MAE
  4. LASSO / Elastic Net coefficients

Note: The data keeps repeated measurements from the same plan
(identical features with different GPR values).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend and save plots directly.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "3_one_hot_encoding.csv"
KEPT_FEATURES_FILE = SCRIPT_DIR / "4_kept_features_targets.csv"
OUTPUT_DIR = SCRIPT_DIR / "4_feature_importance"
PREPROCESSING_DIR = OUTPUT_DIR / "preprocessing"
FEATURE_CORRELATION_DIR = OUTPUT_DIR / "feature_correlation"
SPEARMAN_DIR = OUTPUT_DIR / "spearman"
RANDOM_FOREST_DIR = OUTPUT_DIR / "random_forest"
ELASTIC_NET_DIR = OUTPUT_DIR / "elastic_net_lasso"
OUTPUT_SUBDIRS = (
    PREPROCESSING_DIR,
    FEATURE_CORRELATION_DIR,
    SPEARMAN_DIR,
    RANDOM_FOREST_DIR,
    ELASTIC_NET_DIR,
)

TARGETS = ["GPR3mm", "GPR2mm", "GPR1mm"]
RANDOM_STATE = 42
RF_N_ESTIMATORS = 400
RF_CV_SPLITS = 5
RF_CV_REPEATS = 5
RF_PERMUTATION_REPEATS = 10
SUMMARY_TOP_N = 30
PLOT_DPI = 300
HIGH_CORR_THRESHOLD = 0.8  # Thesis stage 2: pairwise correlation > 0.8 is considered strong.
EXCLUDED_FEATURE_PREFIXES = ("Site_", "GPR", "Energy_6F")
TARGET_COLORS = {"GPR3mm": "#a8dadc", "GPR2mm": "#457b9d", "GPR1mm": "#1d3557"}


def make_output_dirs() -> None:
    for path in OUTPUT_SUBDIRS:
        path.mkdir(parents=True, exist_ok=True)


def require_numeric(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Ensure columns can be converted to numeric values."""
    converted = df.apply(pd.to_numeric, errors="coerce")
    invalid_cols = converted.columns[converted.isna().any()].tolist()
    if invalid_cols:
        raise ValueError(
            f"{label} contains missing or non-numeric columns. Check preprocessing output first: "
            f"{', '.join(invalid_cols)}"
        )
    return converted


def load_features_and_targets() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the CSV and return features, targets, and dropped-feature notes."""
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows.")

    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        raise ValueError(f"CSV is missing target columns: {', '.join(missing_targets)}")

    # Exclude Site one-hot columns and all GPR columns, including targets.
    drop_cols = [c for c in df.columns if c.startswith(EXCLUDED_FEATURE_PREFIXES)]
    targets = require_numeric(df[TARGETS], "Target columns")
    features = require_numeric(df.drop(columns=drop_cols), "Feature columns")

    constant_cols = [
        c for c in features.columns if features[c].nunique(dropna=False) <= 1
    ]
    dropped = [f"{c}: zero variance (single unique value)" for c in constant_cols]
    features = features.drop(columns=constant_cols)

    print(f"Excluded {len(drop_cols)} columns matching prefixes: {EXCLUDED_FEATURE_PREFIXES}.")
    print(f"Using {features.shape[1]} features.")
    if dropped:
        print("Additional dropped features:")
        for line in dropped:
            print(f"  - {line}")

    return features.reset_index(drop=True), targets.reset_index(drop=True), dropped


def write_dropped_features(dropped: list[str]) -> None:
    path = PREPROCESSING_DIR / "dropped_features.txt"
    if dropped:
        text = "Dropped feature columns:\n" + "\n".join(f"- {d}" for d in dropped) + "\n"
    else:
        text = "No columns were dropped.\n"
    path.write_text(text, encoding="utf-8")


def write_kept_features_targets(features: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Write the final retained features together with the target columns."""
    kept_df = pd.concat([features, targets], axis=1)
    kept_df.to_csv(KEPT_FEATURES_FILE, index=False, encoding="utf-8-sig")
    print(
        f"Saved {features.shape[1]} retained features and {targets.shape[1]} targets "
        f"to {KEPT_FEATURES_FILE.name}."
    )


def format_cell(value: object) -> str:
    """Format numeric values compactly for console reports."""
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:g}"
    return str(value)


def print_duplicate_feature_rows(features: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Print CSV line numbers whose retained feature values are duplicated."""
    feature_cols = features.columns.tolist()
    duplicated_mask = features.duplicated(subset=feature_cols, keep=False)
    if not duplicated_mask.any():
        print("No duplicate feature rows found.")
        return

    duplicate_groups = list(
        features.loc[duplicated_mask].groupby(feature_cols, sort=False, dropna=False)
    )
    print(
        f"Duplicate feature rows found: {len(duplicate_groups)} group(s). "
        "CSV line 1 is the header."
    )
    for group_number, (_, group) in enumerate(duplicate_groups, start=1):
        csv_lines = (group.index + 2).tolist()
        print(f"  Group {group_number}: CSV lines {', '.join(map(str, csv_lines))}")
        for row_idx in group.index:
            target_values = ", ".join(
                f"{target}={format_cell(targets.at[row_idx, target])}"
                for target in TARGETS
            )
            print(f"    line {row_idx + 2}: {target_values}")


def make_corr_heatmap(features: pd.DataFrame) -> None:
    """Create a feature-feature Spearman heatmap and high-correlation pair list."""
    corr = features.corr(method="spearman")

    fig_size = max(8, 0.45 * len(corr.columns))
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.6},
        xticklabels=corr.columns,
        yticklabels=corr.columns,
    )
    plt.title("Feature-feature Spearman correlation")
    plt.tight_layout()
    plt.savefig(FEATURE_CORRELATION_DIR / "corr_heatmap.pdf", dpi=150)
    plt.close()

    # Use only the upper triangle to avoid duplicate high-correlation pairs.
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr.iloc[i, j]
            if pd.notna(rho) and abs(rho) > HIGH_CORR_THRESHOLD:
                pairs.append({"feature_1": cols[i], "feature_2": cols[j], "spearman_rho": round(rho, 4)})

    pairs_df = pd.DataFrame(pairs).sort_values(
        "spearman_rho", key=lambda s: s.abs(), ascending=False
    )
    pairs_df.to_csv(
        FEATURE_CORRELATION_DIR / "high_corr_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"Found {len(pairs_df)} high-correlation pairs (|rho|>{HIGH_CORR_THRESHOLD}); "
        "saved to feature_correlation/high_corr_pairs.csv"
    )


def spearman_analysis(X: pd.DataFrame, y: pd.Series, target: str) -> pd.DataFrame:
    """Calculate Spearman rho and p-value for each feature against the target."""
    rows = []
    for col in X.columns:
        rho, p = spearmanr(X[col], y)
        rows.append({"feature": col, "spearman_rho": rho, "p_value": p})

    result = pd.DataFrame(rows)
    result["abs_rho"] = result["spearman_rho"].abs()
    result = result.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    result.to_csv(
        SPEARMAN_DIR / f"spearman_{target}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Horizontal bar chart: color shows sign, and an asterisk marks p<0.05.
    plot_df = result.iloc[::-1]
    colors = ["#d1495b" if v < 0 else "#2e86ab" for v in plot_df["spearman_rho"]]
    plt.figure(figsize=(8, max(6, 0.32 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["spearman_rho"], color=colors)
    for y_pos, (_, r) in enumerate(plot_df.iterrows()):
        if r["p_value"] < 0.05:
            plt.text(
                r["spearman_rho"] + (0.01 if r["spearman_rho"] >= 0 else -0.01),
                y_pos,
                "*",
                va="center",
                ha="left" if r["spearman_rho"] >= 0 else "right",
                fontsize=12,
            )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Spearman rho (* p<0.05)")
    plt.title(f"Spearman correlation with {target}  (n={len(y)})")
    plt.tight_layout()
    plt.savefig(SPEARMAN_DIR / f"spearman_{target}.pdf", dpi=150)
    plt.close()

    return result


def rf_analysis(X: pd.DataFrame, y: pd.Series, target: str) -> tuple[pd.DataFrame, dict]:
    """Run RF CV metrics and permutation importance aggregated over validation folds."""
    cv = RepeatedKFold(
        n_splits=RF_CV_SPLITS,
        n_repeats=RF_CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    r2_scores = []
    mae_scores = []
    fold_importances = []
    for train_idx, test_idx in cv.split(X):
        fitted = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ).fit(X.iloc[train_idx], y.iloc[train_idx])

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        predictions = fitted.predict(X_test)
        r2_scores.append(fitted.score(X_test, y_test))
        mae_scores.append(mean_absolute_error(y_test, predictions))

        perm = permutation_importance(
            fitted,
            X_test,
            y_test,
            n_repeats=RF_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring="r2",
        )
        fold_importances.append(perm.importances_mean)

    r2 = np.asarray(r2_scores)
    mae = np.asarray(mae_scores)
    fold_importances_array = np.vstack(fold_importances)
    n_folds = fold_importances_array.shape[0]
    importances = fold_importances_array.mean(axis=0)
    importance_std = fold_importances_array.std(axis=0, ddof=1)
    positive_fold_count = (fold_importances_array > 0).sum(axis=0)

    scores = {
        "target": target,
        "r2_mean": round(r2.mean(), 4),
        "r2_std": round(r2.std(), 4),
        "mae_mean": round(mae.mean(), 4),
        "mae_std": round(mae.std(), 4),
    }
    print(
        f"  [{target}] RF CV  R2={scores['r2_mean']:.3f}+/-{scores['r2_std']:.3f}  "
        f"MAE={scores['mae_mean']:.3f}+/-{scores['mae_std']:.3f}  folds={n_folds}"
    )

    result = pd.DataFrame(
        {
            "feature": X.columns,
            "perm_importance": importances,
            "perm_importance_std": importance_std,
            "positive_fold_count": positive_fold_count,
            "n_folds": n_folds,
        }
    ).sort_values("perm_importance", ascending=False).reset_index(drop=True)
    result.to_csv(
        RANDOM_FOREST_DIR / f"rf_importance_{target}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_df = result.iloc[::-1]
    plt.figure(figsize=(8, max(6, 0.32 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["perm_importance"], color="#5b8c5a")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Mean permutation importance (drop in R2)")
    plt.title(f"RF repeated-CV permutation importance for {target}  (n={len(y)})")
    plt.tight_layout()
    plt.savefig(RANDOM_FOREST_DIR / f"rf_importance_{target}.pdf", dpi=150)
    plt.close()

    return result, scores


def lasso_analysis(X: pd.DataFrame, y: pd.Series, target: str) -> pd.DataFrame:
    """Fit Elastic Net on standardized features and output coefficients."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=cv,
        random_state=RANDOM_STATE,
        max_iter=10000,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)

    result = pd.DataFrame(
        {"feature": X.columns, "coefficient": model.coef_}
    )
    result["abs_coef"] = result["coefficient"].abs()
    result = result.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    result.to_csv(
        ELASTIC_NET_DIR / f"lasso_{target}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"  [{target}] ElasticNet  alpha={model.alpha_:.4g}  l1_ratio={model.l1_ratio_}  "
        f"nonzero coefficients {int((result['coefficient'] != 0).sum())}/{len(result)}"
    )

    nonzero = result[result["coefficient"] != 0].iloc[::-1]
    if len(nonzero) == 0:
        return result
    colors = ["#d1495b" if v < 0 else "#2e86ab" for v in nonzero["coefficient"]]
    plt.figure(figsize=(8, max(4, 0.32 * len(nonzero))))
    plt.barh(nonzero["feature"], nonzero["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Standardized Elastic Net coefficient")
    plt.title(f"Elastic Net coefficients for {target}  (n={len(y)})")
    plt.tight_layout()
    plt.savefig(ELASTIC_NET_DIR / f"lasso_{target}.pdf", dpi=150)
    plt.close()

    return result


def make_summary_comparison(
    results_by_target: dict[str, pd.DataFrame],
    value_col: str,
    rank_col: str,
    output_path: Path,
    xlabel: str,
    title: str,
    top_n: int = SUMMARY_TOP_N,
) -> None:
    """Compare signed metrics across targets, selecting top features by absolute rank."""
    max_rank = None
    for res in results_by_target.values():
        s = res.set_index("feature")[rank_col].abs()
        max_rank = s if max_rank is None else pd.concat([max_rank, s], axis=1).max(axis=1)
    if max_rank is None or max_rank.empty:
        return
    top_features = max_rank.sort_values(ascending=False).head(top_n).index.tolist()

    data = {}
    for target, res in results_by_target.items():
        s = res.set_index("feature")[value_col]
        data[target] = [s.get(f, 0.0) for f in top_features]

    plot_df = pd.DataFrame(data, index=top_features).iloc[::-1]
    ax = plot_df.plot(
        kind="barh",
        figsize=(9, max(6, 0.5 * len(top_features))),
        color=TARGET_COLORS,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.legend(title="Target")
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def main() -> None:
    make_output_dirs()

    features, targets, dropped = load_features_and_targets()
    write_dropped_features(dropped)
    write_kept_features_targets(features, targets)
    print_duplicate_feature_rows(features, targets)

    make_corr_heatmap(features)

    spearman_by_target: dict[str, pd.DataFrame] = {}
    rf_by_target: dict[str, pd.DataFrame] = {}
    lasso_by_target: dict[str, pd.DataFrame] = {}
    cv_scores: list[dict] = []

    for target in TARGETS:
        print(f"\n=== {target} ===")
        X, y = features, targets[target]
        print(f"  Sample count: {len(y)}")

        spearman_by_target[target] = spearman_analysis(X, y, target)
        rf_by_target[target], scores = rf_analysis(X, y, target)
        cv_scores.append(scores)
        lasso_by_target[target] = lasso_analysis(X, y, target)

    pd.DataFrame(cv_scores).to_csv(
        RANDOM_FOREST_DIR / "cv_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_summary_comparison(
        spearman_by_target,
        value_col="spearman_rho",
        rank_col="abs_rho",
        output_path=SPEARMAN_DIR / "spearman_summary_comparison.pdf",
        xlabel="Spearman rho with GPR",
        title="Spearman correlation vs gamma criteria (signed rho, top features)",
        top_n=SUMMARY_TOP_N,
    )
    make_summary_comparison(
        rf_by_target,
        value_col="perm_importance",
        rank_col="perm_importance",
        output_path=RANDOM_FOREST_DIR / "rf_summary_comparison.pdf",
        xlabel="Permutation importance (drop in R2)",
        title="RF permutation importance vs gamma criteria (top features)",
        top_n=SUMMARY_TOP_N,
    )
    make_summary_comparison(
        lasso_by_target,
        value_col="coefficient",
        rank_col="abs_coef",
        output_path=ELASTIC_NET_DIR / "lasso_summary_comparison.pdf",
        xlabel="Standardized Elastic Net coefficient",
        title="Elastic Net coefficients vs gamma criteria (signed, top features)",
        top_n=SUMMARY_TOP_N,
    )

    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
