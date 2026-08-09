"""Select top features per GPR target by averaging ranks across three methods.

Inputs (under Model/4_feature_importance/, produced by 4_feature_importance.py):
  - elastic_net_lasso/lasso_{target}.csv        (feature, coefficient, abs_coef)
  - random_forest/rf_importance_{target}.csv    (feature, perm_importance, ...)
  - spearman/spearman_{target}.csv              (feature, spearman_rho, p_value, abs_rho)
  - feature_correlation/high_corr_pairs.csv     (feature_1, feature_2, spearman_rho)

Outputs: CSV files under Model/5_feature_selection/, per target:
  - selected_features_{target}.csv  (the top-N list after redundancy filtering)
  - skipped_features_{target}.csv   (features skipped due to high correlation)

Algorithm (run separately for each GPR target):
  1. Rank features within each method by absolute importance
     (ties get the average rank, so zero LASSO coefficients share one rank).
  2. Average the three ranks into avg_rank and sort ascending.
  3. Walk down the averaged ranking greedily: skip any candidate that appears
     in high_corr_pairs.csv (|rho| > 0.8) with an already-selected feature,
     otherwise select it, until TOP_N features are chosen.
"""

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
IMPORTANCE_DIR = SCRIPT_DIR / "4_feature_importance"
LASSO_DIR = IMPORTANCE_DIR / "elastic_net_lasso"
RANDOM_FOREST_DIR = IMPORTANCE_DIR / "random_forest"
SPEARMAN_DIR = IMPORTANCE_DIR / "spearman"
HIGH_CORR_FILE = IMPORTANCE_DIR / "feature_correlation" / "high_corr_pairs.csv"
OUTPUT_DIR = SCRIPT_DIR / "5_feature_selection"

TARGETS = ["GPR3mm", "GPR2mm", "GPR1mm"]
TOP_N = 15


def read_method_csv(path: Path, sort_col: str) -> pd.DataFrame:
    """Read one method's importance CSV and check the required columns exist."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("feature", sort_col):
        if col not in df.columns:
            raise ValueError(f"{path.name} is missing column '{col}'.")
    if df["feature"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicated feature names.")
    return df.set_index("feature")


def load_high_corr_pairs() -> dict[frozenset, float]:
    """Load high-correlation pairs as {frozenset({f1, f2}): rho}."""
    if not HIGH_CORR_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {HIGH_CORR_FILE}")
    df = pd.read_csv(HIGH_CORR_FILE, encoding="utf-8-sig")
    return {
        frozenset((row.feature_1, row.feature_2)): row.spearman_rho
        for row in df.itertuples(index=False)
    }


def build_rank_table(target: str) -> pd.DataFrame:
    """Combine the three methods' rankings into one avg_rank table."""
    lasso = read_method_csv(LASSO_DIR / f"lasso_{target}.csv", "abs_coef")
    rf = read_method_csv(RANDOM_FOREST_DIR / f"rf_importance_{target}.csv", "perm_importance")
    spearman = read_method_csv(SPEARMAN_DIR / f"spearman_{target}.csv", "abs_rho")

    if not (set(lasso.index) == set(rf.index) == set(spearman.index)):
        raise ValueError(
            f"[{target}] Feature sets differ between the three method CSVs. "
            "Re-run 4_feature_importance.py so all outputs match."
        )

    table = pd.DataFrame(index=sorted(lasso.index))
    table["lasso_rank"] = lasso["abs_coef"].rank(ascending=False, method="average")
    table["rf_rank"] = rf["perm_importance"].rank(ascending=False, method="average")
    table["spearman_rank"] = spearman["abs_rho"].rank(ascending=False, method="average")
    table["avg_rank"] = table[["lasso_rank", "rf_rank", "spearman_rank"]].mean(axis=1)
    table["lasso_coef"] = lasso["coefficient"]
    table["rf_perm_importance"] = rf["perm_importance"]
    table["spearman_rho"] = spearman["spearman_rho"]

    # Sort by avg_rank; the index (feature name) breaks ties reproducibly.
    return table.rename_axis("feature").sort_index().sort_values("avg_rank", kind="stable")


def greedy_select(
    table: pd.DataFrame, high_corr: dict[frozenset, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk down avg_rank, skipping candidates highly correlated with a selected one."""
    selected: list[str] = []
    skipped_rows = []
    for feature, row in table.iterrows():
        if len(selected) >= TOP_N:
            break
        conflict = next(
            (s for s in selected if frozenset((feature, s)) in high_corr), None
        )
        if conflict is not None:
            skipped_rows.append(
                {
                    "feature": feature,
                    "avg_rank": row["avg_rank"],
                    "correlated_with": conflict,
                    "spearman_rho": high_corr[frozenset((feature, conflict))],
                }
            )
        else:
            selected.append(feature)

    selected_df = table.loc[selected].reset_index()
    selected_df.insert(0, "rank", range(1, len(selected_df) + 1))
    skipped_df = pd.DataFrame(
        skipped_rows, columns=["feature", "avg_rank", "correlated_with", "spearman_rho"]
    )
    return selected_df, skipped_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    high_corr = load_high_corr_pairs()
    print(f"Loaded {len(high_corr)} high-correlation pairs from {HIGH_CORR_FILE.name}.")

    selected_by_target: dict[str, list[str]] = {}
    for target in TARGETS:
        print(f"\n=== {target} ===")
        table = build_rank_table(target)
        selected_df, skipped_df = greedy_select(table, high_corr)
        selected_by_target[target] = selected_df["feature"].tolist()

        selected_df.to_csv(
            OUTPUT_DIR / f"selected_features_{target}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        skipped_df.to_csv(
            OUTPUT_DIR / f"skipped_features_{target}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(f"  Selected top {len(selected_df)} features:")
        for row in selected_df.itertuples(index=False):
            print(f"    {row.rank:>2}. {row.feature}  (avg_rank={row.avg_rank:.2f})")
        if len(skipped_df) > 0:
            print(f"  Skipped {len(skipped_df)} features due to |rho|>0.8:")
            for row in skipped_df.itertuples(index=False):
                print(
                    f"    - {row.feature}  (correlated with {row.correlated_with}, "
                    f"rho={row.spearman_rho:.2f})"
                )
        else:
            print("  No features skipped.")

    # Compare the three target lists for the write-up.
    common = set.intersection(*(set(v) for v in selected_by_target.values()))
    print(f"\nFeatures selected for all three targets ({len(common)}):")
    print(f"  {', '.join(sorted(common))}")
    for target, feats in selected_by_target.items():
        unique = set(feats) - set.union(
            *(set(v) for t, v in selected_by_target.items() if t != target)
        )
        if unique:
            print(f"Only in {target}: {', '.join(sorted(unique))}")

    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
