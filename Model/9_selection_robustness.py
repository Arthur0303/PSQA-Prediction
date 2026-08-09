"""Check how much the step-4/5 feature selection biases the step-6 CV scores.

Steps 4-5 rank and select features on ALL rows of the dataset; step 6 then cross-validates
on the same rows, so the scores of the 5/10/15-feature configurations are
optimistically biased (the full-feature scores are not). This script quantifies
that bias for the configurations chosen at the stage-2/4 operating point:
the whole selection procedure (three-method ranking + greedy redundancy
filter) is re-run INSIDE each outer-training fold, the chosen model is tuned
and tested with the step-6 protocol, and the per-fold scores are compared
pairwise against the step-6 results. Both numbers go into the dissertation.

Inputs:
  - Model/4_kept_features_targets.csv           (features + targets)
  - Model/6_model_training/fold_results.csv     (step-6 per-fold scores)
  - Model/8_operating_point/operating_point.json (chosen model/n per target)

Outputs (under Model/9_selection_robustness/):
  - fold_details.csv                (per fold: in-fold scores, step-6 scores,
                                     paired deltas, selected features)
  - comparison_summary.csv          (per target: means, paired delta, Wilcoxon p)
  - feature_selection_frequency.csv (how often each feature is picked in 25 folds)
  - selection_frequency.pdf         (per-target selection-share bar chart)

Protocol fidelity: the model registry, hyperparameter grids, inner CV and the
25 outer folds are imported from 6_model_training.py, so any protocol change
there propagates here. The in-fold ranking replicates 4_feature_importance.py
(Spearman |rho|, ElasticNetCV |coef|, RF permutation importance) and
5_feature_selection.py (average rank + greedy |rho|>0.8 filter) on the outer
TRAINING rows only. Exception: the RF permutation importance is aggregated
over 5x2 inner folds instead of 5x5 to keep the runtime workable (constant
RF_SUBCV_REPEATS below).

Set SANITY_FIXED_SELECTION = True to bypass the in-fold selection and reuse
the fixed step-5 feature list: every per-fold score must then reproduce
fold_results.csv exactly, which validates that the training protocol matches.
"""

import importlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend and save plots directly.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
step6 = importlib.import_module("6_model_training")

KEPT_FILE = SCRIPT_DIR / "4_kept_features_targets.csv"
FOLD_RESULTS_FILE = SCRIPT_DIR / "6_model_training" / "fold_results.csv"
OPERATING_POINT_FILE = SCRIPT_DIR / "8_operating_point" / "operating_point.json"
SELECTION_DIR = SCRIPT_DIR / "5_feature_selection"
OUTPUT_DIR = SCRIPT_DIR / "9_selection_robustness"

RANDOM_STATE = step6.RANDOM_STATE
HIGH_CORR_THRESHOLD = 0.8  # same as 4_feature_importance.py
RF_N_ESTIMATORS = 400      # same as 4_feature_importance.py
RF_PERMUTATION_REPEATS = 10
RF_SUBCV_SPLITS = 5
RF_SUBCV_REPEATS = 2       # step 4 uses 5; reduced for runtime (documented)
ELASTICNET_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
SANITY_FIXED_SELECTION = False

INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_BLUE = "#2a78d6"  # same series accent as steps 7/8


def load_chosen_configs() -> dict[str, tuple[str, int]]:
    if not OPERATING_POINT_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {OPERATING_POINT_FILE}. Run 8_operating_point.py first."
        )
    with OPERATING_POINT_FILE.open(encoding="utf-8") as handle:
        operating_point = json.load(handle)
    configs = {
        target: (spec["model"], int(spec["n_features"]))
        for target, spec in operating_point["combo"].items()
    }
    for target, (model, n_features) in configs.items():
        print(f"Chosen config for {target}: {model} (n={n_features})")
    return configs


def rank_features_in_fold(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Average rank over the three step-4 methods, computed on training rows only."""
    spearman_abs = pd.Series(
        [abs(spearmanr(X[col], y)[0]) for col in X.columns], index=X.columns
    )

    scaler = StandardScaler()
    elastic_net = ElasticNetCV(
        l1_ratio=ELASTICNET_L1_RATIOS,
        cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        max_iter=10000,
        n_jobs=1,
    )
    elastic_net.fit(scaler.fit_transform(X), y)
    lasso_abs = pd.Series(np.abs(elastic_net.coef_), index=X.columns)

    sub_cv = RepeatedKFold(
        n_splits=RF_SUBCV_SPLITS, n_repeats=RF_SUBCV_REPEATS, random_state=RANDOM_STATE
    )
    fold_importances = []
    for train_idx, test_idx in sub_cv.split(X):
        forest = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1
        ).fit(X.iloc[train_idx], y.iloc[train_idx])
        perm = permutation_importance(
            forest,
            X.iloc[test_idx],
            y.iloc[test_idx],
            n_repeats=RF_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring="r2",
            n_jobs=-1,
        )
        fold_importances.append(perm.importances_mean)
    rf_importance = pd.Series(np.vstack(fold_importances).mean(axis=0), index=X.columns)

    ranks = pd.DataFrame(
        {
            "lasso_rank": lasso_abs.rank(ascending=False, method="average"),
            "rf_rank": rf_importance.rank(ascending=False, method="average"),
            "spearman_rank": spearman_abs.rank(ascending=False, method="average"),
        }
    )
    return ranks.mean(axis=1)


def greedy_select_in_fold(X: pd.DataFrame, avg_rank: pd.Series, top_n: int) -> list[str]:
    """Step-5 greedy walk with the high-correlation pairs computed on X only."""
    corr = X.corr(method="spearman").abs()
    # Sort by avg_rank with the feature name as a reproducible tie-break.
    order = avg_rank.sort_index().sort_values(kind="stable").index
    selected: list[str] = []
    for feature in order:
        if len(selected) >= top_n:
            break
        if any(corr.at[feature, kept] > HIGH_CORR_THRESHOLD for kept in selected):
            continue
        selected.append(feature)
    return selected


def fixed_step5_features(target: str, top_n: int) -> list[str]:
    ranking = pd.read_csv(
        SELECTION_DIR / f"selected_features_{target}.csv", encoding="utf-8-sig"
    )
    return ranking["feature"].tolist()[:top_n]


def evaluate_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> tuple[float, float]:
    """Tune and score one outer fold with the exact step-6 protocol."""
    spec = step6.build_model_registry()[model_name]
    estimator = spec["estimator"]
    param_grid = spec["param_grid"]
    if spec["scale"]:
        estimator = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        param_grid = {f"model__{k}": v for k, v in param_grid.items()}
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=KFold(n_splits=step6.INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring=step6.INNER_SCORING,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)
    return r2_score(y_test, predictions), mean_absolute_error(y_test, predictions)


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Recessive chrome: hairline grid, light baseline, muted tick labels."""
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRIDLINE, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor="#52514e")


def plot_selection_frequency(frequency: pd.DataFrame) -> None:
    """Horizontal bars of the per-fold selection share, one panel per target."""
    targets = list(dict.fromkeys(frequency["target"]))
    n_max = int(frequency.groupby("target").size().max())
    fig, axes = plt.subplots(
        1, len(targets), figsize=(6.0 * len(targets), max(4.8, 0.28 * n_max))
    )
    for ax, target in zip(np.atleast_1d(axes), targets):
        sub = frequency[frequency["target"] == target].sort_values("selection_share")
        ax.barh(sub["feature"], sub["selection_share"], color=SERIES_BLUE)
        style_axes(ax, grid_axis="x")
        ax.set_xlim(0, 1.0)
        n_folds = int(sub["n_folds"].iloc[0])
        ax.set_xlabel(f"Selection share over {n_folds} outer folds", color="#52514e")
        ax.set_title(target, color=INK)
    fig.suptitle(
        "In-fold feature-selection frequency for the chosen configurations",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT_DIR / "selection_frequency.pdf")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = load_chosen_configs()
    df = pd.read_csv(KEPT_FILE, encoding="utf-8-sig")
    features = df.drop(columns=step6.TARGETS)
    step6_folds = pd.read_csv(FOLD_RESULTS_FILE, encoding="utf-8-sig")

    outer_cv = RepeatedKFold(
        n_splits=step6.OUTER_CV_SPLITS,
        n_repeats=step6.OUTER_CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    outer_splits = list(outer_cv.split(features))

    detail_rows = []
    summary_rows = []
    frequency_rows = []
    for target, (model_name, top_n) in configs.items():
        if top_n >= features.shape[1]:
            print(
                f"\n[{target}] chosen config uses the full feature set (n={top_n}); "
                "selection bias does not apply - skipped."
            )
            continue
        print(f"\n=== {target}: {model_name}, in-fold top-{top_n} selection ===")
        y = df[target]
        reference = step6_folds[
            (step6_folds["target"] == target)
            & (step6_folds["model"] == model_name)
            & (step6_folds["n_features"] == top_n)
        ].set_index("fold")
        if len(reference) != len(outer_splits):
            raise ValueError(
                f"fold_results.csv has {len(reference)} folds for this config; "
                f"expected {len(outer_splits)}. Re-run 6_model_training.py."
            )

        selection_counts: dict[str, int] = {}
        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
            X_train, y_train = features.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = features.iloc[test_idx], y.iloc[test_idx]

            if SANITY_FIXED_SELECTION:
                selected = fixed_step5_features(target, top_n)
            else:
                avg_rank = rank_features_in_fold(X_train, y_train)
                selected = greedy_select_in_fold(X_train, avg_rank, top_n)
            for feature in selected:
                selection_counts[feature] = selection_counts.get(feature, 0) + 1

            r2, mae = evaluate_fold(
                X_train[selected], y_train, X_test[selected], y_test, model_name
            )
            detail_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "n_features": top_n,
                    "fold": fold_idx,
                    "mae_infold_selection": mae,
                    "r2_infold_selection": r2,
                    "mae_step6": reference.at[fold_idx, "mae"],
                    "r2_step6": reference.at[fold_idx, "r2"],
                    "delta_mae": mae - reference.at[fold_idx, "mae"],
                    "delta_r2": r2 - reference.at[fold_idx, "r2"],
                    "selected_features": ";".join(selected),
                }
            )
            print(
                f"  fold {fold_idx:>2}: MAE {mae:.3f} (step 6 "
                f"{reference.at[fold_idx, 'mae']:.3f})"
            )

        target_details = pd.DataFrame([r for r in detail_rows if r["target"] == target])
        if SANITY_FIXED_SELECTION:
            if not np.allclose(
                target_details["mae_infold_selection"], target_details["mae_step6"]
            ):
                raise AssertionError(
                    f"[{target}] sanity mode: per-fold MAE does not reproduce "
                    "fold_results.csv - the training protocol has diverged from step 6."
                )
            print(f"  [{target}] sanity mode: all folds reproduce step 6 exactly.")
        deltas = target_details["delta_mae"]
        try:
            wilcoxon_p = float(wilcoxon(deltas).pvalue) if (deltas != 0).any() else 1.0
        except ValueError:
            wilcoxon_p = float("nan")
        summary_rows.append(
            {
                "target": target,
                "model": model_name,
                "n_features": top_n,
                "n_folds": len(target_details),
                "mae_step6_mean": target_details["mae_step6"].mean(),
                "mae_infold_mean": target_details["mae_infold_selection"].mean(),
                "mae_infold_std": target_details["mae_infold_selection"].std(ddof=1),
                "delta_mae_mean": deltas.mean(),
                "delta_mae_std": deltas.std(ddof=1),
                "delta_mae_wilcoxon_p": wilcoxon_p,
                "r2_step6_mean": target_details["r2_step6"].mean(),
                "r2_infold_mean": target_details["r2_infold_selection"].mean(),
            }
        )
        for feature, count in sorted(
            selection_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            frequency_rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "times_selected": count,
                    "n_folds": len(outer_splits),
                    "selection_share": round(count / len(outer_splits), 3),
                }
            )

    if not detail_rows:
        print("\nNothing to check: every chosen config uses the full feature set.")
        return

    pd.DataFrame(detail_rows).round(6).to_csv(
        OUTPUT_DIR / "fold_details.csv", index=False, encoding="utf-8-sig"
    )
    summary = pd.DataFrame(summary_rows).round(4)
    summary.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    frequency = pd.DataFrame(frequency_rows)
    frequency.to_csv(
        OUTPUT_DIR / "feature_selection_frequency.csv", index=False, encoding="utf-8-sig"
    )
    plot_selection_frequency(frequency)

    print("\nSummary (positive delta_mae = in-fold selection scores worse, i.e. the")
    print("step-6 number was optimistic by about that much):")
    print(summary.to_string(index=False))
    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
