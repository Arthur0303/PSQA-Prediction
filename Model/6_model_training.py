"""Train and compare regression models across feature-count levels.

Inputs:
  - Model/4_kept_features_targets.csv                        (31 features + 3 GPR targets)
  - Model/5_feature_selection/selected_features_{target}.csv (per-target feature ranking)

Outputs (under Model/6_model_training/):
  - fold_results.csv  (per outer-fold metrics for every target/model/feature-count)
  - summary.csv       (mean/std metrics per target/model/feature-count)
  - feature_sets.csv  (the exact features used for each target/feature-count)
  - predictions.csv   (out-of-fold predictions per row for every configuration,
                       consumed by 8_operating_point.py for the margin scan)

Protocol (run separately for each GPR target and feature count):
  Nested cross-validation. The outer RepeatedKFold (5 splits x 5 repeats,
  random_state=42, same as 4_feature_importance.py) measures generalization
  (R2 / MAE / RMSE per fold); within each outer training set an inner 5-fold
  GridSearchCV (MAE scoring) picks hyperparameters. Models that need feature
  scaling (Ridge, SVR, MLP) run inside a Pipeline with StandardScaler so the
  scaler is fit on each training fold only.

Note: The data keeps repeated measurements from the same plan (identical
features with different GPR values) and plain RepeatedKFold is used
deliberately, matching 4_feature_importance.py. The outer splits are computed
once from the row count, so every model and feature count is evaluated on the
same 25 folds and results are directly comparable.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "4_kept_features_targets.csv"
SELECTION_DIR = SCRIPT_DIR / "5_feature_selection"
OUTPUT_DIR = SCRIPT_DIR / "6_model_training"

TARGETS = ["GPR3mm", "GPR2mm", "GPR1mm"]
SELECTED_COUNTS = [5, 10, 15]  # taken from the step-5 ranking; the full set (31) is appended
RANDOM_STATE = 42
OUTER_CV_SPLITS = 5
OUTER_CV_REPEATS = 5
INNER_CV_SPLITS = 5
INNER_SCORING = "neg_mean_absolute_error"


def build_model_registry() -> dict[str, dict]:
    """Model name -> estimator, hyperparameter grid, and whether to standardize.

    Estimators that support n_jobs are kept single-threaded; parallelism happens
    across GridSearchCV candidates instead.
    """
    return {
        "Dummy": {
            "estimator": DummyRegressor(strategy="mean"),
            "param_grid": {},
            "scale": False,
        },
        "Ridge": {
            "estimator": Ridge(),
            "param_grid": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "scale": True,
        },
        "SVR": {
            "estimator": SVR(kernel="rbf"),
            "param_grid": {
                "C": [1, 10, 100],
                "gamma": ["scale", 0.03, 0.1],
                "epsilon": [0.1, 0.5],
            },
            "scale": True,
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(
                n_estimators=400, random_state=RANDOM_STATE, n_jobs=1
            ),
            "param_grid": {
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 3, 5],
                "max_features": [1.0, "sqrt"],
            },
            "scale": False,
        },
        "XGBoost": {
            "estimator": XGBRegressor(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=1, verbosity=0
            ),
            "param_grid": {
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3, 5],
                "subsample": [0.7, 1.0],
            },
            "scale": False,
        },
        "MLP": {
            "estimator": MLPRegressor(
                solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE
            ),
            "param_grid": {
                "hidden_layer_sizes": [(16,), (32,), (32, 16)],
                "alpha": [1e-4, 1e-2, 1.0],
            },
            "scale": True,
        },
    }


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


def load_features_and_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {INPUT_FILE.name}.")

    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        raise ValueError(f"CSV is missing target columns: {', '.join(missing_targets)}")

    targets = require_numeric(df[TARGETS], "Target columns")
    features = require_numeric(df.drop(columns=TARGETS), "Feature columns")
    print(f"Using {features.shape[1]} features.")
    return features.reset_index(drop=True), targets.reset_index(drop=True)


def load_feature_sets(all_features: list[str]) -> dict[str, dict[int, list[str]]]:
    """Build {target: {n_features: [feature, ...]}} from the step-5 rankings.

    Counts in SELECTED_COUNTS take the top-N of the per-target ranking; the
    full feature set is added as its own count (currently 31).
    """
    sets_by_target: dict[str, dict[int, list[str]]] = {}
    for target in TARGETS:
        path = SELECTION_DIR / f"selected_features_{target}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        ranking = pd.read_csv(path, encoding="utf-8-sig")
        if "feature" not in ranking.columns:
            raise ValueError(f"{path.name} is missing column 'feature'.")
        ranked = ranking["feature"].tolist()
        if len(ranked) < max(SELECTED_COUNTS):
            raise ValueError(
                f"{path.name} lists {len(ranked)} features but {max(SELECTED_COUNTS)} are needed."
            )
        unknown = sorted(set(ranked) - set(all_features))
        if unknown:
            raise ValueError(
                f"{path.name} contains features missing from {INPUT_FILE.name}: "
                f"{', '.join(unknown)}. Re-run steps 4 and 5 so the outputs match."
            )
        sets_by_target[target] = {n: ranked[:n] for n in SELECTED_COUNTS}
        sets_by_target[target][len(all_features)] = list(all_features)
    return sets_by_target


def write_feature_sets(sets_by_target: dict[str, dict[int, list[str]]]) -> None:
    rows = [
        {"target": target, "n_features": n, "rank": rank, "feature": feature}
        for target, by_count in sets_by_target.items()
        for n, feats in by_count.items()
        for rank, feature in enumerate(feats, start=1)
    ]
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "feature_sets.csv", index=False, encoding="utf-8-sig"
    )


def evaluate_config(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    param_grid: dict,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict], list[dict]]:
    """Nested CV for one estimator: per outer fold, tune on train, score on test.

    Returns (per-fold metric rows, per-row out-of-fold prediction rows).
    """
    inner_cv = KFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    pred_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        search = GridSearchCV(
            estimator,
            param_grid,
            cv=inner_cv,
            scoring=INNER_SCORING,
            n_jobs=-1,
        )
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = search.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx]
        rows.append(
            {
                "fold": fold_idx,
                "r2": r2_score(y_test, predictions),
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": root_mean_squared_error(y_test, predictions),
                "best_params": json.dumps(search.best_params_, default=str),
            }
        )
        repeat = fold_idx // OUTER_CV_SPLITS
        for row_index, y_true, y_pred in zip(test_idx, y_test, predictions):
            pred_rows.append(
                {
                    "fold": fold_idx,
                    "repeat": repeat,
                    "row_index": int(row_index),
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
    return rows, pred_rows


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features, targets = load_features_and_targets()
    sets_by_target = load_feature_sets(features.columns.tolist())
    write_feature_sets(sets_by_target)

    # Outer splits depend only on the row count, so computing them once keeps
    # the exact same folds for every target, model, and feature count.
    outer_cv = RepeatedKFold(
        n_splits=OUTER_CV_SPLITS,
        n_repeats=OUTER_CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    outer_splits = list(outer_cv.split(features))
    registry = build_model_registry()

    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for target in TARGETS:
        print(f"\n=== {target} ===")
        y = targets[target]
        for n_features, feature_list in sets_by_target[target].items():
            X = features[feature_list]
            for model_name, spec in registry.items():
                estimator = spec["estimator"]
                param_grid = spec["param_grid"]
                if spec["scale"]:
                    estimator = Pipeline(
                        [("scaler", StandardScaler()), ("model", estimator)]
                    )
                    param_grid = {f"model__{k}": v for k, v in param_grid.items()}

                start = time.perf_counter()
                rows, pred_rows = evaluate_config(X, y, estimator, param_grid, outer_splits)
                elapsed = time.perf_counter() - start

                for row in rows:
                    fold_rows.append(
                        {"target": target, "model": model_name, "n_features": n_features, **row}
                    )
                for row in pred_rows:
                    prediction_rows.append(
                        {"target": target, "model": model_name, "n_features": n_features, **row}
                    )
                r2 = np.array([r["r2"] for r in rows])
                mae = np.array([r["mae"] for r in rows])
                print(
                    f"  [{target}] n_features={n_features:>2}  {model_name:<12} "
                    f"R2={r2.mean():.3f}+/-{r2.std():.3f}  "
                    f"MAE={mae.mean():.3f}+/-{mae.std():.3f}  ({elapsed:.1f}s)"
                )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUTPUT_DIR / "fold_results.csv", index=False, encoding="utf-8-sig")

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(
        OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Saved {len(predictions_df)} out-of-fold prediction rows.")

    summary = (
        fold_df.groupby(["target", "model", "n_features"], sort=False)[["r2", "mae", "rmse"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["n_folds"] = len(outer_splits)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved {len(fold_df)} fold rows and {len(summary)} summary rows.")
    print(f"Done. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
