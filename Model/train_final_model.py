"""Train the deployment models on ALL rows and save them (stage 5, training).

For the combo chosen in 8_operating_point.py (one model x feature-count per
target), hyperparameters are re-tuned with the step-6 grid on the full data
set (5-fold KFold, shuffle, random_state 42, MAE scoring) and the best
estimator is refit on all rows. The decision thresholds are NOT refit here:
the margins come from the residual-quantile margin analysis on the
out-of-fold predictions (operating_point.json), which is the honest error
estimate for unseen plans.

Inputs:
  - Model/4_kept_features_targets.csv
  - Model/5_feature_selection/selected_features_{target}.csv
  - Model/8_operating_point/operating_point.json

Outputs (under Model/final_model/):
  - model_GPR3mm.joblib / model_GPR2mm.joblib  (fitted estimator or pipeline)
  - final_model_meta.json  (feature lists, best params, margins, decision
    thresholds, per-feature training ranges for out-of-range warnings,
    library versions)

Protocol fidelity: the model registry and grids are imported from
6_model_training.py, so the deployment models are tuned exactly like the
cross-validated ones.
"""

import importlib
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
step6 = importlib.import_module("6_model_training")

KEPT_FILE = SCRIPT_DIR / "4_kept_features_targets.csv"
SELECTION_DIR = SCRIPT_DIR / "5_feature_selection"
OPERATING_POINT_FILE = SCRIPT_DIR / "8_operating_point" / "operating_point.json"
OUTPUT_DIR = SCRIPT_DIR / "final_model"

CLS_TARGETS = ("GPR3mm", "GPR2mm")


def load_operating_point() -> dict:
    if not OPERATING_POINT_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {OPERATING_POINT_FILE}. Run 8_operating_point.py first."
        )
    with OPERATING_POINT_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def feature_list_for(target: str, n_features: int, all_features: list[str]) -> list[str]:
    """Top-N of the step-5 ranking, or the full set when N covers everything."""
    if n_features >= len(all_features):
        return list(all_features)
    ranking = pd.read_csv(
        SELECTION_DIR / f"selected_features_{target}.csv", encoding="utf-8-sig"
    )
    selected = ranking["feature"].tolist()[:n_features]
    unknown = sorted(set(selected) - set(all_features))
    if unknown:
        raise ValueError(
            f"selected_features_{target}.csv contains unknown features: {', '.join(unknown)}"
        )
    return selected


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    operating_point = load_operating_point()
    df = pd.read_csv(KEPT_FILE, encoding="utf-8-sig")
    all_features = [c for c in df.columns if c not in step6.TARGETS]
    print(f"Loaded {len(df)} rows, {len(all_features)} candidate features.")

    registry = step6.build_model_registry()
    cv = KFold(n_splits=step6.INNER_CV_SPLITS, shuffle=True, random_state=step6.RANDOM_STATE)

    models_meta = {}
    for target in CLS_TARGETS:
        combo = operating_point["combo"][target]
        model_name = combo["model"]
        n_features = int(combo["n_features"])
        feature_list = feature_list_for(target, n_features, all_features)
        X = df[feature_list]
        y = df[target]

        spec = registry[model_name]
        estimator = spec["estimator"]
        param_grid = spec["param_grid"]
        if spec["scale"]:
            estimator = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
            param_grid = {f"model__{k}": v for k, v in param_grid.items()}

        search = GridSearchCV(
            estimator, param_grid, cv=cv, scoring=step6.INNER_SCORING, n_jobs=-1
        )
        search.fit(X, y)
        in_sample_mae = float((y - search.predict(X)).abs().mean())
        print(
            f"[{target}] {model_name} (n={n_features})  "
            f"tuning CV MAE={-search.best_score_:.3f}  in-sample MAE={in_sample_mae:.3f}  "
            f"params={search.best_params_}"
        )

        model_path = OUTPUT_DIR / f"model_{target}.joblib"
        joblib.dump(search.best_estimator_, model_path)
        models_meta[target] = {
            "model": model_name,
            "n_features": n_features,
            "features": feature_list,
            "best_params": search.best_params_,
            "tuning_cv_mae": round(-float(search.best_score_), 4),
            "outer_cv_mae": combo["cv_mae"],
            "in_sample_mae": round(in_sample_mae, 4),
            "model_file": model_path.name,
            "feature_ranges": {
                feature: {
                    "min": float(df[feature].min()),
                    "max": float(df[feature].max()),
                }
                for feature in feature_list
            },
        }

    meta = {
        "created": date.today().isoformat(),
        "n_training_rows": len(df),
        "training_file": KEPT_FILE.name,
        "operating_point_file": str(OPERATING_POINT_FILE.relative_to(SCRIPT_DIR)),
        "thresholds": operating_point["thresholds"],
        "margin_method": operating_point["margin_method"],
        "margins_pp": operating_point["margins_pp"],
        "decision_thresholds": operating_point["decision_thresholds"],
        "models": models_meta,
        "versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    with (OUTPUT_DIR / "final_model_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False, default=str)

    print(f"\nDecision thresholds saved with the models:")
    for cls in ("class2", "class0"):
        values = meta["decision_thresholds"][cls]
        print(f"  {cls}: " + ", ".join(f"{t}={values[t]}" for t in CLS_TARGETS))
    print(f"\nDone. All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
