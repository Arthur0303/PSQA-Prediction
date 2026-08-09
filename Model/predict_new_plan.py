"""Classify new plans with the saved deployment models (stage 5, inference).

Workflow for a new plan: run Data_extract_v7/extract_data_v7.py to compute the
complexity metrics, copy the resulting row(s) into Model/Data/new_plans.xlsx
(sheet `RadCalc`, same column layout as data.xlsx / New80_v4.xlsx - GPR columns
may be empty), then run this script.

For every row it predicts GPR3mm and GPR2mm and applies the operating-point
decision rule stored with the models:
  2 = both predictions clear the margin-tightened class-2 thresholds -> auto-pass
  0 = both predictions fall below the margin-tightened class-0 thresholds -> replan
  1 = everything else -> manual review

Inputs:
  - Model/final_model/model_GPR3mm.joblib / model_GPR2mm.joblib
  - Model/final_model/final_model_meta.json
  - Model/Data/new_plans.xlsx  (sheet `RadCalc`)

Output:
  - Model/predictions.csv  (one row per plan: ID/Plan if present, predicted
    GPRs, decision 2/1/0, reason, out-of-range feature warnings)

Features whose value lies outside the training min/max are flagged - the model
extrapolates there, so treat the prediction with extra caution (the flag does
not change the decision).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "final_model"
META_FILE = MODEL_DIR / "final_model_meta.json"
INPUT_FILE = SCRIPT_DIR / "Data" / "new_plans.xlsx"
SHEET_NAME = "RadCalc"
OUTPUT_FILE = SCRIPT_DIR / "predictions.csv"

CLS_TARGETS = ("GPR3mm", "GPR2mm")
ID_COLUMNS = ("ID", "Plan", "Date")  # carried through to the output if present
ENERGY_COLUMN = "Energy"
# One-hot columns used in training; any other Energy value (e.g. 6F) is the
# reference category with all indicator columns at 0, matching step 3.
ENERGY_ONE_HOT = {"Energy_10": "10", "Energy_6": "6"}


def load_bundle() -> tuple[dict, dict]:
    if not META_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {META_FILE}. Run train_final_model.py first."
        )
    with META_FILE.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    models = {
        target: joblib.load(MODEL_DIR / meta["models"][target]["model_file"])
        for target in CLS_TARGETS
    }
    return meta, models


def normalize_energy(value: object) -> str:
    """Excel may deliver Energy as 6, 6.0, '6' or '6F'; compare as clean text."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def build_feature_frame(raw: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Reproduce the step-3 encoding for exactly the features the model needs."""
    frame = pd.DataFrame(index=raw.index)
    missing_columns = []
    for feature in feature_list:
        if feature in ENERGY_ONE_HOT:
            if ENERGY_COLUMN not in raw.columns:
                missing_columns.append(ENERGY_COLUMN)
                continue
            energies = raw[ENERGY_COLUMN].map(normalize_energy)
            frame[feature] = (energies == ENERGY_ONE_HOT[feature]).astype(int)
        elif feature in raw.columns:
            frame[feature] = pd.to_numeric(raw[feature], errors="coerce")
        else:
            missing_columns.append(feature)
    if missing_columns:
        raise ValueError(
            f"{INPUT_FILE.name} is missing required columns: "
            f"{', '.join(sorted(set(missing_columns)))}"
        )
    bad = frame.columns[frame.isna().any()].tolist()
    if bad:
        rows = sorted(frame.index[frame[bad].isna().any(axis=1)] + 2)  # +2: header row
        raise ValueError(
            f"Missing or non-numeric feature values in columns {', '.join(bad)} "
            f"(xlsx rows {', '.join(map(str, rows))}). Fill them in before predicting."
        )
    return frame


def out_of_range_features(row: pd.Series, ranges: dict[str, dict]) -> list[str]:
    flagged = []
    for feature, bounds in ranges.items():
        value = row[feature]
        if value < bounds["min"] or value > bounds["max"]:
            flagged.append(
                f"{feature}={value:g} outside [{bounds['min']:g}, {bounds['max']:g}]"
            )
    return flagged


def main() -> None:
    meta, models = load_bundle()
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. Copy the extractor output rows "
            f"into that workbook (sheet {SHEET_NAME}) first."
        )
    raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"{INPUT_FILE.name} contains no data rows.")
    print(f"Loaded {len(raw)} plan row(s) from {INPUT_FILE.name}.")

    upper = meta["decision_thresholds"]["class2"]
    lower = meta["decision_thresholds"]["class0"]
    class0_enabled = all(lower[t] is not None for t in CLS_TARGETS)
    if not class0_enabled:
        print("Note: no class-0 operating point was found in step 8; replan is disabled.")

    predictions = {}
    # dict keyed by warning text: deduplicates features shared by both models
    warnings_per_row: list[dict[str, None]] = [{} for _ in range(len(raw))]
    for target in CLS_TARGETS:
        model_meta = meta["models"][target]
        X = build_feature_frame(raw, model_meta["features"])
        predictions[target] = models[target].predict(X)
        for i, (_, row) in enumerate(X.iterrows()):
            for warning in out_of_range_features(row, model_meta["feature_ranges"]):
                warnings_per_row[i][warning] = None

    pred3 = np.asarray(predictions["GPR3mm"], dtype=float)
    pred2 = np.asarray(predictions["GPR2mm"], dtype=float)
    is_class2 = (pred3 >= upper["GPR3mm"]) & (pred2 >= upper["GPR2mm"])
    if class0_enabled:
        is_class0 = (pred3 < lower["GPR3mm"]) & (pred2 < lower["GPR2mm"])
    else:
        is_class0 = np.zeros(len(raw), dtype=bool)
    # With negative margins the class-2 and class-0 boxes can overlap; a plan
    # matching both is contradictory evidence and stays in manual review,
    # mirroring predicted_classes() in 8_operating_point.py.
    decisions = np.where(
        is_class2 & ~is_class0, 2, np.where(is_class0 & ~is_class2, 0, 1)
    )

    reasons = []
    for flag2, flag0 in zip(is_class2, is_class0):
        if flag2 and flag0:
            reasons.append(
                "manual review: predictions satisfy both the class-2 and the "
                "class-0 rule (contradictory evidence)"
            )
        elif flag2:
            reasons.append(
                f"auto-pass: pred GPR3mm >= {upper['GPR3mm']:.2f} and "
                f"pred GPR2mm >= {upper['GPR2mm']:.2f}"
            )
        elif flag0:
            reasons.append(
                f"replan: pred GPR3mm < {lower['GPR3mm']:.2f} and "
                f"pred GPR2mm < {lower['GPR2mm']:.2f}"
            )
        else:
            reasons.append("manual review: predictions between decision thresholds")

    output = pd.DataFrame()
    for column in ID_COLUMNS:
        if column in raw.columns:
            output[column] = raw[column]
    output["pred_GPR3mm"] = np.round(pred3, 2)
    output["pred_GPR2mm"] = np.round(pred2, 2)
    output["decision"] = decisions
    output["reason"] = reasons
    output["out_of_range_features"] = ["; ".join(w) for w in warnings_per_row]
    output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nDecisions: " + ", ".join(
        f"{cls}->{int((decisions == cls).sum())}" for cls in (2, 1, 0)
    ))
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(output.drop(columns=["reason"]).to_string(index=False))
    flagged_rows = sum(1 for w in warnings_per_row if w)
    if flagged_rows:
        print(
            f"\nWarning: {flagged_rows} row(s) have feature values outside the "
            "training range (see out_of_range_features) - the model extrapolates "
            "there, so review those plans manually regardless of the decision."
        )
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
