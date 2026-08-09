# PSQA Prediction: VMAT Plan-Complexity-Based QA Outcome Prediction

Predicting patient-specific QA (PSQA) outcomes for VMAT radiotherapy plans — gamma pass rates at three criteria (GPR 3%/3mm, 2%/2mm, 1%/1mm) — from treatment-plan complexity metrics, so that likely-passing plans can be triaged before measurement. Developed as part of an MSc dissertation in Data Science and Machine Learning at UCL.

The pipeline extracts complexity metrics (MCSv, AAV, LSV, SAS, EM, PI, PA, AFW, LT, PMU, …) from TPS RTPLAN DICOM files, pairs them with measured gamma pass rates parsed from RadCalc QA reports, trains and compares regression models under nested cross-validation, and derives a three-way clinical operating point:

- **2 — auto-pass**: predicted GPRs clear the class-2 thresholds; no manual review needed
- **1 — manual review**: everything in between
- **0 — replan**: predicted GPRs fall below the class-0 thresholds

## Repository structure

```
Data_deidentification/
  deidentify_patient_files.py   De-identify RadCalc PDFs and DICOM files before any
                                further processing (GUI, writes mock copies to Mock/)

Data_extract_v7/
  extract_data_v7.py            Main extraction tool (GUI): parses GPRs from RadCalc
                                PDFs, computes complexity metrics from RTPLAN DICOM
                                control points and PTV/CTV volumes from RT Structure
                                Sets, writes everything to data.xlsx
  extract_data_v6.py            Previous version, kept for reference

Model/
  Data/
    training.xlsx               Empty template: paste extractor output rows here
    new_plans.xlsx              Empty template: rows for plans to be predicted
  0_remove_columns.py           Drop identifier and label-leakage columns
  1_remove_missing_rows.py      Drop rows with missing values
  2_count_site_energy.py        Descriptive statistics (site/energy counts, GPR
                                distributions, class balance)
  3_one_hot_encoding.py         One-hot encode Site and Energy
  4_feature_importance.py       Spearman correlations, RF permutation importance,
                                Elastic Net coefficients (per GPR target)
  5_feature_selection.py        Average the three rankings, greedy redundancy filter
  6_model_training.py           Nested CV (RepeatedKFold 5x5 outer, GridSearchCV
                                inner) over 6 models x feature counts
  7_model_comparison.py         Comparison plots/tables from the step-6 results
  8_operating_point.py          Turn predicted GPRs into the 2/1/0 decision;
                                residual-quantile margin search
  9_selection_robustness.py     Re-run feature selection inside each fold to
                                quantify selection bias
  10_site_error_analysis.py     Post-hoc per-site residual analysis
  train_final_model.py          Retune the chosen models on all rows, save them
  predict_new_plan.py           Predict and classify new plans with the saved models
```

## Requirements

Python 3.12. Install the dependencies with:

```
pip install numpy pandas scikit-learn scipy matplotlib seaborn xgboost joblib openpyxl pdfplumber pydicom PyPDF2
```

## How to run

Every script is run directly (`python <script>.py`), takes no command-line arguments, and reads/writes files relative to its own directory.

1. **De-identify** (if working with identifiable data): run `Data_deidentification/deidentify_patient_files.py` and select the Structure Set / RTPLAN / RadCalc PDF files in the dialogs. Mock copies are written to a `Mock/` folder.
2. **Extract**: run `Data_extract_v7/extract_data_v7.py` and select the RadCalc PDFs, the matching TPS RTPLAN DICOM files, and the RT Structure Sets. Results accumulate in `data.xlsx` next to the script.
3. **Model**: copy the extracted rows into `Model/Data/training.xlsx` (sheet `RadCalc`), then run the numbered scripts **in order 0 → 10** — each step reads the previous step's output by fixed filename. Finally run `train_final_model.py` to fit and save the deployment models.
4. **Predict new plans**: put extractor-format rows for new plans into `Model/Data/new_plans.xlsx` (GPR columns may be empty) and run `predict_new_plan.py`. Output: `Model/predictions.csv` with predicted GPRs, the 2/1/0 decision, and out-of-training-range feature warnings.

## Data availability

**This repository contains no patient data.** `Model/Data/training.xlsx` and `Model/Data/new_plans.xlsx` are empty templates; the clinical dataset used in the dissertation cannot be shared. To reproduce the pipeline, supply your own institutional RadCalc reports and DICOM files (de-identified with the included tool or your local procedure).

## Notes

- The extraction tools are GUI-driven (tkinter file dialogs) and were built against RadCalc PDF report layouts and Elekta Monaco RTPLAN exports; other TPS/QA-software combinations may need parser adjustments.
- Step 6 is by far the slowest step (nested cross-validation over all models and feature counts); expect a long runtime.
- All figures are written as vector PDF.
