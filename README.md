# PSQA Prediction: VMAT Plan-Complexity-Based QA Outcome Prediction

Source code and selected results accompanying an MSc dissertation in Data Science and Machine Learning at UCL. The project predicts patient-specific quality assurance (PSQA) gamma passing rates from VMAT treatment-plan complexity metrics.

The pipeline extracts features from RadCalc QA reports and TPS DICOM files, compares regression models for GPR 3%/3 mm, 2%/2 mm and 1%/1 mm, and combines the first two targets into research triage recommendations:

| Code | Recommendation | Decision rule |
| --- | --- | --- |
| `2` | Auto-pass | Both predictions meet or exceed their adjusted upper thresholds, without also satisfying the replan rule. |
| `1` | Manual review | Neither extreme rule is satisfied, or both are satisfied. |
| `0` | Replan | Both predictions fall below their adjusted lower thresholds, without also satisfying the auto-pass rule. |

Step 8 selects the model pair and residual-quantile margins. Final-model training saves these thresholds with the fitted models for use when predicting new plans.

## Repository contents

```text
Data_deidentification/
  deidentify_patient_files.py   PDF and DICOM pseudonymisation helper

Data_extract_v7/
  extract_data_v7.py            Main extraction tool with file-selection dialogs
  extract_data_v6.py            Previous version retained for reference
  xlsx_columns_explain.txt      Definitions, units and sources for all 43 columns

Model/
  Data/
    training.xlsx              Placeholder for the training workbook
    new_plans.xlsx             Placeholder for the new-plan workbook
  0_remove_columns.py          Remove identifiers and label-leakage columns
  1_remove_missing_rows.py     Remove rows with missing values
  2_count_site_energy.py       Dataset counts, GPR distributions and class balance
  3_one_hot_encoding.py        Encode Site and Energy categories
  4_feature_importance.py      Spearman, RF permutation and Elastic Net rankings
  5_feature_selection.py       Combine rankings and filter redundant features
  6_model_training.py          Benchmark regressors with nested cross-validation
  7_model_comparison.py        Comparison tables and plots from step-6 results
  8_operating_point.py         Model-pair and residual-quantile margin search
  9_selection_robustness.py    Repeat feature selection within each outer fold
  10_site_error_analysis.py    Post-hoc analysis of residuals by treatment site
  11_roc_analysis.py           ROC curves and AUCs for the two extreme decisions
  train_final_model.py         Retune on all training rows and save model bundles
  predict_new_plan.py          Predict GPRs and apply the saved decision rule
```

The [column reference](Data_extract_v7/xlsx_columns_explain.txt) documents report fields, complexity metrics and the implemented DICOM conventions, including control-point carry-forward, MU weighting, jaw clipping and multi-arc aggregation.

Selected results are included for inspection without rerunning the models:

| Directory under `Model/` | Included result files |
| --- | --- |
| `6_model_training/` | `summary.csv`, `fold_results.csv` |
| `8_operating_point/` | `margin_search.csv`, `operating_point.json` |
| `9_selection_robustness/` | `fold_details.csv` |
| `11_roc_analysis/` | `roc_auc.csv`, `roc_points.csv`, `roc_curves.pdf` |

The benchmark summary contains mean and fold-level standard deviation of MAE, RMSE and R-squared for 72 configurations: six regressors, four feature-set sizes and three GPR targets.

## Requirements

The project was developed with Python 3.12. Install the dependencies in your Python environment:

```bash
python -m pip install numpy pandas scikit-learn scipy matplotlib seaborn xgboost joblib openpyxl pdfplumber pydicom PyPDF2
```

The extraction and pseudonymisation tools also require `tkinter` and a desktop session for file-selection dialogs. The commands below are run from the repository root. Scripts accept no command-line arguments and resolve their input and output paths relative to their own directories.

## Running the pipeline

### 1. Prepare and extract source files

Prepare pseudonymised RadCalc reports, TPS RTPLAN files and RT Structure Sets using the approved procedure in your clinical environment. The helper can be run with:

```bash
python Data_deidentification/deidentify_patient_files.py
```

It writes separate copies with the `_MOCK` suffix. For PDFs, it retains only the matched calculation-summary page; it does not reproduce the complete historical preparation of multipage reports. Retain the report pages needed to extract measured gamma passing rates when preparing the extraction inputs.

Run the extractor and select the prepared reports and matching DICOM files:

```bash
python Data_extract_v7/extract_data_v7.py
```

The output is `Data_extract_v7/data.xlsx`, with one row per report on the `RadCalc` worksheet. The parsers were developed for RadCalc report layouts and Elekta Monaco DICOM exports; other formats may require parser changes.

### 2. Prepare the training workbook

Copy the extractor's `RadCalc` worksheet, including its 43 column headers and data rows, into `Model/Data/training.xlsx`. The worksheet must be named exactly `RadCalc`; create or rename it if necessary. Preserve the column names documented in the column reference.

The distributed Excel files are empty placeholders. Supply your own extracted data before running the modelling scripts.

### 3. Run the numbered scripts

Run steps **0 through 11 in numerical order**:

```bash
python Model/0_remove_columns.py
python Model/1_remove_missing_rows.py
python Model/2_count_site_energy.py
python Model/3_one_hot_encoding.py
python Model/4_feature_importance.py
python Model/5_feature_selection.py
python Model/6_model_training.py
python Model/7_model_comparison.py
python Model/8_operating_point.py
python Model/9_selection_robustness.py
python Model/10_site_error_analysis.py
python Model/11_roc_analysis.py
```

Each script reads the earlier outputs it needs using fixed filenames. Steps 2, 7, 9, 10 and 11 are auxiliary analyses. Generated figures are saved as vector PDFs.

Step 6 uses five outer folds repeated five times, with five-fold inner grid search. It is the most computationally expensive step. Step 8 writes the chosen configuration and margins to `Model/8_operating_point/operating_point.json`; its measured-versus-predicted scatter plots use equal axis scales and an identity line. Step 11 reports ROC discrimination without changing the operating point.

### 4. Train the final models

```bash
python Model/train_final_model.py
```

This retunes the selected GPR3mm and GPR2mm models on all training rows and saves them under `Model/final_model/`, together with metadata containing feature lists, decision thresholds, training ranges and library versions. The step-8 margins are retained.

### 5. Predict new plans

Extract the new plans and copy their worksheet, including the column headers, into `Model/Data/new_plans.xlsx`. Use the worksheet name `RadCalc`. Measured GPR columns may be empty for prediction; required predictor values must be present.

```bash
python Model/predict_new_plan.py
```

The output, `Model/predictions.csv`, contains identifiers carried through from the input, predicted GPR3mm and GPR2mm, the recommendation code, its reason, and any feature-range warnings.

Predicted GPRs are capped at **100%** before classification and output. Decisions use the capped values before rounding; the CSV displays predictions to two decimal places. Features outside the training minimum-to-maximum range are flagged. These flags do not automatically change the recommendation code, and the script advises manual review of flagged plans.

## Data availability and reproducibility

The public repository contains source code, empty workbook placeholders and selected aggregate results. The clinical source files, populated study workbooks, per-record prediction outputs and fitted deployment models are not distributed. Running the complete workflow requires your own prepared data; the saved result tables can be inspected directly.

The modelling workflow uses a fixed random seed of 42. Dependencies are not pinned, so reproducing the dissertation's exact numerical results also requires the original study dataset and matching library versions. Running the scripts on another dataset regenerates results for that dataset and may overwrite the supplied result files.

Feature ranking and selection in steps 4 and 5 use the full development dataset, which can make the selected-feature benchmark scores optimistic. Step 9 assesses this effect by repeating selection within the outer folds. The operating-point search and ROC analysis also reuse development predictions; the dissertation discusses the resulting evaluation limitations.
