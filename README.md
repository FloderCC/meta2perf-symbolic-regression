### Repository of the work entitled “Predicting ML Performance from Meta-Features: A Closed-Form Symbolic Regression Model”

This project builds a **meta-dataset** that links:
- dataset **meta-features** (extracted with `pymfe`),
- model **descriptors / type**,
- and the **observed performance** (MCC, etc.) of several ML algorithms,

and then learns **symbolic regression (SR)** models and **black-box baselines** to infer performance directly from those descriptors.

---

## 1. Repository Structure

The current repository looks like this:

```text
.
├── README.md
└── src/
    ├── datasets/                 # per-dataset READMEs (no raw CSVs tracked here)
    ├── logs/                     # logs of the SR and baseline runs
    ├── plots/                    # PDF/PNG plots for relevance and regression
    ├── results/                  # CSVs produced by the experiment stages
    │   ├── meta_dataset.csv
    │   ├── results_stage_dataset_description.csv
    │   ├── results_stage_ml_eval.csv
    │   └── results_stage_ml_tune.csv
    ├── utils/
    │   ├── dataset_utils.py
    │   └── model_utils.py
    ├── exp_setup.py
    ├── exp_stage_ml_tune.py
    ├── exp_stage_ml_eval.py
    ├── exp_stage_dataset_description.py
    ├── exp_stage_create_meta_dataset.py
    ├── exp_stage_create_sr_f.py
    ├── exp_stage_create_sr_f_s.py
    ├── exp_stage_create_f_bm.py
    ├── exp_stage_create_f_s_bm.py
    ├── explain_best_bm.py
    ├── plot_bm_f_relevance.py
    ├── plot_correlations.py
    ├── plot_sr_f.py
    └── plot_sr_f_s.py
```


### Files in `src/`

- `exp_setup.py` — Central experiment config: dataset list (and columns to drop/target), model list, seeds, sample-size percentages, and test size.
- `exp_stage_ml_tune.py` — Hyperparameter tuning for every (dataset, sample size, model, seed); writes best params to results.
- `exp_stage_ml_eval.py` — Trains and evaluates the tuned models; logs metrics (e.g., MCC, accuracy, precision, recall, F1).
- `exp_stage_dataset_description.py` — Builds dataset meta-features with `pymfe` for each dataset/sample split.
- `exp_stage_create_meta_dataset.py` — Joins outputs from previous stages and adds model descriptions → `results/meta_dataset.csv`.
- `exp_stage_create_sr_f.py` — Trains symbolic regression using the meta-dataset.
- `exp_stage_create_sr_f_s.py` — Trains symbolic regression using the normalized meta-dataset.
- `exp_stage_create_f_bm.py` — Trains black-box baseline regressors (e.g., RF/GB/MLP) on the meta-dataset for comparison.
- `explain_best_bm.py` — Runs SHAP on the best baseline to obtain/globalize feature relevance.
- `plot_bm_f_relvance.py` — Plots SHAP feature relevance bars for the baseline model.
- `plot_correlations.py` — Computes and plots Pearson/Spearman correlations of meta-features with the target.
- `plot_sr_f.py` — Plots SR results for the meta-dataset
- `plot_sr_f_s.py` — Plots SR results for the normalized meta-dataset

#### Utilities (`src/utils/`)
- `dataset_utils.py` — Sampling, preprocessing (NaNs, encoding), dataset size limiting, and `pymfe` meta-feature extraction helpers.
- `model_utils.py` — Model factory, grid search wrappers, scorers (e.g., MCC), seeding/reset helpers, and training of best configurations.

#### Folders
- `datasets/` — Per-dataset notes/READMEs.
- `results/` — CSV outputs for each stage.
- `logs/` — TXT logs for SR/baselines/SHAP runs.
- `plots/` — Saved figures (PDF/PNG) produced by the plot scripts.


## Complementary information (paper-aligned)

To keep the main README short, the complementary information from the paper are in separate files:

- [Datasets used in this study](src/docs/datasets.md)
- [Hyperparameter space for the ML models](src/docs/ml-hyperparams.md)
- [Model descriptors and complexity (proc_units, train/inf complexity, encodings)](src/docs/model-descriptors.md)
- [Hyperparameter space for the black-box regressors](src/docs/baselines-hyperparams.md)
