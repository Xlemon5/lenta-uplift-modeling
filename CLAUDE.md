# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Uplift Modeling project on the Lenta retail dataset (BigTarget Hackathon, Лента + Microsoft, 2020). Binary classification task estimating the causal effect of marketing communications (treatment) on customer store visits/purchases (target: `response_att`).

**Current status:** EDA, preprocessing, baseline modeling, and hyperparameter tuning complete. Best model: **Class Transformation + LightGBM** (Qini AUC = 0.0752). See `reports/modeling_results.md` for detailed results.

## Setup

```bash
pip install scikit-uplift pandas numpy matplotlib seaborn scipy scikit-learn lightgbm catboost
```

## Project Structure

```
├── eda.ipynb                    # EDA & preprocessing (stages 1–7 from todo.md)
├── modeling.ipynb               # Uplift model training, evaluation, tuning
├── data/
│   └── processed/               # Train/test splits (CSV) + preprocessing pipeline (pkl)
│       ├── X_train.csv          # 480,920 × 244 features
│       ├── X_test.csv           # 206,109 × 244 features
│       ├── y_train.csv / y_test.csv
│       ├── treatment_train.csv / treatment_test.csv  (strings 'test'/'control')
│       └── preprocessing_pipeline.pkl
├── reports/
│   ├── eda_summary.md           # Detailed EDA findings (in Russian)
│   └── modeling_results.md      # Modeling results & analysis (in Russian)
├── columns_description.md       # Feature descriptions from dataset source
├── todo.md                      # 7-stage project checklist with results (in Russian)
└── catboost_info/               # CatBoost training logs (auto-generated, not tracked)
```

## Data Loading

Raw data: `from sklift.datasets import fetch_lenta` → returns `data` (features), `target`, `treatment`.

Processed data (for modeling):
```python
import pandas as pd
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
treatment_train = pd.read_csv('data/processed/treatment_train.csv').squeeze()
treatment_test = pd.read_csv('data/processed/treatment_test.csv').squeeze()

# IMPORTANT: treatment files contain strings — convert to int
treatment_train = (treatment_train == 'test').astype(int)
treatment_test = (treatment_test == 'test').astype(int)
```

## sklift Compatibility Fix

sklearn 1.8+ moved `check_matplotlib_support`. Apply this patch before importing `sklift.viz`:
```python
import sklearn.utils
from sklearn.utils._plotting import check_matplotlib_support
sklearn.utils.check_matplotlib_support = check_matplotlib_support
```

## Key Dataset Facts

- 687,029 customers, 193 raw features → 244 after preprocessing (142 original + 102 `_missing` flags)
- Treatment/control split: 75.1% / 24.9% (strings `'test'`/`'control'` in raw)
- Overall uplift: +0.75 п.п. (CR treatment=11.01%, CR control=10.26%)
- Train/test: 70/30 stratified by treatment × target, `random_state=42`
- 51 features removed (|corr| > 0.8), winsorization 1–99%, median imputation, gender LabelEncoded

## Domain Notes

- **Never remove outlier observations** — every customer matters for uplift estimation. Use clipping/winsorization instead.
- **No post-treatment features** — all features measured before the marketing communication.
- High correlation with target does NOT imply usefulness for uplift. Look for **effect modifiers** (features where uplift differs across subgroups).
- Tree-based models (LightGBM, CatBoost) don't need feature scaling; only scale if using logistic regression as base learner.
- Key effect modifiers found in EDA: `age` (monotonic uplift growth with age), `gender`, `response_sms`/`response_viber`, `main_format`.
- `_missing` flags are informative signals (no purchases in period), not noise — keep them.

## Modeling Results Summary

### Baseline (5 approaches, CatBoost `iterations=200, depth=6, lr=0.1`):

| Model | Qini AUC |
|-------|----------|
| Class Transformation | **0.0730** |
| S-Learner (interaction) | 0.0148 |
| T-Learner (DDR) | 0.0144 |
| S-Learner (dummy) | 0.0100 |
| T-Learner (vanilla) | 0.0066 |

### After tuning (best configurations):

| Model | Qini AUC | Improvement |
|-------|----------|-------------|
| **CT + LightGBM** (d=6, leaves=31, iter=500) | **0.0752** | **+3.1%** |
| CT + LightGBM (d=4, leaves=15, iter=500) | 0.0743 | +1.8% |
| CT + CatBoost (iter=500, lr=0.05, d=6) | 0.0741 | +1.5% |

Top features: `response_sms` (26.07), `response_viber` (20.59) dominate by a large margin.

## Modeling Guidance

- Use scikit-uplift (`sklift`) for uplift approaches: `SoloModel` (S-learner), `TwoModels` (T-learner), `ClassTransformation`
- Best base learner: **LightGBM** (faster and slightly better than CatBoost for this task)
- Evaluation metrics: uplift@k, Qini AUC, Uplift AUC (NOT standard AUC/accuracy)
- **Do NOT use `ClassTransformationReg`** — it requires `propensity_val` parameter and produces near-zero Qini AUC on this dataset. Use `ClassTransformation` (classifier version) instead.
- Class Transformation predicted uplift values are often negative — this is expected; only the **ranking** matters, not absolute values
- `response_sms`/`response_viber` have slight imbalance between treatment/control — account for in interpretation

## Running Notebooks

Execute via command line (VS Code kernel may use wrong Python):
```bash
python3 -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 eda.ipynb --output eda.ipynb
python3 -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 modeling.ipynb --output modeling.ipynb
```

## Library Documentation

Use the Context7 MCP tool (`resolve-library-id` → `query-docs`) to look up documentation for scikit-uplift, scikit-learn, LightGBM, CatBoost, and other libraries used in this project.
