# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Uplift Modeling project on the Lenta retail dataset (BigTarget Hackathon, Лента + Microsoft, 2020). Binary classification task estimating the causal effect of marketing communications (treatment) on customer store visits/purchases (target: `response_att`).

**Current status:** EDA, preprocessing, baseline modeling, hyperparameter tuning, balance experiments, and advanced meta-learner comparison complete. Best model: **Class Transformation + LightGBM** (Qini AUC = 0.0752). See `reports/modeling_results.md` for detailed results and `reports/eda_summary.md` for EDA findings.

## Setup

```bash
pip install scikit-uplift pandas numpy matplotlib seaborn scipy scikit-learn lightgbm catboost causalml econml
```

After cloning (large CSV files tracked via Git LFS):
```bash
git lfs install && git lfs pull
```

## Running Notebooks

Execute via command line (VS Code kernel may use wrong Python):
```bash
python3 -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 eda.ipynb --output eda.ipynb
python3 -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 modeling.ipynb --output modeling.ipynb
```

## Key Files

Notebooks live at the repo root (no `src/` or `notebooks/` subdirs):
- `eda.ipynb` — EDA & preprocessing pipeline (stages 1–7 from `todo.md`)
- `modeling.ipynb` — Uplift model training, evaluation, tuning, balance experiments
- `data/processed/` — Train/test CSV splits + `preprocessing_pipeline.pkl`. X_train (~420MB) and X_test (~180MB) tracked via Git LFS
- `reports/modeling_results.md` — Full modeling results & analysis (in Russian)
- `reports/eda_summary.md` — Detailed EDA findings (in Russian)
- `columns_description.md` — Feature descriptions from dataset source
- `todo.md` — 7-stage project checklist with results (in Russian)

## Data Loading

Raw data: `from sklift.datasets import fetch_lenta` → returns `data` (features), `target`, `treatment`.

Load preprocessing pipeline (contains LabelEncoder, medians, scaler, lists of removed/skewed features):
```python
import pickle
with open('data/processed/preprocessing_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)
```

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
- Preprocessing: 51 features removed (|corr| > 0.8), winsorization 1–99%, median imputation, gender LabelEncoded

## Domain Notes

- **Never remove outlier observations** — every customer matters for uplift estimation. Use clipping/winsorization instead.
- **No post-treatment features** — all features measured before the marketing communication.
- High correlation with target does NOT imply usefulness for uplift. Look for **effect modifiers** (features where uplift differs across subgroups).
- Tree-based models (LightGBM, CatBoost) don't need feature scaling; only scale if using logistic regression as base learner.
- Key effect modifiers found in EDA: `age` (monotonic uplift growth with age), `gender`, `response_sms`/`response_viber`, `main_format`.
- `_missing` flags are informative signals (no purchases in period), not noise — keep them.

## Modeling Guidance

- Use scikit-uplift (`sklift`) for uplift approaches: `SoloModel` (S-learner), `TwoModels` (T-learner), `ClassTransformation`
- Best base learner: **LightGBM** (faster and slightly better than CatBoost for this task)
- Evaluation metrics: uplift@k, Qini AUC, Uplift AUC (NOT standard AUC/accuracy)
- **Do NOT use `ClassTransformationReg`** — it requires `propensity_val` parameter and produces near-zero Qini AUC on this dataset. Use `ClassTransformation` (classifier version) instead.
- **Do NOT balance treatment/control for Class Transformation** — the method has built-in IPW correction via the Z-transform formula (divides by propensity). Additional balancing (downsampling, oversampling, IPW weighting) causes double correction and degrades Qini AUC by ~60%. Balancing slightly helps T-Learner (+33%) but its absolute performance remains 4x worse than CT.
- Class Transformation predicted uplift values are often negative — this is expected; only the **ranking** matters, not absolute values
- Top features: `response_sms` (26.07), `response_viber` (20.59) dominate by a large margin. They have slight imbalance between treatment/control — account for in interpretation.

## Best Model Configuration

**Class Transformation + LGBMClassifier** — Qini AUC = **0.0752** (+3.1% over CatBoost baseline):
```python
LGBMClassifier(n_estimators=500, max_depth=6, num_leaves=31, learning_rate=0.05, random_state=42, verbose=-1)
```

Class Transformation dramatically outperforms S-Learner and T-Learner (~5x better Qini AUC) because it directly optimizes uplift ranking rather than estimating conditional outcomes.

## Completed Experiments (beyond baseline)

### Train vs Test overfitting (modeling.ipynb section 5.1)
T-Learner massively overfits: Qini AUC Train=0.278 vs Test=0.007 (42× gap). Class Transformation is most robust: Train=0.100 vs Test=0.073 (1.37× gap). Results in `reports/modeling_results.md` section 10.

### SMOTE generator for control group balancing (modeling.ipynb section 10)
Implemented `ControlGroupSMOTE` — random-pair interpolation (SMOTE without KNN, O(n×d)). Generated 241K synthetic control observations to achieve 50/50 balance. **Results: S-Learner −1%, T-Learner −41%.** Synthetic data hurts because test set is real-only. Full analysis in `reports/modeling_results.md` section 11.

**Confirmed: no balancing strategy improves results. Original 75/25 + Class Transformation is optimal.**

### Advanced meta-learners: X-Learner, DR-Learner, R-Learner, CausalForestDML (modeling.ipynb section 11)
Libraries: `causalml` 0.16.0, `econml` 0.16.0. **All dramatically worse than CT baseline (~90% lower Qini AUC):**
- DR-Learner: 0.0073, R-Learner: 0.0072, X-Learner: 0.0060, CausalForestDML (80K subsample): 0.0048
- Root cause: MSE-optimizing CATE estimators ≠ good rankers for Qini AUC. CT's Z-transform binary classifier directly optimizes ranking. See `reports/modeling_results.md` section 12.
- causalml API notes: `BaseDRLearner` takes no `propensity_learner` arg — pass `p=` to `fit()` and `predict()` instead. `BaseXRegressor.predict()` requires `p=` kwarg.
- CausalForestDML on 480K×244 exceeds 900s per-cell timeout — use 80K subsample for notebook execution.

## Repository

GitHub: `Xlemon5/lenta-uplift-modeling` (master branch).

## Library Documentation

Use the Context7 MCP tool (`resolve-library-id` → `query-docs`) to look up documentation for scikit-uplift, scikit-learn, LightGBM, CatBoost, and other libraries used in this project.
