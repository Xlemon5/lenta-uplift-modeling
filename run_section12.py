"""Standalone script: CT-guided SMOTE — top-10% control as template for full dataset."""
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test  = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()
treatment_train = pd.read_csv('data/processed/treatment_train.csv').squeeze()
treatment_test  = pd.read_csv('data/processed/treatment_test.csv').squeeze()

treatment_train = (treatment_train == 'test').astype(int)
treatment_test  = (treatment_test  == 'test').astype(int)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"T/C: {treatment_train.mean():.4f} / {1-treatment_train.mean():.4f}")

# ── Imports ───────────────────────────────────────────────────────────────────
from lightgbm import LGBMClassifier
from sklift.models import ClassTransformation, SoloModel, TwoModels
from sklift.metrics import qini_auc_score, uplift_auc_score

def asd_score(y_true, uplift_scores, treatment, n_bins=10):
    df = pd.DataFrame({'y': y_true, 'score': uplift_scores, 't': treatment})
    df['bin'] = pd.qcut(df['score'].rank(method='first'), n_bins, labels=False)
    deviations = []
    for b in range(n_bins):
        sub = df[df['bin'] == b]
        if sub['t'].nunique() < 2:
            continue
        actual    = sub.loc[sub['t']==1,'y'].mean() - sub.loc[sub['t']==0,'y'].mean()
        predicted = sub['score'].mean()
        deviations.append((actual - predicted) ** 2)
    return float(np.mean(deviations)) if deviations else np.nan

# ── ControlGroupSMOTE ─────────────────────────────────────────────────────────
class ControlGroupSMOTE:
    def __init__(self, random_state=42):
        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y):
        self.X_ = X.values if hasattr(X, 'values') else X
        self.y_ = y.values if hasattr(y, 'values') else y
        self.binary_cols_ = [
            j for j in range(self.X_.shape[1])
            if np.all(np.isin(self.X_[:, j], [0, 1]))
        ]
        self.columns_ = X.columns if hasattr(X, 'columns') else None
        return self

    def generate(self, n_samples):
        n = len(self.X_)
        idx_a = self.rng.randint(0, n, size=n_samples)
        idx_b = self.rng.randint(0, n, size=n_samples)
        alpha = self.rng.uniform(0, 1, size=(n_samples, 1))
        X_syn = self.X_[idx_a] + alpha * (self.X_[idx_b] - self.X_[idx_a])
        for j in self.binary_cols_:
            X_syn[:, j] = np.round(X_syn[:, j]).astype(int)
        y_syn = np.where(alpha.flatten() <= 0.5, self.y_[idx_a], self.y_[idx_b])
        cols = self.columns_ if self.columns_ is not None else range(self.X_.shape[1])
        return pd.DataFrame(X_syn, columns=cols), pd.Series(y_syn)

# ── Step 1: Fit CT and identify top-10% control observations ─────────────────
print("\nFitting CT+LGB to score training data...")
t0 = time.time()
ct_best = ClassTransformation(LGBMClassifier(
    n_estimators=500, max_depth=6, num_leaves=31,
    learning_rate=0.05, random_state=42, verbose=-1
))
ct_best.fit(X_train, y_train, treatment_train)
ct_train_scores = ct_best.predict(X_train)
print(f"  done in {time.time()-t0:.1f}s")

threshold = np.percentile(ct_train_scores, 90)
top10_mask = ct_train_scores >= threshold

# Control observations in top-10% — these are our SMOTE template
ctrl_mask    = treatment_train == 0
top10_ctrl_mask = top10_mask & ctrl_mask

X_ctrl_top10 = X_train[top10_ctrl_mask].reset_index(drop=True)
y_ctrl_top10 = y_train[top10_ctrl_mask].reset_index(drop=True)

# Full control pool (for comparison)
X_ctrl_full  = X_train[ctrl_mask].reset_index(drop=True)
y_ctrl_full  = y_train[ctrl_mask].reset_index(drop=True)

X_trt        = X_train[~ctrl_mask].reset_index(drop=True)
y_trt        = y_train[~ctrl_mask].reset_index(drop=True)

n_generate   = len(X_trt) - len(X_ctrl_full)  # ~241K to reach 50/50

print(f"\nFull control pool:         {len(X_ctrl_full):,} rows, CR={y_ctrl_full.mean():.4f}")
print(f"Top-10% control template:  {len(X_ctrl_top10):,} rows, CR={y_ctrl_top10.mean():.4f}")
print(f"Treatment:                 {len(X_trt):,} rows")
print(f"To generate (50/50 full):  {n_generate:,} rows")

# ── Step 2: SMOTE from top-10% control template ───────────────────────────────
print("\nGenerating synthetic control from top-10% template...")
t0 = time.time()
smote_top10 = ControlGroupSMOTE(random_state=42)
smote_top10.fit(X_ctrl_top10, y_ctrl_top10)
X_syn_top10, y_syn_top10 = smote_top10.generate(n_generate)
print(f"  Generated {len(X_syn_top10):,} rows in {time.time()-t0:.1f}s")
print(f"  Synthetic CR: {y_syn_top10.mean():.4f} (source CR: {y_ctrl_top10.mean():.4f})")

# Build augmented dataset: full real train + synthetic top-10% control
t_syn_ser = pd.Series([0] * n_generate, name='treatment')

X_aug = pd.concat([X_train, X_syn_top10], ignore_index=True)
y_aug = pd.concat([y_train, y_syn_top10], ignore_index=True)
t_aug = pd.concat([treatment_train, t_syn_ser], ignore_index=True)

shuffle_idx = np.random.RandomState(42).permutation(len(X_aug))
X_aug = X_aug.iloc[shuffle_idx].reset_index(drop=True)
y_aug = y_aug.iloc[shuffle_idx].reset_index(drop=True)
t_aug = t_aug.iloc[shuffle_idx].reset_index(drop=True)

print(f"\nAugmented dataset: {len(X_aug):,} rows, T rate: {t_aug.mean():.4f}")

# ── Step 3: Also build Section 11 style dataset (SMOTE from full control) ─────
print("\nGenerating Section 11 baseline (SMOTE from full control)...")
t0 = time.time()
smote_full = ControlGroupSMOTE(random_state=42)
smote_full.fit(X_ctrl_full, y_ctrl_full)
X_syn_full, y_syn_full = smote_full.generate(n_generate)
print(f"  Generated {len(X_syn_full):,} rows in {time.time()-t0:.1f}s")
print(f"  Synthetic CR: {y_syn_full.mean():.4f} (source CR: {y_ctrl_full.mean():.4f})")

t_syn_ser2 = pd.Series([0] * n_generate, name='treatment')
X_aug_full = pd.concat([X_train, X_syn_full], ignore_index=True)
y_aug_full = pd.concat([y_train, y_syn_full], ignore_index=True)
t_aug_full = pd.concat([treatment_train, t_syn_ser2], ignore_index=True)

shuffle_idx2 = np.random.RandomState(42).permutation(len(X_aug_full))
X_aug_full = X_aug_full.iloc[shuffle_idx2].reset_index(drop=True)
y_aug_full = y_aug_full.iloc[shuffle_idx2].reset_index(drop=True)
t_aug_full = t_aug_full.iloc[shuffle_idx2].reset_index(drop=True)

# ── Step 4: Train and evaluate ────────────────────────────────────────────────
lgbm_params = dict(n_estimators=500, max_depth=6, num_leaves=31,
                   learning_rate=0.05, random_state=42, verbose=-1)

configs = [
    ('S-Learner (original 75/25)',       X_train,    y_train,    treatment_train),
    ('S-Learner (SMOTE full ctrl)',       X_aug_full, y_aug_full, t_aug_full),
    ('S-Learner (SMOTE top-10% ctrl)',   X_aug,      y_aug,      t_aug),
    ('T-Learner (original 75/25)',       X_train,    y_train,    treatment_train),
    ('T-Learner (SMOTE full ctrl)',       X_aug_full, y_aug_full, t_aug_full),
    ('T-Learner (SMOTE top-10% ctrl)',   X_aug,      y_aug,      t_aug),
]

results = []
for label, X_tr, y_tr, t_tr in configs:
    t0 = time.time()
    if label.startswith('S'):
        m = SoloModel(LGBMClassifier(**lgbm_params), method='dummy')
    else:
        m = TwoModels(LGBMClassifier(**lgbm_params), LGBMClassifier(**lgbm_params), method='vanilla')
    m.fit(X_tr, y_tr, t_tr)
    pred = m.predict(X_test)
    elapsed = time.time() - t0
    row = {
        'Model':      label,
        'Qini AUC':   qini_auc_score(y_test, pred, treatment_test),
        'Uplift AUC': uplift_auc_score(y_test, pred, treatment_test),
        'ASD':        asd_score(y_test, pred, treatment_test),
        'time_s':     round(elapsed, 1),
    }
    results.append(row)
    print(f"{label}: Qini={row['Qini AUC']:.4f}  Uplift={row['Uplift AUC']:.4f}  ASD={row['ASD']:.6f}  ({elapsed:.1f}s)")

# ── Step 5: Summary ───────────────────────────────────────────────────────────
df = pd.DataFrame(results).set_index('Model')
print('\n=== CT-guided SMOTE (top-10% ctrl as template) ===')
print(df.to_string(float_format='{:.6f}'.format))

print('\n=== Delta vs original ===')
for base, comp in [
    ('S-Learner (original 75/25)', 'S-Learner (SMOTE full ctrl)'),
    ('S-Learner (original 75/25)', 'S-Learner (SMOTE top-10% ctrl)'),
    ('T-Learner (original 75/25)', 'T-Learner (SMOTE full ctrl)'),
    ('T-Learner (original 75/25)', 'T-Learner (SMOTE top-10% ctrl)'),
]:
    orig = df.loc[base, 'Qini AUC']
    new  = df.loc[comp, 'Qini AUC']
    print(f"  {comp}: {orig:.4f} → {new:.4f}  ({(new-orig)/orig*100:+.1f}%)")
