"""Standalone script: CT-guided SMOTE on top-10% experiment (Section 12)."""
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

# ── Imports ───────────────────────────────────────────────────────────────────
from lightgbm import LGBMClassifier
from sklift.models import ClassTransformation, SoloModel, TwoModels
from sklift.metrics import qini_auc_score, uplift_auc_score

# ASD helper (same as notebook)
def asd_score(y_true, uplift_scores, treatment, n_bins=10):
    df = pd.DataFrame({'y': y_true, 'score': uplift_scores, 't': treatment})
    df['bin'] = pd.qcut(df['score'].rank(method='first'), n_bins, labels=False)
    deviations = []
    for b in range(n_bins):
        mask = df['bin'] == b
        sub  = df[mask]
        if sub['t'].nunique() < 2:
            continue
        actual_uplift    = sub.loc[sub['t']==1,'y'].mean() - sub.loc[sub['t']==0,'y'].mean()
        predicted_uplift = sub['score'].mean()
        deviations.append((actual_uplift - predicted_uplift) ** 2)
    return float(np.mean(deviations)) if deviations else np.nan

# ── ControlGroupSMOTE (same as Section 10) ───────────────────────────────────
class ControlGroupSMOTE:
    def __init__(self, random_state=42):
        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y):
        self.X_ = X.values if hasattr(X, 'values') else X
        self.y_ = y.values if hasattr(y, 'values') else y
        # Identify binary columns (0/1 only)
        self.binary_cols_ = [
            j for j in range(self.X_.shape[1])
            if np.all(np.isin(self.X_[:, j], [0, 1]))
        ]
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
        cols = pd.RangeIndex(self.X_.shape[1])
        return pd.DataFrame(X_syn, columns=cols), pd.Series(y_syn)

# ── Step 1: Fit CT and score training data ────────────────────────────────────
print("\nFitting CT+LGB to score training data...")
t0 = time.time()
ct_best = ClassTransformation(LGBMClassifier(
    n_estimators=500, max_depth=6, num_leaves=31,
    learning_rate=0.05, random_state=42, verbose=-1
))
ct_best.fit(X_train, y_train, treatment_train)
ct_train_scores = ct_best.predict(X_train)
print(f"  CT fit+predict done in {time.time()-t0:.1f}s")

# ── Step 2: Filter top 10% ────────────────────────────────────────────────────
threshold = np.percentile(ct_train_scores, 90)
top10_mask = ct_train_scores >= threshold

X_top = X_train[top10_mask].reset_index(drop=True)
y_top = y_train[top10_mask].reset_index(drop=True)
t_top = treatment_train[top10_mask].reset_index(drop=True)

print(f"\nTop 10% subset: {top10_mask.sum():,} rows")
print(f"T/C ratio: {t_top.mean():.4f} / {1-t_top.mean():.4f}")
print(f"CR: {y_top.mean():.4f}")

# ── Step 3: SMOTE within top-10% ──────────────────────────────────────────────
ctrl_mask_top = t_top == 0
X_ctrl_top = X_top[ctrl_mask_top].reset_index(drop=True)
y_ctrl_top = y_top[ctrl_mask_top].reset_index(drop=True)
X_trt_top  = X_top[~ctrl_mask_top].reset_index(drop=True)
y_trt_top  = y_top[~ctrl_mask_top].reset_index(drop=True)

n_gen = len(X_trt_top) - len(X_ctrl_top)
print(f'\nTreatment: {len(X_trt_top):,}  Control: {len(X_ctrl_top):,}  Generate: {n_gen:,}')

smote_top = ControlGroupSMOTE(random_state=42)
smote_top.fit(X_ctrl_top, y_ctrl_top)
X_syn_top, y_syn_top = smote_top.generate(n_gen)
# Restore column names
X_syn_top.columns = X_ctrl_top.columns

t_ctrl_ser = pd.Series([0]*len(X_ctrl_top), name='treatment')
t_syn_ser  = pd.Series([0]*n_gen,            name='treatment')
t_trt_ser  = pd.Series([1]*len(X_trt_top),  name='treatment')

X_top_bal = pd.concat([X_ctrl_top, X_syn_top, X_trt_top], ignore_index=True)
y_top_bal = pd.concat([y_ctrl_top, y_syn_top, y_trt_top], ignore_index=True)
t_top_bal = pd.concat([t_ctrl_ser, t_syn_ser,  t_trt_ser], ignore_index=True)

shuffle_idx = np.random.RandomState(42).permutation(len(X_top_bal))
X_top_bal = X_top_bal.iloc[shuffle_idx].reset_index(drop=True)
y_top_bal = y_top_bal.iloc[shuffle_idx].reset_index(drop=True)
t_top_bal = t_top_bal.iloc[shuffle_idx].reset_index(drop=True)

print(f'\nBalanced top-10% subset: {len(X_top_bal):,} rows, T rate: {t_top_bal.mean():.4f}')

# ── Step 4: Train S/T-Learner, evaluate on full test ──────────────────────────
lgbm_params = dict(n_estimators=500, max_depth=6, num_leaves=31,
                   learning_rate=0.05, random_state=42, verbose=-1)

top10_results = []

configs = [
    ('S-Learner (full 75/25)',    X_train,   y_train,   treatment_train),
    ('S-Learner (top-10% SMOTE)', X_top_bal, y_top_bal, t_top_bal),
    ('T-Learner (full 75/25)',    X_train,   y_train,   treatment_train),
    ('T-Learner (top-10% SMOTE)', X_top_bal, y_top_bal, t_top_bal),
]

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
    top10_results.append(row)
    print(f"{label}: Qini AUC={row['Qini AUC']:.4f}  Uplift AUC={row['Uplift AUC']:.4f}  ASD={row['ASD']:.6f}  ({elapsed:.1f}s)")

# ── Step 5: Summary table ─────────────────────────────────────────────────────
top10_df = pd.DataFrame(top10_results).set_index('Model')
print('\n=== CT-guided SMOTE на топ-10% ===')
print(top10_df.to_string(float_format='{:.6f}'.format))

print('\n=== Delta vs baseline ===')
for base_label, smote_label in [
    ('S-Learner (full 75/25)',    'S-Learner (top-10% SMOTE)'),
    ('T-Learner (full 75/25)',    'T-Learner (top-10% SMOTE)'),
]:
    orig = top10_df.loc[base_label, 'Qini AUC']
    new  = top10_df.loc[smote_label,'Qini AUC']
    print(f"{smote_label.split('(')[0].strip()}: {orig:.4f} → {new:.4f}  ({(new-orig)/orig*100:+.1f}%)")
