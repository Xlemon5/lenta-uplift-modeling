"""Standalone script: UpliftRandomForestClassifier — Section 13 experiment."""
import time, warnings, os
warnings.filterwarnings('ignore')
os.chdir('/Users/ilya/Desktop/lenta')

import numpy as np
import pandas as pd

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

from lightgbm import LGBMClassifier
from sklift.models import ClassTransformation
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k
from causalml.inference.tree import UpliftRandomForestClassifier

def asd_score(y_true, uplift_pred, treatment, n_bins=10):
    df = pd.DataFrame({'y': np.asarray(y_true),
                       'uplift': np.asarray(uplift_pred),
                       'treatment': np.asarray(treatment)})
    df['decile'] = pd.qcut(df['uplift'].rank(method='first'), q=n_bins,
                            labels=False, duplicates='drop')
    sq = []
    for _, grp in df.groupby('decile'):
        trt  = grp[grp['treatment'] == 1]['y']
        ctrl = grp[grp['treatment'] == 0]['y']
        if len(trt) == 0 or len(ctrl) == 0:
            continue
        sq.append((trt.mean() - ctrl.mean() - grp['uplift'].mean()) ** 2)
    return float(np.mean(sq)) if sq else np.nan

# ── String treatment labels (URF requires strings, not ints) ─────────────────
treatment_train_str = treatment_train.map({1: 'test', 0: 'control'})
treatment_test_str  = treatment_test.map({1: 'test', 0: 'control'})

# ── ControlGroupSMOTE ─────────────────────────────────────────────────────────
class ControlGroupSMOTE:
    def __init__(self, random_state=42):
        self.rng = np.random.RandomState(random_state)
    def fit(self, X, y):
        self.X_ = X.values if hasattr(X, 'values') else X
        self.y_ = y.values if hasattr(y, 'values') else y
        self.binary_cols_ = [j for j in range(self.X_.shape[1])
                              if np.all(np.isin(self.X_[:, j], [0, 1]))]
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

# ── Build CT-guided SMOTE augmented dataset (same as Section 12) ─────────────
print("\nBuilding CT-guided SMOTE augmented dataset...")
ct_best = ClassTransformation(LGBMClassifier(
    n_estimators=500, max_depth=6, num_leaves=31,
    learning_rate=0.05, random_state=42, verbose=-1
))
ct_best.fit(X_train, y_train, treatment_train)
ct_scores = ct_best.predict(X_train)

threshold = np.percentile(ct_scores, 90)
ctrl_mask = treatment_train == 0
top10_ctrl_mask = (ct_scores >= threshold) & ctrl_mask

X_ctrl_top10 = X_train[top10_ctrl_mask].reset_index(drop=True)
y_ctrl_top10 = y_train[top10_ctrl_mask].reset_index(drop=True)

n_generate = int((treatment_train == 1).sum()) - int((treatment_train == 0).sum())
smote = ControlGroupSMOTE(random_state=42)
smote.fit(X_ctrl_top10, y_ctrl_top10)
X_syn, y_syn = smote.generate(n_generate)

t_syn = pd.Series([0] * n_generate, name='treatment')
X_aug = pd.concat([X_train, X_syn], ignore_index=True)
y_aug = pd.concat([y_train, y_syn], ignore_index=True)
t_aug = pd.concat([treatment_train, t_syn], ignore_index=True)

shuffle_idx = np.random.RandomState(42).permutation(len(X_aug))
X_aug = X_aug.iloc[shuffle_idx].reset_index(drop=True)
y_aug = y_aug.iloc[shuffle_idx].reset_index(drop=True)
t_aug = t_aug.iloc[shuffle_idx].reset_index(drop=True)

# Subsample 200K to match notebook config (same as modeling.ipynb cell 90)
_aug_idx = np.random.RandomState(42).choice(len(X_aug), size=200_000, replace=False)
X_aug_urf = X_aug.iloc[_aug_idx].reset_index(drop=True)
y_aug_urf = y_aug.iloc[_aug_idx].reset_index(drop=True)
t_aug_urf = t_aug.iloc[_aug_idx].reset_index(drop=True)
t_aug_urf_str = t_aug_urf.map({1: 'test', 0: 'control'})
print(f"Augmented: {len(X_aug):,} rows total; URF subsample: {len(X_aug_urf):,}, T rate: {t_aug_urf.mean():.4f}")

t_aug_str = t_aug.map({1: 'test', 0: 'control'})

# ── URF config (matches modeling.ipynb cell 90) ───────────────────────────────
def make_urf():
    return UpliftRandomForestClassifier(
        control_name='control',
        n_estimators=50,
        max_depth=5,
        max_features=10,
        min_samples_leaf=1000,
        min_samples_treatment=200,
        n_reg=50,
        evaluationFunction='KL',
        normalization=True,
        random_state=42,
        n_jobs=-1
    )

# ── Run experiments ───────────────────────────────────────────────────────────
configs = [
    ('URF (original 75/25)',      X_train,   y_train,   treatment_train_str),
    ('URF (SMOTE top-10% 200K)', X_aug_urf, y_aug_urf, t_aug_urf_str),
]

results = []
for label, X_tr, y_tr, t_tr in configs:
    print(f"\nTraining {label}...")
    t0 = time.time()
    urf = make_urf()
    urf.fit(X_tr.values, treatment=t_tr.values, y=y_tr.values)
    pred = urf.predict(X_test.values).flatten()
    elapsed = time.time() - t0
    row = {
        'Model':      label,
        'uplift@10%': uplift_at_k(y_test, pred, treatment_test, strategy='by_group', k=0.1),
        'uplift@30%': uplift_at_k(y_test, pred, treatment_test, strategy='by_group', k=0.3),
        'Qini AUC':   qini_auc_score(y_test, pred, treatment_test),
        'Uplift AUC': uplift_auc_score(y_test, pred, treatment_test),
        'ASD':        asd_score(y_test, pred, treatment_test),
        'time_s':     round(elapsed, 1),
    }
    results.append(row)
    print(f"  uplift@10%={row['uplift@10%']:.4f}  uplift@30%={row['uplift@30%']:.4f}  "
          f"Qini={row['Qini AUC']:.4f}  Uplift={row['Uplift AUC']:.4f}  "
          f"ASD={row['ASD']:.6f}  ({elapsed:.1f}s)")

# ── Summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(results).set_index('Model')
print('\n=== UpliftRandomForestClassifier ===')
print(df.to_string(float_format='{:.6f}'.format))

ct_baseline = 0.0752
print('\n=== Delta vs CT baseline (0.0752) ===')
for _, row in df.iterrows():
    delta = (row['Qini AUC'] - ct_baseline) / ct_baseline * 100
    print(f"  {row.name}: {row['Qini AUC']:.4f}  ({delta:+.1f}% vs CT)")
