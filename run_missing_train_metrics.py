"""Compute train metrics for DR-Learner, CausalForestDML, URF original, URF SMOTE 200K."""
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMClassifier, LGBMRegressor
from sklift.models import ClassTransformation
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k
from causalml.inference.meta import BaseDRLearner
from econml.dml import CausalForestDML
from causalml.inference.tree import UpliftRandomForestClassifier

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

X_tr = X_train.values; X_te = X_test.values
t_tr = treatment_train.values; t_te = treatment_test.values
y_tr = y_train.values.astype(float); y_te = y_test.values.astype(float)

def asd_score(y_true, uplift_pred, treatment, n_bins=10):
    df = pd.DataFrame({'y': np.asarray(y_true), 'uplift': np.asarray(uplift_pred),
                       'treatment': np.asarray(treatment)})
    df['decile'] = pd.qcut(df['uplift'].rank(method='first'), q=n_bins,
                            labels=False, duplicates='drop')
    sq = []
    for _, grp in df.groupby('decile'):
        trt  = grp[grp['treatment'] == 1]['y']
        ctrl = grp[grp['treatment'] == 0]['y']
        if len(trt) == 0 or len(ctrl) == 0: continue
        sq.append((trt.mean() - ctrl.mean() - grp['uplift'].mean()) ** 2)
    return float(np.mean(sq)) if sq else np.nan

def m(y, pred, t):
    return {
        'uplift@10%': uplift_at_k(y, pred, t, strategy='by_group', k=0.1),
        'uplift@30%': uplift_at_k(y, pred, t, strategy='by_group', k=0.3),
        'Qini AUC':   qini_auc_score(y, pred, t),
        'Uplift AUC': uplift_auc_score(y, pred, t),
        'ASD':        asd_score(y, pred, t),
    }

# ── Propensity ─────────────────────────────────────────────────────────────────
print("\nPropensity (cv=3)...")
t0 = time.time()
prop = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                       random_state=42, verbose=-1, n_jobs=-1)
p_cv = cross_val_predict(prop, X_tr, t_tr, cv=3, method='predict_proba')[:, 1]
p_cv = np.clip(p_cv, 0.01, 0.99)
prop.fit(X_tr, t_tr)
p_test = np.clip(prop.predict_proba(X_te)[:, 1], 0.01, 0.99)
p_train_full = np.clip(prop.predict_proba(X_tr)[:, 1], 0.01, 0.99)
print(f"  Done ({time.time()-t0:.0f}s)")

results = {}

# ── DR-Learner ─────────────────────────────────────────────────────────────────
print("\n=== DR-Learner ===")
t0 = time.time()
dr = BaseDRLearner(learner=LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                          random_state=42, verbose=-1, n_jobs=-1))
dr.fit(X=X_tr, treatment=t_tr, y=y_tr, p=p_cv)
dr_train = dr.predict(X=X_tr, p=p_train_full).flatten()
dr_test  = dr.predict(X=X_te, p=p_test).flatten()
elapsed = time.time() - t0
tr = m(y_train, dr_train, treatment_train)
te = m(y_test,  dr_test,  treatment_test)
print(f"  Done ({elapsed:.0f}s)")
print(f"  Train: Qini={tr['Qini AUC']:.4f}  u@10%={tr['uplift@10%']:.4f}  ASD={tr['ASD']:.6f}")
print(f"  Test:  Qini={te['Qini AUC']:.4f}  u@10%={te['uplift@10%']:.4f}  ASD={te['ASD']:.6f}")
results['DR-Learner'] = {'train': tr, 'test': te}

# ── CausalForestDML (80K subsample) ────────────────────────────────────────────
print("\n=== CausalForestDML (80K subsample) ===")
t0 = time.time()
np.random.seed(42)
idx_sub = np.random.choice(len(X_tr), size=80000, replace=False)
X_sub = X_tr[idx_sub]; y_sub = y_tr[idx_sub]; t_sub = t_tr[idx_sub]
cf = CausalForestDML(
    model_y=LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                           random_state=42, verbose=-1, n_jobs=-1),
    model_t=LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                            random_state=42, verbose=-1, n_jobs=-1),
    n_estimators=100, max_depth=6, random_state=42, discrete_treatment=True, verbose=0
)
cf.fit(y_sub, t_sub, X=X_sub)
# Train metrics on subsample (the data it was trained on)
cf_train_sub = cf.effect(X_sub).flatten()
cf_test = cf.effect(X_te).flatten()
elapsed = time.time() - t0
# Use subsample labels/treatment for train metrics
tr = m(y_sub, cf_train_sub, t_sub)
te = m(y_test, cf_test, treatment_test)
print(f"  Done ({elapsed:.0f}s)")
print(f"  Train (80K): Qini={tr['Qini AUC']:.4f}  u@10%={tr['uplift@10%']:.4f}  ASD={tr['ASD']:.6f}")
print(f"  Test:        Qini={te['Qini AUC']:.4f}  u@10%={te['uplift@10%']:.4f}  ASD={te['ASD']:.6f}")
results['CausalForestDML (80K)'] = {'train': tr, 'test': te, 'train_note': '80K subsample'}

# ── URF (original 75/25, full 480K) ────────────────────────────────────────────
print("\n=== URF (original 75/25, 480K) ===")
t_str_train = treatment_train.map({1: 'test', 0: 'control'})
t0 = time.time()
urf_orig = UpliftRandomForestClassifier(
    control_name='control', n_estimators=50, max_depth=5, max_features=10,
    min_samples_leaf=1000, min_samples_treatment=200, n_reg=50,
    evaluationFunction='KL', normalization=True, random_state=42, n_jobs=-1
)
urf_orig.fit(X_tr, treatment=t_str_train.values, y=y_tr)
urf_orig_train = urf_orig.predict(X_tr).flatten()
urf_orig_test  = urf_orig.predict(X_te).flatten()
elapsed = time.time() - t0
tr = m(y_train, urf_orig_train, treatment_train)
te = m(y_test,  urf_orig_test,  treatment_test)
print(f"  Done ({elapsed:.0f}s)")
print(f"  Train: Qini={tr['Qini AUC']:.4f}  u@10%={tr['uplift@10%']:.4f}  ASD={tr['ASD']:.6f}")
print(f"  Test:  Qini={te['Qini AUC']:.4f}  u@10%={te['uplift@10%']:.4f}  ASD={te['ASD']:.6f}")
results['URF (original 75/25)'] = {'train': tr, 'test': te}

# ── URF (SMOTE top-10% 200K) ───────────────────────────────────────────────────
print("\n=== URF (SMOTE top-10% 200K) ===")

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

ct_scorer = ClassTransformation(LGBMClassifier(
    n_estimators=500, max_depth=6, num_leaves=31,
    learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1))
ct_scorer.fit(X_train, y_train, treatment_train)
ct_scores = ct_scorer.predict(X_train)
threshold = np.percentile(ct_scores, 90)
ctrl_mask = treatment_train == 0
top10_ctrl = (ct_scores >= threshold) & ctrl_mask
X_ctrl_top10 = X_train[top10_ctrl].reset_index(drop=True)
y_ctrl_top10 = y_train[top10_ctrl].reset_index(drop=True)
n_generate = int((treatment_train == 1).sum()) - int((treatment_train == 0).sum())
smote = ControlGroupSMOTE(random_state=42)
smote.fit(X_ctrl_top10, y_ctrl_top10)
X_syn, y_syn = smote.generate(n_generate)
t_syn = pd.Series([0] * n_generate)
X_aug = pd.concat([X_train, X_syn], ignore_index=True)
y_aug = pd.concat([y_train, y_syn], ignore_index=True)
t_aug = pd.concat([treatment_train, t_syn], ignore_index=True)
shuffle_idx = np.random.RandomState(42).permutation(len(X_aug))
X_aug = X_aug.iloc[shuffle_idx].reset_index(drop=True)
y_aug = y_aug.iloc[shuffle_idx].reset_index(drop=True)
t_aug = t_aug.iloc[shuffle_idx].reset_index(drop=True)
_aug_idx = np.random.RandomState(42).choice(len(X_aug), size=200_000, replace=False)
X_aug_urf = X_aug.iloc[_aug_idx].reset_index(drop=True)
y_aug_urf = y_aug.iloc[_aug_idx].reset_index(drop=True)
t_aug_urf = t_aug.iloc[_aug_idx].reset_index(drop=True)
t_aug_urf_str = t_aug_urf.map({1: 'test', 0: 'control'})

t0 = time.time()
urf_smote = UpliftRandomForestClassifier(
    control_name='control', n_estimators=50, max_depth=5, max_features=10,
    min_samples_leaf=1000, min_samples_treatment=200, n_reg=50,
    evaluationFunction='KL', normalization=True, random_state=42, n_jobs=-1
)
urf_smote.fit(X_aug_urf.values, treatment=t_aug_urf_str.values, y=y_aug_urf.values)
# Train on the 200K it was trained on
urf_smote_train = urf_smote.predict(X_aug_urf.values).flatten()
urf_smote_test  = urf_smote.predict(X_te).flatten()
elapsed = time.time() - t0
tr = m(y_aug_urf, urf_smote_train, t_aug_urf)
te = m(y_test,    urf_smote_test,  treatment_test)
print(f"  Done ({elapsed:.0f}s)")
print(f"  Train (200K): Qini={tr['Qini AUC']:.4f}  u@10%={tr['uplift@10%']:.4f}  ASD={tr['ASD']:.6f}")
print(f"  Test:         Qini={te['Qini AUC']:.4f}  u@10%={te['uplift@10%']:.4f}  ASD={te['ASD']:.6f}")
results['URF (SMOTE 200K)'] = {'train': tr, 'test': te, 'train_note': '200K subsample'}

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("TRAIN metrics")
print("="*80)
for name, r in results.items():
    note = r.get('train_note', 'full train')
    tr = r['train']
    print(f"  {name} ({note}): u@10%={tr['uplift@10%']:.4f}  u@30%={tr['uplift@30%']:.4f}  "
          f"Qini={tr['Qini AUC']:.4f}  UpliftAUC={tr['Uplift AUC']:.4f}  ASD={tr['ASD']:.6f}")

print("\n" + "="*80)
print("TEST metrics")
print("="*80)
for name, r in results.items():
    te = r['test']
    print(f"  {name}: u@10%={te['uplift@10%']:.4f}  u@30%={te['uplift@30%']:.4f}  "
          f"Qini={te['Qini AUC']:.4f}  UpliftAUC={te['Uplift AUC']:.4f}  ASD={te['ASD']:.6f}")

print("\n=== Overfitting ratio (Qini AUC Train/Test) ===")
for name, r in results.items():
    q_tr = r['train']['Qini AUC']
    q_te = r['test']['Qini AUC']
    ratio = q_tr / q_te if q_te > 0 else float('inf')
    note = r.get('train_note', '')
    print(f"  {name}: Train={q_tr:.4f}  Test={q_te:.4f}  Ratio={ratio:.1f}x  {note}")
