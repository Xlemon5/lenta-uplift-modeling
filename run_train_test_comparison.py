"""Comprehensive train+test metrics table for all key models.

Train metrics: in-sample predictions (shows overfitting gap).
CT+Isotonic train: OOF predictions (unbiased).
"""
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict, KFold
from lightgbm import LGBMClassifier, LGBMRegressor
from sklift.models import ClassTransformation, SoloModel, TwoModels
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k
from causalml.inference.meta import BaseXRegressor, BaseRRegressor

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

def row_metrics(label, pred_train, pred_test):
    def m(y, pred, t):
        return {
            'uplift@10%': uplift_at_k(y, pred, t, strategy='by_group', k=0.1),
            'uplift@30%': uplift_at_k(y, pred, t, strategy='by_group', k=0.3),
            'Qini AUC':   qini_auc_score(y, pred, t),
            'Uplift AUC': uplift_auc_score(y, pred, t),
            'ASD':        asd_score(y, pred, t),
        }
    tr = m(y_train, pred_train, treatment_train)
    te = m(y_test,  pred_test,  treatment_test)
    print(f"  {label}")
    print(f"    Train: Qini={tr['Qini AUC']:.4f}  u@10%={tr['uplift@10%']:.4f}  u@30%={tr['uplift@30%']:.4f}  ASD={tr['ASD']:.6f}")
    print(f"    Test:  Qini={te['Qini AUC']:.4f}  u@10%={te['uplift@10%']:.4f}  u@30%={te['uplift@30%']:.4f}  ASD={te['ASD']:.6f}")
    return {'label': label, 'train': tr, 'test': te}

lgbm_params = dict(n_estimators=500, max_depth=6, num_leaves=31,
                   learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)

records = []

# ── CT baseline ────────────────────────────────────────────────────────────────
print("\n=== CT + LightGBM (baseline) ===")
t0 = time.time()
ct = ClassTransformation(LGBMClassifier(**lgbm_params))
ct.fit(X_train, y_train, treatment_train)
ct_train_pred = ct.predict(X_train)
ct_test_pred  = ct.predict(X_test)
print(f"  Fit done ({time.time()-t0:.0f}s)")
records.append(row_metrics('CT + LightGBM', ct_train_pred, ct_test_pred))

# ── Propensity (for meta-learners) ────────────────────────────────────────────
print("\n=== Propensity (cv=3) ===")
t0 = time.time()
prop_clf = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                           random_state=42, verbose=-1, n_jobs=-1)
p_cv = cross_val_predict(prop_clf, X_tr, t_tr, cv=3, method='predict_proba')[:, 1]
p_cv = np.clip(p_cv, 0.01, 0.99)
prop_clf.fit(X_tr, t_tr)
p_test = np.clip(prop_clf.predict_proba(X_te)[:, 1], 0.01, 0.99)
p_train_full = np.clip(prop_clf.predict_proba(X_tr)[:, 1], 0.01, 0.99)
print(f"  Done ({time.time()-t0:.0f}s)")

# ── CT + Isotonic calibration ─────────────────────────────────────────────────
print("\n=== CT + Isotonic calibration ===")
t0 = time.time()
print("  OOF CT scores (3-fold)...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
ct_oof = np.zeros(len(X_train))
for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
    print(f"    Fold {fold_i+1}/3...")
    ct_fold = ClassTransformation(LGBMClassifier(**lgbm_params))
    ct_fold.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], treatment_train.iloc[tr_idx])
    ct_oof[val_idx] = ct_fold.predict(X_train.iloc[val_idx])
pseudo = t_tr * y_tr / p_cv - (1 - t_tr) * y_tr / (1 - p_cv)
iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
iso.fit(ct_oof, pseudo)
ct_iso_train = iso.predict(ct_oof)       # OOF = unbiased train metric
ct_iso_test  = iso.predict(ct_test_pred)
print(f"  Done ({time.time()-t0:.0f}s)")
records.append(row_metrics('CT + Isotonic', ct_iso_train, ct_iso_test))

# ── S-Learner ─────────────────────────────────────────────────────────────────
print("\n=== S-Learner (SoloModel, dummy) ===")
t0 = time.time()
sl = SoloModel(LGBMClassifier(**lgbm_params), method='dummy')
sl.fit(X_train, y_train, treatment_train)
sl_train = sl.predict(X_train)
sl_test  = sl.predict(X_test)
print(f"  Fit done ({time.time()-t0:.0f}s)")
records.append(row_metrics('S-Learner', sl_train, sl_test))

# ── T-Learner ─────────────────────────────────────────────────────────────────
print("\n=== T-Learner (TwoModels, vanilla) ===")
t0 = time.time()
tl = TwoModels(LGBMClassifier(**lgbm_params), LGBMClassifier(**lgbm_params), method='vanilla')
tl.fit(X_train, y_train, treatment_train)
tl_train = tl.predict(X_train)
tl_test  = tl.predict(X_test)
print(f"  Fit done ({time.time()-t0:.0f}s)")
records.append(row_metrics('T-Learner', tl_train, tl_test))

# ── T-Learner + SMOTE top-10% ctrl ────────────────────────────────────────────
print("\n=== T-Learner + SMOTE top-10% ctrl ===")
t0 = time.time()

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

ct_scorer = ClassTransformation(LGBMClassifier(**lgbm_params))
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

tl_smote = TwoModels(LGBMClassifier(**lgbm_params), LGBMClassifier(**lgbm_params), method='vanilla')
tl_smote.fit(X_aug, y_aug, t_aug)
# Train metric on original train (fair comparison)
tl_smote_train = tl_smote.predict(X_train)
tl_smote_test  = tl_smote.predict(X_test)
print(f"  Done ({time.time()-t0:.0f}s)")
records.append(row_metrics('T-Learner + SMOTE top-10%', tl_smote_train, tl_smote_test))

# ── X-Learner ─────────────────────────────────────────────────────────────────
print("\n=== X-Learner ===")
t0 = time.time()
xl = BaseXRegressor(learner=LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                           random_state=42, verbose=-1, n_jobs=-1))
xl.fit(X=X_tr, treatment=t_tr, y=y_tr, p=p_cv)
xl_train = xl.predict(X=X_tr, p=p_train_full).flatten()
xl_test  = xl.predict(X=X_te, p=p_test).flatten()
print(f"  Done ({time.time()-t0:.0f}s)")
records.append(row_metrics('X-Learner', xl_train, xl_test))

# ── R-Learner ─────────────────────────────────────────────────────────────────
print("\n=== R-Learner ===")
t0 = time.time()
rl = BaseRRegressor(learner=LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                           random_state=42, verbose=-1, n_jobs=-1))
rl.fit(X=X_tr, treatment=t_tr, y=y_tr, p=p_cv)
rl_train = rl.predict(X=X_tr).flatten()
rl_test  = rl.predict(X=X_te).flatten()
print(f"  Done ({time.time()-t0:.0f}s)")
records.append(row_metrics('R-Learner', rl_train, rl_test))

# ── Format tables ──────────────────────────────────────────────────────────────
cols = ['uplift@10%', 'uplift@30%', 'Qini AUC', 'Uplift AUC', 'ASD']

def make_df(records, split):
    rows = []
    for r in records:
        d = {'Model': r['label']}
        d.update(r[split])
        rows.append(d)
    return pd.DataFrame(rows)[['Model'] + cols]

train_df = make_df(records, 'train')
test_df  = make_df(records, 'test')

print("\n" + "="*90)
print("TRAIN METRICS (in-sample; CT+Isotonic uses OOF predictions)")
print("="*90)
print(train_df.to_string(index=False, float_format='{:.4f}'.format))

print("\n" + "="*90)
print("TEST METRICS")
print("="*90)
print(test_df.to_string(index=False, float_format='{:.4f}'.format))

print("\n" + "="*90)
print("OVERFITTING GAP (Train - Test, Qini AUC)")
print("="*90)
for r in records:
    gap = r['train']['Qini AUC'] - r['test']['Qini AUC']
    ratio = r['train']['Qini AUC'] / r['test']['Qini AUC'] if r['test']['Qini AUC'] > 0 else float('inf')
    print(f"  {r['label']:<30} Train={r['train']['Qini AUC']:.4f}  Test={r['test']['Qini AUC']:.4f}  "
          f"Gap={gap:+.4f}  Ratio={ratio:.2f}x")
