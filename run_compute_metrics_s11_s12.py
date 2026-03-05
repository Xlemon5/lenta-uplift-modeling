"""Compute uplift@10% and uplift@30% for sections 11 (meta-learners) and 12 (CT-guided SMOTE)."""
import time, warnings
warnings.filterwarnings('ignore')
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

from lightgbm import LGBMClassifier, LGBMRegressor
from sklift.models import ClassTransformation, SoloModel, TwoModels
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k
from sklearn.model_selection import cross_val_predict
from sklearn.utils import resample

def asd_score(y_true, uplift_pred, treatment, n_bins=10):
    df = pd.DataFrame({'y': np.asarray(y_true), 'uplift': np.asarray(uplift_pred),
                       'treatment': np.asarray(treatment)})
    df['decile'] = pd.qcut(df['uplift'].rank(method='first'), q=n_bins, labels=False, duplicates='drop')
    sq = []
    for _, grp in df.groupby('decile'):
        trt  = grp[grp['treatment'] == 1]['y']
        ctrl = grp[grp['treatment'] == 0]['y']
        if len(trt) == 0 or len(ctrl) == 0: continue
        sq.append((trt.mean() - ctrl.mean() - grp['uplift'].mean()) ** 2)
    return float(np.mean(sq)) if sq else np.nan

def metrics(y, pred, t):
    return {
        'uplift@10%': uplift_at_k(y, pred, t, strategy='by_group', k=0.1),
        'uplift@30%': uplift_at_k(y, pred, t, strategy='by_group', k=0.3),
        'Qini AUC':   qini_auc_score(y, pred, t),
        'Uplift AUC': uplift_auc_score(y, pred, t),
        'ASD':        asd_score(y, pred, t),
    }

def make_lgbm_reg():
    return LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1, n_jobs=-1)
def make_lgbm_clf():
    return LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1, n_jobs=-1)

lgbm_params = dict(n_estimators=500, max_depth=6, num_leaves=31,
                   learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)

X_tr = X_train.values; X_te = X_test.values
t_tr = treatment_train.values.astype(int); t_te = treatment_test.values.astype(int)
y_tr = y_train.values.astype(float); y_te = y_test.values.astype(float)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 11: Meta-learners")
print("="*80)

from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseDRLearner
from econml.dml import CausalForestDML

print("  Computing cross-fitted propensity scores (cv=3)...")
t0 = time.time()
p_train_cv = cross_val_predict(make_lgbm_clf(), X_tr, t_tr, cv=3, method='predict_proba')[:, 1]
prop_model = make_lgbm_clf()
prop_model.fit(X_tr, t_tr)
p_test = prop_model.predict_proba(X_te)[:, 1]
p_train_cv = np.clip(p_train_cv, 0.01, 0.99)
p_test     = np.clip(p_test,     0.01, 0.99)
print(f"  Propensity done ({time.time()-t0:.1f}s). mean train={p_train_cv.mean():.4f}")

meta_results = []

# X-Learner
print("  X-Learner...")
t0 = time.time()
xl = BaseXRegressor(learner=make_lgbm_reg())
xl.fit(X=X_tr, treatment=t_tr, y=y_tr, p=p_train_cv)
xl_pred = xl.predict(X=X_te, p=p_test).flatten()
row = {'Model': 'X-Learner (LightGBM)', **metrics(y_te, xl_pred, t_te)}
meta_results.append(row)
print(f"    Qini={row['Qini AUC']:.4f}  u@10%={row['uplift@10%']:.4f}  u@30%={row['uplift@30%']:.4f}  ({time.time()-t0:.1f}s)")

# DR-Learner
print("  DR-Learner...")
t0 = time.time()
dr = BaseDRLearner(learner=make_lgbm_reg())
dr.fit(X=X_tr, treatment=t_tr, y=y_tr, p=p_train_cv)
dr_pred = dr.predict(X=X_te, p=p_test).flatten()
row = {'Model': 'DR-Learner (LightGBM)', **metrics(y_te, dr_pred, t_te)}
meta_results.append(row)
print(f"    Qini={row['Qini AUC']:.4f}  u@10%={row['uplift@10%']:.4f}  u@30%={row['uplift@30%']:.4f}  ({time.time()-t0:.1f}s)")

# R-Learner
print("  R-Learner...")
t0 = time.time()
rl = BaseRRegressor(learner=make_lgbm_reg())
rl.fit(X=X_tr, treatment=t_tr, y=y_tr, p=p_train_cv)
rl_pred = rl.predict(X=X_te).flatten()
row = {'Model': 'R-Learner (LightGBM)', **metrics(y_te, rl_pred, t_te)}
meta_results.append(row)
print(f"    Qini={row['Qini AUC']:.4f}  u@10%={row['uplift@10%']:.4f}  u@30%={row['uplift@30%']:.4f}  ({time.time()-t0:.1f}s)")

# CausalForestDML (80K subsample)
print("  CausalForestDML (80K subsample)...")
t0 = time.time()
np.random.seed(42)
idx_sub = np.random.choice(len(X_tr), size=80000, replace=False)
X_sub = X_tr[idx_sub]; y_sub = y_tr[idx_sub]; t_sub = t_tr[idx_sub]
cf = CausalForestDML(model_y=make_lgbm_reg(), model_t=make_lgbm_clf(),
                     n_estimators=100, max_depth=6, random_state=42,
                     discrete_treatment=True, verbose=0)
cf.fit(y_sub, t_sub, X=X_sub)
cf_pred = cf.effect(X_te).flatten()
row = {'Model': 'CausalForestDML (80K)', **metrics(y_te, cf_pred, t_te)}
meta_results.append(row)
print(f"    Qini={row['Qini AUC']:.4f}  u@10%={row['uplift@10%']:.4f}  u@30%={row['uplift@30%']:.4f}  ({time.time()-t0:.1f}s)")

meta_df = pd.DataFrame(meta_results)
print("\n=== Section 11 full table ===")
print(meta_df.to_string(index=False, float_format='{:.4f}'.format))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 12: CT-guided SMOTE (top-10% ctrl)")
print("="*80)

lgbm = lambda: LGBMClassifier(**lgbm_params)

print("  Fitting CT for scoring...")
ct_scorer = ClassTransformation(lgbm())
ct_scorer.fit(X_train, y_train, treatment_train)
ct_scores = ct_scorer.predict(X_train)

threshold = np.percentile(ct_scores, 90)
ctrl_mask  = treatment_train == 0
top10_ctrl = (ct_scores >= threshold) & ctrl_mask
n_generate = int((treatment_train == 1).sum()) - int((treatment_train == 0).sum())

class ControlGroupSMOTE:
    def __init__(self, random_state=42):
        self.rng = np.random.RandomState(random_state)
    def fit(self, X, y):
        self.X_ = X.values if hasattr(X, 'values') else X
        self.y_ = y.values if hasattr(y, 'values') else y
        self.binary_cols_ = [j for j in range(self.X_.shape[1]) if np.all(np.isin(self.X_[:, j], [0, 1]))]
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

t_syn = pd.Series([0]*n_generate)

smote_full = ControlGroupSMOTE(random_state=42)
smote_full.fit(X_train[ctrl_mask].reset_index(drop=True), y_train[ctrl_mask].reset_index(drop=True))
X_syn_full, y_syn_full = smote_full.generate(n_generate)
X_aug_full  = pd.concat([X_train, X_syn_full], ignore_index=True)
y_aug_full  = pd.concat([y_train, y_syn_full], ignore_index=True)
t_aug_full  = pd.concat([treatment_train, t_syn], ignore_index=True)

X_ctrl_top10 = X_train[top10_ctrl].reset_index(drop=True)
y_ctrl_top10 = y_train[top10_ctrl].reset_index(drop=True)
smote_top10 = ControlGroupSMOTE(random_state=42)
smote_top10.fit(X_ctrl_top10, y_ctrl_top10)
X_syn_top10, y_syn_top10 = smote_top10.generate(n_generate)
X_aug_top10 = pd.concat([X_train, X_syn_top10], ignore_index=True)
y_aug_top10 = pd.concat([y_train, y_syn_top10], ignore_index=True)
t_aug_top10 = pd.concat([treatment_train, t_syn], ignore_index=True)

smote_configs = [
    ('S-Learner (original 75/25)',      X_train,     y_train,     treatment_train),
    ('S-Learner (SMOTE full ctrl)',      X_aug_full,  y_aug_full,  t_aug_full),
    ('S-Learner (SMOTE top-10% ctrl)',  X_aug_top10, y_aug_top10, t_aug_top10),
    ('T-Learner (original 75/25)',      X_train,     y_train,     treatment_train),
    ('T-Learner (SMOTE full ctrl)',      X_aug_full,  y_aug_full,  t_aug_full),
    ('T-Learner (SMOTE top-10% ctrl)',  X_aug_top10, y_aug_top10, t_aug_top10),
]

smote_results = []
for label, X_tr2, y_tr2, t_tr2 in smote_configs:
    t0 = time.time()
    m = SoloModel(lgbm(), method='dummy') if label.startswith('S') else TwoModels(lgbm(), lgbm(), method='vanilla')
    m.fit(X_tr2, y_tr2, t_tr2)
    pred = m.predict(X_test)
    row = {'Model': label, **metrics(y_test, pred, treatment_test), 'time_s': round(time.time()-t0, 1)}
    smote_results.append(row)
    print(f"  {label}: Qini={row['Qini AUC']:.4f}  u@10%={row['uplift@10%']:.4f}  u@30%={row['uplift@30%']:.4f}  ({row['time_s']}s)")

smote_df = pd.DataFrame(smote_results)
print("\n=== Section 12 full table ===")
print(smote_df.to_string(index=False, float_format='{:.4f}'.format))

print("\nDone.")
