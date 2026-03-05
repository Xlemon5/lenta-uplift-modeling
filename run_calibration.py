"""Calibration experiments: reduce ASD for CT+LGB without losing Qini AUC.

Methods:
  1. Isotonic regression calibration (OOF CT scores → IPW pseudo-outcomes)
  2. Ensemble: CT z-score + R-Learner z-score (blended, various alpha)
     + Ensemble: isotonic-calibrated CT + R-Learner (both on CATE scale)
  3. CalibratedClassifierCV (isotonic) on base LGB inside CT

Root cause: CT formula uplift = 2*P(Z=1|X)-1 is unbiased only for p=0.5.
At p=0.75: E[CT] ≈ 1.5*CR_T - 0.5*CR_C - 0.5 ≈ -0.385, vs true uplift ~+0.007.
Isotonic calibration maps CT scores → actual CATE scale via OOF + IPW pseudo-outcomes.
"""
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, KFold
from lightgbm import LGBMClassifier, LGBMRegressor
from sklift.models import ClassTransformation
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k
from causalml.inference.meta import BaseRRegressor

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
print(f"Treatment rate train: {treatment_train.mean():.4f}")

X_tr = X_train.values; X_te = X_test.values
t_tr = treatment_train.values; t_te = treatment_test.values
y_tr = y_train.values.astype(float); y_te = y_test.values.astype(float)

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

def report(label, pred, t_start=None):
    q   = qini_auc_score(y_test, pred, treatment_test)
    u10 = uplift_at_k(y_test, pred, treatment_test, strategy='by_group', k=0.1)
    u30 = uplift_at_k(y_test, pred, treatment_test, strategy='by_group', k=0.3)
    a   = asd_score(y_test, pred, treatment_test)
    elapsed = f"  ({time.time()-t_start:.0f}s)" if t_start else ""
    print(f"  {label}: Qini={q:.4f}  u@10%={u10:.4f}  u@30%={u30:.4f}  ASD={a:.6f}{elapsed}")
    return {'Model': label, 'uplift@10%': u10, 'uplift@30%': u30,
            'Qini AUC': q, 'Uplift AUC': uplift_auc_score(y_test, pred, treatment_test), 'ASD': a}

lgbm_params = dict(n_estimators=500, max_depth=6, num_leaves=31,
                   learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)

results = []

# ── Baseline CT ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Baseline CT + LGB")
print("="*70)
t0 = time.time()
ct = ClassTransformation(LGBMClassifier(**lgbm_params))
ct.fit(X_train, y_train, treatment_train)
ct_pred = ct.predict(X_test)
results.append(report('CT baseline', ct_pred, t0))
print(f"  CT score range: [{ct_pred.min():.4f}, {ct_pred.max():.4f}], mean={ct_pred.mean():.4f}")

# ── Propensity (for IPW pseudo-outcomes and R-Learner) ────────────────────────
print("\n--- Propensity scores (cv=3, fast LGBM) ---")
t0 = time.time()
prop_clf = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                           random_state=42, verbose=-1, n_jobs=-1)
p_cv = cross_val_predict(prop_clf, X_tr, t_tr, cv=3, method='predict_proba')[:, 1]
p_cv = np.clip(p_cv, 0.01, 0.99)
print(f"  Done ({time.time()-t0:.0f}s), mean={p_cv.mean():.4f}")

# IPW pseudo-outcomes: unbiased per-obs CATE estimate (very noisy, but unbiased)
pseudo = t_tr * y_tr / p_cv - (1 - t_tr) * y_tr / (1 - p_cv)
print(f"  IPW pseudo-outcome: mean={pseudo.mean():.4f}, std={pseudo.std():.4f}")

# ── Method 1: Isotonic regression calibration ──────────────────────────────────
print("\n" + "="*70)
print("Method 1: Isotonic Regression Calibration")
print("="*70)
print("  Computing OOF CT scores (3-fold)...")
t0 = time.time()
kf = KFold(n_splits=3, shuffle=True, random_state=42)
ct_oof = np.zeros(len(X_train))
for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
    print(f"    Fold {fold_i+1}/3...")
    ct_fold = ClassTransformation(LGBMClassifier(**lgbm_params))
    ct_fold.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], treatment_train.iloc[tr_idx])
    ct_oof[val_idx] = ct_fold.predict(X_train.iloc[val_idx])

print(f"  OOF scores done ({time.time()-t0:.0f}s).")
print(f"  OOF score range: [{ct_oof.min():.4f}, {ct_oof.max():.4f}], mean={ct_oof.mean():.4f}")

# Fit isotonic regression: CT score → CATE pseudo-outcome
# increasing=True because higher CT score should mean higher uplift
iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
iso.fit(ct_oof, pseudo)
ct_iso_pred = iso.predict(ct_pred)
print(f"  Calibrated score range: [{ct_iso_pred.min():.4f}, {ct_iso_pred.max():.4f}], mean={ct_iso_pred.mean():.4f}")
results.append(report('CT + Isotonic (OOF+IPW)', ct_iso_pred))

# ── Method 2: Ensemble CT + R-Learner ─────────────────────────────────────────
print("\n" + "="*70)
print("Method 2: Ensemble CT + R-Learner")
print("="*70)
print("  Training R-Learner (LGBMRegressor, 100 trees)...")
t0 = time.time()
rl = BaseRRegressor(learner=LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                           random_state=42, verbose=-1, n_jobs=-1))
rl.fit(X=X_tr, treatment=t_tr, y=y_tr, p=p_cv)
rl_pred = rl.predict(X=X_te).flatten()
print(f"  R-Learner done ({time.time()-t0:.0f}s). Score range: [{rl_pred.min():.4f}, {rl_pred.max():.4f}], mean={rl_pred.mean():.4f}")

# 2a: z-score normalize both, then blend (preserves Qini of dominant component)
ct_z  = (ct_pred  - ct_pred.mean())  / ct_pred.std()
rl_z  = (rl_pred  - rl_pred.mean())  / rl_pred.std()
print("\n  2a. Z-score blend (raw CT + R-Learner):")
for alpha in [0.95, 0.90, 0.80, 0.70, 0.50]:
    combined = alpha * ct_z + (1 - alpha) * rl_z
    results.append(report(f'CT+RL z-score α={alpha}', combined))

# 2b: Both on CATE scale — isotonic-calibrated CT + R-Learner
print("\n  2b. CATE-scale blend (isotonic CT + R-Learner):")
for alpha in [0.95, 0.90, 0.80, 0.70, 0.50]:
    combined = alpha * ct_iso_pred + (1 - alpha) * rl_pred
    results.append(report(f'CT_iso+RL α={alpha}', combined))

# ── Method 3: CalibratedClassifierCV ──────────────────────────────────────────
print("\n" + "="*70)
print("Method 3: CalibratedClassifierCV (isotonic, cv=3) inside CT")
print("="*70)
t0 = time.time()
cal_base = CalibratedClassifierCV(
    LGBMClassifier(**lgbm_params), method='isotonic', cv=3
)
ct_cal = ClassTransformation(cal_base)
ct_cal.fit(X_train, y_train, treatment_train)
ct_cal_pred = ct_cal.predict(X_test)
print(f"  Calibrated score range: [{ct_cal_pred.min():.4f}, {ct_cal_pred.max():.4f}], mean={ct_cal_pred.mean():.4f}")
results.append(report('CT + CalibratedClassifierCV', ct_cal_pred, t0))

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
df = pd.DataFrame(results)
print(df[['Model', 'uplift@10%', 'uplift@30%', 'Qini AUC', 'ASD']].to_string(
    index=False, float_format='{:.6f}'.format))

print("\n=== ASD improvement vs baseline ===")
baseline_asd   = results[0]['ASD']
baseline_qini  = results[0]['Qini AUC']
for row in results[1:]:
    asd_delta  = (row['ASD']      - baseline_asd)  / baseline_asd  * 100
    qini_delta = (row['Qini AUC'] - baseline_qini) / baseline_qini * 100
    print(f"  {row['Model']:<35} ASD: {row['ASD']:.6f} ({asd_delta:+.1f}%)  "
          f"Qini: {row['Qini AUC']:.4f} ({qini_delta:+.1f}%)")
