"""Save decile bar charts for CT baseline, CT+Linear OLS, CT+Isotonic to reports/figures/."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from lightgbm import LGBMClassifier
from sklift.models import ClassTransformation
from sklift.metrics import qini_auc_score

print("Loading data...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test  = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()
treatment_train = pd.read_csv('data/processed/treatment_train.csv').squeeze()
treatment_test  = pd.read_csv('data/processed/treatment_test.csv').squeeze()
treatment_train = (treatment_train == 'test').astype(int)
treatment_test  = (treatment_test  == 'test').astype(int)

X_arr = X_train.values
t_arr = treatment_train.values
y_arr = y_train.values.astype(float)

lgbm_p = dict(n_estimators=500, max_depth=6, num_leaves=31,
              learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)

# ── 1. CT baseline ────────────────────────────────────────────────────────────
print("Training CT baseline...")
ct = ClassTransformation(LGBMClassifier(**lgbm_p))
ct.fit(X_train, y_train, treatment_train)
ct_test = ct.predict(X_test)
print(f"  Qini test: {qini_auc_score(y_test, ct_test, treatment_test):.4f}")

# ── 2. Propensity + OOF + calibrations ───────────────────────────────────────
print("Propensity (cv=3)...")
prop = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                      random_state=42, verbose=-1, n_jobs=-1)
p_cv = cross_val_predict(prop, X_arr, t_arr, cv=3, method='predict_proba')[:, 1]
p_cv = np.clip(p_cv, 0.01, 0.99)
pseudo = t_arr * y_arr / p_cv - (1 - t_arr) * y_arr / (1 - p_cv)

print("OOF CT scores (3-fold)...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
ct_oof = np.zeros(len(X_train))
for i, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    print(f"  Fold {i+1}/3...")
    cf = ClassTransformation(LGBMClassifier(**lgbm_p))
    cf.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], treatment_train.iloc[tr_idx])
    ct_oof[val_idx] = cf.predict(X_train.iloc[val_idx])

# Isotonic
iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
iso.fit(ct_oof, pseudo)
ct_iso_test = iso.predict(ct_test)

# Linear OLS
lr = LinearRegression()
lr.fit(ct_oof.reshape(-1, 1), pseudo)
ct_lin_test = lr.predict(ct_test.reshape(-1, 1))

print(f"  CT+Isotonic  Qini: {qini_auc_score(y_test, ct_iso_test, treatment_test):.4f}")
print(f"  CT+Linear    Qini: {qini_auc_score(y_test, ct_lin_test, treatment_test):.4f}")

# ── 3. Decile function ────────────────────────────────────────────────────────
def decile_stats(pred, y, t):
    df = pd.DataFrame({'pred': np.asarray(pred),
                       'y': np.asarray(y), 't': np.asarray(t)})
    df['bin'] = pd.qcut(df['pred'].rank(method='first'), q=10,
                        labels=False, duplicates='drop')
    rows = []
    for b in sorted(df['bin'].unique()):
        g = df[df['bin'] == b]
        cr_t = g.loc[g['t'] == 1, 'y'].mean()
        cr_c = g.loc[g['t'] == 0, 'y'].mean()
        rows.append({'Дециль': int(b)+1, 'CR_T': cr_t,
                     'CR_C': cr_c, 'Uplift': cr_t - cr_c})
    return pd.DataFrame(rows)

# ── 4. Plot and save ──────────────────────────────────────────────────────────
variants = [
    ('CT + LightGBM (baseline)', ct_test,     'ct_baseline'),
    ('CT + Linear OLS',          ct_lin_test,  'ct_linear_ols'),
    ('CT + Isotonic (OOF+IPW)',  ct_iso_test,  'ct_isotonic'),
]

for name, pred, fname in variants:
    bdf = decile_stats(pred, y_test, treatment_test)
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_c = ['#2ecc71' if u > 0 else '#e74c3c' for u in bdf['Uplift']]
    ax.bar(bdf['Дециль'], bdf['Uplift'], color=bar_c, edgecolor='white', width=0.7)
    ax.axhline(y=0, color='black', linewidth=0.8)
    for _, row in bdf.iterrows():
        offset = 0.0008 if row['Uplift'] >= 0 else -0.0020
        ax.text(row['Дециль'], row['Uplift'] + offset,
                f"{row['Uplift']:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Дециль (1 = наименьший predicted uplift, 10 = наибольший)')
    ax.set_ylabel('Фактический uplift (CR_T − CR_C)')
    ax.set_title(f'Uplift по децилям — {name} (тест)', fontsize=12, fontweight='bold')
    ax.set_xticks(bdf['Дециль'])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = f'reports/figures/decile_{fname}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print(bdf[['Дециль', 'Uplift']].to_string(index=False))

print("\nDone.")
