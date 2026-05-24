"""
train_model.py — LightGBM signal classifier (optimised)
========================================================
Run this ONCE to build trade_classifier_lgbm.pkl, then place that file
in the same directory as your engine script (data_implementation.py).

Changes applied to produce a reliable model:
  • Uses ALL 31 pairs from the engine (ALL_PAIRS)
  • Lowered min_confidence to 0.4 to generate more training signals
  • Recommended to extend yfinance data period to 180d in MultiTimeframeDataManager
  • All previous fixes preserved
"""

import sys
import pandas as pd
import numpy as np
import joblib

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
except ImportError:
    sys.exit("lightgbm not installed. Run: pip install lightgbm")

try:
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    sys.exit("scikit-learn not installed. Run: pip install scikit-learn")

# ── Import engine functions and the full pair list ────────────────────────────
try:
    from data_implementation import create_training_dataset, extract_features, ALL_PAIRS
except ModuleNotFoundError:
    sys.exit(
        "Cannot find data_implementation.py in this directory.\n"
        "Make sure train_model.py sits in the same folder as your engine file."
    )

# =============================================================================
# 1. GENERATE TRAINING DATA
# =============================================================================
print("=" * 60)
print("Step 1 — Generating training data")
print("=" * 60)

df = create_training_dataset(
    pairs=ALL_PAIRS,                # use all 31 pairs
    timeframes=["1h", "4h"],
    min_confidence=0.4,             # lower threshold → more training samples
    min_rr=1.5,
    abort_if_insufficient=False,
)

if df is None or df.empty:
    sys.exit(
        "Training dataset is empty. Possible causes:\n"
        "  • yfinance rate-limited — wait a few minutes and retry\n"
        "  • All pairs returning no data\n"
        "  • create_training_dataset raised a silent exception"
    )

print(f"\nDataset shape:        {df.shape}")
print(f"Win rate in data:     {df['outcome'].mean():.2%}")
print(f"Total labelled rows:  {len(df)}")

if len(df) < 800:
    print(
        "\nWARNING: only {n} samples — minimum 800 recommended.\n"
        "The model will likely overfit. Options:\n"
        "  • Extend yfinance period in MultiTimeframeDataManager\n"
        "  • Add more pairs or timeframes\n"
        "  • Lower min_confidence further to 0.3".format(n=len(df))
    )

# =============================================================================
# 2. PREPARE FEATURES
# =============================================================================
print("\n" + "=" * 60)
print("Step 2 — Preparing features")
print("=" * 60)

drop_cols = ['outcome', 'pair', 'timeframe']
feature_cols = [c for c in df.columns if c not in drop_cols]

# Chronological sort before splitting (if timestamp column exists)
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp').reset_index(drop=True)

X = df[feature_cols].fillna(0)
y = df['outcome'].astype(int)

print(f"Features:  {len(feature_cols)}")
print(f"Positives (wins):  {y.sum()} ({y.mean():.1%})")
print(f"Negatives (losses): {(1 - y).sum()} ({(1-y).mean():.1%})")

# =============================================================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "=" * 60)
print("Step 3 — Chronological train/test split (80/20)")
print("=" * 60)

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

print(f"Train rows: {len(X_train)}")
print(f"Test rows:  {len(X_test)}")

if len(X_test) < 50:
    print(
        "WARNING: test set has fewer than 50 rows — "
        "metrics will be unreliable. Generate more data."
    )

# =============================================================================
# 4. TRAIN LIGHTGBM
# =============================================================================
print("\n" + "=" * 60)
print("Step 4 — Training LightGBM classifier")
print("=" * 60)

model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_samples=20,       # prevents overfitting on small datasets
    num_leaves=31,
    class_weight='balanced',    # handles win/loss imbalance
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=50),   # prints every 50 rounds
    ],
)

print(f"\nBest iteration: {model.best_iteration_}")

# =============================================================================
# 5. EVALUATE
# =============================================================================
print("\n" + "=" * 60)
print("Step 5 — Evaluation on held-out test set")
print("=" * 60)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))

# Threshold tuning
print("Win precision at different probability thresholds:")
for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    y_thresh = (y_proba >= threshold).astype(int)
    if y_thresh.sum() == 0:
        print(f"  {threshold:.2f} → no trades would be taken")
        continue
    report = classification_report(
        y_test, y_thresh, target_names=['Loss', 'Win'],
        output_dict=True, zero_division=0
    )
    prec   = report['Win']['precision']
    recall = report['Win']['recall']
    taken  = y_thresh.sum()
    print(f"  {threshold:.2f} → precision {prec:.2%}, recall {recall:.2%}, trades taken {taken}/{len(y_test)}")

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 60)
print("Step 6 — Feature importance")
print("=" * 60)

importance = pd.DataFrame({
    'feature':    feature_cols,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)

print("\nTop 10 features:")
print(importance.head(10).to_string(index=False))

top_share = importance['importance'].iloc[0] / importance['importance'].sum()
if top_share > 0.4:
    print(
        f"\nWARNING: top feature '{importance['feature'].iloc[0]}' accounts for "
        f"{top_share:.0%} of importance — possible overfitting or data leakage."
    )

# =============================================================================
# 7. SANITY CHECKS
# =============================================================================
print("\n" + "=" * 60)
print("Step 7 — Sanity checks")
print("=" * 60)

report_dict   = classification_report(y_test, y_pred, target_names=['Loss','Win'], output_dict=True, zero_division=0)
win_precision = report_dict['Win']['precision']
win_recall    = report_dict['Win']['recall']
accuracy      = accuracy_score(y_test, y_pred)

print(f"Win precision: {win_precision:.2%}")
print(f"Win recall:    {win_recall:.2%}")
print(f"Accuracy:      {accuracy:.2%}")

if win_precision < 0.55:
    verdict = "FAIL — model cannot identify wins reliably. Do NOT use live."
elif win_precision < 0.65:
    verdict = "MARGINAL — paper trade only. Collect more data before going live."
elif accuracy > 0.80:
    verdict = "SUSPICIOUS — accuracy too high, likely look-ahead bias in features. Do not use live."
else:
    verdict = "PASS — acceptable for careful paper trading. Monitor live results weekly."

print(f"\nVerdict: {verdict}")

# =============================================================================
# 8. SAVE AND VERIFY
# =============================================================================
print("\n" + "=" * 60)
print("Step 8 — Saving model")
print("=" * 60)

MODEL_PATH = 'trade_classifier_lgbm.pkl'
joblib.dump(model, MODEL_PATH)
print(f"Saved → {MODEL_PATH}")

# Verify the saved model
try:
    loaded = joblib.load(MODEL_PATH)
    _ = loaded.feature_name_
    assert list(loaded.feature_name_) == feature_cols, "Feature names mismatch after reload"
    print("Verification: model reloaded successfully and feature names match.")
except AttributeError:
    print(
        "WARNING: loaded model has no feature_name_ attribute.\n"
        "The engine's apply_ml_score() will crash. Check your LightGBM version."
    )
except AssertionError as e:
    print(f"WARNING: {e}")
except Exception as e:
    print(f"WARNING: model reload failed — {e}")

print("\nDone. Place trade_classifier_lgbm.pkl in the same folder as data_implementation.py")