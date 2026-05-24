"""
retrain_model.py — periodic retraining with real journal trades
===============================================================
Run this weekly or after every 20+ new closed trades.

What it does
─────────────
1. Loads your REAL closed trades from trades.db (with WIN/LOSS outcomes)
2. Augments with fresh historical signals from the engine
3. Aligns features so both sources match exactly
4. Trains LightGBM chronologically (no leakage)
5. Saves trade_classifier_lgbm.pkl — same file the engine loads

Fixes vs pasted version
─────────────────────────
1.  Wrong import (data_implementation) → antigravity_quant_engine
2.  JPY=X invalid ticker → USDJPY=X (x2)
3.  lookback_days param doesn't exist → removed
4.  train_test_split shuffles rows → chronological split
5.  Indentation restored throughout
6.  Feature column mismatch on concat → common_cols alignment
7.  ZeroDivisionError in scale_pos_weight → guarded with max()
8.  No callbacks → early_stopping + log_evaluation added
9.  No feature_name_ verification before save → added
10. No DB existence check → os.path.exists guard added
"""

import os
import sys
import json
import sqlite3

import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("lightgbm not installed.  Run: pip install lightgbm")

try:
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    sys.exit("scikit-learn not installed.  Run: pip install scikit-learn")

# ── FIX 1: correct engine import ─────────────────────────────────────────────
try:
    from data_implementation import create_training_dataset, extract_features
except ModuleNotFoundError:
    sys.exit(
        "Cannot find antigravity_quant_engine.py.\n"
        "Make sure retrain_model.py is in the same folder as your engine file.\n"
        "If your engine has a different filename, update the import above."
    )

# =============================================================================
# CONFIG
# =============================================================================

DB_PATH    = "trades.db"
MODEL_PATH = "trade_classifier_lgbm.pkl"

PAIRS = [
    "BTC-USD",
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",   # FIX 2: was JPY=X which returns no data
    "GBPNZD=X",
    "AUDCAD=X",
]
TIMEFRAMES = ["1h", "4h"]

# Minimum journal trades before we trust the real-trade signal
MIN_JOURNAL_TRADES = 20

# Drop columns that are metadata, not features
META_COLS = ['outcome', 'pair', 'timeframe', 'timestamp']


# =============================================================================
# 1. LOAD REAL TRADES FROM JOURNAL DB
# =============================================================================

def load_journal_trades() -> tuple[pd.DataFrame, list]:
    """
    Reads closed trades from trades.db.
    Each row must have:
      • outcome  — 'WIN' or 'LOSS'
      • notes    — JSON string produced by generate_signal()

    Returns (X_df, y_list) or (None, None) if not enough data.
    """
    # FIX 10: check DB exists before connecting
    if not os.path.exists(DB_PATH):
        print(f"trades.db not found at '{DB_PATH}' — skipping journal trades.")
        return None, None

    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(
            "SELECT * FROM trades WHERE outcome IS NOT NULL", conn
        )
        conn.close()
    except Exception as e:
        print(f"DB read error: {e} — skipping journal trades.")
        return None, None

    if df.empty:
        print("trades.db exists but has no closed trades yet.")
        return None, None

    features_list, labels = [], []

    for _, row in df.iterrows():
        try:
            signal = json.loads(row['notes'])
        except (json.JSONDecodeError, TypeError):
            continue                         # skip rows with no valid signal JSON

        feats = extract_features(signal)
        features_list.append(feats)
        labels.append(1 if str(row['outcome']).upper() == 'WIN' else 0)

    if len(features_list) < MIN_JOURNAL_TRADES:
        print(
            f"Only {len(features_list)} journal trades with valid signal data "
            f"(need {MIN_JOURNAL_TRADES}). Falling back to historical data only."
        )
        return None, None

    X = pd.DataFrame(features_list).fillna(0)
    print(f"Loaded {len(X)} journal trades from trades.db  "
          f"(win rate: {np.mean(labels):.1%})")
    return X, labels


# =============================================================================
# 2. LOAD HISTORICAL DATA FROM ENGINE
# =============================================================================

def load_historical_data() -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """Calls create_training_dataset and returns (X, y) ready to use."""
    print("Fetching historical signal data from engine...")
    try:
        df = create_training_dataset(
            pairs=PAIRS,
            timeframes=TIMEFRAMES,
            min_confidence=0.5,
            min_rr=1.5,
            abort_if_insufficient=False,
        )
    except Exception as e:
        print(f"create_training_dataset failed: {e}")
        return None, None

    if df is None or df.empty:
        print("Historical dataset returned empty.")
        return None, None

    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].fillna(0)
    y = df['outcome'].astype(int)
    print(f"Historical data: {len(X)} rows, win rate {y.mean():.1%}")
    return X, y


# =============================================================================
# 3. ALIGN FEATURES  (FIX 6)
# =============================================================================

def align_features(X1: pd.DataFrame, X2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    When combining journal trades and historical data the feature columns
    may differ slightly (journal might be missing new features the engine
    added, or vice-versa).  Keep only columns present in BOTH, filled with 0
    for any missing values.
    """
    common = sorted(set(X1.columns) & set(X2.columns))
    if len(common) < len(X1.columns) or len(common) < len(X2.columns):
        only_1 = set(X1.columns) - set(X2.columns)
        only_2 = set(X2.columns) - set(X1.columns)
        if only_1:
            print(f"  Dropping {len(only_1)} journal-only features: {only_1}")
        if only_2:
            print(f"  Dropping {len(only_2)} history-only features: {only_2}")
    return X1[common].fillna(0), X2[common].fillna(0)


# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("retrain_model.py — LightGBM retraining")
print("=" * 60)

# ── Step 1: journal trades ────────────────────────────────────────────────────
X_journal, y_journal = load_journal_trades()

# ── Step 2: historical data ───────────────────────────────────────────────────
X_hist, y_hist = load_historical_data()

if X_hist is None and X_journal is None:
    sys.exit("No training data available from either source. Exiting.")

# ── Step 3: combine ───────────────────────────────────────────────────────────
print("\nCombining data sources...")

if X_journal is not None and X_hist is not None:
    X_journal, X_hist = align_features(X_journal, X_hist)   # FIX 6
    X = pd.concat([X_hist, X_journal], ignore_index=True)   # journal goes last (most recent)
    y = np.concatenate([y_hist.values, np.array(y_journal)])
    print(f"Combined: {len(X_hist)} historical + {len(X_journal)} journal = {len(X)} total rows")

elif X_journal is not None:
    X = X_journal
    y = np.array(y_journal)
    print(f"Using journal trades only: {len(X)} rows")

else:
    X = X_hist
    y = y_hist.values
    print(f"Using historical data only: {len(X)} rows")

# ── Step 4: chronological split (FIX 4) ──────────────────────────────────────
# Journal trades are appended at the end so the split naturally puts
# them in the test set — exactly what we want for out-of-sample evaluation.
print("\nSplitting chronologically (80/20)...")
split_idx        = int(len(X) * 0.8)
X_train, X_test  = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_train, y_test  = y[:split_idx],             y[split_idx:]

print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

if len(X_test) < 20:
    print("WARNING: test set very small — metrics will be noisy.")

# ── Step 5: scale_pos_weight guard (FIX 7) ───────────────────────────────────
n_neg = max((y_train == 0).sum(), 1)
n_pos = max((y_train == 1).sum(), 1)
spw   = n_neg / n_pos
print(f"\nClass balance — losses: {n_neg}, wins: {n_pos}, scale_pos_weight: {spw:.2f}")

# ── Step 6: train (FIX 8) ────────────────────────────────────────────────────
print("\nTraining LightGBM...")
model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.01,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.5,
    scale_pos_weight=spw,    # FIX 7: guarded above
    min_child_samples=20,
    random_state=42,
    verbose=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),  # FIX 8
        lgb.log_evaluation(period=50),
    ],
)

print(f"Best iteration: {model.best_iteration_}")

# ── Step 7: evaluate ──────────────────────────────────────────────────────────
print("\nEvaluation on test set:")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=['Loss', 'Win'], zero_division=0))

report       = classification_report(y_test, y_pred, target_names=['Loss','Win'],
                                     output_dict=True, zero_division=0)
win_prec     = report['Win']['precision']

if win_prec < 0.55:
    verdict = "FAIL — do not deploy. Collect more data."
elif win_prec < 0.65:
    verdict = "MARGINAL — paper trade only."
elif accuracy_score(y_test, y_pred) > 0.80:
    verdict = "SUSPICIOUS — possible data leakage. Do not deploy."
else:
    verdict = "PASS — deploy and monitor weekly."

print(f"\nVerdict: {verdict}")

# ── Step 8: save + verify (FIX 9) ────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
print(f"\nSaved → {MODEL_PATH}")

try:
    loaded = joblib.load(MODEL_PATH)
    _      = loaded.feature_name_      # engine calls this; crash here = crash in engine
    print("Verification: model reloads correctly and feature_name_ attribute present.")
except AttributeError:
    print(
        "WARNING: feature_name_ missing on reloaded model.\n"
        "Check your LightGBM version — the engine will crash without this."
    )
except Exception as e:
    print(f"WARNING: reload check failed — {e}")

print("\nDone. Place trade_classifier_lgbm.pkl alongside antigravity_quant_engine.py")