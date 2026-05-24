"""
trade_db.py
===========
SQLite trade journal database for OQPE dashboard.

RENDER DEPLOYMENT NOTE:
  The free tier has an ephemeral filesystem — trades.db is wiped on every
  redeploy or restart. To persist trades across deploys you have two options:

  Option A (recommended free): Upgrade to Render Starter ($7/mo) and attach
    a Persistent Disk. Set DB_PATH to the mounted disk path, e.g. /data/trades.db
    and set the disk mount point to /data in your Render dashboard.

  Option B (free): Use Render Postgres instead of SQLite.
    Replace the SQLite functions below with psycopg2 + DATABASE_URL env var.
    Render provides one free Postgres instance (expires after 30 days).

For local development, SQLite works perfectly with no changes needed.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional

# ── Path config ───────────────────────────────────────────────────────────────
# On Render with a persistent disk mounted at /data, set:
#   DB_PATH = os.environ.get("DB_PATH", "/data/trades.db")
# For local dev or free-tier ephemeral storage:
DB_PATH = os.environ.get("DB_PATH", "trades.db")


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# =============================================================================
# SCHEMA
# =============================================================================

def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                pair         TEXT    NOT NULL,
                timeframe    TEXT,
                signal       TEXT    NOT NULL,
                entry_price  REAL,
                stop_loss    REAL,
                take_profit  REAL,
                exit_price   REAL,
                confidence   REAL,
                risk_reward  REAL,
                status       TEXT    DEFAULT 'PENDING',
                outcome      TEXT,
                profit_loss  REAL,
                entry_time   TEXT,
                exit_time    TEXT,
                notes        TEXT
            )
        """)
        con.commit()


# =============================================================================
# WRITE OPERATIONS
# =============================================================================

def insert_trade(trade: dict) -> int:
    """Insert a new trade. Returns the new row id."""
    cols   = ", ".join(trade.keys())
    places = ", ".join("?" * len(trade))
    with _conn() as con:
        cur = con.execute(
            f"INSERT INTO trades ({cols}) VALUES ({places})",
            list(trade.values()),
        )
        con.commit()
        return cur.lastrowid


def update_trade(trade_id: int, updates: dict) -> None:
    """Patch specific columns on an existing trade row."""
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with _conn() as con:
        con.execute(
            f"UPDATE trades SET {set_clause} WHERE id = ?",
            [*updates.values(), trade_id],
        )
        con.commit()


def delete_trade(trade_id: int) -> None:
    with _conn() as con:
        con.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
        con.commit()


# =============================================================================
# READ OPERATIONS
# =============================================================================

def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    try:
        with _conn() as con:
            df = pd.read_sql_query(sql, con, params=params)
        return df
    except Exception:
        return pd.DataFrame()


def get_all_trades() -> pd.DataFrame:
    """All trades, newest first."""
    return _query("SELECT * FROM trades ORDER BY id DESC")


def get_open_trades() -> pd.DataFrame:
    """Active + pending trades."""
    return _query(
        "SELECT * FROM trades WHERE status IN ('ACTIVE','PENDING') ORDER BY id DESC"
    )


def get_active_trades() -> pd.DataFrame:
    return _query("SELECT * FROM trades WHERE status = 'ACTIVE' ORDER BY id DESC")


def get_pending_trades() -> pd.DataFrame:
    return _query("SELECT * FROM trades WHERE status = 'PENDING' ORDER BY id DESC")


def get_closed_trades() -> pd.DataFrame:
    return _query(
        "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY entry_time DESC"
    )


# =============================================================================
# STATISTICS
# =============================================================================

def calculate_stats() -> dict:
    """Summary stats for the analytics panel."""
    df = get_all_trades()
    closed = df[df["outcome"].notna()] if not df.empty else pd.DataFrame()

    if closed.empty:
        return {
            "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0,
        }

    wins   = closed[closed["outcome"] == "WIN"]
    losses = closed[closed["outcome"] == "LOSS"]

    gross_profit = float(wins["profit_loss"].sum())   if not wins.empty   else 0.0
    gross_loss   = abs(float(losses["profit_loss"].sum())) if not losses.empty else 0.0

    return {
        "total_trades":  len(closed),
        "win_rate":      round(len(wins) / len(closed) * 100, 1) if len(closed) > 0 else 0.0,
        "total_pnl":     round(float(closed["profit_loss"].sum()), 2),
        "avg_win":       round(float(wins["profit_loss"].mean()),   2) if not wins.empty   else 0.0,
        "avg_loss":      round(float(losses["profit_loss"].mean()), 2) if not losses.empty else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        "best_trade":    round(float(closed["profit_loss"].max()), 2) if not closed.empty else 0.0,
        "worst_trade":   round(float(closed["profit_loss"].min()), 2) if not closed.empty else 0.0,
    }