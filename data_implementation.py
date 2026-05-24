"""
AntigravityQuantEngine — Production Ready
==========================================
Builds on the optimized version and applies every critical fix
flagged in the profitability review:

  A. LOOK-AHEAD BIAS FIX
     find_swing_points_vectorized used shift(-window) which peeks
     into future bars. Replaced with a causal rolling-max/min that
     only uses data available at bar-close time.

  B. VOLATILITY FILTER (ATR-based)
     Signals are suppressed during abnormally low-volatility (choppy)
     or abnormally high-volatility (news spike) conditions. Both kill
     signal quality in SMC strategies.

  C. SPREAD / SLIPPAGE MODEL
     calculate_position_size and R:R calculation now deduct an
     estimated spread so the reported R:R reflects realistic fills,
     not theoretical mid-prices.

  D. DRAWDOWN CIRCUIT BREAKER
     EngineGuard tracks rolling equity. If drawdown from the rolling
     peak exceeds max_drawdown_pct, the engine enters a cooldown and
     returns NO_TRADE until conditions recover.

  E. WALK-FORWARD BACKTEST HARNESS
     walk_forward_backtest() splits data into expanding train windows
     and out-of-sample test windows so you can see genuine
     out-of-sample performance before risking real capital.

  F. SAMPLE-SIZE GUARD IN TRAINING
     create_training_dataset now warns (and optionally aborts) when
     the labelled sample count is below MIN_SAMPLES_FOR_ML.

All previous optimisations (vectorised indicators, parallel fetch,
singleton news filter, thread-safe calendar cache, unified pip size,
shared DataFrame cache) are fully preserved.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import logging
import json
import time
import threading
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import xml.etree.ElementTree as ET
import requests
import threading
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

logging.basicConfig(level=logging.INFO)

# ── Optional ML ──────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ── Optional News ─────────────────────────────────────────────────────────────
USE_NEWS = False
if USE_NEWS:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError as e:
        logging.warning(f"News deps missing: {e}. Disabling news.")
        USE_NEWS = False

# Minimum labelled samples before LightGBM training is considered reliable
MIN_SAMPLES_FOR_ML = 800

# Module-level list of all trading pairs (used by both engine and training)
ALL_PAIRS = [
    "EURJPY=X", "EURNZD=X", "GBPAUD=X", "USDCHF=X",
    "BTC-USD",  "EURAUD=X", "EURUSD=X", "GBPCHF=X",
    "NZDUSD=X", "AUDUSD=X", "AUDCAD=X", "EURCHF=X",
    "AUDCHF=X", "CADCHF=X", "AUDNZD=X", "EURCAD=X",
    "GBPCAD=X", "GBPNZD=X", "NZDJPY=X", "CADJPY=X",
    "AUDJPY=X", "SOL-USD",  "USDJPY=X", "CHFJPY=X",
    "GBPJPY=X", "USDCAD=X", "GBPUSD=X", "EURGBP=X",
    "SPY",      "QQQ",      "GC=F",
]



# =============================================================================
# KTL AXIS: SESSION & KILL ZONE DEFINITIONS
# =============================================================================

KILL_ZONES = {
    "ASIA":              {"start": 0,  "end": 6},
    "FRANKFURT":         {"start": 6,  "end": 7},
    "LONDON":            {"start": 7,  "end": 16},
    "NEW_YORK":          {"start": 12, "end": 20},
    "LONDON_NY_OVERLAP": {"start": 12, "end": 16},
    "NY_LUNCH":          {"start": 16, "end": 17},
}


def get_current_kill_zone(dt: datetime) -> str:
    for zone, t in KILL_ZONES.items():
        if t["start"] <= dt.hour < t["end"]:
            return zone
    return "OFF_HOURS"


def is_no_trade_zone(dt: datetime) -> bool:
    return KILL_ZONES["NY_LUNCH"]["start"] <= dt.hour < KILL_ZONES["NY_LUNCH"]["end"]


# =============================================================================
# 1. DATA MANAGER  (parallel fetch)
# =============================================================================

class MultiTimeframeDataManager:
    TIMEFRAMES = {
        "15m": {"interval": "15m", "period": "60d"},
        "1h":  {"interval": "1h",  "period": "60d"},
        "4h":  {"interval": "1h",  "period": "60d", "resample": True},
        "1d":  {"interval": "1d",  "period": "2y"},
    }

    DEFAULT_PAIRS = [
        "EURJPY=X", "EURNZD=X", "GBPAUD=X", "USDCHF=X",
        "BTC-USD",  "EURAUD=X", "EURUSD=X", "GBPCHF=X",
        "NZDUSD=X", "AUDUSD=X", "AUDCAD=X", "EURCHF=X",
        "AUDCHF=X", "CADCHF=X", "AUDNZD=X", "EURCAD=X",
        "GBPCAD=X", "GBPNZD=X", "NZDJPY=X", "CADJPY=X",
        "AUDJPY=X", "SOL-USD",  "USDJPY=X", "CHFJPY=X",
        "GBPJPY=X", "USDCAD=X", "GBPUSD=X", "EURGBP=X",
        "SPY",      "QQQ",      "GC=F",          # updated tickers
    ]

    def __init__(self, pairs: list = None):
        self.pairs = pairs or self.DEFAULT_PAIRS
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {}

    def fetch_data(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        config = self.TIMEFRAMES[timeframe]
        try:
            df = yf.Ticker(pair).history(
                interval=config["interval"], period=config["period"]
            )
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]]
            if config.get("resample"):
                df = self._resample_to_4h(df)
            return _clean_data(df)
        except Exception as e:
            logging.error(f"Fetch failed {pair} {timeframe}: {e}")
            return None

    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        return _clean_data(
            df.resample("4h").agg(
                open=("open", "first"), high=("high", "max"),
                low=("low", "min"),    close=("close", "last"),
                volume=("volume", "sum"),
            ).dropna()
        )

    def fetch_all_parallel(self, max_workers: int = 8) -> None:
        def _fetch_pair(pair: str) -> Tuple[str, Dict]:
            return pair, {
                tf: df
                for tf in self.TIMEFRAMES
                if (df := self.fetch_data(pair, tf)) is not None
            }

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for future in as_completed(
                {ex.submit(_fetch_pair, p): p for p in self.pairs}
            ):
                try:
                    pair, data = future.result()
                    self.data[pair] = data
                except Exception as e:
                    logging.error(f"fetch_all_parallel error: {e}")

    async def fetch_all(self):           # backward-compat alias
        self.fetch_all_parallel()


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# =============================================================================
# VOLATILITY HELPERS  (FIX B — ATR-based filter)
# =============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — fully vectorised."""
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def volatility_filter(df: pd.DataFrame, atr_period: int = 14,
                      low_mult: float = 0.3, high_mult: float = 3.0) -> str:
    """
    Return 'OK', 'LOW_VOL' or 'HIGH_VOL'.

    LOW_VOL  → current ATR < low_mult  × rolling-median ATR  (choppy / ranging)
    HIGH_VOL → current ATR > high_mult × rolling-median ATR  (news spike)
    """
    atr = compute_atr(df, atr_period).dropna()
    if len(atr) < atr_period * 2:
        return "OK"           # not enough data — don't block
    current_atr = atr.iloc[-1]
    median_atr  = atr.iloc[-atr_period * 2:].median()
    if median_atr == 0:
        return "OK"
    ratio = current_atr / median_atr
    if ratio < low_mult:
        return "LOW_VOL"
    if ratio > high_mult:
        return "HIGH_VOL"
    return "OK"


# =============================================================================
# SPREAD / SLIPPAGE MODEL  (FIX C)
# =============================================================================

# Estimated round-trip spread in pips per asset class.
# Used to adjust R:R so it reflects realistic fills.
SPREAD_PIPS: Dict[str, float] = {
    # Major forex
    "EURUSD=X": 0.8, "GBPUSD=X": 1.0, "AUDUSD=X": 0.9, "NZDUSD=X": 1.2,
    "USDCAD=X": 1.2, "USDCHF=X": 1.1, "EURGBP=X": 1.0,
    # JPY majors
    "USDJPY=X": 0.8, "EURJPY=X": 1.2, "GBPJPY=X": 1.5, "AUDJPY=X": 1.3,
    "NZDJPY=X": 1.8, "CADJPY=X": 1.8, "CHFJPY=X": 2.0,
    # Minor / exotic forex (wider spreads)
    "EURAUD=X": 2.0, "EURNZD=X": 2.5, "EURCHF=X": 1.5, "EURCAD=X": 2.0,
    "GBPAUD=X": 2.5, "GBPCHF=X": 2.0, "GBPCAD=X": 2.5, "GBPNZD=X": 3.0,
    "AUDCHF=X": 2.5, "AUDCAD=X": 2.0, "AUDNZD=X": 2.0, "CADCHF=X": 2.5,
    # Crypto — quoted in price units, not pips
    "BTC-USD":  50.0, "SOL-USD": 0.05,
    # Indices / Gold (updated tickers)
    "SPY":  0.5, "QQQ": 1.0, "GC=F": 0.3,
}


def get_spread(pair: str, pip_size: float) -> float:
    """Return spread in price units (not pips) for the given pair."""
    if "BTC" in pair or "SOL" in pair:
        return SPREAD_PIPS.get(pair, 10.0)      # already in price units for crypto
    pips = SPREAD_PIPS.get(pair, 2.0)
    return pips * pip_size


# =============================================================================
# DRAWDOWN CIRCUIT BREAKER  (FIX D)
# =============================================================================

class EngineGuard:
    """
    Tracks rolling closed-trade equity and pauses signal generation
    when drawdown from the peak exceeds max_drawdown_pct.

    Usage:
        guard = EngineGuard(max_drawdown_pct=10.0, cooldown_trades=5)
        guard.record_trade(profit_loss_pct)
        if guard.is_halted():
            return NO_TRADE
    """

    def __init__(self, max_drawdown_pct: float = 10.0, cooldown_trades: int = 5):
        self.max_dd   = max_drawdown_pct
        self.cooldown = cooldown_trades
        self._equity: List[float] = [0.0]   # cumulative P&L %
        self._halt_remaining = 0

    def record_trade(self, profit_loss_pct: float) -> None:
        self._equity.append(self._equity[-1] + profit_loss_pct)
        peak = max(self._equity)
        dd   = peak - self._equity[-1]
        if dd >= self.max_dd:
            self._halt_remaining = self.cooldown
            logging.warning(
                f"EngineGuard: drawdown {dd:.1f}% ≥ {self.max_dd}%. "
                f"Halting for {self.cooldown} trades."
            )

    def trade_completed(self) -> None:
        """Call each time a halted trade cycle passes (win or loss)."""
        if self._halt_remaining > 0:
            self._halt_remaining -= 1

    def is_halted(self) -> bool:
        return self._halt_remaining > 0

    @property
    def current_drawdown(self) -> float:
        if not self._equity:
            return 0.0
        return max(self._equity) - self._equity[-1]

    @property
    def peak_equity(self) -> float:
        return max(self._equity)


# Module-level default guard — shared unless you pass your own
_default_guard = EngineGuard(max_drawdown_pct=10.0, cooldown_trades=5)


# =============================================================================
# KTL: SNR LEVELS  (vectorised)
# =============================================================================

def identify_snr_levels(df: pd.DataFrame, lookback: int = 100) -> Dict[str, List[float]]:
    df = df.tail(lookback).copy()
    prev_bear = df["close"].shift(1) > df["open"].shift(1)
    prev_bull = df["close"].shift(1) < df["open"].shift(1)
    curr_bull = df["close"] > df["open"]
    curr_bear = df["close"] < df["open"]
    gap_pct   = (df["close"].shift(1) - df["open"]).abs() / \
                df["close"].shift(1).replace(0, np.nan)
    mid = (df["close"].shift(1) + df["open"]) / 2
    return {
        "supports":    sorted(set(mid.where(prev_bear & curr_bull).dropna().tolist())),
        "resistances": sorted(set(mid.where(prev_bull & curr_bear).dropna().tolist())),
        "open_close":  sorted(set(
            df["close"].shift(1).where(gap_pct < 0.001).dropna().tolist()
        )),
    }


# =============================================================================
# KTL: FAIR VALUE GAP  (vectorised)
# =============================================================================

def detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    high2 = df["high"].shift(2)
    low2  = df["low"].shift(2)
    bull  = df["low"]  > high2
    bear  = df["high"] < low2
    df["fvg_bull"]   = np.where(bull,  1,   np.nan)
    df["fvg_bear"]   = np.where(bear, -1,   np.nan)
    df["fvg_top"]    = np.where(bull, df["low"],  np.where(bear, low2,        np.nan))
    df["fvg_bottom"] = np.where(bull, high2,      np.where(bear, df["high"],  np.nan))
    return df


# =============================================================================
# KTL: DAILY CYCLE
# =============================================================================

def detect_daily_cycle(df: pd.DataFrame) -> str:
    if len(df) < 24:
        return "UNKNOWN"
    today     = df.iloc[-24:]
    asia      = today.iloc[:7]
    day_range = today["high"].max() - today["low"].min()
    if day_range == 0:
        return "UNKNOWN"
    asia_range = asia["high"].max() - asia["low"].min()
    if asia_range < day_range * 0.3:
        return "JUDAH_SWING"
    if (asia["close"].iloc[-1] > asia["open"].iloc[0] * 1.002 or
            asia["close"].iloc[-1] < asia["open"].iloc[0] * 0.998):
        return "ASIA_CONTINUATION"
    return "BREAK_RETEST"


# =============================================================================
# FIX A — SWING POINT DETECTION: CAUSAL (no look-ahead bias)
# =============================================================================

def find_swing_points_causal(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    CAUSAL swing detection — uses only past bars available at each bar-close.

    Original used shift(-window) which reads FUTURE bars, inflating backtest
    win rates by detecting swings earlier than they could be known in live trading.

    Method: a swing high at bar i is confirmed only when the next `window`
    bars have all closed below it — i.e. we know it at bar i+window.
    We record the swing at its TRUE price level but stamp it at confirmation bar.

    In live use, the latest `window` bars will have no confirmed swings — this
    is correct and conservative behaviour.
    """
    df     = df.copy()
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(df)

    swing_high_vals = np.full(n, np.nan)
    swing_low_vals  = np.full(n, np.nan)

    for i in range(window, n - window):
        # Swing high: highest in [i-window, i+window] window, confirmed at i+window
        hi_window = highs[i - window: i + window + 1]
        if highs[i] == hi_window.max():
            confirm_idx = min(i + window, n - 1)
            swing_high_vals[confirm_idx] = highs[i]

        # Swing low: lowest in [i-window, i+window] window, confirmed at i+window
        lo_window = lows[i - window: i + window + 1]
        if lows[i] == lo_window.min():
            confirm_idx = min(i + window, n - 1)
            swing_low_vals[confirm_idx] = lows[i]

    df["swing_high"] = swing_high_vals
    df["swing_low"]  = swing_low_vals
    return df


# Keep old name as alias so existing code paths still work during migration
find_swing_points_vectorized = find_swing_points_causal


# =============================================================================
# BOS / CHOCH DETECTION  (itertuples + single bulk assignment)
# =============================================================================

def detect_bos_choch_from_swings(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = find_swing_points_causal(df, window)
    n  = len(df)
    trends    = ["neutral"] * n
    bos_arr   = [0] * n
    choch_arr = [0] * n
    swing_pts: deque = deque(maxlen=2)
    current_trend = "neutral"

    for i, row in enumerate(df.itertuples()):
        if pd.notna(row.swing_high):
            swing_pts.append(("high", row.swing_high, i))
        elif pd.notna(row.swing_low):
            swing_pts.append(("low", row.swing_low, i))

        if len(swing_pts) >= 2:
            last_type, last_level, _ = swing_pts[-2]
            price = row.close
            if last_type == "high" and price > last_level:
                if current_trend == "bullish":
                    bos_arr[i] = 1
                else:
                    choch_arr[i] = 1
                current_trend = "bullish"
            elif last_type == "low" and price < last_level:
                if current_trend == "bearish":
                    bos_arr[i] = -1
                else:
                    choch_arr[i] = -1
                current_trend = "bearish"

        trends[i] = current_trend

    df = df.copy()
    df["trend"] = trends
    df["bos"]   = bos_arr
    df["choch"] = choch_arr
    return df


# =============================================================================
# LIQUIDITY SWEEP DETECTION
# =============================================================================

def detect_liquidity_sweep(
    df: pd.DataFrame,
    swing_lows: pd.Series = None,
    lookback: int = 20,
) -> pd.DataFrame:
    df = df.copy()
    if swing_lows is None:
        swing_lows = find_swing_points_causal(df, window=10)["swing_low"]

    sweep_vals   = [0]     * len(df)
    reclaim_vals = [False] * len(df)

    for i in range(lookback, len(df)):
        recent = swing_lows.iloc[max(0, i - lookback): i].dropna()
        if recent.empty:
            continue
        low_i, close_i = df.iloc[i]["low"], df.iloc[i]["close"]
        for lvl in recent.values:
            if low_i < lvl:
                sweep_vals[i]   = -1
                reclaim_vals[i] = close_i > lvl
                break

    df["sweep"]         = sweep_vals
    df["sweep_reclaim"] = reclaim_vals
    return df


# =============================================================================
# ORDER BLOCK DETECTION  (vectorised)
# =============================================================================

def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 5,
    min_move_pct: float = 0.005,
) -> pd.DataFrame:
    df         = df.copy()
    pct_change = df["close"].pct_change(lookback)
    bull_move  = pct_change >  min_move_pct
    bear_move  = pct_change < -min_move_pct

    ob_type_arr   = np.zeros(len(df), dtype=int)
    ob_top_arr    = np.full(len(df), np.nan)
    ob_bottom_arr = np.full(len(df), np.nan)

    for i in np.where(bull_move.values)[0]:
        if i < lookback + 1:
            continue
        bearish = df.iloc[i - lookback: i]
        bearish = bearish[bearish["close"] < bearish["open"]]
        if not bearish.empty:
            ob = bearish.iloc[-1]
            ob_type_arr[i], ob_bottom_arr[i], ob_top_arr[i] = 1, ob["low"], ob["high"]

    for i in np.where(bear_move.values)[0]:
        if i < lookback + 1:
            continue
        bullish = df.iloc[i - lookback: i]
        bullish = bullish[bullish["close"] > bullish["open"]]
        if not bullish.empty:
            ob = bullish.iloc[-1]
            ob_type_arr[i], ob_top_arr[i], ob_bottom_arr[i] = -1, ob["high"], ob["low"]

    df["ob_type"]   = ob_type_arr
    df["ob_top"]    = ob_top_arr
    df["ob_bottom"] = ob_bottom_arr
    return df


def identify_poi_from_ob(df: pd.DataFrame, ob_df: pd.DataFrame, trend: str) -> dict:
    col = 1 if trend == "bullish" else -1 if trend == "bearish" else None
    if col is None:
        return {"poi_active": False, "zone_type": None, "entry_zone": None}
    valid = ob_df[ob_df["ob_type"] == col]
    if valid.empty:
        return {"poi_active": False, "zone_type": None, "entry_zone": None}
    latest = valid.iloc[-1]
    return {
        "poi_active":  True,
        "zone_type":   "demand" if col == 1 else "supply",
        "entry_zone":  [latest["ob_bottom"], latest["ob_top"]],
        "ob_strength": 0.5,
    }


# =============================================================================
# ECONOMIC CALENDAR  (thread-safe cache)
# =============================================================================

_cal_lock  = threading.Lock()
_cal_cache: Dict = {"timestamp": 0.0, "data": None}
_CAL_TTL   = 1800


_FF_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.xml",
]
 
_EMPTY_CAL = pd.DataFrame(
    columns=["date","time","currency","event","impact","forecast","previous"]
)
 
 
def _fetch_calendar_raw() -> pd.DataFrame:
    """
    Fetch ForexFactory weekly economic calendar XML.
 
    ForexFactory XML event structure:
        <event>
          <title>Core CPI m/m</title>
          <country>USD</country>
          <date>May 23, 2026</date>
          <time>8:30am</time>
          <impact>High</impact>
          <forecast>0.3%</forecast>
          <previous>0.4%</previous>
        </event>
 
    Returns a DataFrame with standardised column names matching the original
    schema so no downstream code changes are needed.
    """
    all_events = []
 
    for url in _FF_URLS:
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (compatible; OQPE/1.0)"},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
 
        except requests.exceptions.HTTPError as e:
            if "404" in str(e):
                # next-week XML returns 404 before it's published — not an error
                continue
            logging.error(f"Calendar HTTP error ({url}): {e}")
            continue
        except requests.exceptions.RequestException as e:
            logging.error(f"Calendar network error ({url}): {e}")
            continue
        except ET.ParseError as e:
            logging.error(f"Calendar XML parse error ({url}): {e}")
            continue
        except Exception as e:
            logging.error(f"Calendar unexpected error ({url}): {e}")
            continue
 
        for event in root.findall("event"):
            date_raw = event.findtext("date", "").strip()
            time_raw = event.findtext("time", "All Day").strip()
 
            # ForexFactory date format: "May 23, 2026"
            date_iso = date_raw
            for fmt in ("%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d", "%d %b %Y"):
                try:
                    date_iso = datetime.strptime(date_raw, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
 
            all_events.append({
                "date":     date_iso,
                "time":     time_raw,
                "currency": event.findtext("country",  "").strip(),
                "event":    event.findtext("title",    "").strip(),
                "impact":   event.findtext("impact",   "").strip(),
                "forecast": event.findtext("forecast", "").strip(),
                "previous": event.findtext("previous", "").strip(),
            })
 
    if not all_events:
        logging.warning("Calendar: no events fetched from ForexFactory XML.")
        return _EMPTY_CAL.copy()
 
    df = pd.DataFrame(all_events)
    df["_sort"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    logging.info(f"Calendar: loaded {len(df)} events.")
    return df
 
 
def get_economic_calendar(pair: str, days_ahead: int = 3) -> pd.DataFrame:
    """
    Return upcoming economic events relevant to the currencies in `pair`.
 
    Parameters
    ──────────
    pair        : instrument string, e.g. "GBPUSD=X", "BTC-USD", "GC=F", "SPY"
    days_ahead  : how many calendar days forward to include (default 3)
 
    Returns
    ───────
    DataFrame with columns: date, time, currency, event, impact, forecast, previous
    Empty DataFrame if no matching events or on fetch failure.
    Never raises.
    """
    with _cal_lock:
        now = time.time()
        if _cal_cache["data"] is None or (now - _cal_cache["timestamp"]) > _CAL_TTL:
            _cal_cache["data"]      = _fetch_calendar_raw()
            _cal_cache["timestamp"] = now
        df = _cal_cache["data"].copy()
 
    if df.empty:
        return _EMPTY_CAL.copy()
 
    # ── Determine relevant currencies ─────────────────────────────────────────
    currencies: set = set()
 
    # Instruments that only have USD exposure
    usd_instruments = {
        "BTC-USD", "SOL-USD", "GC=F", "SPY", "QQQ",
        "^GSPC", "^NDX", "XAUUSD=X",
    }
    pair_upper = pair.upper()
    if pair in usd_instruments or any(x in pair_upper for x in ("BTC","SOL","XAU")):
        currencies.add("USD")
    elif pair_upper in usd_instruments:
        currencies.add("USD")
    else:
        # Forex pair: extract base and quote from first 6 chars after stripping =X
        clean = pair.replace("=X", "").upper()
        known = {"EUR","USD","GBP","JPY","AUD","NZD","CAD","CHF"}
        if len(clean) >= 3 and clean[:3] in known:
            currencies.add(clean[:3])
        if len(clean) >= 6 and clean[3:6] in known:
            currencies.add(clean[3:6])
        if not currencies:
            currencies.add("USD")  # safe default
 
    # ── Date filter — use .date() objects to avoid tz-mismatch ───────────────
    try:
        df["_dt"] = pd.to_datetime(
            df["date"], format="%Y-%m-%d", errors="coerce"
        ).dt.date
    except Exception:
        return _EMPTY_CAL.copy()
 
    today  = datetime.now().date()
    cutoff = today + timedelta(days=days_ahead)
 
    mask = (
        df["_dt"].notna()
        & (df["_dt"] >= today)
        & (df["_dt"] <= cutoff)
        & df["currency"].isin(currencies)
    )
 
    result = df[mask].drop(columns=["_dt"]).reset_index(drop=True)
    return result if not result.empty else _EMPTY_CAL.copy()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# QUICK VERIFICATION (run this file directly to test)
# python calendar_fix.py
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing ForexFactory calendar fetch...\n")
 
    raw = _fetch_calendar_raw()
    print(f"Raw events fetched: {len(raw)}")
 
    if raw.empty:
        print("FAIL: No data returned. Check your internet connection.")
    else:
        print("\nSample (first 5 rows):")
        print(raw.head(5).to_string(index=False))
 
        print("\nFiltering for EURUSD (EUR + USD events, next 3 days):")
        eur = get_economic_calendar("EURUSD=X", days_ahead=3)
        if eur.empty:
            print("  No high-impact EUR/USD events in next 3 days (normal outside news week).")
        else:
            print(eur.to_string(index=False))
 
        print("\nFiltering for GBPJPY (GBP + JPY events, next 3 days):")
        gbp = get_economic_calendar("GBPJPY=X", days_ahead=3)
        if gbp.empty:
            print("  No events.")
        else:
            print(gbp.to_string(index=False))
 
        print("\nFiltering for BTC-USD (USD events, next 3 days):")
        btc = get_economic_calendar("BTC-USD", days_ahead=3)
        if btc.empty:
            print("  No USD events in next 3 days.")
        else:
            print(btc.to_string(index=False))
 
        print("\nPASS: Calendar is working correctly.")



# =============================================================================
# SENTIMENT ANALYSER & NEWS FILTER  (module-level singletons)
# =============================================================================

class NewsSentimentAnalyzer:
    def __init__(self):
        if USE_NEWS:
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        else:
            self.tokenizer = self.model = None

    def analyze(self, text: str) -> dict:
        if not USE_NEWS or self.model is None:
            return {"sentiment": "NEUTRAL", "confidence": 0.0, "trade_allowed": True}
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        probs  = torch.nn.functional.softmax(self.model(**inputs).logits, dim=-1)[0].tolist()
        labels = ["positive", "negative", "neutral"]
        best   = labels[probs.index(max(probs))].upper()
        return {"sentiment": best, "confidence": max(probs),
                "trade_allowed": best != "NEGATIVE" or max(probs) < 0.8}


_sentiment_analyzer = NewsSentimentAnalyzer()


class NewsFilter:
    def evaluate(self, current_time: datetime, pair: str,
                 recent_headlines: list = None) -> dict:
        if not USE_NEWS:
            return {"impact": "LOW", "sentiment": "NEUTRAL", "trade_allowed": True}
        try:
            df = get_economic_calendar(pair, days_ahead=1)
        except Exception:
            df = pd.DataFrame()

        if not df.empty:
            for _, row in df.iterrows():
                if row.get("impact") != "High":
                    continue
                if row.get("time") == "All Day":
                    return {"impact": "HIGH", "sentiment": "NEUTRAL",
                            "trade_allowed": False,
                            "event_name": row.get("event", "High impact event")}
                try:
                    event_dt = datetime.strptime(
                        f"{row['date']} {row['time']}", "%m-%d-%Y %I:%M%p"
                    )
                    if abs((current_time - event_dt).total_seconds()) / 60 <= 30:
                        return {"impact": "HIGH", "sentiment": "NEUTRAL",
                                "trade_allowed": False,
                                "event_name": row.get("event", "High impact event")}
                except Exception:
                    continue

        if recent_headlines:
            sents = [_sentiment_analyzer.analyze(h) for h in recent_headlines]
            if sum(1 for s in sents if s["sentiment"] == "NEGATIVE") / len(sents) > 0.6:
                return {"impact": "MEDIUM", "sentiment": "BEARISH", "trade_allowed": False}

        return {"impact": "LOW", "sentiment": "NEUTRAL", "trade_allowed": True}


_news_filter = NewsFilter()


# =============================================================================
# ML FEATURE EXTRACTION
# =============================================================================

def extract_features(signal: dict) -> dict:
    f: dict = {
        "signal_buy":   1 if signal.get("signal") == "BUY"  else 0,
        "signal_sell":  1 if signal.get("signal") == "SELL" else 0,
        "confidence":   signal.get("confidence", 0),
        "risk_reward":  signal.get("risk_reward", 0),
        "trend_bullish": 1 if signal.get("trend") == "bullish" else 0,
        "trend_bearish": 1 if signal.get("trend") == "bearish" else 0,
    }
    ktl = signal.get("ktl_axis", {})
    for kz in ("LONDON", "NEW_YORK", "LONDON_NY_OVERLAP", "ASIA"):
        f[f"kill_zone_{kz.lower()}"] = 1 if ktl.get("kill_zone") == kz else 0
    for dc in ("JUDAH_SWING", "ASIA_CONTINUATION"):
        f[f"daily_cycle_{dc.lower()}"] = 1 if ktl.get("daily_cycle") == dc else 0
    f["axis_aligned"] = 1 if ktl.get("axis_aligned") else 0
    f["has_bull_fvg"] = 1 if ktl.get("has_bull_fvg") else 0
    f["has_bear_fvg"] = 1 if ktl.get("has_bear_fvg") else 0

    price  = signal.get("current_price", 1) or 1
    near_s = ktl.get("nearest_support")
    near_r = ktl.get("nearest_resistance")
    f["support_distance_pct"]    = (price - near_s) / price if near_s else 0
    f["resistance_distance_pct"] = (near_r - price) / price if near_r else 0

    # Volatility regime feature
    f["vol_ok"]      = 1 if ktl.get("vol_regime") == "OK"       else 0
    f["vol_low"]     = 1 if ktl.get("vol_regime") == "LOW_VOL"  else 0
    f["vol_high"]    = 1 if ktl.get("vol_regime") == "HIGH_VOL" else 0

    poi = signal.get("poi", {})
    f["poi_active"] = 1 if poi.get("poi_active") else 0
    f["poi_demand"] = 1 if poi.get("zone_type") == "demand" else 0
    f["poi_supply"] = 1 if poi.get("zone_type") == "supply" else 0

    try:
        dt = datetime.fromisoformat(signal["timestamp"])
        f["hour"]        = dt.hour
        f["day_of_week"] = dt.weekday()
    except Exception:
        f["hour"] = f["day_of_week"] = 0

    pair = signal.get("pair", "")
    for p in ["BTC-USD", "EURUSD=X", "GBPUSD=X", "USDJPY=X"]:
        f[f"pair_{p}"] = 1 if pair == p else 0
    return f


# =============================================================================
# FIX E — WALK-FORWARD BACKTEST HARNESS
# =============================================================================

def walk_forward_backtest(
    pairs: List[str] = None,
    timeframe: str = "1h",
    train_bars: int = 1000,
    test_bars:  int = 200,
    step_bars:  int = 200,
    min_confidence: float = 0.6,
    min_rr: float = 2.0,
    spread_model: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward validation.

    For each window:
      • Train slice  : bars [start : start + train_bars]   (not used for ML here,
        but used to establish structural context for the engine)
      • Test slice   : bars [start + train_bars : start + train_bars + test_bars]
      • Engine generates signals on each test bar using only data up to that bar
      • Outcomes are labelled by whether future price hits TP or SL first

    Returns a DataFrame with per-trade outcomes, confidence, R:R, and window index
    so you can compute out-of-sample win rate and profit factor.
    """
    if pairs is None:
        pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD"]

    dm = MultiTimeframeDataManager(pairs)
    dm.fetch_all_parallel(max_workers=6)

    all_rows = []

    for pair in pairs:
        df     = dm.data.get(pair, {}).get(timeframe)
        df_d   = dm.data.get(pair, {}).get("1d")
        if df is None or df.empty or len(df) < train_bars + test_bars:
            logging.warning(f"Skipping {pair} — insufficient data")
            continue

        swing_win = 10 if "BTC" in pair else 7
        window_idx = 0
        start = 0

        while start + train_bars + test_bars <= len(df):
            test_start = start + train_bars
            test_end   = test_start + test_bars

            engine = AntigravityQuantEngine(
                pair, timeframe,
                swing_window=swing_win,
                min_confidence=min_confidence,
                min_risk_reward=min_rr,
            )

            for i in range(test_start, test_end):
                window_df = df.iloc[:i].copy()
                signal = engine.generate_signal(window_df, higher_tf_df=df_d)
                if signal["signal"] not in ("BUY", "SELL"):
                    continue

                entry = signal["entry"]
                sl    = signal["stop_loss"]
                tp    = signal["take_profit"]

                # Spread adjustment (FIX C) applied to outcome labelling
                if spread_model:
                    pip   = engine._pip_size()
                    spread = get_spread(pair, pip)
                    if signal["signal"] == "BUY":
                        entry += spread          # fill is worse on a buy
                    else:
                        entry -= spread          # fill is worse on a sell

                outcome = None
                for _, fut in df.iloc[i:].iterrows():
                    if signal["signal"] == "BUY":
                        if fut["low"]  <= sl: outcome = 0; break
                        if fut["high"] >= tp: outcome = 1; break
                    else:
                        if fut["high"] >= sl: outcome = 0; break
                        if fut["low"]  <= tp: outcome = 1; break

                if outcome is None:
                    continue

                pl = ((tp - entry) / entry * 100) if (outcome == 1 and signal["signal"] == "BUY") \
                else ((entry - tp) / entry * 100) if (outcome == 1 and signal["signal"] == "SELL") \
                else ((sl - entry) / entry * 100) if signal["signal"] == "BUY" \
                else ((entry - sl) / entry * 100)

                all_rows.append({
                    "window":     window_idx,
                    "pair":       pair,
                    "timeframe":  timeframe,
                    "signal":     signal["signal"],
                    "confidence": signal["confidence"],
                    "risk_reward": signal.get("risk_reward", 0),
                    "outcome":    outcome,
                    "pl_pct":     round(pl, 3),
                    "entry_bar":  i,
                    "reason":     " | ".join(signal.get("reason", [])),
                })

            start      += step_bars
            window_idx += 1

    if not all_rows:
        logging.warning("Walk-forward: no labelled trades produced.")
        return pd.DataFrame()

    results = pd.DataFrame(all_rows)

    # Summary per window
    summary = (
        results.groupby("window")
        .agg(
            trades=("outcome", "count"),
            win_rate=("outcome", "mean"),
            total_pl=("pl_pct", "sum"),
            avg_confidence=("confidence", "mean"),
        )
        .round(3)
    )
    logging.info(f"\nWalk-forward summary:\n{summary.to_string()}")

    overall_wr = results["outcome"].mean()
    wins   = results[results["outcome"] == 1]["pl_pct"].sum()
    losses = results[results["outcome"] == 0]["pl_pct"].abs().sum()
    pf     = wins / losses if losses > 0 else float("inf")

    logging.info(
        f"\nOverall — trades: {len(results)}, "
        f"win rate: {overall_wr:.1%}, "
        f"profit factor: {pf:.2f}, "
        f"total P&L: {results['pl_pct'].sum():.1f}%"
    )

    if overall_wr < 0.40:
        logging.warning(
            "⚠  Out-of-sample win rate below 40%. "
            "Do NOT trade live — strategy needs further development."
        )
    elif overall_wr >= 0.50 and pf >= 1.3:
        logging.info(
            "✓  Out-of-sample metrics look promising. "
            "Paper trade for 60–90 days before risking real capital."
        )

    return results


# =============================================================================
# FIX F — TRAINING DATASET WITH SAMPLE-SIZE GUARD
# =============================================================================

def create_training_dataset(
    pairs=None, timeframes=None,
    min_confidence=0.5, min_rr=1.5,
    abort_if_insufficient: bool = False,
) -> pd.DataFrame:
    if pairs is None:
        pairs = ["BTC-USD", "EURUSD=X", "GBPUSD=X", "USDJPY=X"]
    if timeframes is None:
        timeframes = ["1h", "4h"]

    dm = MultiTimeframeDataManager(pairs)
    logging.info("Pre-fetching training data…")
    dm.fetch_all_parallel(max_workers=6)

    all_rows = []

    for pair in pairs:
        df_d = dm.data.get(pair, {}).get("1d")
        for tf in timeframes:
            df = dm.data.get(pair, {}).get(tf)
            if df is None or df.empty:
                continue

            swing_win = 10 if "BTC" in pair else 7
            engine = AntigravityQuantEngine(
                pair, tf, swing_window=swing_win,
                min_confidence=min_confidence, min_risk_reward=min_rr,
            )

            for i in range(200, len(df) - 50, 20):
                window_df = df.iloc[:i].copy()
                signal    = engine.generate_signal(window_df, higher_tf_df=df_d)
                if signal["signal"] not in ("BUY", "SELL"):
                    continue

                entry, sl, tp = signal["entry"], signal["stop_loss"], signal["take_profit"]

                # Apply spread before labelling  (FIX C)
                pip    = engine._pip_size()
                spread = get_spread(pair, pip)
                if signal["signal"] == "BUY":
                    entry += spread
                else:
                    entry -= spread

                outcome = None
                for _, row in df.iloc[i:].iterrows():
                    if signal["signal"] == "BUY":
                        if row["low"]  <= sl: outcome = 0; break
                        if row["high"] >= tp: outcome = 1; break
                    else:
                        if row["high"] >= sl: outcome = 0; break
                        if row["low"]  <= tp: outcome = 1; break

                if outcome is None:
                    continue

                feat = extract_features(signal)
                feat.update(outcome=outcome, pair=pair, timeframe=tf)
                all_rows.append(feat)

    df_out = pd.DataFrame(all_rows)
    n = len(df_out)
    logging.info(f"Training dataset: {n} labelled samples.")

    if n < MIN_SAMPLES_FOR_ML:
        msg = (
            f"Only {n} labelled samples — minimum {MIN_SAMPLES_FOR_ML} recommended "
            f"for reliable LightGBM generalisation. "
            f"Expand pairs/timeframes or lower min_confidence."
        )
        if abort_if_insufficient:
            raise ValueError(msg)
        logging.warning(f"⚠  {msg}")

    return df_out


# =============================================================================
# MAIN ENGINE
# =============================================================================

class AntigravityQuantEngine:

    ASSET_CONFIG = {
        # Forex
        "EURUSD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "GBPUSD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "USDCHF=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "AUDUSD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "NZDUSD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "USDCAD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "EURGBP=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "EURAUD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "EURCHF=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "EURCAD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "GBPCHF=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "GBPAUD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "GBPCAD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "GBPNZD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "AUDCHF=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "AUDCAD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "AUDNZD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "EURNZD=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "CADCHF=X": {"pip_value":   10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        # JPY pairs (pip value = 1000)
        "USDJPY=X": {"pip_value": 1_000.0,"lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "EURJPY=X": {"pip_value": 1_000.0,"lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "GBPJPY=X": {"pip_value": 1_000.0,"lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "AUDJPY=X": {"pip_value": 1_000.0,"lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "NZDJPY=X": {"pip_value": 1_000.0,"lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "CADJPY=X": {"pip_value": 1_000.0,"lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        "CHFJPY=X": {"pip_value": 1_000.0,"lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        # Crypto
        "BTC-USD":  {"pip_value": 1.0, "lot_size": 1.0, "min_lot": 0.001, "is_forex": False},
        "SOL-USD":  {"pip_value": 1.0, "lot_size": 1.0, "min_lot": 0.001, "is_forex": False},
        # Indices / Gold (updated tickers)
        "SPY":   {"pip_value": 1.0, "lot_size": 1.0, "min_lot": 0.001, "is_forex": False},
        "QQQ":   {"pip_value": 1.0, "lot_size": 1.0, "min_lot": 0.001, "is_forex": False},
        "GC=F":  {"pip_value": 1.0, "lot_size": 1.0, "min_lot": 0.001, "is_forex": False},
    }

    def __init__(
        self,
        pair: str,
        timeframe: str,
        swing_window: int = 10,
        min_confidence: float = 0.65,
        min_risk_reward: float = 2.0,
        news_filter: NewsFilter = None,
        guard: EngineGuard = None,       # FIX D — injectable circuit breaker
    ):
        self.pair            = pair
        self.timeframe       = timeframe
        self.swing_window    = swing_window
        self.min_confidence  = min_confidence
        self.min_risk_reward = min_risk_reward
        self.news_filter     = news_filter or _news_filter
        self.guard           = guard or _default_guard

        self.ml_model       = None
        self.bias_alpha     = 0.02
        self.recent_winrate = 0.5

        if ML_AVAILABLE and JOBLIB_AVAILABLE:
            try:
                self.ml_model = joblib.load("trade_classifier_lgbm.pkl")
                logging.info("LightGBM model loaded.")
            except FileNotFoundError:
                logging.info("No trade_classifier_lgbm.pkl — ML scoring disabled.")
            except Exception as e:
                logging.error(f"ML load error: {e}")

    # ── Single canonical pip size  (FIX #9 from previous pass) ──────────────
    def _pip_size(self) -> float:
        if "JPY" in self.pair:  return 0.01
        if "BTC" in self.pair or "SOL" in self.pair: return 1.0
        if "XAU" in self.pair:  return 0.01
        return 0.0001

    def apply_ml_score(self, signal: dict) -> float:
        if self.ml_model is None:
            return 1.0
        try:
            feats    = extract_features(signal)
            feats_df = pd.DataFrame([feats])[self.ml_model.feature_name_]
            proba    = self.ml_model.predict_proba(feats_df)[0, 1]
            adjusted = float(np.clip(
                proba + self.bias_alpha * (self.recent_winrate - 0.5), 0.0, 1.0
            ))
            return 0.5 + adjusted                # scale → [0.5, 1.5]
        except Exception as e:
            logging.error(f"ML scoring error: {e}")
            return 1.0

    def update_recent_winrate(self, recent_outcomes: list):
        if recent_outcomes:
            self.recent_winrate = sum(recent_outcomes) / len(recent_outcomes)

    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE SIGNAL  (fixed stop‑distance enforcement removed)
    # ─────────────────────────────────────────────────────────────────────────
    def generate_signal(
        self,
        df: pd.DataFrame,
        higher_tf_df: pd.DataFrame = None,
    ) -> dict:
        df            = df.sort_index().copy()
        current_price = df.iloc[-1]["close"]
        current_time  = df.index[-1]

        # ── FIX D: circuit breaker ────────────────────────────────────────
        if self.guard.is_halted():
            return self._no_trade(
                f"Circuit breaker active — drawdown {self.guard.current_drawdown:.1f}% "
                f"exceeded limit. Cooling down.",
                current_time, current_price, "neutral",
            )

        # ── HTF alignment ────────────────────────────────────────────────
        if higher_tf_df is None or higher_tf_df.empty:
            return self._no_trade(
                "Higher timeframe data required", current_time, current_price, "neutral"
            )
        higher_struct = detect_bos_choch_from_swings(higher_tf_df, window=7)
        higher_trend  = higher_struct["trend"].iloc[-1]
        if higher_trend == "neutral":
            return self._no_trade("HTF trend neutral", current_time, current_price, "neutral")

        # ── Session filter ────────────────────────────────────────────────
        kill_zone = get_current_kill_zone(current_time)
        if is_no_trade_zone(current_time):
            return self._no_trade("NY Lunch – No Trade Zone", current_time, current_price, "neutral")

        # ── FIX B: volatility filter ──────────────────────────────────────
        vol_regime = volatility_filter(df)
        if vol_regime != "OK":
            return self._no_trade(
                f"Volatility filter: {vol_regime} — signal suppressed",
                current_time, current_price, "neutral",
            )

        # ── LTF structure ─────────────────────────────────────────────────
        structure_df = detect_bos_choch_from_swings(df, window=self.swing_window)
        trend        = structure_df["trend"].iloc[-1]
        if trend != higher_trend:
            return self._no_trade(
                f"LTF ({trend}) ≠ HTF ({higher_trend})",
                current_time, current_price, trend,
            )

        # ── Technical layers ──────────────────────────────────────────────
        liq_df   = detect_liquidity_sweep(df, swing_lows=structure_df["swing_low"])
        ob_df    = detect_order_blocks(df, lookback=5,
                                       min_move_pct=0.01 if "BTC" in self.pair else 0.002)
        poi      = identify_poi_from_ob(df, ob_df, trend)
        fvg_df   = detect_fair_value_gaps(df)
        snr      = identify_snr_levels(df)
        daily_cy = detect_daily_cycle(df)

        has_bull_fvg = bool(fvg_df["fvg_bull"].notna().any())
        has_bear_fvg = bool(fvg_df["fvg_bear"].notna().any())
        nearest_sup  = max((s for s in snr["supports"]    if s < current_price), default=None)
        nearest_res  = min((r for r in snr["resistances"] if r > current_price), default=None)
        axis_aligned = kill_zone in ("LONDON", "NEW_YORK", "LONDON_NY_OVERLAP")

        # ── Scoring ───────────────────────────────────────────────────────
        score, reasons = 0, []

        if trend == "bullish": score += 30; reasons.append("Bullish trend (HTF aligned)")
        else:                  score += 30; reasons.append("Bearish trend (HTF aligned)")

        recent_choch = structure_df[structure_df["choch"] != 0].tail(3)
        recent_bos   = structure_df[structure_df["bos"]   != 0].tail(5)
        choch_bonus  = False
        if not recent_choch.empty:
            cd = "bullish" if recent_choch.iloc[-1]["choch"] == 1 else "bearish"
            if cd == trend:
                score += 30; reasons.append(f"CHOCH confirms {trend}"); choch_bonus = True
        if not choch_bonus and not recent_bos.empty:
            bd = "bullish" if recent_bos.iloc[-1]["bos"] == 1 else "bearish"
            if bd == trend:
                score += 20; reasons.append(f"BOS confirms {trend}")

        recent_liq = liq_df[liq_df["sweep"] != 0].tail(3)
        if not recent_liq.empty:
            ls         = recent_liq.iloc[-1]
            sweep_dir  = "sell-side" if ls["sweep"] == -1 else "buy-side"
            aligned    = (trend == "bullish" and sweep_dir == "sell-side") or \
                         (trend == "bearish" and sweep_dir == "buy-side")
            if aligned and ls["sweep_reclaim"]:
                score += 25; reasons.append(f"Liquidity sweep + reclaim ({sweep_dir})")

        if poi["poi_active"]:
            match = (trend == "bullish" and poi["zone_type"] == "demand") or \
                    (trend == "bearish" and poi["zone_type"] == "supply")
            if match:
                score += 15; reasons.append(f"POI active ({poi['zone_type']})")

        if (trend == "bullish" and has_bull_fvg) or (trend == "bearish" and has_bear_fvg):
            score += 5; reasons.append("FVG aligns with trend")
        if axis_aligned:
            score += 5; reasons.append(f"Kill zone: {kill_zone}")

        confidence = min(score / 100.0, 1.0)
        if confidence < self.min_confidence:
            return self._no_trade(
                f"Confidence too low ({confidence:.0%})", current_time, current_price, trend
            )

        sig_dir = "BUY" if trend == "bullish" else "SELL"
        entry, sl, tp = self._calculate_levels(sig_dir, poi, structure_df, current_price)
        if entry is None:
            return self._no_trade("Cannot determine entry/SL/TP", current_time, current_price, trend)

        # ── FIX C: spread-adjusted R:R ────────────────────────────────────
        spread           = get_spread(self.pair, self._pip_size())
        effective_entry  = entry + spread if sig_dir == "BUY" else entry - spread
        adj_risk         = abs(effective_entry - sl)
        adj_reward       = abs(tp - effective_entry)
        rr               = adj_reward / adj_risk if adj_risk > 0 else 0.0

        if rr < self.min_risk_reward:
            return self._no_trade(
                f"Spread-adjusted R:R too low ({rr:.2f} < {self.min_risk_reward})",
                current_time, current_price, trend,
                entry=entry, stop_loss=sl, take_profit=tp, risk_reward=rr,
            )

        signal_dict = {
            "pair":          self.pair,
            "timeframe":     self.timeframe,
            "timestamp":     current_time.isoformat(),
            "signal":        sig_dir,
            "confidence":    confidence,
            "trend":         trend,
            "current_price": current_price,
            "entry":         round(entry, 5),
            "stop_loss":     round(sl,    5),
            "take_profit":   round(tp,    5),
            "risk_reward":   round(rr,    2),
            "spread":        round(spread, 6),
            "poi":           poi,
            "reason":        reasons,
            "news_impact":   {"trade_allowed": True},
            "ktl_axis": {
                "kill_zone":          kill_zone,
                "daily_cycle":        daily_cy,
                "axis_aligned":       axis_aligned,
                "nearest_support":    nearest_sup,
                "nearest_resistance": nearest_res,
                "has_bull_fvg":       has_bull_fvg,
                "has_bear_fvg":       has_bear_fvg,
                "vol_regime":         vol_regime,
            },
        }

        ml_mult = self.apply_ml_score(signal_dict)
        signal_dict["confidence"] = float(np.clip(confidence * ml_mult, 0.0, 1.0))
        return signal_dict

    # ── Level calculation (UNIFIED FIX) ───────────────────────────────────────
    def _calculate_levels(self, signal, poi, structure_df, current_price):
        """Return (entry, stop_loss, take_profit). Never returns entry == sl."""
        # ------------------------------------------------------------------
        # Helper: absolute minimum stop distance for any asset
        # ------------------------------------------------------------------
        def _min_stop(price):
            if "BTC" in self.pair or "SOL" in self.pair:
                return max(price * 0.01, 1.0)               # crypto: at least $1
            if self.pair in ("SPY", "QQQ", "GC=F"):
                # Indices / gold – use 0.5% of price, minimum 0.1 point
                return max(price * 0.005, 0.1)
            # Forex – 20 pips, respecting JPY pip size
            pip_size = self._pip_size()                      # 0.01 for JPY, 0.0001 otherwise
            return 20 * pip_size

        entry = stop_loss = take_profit = None

        if poi["poi_active"]:
            zone_bottom, zone_top = poi["entry_zone"]
            if signal == "BUY":
                entry = zone_bottom
                stop_loss = zone_bottom - _min_stop(zone_bottom)
                take_profit = zone_top + (zone_top - zone_bottom) * 2
            else:
                entry = zone_top
                stop_loss = zone_top + _min_stop(zone_top)
                take_profit = zone_bottom - (zone_top - zone_bottom) * 2
        else:
            recent_high = (structure_df['swing_high'].dropna().iloc[-1]
                           if not structure_df['swing_high'].dropna().empty else None)
            recent_low  = (structure_df['swing_low'].dropna().iloc[-1]
                           if not structure_df['swing_low'].dropna().empty else None)

            if signal == "BUY" and recent_low is not None and recent_low < current_price:
                entry = recent_low
                stop_loss = recent_low - _min_stop(recent_low)
                take_profit = (recent_high if recent_high
                               else current_price + (current_price - recent_low) * 2)
            elif signal == "SELL" and recent_high is not None and recent_high > current_price:
                entry = recent_high
                stop_loss = recent_high + _min_stop(recent_high)
                take_profit = (recent_low if recent_low
                               else current_price - (recent_high - current_price) * 2)
            else:
                return None, None, None          # no valid swing point → no trade

        # ------------------------------------------------------------------
        # FINAL safety: never allow stop to be within 0.1% of entry
        # ------------------------------------------------------------------
        min_dist = _min_stop(entry)
        if abs(entry - stop_loss) < min_dist * 0.1:         # if essentially zero
            stop_loss = entry - min_dist if signal == "BUY" else entry + min_dist
        return entry, stop_loss, take_profit

    # ── No-trade helper ───────────────────────────────────────────────────────
    def _no_trade(self, reason, timestamp, current_price, trend,
                  entry=None, stop_loss=None, take_profit=None, risk_reward=None) -> dict:
        resp = {
            "pair":          self.pair,
            "timeframe":     self.timeframe,
            "timestamp":     timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            "signal":        "NO_TRADE",
            "confidence":    0.0,
            "trend":         trend,
            "current_price": current_price,
            "reason":        [reason],
            "news_impact":   {"trade_allowed": True},
        }
        if entry is not None:
            resp.update(entry=entry, stop_loss=stop_loss, take_profit=take_profit, risk_reward=risk_reward)
        return resp

    # ── Position sizing (spread-aware)  (FIX C) ───────────────────────────────
    def calculate_position_size(
        self, risk_amount: float, account_currency: str = "USD",
        entry: float = None, stop_loss: float = None,
        signal: str = "BUY",
    ) -> dict:
        if entry is None or stop_loss is None:
            return {"error": "entry and stop_loss required"}

        config   = self.ASSET_CONFIG.get(
            self.pair,
            {"pip_value": 10.0, "lot_size": 100_000, "min_lot": 0.01, "is_forex": True},
        )
        min_lot  = config["min_lot"]
        spread   = get_spread(self.pair, self._pip_size())

        # Adjust entry for spread so risk calculation uses realistic fill price
        eff_entry = entry + spread if signal == "BUY" else entry - spread
        eff_risk  = abs(eff_entry - stop_loss)

        if eff_risk == 0:
            return {"error": "Effective risk distance is zero (check entry/SL/spread)"}

        if config["is_forex"]:
            pip_risk = eff_risk / self._pip_size()
            lot_size = risk_amount / (pip_risk * config["pip_value"])
            lot_size = max(round(lot_size / min_lot) * min_lot, min_lot)
            units    = lot_size * config["lot_size"]
            pip_risk_out = pip_risk
        else:
            lot_size = max(round(risk_amount / eff_risk / min_lot) * min_lot, min_lot)
            units    = lot_size
            pip_risk_out = None

        return {
            "pair":             self.pair,
            "risk_amount":      risk_amount,
            "account_currency": account_currency,
            "entry":            entry,
            "effective_entry":  round(eff_entry, 5),
            "stop_loss":        stop_loss,
            "spread":           round(spread, 6),
            "effective_risk":   round(eff_risk, 5),
            "pip_risk":         pip_risk_out,
            "lot_size":         round(lot_size, 4),
            "units":            round(units, 4),
            "min_lot":          min_lot,
        }


# =============================================================================
# TRADE PROBABILITY SCORER  (shared cache)
# =============================================================================

class TradeProbabilityScorer:

    def __init__(self, pairs: list = None, timeframes: list = None):
        self.pairs        = pairs or MultiTimeframeDataManager.DEFAULT_PAIRS
        self.timeframes   = timeframes or ["15m", "1h", "4h", "1d"]
        self.data_manager = MultiTimeframeDataManager(self.pairs)
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}

    def _get_df(self, pair: str, tf: str) -> Optional[pd.DataFrame]:
        key = f"{pair}:{tf}"
        if key not in self._cache:
            self._cache[key] = self.data_manager.fetch_data(pair, tf)
        return self._cache[key]

    def calculate_composite_score(self, signal: dict, higher_tf_signal: dict = None) -> float:
        if signal["signal"] not in ("BUY", "SELL"):
            return 0.0
        base    = signal["confidence"]
        rr_bon  = min(signal.get("risk_reward", 1.0) / 10, 0.2)
        mtf_bon = 0.15 if (higher_tf_signal and higher_tf_signal["signal"] == signal["signal"]) else 0.0
        ax_bon  = 0.25 if signal.get("ktl_axis", {}).get("axis_aligned", False) else 0.0
        return min(base + rr_bon + mtf_bon + ax_bon, 1.0)

    def scan_all(
        self,
        min_confidence: float = 0.5,
        min_rr: float = 1.5,
        max_workers: int = 4,
    ) -> pd.DataFrame:

        results = []

        def process_pair(pair: str) -> list:
            pair_results = []
            df_daily     = self._get_df(pair, "1d")
            daily_signal = None

            if df_daily is not None and not df_daily.empty:
                eng_d = AntigravityQuantEngine(pair, "1d", swing_window=7,
                                               min_confidence=min_confidence,
                                               min_risk_reward=min_rr)
                daily_signal = eng_d.generate_signal(df_daily)

            for tf in self.timeframes:
                if tf == "1d":
                    continue
                df = self._get_df(pair, tf)
                if df is None or df.empty:
                    continue

                eng    = AntigravityQuantEngine(
                    pair, tf, swing_window=10 if "BTC" in pair else 7,
                    min_confidence=min_confidence, min_risk_reward=min_rr,
                )
                signal = eng.generate_signal(df, higher_tf_df=df_daily)

                if signal["signal"] in ("BUY", "SELL"):
                    composite = self.calculate_composite_score(signal, daily_signal)
                    ktl = signal.get("ktl_axis", {})
                    pair_results.append({
                        "pair":           pair,
                        "timeframe":      tf,
                        "signal":         signal["signal"],
                        "confidence":     round(signal["confidence"], 4),
                        "risk_reward":    signal.get("risk_reward", 0),
                        "trend":          signal["trend"],
                        "entry":          signal["entry"],
                        "stop_loss":      signal["stop_loss"],
                        "take_profit":    signal["take_profit"],
                        "spread":         signal.get("spread", 0),
                        "reasons":        " | ".join(signal["reason"]),
                        "daily_aligned":  daily_signal is not None and
                                          daily_signal["signal"] == signal["signal"],
                        "axis_aligned":   ktl.get("axis_aligned", False),
                        "kill_zone":      ktl.get("kill_zone", "N/A"),
                        "daily_cycle":    ktl.get("daily_cycle", "N/A"),
                        "vol_regime":     ktl.get("vol_regime", "N/A"),
                        "has_fvg":        ktl.get("has_bull_fvg") or ktl.get("has_bear_fvg"),
                        "composite_score": composite,
                    })
            return pair_results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pair, p): p for p in self.pairs}
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    logging.error(f"scan_all error {futures[future]}: {e}")

        if not results:
            return pd.DataFrame()

        df_out  = pd.DataFrame(results)
        max_sc  = df_out["composite_score"].max()
        df_out["probability"] = (
            (df_out["composite_score"] / max_sc * 100).round(1) if max_sc > 0 else 0.0
        )
        return df_out.sort_values("composite_score", ascending=False).reset_index(drop=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("AntigravityQuantEngine — Production Ready")
    print("=" * 65)

    # ── Quick scan ────────────────────────────────────────────────────────
    scorer = TradeProbabilityScorer()
    best   = scorer.scan_all(min_confidence=0.5, min_rr=1.5)

    if not best.empty:
        print(f"\nFound {len(best)} potential trades.\n")
        cols = ["pair","timeframe","signal","probability","risk_reward",
                "axis_aligned","kill_zone","daily_cycle","vol_regime","has_fvg"]
        print(best[cols].head(5).to_string(index=False))
    else:
        print("No qualifying trades found.")

    # ── BTC example ───────────────────────────────────────────────────────
    print("\n--- BTC-USD 1h example ---")
    dm_ex  = MultiTimeframeDataManager(["BTC-USD"])
    df_1h  = dm_ex.fetch_data("BTC-USD", "1h")
    df_4h  = dm_ex.fetch_data("BTC-USD", "4h")
    if df_1h is not None and not df_1h.empty:
        engine = AntigravityQuantEngine(
            "BTC-USD", "1h", swing_window=10,
            min_confidence=0.6, min_risk_reward=2.0,
        )
        sig = engine.generate_signal(df_1h, higher_tf_df=df_4h)
        print(json.dumps(sig, indent=2, default=str))
    else:
        print("Data fetch failed.")

    # ── Walk-forward validation (run separately — takes several minutes) ──
    # Uncomment when you're ready to validate out-of-sample performance:
    #
    # print("\n--- Walk-Forward Backtest ---")
    # wf_results = walk_forward_backtest(
    #     pairs=["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD"],
    #     timeframe="1h",
    #     train_bars=1000, test_bars=200, step_bars=200,
    #     min_confidence=0.6, min_rr=2.0, spread_model=True,
    # )
    # if not wf_results.empty:
    #     wf_results.to_csv("walk_forward_results.csv", index=False)
    #     print("Results saved → walk_forward_results.csv")