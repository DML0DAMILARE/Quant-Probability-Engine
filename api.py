from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
from typing import Optional

from data_implementation import (
    MultiTimeframeDataManager,
    AntigravityQuantEngine,
    TradeProbabilityScorer,
    get_economic_calendar,
    detect_bos_choch_from_swings,
    detect_liquidity_sweep,
    detect_order_blocks
)

app = FastAPI(title="Antigravity Quant Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dm = MultiTimeframeDataManager([
    "BTC-USD", "EURUSD=X", "GBPUSD=X", "JPY=X",
    "EURJPY=X", "EURNZD=X", "GBPAUD=X", "USDCHF=X",
    "GBPCHF=X", "AUDUSD=X", "GBPCAD=X", "USDCAD=X"
])

def df_to_dict(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    df_out = df.reset_index()
    if 'Datetime' not in df_out.columns and 'index' in df_out.columns:
        df_out.rename(columns={'index': 'Datetime'}, inplace=True)
    if 'Datetime' in df_out.columns:
        df_out['Datetime'] = df_out['Datetime'].astype(str)
    return df_out.to_dict(orient='records')


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/signal/{pair}")
async def get_signal(
    pair: str,
    timeframe: str = Query("1h", enum=["15m", "1h", "4h", "1d"]),
    risk_amount: Optional[float] = Query(None),
    min_confidence: float = Query(0.6),
    min_rr: float = Query(2.0),
    use_higher_tf: bool = Query(True)   # NEW: whether to fetch daily for alignment
):
    """
    Generate KTL-enhanced trading signal for a specific pair and timeframe.
    """
    df = dm.fetch_data(pair, timeframe)
    if df is None or df.empty:
        return {"error": f"Failed to fetch data for {pair} {timeframe}"}

    higher_tf_df = None
    if use_higher_tf and timeframe != "1d":
        higher_tf_df = dm.fetch_data(pair, "1d")

    swing_win = 10 if "BTC" in pair else 7
    engine = AntigravityQuantEngine(
        pair, timeframe,
        swing_window=swing_win,
        min_confidence=min_confidence,
        min_risk_reward=min_rr
    )
    signal = engine.generate_signal(df, higher_tf_df=higher_tf_df)

    if risk_amount and signal["signal"] in ["BUY", "SELL"]:
        pos = engine.calculate_position_size(
            risk_amount=risk_amount,
            entry=signal["entry"],
            stop_loss=signal["stop_loss"]
        )
        signal["position_sizing"] = pos

    return signal


@app.get("/structure/{pair}")
async def get_structure(pair: str, timeframe: str = "1h", candles: int = 300):
    df = dm.fetch_data(pair, timeframe)
    if df is None or df.empty:
        return {"error": f"Failed to fetch data for {pair} {timeframe}"}

    swing_win = 10 if "BTC" in pair else 7
    structure_df = detect_bos_choch_from_swings(df, window=swing_win)
    liq_df = detect_liquidity_sweep(df, swing_lows=structure_df['swing_low'], lookback=20)
    move_pct = 0.01 if "BTC" in pair else 0.002
    ob_df = detect_order_blocks(df, lookback=5, min_move_pct=move_pct)

    return {
        "pair": pair,
        "timeframe": timeframe,
        "ohlcv": df_to_dict(df.tail(candles)),
        "structure": df_to_dict(structure_df.tail(candles)),
        "liquidity": df_to_dict(liq_df.tail(candles)),
        "order_blocks": df_to_dict(ob_df.tail(candles))
    }


@app.get("/calendar/{pair}")
async def get_calendar_events(pair: str, days: int = 3):
    try:
        events_df = get_economic_calendar(pair, days_ahead=days)
        if events_df.empty:
            return {"events": []}
        events_list = events_df.to_dict(orient='records')
        return {"events": events_list}
    except Exception as e:
        return {"error": str(e), "events": []}


@app.get("/best-trades")
async def get_best_trades(
    min_confidence: float = Query(0.5),
    min_rr: float = Query(1.5)
):
    """
    Scan all pairs and timeframes, return KTL-ranked opportunities.
    """
    scorer = TradeProbabilityScorer()
    df = scorer.scan_all(min_confidence=min_confidence, min_rr=min_rr)
    if df.empty:
        return {"trades": [], "message": "No qualifying trades found"}
    return {"trades": df.to_dict(orient='records')}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)