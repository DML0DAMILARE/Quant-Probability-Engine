import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import dash
from dash import Input, Output, State, ctx, dcc, html, no_update
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import yfinance as yf

from data_implementation import (
    AntigravityQuantEngine,
    MultiTimeframeDataManager,
    EngineGuard,
    detect_bos_choch_from_swings,
    get_economic_calendar,
    get_current_kill_zone,
    _default_guard,
)
from trade_db import (
    calculate_stats,
    delete_trade,
    get_active_trades,
    get_all_trades,
    get_open_trades,
    get_pending_trades,
    init_db,
    insert_trade,
    update_trade,
)


# BOOTSTRAP
init_db()
ML_ACTIVE = os.path.exists("trade_classifier_lgbm.pkl")

ALL_PAIRS = [
    "EURJPY=X","EURNZD=X","GBPAUD=X","USDCHF=X",
    "BTC-USD", "EURAUD=X","EURUSD=X","GBPCHF=X",
    "NZDUSD=X","AUDUSD=X","AUDCAD=X","EURCHF=X",
    "AUDCHF=X","CADCHF=X","AUDNZD=X","EURCAD=X",
    "GBPCAD=X","GBPNZD=X","NZDJPY=X","CADJPY=X",
    "AUDJPY=X","SOL-USD", "USDJPY=X","CHFJPY=X",
    "GBPJPY=X","USDCAD=X","GBPUSD=X","EURGBP=X",
    "SPY","QQQ","GC=F",
]

STRIP_PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","GBPJPY=X",
    "AUDUSD=X","USDCAD=X","GC=F","BTC-USD","NZDUSD=X",
]

PAIR_LABELS = {
    "SPY":"S&P500","QQQ":"NAS100","BTC-USD":"BTCUSD",
    "SOL-USD":"SOLUSD","GC=F":"XAUUSD","XAUUSD=X":"XAUUSD",
}

dm = MultiTimeframeDataManager(ALL_PAIRS)

def label(pair: str) -> str:
    return PAIR_LABELS.get(pair, pair.replace("=X",""))

def fmt_price(price, pair: str) -> str:
    if price is None: return "—"
    if "BTC" in pair: return f"{price:,.0f}"
    if "SOL" in pair or pair in ("SPY","QQQ","GC=F","XAUUSD=X"): return f"{price:,.2f}"
    if "JPY" in pair: return f"{price:.3f}"
    return f"{price:.5f}"

def fmt_pnl(v: float) -> str:
    if v is None: return "—"
    return f"+{v:.2f}%" if v > 0 else f"{v:.2f}%"

def fmt_num(v, decimals=2, suffix="") -> str:
    if v is None: return "—"
    try:
        f = float(v)
        if f > 0 and suffix == "%": return f"+{f:.{decimals}f}{suffix}"
        return f"{f:.{decimals}f}{suffix}"
    except Exception:
        return str(v)

# DESIGN TOKENS v9 — "Obsidian Pro"


BG      = "#070c11"
SURF    = "#0b1520"
SURF2   = "#0f1e2d"
SURF3   = "#142435"
LINE    = "#1c3048"
LINE2   = "#112030"
LINE3   = "#0d1a28"
DIM     = "#28455e"
MUTED   = "#4a7292"
TEXT    = "#84b2cc"
BRIGHT  = "#cae2f2"
WHITE   = "#e6f2fc"

POS     = "#00d47e"
POS2    = "#00f090"   
NEG     = "#e8334a"
WARN    = "#e89c00"
DATA    = "#5ba8d4"
CYAN    = "#00b8d4"

# Fonts
DISPLAY = "'Chakra Petch','Space Mono',monospace"  # display numbers
MONO    = "'DM Mono','Space Mono',monospace"        # data / labels

PAGE_SIZE = 12

_CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=MONO, color=MUTED, size=10),
    margin=dict(l=10, r=10, t=30, b=10),
)

# PRICE CACHE

_price_cache: dict  = {}
_price_cache_ts: float = 0.0
_price_lock = threading.Lock()
_PRICE_TTL  = 6

def get_live_prices(pairs: list) -> dict:
    global _price_cache, _price_cache_ts
    now = time.time()
    with _price_lock:
        if now - _price_cache_ts < _PRICE_TTL and _price_cache:
            return dict(_price_cache)
    def _fetch(pair):
        try:
            info = yf.Ticker(pair).fast_info
            p = info.get("lastPrice") or info.get("regularMarketPrice")
            if p is None:
                h = yf.Ticker(pair).history(period="1d", interval="1m")
                p = float(h["Close"].iloc[-1]) if not h.empty else None
            return pair, p
        except Exception:
            return pair, None
    result = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_fetch, p): p for p in pairs}
        for fut in as_completed(futures):
            try:
                pair, price = fut.result()
                if price is not None: result[pair] = price
            except Exception:
                pass
    with _price_lock:
        _price_cache.update(result)
        _price_cache_ts = time.time()
        return dict(_price_cache)

_prev_prices: dict = {}
_prev_prices_lock = threading.Lock()

# SCANNER

_scan_lock    = threading.Lock()
_scan_result: list  = []
_scan_ts:     float = 0.0
_scan_running: bool = False
_SCAN_TTL = 90

def _do_scan(force: bool = False) -> None:
    global _scan_result, _scan_ts, _scan_running
    with _scan_lock:
        if _scan_running: return
        if not force and (time.time() - _scan_ts) < _SCAN_TTL: return
        _scan_running = True
    results = []
    try:
        def _scan_pair(pair):
            df_1h    = dm.fetch_data(pair,"1h")
            df_daily = dm.fetch_data(pair,"1d")
            df_4h    = dm.fetch_data(pair,"4h")
            if df_1h is None or df_1h.empty: return None
            price = df_1h["close"].iloc[-1]
            eng = AntigravityQuantEngine(pair,"1h",
                swing_window=10 if "BTC" in pair else 7,
                min_confidence=0.6, min_risk_reward=2.0)
            sig = eng.generate_signal(df_1h, higher_tf_df=df_daily)
            if sig["signal"] in ("BUY","SELL") and df_4h is not None and not df_4h.empty:
                s4h = detect_bos_choch_from_swings(df_4h, window=10 if "BTC" in pair else 7)
                t4h = s4h["trend"].iloc[-1]
                if (sig["signal"]=="BUY" and t4h!="bullish") or \
                   (sig["signal"]=="SELL" and t4h!="bearish"):
                    sig["signal"] = "NO_TRADE"
                    sig["reason"] = ["4h trend not aligned"]
            return {**sig, "current_price": price, "pair": pair}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_scan_pair,p):p for p in ALL_PAIRS}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r: results.append(r)
                except Exception as e:
                    print(f"Scan error: {e}")
        with _scan_lock:
            _scan_result = results
            _scan_ts     = time.time()
    finally:
        with _scan_lock:
            _scan_running = False

def kick_scan(force: bool = False) -> None:
    threading.Thread(target=_do_scan, args=(force,), daemon=True).start()

_chart_cache: dict = {}
def _cached_fig(key, h, fn):
    if key in _chart_cache and _chart_cache[key]["h"] == h:
        return _chart_cache[key]["f"]
    f = fn()
    _chart_cache[key] = {"h": h, "f": f}
    return f

# SESSION HELPER

def get_session() -> tuple:
    try:
        kz = get_current_kill_zone(datetime.now(timezone.utc))
    except Exception:
        kz = "UNKNOWN"
    lbls = {"LONDON":"LONDON","LONDON_NY_OVERLAP":"LON·NY","NEW_YORK":"NEW YORK",
            "ASIA":"ASIA","FRANKFURT":"FRANKFURT","NY_LUNCH":"NY LUNCH ⊘",
            "OFF_HOURS":"CLOSED","UNKNOWN":"—"}
    cols = {"LONDON":POS,"LONDON_NY_OVERLAP":POS,"NEW_YORK":DATA,
            "ASIA":WARN,"FRANKFURT":DATA,"NY_LUNCH":NEG,"OFF_HOURS":MUTED,"UNKNOWN":MUTED}
    return lbls.get(kz, kz), cols.get(kz, MUTED)

# ADVANCED STATS CALCULATOR

def compute_advanced_stats(closed_df: pd.DataFrame) -> dict:
    """
    Computes the metrics that actually matter for evaluating a trading edge.
    Based on TradeZella / JournalPlus research:
      - Expectancy: most important single number
      - R-Multiple average
      - Consistency score (std dev of P&L)
      - Composite edge score
      - Best/worst trade
      - Profit factor
    """
    s = {}

    if closed_df.empty or "profit_loss" not in closed_df.columns:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
            "expectancy": 0, "max_dd": 0, "consistency": 0,
            "best_trade": 0, "worst_trade": 0, "r_avg": 0,
            "edge_score": 0, "edge_status": "INSUFFICIENT DATA",
            "edge_color": MUTED,
            "total_pnl": 0, "net_pnl": 0,
        }

    df = closed_df.dropna(subset=["profit_loss"]).copy()
    wins_df  = df[df["profit_loss"] > 0]
    loss_df  = df[df["profit_loss"] < 0]

    total  = len(df)
    wins   = len(wins_df)
    losses = len(loss_df)
    win_rate = wins / total * 100 if total > 0 else 0

    avg_win  = float(wins_df["profit_loss"].mean()) if not wins_df.empty else 0
    avg_loss = abs(float(loss_df["profit_loss"].mean())) if not loss_df.empty else 0

    gross_profit = float(wins_df["profit_loss"].sum()) if not wins_df.empty else 0
    gross_loss   = abs(float(loss_df["profit_loss"].sum())) if not loss_df.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0
    )
    profit_factor = min(profit_factor, 99.9)  # cap display

    # Expectancy: (Win% × Avg Win) - (Loss% × Avg Loss)
    wr_dec = win_rate / 100
    lr_dec = 1 - wr_dec
    expectancy = (wr_dec * avg_win) - (lr_dec * avg_loss)

    # R-Multiple: avg win / avg loss ratio
    r_avg = avg_win / avg_loss if avg_loss > 0 else 0

    # Consistency score: inverse of std dev (lower std dev = more consistent)
    pnl_std = float(df["profit_loss"].std()) if len(df) > 1 else 0

    # Max drawdown from cumulative P&L
    df["cum"] = df.sort_values("entry_time", na_position="last")["profit_loss"].cumsum()
    rolling_peak = df["cum"].cummax()
    drawdown_series = rolling_peak - df["cum"]
    max_dd = float(drawdown_series.max()) if not drawdown_series.empty else 0

    best_trade  = float(df["profit_loss"].max())
    worst_trade = float(df["profit_loss"].min())
    net_pnl     = float(df["profit_loss"].sum())


    # Profit Factor: 30%, Max DD: 25%, Expectancy: 20%, Consistency: 15%, R/R: 10%
    def _pf_score(pf):
        if pf < 1.0:   return 20
        if pf < 1.25:  return 40
        if pf < 1.5:   return 60
        if pf < 2.0:   return 75
        return 90

    def _dd_score(dd):
        if dd > 20:  return 20
        if dd > 15:  return 40
        if dd > 10:  return 55
        if dd > 5:   return 70
        return 85

    def _exp_score(exp):
        if exp <= 0:    return 20
        if exp < 0.1:   return 45
        if exp < 0.3:   return 65
        if exp < 0.5:   return 80
        return 90

    def _cons_score(std, n):
        if n < 10: return 50  # not enough data
        if std > 3:   return 30
        if std > 2:   return 50
        if std > 1:   return 70
        return 85

    def _rr_score(r):
        if r < 1.0:  return 20
        if r < 1.5:  return 50
        if r < 2.0:  return 70
        if r < 3.0:  return 85
        return 95

    pf_s   = _pf_score(profit_factor)
    dd_s   = _dd_score(max_dd)
    exp_s  = _exp_score(expectancy)
    cons_s = _cons_score(pnl_std, total)
    rr_s   = _rr_score(r_avg)

    edge_score = (pf_s*0.30 + dd_s*0.25 + exp_s*0.20 + cons_s*0.15 + rr_s*0.10)

    if total < 20:
        edge_status = "INSUFFICIENT DATA"
        edge_color  = MUTED
    elif edge_score >= 75:
        edge_status = "STRONG EDGE"
        edge_color  = POS2
    elif edge_score >= 55:
        edge_status = "MARGINAL EDGE"
        edge_color  = WARN
    else:
        edge_status = "REVIEW STRATEGY"
        edge_color  = NEG

    return {
        "total": total, "wins": wins, "losses": losses,
        "win_rate": round(win_rate, 1),
        "avg_win":  round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 3),
        "max_dd": round(max_dd, 2),
        "consistency": round(pnl_std, 3),
        "best_trade": round(best_trade, 2),
        "worst_trade": round(worst_trade, 2),
        "r_avg": round(r_avg, 2),
        "edge_score": round(edge_score, 1),
        "edge_status": edge_status,
        "edge_color": edge_color,
        "net_pnl": round(net_pnl, 2),
    }


def build_analytics_data(prices: dict = None) -> dict:
    if prices is None: prices = {}
    closed_df = get_all_trades()
    closed_df = closed_df[closed_df["outcome"].notna()].copy() if not closed_df.empty else pd.DataFrame()
    active_df = get_active_trades()
    base_stats = calculate_stats()

    adv = compute_advanced_stats(closed_df)

    active_rows = []
    total_unreal = 0.0
    for _, t in active_df.iterrows():
        price = prices.get(t["pair"], t.get("entry_price",0))
        ep    = t["entry_price"] or 1
        unreal = ((price-ep)/ep*100) if t["signal"]=="BUY" else ((ep-price)/ep*100)
        total_unreal += unreal
        active_rows.append({
            "pair":t["pair"],"signal":t["signal"],
            "entry_price":ep,"current_price":price,
            "stop_loss":t["stop_loss"],"take_profit":t["take_profit"],
            "unrealised":round(unreal,2),"entry_time":t.get("entry_time",""),
        })

    return {
        "adv": adv,
        "closed_df": closed_df,
        "active_df": active_df,
        "active_rows": active_rows,
        "total_unreal": round(total_unreal, 2),
        "pending_count": len(get_pending_trades()),
    }

# CHART BUILDERS

def _fig_equity(closed_df, active_rows):
    fig = go.Figure(layout={**_CHART,"height":200})
    d   = pd.DataFrame()
    if closed_df.empty and not active_rows:
        fig.add_annotation(text="NO TRADE DATA",showarrow=False,
            font=dict(color=DIM,size=10,family=MONO),xref="paper",yref="paper",x=.5,y=.5)
        return fig
    if not closed_df.empty:
        d = closed_df.dropna(subset=["profit_loss"]).sort_values("entry_time").copy()
        d["cum"] = d["profit_loss"].cumsum()
        peak     = d["cum"].cummax()
        dd       = d["cum"] - peak
        fig.add_trace(go.Scatter(x=list(range(len(d))),y=dd.tolist(),
            fill="tozeroy",fillcolor="rgba(232,51,74,0.05)",
            line=dict(color="rgba(0,0,0,0)"),showlegend=False,hoverinfo="skip"))
        col = POS if d["cum"].iloc[-1] >= 0 else NEG
        fig.add_trace(go.Scatter(x=list(range(len(d))),y=d["cum"].tolist(),
            mode="lines",name="P&L",
            line=dict(color=col,width=1.5),
            fill="tozeroy",fillcolor=f"rgba(0,212,126,0.04)",
            hovertemplate="%{y:+.2f}%<extra></extra>"))
    if active_rows:
        base = float(d["cum"].iloc[-1]) if not d.empty else 0.
        proj = base + sum(r["unrealised"] for r in active_rows)
        xe   = len(d) if not d.empty else 0
        fig.add_trace(go.Scatter(x=[max(0,xe-1),xe+1],y=[base,proj],
            mode="lines+markers",name="Projected",
            line=dict(color=WARN,width=1,dash="dot"),
            marker=dict(size=3,color=WARN),
            hovertemplate="Projected: %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0,line_color=LINE,line_width=1)
    fig.update_layout(
        title=dict(text="EQUITY CURVE",font=dict(size=9,color=MUTED,family=MONO),x=0.01),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showline=False),
        yaxis=dict(gridcolor=LINE2,zeroline=False,ticksuffix="%",
                   tickfont=dict(size=9),tickformat="+.1f"),
        legend=dict(orientation="h",x=0.01,y=1.18,font=dict(size=9)),
    )
    return fig

def _fig_donut(wins, losses, win_rate):
    fig = go.Figure(layout={**_CHART,"height":180})
    if wins+losses == 0:
        fig.add_annotation(text="NO DATA",showarrow=False,
            font=dict(color=DIM,size=10,family=MONO),xref="paper",yref="paper",x=.5,y=.5)
        return fig
    fig.add_trace(go.Pie(
        values=[wins,losses],labels=["WIN","LOSS"],hole=0.74,
        marker=dict(colors=[POS,NEG],line=dict(color=BG,width=6)),
        textinfo="none",hovertemplate="%{label}: %{value}<extra></extra>"))
    fig.add_annotation(text=f"{win_rate:.0f}",x=.5,y=.55,showarrow=False,
        font=dict(size=26,color=POS if win_rate>=50 else NEG,family=DISPLAY),
        xref="paper",yref="paper")
    fig.add_annotation(text="% WIN",x=.5,y=.34,showarrow=False,
        font=dict(size=8,color=MUTED,family=MONO),xref="paper",yref="paper")
    fig.update_layout(showlegend=False,margin=dict(l=4,r=4,t=4,b=4))
    return fig

def _fig_heatmap(results):
    fig = go.Figure(layout=_CHART)
    if not results:
        fig.add_annotation(text="RUN SCAN TO POPULATE",showarrow=False,
            font=dict(color=DIM,size=10,family=MONO),xref="paper",yref="paper",x=.5,y=.5)
        fig.update_layout(height=80)
        return fig
    tfs   = ["15m","1h","4h","1d"]
    pairs = list(dict.fromkeys(r["pair"] for r in results))
    z,txt = [],[]
    for p in pairs:
        rz,rt = [],[]
        for tf in tfs:
            hit = next((r for r in results if r["pair"]==p and r["timeframe"]==tf),None)
            if hit and hit.get("signal") in ("BUY","SELL"):
                v = (1 if hit["signal"]=="BUY" else -1)*hit.get("confidence",.5)
                rz.append(v); rt.append(f"{hit['signal']}\n{hit.get('confidence',0):.0%}")
            else:
                rz.append(0); rt.append("—")
        z.append(rz); txt.append(rt)
    fig.add_trace(go.Heatmap(
        z=z,x=tfs,y=[label(p) for p in pairs],
        colorscale=[[0,NEG],[0.5,SURF2],[1,POS]],
        zmid=0,zmin=-1,zmax=1,text=txt,texttemplate="%{text}",
        textfont=dict(size=8,family=MONO),showscale=False,
        hovertemplate="<b>%{y}</b> %{x}<br>%{text}<extra></extra>"))
    fig.update_layout(
        title=dict(text="SIGNAL MAP",font=dict(size=9,color=MUTED,family=MONO),x=.01),
        xaxis=dict(side="top",showgrid=False,tickfont=dict(size=9,family=MONO)),
        yaxis=dict(showgrid=False,autorange="reversed",tickfont=dict(size=9,family=MONO)),
        height=max(120,len(pairs)*20+48))
    return fig

def _fig_bar_chart(closed_df, group_col, title, height=185):
    fig = go.Figure(layout={**_CHART,"height":height})
    if closed_df.empty: return fig
    d = closed_df.copy()
    if group_col == "month":
        d["month"] = pd.to_datetime(d["entry_time"],errors="coerce").dt.to_period("M").astype(str)
    by = d.groupby(group_col)["profit_loss"].sum().reset_index()
    if group_col == "pair": by[group_col] = by[group_col].apply(label)
    by = by.sort_values("profit_loss")
    fig.add_trace(go.Bar(
        x=by[group_col],y=by["profit_loss"],
        marker_color=[POS if v>=0 else NEG for v in by["profit_loss"]],
        marker_line_width=0,hovertemplate="%{x}: %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0,line_color=LINE,line_width=1)
    fig.update_layout(
        title=dict(text=title,font=dict(size=9,color=MUTED,family=MONO),x=.01),
        xaxis=dict(showgrid=False,tickfont=dict(size=8,family=MONO)),
        yaxis=dict(gridcolor=LINE2,zeroline=False,ticksuffix="%",
                   tickfont=dict(size=9),tickformat="+.1f"),
        bargap=0.35,showlegend=False)
    return fig

def _fig_rolling(closed_df, window=10):
    fig = go.Figure(layout={**_CHART,"height":185})
    if closed_df.empty or len(closed_df)<3:
        fig.add_annotation(text="NEED MORE TRADES",showarrow=False,
            font=dict(color=DIM,size=10,family=MONO),xref="paper",yref="paper",x=.5,y=.5)
        return fig
    d = closed_df.sort_values("entry_time").copy()
    d["win"]  = (d["outcome"]=="WIN").astype(int)
    d["roll"] = d["win"].rolling(min(window,len(d)),min_periods=1).mean()*100
    fig.add_hrect(y0=0,y1=50,fillcolor="rgba(232,51,74,0.03)",line_width=0)
    fig.add_hline(y=50,line_color=LINE,line_dash="dot",line_width=1)
    fig.add_trace(go.Scatter(x=list(range(len(d))),y=d["roll"].tolist(),
        mode="lines",line=dict(color=DATA,width=1.5),
        fill="tozeroy",fillcolor="rgba(91,168,212,0.06)",
        hovertemplate="Trade %{x}: %{y:.1f}%<extra></extra>"))
    fig.update_layout(
        title=dict(text=f"ROLLING {window}-TRADE WIN RATE",font=dict(size=9,color=MUTED,family=MONO),x=.01),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(gridcolor=LINE2,zeroline=False,ticksuffix="%",
                   tickfont=dict(size=9),range=[0,100]),showlegend=False)
    return fig

# UI PRIMITIVES

_TH = {
    "background": BG, "color": MUTED, "padding": "5px 10px",
    "fontFamily": MONO, "fontSize": "9px", "letterSpacing": "0.08em",
    "textTransform": "uppercase", "borderBottom": f"1px solid {LINE}",
    "borderRight": f"1px solid {LINE2}", "whiteSpace": "nowrap",
    "verticalAlign": "middle", "fontWeight": "400",
}
_TD = {
    "padding": "5px 10px", "borderBottom": f"1px solid {LINE2}",
    "borderRight": f"1px solid {LINE2}", "color": TEXT,
    "fontFamily": MONO, "fontSize": "11px", "whiteSpace": "nowrap",
    "fontVariantNumeric": "tabular-nums", "transition": "background 0.08s",
}

def _th(t, align="left"):
    return html.Th(t, style={**_TH,"textAlign":align})

def _td(c, color=None, bold=False, right=False, dim=False):
    s = {**_TD}
    if color: s["color"] = color
    if bold:  s["fontWeight"] = "600"
    if right: s["textAlign"] = "right"
    if dim:   s["color"] = DIM
    return html.Td(c, style=s)

def _del_btn(tid):
    return html.Button("✕", id={"type":"delete-btn","index":tid}, style={
        "background":"transparent","color":MUTED,"border":"none",
        "cursor":"pointer","fontSize":"11px","padding":"0 4px",
        "fontFamily":MONO,"transition":"color 0.1s",
    })

def _table(headers, rows, empty="NO DATA"):
    if not rows:
        return html.Div(empty, style={
            "color":DIM,"padding":"22px 10px","fontFamily":MONO,
            "fontSize":"9px","letterSpacing":"0.08em","textAlign":"center",
        })
    return html.Div(
        html.Table(
            [html.Thead(html.Tr([_th(h) for h in headers])),html.Tbody(rows)],
            style={"width":"100%","borderCollapse":"collapse",
                   "tableLayout":"fixed","borderSpacing":"0"}),
        style={"overflowX":"auto","border":f"1px solid {LINE}"})

def _conf_bar(conf):
    pct = round(conf*100)
    col = POS if conf>=0.75 else (WARN if conf>=0.5 else NEG)
    return html.Div([
        html.Span(f"{pct}%",style={"color":col,"fontSize":"11px","fontFamily":MONO,
            "fontWeight":"600","display":"inline-block","width":"32px"}),
        html.Div(html.Div(style={"width":f"{pct}%","height":"2px","backgroundColor":col}),
            style={"width":"80px","height":"2px","backgroundColor":LINE2,
                   "display":"inline-block","verticalAlign":"middle","marginLeft":"6px"}),
    ],style={"display":"flex","alignItems":"center"})

def _rr_badge(rr):
    col = POS if rr>=3 else (WARN if rr>=2 else NEG)
    return html.Span(f"{rr:.1f}R",style={"color":col,"fontFamily":MONO,"fontSize":"11px","fontWeight":"700"})

def _sig_chip(sig):
    if sig=="BUY":  return html.Span("▲ BUY",style={"color":POS,"fontFamily":MONO,"fontWeight":"700","fontSize":"11px"})
    if sig=="SELL": return html.Span("▼ SELL",style={"color":NEG,"fontFamily":MONO,"fontWeight":"700","fontSize":"11px"})
    return html.Span("—",style={"color":DIM,"fontFamily":MONO})

def _vol_dot(regime):
    c = POS if regime=="OK" else (WARN if regime=="LOW_VOL" else NEG)
    t = {"OK":"OK","LOW_VOL":"LO-VOL","HIGH_VOL":"HI-VOL"}.get(regime,"?")
    return html.Span([html.Span("●",style={"color":c,"fontSize":"7px","marginRight":"4px"}),
                      html.Span(t,style={"color":c,"fontSize":"9px","fontFamily":MONO})])

def _kz_label(kz):
    l = {"LONDON":"LON","LONDON_NY_OVERLAP":"L/N","NEW_YORK":"NY",
         "ASIA":"ASIA","FRANKFURT":"FFM","NY_LUNCH":"—","OFF_HOURS":"OFF"}
    c = {"LONDON":POS,"LONDON_NY_OVERLAP":POS,"NEW_YORK":DATA,
         "ASIA":WARN,"FRANKFURT":DATA,"NY_LUNCH":NEG,"OFF_HOURS":DIM}
    return html.Span(l.get(kz,"—"),style={"color":c.get(kz,DIM),"fontFamily":MONO,"fontSize":"9px"})

def _exec_btn(pair, sig):
    bg = POS if sig=="BUY" else NEG
    return html.Button("EXEC",id={"type":"exec-btn","index":pair},style={
        "backgroundColor":bg,"color":BG,"border":"none","padding":"2px 9px",
        "cursor":"pointer","fontFamily":MONO,"fontWeight":"700",
        "fontSize":"9px","letterSpacing":"0.06em","borderRadius":"1px"})

def _pager(prev_id, next_id, cur, total):
    b = {"backgroundColor":"transparent","color":MUTED,"border":f"1px solid {LINE}",
         "padding":"2px 8px","cursor":"pointer","fontFamily":MONO,"fontSize":"9px"}
    return html.Div([
        html.Button("←",id=prev_id,n_clicks=0,style=b),
        html.Span(f"  {cur+1} / {total}  ",style={"color":MUTED,"fontSize":"9px","fontFamily":MONO}),
        html.Button("→",id=next_id,n_clicks=0,style=b),
    ],style={"display":"flex","alignItems":"center","padding":"8px 0 4px"})

def _rule(label_text):
    """Horizontal rule with embedded label — used as section divider."""
    return html.Div([
        html.Div(style={"flex":"1","height":"1px","backgroundColor":LINE2}),
        html.Span(label_text,style={
            "color":DIM,"fontSize":"8px","fontFamily":MONO,"letterSpacing":"0.12em",
            "textTransform":"uppercase","margin":"0 10px","whiteSpace":"nowrap",
        }),
        html.Div(style={"flex":"1","height":"1px","backgroundColor":LINE2}),
    ],style={"display":"flex","alignItems":"center","margin":"12px 0 10px"})

def _status_dot(val, thresholds, labels=None):
    """Traffic-light dot based on value vs thresholds (good, warn, bad)."""
    good_t, warn_t = thresholds
    if val >= good_t: col = POS
    elif val >= warn_t: col = WARN
    else: col = NEG
    return html.Span("●",style={"color":col,"fontSize":"8px","marginLeft":"6px"})

_PANEL = {"backgroundColor":SURF,"border":f"1px solid {LINE}","marginBottom":"1px"}
_BTN_P = {"backgroundColor":POS,"color":BG,"border":"none","padding":"6px 16px",
           "cursor":"pointer","fontFamily":MONO,"fontWeight":"700","fontSize":"10px",
           "letterSpacing":"0.06em","borderRadius":"1px","transition":"opacity 0.1s"}
_BTN_S = {"backgroundColor":"transparent","color":MUTED,"border":f"1px solid {LINE}",
           "padding":"5px 12px","cursor":"pointer","fontFamily":MONO,"fontSize":"9px",
           "letterSpacing":"0.04em","borderRadius":"1px"}

# PERFORMANCE SUMMARY COMPONENTS (the main fix)

def _tier1_metric(val_str, label_text, color, accent_color, sub=None, dot_thresh=None, dot_val=None):
    """
    TIER 1 — headline KPI cell.
    Large Chakra Petch display number, label below, semantic top border.
    """
    dot = html.Span()
    if dot_thresh is not None and dot_val is not None:
        dot = _status_dot(dot_val, dot_thresh)

    return html.Div([
        html.Div(style={
            "height":"2px","backgroundColor":accent_color,
            "width":"100%","marginBottom":"0",
        }),
        html.Div([
            html.Div([
                html.Span(val_str,style={
                    "color":color,"fontFamily":DISPLAY,"fontSize":"32px",
                    "fontWeight":"500","letterSpacing":"0.01em","lineHeight":"1",
                }),
                dot,
            ],style={"display":"flex","alignItems":"center","marginBottom":"4px"}),
            html.Div(label_text,style={
                "color":MUTED,"fontSize":"8px","fontFamily":MONO,
                "letterSpacing":"0.1em","textTransform":"uppercase",
            }),
            *([html.Div(sub,style={"color":DIM,"fontSize":"9px","fontFamily":MONO,"marginTop":"3px"})] if sub else []),
        ],style={"padding":"12px 18px 14px"}),
    ],style={
        "flex":"1","minWidth":"120px",
        "borderRight":f"1px solid {LINE}",
        "backgroundColor":SURF,
    })

def _tier2_row(label_text, val_str, color=None, bar=None, bar_color=None, bar_pct=None):
    """TIER 2 — compact stat row within a breakdown column."""
    return html.Div([
        html.Div(label_text,style={
            "color":MUTED,"fontSize":"9px","fontFamily":MONO,
            "letterSpacing":"0.06em","textTransform":"uppercase",
            "flex":"1",
        }),
        html.Div([
            html.Span(val_str,style={
                "color":color or BRIGHT,"fontFamily":DISPLAY,"fontSize":"15px",
                "fontWeight":"500","letterSpacing":"0.01em",
            }),
        ]),
    ] + ([
        html.Div(
            html.Div(style={
                "width":f"{min(bar_pct,100)}%","height":"2px",
                "background":f"linear-gradient(90deg, {POS}, {WARN} 60%, {NEG})" if bar_color=="gradient" else bar_color,
                "borderRadius":"1px",
            }),
            style={"width":"100%","height":"2px","backgroundColor":LINE2,
                   "borderRadius":"1px","marginTop":"4px"},
        ),
    ] if bar and bar_pct is not None else []),
    style={
        "display":"flex","alignItems":"center","flexWrap":"wrap",
        "padding":"7px 0","borderBottom":f"1px solid {LINE2}",
    })

def _breakdown_col(title, rows_html):
    """TIER 2 column wrapper."""
    return html.Div([
        html.Div(title,style={
            "color":DIM,"fontSize":"8px","fontFamily":MONO,"letterSpacing":"0.12em",
            "textTransform":"uppercase","marginBottom":"6px",
            "paddingBottom":"6px","borderBottom":f"1px solid {LINE}",
        }),
        *rows_html,
    ],style={
        "flex":"1","padding":"12px 16px",
        "borderRight":f"1px solid {LINE}",
        "minWidth":"160px",
    })

def _edge_badge(status, color, score):
    """TIER 3 — composite edge status badge."""
    return html.Div([
        html.Div([
            html.Span("EDGE STATUS",style={
                "color":DIM,"fontSize":"7px","fontFamily":MONO,
                "letterSpacing":"0.12em","display":"block","marginBottom":"3px",
            }),
            html.Span(status,style={
                "color":color,"fontFamily":DISPLAY,"fontSize":"13px",
                "fontWeight":"500","letterSpacing":"0.06em",
            }),
        ]),
        html.Div([
            html.Div(f"{score:.0f}",style={
                "color":color,"fontFamily":DISPLAY,"fontSize":"28px",
                "fontWeight":"500","lineHeight":"1",
            }),
            html.Div("/100",style={"color":DIM,"fontSize":"9px","fontFamily":MONO,"marginTop":"2px"}),
        ],style={"textAlign":"right"}),
    ],style={
        "display":"flex","justifyContent":"space-between","alignItems":"center",
        "padding":"10px 18px","borderTop":f"1px solid {LINE}",
        "borderLeft":f"2px solid {color}",
        "backgroundColor":f"{color}0a",
    })

def build_performance_summary(adv: dict, active_rows: list, total_unreal: float,
                               pending_count: int, closed_df: pd.DataFrame) -> html.Div:
    """
    The structured 3-tier performance summary.
    Research-backed layout: headlines → breakdown → edge status
    """

    realised  = adv["net_pnl"]
    combined  = round(realised + total_unreal, 2)
    active_c  = len(active_rows)

    # ── TIER 1: 4 headline KPIs 
    tier1 = html.Div([
        _tier1_metric(
            fmt_pnl(combined), "COMBINED P&L",
            color=POS if combined>=0 else NEG,
            accent_color=POS if combined>=0 else NEG,
            sub=f"Realised {fmt_pnl(realised)}  ·  Float {fmt_pnl(total_unreal)}",
        ),
        _tier1_metric(
            f"{adv['win_rate']:.1f}%", "WIN RATE",
            color=POS if adv["win_rate"]>=50 else (WARN if adv["win_rate"]>=40 else NEG),
            accent_color=POS if adv["win_rate"]>=50 else NEG,
            sub=f"{adv['wins']}W  ·  {adv['losses']}L  ·  {adv['total']} trades",
            dot_thresh=(50, 40), dot_val=adv["win_rate"],
        ),
        _tier1_metric(
            f"{adv['profit_factor']:.2f}", "PROFIT FACTOR",
            color=POS if adv["profit_factor"]>=1.5 else (WARN if adv["profit_factor"]>=1.0 else NEG),
            accent_color=POS if adv["profit_factor"]>=1.5 else (WARN if adv["profit_factor"]>=1.0 else NEG),
            sub="Gross profit ÷ gross loss",
            dot_thresh=(1.5, 1.0), dot_val=adv["profit_factor"],
        ),
        _tier1_metric(
            fmt_num(adv["max_dd"],2,"%"), "MAX DRAWDOWN",
            color=POS if adv["max_dd"]<5 else (WARN if adv["max_dd"]<10 else NEG),
            accent_color=NEG if adv["max_dd"]>=10 else (WARN if adv["max_dd"]>=5 else POS),
            sub="From equity peak",
            dot_thresh=(5, 10), dot_val=100-adv["max_dd"],  # inverted: less DD = better
        ),
    ],style={"display":"flex","borderBottom":f"1px solid {LINE}"})

    # ── TIER 2: 3-column breakdown 
    # Drawdown bar percentage vs 10% limit
    dd_pct = min((adv["max_dd"] / 10.0) * 100, 100)

    col_left = _breakdown_col("TRADE STATISTICS", [
        _tier2_row("TOTAL TRADES",  str(adv["total"]),  BRIGHT),
        _tier2_row("WINNERS",       str(adv["wins"]),   POS),
        _tier2_row("LOSERS",        str(adv["losses"]), NEG),
        _tier2_row("AVG WIN",       fmt_num(adv["avg_win"],2,"%"),  POS),
        _tier2_row("AVG LOSS",      fmt_num(adv["avg_loss"],2,"%"), NEG),
        _tier2_row("BEST TRADE",    fmt_pnl(adv["best_trade"]),    POS),
        _tier2_row("WORST TRADE",   fmt_pnl(adv["worst_trade"]),   NEG),
    ])

    col_center = _breakdown_col("EDGE & RISK METRICS", [
        _tier2_row("EXPECTANCY",
            fmt_num(adv["expectancy"],3,"%"),
            color=POS if adv["expectancy"]>0 else NEG,
        ),
        _tier2_row("R-MULTIPLE AVG", f"{adv['r_avg']:.2f}R",
            color=POS if adv["r_avg"]>=1.5 else (WARN if adv["r_avg"]>=1.0 else NEG)),
        _tier2_row("CURRENT DD",
            fmt_num(adv["max_dd"],2,"%"), NEG,
            bar=True, bar_color="gradient", bar_pct=dd_pct,
        ),
        _tier2_row("CONSISTENCY",
            f"{adv['consistency']:.3f} σ",
            color=POS if adv["consistency"]<1 else (WARN if adv["consistency"]<2 else NEG)),
        _tier2_row("EDGE SCORE",
            f"{adv['edge_score']:.0f} / 100",
            color=adv["edge_color"]),
    ])

    col_right = _breakdown_col("ACCOUNT STATUS", [
        _tier2_row("ACTIVE NOW",    str(active_c),      POS if active_c else MUTED),
        _tier2_row("PENDING",       str(pending_count), WARN if pending_count else MUTED),
        _tier2_row("REALISED P&L",  fmt_pnl(realised),  POS if realised>=0 else NEG),
        _tier2_row("UNREALISED",    fmt_pnl(total_unreal), POS if total_unreal>=0 else NEG),
        _tier2_row("COMBINED",      fmt_pnl(combined),  POS if combined>=0 else NEG),
    ])

    tier2 = html.Div([col_left, col_center, col_right],
        style={"display":"flex","borderBottom":f"1px solid {LINE}"})

    # TIER 3: Edge status badge 
    tier3 = _edge_badge(adv["edge_status"], adv["edge_color"], adv["edge_score"])

    return html.Div([tier1, tier2, tier3],
        style={**_PANEL, "marginBottom":"12px"})

# APP LAYOUT

def _nav_item(text, tid, ico):
    return html.Div([
        html.Span(ico,style={"color":DIM,"marginRight":"10px","fontSize":"10px","fontFamily":MONO}),
        html.Span(text,style={"fontFamily":MONO,"fontSize":"10px","letterSpacing":"0.08em"}),
    ],id={"type":"nav-item","index":tid},style={
        "padding":"9px 14px","cursor":"pointer","color":MUTED,
        "display":"flex","alignItems":"center",
        "borderLeft":"2px solid transparent","transition":"all 0.1s",
    })

app = dash.Dash(__name__, title="OQPE",
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Chakra+Petch:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap",
    ])
server = app.server   # ← required for gunicorn / Render
app.config.suppress_callback_exceptions = True

app.layout = html.Div([

    dcc.Interval(id="price-interval",      interval=60_000),
    dcc.Interval(id="scan-poll-interval",  interval=3_000),
    dcc.Interval(id="scanner-auto",        interval=300_000),
    dcc.Interval(id="clock-interval",      interval=1_000),
    dcc.Store(id="active-tab",              data="scanner"),
    dcc.Store(id="scan-results",            data=[]),
    dcc.Store(id="scan-is-running",         data=False),
    dcc.Store(id="pending-page",            data=0),
    dcc.Store(id="closed-page",             data=0),
    dcc.Store(id="journal-refresh-trigger", data=0),
    dcc.Download(id="download-csv"),

    # BLOOMBERG STRIP
    html.Div([
        html.Div([
            html.Span("OQPE",style={
                "color":WHITE,"fontFamily":DISPLAY,"fontWeight":"600",
                "fontSize":"14px","letterSpacing":"0.2em","marginRight":"4px",
            }),
            html.Span("TERMINAL",style={
                "color":DIM,"fontFamily":DISPLAY,"fontSize":"9px",
                "letterSpacing":"0.22em","fontWeight":"300",
            }),
        ],style={"display":"flex","alignItems":"center","padding":"0 16px",
                 "borderRight":f"1px solid {LINE}","minWidth":"155px"}),
        html.Div(
            html.Div(id="price-strip-inner",style={"display":"inline-flex","gap":"0","whiteSpace":"nowrap"}),
            style={"flex":"1","overflow":"hidden","position":"relative"},
        ),
        html.Div([
            html.Div(id="strip-clock",style={
                "color":WHITE,"fontFamily":MONO,"fontSize":"13px","fontWeight":"500","letterSpacing":"0.04em",
            }),
            html.Div(id="strip-session",style={
                "fontFamily":MONO,"fontSize":"9px","letterSpacing":"0.06em","marginTop":"2px",
            }),
        ],style={"padding":"0 14px","textAlign":"right","borderLeft":f"1px solid {LINE}",
                 "minWidth":"145px","display":"flex","flexDirection":"column","justifyContent":"center"}),
    ],style={
        "display":"flex","alignItems":"stretch","height":"40px",
        "backgroundColor":BG,"borderBottom":f"1px solid {LINE}",
        "position":"sticky","top":"0","zIndex":"100",
    }),

    #  BODY 
    html.Div([

        # Sidebar
        html.Div([
            html.Div([
                _nav_item("SCANNER",   "scanner",  "◈"),
                _nav_item("JOURNAL",   "journal",  "≡"),
                _nav_item("CALENDAR",  "calendar", "◷"),
                _nav_item("PIP CALC",  "pipcalc",  "σ"),
                _nav_item("ANALYTICS", "analytics","◉"),
            ],style={"paddingTop":"8px","borderBottom":f"1px solid {LINE}","paddingBottom":"8px"}),

            html.Div([
                html.Div([
                    html.Span("●",className="blink-live",
                        style={"color":POS,"marginRight":"6px","fontSize":"7px"}),
                    html.Span("LIVE DATA",style={"color":MUTED,"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em"}),
                ],style={"marginBottom":"5px","display":"flex","alignItems":"center"}),
                html.Div([
                    html.Span("●",style={"color":POS if ML_ACTIVE else NEG,"marginRight":"6px","fontSize":"7px"}),
                    html.Span("ML ACTIVE" if ML_ACTIVE else "ML OFFLINE",
                        style={"color":POS if ML_ACTIVE else NEG,"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em"}),
                ],style={"display":"flex","alignItems":"center"}),
            ],style={"padding":"10px 14px","borderBottom":f"1px solid {LINE}"}),

            html.Div(id="guard-banner"),

            html.Div([
                html.Button(html.Span(id="scan-btn-label",children="⟳ SCAN MARKET"),
                    id="scan-market-btn",n_clicks=0,
                    style={**_BTN_P,"width":"100%","marginBottom":"5px"}),
                html.Button("RETRAIN MODEL",id="retrain-model-btn",n_clicks=0,
                    style={**_BTN_S,"width":"100%","marginBottom":"4px"}),
                html.Button("⬇ EXPORT CSV",id="export-csv-btn",n_clicks=0,
                    style={**_BTN_S,"width":"100%"}),
                html.Div(id="retrain-feedback",style={"marginTop":"5px"}),
            ],style={"padding":"10px"}),

        ],style={
            "width":"158px","minWidth":"158px","backgroundColor":BG,
            "borderRight":f"1px solid {LINE}",
            "display":"flex","flexDirection":"column",
            "height":"calc(100vh - 40px)","overflow":"hidden",
            "position":"sticky","top":"40px",
        }),

        # Main panel
        html.Div([
            # Subheader
            html.Div([
                html.Div([
                    html.Span(id="page-title",style={
                        "color":WHITE,"fontFamily":DISPLAY,"fontSize":"15px",
                        "fontWeight":"500","letterSpacing":"0.08em",
                    }),
                    html.Span("  "),
                    html.Span(id="page-subtitle",style={
                        "color":DIM,"fontFamily":MONO,"fontSize":"9px","letterSpacing":"0.05em",
                    }),
                ]),
                html.Div(id="header-stats",
                    style={"display":"flex","gap":"0","alignItems":"stretch"}),
            ],style={
                "display":"flex","justifyContent":"space-between","alignItems":"center",
                "padding":"5px 18px","borderBottom":f"1px solid {LINE}",
                "backgroundColor":SURF,"minHeight":"38px",
            }),

            # Position ticker
            html.Div(id="position-ticker",style={"borderBottom":f"1px solid {LINE2}"}),

            # Toast
            html.Div(id="exec-feedback"),

            # Content
            html.Div(id="tab-content",style={"padding":"14px 16px"}),

            # Hidden pager buttons
            html.Div([
                html.Button("←",id="pending-prev-btn",n_clicks=0,style={"display":"none"}),
                html.Button("→",id="pending-next-btn",n_clicks=0,style={"display":"none"}),
                html.Button("←",id="closed-prev-btn", n_clicks=0,style={"display":"none"}),
                html.Button("→",id="closed-next-btn", n_clicks=0,style={"display":"none"}),
            ],style={"display":"none"}),

        ],style={"flex":"1","overflowY":"auto","overflowX":"hidden",
                 "height":"calc(100vh - 40px)","minWidth":"0"}),

    ],style={"display":"flex","height":"calc(100vh - 40px)"}),

],style={"backgroundColor":BG,"fontFamily":MONO,"minHeight":"100vh"})

# CSS
# ─────────────────────────────────────────────────────────────────────────────

app.index_string = f"""<!DOCTYPE html><html>
<head>
  {{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
  <style>
    *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
    html {{ font-size:14px; }}
    body {{ background:{BG}; color:{TEXT}; font-family:{MONO}; overflow:hidden; }}
    ::-webkit-scrollbar {{ width:3px; height:3px; }}
    ::-webkit-scrollbar-track {{ background:{BG}; }}
    ::-webkit-scrollbar-thumb {{ background:{LINE}; }}
    * {{ font-variant-numeric:tabular-nums; font-feature-settings:"tnum"; }}

    .nav-active {{
      color:{WHITE} !important;
      border-left-color:{POS} !important;
      background:{SURF2} !important;
    }}
    .sig-buy  {{ color:{POS};  font-weight:700; }}
    .sig-sell {{ color:{NEG}; font-weight:700; }}

    tbody tr:hover td {{ background:{SURF2} !important; }}

    @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.1}} }}
    .blink-live {{ animation:blink 2s ease infinite; display:inline-block; }}

    @keyframes panelIn {{ from{{opacity:0;transform:translateY(4px)}} to{{opacity:1;transform:none}} }}
    .panel-in {{ animation:panelIn 0.14s ease forwards; }}

    /* Staggered analytics reveal */
    .analytics-t1 {{ animation:panelIn 0.14s ease 0.00s both; }}
    .analytics-t2 {{ animation:panelIn 0.14s ease 0.06s both; }}
    .analytics-t3 {{ animation:panelIn 0.14s ease 0.12s both; }}
    .analytics-t4 {{ animation:panelIn 0.14s ease 0.18s both; }}
    .analytics-t5 {{ animation:panelIn 0.14s ease 0.24s both; }}

    @keyframes spin {{ to{{transform:rotate(360deg)}} }}
    .spin {{ display:inline-block; animation:spin 0.7s linear infinite; }}

    @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.35}} }}
    .pulse {{ animation:pulse 2s ease infinite; }}

    @keyframes stripScroll {{ 0%{{transform:translateX(0)}} 100%{{transform:translateX(-50%)}} }}
    .strip-scroll {{ animation:stripScroll 45s linear infinite; }}
    .strip-scroll:hover {{ animation-play-state:paused; }}

    @keyframes toastIn {{ from{{opacity:0;transform:translateX(10px)}} to{{opacity:1;transform:none}} }}
    .toast {{
      position:fixed; bottom:18px; right:18px; z-index:9999;
      animation:toastIn 0.16s ease forwards;
      font-family:{MONO}; font-size:11px; padding:8px 14px;
      border-left:2px solid; background:{SURF2};
    }}

    .strip-item {{
      display:inline-flex; flex-direction:column; justify-content:center;
      padding:0 14px; border-right:1px solid {LINE2};
      height:40px; min-width:85px;
    }}
    .strip-pair  {{ color:{DIM};  font-size:7px;  letter-spacing:0.12em; font-family:{MONO}; }}
    .strip-price {{ color:{TEXT}; font-size:12px; font-weight:500; font-family:{MONO}; }}

    .pl-pos {{ border-left:2px solid {POS}; }}
    .pl-neg {{ border-left:2px solid {NEG}; }}

    button:hover  {{ opacity:0.8; cursor:pointer; }}
    button:active {{ opacity:0.6; }}

    .Select-control {{background:{SURF}!important;border-color:{LINE}!important;color:{TEXT}!important;border-radius:0!important;}}
    .Select-value-label,.Select-placeholder {{color:{TEXT}!important;font-family:{MONO}!important;font-size:11px!important;}}
    .Select-menu-outer {{background:{SURF}!important;border-color:{LINE}!important;border-radius:0!important;}}
    .VirtualizedSelectOption {{color:{TEXT}!important;font-family:{MONO}!important;font-size:11px!important;}}
    .VirtualizedSelectFocusedOption {{background:{SURF2}!important;}}

    /* Scanline texture */
    body::before {{
      content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
      background-image:repeating-linear-gradient(
        0deg,transparent,transparent 1px,
        rgba(255,255,255,0.007) 1px,rgba(255,255,255,0.007) 2px
      );
    }}
  </style>
</head>
<body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body>
</html>"""

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("price-strip-inner","children"),
    Output("strip-clock","children"),
    Output("strip-session","children"),
    Input("clock-interval","n_intervals"),
)
def update_strip(_):
    now  = datetime.now(timezone(timedelta(hours=1)))
    slbl, scol = get_session()
    prices = get_live_prices(STRIP_PAIRS)
    items  = [html.Div([
        html.Div(label(p), className="strip-pair"),
        html.Div(fmt_price(prices.get(p),p) if prices.get(p) else "—", className="strip-price"),
    ], className="strip-item") for p in STRIP_PAIRS]
    strip = html.Div(items*2, className="strip-scroll",
                     style={"display":"inline-flex"})
    return strip, now.strftime("%H:%M:%S"), html.Span(slbl, style={"color":scol})


@app.callback(Output("position-ticker","children"), Input("price-interval","n_intervals"))
def update_pos_ticker(_n):
    active_df = get_active_trades()
    if active_df.empty: return html.Div()
    prices = get_live_prices(active_df["pair"].unique().tolist())
    chips  = []
    for _, t in active_df.iterrows():
        p   = prices.get(t["pair"], t["entry_price"])
        ep  = t["entry_price"] or 1
        unr = ((p-ep)/ep*100) if t["signal"]=="BUY" else ((ep-p)/ep*100)
        col = POS if unr>=0 else NEG
        chips.append(html.Div([
            html.Span(label(t["pair"]),style={"color":BRIGHT,"fontWeight":"500","fontFamily":MONO,"fontSize":"10px","marginRight":"6px"}),
            html.Span("▲" if t["signal"]=="BUY" else "▼",style={"color":col,"fontSize":"8px","marginRight":"3px"}),
            html.Span(fmt_pnl(unr),style={"color":col,"fontWeight":"700","fontFamily":MONO,"fontSize":"11px"}),
        ],style={"display":"inline-flex","alignItems":"center","padding":"3px 12px",
                 "borderRight":f"1px solid {LINE2}"}))
    return html.Div(chips,style={"display":"flex","backgroundColor":SURF2,"overflowX":"auto"})


@app.callback(Output("header-stats","children"), Input("price-interval","n_intervals"))
def update_header_stats(_n):
    open_t   = get_open_trades()
    active_c = len(open_t[open_t["status"]=="ACTIVE"])  if not open_t.empty else 0
    pend_c   = len(open_t[open_t["status"]=="PENDING"]) if not open_t.empty else 0
    closed_d = get_all_trades()
    closed_d = closed_d[closed_d["outcome"].notna()] if not closed_d.empty else pd.DataFrame()
    wins   = int((closed_d["outcome"]=="WIN").sum())  if not closed_d.empty else 0
    losses = int((closed_d["outcome"]=="LOSS").sum()) if not closed_d.empty else 0
    wr     = wins/(wins+losses)*100 if (wins+losses)>0 else 0

    def _s(val, lbl, col=TEXT, accent=None):
        return html.Div([
            html.Div(str(val),style={"color":col,"fontSize":"15px","fontFamily":DISPLAY,
                                     "fontWeight":"500","lineHeight":"1","letterSpacing":"0.02em"}),
            html.Div(lbl,style={"color":DIM,"fontSize":"7px","letterSpacing":"0.1em",
                                "fontFamily":MONO,"marginTop":"3px","textTransform":"uppercase"}),
        ],style={"padding":"5px 14px","borderLeft":f"1px solid {LINE}","textAlign":"center",
                 "minWidth":"65px",**({"borderTop":f"2px solid {accent}"} if accent else {})})

    return [
        _s(active_c,    "ACTIVE",   POS  if active_c else MUTED, POS  if active_c else None),
        _s(pend_c,      "PENDING",  WARN if pend_c   else MUTED, WARN if pend_c   else None),
        _s(wins+losses, "CLOSED",   DATA),
        _s(f"{wr:.0f}%","WIN RATE", POS if wr>=50 else NEG, POS if wr>=50 else NEG),
    ]


@app.callback(
    Output("active-tab","data"),
    Output({"type":"nav-item","index":dash.dependencies.ALL},"className"),
    Input({"type":"nav-item","index":dash.dependencies.ALL},"n_clicks"),
    State("active-tab","data"),
    prevent_initial_call=True,
)
def switch_tab(n_clicks, current):
    tab_ids = ["scanner","journal","calendar","pipcalc","analytics"]
    new_tab = current
    if ctx.triggered and any(n_clicks):
        try:
            new_tab = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
        except Exception:
            pass
    classes = ["nav-active" if t==new_tab else "" for t in tab_ids]
    return new_tab, classes


@app.callback(
    Output("tab-content",  "children"),
    Output("page-subtitle","children"),
    Output("page-title",   "children"),
    Input("active-tab",    "data"),
    Input("scan-results",  "data"),
    Input("pending-page",  "data"),
    Input("closed-page",   "data"),
)
def render_tab(tab, scan_results, pending_page, closed_page):
    if tab=="scanner":
        return _render_scanner(scan_results),"signals · 1h primary · 4h confirm","MARKET SCANNER"
    if tab=="journal":
        ot = get_open_trades()
        pr = get_live_prices(ot["pair"].unique().tolist()) if not ot.empty else {}
        return _render_journal(pending_page,closed_page,pr),"positions · live prices","TRADE JOURNAL"
    if tab=="calendar":
        return _render_calendar(),"high-impact events · next 3 days","ECO CALENDAR"
    if tab=="pipcalc":
        return _render_pipcalc(),"position sizing","PIP CALCULATOR"
    if tab=="analytics":
        ot = get_open_trades()
        pr = get_live_prices(ot["pair"].unique().tolist()) if not ot.empty else {}
        return _render_analytics(pr),"performance analytics · all trades","ANALYTICS"
    return html.Div(),"",""


@app.callback(Output("guard-banner","children"), Input("price-interval","n_intervals"))
def guard_banner(_n):
    if _default_guard.is_halted():
        return html.Div([
            html.Span("⊘ CIRCUIT BREAKER  ",style={"fontWeight":"700"}),
            html.Span(f"{_default_guard.current_drawdown:.1f}% DD exceeded"),
        ],style={"padding":"7px 14px","backgroundColor":"rgba(232,51,74,0.07)",
                 "borderBottom":f"1px solid {NEG}","fontFamily":MONO,
                 "fontSize":"9px","letterSpacing":"0.05em","color":NEG})
    return html.Div()


@app.callback(
    Output("scan-is-running","data"),
    Input("scan-market-btn","n_clicks"),
    Input("scanner-auto","n_intervals"),
    prevent_initial_call=True,
)
def trigger_scan(n_clicks, _auto):
    force = "scan-market-btn" in (ctx.triggered[0]["prop_id"] if ctx.triggered else "")
    kick_scan(force=force)
    return True


@app.callback(
    Output("scan-results","data"),
    Output("scan-is-running","data",allow_duplicate=True),
    Input("scan-poll-interval","n_intervals"),
    State("scan-is-running","data"),
    prevent_initial_call=True,
)
def poll_scan(_,is_running):
    with _scan_lock:
        running = _scan_running; results = list(_scan_result)
    if not running and results: return results, False
    return no_update, running


@app.callback(Output("scan-btn-label","children"), Input("scan-is-running","data"))
def scan_btn_label(running):
    if running: return [html.Span("◌",className="spin",style={"marginRight":"5px"}),"SCANNING"]
    return "⟳  SCAN MARKET"


@app.callback(Output("retrain-feedback","children"),
              Input("retrain-model-btn","n_clicks"),prevent_initial_call=True)
def retrain(_n):
    def _run():
        try:
            r = subprocess.run(["python","retrain_model.py"],
                capture_output=True,text=True,timeout=180)
            print("Retrain:", r.stdout or r.stderr)
        except Exception as e:
            print("Retrain failed:", e)
    threading.Thread(target=_run,daemon=True).start()
    return html.Span("RETRAINING…",style={"color":WARN,"fontSize":"9px","fontFamily":MONO,"letterSpacing":"0.06em"})


@app.callback(Output("download-csv","data"),Input("export-csv-btn","n_clicks"),prevent_initial_call=True)
def export_csv(_n):
    df = get_all_trades()
    return no_update if df.empty else dcc.send_data_frame(df.to_csv,"oqpe_trades.csv",index=False)


@app.callback(
    Output("exec-feedback","children"),
    Input({"type":"exec-btn","index":dash.dependencies.ALL},"n_clicks"),
    prevent_initial_call=True,
)
def execute_trade(n_clicks):
    if not ctx.triggered or not any(n_clicks): raise dash.exceptions.PreventUpdate
    pair = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    df_1h    = dm.fetch_data(pair,"1h")
    df_daily = dm.fetch_data(pair,"1d")
    if df_1h is None or df_1h.empty:
        return html.Div(f"⊘ NO DATA: {label(pair)}",className="toast",
                        style={"borderColor":NEG,"color":NEG})
    eng = AntigravityQuantEngine(pair,"1h",swing_window=10 if "BTC" in pair else 7,
        min_confidence=0.6,min_risk_reward=2.0)
    sig = eng.generate_signal(df_1h,higher_tf_df=df_daily)
    if sig["signal"] not in ("BUY","SELL"):
        return html.Div("⊘ SIGNAL EXPIRED",className="toast",style={"borderColor":NEG,"color":NEG})
    tid = insert_trade({"pair":pair,"timeframe":"1h","signal":sig["signal"],
        "entry_price":sig["entry"],"stop_loss":sig["stop_loss"],"take_profit":sig["take_profit"],
        "confidence":sig["confidence"],"risk_reward":sig["risk_reward"],"status":"PENDING",
        "notes":json.dumps(sig,default=str)})
    return html.Div(f"✓ #{tid} {label(pair)} {sig['signal']} @ {fmt_price(sig['entry'],pair)}",
                    className="toast",style={"borderColor":POS,"color":POS})


@app.callback(
    Output("journal-refresh-trigger","data"),
    Input({"type":"delete-btn","index":dash.dependencies.ALL},"n_clicks"),
    prevent_initial_call=True,
)
def handle_delete(n_clicks):
    if not ctx.triggered or not any(n_clicks): raise dash.exceptions.PreventUpdate
    try:
        tid = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
        delete_trade(tid)
    except Exception as e:
        print("Delete error:", e)
    return time.time()


@app.callback(
    Output("pending-page","data"),
    Input("pending-prev-btn","n_clicks"),Input("pending-next-btn","n_clicks"),
    State("pending-page","data"),prevent_initial_call=True,
)
def pending_nav(prev,nxt,page):
    pages = max(1,(len(get_pending_trades())+PAGE_SIZE-1)//PAGE_SIZE)
    return max(0,page-1) if "prev" in ctx.triggered[0]["prop_id"] else min(pages-1,page+1)


@app.callback(
    Output("closed-page","data"),
    Input("closed-prev-btn","n_clicks"),Input("closed-next-btn","n_clicks"),
    State("closed-page","data"),prevent_initial_call=True,
)
def closed_nav(prev,nxt,page):
    df = get_all_trades()
    df = df[df["outcome"].notna()] if not df.empty else df
    pages = max(1,(len(df)+PAGE_SIZE-1)//PAGE_SIZE)
    return max(0,page-1) if "prev" in ctx.triggered[0]["prop_id"] else min(pages-1,page+1)


@app.callback(
    Output("tab-content","children",allow_duplicate=True),
    Input("price-interval","n_intervals"),
    Input("journal-refresh-trigger","data"),
    State("active-tab","data"),
    State("pending-page","data"),
    State("closed-page","data"),
    prevent_initial_call=True,
)
def monitor_live(n,_refresh,tab,pending_page,closed_page):
    global _prev_prices
    open_trades = get_open_trades()
    prices: dict = {}
    if not open_trades.empty:
        prices = get_live_prices(open_trades["pair"].unique().tolist())
        with _prev_prices_lock:
            prev_snap = dict(_prev_prices)
        for _,t in open_trades.iterrows():
            price = prices.get(t["pair"])
            if price is None: continue
            if t["status"]=="PENDING":
                entry = t["entry_price"]; prev = prev_snap.get(t["pair"])
                activate = False
                if prev is not None:
                    if t["signal"]=="BUY"  and prev>entry>=price: activate=True
                    if t["signal"]=="SELL" and prev<entry<=price: activate=True
                if activate:
                    update_trade(t["id"],{"status":"ACTIVE","entry_time":datetime.now().isoformat()})
                    _default_guard.trade_completed()
            elif t["status"]=="ACTIVE":
                sl_hit = (t["signal"]=="BUY" and price<=t["stop_loss"]) or \
                         (t["signal"]=="SELL" and price>=t["stop_loss"])
                tp_hit = (t["signal"]=="BUY" and price>=t["take_profit"]) or \
                         (t["signal"]=="SELL" and price<=t["take_profit"])
                if sl_hit or tp_hit:
                    ep = t["entry_price"] or 1
                    pl = ((price-ep)/ep*100) if t["signal"]=="BUY" else ((ep-price)/ep*100)
                    update_trade(t["id"],{"status":"CLOSED","exit_price":price,
                        "outcome":"LOSS" if sl_hit else "WIN",
                        "profit_loss":round(pl,2),"exit_time":datetime.now().isoformat()})
                    _default_guard.record_trade(pl)
        with _prev_prices_lock:
            _prev_prices = prices.copy()
    if tab=="journal":   return _render_journal(pending_page,closed_page,prices)
    if tab=="analytics": return _render_analytics(prices)
    return no_update


@app.callback(Output("calendar-table-div","children"),Input("calendar-currency","value"))
def update_calendar(currency):
    pm = {"USD":"EURUSD=X","EUR":"EURUSD=X","GBP":"GBPUSD=X","JPY":"USDJPY=X"}
    try:
        events = get_economic_calendar(pm.get(currency,"EURUSD=X"),days_ahead=3)
    except Exception as e:
        return html.Div(str(e),style={"color":NEG,"fontFamily":MONO,"fontSize":"10px"})
    if events.empty:
        return html.Div("NO HIGH-IMPACT EVENTS IN NEXT 3 DAYS",
            style={"color":DIM,"fontFamily":MONO,"fontSize":"9px","letterSpacing":"0.06em","padding":"12px 0"})
    ic = {"High":NEG,"Medium":WARN,"Low":DIM}
    rows=[html.Tr([
        _td(ev.get("date","—")),_td(ev.get("time","—")),
        _td(ev.get("currency","—"),color=DATA,bold=True),
        _td(ev.get("event","—")),
        _td(ev.get("impact","—"),color=ic.get(ev.get("impact",""),TEXT),bold=True),
    ]) for _,ev in events.iterrows()]
    return _table(["DATE","TIME","CCY","EVENT","IMPACT"],rows)


def calculate_lot_size(pair,risk_amount,stop_loss_pips):
    if "BTC" in pair or "SOL" in pair: ps=1.0; lu=1
    elif pair in ("SPY","QQQ","GC=F","XAUUSD=X"): ps=0.01; lu=1
    elif "JPY" in pair: ps=0.01; lu=100_000
    else: ps=0.0001; lu=100_000
    pv = ps*lu
    if pv==0: return {"error":"Cannot compute pip value"}
    ml = 0.001 if ("BTC" in pair or "SOL" in pair) else 0.01
    lot = max(ml, round((risk_amount/(stop_loss_pips*pv))/ml)*ml)
    units = lot*lu if lu!=1 else lot
    try:
        from data_implementation import SPREAD_PIPS
        sp = SPREAD_PIPS.get(pair,2.0)
    except Exception:
        sp = 2.0
    return {"lot_size":round(lot,4),"units":round(units,4),"pip_value_per_lot":pv,
            "spread_pips":sp,"spread_cost":round(sp*pv*lot,2),"pair":label(pair)}


@app.callback(
    Output("pip-result","children"),
    Input("calc-btn","n_clicks"),
    State("pip-pair","value"),State("risk-amount","value"),State("stop-loss-pips","value"),
    prevent_initial_call=True,
)
def compute_lot_size(n,pair,risk,sl_pips):
    if not pair or not risk or not sl_pips:
        return html.Div("FILL ALL FIELDS",style={"color":NEG,"fontFamily":MONO,"fontSize":"9px"})
    try: risk=float(risk); sl_pips=float(sl_pips)
    except: return html.Div("INVALID INPUT",style={"color":NEG,"fontFamily":MONO})
    res = calculate_lot_size(pair,risk,sl_pips)
    if "error" in res: return html.Div(res["error"],style={"color":NEG})
    def _r(lbl,val,col=BRIGHT):
        return html.Tr([_td(lbl,color=MUTED),_td(val,color=col,right=True)])
    return html.Div([
        html.Table([html.Tbody([
            _r("LOT SIZE",   f"{res['lot_size']} lots",POS),
            _r("UNITS",      f"{res['units']:,.0f}"),
            _r("PIP VALUE",  f"${res['pip_value_per_lot']:.4f}"),
            _r("SPREAD COST",f"${res['spread_cost']:.2f}  ({res['spread_pips']} pips)",WARN),
        ])],style={"width":"100%","borderCollapse":"collapse","tableLayout":"fixed"}),
    ],style={"border":f"1px solid {LINE}","marginTop":"12px","borderLeft":f"2px solid {POS}"})

# ─────────────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def _render_scanner(scan_results):
    active  = [r for r in (scan_results or []) if r.get("signal") in ("BUY","SELL")]
    buy_c   = sum(1 for r in active if r.get("signal")=="BUY")
    sell_c  = sum(1 for r in active if r.get("signal")=="SELL")

    stats_row = html.Div([
        html.Div([
            html.Span(str(buy_c),style={"color":POS,"fontFamily":DISPLAY,"fontSize":"26px","fontWeight":"500"}),
            html.Span(" BUY",style={"color":MUTED,"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em"}),
        ],style={"padding":"8px 20px","borderRight":f"1px solid {LINE}","display":"flex","alignItems":"baseline","gap":"5px"}),
        html.Div([
            html.Span(str(sell_c),style={"color":NEG,"fontFamily":DISPLAY,"fontSize":"26px","fontWeight":"500"}),
            html.Span(" SELL",style={"color":MUTED,"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em"}),
        ],style={"padding":"8px 20px","borderRight":f"1px solid {LINE}","display":"flex","alignItems":"baseline","gap":"5px"}),
        html.Div([
            html.Span(str(len(scan_results or [])),style={"color":DATA,"fontFamily":DISPLAY,"fontSize":"26px","fontWeight":"500"}),
            html.Span(" SCANNED",style={"color":MUTED,"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em"}),
        ],style={"padding":"8px 20px","display":"flex","alignItems":"baseline","gap":"5px"}),
    ],style={"display":"flex","backgroundColor":SURF,"border":f"1px solid {LINE}","marginBottom":"10px"}) if scan_results else html.Div()

    heatmap = html.Div()
    if active:
        h = hash(json.dumps([(r.get("pair"),r.get("signal"),r.get("confidence")) for r in active],sort_keys=True))
        heatmap = html.Div([
            html.Div([html.Span("▦",style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                      html.Span("SIGNAL MAP",style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
                style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
            dcc.Graph(figure=_cached_fig("hm",h,lambda r=active:_fig_heatmap(r)),
                      config={"displayModeBar":False},style={"padding":"0 8px 6px"}),
        ],style={**_PANEL,"marginBottom":"10px"})

    def _ord(r):
        s=r.get("signal",""); return 0 if s=="BUY" else (1 if s=="SELL" else 2)

    rows=[]
    for r in sorted(scan_results or [],key=lambda x:(_ord(x),-x.get("confidence",0))):
        pair=r["pair"]; sig=r.get("signal","NO_TRADE")
        price=r.get("current_price"); conf=r.get("confidence",0)
        vol=r.get("ktl_axis",{}).get("vol_regime","OK")
        kz=r.get("ktl_axis",{}).get("kill_zone","OFF_HOURS")
        rr_v=r.get("risk_reward",0)
        if sig in ("BUY","SELL"):
            rows.append(html.Tr([
                _td(html.Span(label(pair),style={"color":BRIGHT,"fontWeight":"600","fontFamily":MONO})),
                _td(fmt_price(price,pair),right=True),
                _td(_sig_chip(sig)),
                _td(_conf_bar(conf)),
                _td(_rr_badge(rr_v)),
                _td(fmt_price(r.get("entry"),    pair),right=True),
                _td(fmt_price(r.get("stop_loss"),pair),right=True,color=NEG),
                _td(fmt_price(r.get("take_profit"),pair),right=True,color=POS),
                _td(html.Span([_vol_dot(vol),html.Span("  "),_kz_label(kz)])),
                _td(_exec_btn(pair,sig)),
            ]))
        else:
            reason=(r.get("reason") or ["—"])[0]
            rows.append(html.Tr([
                _td(html.Span(label(pair),style={"color":DIM,"fontFamily":MONO})),
                _td(fmt_price(price,pair),right=True,dim=True),
                _td(html.Span(reason[:36],style={"color":DIM,"fontSize":"9px","fontFamily":MONO})),
                _td("—",dim=True),_td("—",dim=True),_td("—",dim=True),
                _td("—",dim=True),_td("—",dim=True),_td("—",dim=True),_td(""),
            ]))

    table_panel = html.Div([
        html.Div([html.Span("◈",style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                  html.Span("LIVE SIGNALS",style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
            style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
        _table(["PAIR","PRICE","SIGNAL","CONFIDENCE","R/R","ENTRY","STOP LOSS","TAKE PROFIT","INFO",""],
               rows,"RUN SCAN TO SEE SIGNALS"),
    ],style=_PANEL)

    return html.Div([stats_row,heatmap,table_panel],className="panel-in")


def _render_journal(pending_page,closed_page,prices=None):
    if prices is None: prices={}
    active_df  = get_active_trades()
    pending_df = get_pending_trades()
    closed_df  = get_all_trades()
    closed_df  = closed_df[closed_df["outcome"].notna()] if not closed_df.empty else closed_df
    if not closed_df.empty:
        closed_df = closed_df.sort_values("entry_time",ascending=False,na_position="last")

    active_rows=[]
    for _,t in active_df.iterrows():
        p=prices.get(t["pair"],t["entry_price"]); ep=t["entry_price"] or 1
        pl=((p-ep)/ep*100) if t["signal"]=="BUY" else ((ep-p)/ep*100)
        active_rows.append(html.Tr(className="pl-pos" if pl>=0 else "pl-neg",children=[
            _td(label(t["pair"]),bold=True,color=BRIGHT),_td(t["timeframe"]),
            _td(_sig_chip(t["signal"])),
            _td(fmt_price(t["entry_price"],t["pair"]),right=True),
            _td(fmt_price(p,t["pair"]),right=True,color=POS if pl>=0 else NEG),
            _td(fmt_price(t["stop_loss"],t["pair"]),right=True,color=NEG),
            _td(fmt_price(t["take_profit"],t["pair"]),right=True,color=POS),
            _td(fmt_pnl(pl),color=POS if pl>=0 else NEG,bold=True,right=True),
            _td(_del_btn(t["id"])),
        ]))

    pp=max(1,(len(pending_df)+PAGE_SIZE-1)//PAGE_SIZE)
    pg=max(0,min(pending_page,pp-1))
    pend_rows=[]
    for _,t in pending_df.iloc[pg*PAGE_SIZE:(pg+1)*PAGE_SIZE].iterrows():
        pend_rows.append(html.Tr([
            _td(label(t["pair"]),bold=True,color=BRIGHT),_td(t["timeframe"]),
            _td(_sig_chip(t["signal"])),
            _td(fmt_price(t["entry_price"],t["pair"]),right=True),
            _td(fmt_price(t["stop_loss"],t["pair"]),right=True,color=NEG),
            _td(fmt_price(t["take_profit"],t["pair"]),right=True,color=POS),
            _td(f"{t.get('risk_reward',0):.2f}",right=True),
            _td(_del_btn(t["id"])),
        ]))

    cp=max(1,(len(closed_df)+PAGE_SIZE-1)//PAGE_SIZE)
    cg=max(0,min(closed_page,cp-1))
    closed_rows=[]
    for _,t in closed_df.iloc[cg*PAGE_SIZE:(cg+1)*PAGE_SIZE].iterrows():
        pl=t.get("profit_loss",0) or 0
        closed_rows.append(html.Tr([
            _td(label(t["pair"]),bold=True,color=BRIGHT),_td(t["timeframe"]),
            _td(_sig_chip(t["signal"])),
            _td(fmt_price(t.get("entry_price"),t["pair"]),right=True),
            _td(fmt_price(t.get("exit_price"),t["pair"]),right=True),
            _td(t.get("outcome","—"),color=POS if t.get("outcome")=="WIN" else NEG,bold=True),
            _td(fmt_pnl(pl),color=POS if pl>=0 else NEG,right=True),
            _td(str(t.get("entry_time",""))[:10]),
        ]))

    def _panel(title,glyph,headers,rows_,empty,pager=None):
        return html.Div([
            html.Div([html.Span(glyph,style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                      html.Span(title,style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
                style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
            _table(headers,rows_,empty),
            *([pager] if pager else []),
        ],style={**_PANEL,"marginBottom":"10px"})

    return html.Div([
        _panel("ACTIVE POSITIONS","▶",
               ["PAIR","TF","DIR","ENTRY","CURRENT","SL","TP","P&L",""],
               active_rows,"NO ACTIVE POSITIONS"),
        _panel("PENDING ORDERS","◎",
               ["PAIR","TF","DIR","ENTRY","SL","TP","R/R",""],
               pend_rows,"NO PENDING ORDERS",
               pager=_pager("pending-prev-btn","pending-next-btn",pg,pp)),
        _panel("TRADE HISTORY","≡",
               ["PAIR","TF","DIR","ENTRY","EXIT","RESULT","P&L","DATE"],
               closed_rows,"NO CLOSED TRADES",
               pager=_pager("closed-prev-btn","closed-next-btn",cg,cp)),
    ],className="panel-in")


def _render_calendar():
    return html.Div([html.Div([
        html.Div([html.Span("◷",style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                  html.Span("ECONOMIC CALENDAR",style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
            style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
        html.Div([
            dcc.Dropdown(id="calendar-currency",
                options=[{"label":c,"value":c} for c in ("USD","EUR","GBP","JPY")],
                value="USD",clearable=False,
                style={"width":"130px","fontFamily":MONO,"fontSize":"11px","marginBottom":"8px"}),
        ],style={"padding":"8px 12px 0"}),
        html.Div(id="calendar-table-div"),
    ],style=_PANEL)],className="panel-in")


def _render_pipcalc():
    pair_opts=[{"label":label(p),"value":p} for p in ALL_PAIRS]
    inp={"width":"180px","padding":"6px 8px","border":f"1px solid {LINE}",
         "background":SURF2,"color":TEXT,"fontFamily":MONO,"fontSize":"11px",
         "outline":"none","borderRadius":"0"}
    lbl={"color":MUTED,"fontSize":"8px","letterSpacing":"0.1em","fontFamily":MONO,
         "textTransform":"uppercase","display":"block","marginBottom":"5px"}
    return html.Div([html.Div([
        html.Div([html.Span("σ",style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                  html.Span("POSITION SIZE CALCULATOR",style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
            style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
        html.Div([
            html.Div([html.Span("INSTRUMENT",style=lbl),
                dcc.Dropdown(id="pip-pair",options=pair_opts,value="EURUSD=X",
                    clearable=False,style={"width":"180px"})],style={"marginBottom":"12px"}),
            html.Div([html.Span("RISK AMOUNT ($)",style=lbl),
                dcc.Input(id="risk-amount",type="number",value=100,min=1,step=10,style=inp)],
                style={"marginBottom":"12px"}),
            html.Div([html.Span("STOP LOSS (PIPS)",style=lbl),
                dcc.Input(id="stop-loss-pips",type="number",value=20,min=1,step=1,style=inp)],
                style={"marginBottom":"14px"}),
            html.Button("CALCULATE",id="calc-btn",n_clicks=0,style=_BTN_P),
            html.Div(id="pip-result",style={"marginTop":"12px"}),
        ],style={"padding":"0 12px 12px"}),
    ],style={**_PANEL,"maxWidth":"400px"})],className="panel-in")


def _render_analytics(prices=None):
    if prices is None: prices={}
    data   = build_analytics_data(prices)
    adv    = data["adv"]
    closed = data["closed_df"]
    ar     = data["active_rows"]
    unreal = data["total_unreal"]
    pend_c = data["pending_count"]

    chash = hash(str(closed["entry_time"].tolist()) if not closed.empty else "") ^ hash(str(ar))
    gcfg  = {"displayModeBar":False}

    # ── Structured Performance Summary ────────────────────────────────────────
    perf_summary = build_performance_summary(adv, ar, unreal, pend_c, closed)

    # ── Active position cards ─────────────────────────────────────────────────
    active_section = html.Div()
    if ar:
        cards = html.Div([
            html.Div([
                html.Div([
                    html.Span(label(r["pair"]),style={
                        "color":BRIGHT,"fontWeight":"500","fontFamily":DISPLAY,
                        "fontSize":"15px","letterSpacing":"0.05em",
                    }),
                    html.Span(f"  {'▲' if r['signal']=='BUY' else '▼'} {r['signal']}",
                        style={"color":POS if r["signal"]=="BUY" else NEG,
                               "fontFamily":MONO,"fontSize":"10px"}),
                ],style={"marginBottom":"6px"}),
                html.Div(fmt_pnl(r["unrealised"]),style={
                    "color":POS if r["unrealised"]>=0 else NEG,
                    "fontFamily":DISPLAY,"fontSize":"24px","fontWeight":"500","lineHeight":"1",
                },className="pulse" if abs(r["unrealised"])>0.5 else ""),
                html.Div("UNREALISED",style={"color":DIM,"fontFamily":MONO,"fontSize":"7px",
                                              "letterSpacing":"0.1em","marginTop":"4px"}),
                html.Div([
                    html.Span(f"E  {fmt_price(r['entry_price'],r['pair'])}",
                        style={"color":MUTED,"fontFamily":MONO,"fontSize":"10px","marginRight":"14px"}),
                    html.Span(f"C  {fmt_price(r['current_price'],r['pair'])}",
                        style={"color":TEXT,"fontFamily":MONO,"fontSize":"10px"}),
                ],style={"marginTop":"7px"}),
            ],style={
                "backgroundColor":SURF,"border":f"1px solid {LINE}",
                "borderTop":f"2px solid {POS if r['unrealised']>=0 else NEG}",
                "padding":"12px 16px","minWidth":"148px","flex":"1",
            })
            for r in ar
        ],style={"display":"flex","flexWrap":"wrap","gap":"1px"})
        active_section = html.Div([
            html.Div([html.Span("▶",style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                      html.Span("LIVE POSITIONS",style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
                style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
            cards,
        ],style={**_PANEL,"marginBottom":"10px"})

    # ── Charts ────────────────────────────────────────────────────────────────
    equity_panel = html.Div([
        html.Div([html.Span("◉",style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                  html.Span("EQUITY CURVE",style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
            style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
        dcc.Graph(figure=_cached_fig("eq",chash,lambda df=closed,a=ar:_fig_equity(df,a)),config=gcfg),
    ],style={**_PANEL,"marginBottom":"10px"})

    rolling_panel = html.Div([
        html.Div([html.Span("∿",style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                  html.Span("ROLLING WIN RATE  (10-trade window)",style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
            style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
        dcc.Graph(figure=_cached_fig("rw",chash,lambda df=closed:_fig_rolling(df)),config=gcfg),
    ],style={**_PANEL,"marginBottom":"10px"})

    charts_row = html.Div([
        html.Div([
            html.Div([html.Span("●",style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                      html.Span("WIN RATE",style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
                style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
            dcc.Graph(figure=_cached_fig("dn",chash,
                lambda a=adv:_fig_donut(a["wins"],a["losses"],a["win_rate"])),config=gcfg),
        ],style={**_PANEL,"flex":"1","minWidth":"160px","marginBottom":"0"}),
        html.Div([
            html.Div([html.Span("▦",style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                      html.Span("MONTHLY P&L",style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
                style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
            dcc.Graph(figure=_cached_fig("mo",chash,
                lambda df=closed:_fig_bar_chart(df,"month","MONTHLY P&L")),config=gcfg),
        ],style={**_PANEL,"flex":"2","minWidth":"200px","marginBottom":"0","marginLeft":"1px"}),
        html.Div([
            html.Div([html.Span("▧",style={"color":DIM,"marginRight":"8px","fontSize":"8px"}),
                      html.Span("P&L BY PAIR",style={"fontFamily":MONO,"fontSize":"8px","letterSpacing":"0.1em","textTransform":"uppercase","color":MUTED})],
                style={"padding":"7px 12px 0","display":"flex","alignItems":"center"}),
            dcc.Graph(figure=_cached_fig("pb",chash,
                lambda df=closed:_fig_bar_chart(df,"pair","P&L BY PAIR")),config=gcfg),
        ],style={**_PANEL,"flex":"2","minWidth":"200px","marginBottom":"0","marginLeft":"1px"}),
    ],style={"display":"flex","gap":"0","marginBottom":"10px"})

    # Staggered animation classes
    return html.Div([
        html.Div(perf_summary,  className="analytics-t1"),
        html.Div(active_section,className="analytics-t2"),
        html.Div(equity_panel,  className="analytics-t3"),
        html.Div(rolling_panel, className="analytics-t4"),
        html.Div(charts_row,    className="analytics-t5"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kick_scan(force=False)
    app.run(debug=False, port=8050, threaded=True)