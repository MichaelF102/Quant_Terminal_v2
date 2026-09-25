"""Market Explorer & Institutional Screener.

Features:
1. Institutional Multi-Factor Stock Screener powered by TradingView Screener with
   clickable condition filters, quick presets, and numerical inputs.
2. Deep-Dive 2-Stock Comparison with head-to-head performance matrix, fundamental
   scorecard, normalized relative alpha charts, rolling risk, and correlation scatter.
"""

from datetime import date, timedelta
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# Ensure utils directory is in Python path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tradingview_screener import Query, col
from utils.helper import (
    inject_custom_theme,
    load_data,
    fetch_yf_info,
    _fmt_money,
    _fmt_num,
    _fmt_pct,
    CURRENCY_SYMBOLS,
)


# -----------------------------------------------------------------------------
# TradingView Screener Engine
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def run_tradingview_screener(
    market: str = "india",
    exchange: str = "NSE",
    preset: str = "All Stocks (Custom Filters)",
    sector: str = "All",
    min_mcap: float = 0.0,
    max_mcap: float = 1e15,
    min_pe: float = 0.0,
    max_pe: float = 200.0,
    min_rsi: float = 0.0,
    max_rsi: float = 100.0,
    min_rvol: float = 0.0,
    min_change: float = -100.0,
    max_change: float = 100.0,
    min_price: float = 0.0,
    max_price: float = 1000000.0,
    ma_filter: str = "Any",
    sort_by: str = "market_cap_basic",
    sort_asc: bool = False,
    limit: int = 50,
) -> tuple[int, pd.DataFrame]:
    """Execute multi-condition query against TradingView Screener API."""
    try:
        q = Query().set_markets(market)
        cols = [
            "name",
            "description",
            "close",
            "change",
            "volume",
            "relative_volume_10d_calc",
            "RSI",
            "market_cap_basic",
            "price_earnings_ttm",
            "Recommend.All",
            "sector",
            "exchange",
            "SMA50",
            "SMA200",
        ]
        q = q.select(*cols)

        conditions = []

        # Exchange filter
        if exchange and exchange != "All" and "All" not in exchange:
            conditions.append(col("exchange") == exchange)
        elif market == "india" and (exchange == "All" or "All" in exchange):
            conditions.append(col("exchange").isin(["NSE", "BSE"]))

        # Preset rules
        if preset == "🚀 Bullish Momentum Breakout":
            conditions.extend([
                col("RSI") >= 55,
                col("RSI") <= 75,
                col("change") > 0,
                col("close") > col("SMA50"),
            ])
        elif preset == "📉 Oversold Mean Reversion":
            conditions.extend([col("RSI") < 35, col("RSI") > 5])
        elif preset == "⚡ High Volume Accumulation":
            conditions.extend([col("relative_volume_10d_calc") >= 1.5, col("change") > 0])
        elif preset == "👑 Large-Cap Quality Compounders":
            threshold = 50000000000 if market == "india" else 50000000000  # ₹5k Cr or $50B
            conditions.append(col("market_cap_basic") >= threshold)
        elif preset == "💰 High Dividend Value":
            conditions.extend([col("price_earnings_ttm") > 0, col("price_earnings_ttm") <= 25])
        elif preset == "🏆 52-Week High Breakouts":
            conditions.extend([col("change") > 1.0, col("RSI") >= 60])
        elif preset == "🛡️ Strong Buy Technical Consensus":
            conditions.append(col("Recommend.All") >= 0.3)

        # Numerical input condition filters
        if min_mcap > 0:
            conditions.append(col("market_cap_basic") >= min_mcap)
        if max_mcap < 1e15:
            conditions.append(col("market_cap_basic") <= max_mcap)

        if min_pe > 0:
            conditions.append(col("price_earnings_ttm") >= min_pe)
        if max_pe < 200:
            conditions.append(col("price_earnings_ttm") <= max_pe)

        if min_rsi > 0:
            conditions.append(col("RSI") >= min_rsi)
        if max_rsi < 100:
            conditions.append(col("RSI") <= max_rsi)

        if min_rvol > 0:
            conditions.append(col("relative_volume_10d_calc") >= min_rvol)

        if min_change > -100:
            conditions.append(col("change") >= min_change)
        if max_change < 100:
            conditions.append(col("change") <= max_change)

        if min_price > 0:
            conditions.append(col("close") >= min_price)
        if max_price < 1000000:
            conditions.append(col("close") <= max_price)

        if sector != "All" and sector.strip():
            conditions.append(col("sector") == sector.strip())

        # Moving Average condition
        if ma_filter == "Price > 50 SMA":
            conditions.append(col("close") > col("SMA50"))
        elif ma_filter == "Price > 200 SMA":
            conditions.append(col("close") > col("SMA200"))
        elif ma_filter == "Price > 50 SMA & 200 SMA":
            conditions.extend([col("close") > col("SMA50"), col("close") > col("SMA200")])
        elif ma_filter == "Golden Cross (50 SMA > 200 SMA)":
            conditions.append(col("SMA50") > col("SMA200"))

        if conditions:
            q = q.where(*conditions)

        # Order & limit
        q = q.order_by(sort_by, ascending=sort_asc).limit(limit)

        count, df = q.get_scanner_data()
        return count, df
    except Exception as e:
        st.warning(f"TradingView Screener query warning: {e}")
        return 0, pd.DataFrame()


# -----------------------------------------------------------------------------
# Snapshot Data Engine & Deep Comparison Helpers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_universe_snapshots() -> dict:
    """Load and index stocks from India and US CSV snapshot data."""
    root = Path(__file__).resolve().parent.parent
    in_path = root / "data" / "snapshots" / "India_Stocks_Data.csv"
    us_path = root / "data" / "snapshots" / "US_Stocks_Data.csv"

    records_by_label = {}
    records_by_ticker = {}
    india_labels = []
    us_labels = []

    # 1. Process India Snapshot
    if in_path.exists():
        try:
            df_in = pd.read_csv(in_path)
            for _, r in df_in.iterrows():
                sym = str(r["Symbol"]).strip() if pd.notna(r["Symbol"]) else ""
                if not sym:
                    continue
                desc = str(r["Description"]).strip() if pd.notna(r["Description"]) else sym
                ex = str(r["Exchange"]).strip().upper() if pd.notna(r["Exchange"]) else "NSE"
                if ex == "NSE":
                    yf_ticker = f"{sym}.NS"
                elif ex == "BSE":
                    yf_ticker = f"{sym}.BO"
                else:
                    yf_ticker = sym

                label = f"{desc} ({yf_ticker} · {ex})"
                rec = {
                    "yf_ticker": yf_ticker,
                    "symbol": sym,
                    "name": desc,
                    "exchange": ex,
                    "sector": str(r["Sector"]).strip() if pd.notna(r["Sector"]) else "Equity",
                    "price": float(r["Price"]) if pd.notna(r["Price"]) else None,
                    "currency": str(r["Price - Currency"]).strip() if pd.notna(r["Price - Currency"]) else "INR",
                    "change_1d": float(r["Price change %, 1 day"]) if pd.notna(r["Price change %, 1 day"]) else None,
                    "volume_1d": float(r["Volume, 1 day"]) if pd.notna(r["Volume, 1 day"]) else None,
                    "rvol_1d": float(r["Relative volume, 1 day"]) if pd.notna(r["Relative volume, 1 day"]) else None,
                    "mcap": float(r["Market capitalization"]) if pd.notna(r["Market capitalization"]) else None,
                    "mcap_curr": str(r["Market capitalization - Currency"]).strip() if pd.notna(r["Market capitalization - Currency"]) else "INR",
                    "pe": float(r["Price to earnings ratio"]) if pd.notna(r["Price to earnings ratio"]) else None,
                    "eps": float(r["Earnings per share diluted, Trailing 12 months"]) if pd.notna(r["Earnings per share diluted, Trailing 12 months"]) else None,
                    "eps_growth": float(r["Earnings per share diluted growth %, TTM YoY"]) if pd.notna(r["Earnings per share diluted growth %, TTM YoY"]) else None,
                    "div_yield": float(r["Dividend yield %, Trailing 12 months"]) if pd.notna(r["Dividend yield %, Trailing 12 months"]) else None,
                    "market": "India",
                }
                india_labels.append(label)
                records_by_label[label] = rec
                records_by_ticker[yf_ticker.upper()] = rec
                if sym.upper() not in records_by_ticker:
                    records_by_ticker[sym.upper()] = rec
        except Exception as e:
            st.warning(f"Error reading India snapshot: {e}")

    # 2. Process US Snapshot
    if us_path.exists():
        try:
            df_us = pd.read_csv(us_path)
            for _, r in df_us.iterrows():
                sym = str(r["Symbol"]).strip() if pd.notna(r["Symbol"]) else ""
                if not sym:
                    continue
                desc = str(r["Description"]).strip() if pd.notna(r["Description"]) else sym
                ex = str(r["Exchange"]).strip().upper() if pd.notna(r["Exchange"]) else "US"
                yf_ticker = sym

                label = f"{desc} ({yf_ticker} · {ex})"
                rec = {
                    "yf_ticker": yf_ticker,
                    "symbol": sym,
                    "name": desc,
                    "exchange": ex,
                    "sector": str(r["Sector"]).strip() if pd.notna(r["Sector"]) else "Equity",
                    "price": float(r["Price"]) if pd.notna(r["Price"]) else None,
                    "currency": str(r["Price - Currency"]).strip() if pd.notna(r["Price - Currency"]) else "USD",
                    "change_1d": float(r["Price change %, 1 day"]) if pd.notna(r["Price change %, 1 day"]) else None,
                    "volume_1d": float(r["Volume, 1 day"]) if pd.notna(r["Volume, 1 day"]) else None,
                    "rvol_1d": float(r["Relative volume, 1 day"]) if pd.notna(r["Relative volume, 1 day"]) else None,
                    "mcap": float(r["Market capitalization"]) if pd.notna(r["Market capitalization"]) else None,
                    "mcap_curr": str(r["Market capitalization - Currency"]).strip() if pd.notna(r["Market capitalization - Currency"]) else "USD",
                    "pe": float(r["Price to earnings ratio"]) if pd.notna(r["Price to earnings ratio"]) else None,
                    "eps": float(r["Earnings per share diluted, Trailing 12 months"]) if pd.notna(r["Earnings per share diluted, Trailing 12 months"]) else None,
                    "eps_growth": float(r["Earnings per share diluted growth %, TTM YoY"]) if pd.notna(r["Earnings per share diluted growth %, TTM YoY"]) else None,
                    "div_yield": float(r["Dividend yield %, Trailing 12 months"]) if pd.notna(r["Dividend yield %, Trailing 12 months"]) else None,
                    "market": "US",
                }
                us_labels.append(label)
                records_by_label[label] = rec
                records_by_ticker[yf_ticker.upper()] = rec
                if sym.upper() not in records_by_ticker:
                    records_by_ticker[sym.upper()] = rec
        except Exception as e:
            st.warning(f"Error reading US snapshot: {e}")

    all_labels = india_labels + us_labels
    return {
        "india_labels": india_labels,
        "us_labels": us_labels,
        "all_labels": all_labels,
        "by_label": records_by_label,
        "by_ticker": records_by_ticker,
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_comparison_pair_data(
    ticker_a: str, ticker_b: str, period: str = "1y", interval: str = "1d"
) -> tuple[pd.DataFrame, dict, dict]:
    """Fetch synced OHLCV and info metadata for two comparison assets."""
    resolved_a = ticker_a.strip()
    resolved_b = ticker_b.strip()

    raw = yf.download(
        [resolved_a, resolved_b],
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    info_a = fetch_yf_info(resolved_a)
    info_b = fetch_yf_info(resolved_b)

    return raw, info_a, info_b


def build_head_to_head_matrix(
    ca: pd.Series,
    cb: pd.Series,
    info_a: dict,
    info_b: dict,
    snap_a: dict | None,
    snap_b: dict | None,
    name_a: str,
    name_b: str,
) -> pd.DataFrame:
    """Compile comprehensive head-to-head comparison metrics merging snapshots and live market data."""
    rets_a = ca.pct_change().dropna()
    rets_b = cb.pct_change().dropna()

    snap_a = snap_a or {}
    snap_b = snap_b or {}

    # 1. Performance Calculations
    def calc_period_ret(s: pd.Series, n_bars: int) -> float:
        if len(s) > n_bars:
            return float((s.iloc[-1] / s.iloc[-n_bars - 1] - 1.0) * 100.0)
        return float((s.iloc[-1] / s.iloc[0] - 1.0) * 100.0) if len(s) > 1 else 0.0

    ret_1d_a = snap_a.get("change_1d") if snap_a.get("change_1d") is not None else calc_period_ret(ca, 1)
    ret_1d_b = snap_b.get("change_1d") if snap_b.get("change_1d") is not None else calc_period_ret(cb, 1)
    ret_1w_a = calc_period_ret(ca, 5)
    ret_1w_b = calc_period_ret(cb, 5)
    ret_1m_a = calc_period_ret(ca, 21)
    ret_1m_b = calc_period_ret(cb, 21)
    ret_3m_a = calc_period_ret(ca, 63)
    ret_3m_b = calc_period_ret(cb, 63)
    ret_6m_a = calc_period_ret(ca, 126)
    ret_6m_b = calc_period_ret(cb, 126)
    ret_1y_a = calc_period_ret(ca, 252)
    ret_1y_b = calc_period_ret(cb, 252)
    tot_ret_a = float((ca.iloc[-1] / ca.iloc[0] - 1.0) * 100.0) if len(ca) > 1 else 0.0
    tot_ret_b = float((cb.iloc[-1] / cb.iloc[0] - 1.0) * 100.0) if len(cb) > 1 else 0.0

    # Annualized CAGR
    n_days = max(len(ca), 1)
    cagr_a = float(((ca.iloc[-1] / ca.iloc[0]) ** (252.0 / n_days) - 1.0) * 100.0) if n_days > 20 and ca.iloc[0] > 0 else tot_ret_a
    cagr_b = float(((cb.iloc[-1] / cb.iloc[0]) ** (252.0 / n_days) - 1.0) * 100.0) if n_days > 20 and cb.iloc[0] > 0 else tot_ret_b

    win_rate_a = float((rets_a > 0).mean() * 100.0) if len(rets_a) > 0 else 0.0
    win_rate_b = float((rets_b > 0).mean() * 100.0) if len(rets_b) > 0 else 0.0

    best_day_a = float(rets_a.max() * 100.0) if len(rets_a) > 0 else 0.0
    best_day_b = float(rets_b.max() * 100.0) if len(rets_b) > 0 else 0.0
    worst_day_a = float(rets_a.min() * 100.0) if len(rets_a) > 0 else 0.0
    worst_day_b = float(rets_b.min() * 100.0) if len(rets_b) > 0 else 0.0

    # High / Low Distances
    roll_high_a = float(ca.max()) if len(ca) > 0 else float(ca.iloc[-1])
    roll_high_b = float(cb.max()) if len(cb) > 0 else float(cb.iloc[-1])
    roll_low_a = float(ca.min()) if len(ca) > 0 else float(ca.iloc[-1])
    roll_low_b = float(cb.min()) if len(cb) > 0 else float(cb.iloc[-1])

    dist_high_a = float((ca.iloc[-1] / roll_high_a - 1.0) * 100.0) if roll_high_a > 0 else 0.0
    dist_high_b = float((cb.iloc[-1] / roll_high_b - 1.0) * 100.0) if roll_high_b > 0 else 0.0
    dist_low_a = float((ca.iloc[-1] / roll_low_a - 1.0) * 100.0) if roll_low_a > 0 else 0.0
    dist_low_b = float((cb.iloc[-1] / roll_low_b - 1.0) * 100.0) if roll_low_b > 0 else 0.0

    # 2. Volatility, Tail Risk & Statistical Dispersion
    vol_a = float(rets_a.std() * np.sqrt(252) * 100.0) if len(rets_a) > 2 else 0.0
    vol_b = float(rets_b.std() * np.sqrt(252) * 100.0) if len(rets_b) > 2 else 0.0

    dd_a = float(((ca - ca.cummax()) / ca.cummax()).min() * 100.0) if len(ca) > 1 else 0.0
    dd_b = float(((cb - cb.cummax()) / cb.cummax()).min() * 100.0) if len(cb) > 1 else 0.0

    sharpe_a = (
        float((rets_a.mean() * 252) / (rets_a.std() * np.sqrt(252) + 1e-9))
        if rets_a.std() > 0
        else 0.0
    )
    sharpe_b = (
        float((rets_b.mean() * 252) / (rets_b.std() * np.sqrt(252) + 1e-9))
        if rets_b.std() > 0
        else 0.0
    )

    downside_a = rets_a[rets_a < 0]
    downside_b = rets_b[rets_b < 0]
    sortino_a = (
        float((rets_a.mean() * 252) / (downside_a.std() * np.sqrt(252) + 1e-9))
        if len(downside_a) > 1 and downside_a.std() > 0
        else sharpe_a
    )
    sortino_b = (
        float((rets_b.mean() * 252) / (downside_b.std() * np.sqrt(252) + 1e-9))
        if len(downside_b) > 1 and downside_b.std() > 0
        else sharpe_b
    )

    calmar_a = float(abs(cagr_a / dd_a)) if abs(dd_a) > 0.01 else 0.0
    calmar_b = float(abs(cagr_b / dd_b)) if abs(dd_b) > 0.01 else 0.0

    var_95_a = float(abs(np.percentile(rets_a, 5)) * 100.0) if len(rets_a) > 10 else 0.0
    var_95_b = float(abs(np.percentile(rets_b, 5)) * 100.0) if len(rets_b) > 10 else 0.0

    # Conditional VaR (Expected Shortfall)
    tail_a = rets_a[rets_a <= -var_95_a / 100.0]
    cvar_95_a = float(abs(tail_a.mean()) * 100.0) if len(tail_a) > 0 else var_95_a
    tail_b = rets_b[rets_b <= -var_95_b / 100.0]
    cvar_95_b = float(abs(tail_b.mean()) * 100.0) if len(tail_b) > 0 else var_95_b

    semi_dev_a = float(downside_a.std() * np.sqrt(252) * 100.0) if len(downside_a) > 2 else vol_a
    semi_dev_b = float(downside_b.std() * np.sqrt(252) * 100.0) if len(downside_b) > 2 else vol_b

    skew_a = float(rets_a.skew()) if len(rets_a) > 5 else 0.0
    skew_b = float(rets_b.skew()) if len(rets_b) > 5 else 0.0
    kurt_a = float(rets_a.kurtosis()) if len(rets_a) > 5 else 0.0
    kurt_b = float(rets_b.kurtosis()) if len(rets_b) > 5 else 0.0

    beta_a = float(info_a.get("beta", 1.0)) if info_a.get("beta") else 1.0
    beta_b = float(info_b.get("beta", 1.0)) if info_b.get("beta") else 1.0

    # 3. Valuation & Enterprise Multiples
    pe_a = snap_a.get("pe") or (float(info_a.get("trailingPE", 0)) if info_a.get("trailingPE") else None)
    pe_b = snap_b.get("pe") or (float(info_b.get("trailingPE", 0)) if info_b.get("trailingPE") else None)
    fwd_pe_a = float(info_a.get("forwardPE", 0)) if info_a.get("forwardPE") else None
    fwd_pe_b = float(info_b.get("forwardPE", 0)) if info_b.get("forwardPE") else None
    peg_a = float(info_a.get("pegRatio", 0)) if info_a.get("pegRatio") else None
    peg_b = float(info_b.get("pegRatio", 0)) if info_b.get("pegRatio") else None

    pb_a = float(info_a.get("priceToBook", 0)) if info_a.get("priceToBook") else None
    pb_b = float(info_b.get("priceToBook", 0)) if info_b.get("priceToBook") else None
    ps_a = float(info_a.get("priceToSalesTrailing12Months", 0)) if info_a.get("priceToSalesTrailing12Months") else None
    ps_b = float(info_b.get("priceToSalesTrailing12Months", 0)) if info_b.get("priceToSalesTrailing12Months") else None

    mcap_a = snap_a.get("mcap") or (float(info_a.get("marketCap", 0)) if info_a.get("marketCap") else None)
    mcap_b = snap_b.get("mcap") or (float(info_b.get("marketCap", 0)) if info_b.get("marketCap") else None)
    ev_a = float(info_a.get("enterpriseValue", 0)) if info_a.get("enterpriseValue") else None
    ev_b = float(info_b.get("enterpriseValue", 0)) if info_b.get("enterpriseValue") else None

    ev_ebitda_a = float(info_a.get("enterpriseToEbitda", 0)) if info_a.get("enterpriseToEbitda") else None
    ev_ebitda_b = float(info_b.get("enterpriseToEbitda", 0)) if info_b.get("enterpriseToEbitda") else None
    ev_rev_a = float(info_a.get("enterpriseToRevenue", 0)) if info_a.get("enterpriseToRevenue") else None
    ev_rev_b = float(info_b.get("enterpriseToRevenue", 0)) if info_b.get("enterpriseToRevenue") else None

    eps_a = snap_a.get("eps") or (float(info_a.get("trailingEps", 0)) if info_a.get("trailingEps") else None)
    eps_b = snap_b.get("eps") or (float(info_b.get("trailingEps", 0)) if info_b.get("trailingEps") else None)
    fwd_eps_a = float(info_a.get("forwardEps", 0)) if info_a.get("forwardEps") else None
    fwd_eps_b = float(info_b.get("forwardEps", 0)) if info_b.get("forwardEps") else None
    eps_growth_a = snap_a.get("eps_growth") or (float(info_a.get("earningsGrowth", 0) * 100) if info_a.get("earningsGrowth") else None)
    eps_growth_b = snap_b.get("eps_growth") or (float(info_b.get("earningsGrowth", 0) * 100) if info_b.get("earningsGrowth") else None)
    book_val_a = float(info_a.get("bookValue", 0)) if info_a.get("bookValue") else None
    book_val_b = float(info_b.get("bookValue", 0)) if info_b.get("bookValue") else None

    div_a = snap_a.get("div_yield") if snap_a.get("div_yield") is not None else (float(info_a.get("dividendYield", 0) * 100) if info_a.get("dividendYield") else 0.0)
    div_b = snap_b.get("div_yield") if snap_b.get("div_yield") is not None else (float(info_b.get("dividendYield", 0) * 100) if info_b.get("dividendYield") else 0.0)

    # 4. Quality, Profitability & Cash Flow
    roe_a = float(info_a.get("returnOnEquity", 0) * 100) if info_a.get("returnOnEquity") else None
    roe_b = float(info_b.get("returnOnEquity", 0) * 100) if info_b.get("returnOnEquity") else None
    roa_a = float(info_a.get("returnOnAssets", 0) * 100) if info_a.get("returnOnAssets") else None
    roa_b = float(info_b.get("returnOnAssets", 0) * 100) if info_b.get("returnOnAssets") else None

    gross_margin_a = float(info_a.get("grossMargins", 0) * 100) if info_a.get("grossMargins") else None
    gross_margin_b = float(info_b.get("grossMargins", 0) * 100) if info_b.get("grossMargins") else None
    op_margin_a = float(info_a.get("operatingMargins", 0) * 100) if info_a.get("operatingMargins") else None
    op_margin_b = float(info_b.get("operatingMargins", 0) * 100) if info_b.get("operatingMargins") else None
    margin_a = float(info_a.get("profitMargins", 0) * 100) if info_a.get("profitMargins") else None
    margin_b = float(info_b.get("profitMargins", 0) * 100) if info_b.get("profitMargins") else None

    rev_a = float(info_a.get("totalRevenue", 0)) if info_a.get("totalRevenue") else None
    rev_b = float(info_b.get("totalRevenue", 0)) if info_b.get("totalRevenue") else None
    rev_growth_a = float(info_a.get("revenueGrowth", 0) * 100) if info_a.get("revenueGrowth") else None
    rev_growth_b = float(info_b.get("revenueGrowth", 0) * 100) if info_b.get("revenueGrowth") else None
    ebitda_a = float(info_a.get("ebitda", 0)) if info_a.get("ebitda") else None
    ebitda_b = float(info_b.get("ebitda", 0)) if info_b.get("ebitda") else None
    fcf_a = float(info_a.get("freeCashflow", 0)) if info_a.get("freeCashflow") else None
    fcf_b = float(info_b.get("freeCashflow", 0)) if info_b.get("freeCashflow") else None
    ocf_a = float(info_a.get("operatingCashflow", 0)) if info_a.get("operatingCashflow") else None
    ocf_b = float(info_b.get("operatingCashflow", 0)) if info_b.get("operatingCashflow") else None

    # 5. Financial Health & Solvency
    de_a = float(info_a.get("debtToEquity", 0)) if info_a.get("debtToEquity") else None
    de_b = float(info_b.get("debtToEquity", 0)) if info_b.get("debtToEquity") else None
    curr_ratio_a = float(info_a.get("currentRatio", 0)) if info_a.get("currentRatio") else None
    curr_ratio_b = float(info_b.get("currentRatio", 0)) if info_b.get("currentRatio") else None
    quick_ratio_a = float(info_a.get("quickRatio", 0)) if info_a.get("quickRatio") else None
    quick_ratio_b = float(info_b.get("quickRatio", 0)) if info_b.get("quickRatio") else None
    total_cash_a = float(info_a.get("totalCash", 0)) if info_a.get("totalCash") else None
    total_cash_b = float(info_b.get("totalCash", 0)) if info_b.get("totalCash") else None
    total_debt_a = float(info_a.get("totalDebt", 0)) if info_a.get("totalDebt") else None
    total_debt_b = float(info_b.get("totalDebt", 0)) if info_b.get("totalDebt") else None

    # 6. Ownership, Liquidity & Shares
    vol_1d_a = snap_a.get("volume_1d") or (float(info_a.get("volume", 0)) if info_a.get("volume") else None)
    vol_1d_b = snap_b.get("volume_1d") or (float(info_b.get("volume", 0)) if info_b.get("volume") else None)
    rvol_a = snap_a.get("rvol_1d")
    rvol_b = snap_b.get("rvol_1d")
    inst_own_a = float(info_a.get("heldPercentInstitutions", 0) * 100) if info_a.get("heldPercentInstitutions") else None
    inst_own_b = float(info_b.get("heldPercentInstitutions", 0) * 100) if info_b.get("heldPercentInstitutions") else None
    insider_own_a = float(info_a.get("heldPercentInsiders", 0) * 100) if info_a.get("heldPercentInsiders") else None
    insider_own_b = float(info_b.get("heldPercentInsiders", 0) * 100) if info_b.get("heldPercentInsiders") else None
    shares_out_a = float(info_a.get("sharesOutstanding", 0)) if info_a.get("sharesOutstanding") else None
    shares_out_b = float(info_b.get("sharesOutstanding", 0)) if info_b.get("sharesOutstanding") else None
    short_ratio_a = float(info_a.get("shortRatio", 0)) if info_a.get("shortRatio") else None
    short_ratio_b = float(info_b.get("shortRatio", 0)) if info_b.get("shortRatio") else None

    sector_a = snap_a.get("sector") or info_a.get("sector", "Equity")
    sector_b = snap_b.get("sector") or info_b.get("sector", "Equity")
    ex_a = snap_a.get("exchange") or info_a.get("exchange", "Exchange")
    ex_b = snap_b.get("exchange") or info_b.get("exchange", "Exchange")

    # Helpers to decide advantage and delta spread
    def pick_adv(val_a, val_b, higher_is_better=True):
        if val_a is None or val_b is None:
            return "—"
        try:
            fa = float(val_a)
            fb = float(val_b)
        except (ValueError, TypeError):
            return "—"
        if np.isnan(fa) or np.isnan(fb):
            return "—"
        if abs(fa - fb) < 1e-6:
            return "Tied"
        if higher_is_better:
            return f"🟢 {name_a}" if fa > fb else f"🔵 {name_b}"
        else:
            return f"🟢 {name_a}" if fa < fb else f"🔵 {name_b}"

    def calc_delta(val_a, val_b, kind="num"):
        if val_a is None or val_b is None:
            return "—"
        try:
            fa = float(val_a)
            fb = float(val_b)
        except (ValueError, TypeError):
            return "—"
        if np.isnan(fa) or np.isnan(fb):
            return "—"
        d = fa - fb
        if kind == "pct":
            return f"{d:+.2f}%"
        elif kind == "mult":
            return f"{d:+.2f}x"
        elif kind == "curr":
            sign = "+" if d >= 0 else ""
            return f"{sign}{_fmt_num(d)}"
        elif kind == "raw":
            return f"{d:+.2f}"
        return f"{d:+.2f}"

    rows = [
        # 1. Performance & Trajectory
        {"Category": "Performance & Returns", "Metric": "Total Period Trajectory", name_a: f"{tot_ret_a:+.2f}%", name_b: f"{tot_ret_b:+.2f}%", "Spread / Delta": calc_delta(tot_ret_a, tot_ret_b, "pct"), "Advantage": pick_adv(tot_ret_a, tot_ret_b, True), "Institutional Benchmark": "Cumulative window return"},
        {"Category": "Performance & Returns", "Metric": "Annualized Return (CAGR)", name_a: f"{cagr_a:+.2f}%", name_b: f"{cagr_b:+.2f}%", "Spread / Delta": calc_delta(cagr_a, cagr_b, "pct"), "Advantage": pick_adv(cagr_a, cagr_b, True), "Institutional Benchmark": "Hurdle rate target >12.0%"},
        {"Category": "Performance & Returns", "Metric": "1-Day Return (Last Session)", name_a: f"{ret_1d_a:+.2f}%", name_b: f"{ret_1d_b:+.2f}%", "Spread / Delta": calc_delta(ret_1d_a, ret_1d_b, "pct"), "Advantage": pick_adv(ret_1d_a, ret_1d_b, True), "Institutional Benchmark": "Daily market momentum"},
        {"Category": "Performance & Returns", "Metric": "1-Week Return (5 Bars)", name_a: f"{ret_1w_a:+.2f}%", name_b: f"{ret_1w_b:+.2f}%", "Spread / Delta": calc_delta(ret_1w_a, ret_1w_b, "pct"), "Advantage": pick_adv(ret_1w_a, ret_1w_b, True), "Institutional Benchmark": "Short-term swing velocity"},
        {"Category": "Performance & Returns", "Metric": "1-Month Return (21 Bars)", name_a: f"{ret_1m_a:+.2f}%", name_b: f"{ret_1m_b:+.2f}%", "Spread / Delta": calc_delta(ret_1m_a, ret_1m_b, "pct"), "Advantage": pick_adv(ret_1m_a, ret_1m_b, True), "Institutional Benchmark": "Monthly trend continuation"},
        {"Category": "Performance & Returns", "Metric": "3-Month Return (63 Bars)", name_a: f"{ret_3m_a:+.2f}%", name_b: f"{ret_3m_b:+.2f}%", "Spread / Delta": calc_delta(ret_3m_a, ret_3m_b, "pct"), "Advantage": pick_adv(ret_3m_a, ret_3m_b, True), "Institutional Benchmark": "Quarterly earnings cycle"},
        {"Category": "Performance & Returns", "Metric": "6-Month Return (126 Bars)", name_a: f"{ret_6m_a:+.2f}%", name_b: f"{ret_6m_b:+.2f}%", "Spread / Delta": calc_delta(ret_6m_a, ret_6m_b, "pct"), "Advantage": pick_adv(ret_6m_a, ret_6m_b, True), "Institutional Benchmark": "Semi-annual relative strength"},
        {"Category": "Performance & Returns", "Metric": "1-Year Return (252 Bars)", name_a: f"{ret_1y_a:+.2f}%", name_b: f"{ret_1y_b:+.2f}%", "Spread / Delta": calc_delta(ret_1y_a, ret_1y_b, "pct"), "Advantage": pick_adv(ret_1y_a, ret_1y_b, True), "Institutional Benchmark": "52-Week alpha benchmark"},
        {"Category": "Performance & Returns", "Metric": "Distance from Period High (%)", name_a: f"{dist_high_a:+.2f}%", name_b: f"{dist_high_b:+.2f}%", "Spread / Delta": calc_delta(dist_high_a, dist_high_b, "pct"), "Advantage": pick_adv(dist_high_a, dist_high_b, True), "Institutional Benchmark": "Near 0% = Breakout territory"},
        {"Category": "Performance & Returns", "Metric": "Distance from Period Low (%)", name_a: f"{dist_low_a:+.2f}%", name_b: f"{dist_low_b:+.2f}%", "Spread / Delta": calc_delta(dist_low_a, dist_low_b, "pct"), "Advantage": pick_adv(dist_low_a, dist_low_b, True), "Institutional Benchmark": "Rebound from market floor"},
        {"Category": "Performance & Returns", "Metric": "Positive Session Win Rate", name_a: f"{win_rate_a:.1f}%", name_b: f"{win_rate_b:.1f}%", "Spread / Delta": calc_delta(win_rate_a, win_rate_b, "pct"), "Advantage": pick_adv(win_rate_a, win_rate_b, True), "Institutional Benchmark": "Consistent edge > 52.0%"},
        {"Category": "Performance & Returns", "Metric": "Best Single-Day Gain", name_a: f"{best_day_a:+.2f}%", name_b: f"{best_day_b:+.2f}%", "Spread / Delta": calc_delta(best_day_a, best_day_b, "pct"), "Advantage": pick_adv(best_day_a, best_day_b, True), "Institutional Benchmark": "Peak upside session burst"},
        {"Category": "Performance & Returns", "Metric": "Worst Single-Day Loss", name_a: f"{worst_day_a:+.2f}%", name_b: f"{worst_day_b:+.2f}%", "Spread / Delta": calc_delta(worst_day_a, worst_day_b, "pct"), "Advantage": pick_adv(worst_day_a, worst_day_b, True), "Institutional Benchmark": "Left-tail shock threshold"},

        # 2. Valuation & Enterprise Multiples
        {"Category": "Valuation & Fundamentals", "Metric": "Market Capitalization", name_a: _fmt_num(mcap_a) if mcap_a else "—", name_b: _fmt_num(mcap_b) if mcap_b else "—", "Spread / Delta": calc_delta(mcap_a, mcap_b, "curr"), "Advantage": pick_adv(mcap_a, mcap_b, True), "Institutional Benchmark": "Mega-Cap > $100B / Mid > $10B"},
        {"Category": "Valuation & Fundamentals", "Metric": "Enterprise Value (EV)", name_a: _fmt_num(ev_a) if ev_a else "—", name_b: _fmt_num(ev_b) if ev_b else "—", "Spread / Delta": calc_delta(ev_a, ev_b, "curr"), "Advantage": pick_adv(ev_a, ev_b, True), "Institutional Benchmark": "Total theoretical takeover cost"},
        {"Category": "Valuation & Fundamentals", "Metric": "Trailing P/E Ratio", name_a: f"{pe_a:.1f}x" if pe_a else "—", name_b: f"{pe_b:.1f}x" if pe_b else "—", "Spread / Delta": calc_delta(pe_a, pe_b, "mult"), "Advantage": pick_adv(pe_a, pe_b, False), "Institutional Benchmark": "Median peer multiple ~20–25x"},
        {"Category": "Valuation & Fundamentals", "Metric": "Forward P/E Ratio (NTM)", name_a: f"{fwd_pe_a:.1f}x" if fwd_pe_a else "—", name_b: f"{fwd_pe_b:.1f}x" if fwd_pe_b else "—", "Spread / Delta": calc_delta(fwd_pe_a, fwd_pe_b, "mult"), "Advantage": pick_adv(fwd_pe_a, fwd_pe_b, False), "Institutional Benchmark": "Forward valuation multiple"},
        {"Category": "Valuation & Fundamentals", "Metric": "PEG Ratio (PE to Growth)", name_a: f"{peg_a:.2f}x" if peg_a else "—", name_b: f"{peg_b:.2f}x" if peg_b else "—", "Spread / Delta": calc_delta(peg_a, peg_b, "mult"), "Advantage": pick_adv(peg_a, peg_b, False), "Institutional Benchmark": "Fair value < 1.5x / Cheap < 1.0x"},
        {"Category": "Valuation & Fundamentals", "Metric": "Price to Book (P/B)", name_a: f"{pb_a:.2f}x" if pb_a else "—", name_b: f"{pb_b:.2f}x" if pb_b else "—", "Spread / Delta": calc_delta(pb_a, pb_b, "mult"), "Advantage": pick_adv(pb_a, pb_b, False), "Institutional Benchmark": "Tangible asset backing"},
        {"Category": "Valuation & Fundamentals", "Metric": "Price to Sales (P/S TTM)", name_a: f"{ps_a:.2f}x" if ps_a else "—", name_b: f"{ps_b:.2f}x" if ps_b else "—", "Spread / Delta": calc_delta(ps_a, ps_b, "mult"), "Advantage": pick_adv(ps_a, ps_b, False), "Institutional Benchmark": "Revenue multiple < 3.0x preferred"},
        {"Category": "Valuation & Fundamentals", "Metric": "EV / EBITDA Multiple", name_a: f"{ev_ebitda_a:.1f}x" if ev_ebitda_a else "—", name_b: f"{ev_ebitda_b:.1f}x" if ev_ebitda_b else "—", "Spread / Delta": calc_delta(ev_ebitda_a, ev_ebitda_b, "mult"), "Advantage": pick_adv(ev_ebitda_a, ev_ebitda_b, False), "Institutional Benchmark": "Core operational value < 14x"},
        {"Category": "Valuation & Fundamentals", "Metric": "EV / Revenue Multiple", name_a: f"{ev_rev_a:.2f}x" if ev_rev_a else "—", name_b: f"{ev_rev_b:.2f}x" if ev_rev_b else "—", "Spread / Delta": calc_delta(ev_rev_a, ev_rev_b, "mult"), "Advantage": pick_adv(ev_rev_a, ev_rev_b, False), "Institutional Benchmark": "Capital structure adjusted sales"},
        {"Category": "Valuation & Fundamentals", "Metric": "Diluted EPS (TTM)", name_a: f"{eps_a:,.2f}" if eps_a is not None else "—", name_b: f"{eps_b:,.2f}" if eps_b is not None else "—", "Spread / Delta": calc_delta(eps_a, eps_b, "raw"), "Advantage": pick_adv(eps_a, eps_b, True), "Institutional Benchmark": "Earnings power per share"},
        {"Category": "Valuation & Fundamentals", "Metric": "Forward EPS (NTM)", name_a: f"{fwd_eps_a:,.2f}" if fwd_eps_a is not None else "—", name_b: f"{fwd_eps_b:,.2f}" if fwd_eps_b is not None else "—", "Spread / Delta": calc_delta(fwd_eps_a, fwd_eps_b, "raw"), "Advantage": pick_adv(fwd_eps_a, fwd_eps_b, True), "Institutional Benchmark": "Consensus forward outlook"},
        {"Category": "Valuation & Fundamentals", "Metric": "EPS YoY Growth (TTM)", name_a: f"{eps_growth_a:+.1f}%" if eps_growth_a is not None else "—", name_b: f"{eps_growth_b:+.1f}%" if eps_growth_b is not None else "—", "Spread / Delta": calc_delta(eps_growth_a, eps_growth_b, "pct"), "Advantage": pick_adv(eps_growth_a, eps_growth_b, True), "Institutional Benchmark": "Growth threshold >15.0%"},
        {"Category": "Valuation & Fundamentals", "Metric": "Book Value Per Share", name_a: f"{book_val_a:,.2f}" if book_val_a else "—", name_b: f"{book_val_b:,.2f}" if book_val_b else "—", "Spread / Delta": calc_delta(book_val_a, book_val_b, "raw"), "Advantage": pick_adv(book_val_a, book_val_b, True), "Institutional Benchmark": "Net asset accounting base"},
        {"Category": "Valuation & Fundamentals", "Metric": "Dividend Yield", name_a: f"{div_a:.2f}%" if div_a else "0.00%", name_b: f"{div_b:.2f}%" if div_b else "0.00%", "Spread / Delta": calc_delta(div_a or 0, div_b or 0, "pct"), "Advantage": pick_adv(div_a or 0, div_b or 0, True), "Institutional Benchmark": "Cash distribution yield"},

        # 3. Quality & Financial Margins
        {"Category": "Quality & Financials", "Metric": "Return on Equity (ROE)", name_a: f"{roe_a:.1f}%" if roe_a else "—", name_b: f"{roe_b:.1f}%" if roe_b else "—", "Spread / Delta": calc_delta(roe_a, roe_b, "pct"), "Advantage": pick_adv(roe_a, roe_b, True), "Institutional Benchmark": "Buffett standard > 15.0%"},
        {"Category": "Quality & Financials", "Metric": "Return on Assets (ROA)", name_a: f"{roa_a:.1f}%" if roa_a else "—", name_b: f"{roa_b:.1f}%" if roa_b else "—", "Spread / Delta": calc_delta(roa_a, roa_b, "pct"), "Advantage": pick_adv(roa_a, roa_b, True), "Institutional Benchmark": "Asset efficiency target > 6.0%"},
        {"Category": "Quality & Financials", "Metric": "Gross Profit Margin", name_a: f"{gross_margin_a:.1f}%" if gross_margin_a else "—", name_b: f"{gross_margin_b:.1f}%" if gross_margin_b else "—", "Spread / Delta": calc_delta(gross_margin_a, gross_margin_b, "pct"), "Advantage": pick_adv(gross_margin_a, gross_margin_b, True), "Institutional Benchmark": "Pricing power moat > 35.0%"},
        {"Category": "Quality & Financials", "Metric": "Operating Margin", name_a: f"{op_margin_a:.1f}%" if op_margin_a else "—", name_b: f"{op_margin_b:.1f}%" if op_margin_b else "—", "Spread / Delta": calc_delta(op_margin_a, op_margin_b, "pct"), "Advantage": pick_adv(op_margin_a, op_margin_b, True), "Institutional Benchmark": "Core operational leverage"},
        {"Category": "Quality & Financials", "Metric": "Net Profit Margin", name_a: f"{margin_a:.1f}%" if margin_a else "—", name_b: f"{margin_b:.1f}%" if margin_b else "—", "Spread / Delta": calc_delta(margin_a, margin_b, "pct"), "Advantage": pick_adv(margin_a, margin_b, True), "Institutional Benchmark": "Bottom line profitability > 12%"},
        {"Category": "Quality & Financials", "Metric": "Total Revenue (TTM)", name_a: _fmt_num(rev_a) if rev_a else "—", name_b: _fmt_num(rev_b) if rev_b else "—", "Spread / Delta": calc_delta(rev_a, rev_b, "curr"), "Advantage": pick_adv(rev_a, rev_b, True), "Institutional Benchmark": "Top-line commercial scale"},
        {"Category": "Quality & Financials", "Metric": "Revenue YoY Growth", name_a: f"{rev_growth_a:+.1f}%" if rev_growth_a is not None else "—", name_b: f"{rev_growth_b:+.1f}%" if rev_growth_b is not None else "—", "Spread / Delta": calc_delta(rev_growth_a, rev_growth_b, "pct"), "Advantage": pick_adv(rev_growth_a, rev_growth_b, True), "Institutional Benchmark": "Expansion pace > 10.0%"},
        {"Category": "Quality & Financials", "Metric": "EBITDA (TTM)", name_a: _fmt_num(ebitda_a) if ebitda_a else "—", name_b: _fmt_num(ebitda_b) if ebitda_b else "—", "Spread / Delta": calc_delta(ebitda_a, ebitda_b, "curr"), "Advantage": pick_adv(ebitda_a, ebitda_b, True), "Institutional Benchmark": "Operating cash generation"},
        {"Category": "Quality & Financials", "Metric": "Free Cash Flow (TTM)", name_a: _fmt_num(fcf_a) if fcf_a else "—", name_b: _fmt_num(fcf_b) if fcf_b else "—", "Spread / Delta": calc_delta(fcf_a, fcf_b, "curr"), "Advantage": pick_adv(fcf_a, fcf_b, True), "Institutional Benchmark": "Unencumbered cash generation"},
        {"Category": "Quality & Financials", "Metric": "Operating Cash Flow", name_a: _fmt_num(ocf_a) if ocf_a else "—", name_b: _fmt_num(ocf_b) if ocf_b else "—", "Spread / Delta": calc_delta(ocf_a, ocf_b, "curr"), "Advantage": pick_adv(ocf_a, ocf_b, True), "Institutional Benchmark": "Working capital efficiency"},

        # 4. Financial Health & Solvency
        {"Category": "Financial Health & Solvency", "Metric": "Total Debt to Equity", name_a: f"{de_a:.1f}%" if de_a else "—", name_b: f"{de_b:.1f}%" if de_b else "—", "Spread / Delta": calc_delta(de_a, de_b, "pct"), "Advantage": pick_adv(de_a, de_b, False), "Institutional Benchmark": "Conservative leverage < 80%"},
        {"Category": "Financial Health & Solvency", "Metric": "Current Ratio (Liquidity)", name_a: f"{curr_ratio_a:.2f}x" if curr_ratio_a else "—", name_b: f"{curr_ratio_b:.2f}x" if curr_ratio_b else "—", "Spread / Delta": calc_delta(curr_ratio_a, curr_ratio_b, "mult"), "Advantage": pick_adv(curr_ratio_a, curr_ratio_b, True), "Institutional Benchmark": "Solvent liquidity > 1.25x"},
        {"Category": "Financial Health & Solvency", "Metric": "Quick Ratio (Acid Test)", name_a: f"{quick_ratio_a:.2f}x" if quick_ratio_a else "—", name_b: f"{quick_ratio_b:.2f}x" if quick_ratio_b else "—", "Spread / Delta": calc_delta(quick_ratio_a, quick_ratio_b, "mult"), "Advantage": pick_adv(quick_ratio_a, quick_ratio_b, True), "Institutional Benchmark": "Immediate buffer > 1.0x"},
        {"Category": "Financial Health & Solvency", "Metric": "Total Cash Reserves", name_a: _fmt_num(total_cash_a) if total_cash_a else "—", name_b: _fmt_num(total_cash_b) if total_cash_b else "—", "Spread / Delta": calc_delta(total_cash_a, total_cash_b, "curr"), "Advantage": pick_adv(total_cash_a, total_cash_b, True), "Institutional Benchmark": "Balance sheet defense buffer"},
        {"Category": "Financial Health & Solvency", "Metric": "Total Outstanding Debt", name_a: _fmt_num(total_debt_a) if total_debt_a else "—", name_b: _fmt_num(total_debt_b) if total_debt_b else "—", "Spread / Delta": calc_delta(total_debt_a, total_debt_b, "curr"), "Advantage": pick_adv(total_debt_a, total_debt_b, False), "Institutional Benchmark": "Manageable debt burden"},
        {"Category": "Financial Health & Solvency", "Metric": "Primary Sector", name_a: str(sector_a), name_b: str(sector_b), "Spread / Delta": "—", "Advantage": "—", "Institutional Benchmark": "Sector industry peer"},
        {"Category": "Financial Health & Solvency", "Metric": "Primary Exchange", name_a: str(ex_a), name_b: str(ex_b), "Spread / Delta": "—", "Advantage": "—", "Institutional Benchmark": "Listing venue"},

        # 5. Risk, Volatility & Tail Risk
        {"Category": "Risk & Volatility", "Metric": "Annualized Volatility (252D)", name_a: f"{vol_a:.2f}%", name_b: f"{vol_b:.2f}%", "Spread / Delta": calc_delta(vol_a, vol_b, "pct"), "Advantage": pick_adv(vol_a, vol_b, False), "Institutional Benchmark": "Index baseline ~15–20%"},
        {"Category": "Risk & Volatility", "Metric": "Maximum Drawdown (MDD)", name_a: f"{dd_a:.2f}%", name_b: f"{dd_b:.2f}%", "Spread / Delta": calc_delta(dd_a, dd_b, "pct"), "Advantage": pick_adv(dd_a, dd_b, True), "Institutional Benchmark": "Peak-to-trough capital loss"},
        {"Category": "Risk & Volatility", "Metric": "Sharpe Ratio (Annualized)", name_a: f"{sharpe_a:.2f}", name_b: f"{sharpe_b:.2f}", "Spread / Delta": calc_delta(sharpe_a, sharpe_b, "raw"), "Advantage": pick_adv(sharpe_a, sharpe_b, True), "Institutional Benchmark": "Good > 1.0 / Excellent > 2.0"},
        {"Category": "Risk & Volatility", "Metric": "Sortino Ratio (Downside)", name_a: f"{sortino_a:.2f}", name_b: f"{sortino_b:.2f}", "Spread / Delta": calc_delta(sortino_a, sortino_b, "raw"), "Advantage": pick_adv(sortino_a, sortino_b, True), "Institutional Benchmark": "Penalizes only downside volatility"},
        {"Category": "Risk & Volatility", "Metric": "Calmar Ratio (CAGR / MDD)", name_a: f"{calmar_a:.2f}", name_b: f"{calmar_b:.2f}", "Spread / Delta": calc_delta(calmar_a, calmar_b, "raw"), "Advantage": pick_adv(calmar_a, calmar_b, True), "Institutional Benchmark": "Recovery efficiency > 1.0"},
        {"Category": "Risk & Volatility", "Metric": "Daily 95% Value at Risk (VaR)", name_a: f"{var_95_a:.2f}%", name_b: f"{var_95_b:.2f}%", "Spread / Delta": calc_delta(var_95_a, var_95_b, "pct"), "Advantage": pick_adv(var_95_a, var_95_b, False), "Institutional Benchmark": "Maximum 1-day loss in 95% cases"},
        {"Category": "Risk & Volatility", "Metric": "95% Conditional VaR (CVaR)", name_a: f"{cvar_95_a:.2f}%", name_b: f"{cvar_95_b:.2f}%", "Spread / Delta": calc_delta(cvar_95_a, cvar_95_b, "pct"), "Advantage": pick_adv(cvar_95_a, cvar_95_b, False), "Institutional Benchmark": "Average loss beyond 95% threshold"},
        {"Category": "Risk & Volatility", "Metric": "Downside Semi-Deviation", name_a: f"{semi_dev_a:.2f}%", name_b: f"{semi_dev_b:.2f}%", "Spread / Delta": calc_delta(semi_dev_a, semi_dev_b, "pct"), "Advantage": pick_adv(semi_dev_a, semi_dev_b, False), "Institutional Benchmark": "Annualized downside volatility"},
        {"Category": "Risk & Volatility", "Metric": "Daily Return Skewness", name_a: f"{skew_a:+.2f}", name_b: f"{skew_b:+.2f}", "Spread / Delta": calc_delta(skew_a, skew_b, "raw"), "Advantage": pick_adv(skew_a, skew_b, True), "Institutional Benchmark": "Positive skew = right-tail upside"},
        {"Category": "Risk & Volatility", "Metric": "Excess Kurtosis (Fat-Tail)", name_a: f"{kurt_a:.2f}", name_b: f"{kurt_b:.2f}", "Spread / Delta": calc_delta(kurt_a, kurt_b, "raw"), "Advantage": pick_adv(kurt_a, kurt_b, False), "Institutional Benchmark": "Normal distribution = 0.0"},
        {"Category": "Risk & Volatility", "Metric": "Beta (vs Benchmark)", name_a: f"{beta_a:.2f}", name_b: f"{beta_b:.2f}", "Spread / Delta": calc_delta(beta_a, beta_b, "raw"), "Advantage": "—", "Institutional Benchmark": "Market sensitivity (1.0 = equal)"},

        # 6. Ownership, Liquidity & Shares
        {"Category": "Liquidity & Volume", "Metric": "Latest 1-Day Volume", name_a: _fmt_num(vol_1d_a) if vol_1d_a else "—", name_b: _fmt_num(vol_1d_b) if vol_1d_b else "—", "Spread / Delta": calc_delta(vol_1d_a, vol_1d_b, "curr"), "Advantage": pick_adv(vol_1d_a, vol_1d_b, True), "Institutional Benchmark": "Immediate trading liquidity"},
        {"Category": "Liquidity & Volume", "Metric": "Relative Volume (1D RVOL)", name_a: f"{rvol_a:.2f}x" if rvol_a else "—", name_b: f"{rvol_b:.2f}x" if rvol_b else "—", "Spread / Delta": calc_delta(rvol_a, rvol_b, "mult"), "Advantage": pick_adv(rvol_a, rvol_b, True), "Institutional Benchmark": "Baseline = 1.0x / Surge > 1.5x"},
        {"Category": "Liquidity & Volume", "Metric": "Institutional Ownership", name_a: f"{inst_own_a:.1f}%" if inst_own_a else "—", name_b: f"{inst_own_b:.1f}%" if inst_own_b else "—", "Spread / Delta": calc_delta(inst_own_a, inst_own_b, "pct"), "Advantage": pick_adv(inst_own_a, inst_own_b, True), "Institutional Benchmark": "Smart money participation"},
        {"Category": "Liquidity & Volume", "Metric": "Insider Ownership", name_a: f"{insider_own_a:.1f}%" if insider_own_a else "—", name_b: f"{insider_own_b:.1f}%" if insider_own_b else "—", "Spread / Delta": calc_delta(insider_own_a, insider_own_b, "pct"), "Advantage": pick_adv(insider_own_a, insider_own_b, True), "Institutional Benchmark": "Management skin in the game"},
        {"Category": "Liquidity & Volume", "Metric": "Shares Outstanding", name_a: _fmt_num(shares_out_a) if shares_out_a else "—", name_b: _fmt_num(shares_out_b) if shares_out_b else "—", "Spread / Delta": calc_delta(shares_out_a, shares_out_b, "curr"), "Advantage": "—", "Institutional Benchmark": "Total equity float supply"},
        {"Category": "Liquidity & Volume", "Metric": "Short Ratio (Days to Cover)", name_a: f"{short_ratio_a:.1f}d" if short_ratio_a else "—", name_b: f"{short_ratio_b:.1f}d" if short_ratio_b else "—", "Spread / Delta": calc_delta(short_ratio_a, short_ratio_b, "raw"), "Advantage": pick_adv(short_ratio_a, short_ratio_b, False), "Institutional Benchmark": "Squeeze risk elevated > 5.0d"},
    ]

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Technical Screener & Indicator Analysis Engine
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def run_tradingview_technical_screener(
    market: str = "india",
    exchange: str = "NSE",
    preset: str = "All Stocks (Custom Technicals)",
    rsi_min: float = 0.0,
    rsi_max: float = 100.0,
    ma_filter: str = "Any",
    macd_filter: str = "Any",
    stoch_filter: str = "Any",
    bb_filter: str = "Any",
    min_rvol: float = 0.0,
    min_change: float = -100.0,
    max_change: float = 100.0,
    tech_rating: str = "Any",
    sort_by: str = "RSI",
    sort_asc: bool = False,
    limit: int = 50,
) -> tuple[int, pd.DataFrame]:
    """Execute multi-indicator technical query against TradingView Screener API."""
    try:
        q = Query().set_markets(market)
        cols = [
            "name",
            "description",
            "close",
            "change",
            "volume",
            "relative_volume_10d_calc",
            "RSI",
            "MACD.macd",
            "MACD.signal",
            "Stoch.K",
            "Stoch.D",
            "SMA20",
            "SMA50",
            "SMA200",
            "BB.upper",
            "BB.lower",
            "ATR",
            "Recommend.All",
            "Recommend.MA",
            "Recommend.Other",
            "exchange",
            "sector",
        ]
        q = q.select(*cols)

        tech_conditions = []

        # Exchange filter
        if exchange and exchange != "All" and "All" not in exchange:
            tech_conditions.append(col("exchange") == exchange)
        elif market == "india" and (exchange == "All" or "All" in exchange):
            tech_conditions.append(col("exchange").isin(["NSE", "BSE"]))

        # Technical Presets
        if preset == "🚀 Bullish Momentum Expansion (RSI 55-75 + SMA 50 Cross)":
            tech_conditions.extend([
                col("RSI") >= 55,
                col("RSI") <= 75,
                col("close") > col("SMA50"),
                col("change") > 0,
            ])
        elif preset == "📈 Golden Cross Breakout (50 SMA > 200 SMA)":
            tech_conditions.extend([col("SMA50") > col("SMA200"), col("close") > col("SMA50")])
        elif preset == "📉 Oversold Mean Reversion (RSI < 30 + Stoch < 25)":
            tech_conditions.extend([col("RSI") < 35, col("RSI") > 5, col("Stoch.K") < 25])
        elif preset == "⚡ Unusual Volume Surge & Breakout (RVOL > 1.5x + 1D Chg > 0)":
            tech_conditions.extend([col("relative_volume_10d_calc") >= 1.5, col("change") > 0.8])
        elif preset == "🎯 Bollinger Band Squeeze (Low Volatility Compression)":
            tech_conditions.extend([col("RSI") >= 45, col("RSI") <= 55])
        elif preset == "🔥 Strong Technical Consensus (Recommend.All >= 0.3)":
            tech_conditions.append(col("Recommend.All") >= 0.3)
        elif preset == "🌊 Multi-SMA Bullish Alignment (Price > 20 > 50 > 200)":
            tech_conditions.extend([
                col("close") > col("SMA20"),
                col("SMA20") > col("SMA50"),
                col("SMA50") > col("SMA200"),
            ])
        elif preset == "💎 Oversold Bounce at Lower Bollinger Band":
            tech_conditions.extend([col("close") <= col("BB.lower"), col("RSI") < 40])

        # Numerical input condition filters
        if rsi_min > 0:
            tech_conditions.append(col("RSI") >= rsi_min)
        if rsi_max < 100:
            tech_conditions.append(col("RSI") <= rsi_max)

        if min_rvol > 0:
            tech_conditions.append(col("relative_volume_10d_calc") >= min_rvol)

        if min_change > -100:
            tech_conditions.append(col("change") >= min_change)
        if max_change < 100:
            tech_conditions.append(col("change") <= max_change)

        # Moving Average condition
        if ma_filter == "Price > SMA 20":
            tech_conditions.append(col("close") > col("SMA20"))
        elif ma_filter == "Price > SMA 50":
            tech_conditions.append(col("close") > col("SMA50"))
        elif ma_filter == "Price > SMA 200":
            tech_conditions.append(col("close") > col("SMA200"))
        elif ma_filter == "Price > SMA 50 & 200":
            tech_conditions.extend([col("close") > col("SMA50"), col("close") > col("SMA200")])
        elif ma_filter == "Golden Cross (SMA 50 > SMA 200)":
            tech_conditions.append(col("SMA50") > col("SMA200"))
        elif ma_filter == "Multi-SMA Stack (Price > 20 > 50 > 200)":
            tech_conditions.extend([
                col("close") > col("SMA20"),
                col("SMA20") > col("SMA50"),
                col("SMA50") > col("SMA200"),
            ])
        elif ma_filter == "Price < SMA 200 (Discount / Bearish)":
            tech_conditions.append(col("close") < col("SMA200"))

        # MACD condition
        if macd_filter == "Bullish Crossover (MACD > Signal)":
            tech_conditions.append(col("MACD.macd") > col("MACD.signal"))
        elif macd_filter == "Bearish Crossover (MACD < Signal)":
            tech_conditions.append(col("MACD.macd") < col("MACD.signal"))
        elif macd_filter == "MACD Positive (MACD > 0)":
            tech_conditions.append(col("MACD.macd") > 0)
        elif macd_filter == "MACD Negative (MACD < 0)":
            tech_conditions.append(col("MACD.macd") < 0)

        # Stoch condition
        if stoch_filter == "Oversold (Stoch.K < 20)":
            tech_conditions.append(col("Stoch.K") < 20)
        elif stoch_filter == "Overbought (Stoch.K > 80)":
            tech_conditions.append(col("Stoch.K") > 80)
        elif stoch_filter == "Bullish Stoch (K > D)":
            tech_conditions.append(col("Stoch.K") > col("Stoch.D"))

        # Bollinger filter
        if bb_filter == "Price Touching/Below Lower Band":
            tech_conditions.append(col("close") <= col("BB.lower"))
        elif bb_filter == "Price Touching/Above Upper Band":
            tech_conditions.append(col("close") >= col("BB.upper"))

        # Overall technical rating filter
        if tech_rating == "Buy or Strong Buy (Rating > 0.1)":
            tech_conditions.append(col("Recommend.All") >= 0.1)
        elif tech_rating == "Strong Buy Only (Rating > 0.3)":
            tech_conditions.append(col("Recommend.All") >= 0.3)
        elif tech_rating == "Sell or Strong Sell (Rating < -0.1)":
            tech_conditions.append(col("Recommend.All") <= -0.1)

        if tech_conditions:
            q = q.where(*tech_conditions)

        # Order & limit
        q = q.order_by(sort_by, ascending=sort_asc).limit(limit)

        count, df = q.get_scanner_data()
        return count, df
    except Exception as e:
        st.warning(f"Technical Screener query warning: {e}")
        return 0, pd.DataFrame()


def compute_technical_features(df: pd.DataFrame) -> dict:
    """Compute comprehensive institutional technical features and indicators from OHLCV."""
    if df.empty or len(df) < 5:
        return {}

    c = df["Close"].dropna()
    h = df["High"].dropna() if "High" in df.columns else c
    l = df["Low"].dropna() if "Low" in df.columns else c
    o = df["Open"].dropna() if "Open" in df.columns else c
    v = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(1.0, index=c.index)

    last_close = float(c.iloc[-1])
    ret_1d = float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0) if len(c) > 1 else 0.0
    ret_1w = float((c.iloc[-1] / c.iloc[-6] - 1.0) * 100.0) if len(c) > 5 else ret_1d
    ret_1m = float((c.iloc[-1] / c.iloc[-22] - 1.0) * 100.0) if len(c) > 21 else ret_1w

    # 1. Moving Averages & Trend
    sma10 = c.rolling(10, min_periods=3).mean()
    sma20 = c.rolling(20, min_periods=5).mean()
    sma50 = c.rolling(50, min_periods=10).mean()
    sma100 = c.rolling(100, min_periods=15).mean()
    sma200 = c.rolling(200, min_periods=20).mean()
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    v_sma10 = float(sma10.iloc[-1]) if pd.notna(sma10.iloc[-1]) else last_close
    v_sma20 = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else last_close
    v_sma50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else last_close
    v_sma100 = float(sma100.iloc[-1]) if pd.notna(sma100.iloc[-1]) else last_close
    v_sma200 = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else last_close
    v_ema9 = float(ema9.iloc[-1]) if pd.notna(ema9.iloc[-1]) else last_close
    v_ema21 = float(ema21.iloc[-1]) if pd.notna(ema21.iloc[-1]) else last_close
    v_ema50 = float(ema50.iloc[-1]) if pd.notna(ema50.iloc[-1]) else last_close

    dist_sma20 = (last_close / v_sma20 - 1.0) * 100.0
    dist_ema21 = (last_close / v_ema21 - 1.0) * 100.0
    dist_sma50 = (last_close / v_sma50 - 1.0) * 100.0
    dist_sma100 = (last_close / v_sma100 - 1.0) * 100.0
    dist_sma200 = (last_close / v_sma200 - 1.0) * 100.0
    golden_cross = bool(v_sma50 > v_sma200)

    sma50_slope = float((v_sma50 / (sma50.iloc[-20] if len(sma50) > 20 and pd.notna(sma50.iloc[-20]) else v_sma50) - 1.0) * 100.0)
    sma20_50_spread = float((v_sma20 / v_sma50 - 1.0) * 100.0)

    if last_close > v_sma20 > v_sma50 > v_sma200:
        ma_stack = "🟢 Full Bullish Stack (Price > 20 > 50 > 200)"
        trend_regime = "🟢 Strong Bullish Stack"
        trend_score = 92.0
    elif last_close > v_sma50 and v_sma50 > v_sma200:
        ma_stack = "🟢 Bullish Alignment (Price & 50 > 200)"
        trend_regime = "🟢 Moderate Bullish"
        trend_score = 75.0
    elif last_close < v_sma20 < v_sma50 < v_sma200:
        ma_stack = "🔴 Full Bearish Stack (Price < 20 < 50 < 200)"
        trend_regime = "🔴 Strong Bearish Stack"
        trend_score = 15.0
    elif last_close < v_sma50 and v_sma50 < v_sma200:
        ma_stack = "🔴 Bearish Alignment (Price & 50 < 200)"
        trend_regime = "🔴 Moderate Bearish"
        trend_score = 28.0
    else:
        ma_stack = "🟡 Mixed / Consolidating Ribbon"
        trend_regime = "🟡 Neutral / Consolidating"
        trend_score = 50.0

    # 2. RSI 14 & Velocity
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=5).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    v_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0
    rsi_vel = float(rsi.iloc[-1] - rsi.iloc[-4]) if len(rsi) > 4 and pd.notna(rsi.iloc[-4]) else 0.0

    # 3. MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    v_macd = float(macd_line.iloc[-1]) if pd.notna(macd_line.iloc[-1]) else 0.0
    v_signal = float(macd_signal.iloc[-1]) if pd.notna(macd_signal.iloc[-1]) else 0.0
    v_hist = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0.0

    if v_macd > v_signal and v_hist > 0:
        macd_state = "🟢 Bullish Expanding"
    elif v_macd > v_signal and v_hist <= 0:
        macd_state = "🟡 Bullish Decelerating"
    elif v_macd <= v_signal and v_hist < 0:
        macd_state = "🔴 Bearish Expanding"
    else:
        macd_state = "🟡 Bearish Decelerating"

    # 4. Stochastics (14, 3)
    l14 = l.rolling(14, min_periods=5).min()
    h14 = h.rolling(14, min_periods=5).max()
    stoch_k = 100.0 * (c - l14) / (h14 - l14 + 1e-9)
    stoch_d = stoch_k.rolling(3, min_periods=2).mean()
    v_stoch_k = float(stoch_k.iloc[-1]) if pd.notna(stoch_k.iloc[-1]) else 50.0
    v_stoch_d = float(stoch_d.iloc[-1]) if pd.notna(stoch_d.iloc[-1]) else 50.0

    if v_stoch_k > 80:
        stoch_state = "⚠️ Overbought (>80)"
    elif v_stoch_k < 20:
        stoch_state = "🎯 Oversold (<20)"
    elif v_stoch_k > v_stoch_d:
        stoch_state = "🟢 Bullish Crossover (K > D)"
    else:
        stoch_state = "🔴 Bearish Crossover (K < D)"

    # 5. Momentum Oscillators: ROC, Williams %R, CCI
    roc10 = float((c.iloc[-1] / c.iloc[-11] - 1.0) * 100.0) if len(c) > 10 else ret_1w
    roc21 = float((c.iloc[-1] / c.iloc[-22] - 1.0) * 100.0) if len(c) > 21 else ret_1m
    wr14 = float(-100.0 * (h14.iloc[-1] - last_close) / (h14.iloc[-1] - l14.iloc[-1] + 1e-9)) if len(h14) > 0 else -50.0

    tp = (h + l + c) / 3.0
    sma_tp = tp.rolling(20, min_periods=5).mean()
    mad = (tp - sma_tp).abs().rolling(20, min_periods=5).mean()
    cci = (tp - sma_tp) / (0.015 * mad + 1e-9)
    v_cci = float(cci.iloc[-1]) if pd.notna(cci.iloc[-1]) else 0.0

    # 6. ATR & ADX (14) Trend Strength
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    v_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 1.0
    v_atrp = (v_atr / last_close) * 100.0

    up_m = h.diff()
    down_m = -l.diff()
    p_dm = np.where((up_m > down_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((down_m > up_m) & (down_m > 0), down_m, 0.0)
    p_di = 100.0 * (pd.Series(p_dm, index=c.index).rolling(14, min_periods=5).mean() / (atr + 1e-9))
    m_di = 100.0 * (pd.Series(m_dm, index=c.index).rolling(14, min_periods=5).mean() / (atr + 1e-9))
    dx = 100.0 * (p_di - m_di).abs() / (p_di + m_di + 1e-9)
    adx = dx.rolling(14, min_periods=5).mean()
    v_adx = float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 20.0
    v_pdi = float(p_di.iloc[-1]) if pd.notna(p_di.iloc[-1]) else 20.0
    v_mdi = float(m_di.iloc[-1]) if pd.notna(m_di.iloc[-1]) else 20.0

    # 7. Volatility & Realized Volatility
    realized_vol20 = float(c.pct_change().rolling(20, min_periods=5).std().iloc[-1] * np.sqrt(252) * 100.0) if len(c) > 5 else 20.0

    # Bollinger Bands (20, 2.0)
    bb_mid = sma20
    bb_std = c.rolling(20, min_periods=5).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9) * 100.0
    bb_pct_b = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)
    v_bbu = float(bb_upper.iloc[-1]) if pd.notna(bb_upper.iloc[-1]) else last_close
    v_bbl = float(bb_lower.iloc[-1]) if pd.notna(bb_lower.iloc[-1]) else last_close
    v_bbw = float(bb_width.iloc[-1]) if pd.notna(bb_width.iloc[-1]) else 5.0
    v_pct_b = float(bb_pct_b.iloc[-1]) if pd.notna(bb_pct_b.iloc[-1]) else 0.5
    squeeze_state = "⚠️ Coiling Squeeze (<6%)" if v_bbw < 6.0 else "🌊 Volatility Expansion"

    # Price Channel & Daily Range
    h20 = float(h.rolling(20, min_periods=5).max().iloc[-1])
    l20 = float(l.rolling(20, min_periods=5).min().iloc[-1])
    range_20d = float((h20 - l20) / (l20 + 1e-9) * 100.0)
    adr_pct = float(((h - l) / c).rolling(20, min_periods=5).mean().iloc[-1] * 100.0)

    # 8. Volume Flow & Institutional Pressure
    vol_last = float(v.iloc[-1]) if len(v) > 0 else 0.0
    vol_sma20 = float(v.rolling(20, min_periods=5).mean().iloc[-1]) if len(v) > 5 else vol_last
    rvol = vol_last / (vol_sma20 + 1e-9)

    # MFI 14 (Money Flow Index)
    raw_mf = tp * v
    pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(14, min_periods=5).sum()
    neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(14, min_periods=5).sum()
    mfi = 100.0 - (100.0 / (1.0 + (pos_mf / (neg_mf + 1e-9))))
    v_mfi = float(mfi.iloc[-1]) if pd.notna(mfi.iloc[-1]) else 50.0

    # Chaikin Money Flow (CMF 20)
    mfv = ((c - l) - (h - c)) / (h - l + 1e-9) * v
    cmf = mfv.rolling(20, min_periods=5).sum() / (v.rolling(20, min_periods=5).sum() + 1e-9)
    v_cmf = float(cmf.iloc[-1]) if pd.notna(cmf.iloc[-1]) else 0.0

    # On-Balance Volume (OBV)
    direction = np.sign(c.diff().fillna(0))
    obv = (direction * v).cumsum()
    obv_flow = "🟢 Accumulation Flow (+)" if (len(obv) > 10 and obv.diff(10).iloc[-1] > 0) else "🔴 Distribution Flow (-)"

    # Volume Breadth
    up_v = v.where(c > c.shift(), 0).rolling(20, min_periods=5).sum()
    dn_v = v.where(c < c.shift(), 0).rolling(20, min_periods=5).sum()
    vol_breadth = float(up_v.iloc[-1] / (dn_v.iloc[-1] + 1e-9)) if len(up_v) > 0 else 1.0

    # 9. Regime & Composite Scoring
    mom_score = float(np.clip(v_rsi * 0.45 + (50.0 if v_macd > v_signal else 20.0) * 0.25 + min(v_adx, 40.0) * 0.5 + (roc10 * 0.5), 10.0, 95.0))
    volat_score = float(np.clip(100.0 - (v_atrp * 12.0), 15.0, 95.0))
    flow_score = float(np.clip(v_mfi * 0.4 + (50.0 + v_cmf * 100.0) * 0.3 + min(rvol * 20.0, 30.0), 10.0, 95.0))
    reversion_score = float(np.clip((100.0 - v_rsi) * 0.5 + (100.0 - v_stoch_k) * 0.5, 10.0, 95.0))
    overall_score = trend_score * 0.35 + mom_score * 0.25 + flow_score * 0.25 + volat_score * 0.15

    if overall_score >= 75:
        overall_signal = "🚀 Strong Buy"
    elif overall_score >= 60:
        overall_signal = "🟢 Buy"
    elif overall_score <= 30:
        overall_signal = "🚨 Strong Sell"
    elif overall_score <= 45:
        overall_signal = "🔴 Sell"
    else:
        overall_signal = "⚖️ Neutral"

    return {
        "last_close": last_close,
        "ret_1d": ret_1d,
        "ret_1w": ret_1w,
        "ret_1m": ret_1m,
        "sma10": v_sma10,
        "sma20": v_sma20,
        "sma50": v_sma50,
        "sma100": v_sma100,
        "sma200": v_sma200,
        "ema9": v_ema9,
        "ema21": v_ema21,
        "ema50": v_ema50,
        "dist_sma20": dist_sma20,
        "dist_ema21": dist_ema21,
        "dist_sma50": dist_sma50,
        "dist_sma100": dist_sma100,
        "dist_sma200": dist_sma200,
        "golden_cross": golden_cross,
        "sma50_slope": sma50_slope,
        "sma20_50_spread": sma20_50_spread,
        "ma_stack": ma_stack,
        "trend_regime": trend_regime,
        "trend_score": trend_score,
        "rsi": v_rsi,
        "rsi_vel": rsi_vel,
        "macd": v_macd,
        "macd_sig": v_signal,
        "macd_hist": v_hist,
        "macd_state": macd_state,
        "stoch_k": v_stoch_k,
        "stoch_d": v_stoch_d,
        "stoch_state": stoch_state,
        "roc10": roc10,
        "roc21": roc21,
        "wr14": wr14,
        "cci": v_cci,
        "adx": v_adx,
        "p_di": v_pdi,
        "m_di": v_mdi,
        "mom_score": mom_score,
        "atr": v_atr,
        "atr_pct": v_atrp,
        "realized_vol20": realized_vol20,
        "bb_upper": v_bbu,
        "bb_lower": v_bbl,
        "bb_width": v_bbw,
        "bb_pct_b": v_pct_b,
        "squeeze_state": squeeze_state,
        "range_20d": range_20d,
        "adr_pct": adr_pct,
        "volat_score": volat_score,
        "reversion_score": reversion_score,
        "vol_1d": vol_last,
        "vol_sma20": vol_sma20,
        "rvol": rvol,
        "mfi": v_mfi,
        "cmf": v_cmf,
        "obv_flow": obv_flow,
        "vol_breadth": vol_breadth,
        "flow_score": flow_score,
        "overall_score": overall_score,
        "overall_signal": overall_signal,
        # Plotting series
        "series_c": c,
        "series_h": h,
        "series_l": l,
        "series_o": o,
        "series_v": v,
        "series_sma20": sma20,
        "series_sma50": sma50,
        "series_sma200": sma200,
        "series_bbu": bb_upper,
        "series_bbl": bb_lower,
        "series_bb_width": bb_width,
        "series_rsi": rsi,
        "series_macd": macd_line,
        "series_macd_sig": macd_signal,
        "series_macd_hist": macd_hist,
        "series_stoch_k": stoch_k,
        "series_stoch_d": stoch_d,
        "series_mfi": mfi,
        "series_atr": atr,
    }


def build_technical_comparison_matrix(ta: dict, tb: dict, name_a: str, name_b: str) -> pd.DataFrame:
    """Compile comprehensive head-to-head comparison metrics for technical indicators with spread and context."""
    def pick_adv(val_a, val_b, higher_is_better=True):
        if val_a is None or val_b is None:
            return "—"
        try:
            fa = float(val_a)
            fb = float(val_b)
        except (ValueError, TypeError):
            return "—"
        if np.isnan(fa) or np.isnan(fb):
            return "—"
        if abs(fa - fb) < 1e-6:
            return "Tied"
        if higher_is_better:
            return f"🟢 {name_a}" if fa > fb else f"🔵 {name_b}"
        else:
            return f"🟢 {name_a}" if fa < fb else f"🔵 {name_b}"

    def calc_delta(val_a, val_b, kind="num"):
        if val_a is None or val_b is None:
            return "—"
        try:
            fa = float(val_a)
            fb = float(val_b)
        except (ValueError, TypeError):
            return "—"
        if np.isnan(fa) or np.isnan(fb):
            return "—"
        d = fa - fb
        if kind == "pct":
            return f"{d:+.2f}%"
        elif kind == "mult":
            return f"{d:+.2f}x"
        elif kind == "curr":
            return f"{'+' if d >= 0 else ''}{_fmt_num(d)}"
        elif kind == "pts":
            return f"{d:+.1f} pts"
        return f"{d:+.2f}"

    rows = [
        # 1. Trend & Moving Averages
        {"Category": "Trend & Moving Averages", "Metric": "Primary Trend Regime", name_a: ta.get("trend_regime", "—"), name_b: tb.get("trend_regime", "—"), "Spread / Delta": "—", "Advantage": pick_adv(ta.get("trend_score"), tb.get("trend_score"), True), "Signal / Context": "Regime based on moving average stacks"},
        {"Category": "Trend & Moving Averages", "Metric": "Moving Average Ribbon Stack", name_a: ta.get("ma_stack", "—"), name_b: tb.get("ma_stack", "—"), "Spread / Delta": "—", "Advantage": pick_adv(ta.get("trend_score"), tb.get("trend_score"), True), "Signal / Context": "Multi-timeframe ribbon orientation"},
        {"Category": "Trend & Moving Averages", "Metric": "Distance from 20-Day SMA (%)", name_a: f"{ta.get('dist_sma20', 0):+.2f}%", name_b: f"{tb.get('dist_sma20', 0):+.2f}%", "Spread / Delta": calc_delta(ta.get('dist_sma20'), tb.get('dist_sma20'), "pct"), "Advantage": pick_adv(ta.get("dist_sma20"), tb.get("dist_sma20"), True), "Signal / Context": "Short-term mean extension (>5% overextended)"},
        {"Category": "Trend & Moving Averages", "Metric": "Distance from 21-Day EMA (%)", name_a: f"{ta.get('dist_ema21', 0):+.2f}%", name_b: f"{tb.get('dist_ema21', 0):+.2f}%", "Spread / Delta": calc_delta(ta.get('dist_ema21'), tb.get('dist_ema21'), "pct"), "Advantage": pick_adv(ta.get("dist_ema21"), tb.get("dist_ema21"), True), "Signal / Context": "Exponential short-term pullbacks"},
        {"Category": "Trend & Moving Averages", "Metric": "Distance from 50-Day SMA (%)", name_a: f"{ta.get('dist_sma50', 0):+.2f}%", name_b: f"{tb.get('dist_sma50', 0):+.2f}%", "Spread / Delta": calc_delta(ta.get('dist_sma50'), tb.get('dist_sma50'), "pct"), "Advantage": pick_adv(ta.get("dist_sma50"), tb.get("dist_sma50"), True), "Signal / Context": "Intermediate trend anchor (>0% is bullish)"},
        {"Category": "Trend & Moving Averages", "Metric": "Distance from 100-Day SMA (%)", name_a: f"{ta.get('dist_sma100', 0):+.2f}%", name_b: f"{tb.get('dist_sma100', 0):+.2f}%", "Spread / Delta": calc_delta(ta.get('dist_sma100'), tb.get('dist_sma100'), "pct"), "Advantage": pick_adv(ta.get("dist_sma100"), tb.get("dist_sma100"), True), "Signal / Context": "Medium-term institutional trend support"},
        {"Category": "Trend & Moving Averages", "Metric": "Distance from 200-Day SMA (%)", name_a: f"{ta.get('dist_sma200', 0):+.2f}%", name_b: f"{tb.get('dist_sma200', 0):+.2f}%", "Spread / Delta": calc_delta(ta.get('dist_sma200'), tb.get('dist_sma200'), "pct"), "Advantage": pick_adv(ta.get("dist_sma200"), tb.get("dist_sma200"), True), "Signal / Context": "Macro cycle threshold (>0% secular bull)"},
        {"Category": "Trend & Moving Averages", "Metric": "50-Day SMA Velocity Slope", name_a: f"{ta.get('sma50_slope', 0):+.2f}%", name_b: f"{tb.get('sma50_slope', 0):+.2f}%", "Spread / Delta": calc_delta(ta.get('sma50_slope'), tb.get('sma50_slope'), "pct"), "Advantage": pick_adv(ta.get("sma50_slope"), tb.get("sma50_slope"), True), "Signal / Context": "20-bar rate of change in 50 SMA"},
        {"Category": "Trend & Moving Averages", "Metric": "20 SMA vs 50 SMA Spread (%)", name_a: f"{ta.get('sma20_50_spread', 0):+.2f}%", name_b: f"{tb.get('sma20_50_spread', 0):+.2f}%", "Spread / Delta": calc_delta(ta.get('sma20_50_spread'), tb.get('sma20_50_spread'), "pct"), "Advantage": pick_adv(ta.get("sma20_50_spread"), tb.get("sma20_50_spread"), True), "Signal / Context": "MA convergence/divergence velocity"},
        {"Category": "Trend & Moving Averages", "Metric": "Golden Cross (50 > 200 SMA)", name_a: "🟢 Active (Bullish)" if ta.get("golden_cross") else "🔴 Inactive (Bearish)", name_b: "🟢 Active (Bullish)" if tb.get("golden_cross") else "🔴 Inactive (Bearish)", "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "Long-term secular cycle confirmation"},
        {"Category": "Trend & Moving Averages", "Metric": "20-Day SMA Value", name_a: f"{ta.get('sma20', 0):,.2f}", name_b: f"{tb.get('sma20', 0):,.2f}", "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "Short-term benchmark price"},
        {"Category": "Trend & Moving Averages", "Metric": "50-Day SMA Value", name_a: f"{ta.get('sma50', 0):,.2f}", name_b: f"{tb.get('sma50', 0):,.2f}", "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "Intermediate trend benchmark price"},
        {"Category": "Trend & Moving Averages", "Metric": "100-Day SMA Value", name_a: f"{ta.get('sma100', 0):,.2f}", name_b: f"{tb.get('sma100', 0):,.2f}", "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "Medium-term trend benchmark price"},
        {"Category": "Trend & Moving Averages", "Metric": "200-Day SMA Value", name_a: f"{ta.get('sma200', 0):,.2f}", name_b: f"{tb.get('sma200', 0):,.2f}", "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "Secular macro benchmark price"},
        {"Category": "Trend & Moving Averages", "Metric": "21-Day EMA Value", name_a: f"{ta.get('ema21', 0):,.2f}", name_b: f"{tb.get('ema21', 0):,.2f}", "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "Exponential dynamic support level"},

        # 2. Momentum & Oscillators
        {"Category": "Momentum & Oscillators", "Metric": "Relative Strength Index (RSI 14)", name_a: f"{ta.get('rsi', 50):.1f}", name_b: f"{tb.get('rsi', 50):.1f}", "Spread / Delta": calc_delta(ta.get('rsi'), tb.get('rsi'), "num"), "Advantage": pick_adv(ta.get("rsi"), tb.get("rsi"), True), "Signal / Context": "Values >70 overbought, <30 oversold"},
        {"Category": "Momentum & Oscillators", "Metric": "RSI 3-Day Velocity (Δ RSI)", name_a: f"{ta.get('rsi_vel', 0):+.1f}", name_b: f"{tb.get('rsi_vel', 0):+.1f}", "Spread / Delta": calc_delta(ta.get('rsi_vel'), tb.get('rsi_vel'), "num"), "Advantage": pick_adv(ta.get("rsi_vel"), tb.get("rsi_vel"), True), "Signal / Context": "Speed of momentum acceleration"},
        {"Category": "Momentum & Oscillators", "Metric": "MACD Line (12, 26)", name_a: f"{ta.get('macd', 0):+.2f}", name_b: f"{tb.get('macd', 0):+.2f}", "Spread / Delta": calc_delta(ta.get('macd'), tb.get('macd'), "num"), "Advantage": pick_adv(ta.get("macd"), tb.get("macd"), True), "Signal / Context": "Trend-following momentum difference"},
        {"Category": "Momentum & Oscillators", "Metric": "MACD Signal Line (9)", name_a: f"{ta.get('macd_sig', 0):+.2f}", name_b: f"{tb.get('macd_sig', 0):+.2f}", "Spread / Delta": calc_delta(ta.get('macd_sig'), tb.get('macd_sig'), "num"), "Advantage": "—", "Signal / Context": "Exponential trigger signal"},
        {"Category": "Momentum & Oscillators", "Metric": "MACD Histogram (Velocity)", name_a: f"{ta.get('macd_hist', 0):+.2f}", name_b: f"{tb.get('macd_hist', 0):+.2f}", "Spread / Delta": calc_delta(ta.get('macd_hist'), tb.get('macd_hist'), "num"), "Advantage": pick_adv(ta.get("macd_hist"), tb.get("macd_hist"), True), "Signal / Context": "Histogram bar >0 confirms impulse"},
        {"Category": "Momentum & Oscillators", "Metric": "MACD Dynamics State", name_a: ta.get("macd_state", "—"), name_b: tb.get("macd_state", "—"), "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "Impulse divergence vs convergence"},
        {"Category": "Momentum & Oscillators", "Metric": "Stochastic %K (14)", name_a: f"{ta.get('stoch_k', 50):.1f}", name_b: f"{tb.get('stoch_k', 50):.1f}", "Spread / Delta": calc_delta(ta.get('stoch_k'), tb.get('stoch_k'), "num"), "Advantage": pick_adv(ta.get("stoch_k"), tb.get("stoch_k"), True), "Signal / Context": "Fast price location in 14-day range"},
        {"Category": "Momentum & Oscillators", "Metric": "Stochastic %D (3)", name_a: f"{ta.get('stoch_d', 50):.1f}", name_b: f"{tb.get('stoch_d', 50):.1f}", "Spread / Delta": calc_delta(ta.get('stoch_d'), tb.get('stoch_d'), "num"), "Advantage": pick_adv(ta.get("stoch_d"), tb.get("stoch_d"), True), "Signal / Context": "Smoothed stochastic trigger line"},
        {"Category": "Momentum & Oscillators", "Metric": "Stochastic Status", name_a: ta.get("stoch_state", "—"), name_b: tb.get("stoch_state", "—"), "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "Crossover and extreme band readings"},
        {"Category": "Momentum & Oscillators", "Metric": "Rate of Change (10D ROC %)", name_a: f"{ta.get('roc10', 0):+.2f}%", name_b: f"{tb.get('roc10', 0):+.2f}%", "Spread / Delta": calc_delta(ta.get('roc10'), tb.get('roc10'), "pct"), "Advantage": pick_adv(ta.get("roc10"), tb.get("roc10"), True), "Signal / Context": "10-session momentum velocity"},
        {"Category": "Momentum & Oscillators", "Metric": "Rate of Change (21D ROC %)", name_a: f"{ta.get('roc21', 0):+.2f}%", name_b: f"{tb.get('roc21', 0):+.2f}%", "Spread / Delta": calc_delta(ta.get('roc21'), tb.get('roc21'), "pct"), "Advantage": pick_adv(ta.get("roc21"), tb.get("roc21"), True), "Signal / Context": "Monthly price rate of change"},
        {"Category": "Momentum & Oscillators", "Metric": "Williams %R (14)", name_a: f"{ta.get('wr14', -50):.1f}", name_b: f"{tb.get('wr14', -50):.1f}", "Spread / Delta": calc_delta(ta.get('wr14'), tb.get('wr14'), "num"), "Advantage": pick_adv(ta.get("wr14"), tb.get("wr14"), True), "Signal / Context": "Range bound indicator (-20 to -80)"},
        {"Category": "Momentum & Oscillators", "Metric": "Commodity Channel Index (CCI 20)", name_a: f"{ta.get('cci', 0):+.1f}", name_b: f"{tb.get('cci', 0):+.1f}", "Spread / Delta": calc_delta(ta.get('cci'), tb.get('cci'), "num"), "Advantage": pick_adv(ta.get("cci"), tb.get("cci"), True), "Signal / Context": ">+100 bullish surge / <-100 oversold"},
        {"Category": "Momentum & Oscillators", "Metric": "Average Directional Index (ADX 14)", name_a: f"{ta.get('adx', 20):.1f}", name_b: f"{tb.get('adx', 20):.1f}", "Spread / Delta": calc_delta(ta.get('adx'), tb.get('adx'), "num"), "Advantage": pick_adv(ta.get("adx"), tb.get("adx"), True), "Signal / Context": "Trend strength: >25 strong / <20 chop"},
        {"Category": "Momentum & Oscillators", "Metric": "Directional Dominance (+DI vs -DI)", name_a: f"{ta.get('p_di', 20):.1f} vs {ta.get('m_di', 20):.1f}", name_b: f"{tb.get('p_di', 20):.1f} vs {tb.get('m_di', 20):.1f}", "Spread / Delta": "—", "Advantage": pick_adv(ta.get('p_di', 0) - ta.get('m_di', 0), tb.get('p_di', 0) - tb.get('m_di', 0), True), "Signal / Context": "+DI > -DI confirms buyer control"},

        # 3. Volatility & Statistical Bands
        {"Category": "Volatility & Bands", "Metric": "Normalized ATR (NATR % of Price)", name_a: f"{ta.get('atr_pct', 0):.2f}%", name_b: f"{tb.get('atr_pct', 0):.2f}%", "Spread / Delta": calc_delta(ta.get('atr_pct'), tb.get('atr_pct'), "pct"), "Advantage": pick_adv(ta.get("atr_pct"), tb.get("atr_pct"), False), "Signal / Context": "Normalized true range price volatility"},
        {"Category": "Volatility & Bands", "Metric": "Average True Range (ATR 14)", name_a: f"{ta.get('atr', 0):,.2f}", name_b: f"{tb.get('atr', 0):,.2f}", "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "Absolute 14-day candle spread"},
        {"Category": "Volatility & Bands", "Metric": "20-Day Historical Realized Vol", name_a: f"{ta.get('realized_vol20', 0):.2f}%", name_b: f"{tb.get('realized_vol20', 0):.2f}%", "Spread / Delta": calc_delta(ta.get('realized_vol20'), tb.get('realized_vol20'), "pct"), "Advantage": pick_adv(ta.get("realized_vol20"), tb.get("realized_vol20"), False), "Signal / Context": "Annualized realized standard deviation"},
        {"Category": "Volatility & Bands", "Metric": "Bollinger Upper Band (20, 2σ)", name_a: f"{ta.get('bb_upper', 0):,.2f}", name_b: f"{tb.get('bb_upper', 0):,.2f}", "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "+2 sigma resistance ceiling"},
        {"Category": "Volatility & Bands", "Metric": "Bollinger Lower Band (20, 2σ)", name_a: f"{ta.get('bb_lower', 0):,.2f}", name_b: f"{tb.get('bb_lower', 0):,.2f}", "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "-2 sigma support floor"},
        {"Category": "Volatility & Bands", "Metric": "Bollinger Bandwidth %", name_a: f"{ta.get('bb_width', 0):.2f}%", name_b: f"{tb.get('bb_width', 0):.2f}%", "Spread / Delta": calc_delta(ta.get('bb_width'), tb.get('bb_width'), "pct"), "Advantage": "—", "Signal / Context": "Band width relative to middle SMA"},
        {"Category": "Volatility & Bands", "Metric": "Bollinger %B Channel Location", name_a: f"{ta.get('bb_pct_b', 0.5):.2f}", name_b: f"{tb.get('bb_pct_b', 0.5):.2f}", "Spread / Delta": calc_delta(ta.get('bb_pct_b'), tb.get('bb_pct_b'), "num"), "Advantage": pick_adv(ta.get("bb_pct_b"), tb.get("bb_pct_b"), True), "Signal / Context": "0.0 = Lower band / 1.0 = Upper band"},
        {"Category": "Volatility & Bands", "Metric": "Volatility Squeeze State", name_a: ta.get("squeeze_state", "—"), name_b: tb.get("squeeze_state", "—"), "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "<6% compression precedes breakout"},
        {"Category": "Volatility & Bands", "Metric": "20-Day High/Low Range (%)", name_a: f"{ta.get('range_20d', 0):.2f}%", name_b: f"{tb.get('range_20d', 0):.2f}%", "Spread / Delta": calc_delta(ta.get('range_20d'), tb.get('range_20d'), "pct"), "Advantage": "—", "Signal / Context": "Total channel dispersion over 20 bars"},
        {"Category": "Volatility & Bands", "Metric": "Average Daily Range (ADR %)", name_a: f"{ta.get('adr_pct', 0):.2f}%", name_b: f"{tb.get('adr_pct', 0):.2f}%", "Spread / Delta": calc_delta(ta.get('adr_pct'), tb.get('adr_pct'), "pct"), "Advantage": "—", "Signal / Context": "Typical intraday range % of price"},

        # 4. Volume Flow & Institutional Pressure
        {"Category": "Volume Flow & Pressure", "Metric": "Latest 1-Day Trading Volume", name_a: _fmt_num(ta.get("vol_1d", 0)), name_b: _fmt_num(tb.get("vol_1d", 0)), "Spread / Delta": calc_delta(ta.get('vol_1d'), tb.get('vol_1d'), "curr"), "Advantage": pick_adv(ta.get("vol_1d"), tb.get("vol_1d"), True), "Signal / Context": "Raw trading liquidity volume"},
        {"Category": "Volume Flow & Pressure", "Metric": "20-Day Average Volume", name_a: _fmt_num(ta.get("vol_sma20", 0)), name_b: _fmt_num(tb.get("vol_sma20", 0)), "Spread / Delta": calc_delta(ta.get('vol_sma20'), tb.get('vol_sma20'), "curr"), "Advantage": pick_adv(ta.get("vol_sma20"), tb.get("vol_sma20"), True), "Signal / Context": "Average session participation benchmark"},
        {"Category": "Volume Flow & Pressure", "Metric": "Relative Volume (1D RVOL)", name_a: f"{ta.get('rvol', 1):.2f}x", name_b: f"{tb.get('rvol', 1):.2f}x", "Spread / Delta": calc_delta(ta.get('rvol'), tb.get('rvol'), "mult"), "Advantage": pick_adv(ta.get("rvol"), tb.get("rvol"), True), "Signal / Context": ">1.5x indicates institutional participation"},
        {"Category": "Volume Flow & Pressure", "Metric": "Money Flow Index (MFI 14)", name_a: f"{ta.get('mfi', 50):.1f}", name_b: f"{tb.get('mfi', 50):.1f}", "Spread / Delta": calc_delta(ta.get('mfi'), tb.get('mfi'), "num"), "Advantage": pick_adv(ta.get("mfi"), tb.get("mfi"), True), "Signal / Context": "Volume-weighted RSI (>80 overbought)"},
        {"Category": "Volume Flow & Pressure", "Metric": "Chaikin Money Flow (CMF 20)", name_a: f"{ta.get('cmf', 0):+.3f}", name_b: f"{tb.get('cmf', 0):+.3f}", "Spread / Delta": calc_delta(ta.get('cmf'), tb.get('cmf'), "num"), "Advantage": pick_adv(ta.get("cmf"), tb.get("cmf"), True), "Signal / Context": ">+0.05 indicates net accumulation"},
        {"Category": "Volume Flow & Pressure", "Metric": "On-Balance Volume (OBV) Flow", name_a: ta.get("obv_flow", "—"), name_b: tb.get("obv_flow", "—"), "Spread / Delta": "—", "Advantage": "—", "Signal / Context": "Cumulative volume pressure direction"},
        {"Category": "Volume Flow & Pressure", "Metric": "Up / Down Volume Breadth Ratio", name_a: f"{ta.get('vol_breadth', 1):.2f}x", name_b: f"{tb.get('vol_breadth', 1):.2f}x", "Spread / Delta": calc_delta(ta.get('vol_breadth'), tb.get('vol_breadth'), "mult"), "Advantage": pick_adv(ta.get("vol_breadth"), tb.get("vol_breadth"), True), "Signal / Context": ">1.0 denotes buying volume dominance"},

        # 5. Composite Quantitative Scores
        {"Category": "Composite Technical Scores", "Metric": "Overall Technical Verdict", name_a: ta.get("overall_signal", "—"), name_b: tb.get("overall_signal", "—"), "Spread / Delta": "—", "Advantage": pick_adv(ta.get("overall_score"), tb.get("overall_score"), True), "Signal / Context": "Multi-pillar quantitative synthesis"},
        {"Category": "Composite Technical Scores", "Metric": "Composite Quantitative Score", name_a: f"{ta.get('overall_score', 50):.1f}/100", name_b: f"{tb.get('overall_score', 50):.1f}/100", "Spread / Delta": calc_delta(ta.get('overall_score'), tb.get('overall_score'), "pts"), "Advantage": pick_adv(ta.get("overall_score"), tb.get("overall_score"), True), "Signal / Context": "Weighted combination of all pillars"},
        {"Category": "Composite Technical Scores", "Metric": "Trend Strength Pillar Score", name_a: f"{ta.get('trend_score', 50):.1f}/100", name_b: f"{tb.get('trend_score', 50):.1f}/100", "Spread / Delta": calc_delta(ta.get('trend_score'), tb.get('trend_score'), "pts"), "Advantage": pick_adv(ta.get("trend_score"), tb.get("trend_score"), True), "Signal / Context": "Moving average alignment & slope"},
        {"Category": "Composite Technical Scores", "Metric": "Momentum Velocity Pillar", name_a: f"{ta.get('mom_score', 50):.1f}/100", name_b: f"{tb.get('mom_score', 50):.1f}/100", "Spread / Delta": calc_delta(ta.get('mom_score'), tb.get('mom_score'), "pts"), "Advantage": pick_adv(ta.get("mom_score"), tb.get("mom_score"), True), "Signal / Context": "RSI, MACD, ADX, ROC impulse"},
        {"Category": "Composite Technical Scores", "Metric": "Capital Volatility Stability", name_a: f"{ta.get('volat_score', 50):.1f}/100", name_b: f"{tb.get('volat_score', 50):.1f}/100", "Spread / Delta": calc_delta(ta.get('volat_score'), tb.get('volat_score'), "pts"), "Advantage": pick_adv(ta.get("volat_score"), tb.get("volat_score"), True), "Signal / Context": "ATR stability & low drawdown risk"},
        {"Category": "Composite Technical Scores", "Metric": "Institutional Money Flow", name_a: f"{ta.get('flow_score', 50):.1f}/100", name_b: f"{tb.get('flow_score', 50):.1f}/100", "Spread / Delta": calc_delta(ta.get('flow_score'), tb.get('flow_score'), "pts"), "Advantage": pick_adv(ta.get("flow_score"), tb.get("flow_score"), True), "Signal / Context": "MFI, CMF, and RVOL buying intensity"},
        {"Category": "Composite Technical Scores", "Metric": "Mean-Reversion Potential", name_a: f"{ta.get('reversion_score', 50):.1f}/100", name_b: f"{tb.get('reversion_score', 50):.1f}/100", "Spread / Delta": calc_delta(ta.get('reversion_score'), tb.get('reversion_score'), "pts"), "Advantage": pick_adv(ta.get("reversion_score"), tb.get("reversion_score"), True), "Signal / Context": "Bounce edge from oversold exhaustion"},
    ]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main Application Render
# -----------------------------------------------------------------------------
def render_page():
    """Render the Market Explorer & Screener."""
    st.set_page_config(
        page_title="Market Explorer - QuantTerminal",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_theme()

    # Session State defaults
    if "comp_stock_a" not in st.session_state:
        st.session_state["comp_stock_a"] = "RELIANCE.NS"
    if "comp_stock_b" not in st.session_state:
        st.session_state["comp_stock_b"] = "TCS.NS"
    if "tech_stock_a" not in st.session_state:
        st.session_state["tech_stock_a"] = "RELIANCE.NS"
    if "tech_stock_b" not in st.session_state:
        st.session_state["tech_stock_b"] = "TCS.NS"

    # Top Header
    st.markdown(
        """
        <div style="padding-top: 4px; margin-bottom: 12px;">
            <h1 style="margin: 0; font-size: 2.15rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em;">
                🌐 Institutional Market Explorer & Screener
            </h1>
            <p style="margin: 4px 0 0 0; color: #38BDF8; font-size: 0.95rem; font-weight: 500;">
                Multi-Condition Technical & Fundamental Screener · Deep-Dive Head-to-Head 2-Stock Comparison
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs (4 Dedicated Pillars)
    tab_screener, tab_compare, tab_tech_screener, tab_tech_compare = st.tabs([
        "🔍 Institutional Screener (General)",
        "⚖️ Fundamental Comparison (Head-to-Head)",
        "📈 Technicals Screener (Multi-Indicator)",
        "⚡ Technicals Comparison (Charts & Indicators)",
    ])

    # =========================================================================
    # TAB 1: INSTITUTIONAL STOCK SCREENER
    # =========================================================================
    with tab_screener:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        # Quick Preset Chips / Pills
        preset_options = [
            "All Stocks (Custom Filters)",
            "🚀 Bullish Momentum Breakout",
            "📉 Oversold Mean Reversion",
            "⚡ High Volume Accumulation",
            "👑 Large-Cap Quality Compounders",
            "💰 High Dividend Value",
            "🏆 52-Week High Breakouts",
            "🛡️ Strong Buy Technical Consensus",
        ]

        p_col1, p_col2, p_col3 = st.columns([1.1, 1.2, 1.7])
        with p_col1:
            market_choice = st.selectbox(
                "Universe / Region",
                ["India", "US (America)"],
                index=0,
                key="scr_market_select",
            )
            market_val = "india" if "India" in market_choice else "america"
            curr_symbol = "₹" if market_val == "india" else "$"

        with p_col2:
            if market_val == "india":
                exchange_choice = st.selectbox(
                    "Stock Exchange",
                    ["NSE (National Stock Exchange)", "BSE (Bombay Stock Exchange)", "All Exchanges (NSE + BSE)"],
                    index=0,
                    key="scr_exchange_select",
                )
                if exchange_choice.startswith("NSE"):
                    ex_val = "NSE"
                elif exchange_choice.startswith("BSE"):
                    ex_val = "BSE"
                else:
                    ex_val = "All"
            else:
                exchange_choice = st.selectbox(
                    "Stock Exchange",
                    ["All US Exchanges", "NASDAQ", "NYSE", "AMEX"],
                    index=0,
                    key="scr_exchange_select_us",
                )
                if exchange_choice.startswith("NASDAQ"):
                    ex_val = "NASDAQ"
                elif exchange_choice.startswith("NYSE"):
                    ex_val = "NYSE"
                elif exchange_choice.startswith("AMEX"):
                    ex_val = "AMEX"
                else:
                    ex_val = "All"

        with p_col3:
            preset_choice = st.selectbox(
                "⚡ Quick Screener Presets",
                preset_options,
                index=1,
                key="scr_preset_select",
            )

        # ---------------------------------------------------------------------
        # Professional Condition Filter Bar (Easy-to-Click Inputs)
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div style="font-size: 0.82rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin: 12px 0 6px 0;">
                🛠️ Input Value Conditions & Filters
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                sector_input = st.selectbox(
                    "Sector",
                    [
                        "All",
                        "Commercial Services",
                        "Communications",
                        "Consumer Durables",
                        "Consumer Non-Durables",
                        "Consumer Services",
                        "Distribution Services",
                        "Electronic Technology",
                        "Energy Minerals",
                        "Finance",
                        "Health Services",
                        "Health Technology",
                        "Industrial Services",
                        "Miscellaneous",
                        "Non-Energy Minerals",
                        "Process Industries",
                        "Producer Manufacturing",
                        "Retail Trade",
                        "Technology Services",
                        "Transportation",
                        "Utilities",
                    ],
                    index=0,
                    key="cond_sector",
                )
                ma_cond = st.selectbox(
                    "Moving Average Filter",
                    [
                        "Any",
                        "Price > 50 SMA",
                        "Price > 200 SMA",
                        "Price > 50 SMA & 200 SMA",
                        "Golden Cross (50 SMA > 200 SMA)",
                    ],
                    index=0,
                    key="cond_ma",
                )

            with c2:
                # Market Cap condition
                if market_val == "india":
                    mcap_scale = 1e7  # 1 Crore = 10,000,000
                    mcap_unit = "Cr"
                    min_mcap_in = st.number_input(
                        f"Min Market Cap (₹ {mcap_unit})",
                        min_value=0.0,
                        value=500.0,
                        step=500.0,
                        key="cond_min_mcap",
                    )
                    min_mcap_val = min_mcap_in * mcap_scale
                else:
                    mcap_scale = 1e9  # 1 Billion
                    mcap_unit = "B"
                    min_mcap_in = st.number_input(
                        f"Min Market Cap ($ {mcap_unit})",
                        min_value=0.0,
                        value=1.0,
                        step=1.0,
                        key="cond_min_mcap_us",
                    )
                    min_mcap_val = min_mcap_in * mcap_scale

                min_rvol_in = st.number_input(
                    "Min Relative Volume (RVOL)",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.25,
                    key="cond_min_rvol",
                )

            with c3:
                c_rsi1, c_rsi2 = st.columns(2)
                with c_rsi1:
                    min_rsi_in = st.number_input(
                        "Min RSI(14)", 0.0, 100.0, 0.0, step=5.0, key="cond_min_rsi"
                    )
                with c_rsi2:
                    max_rsi_in = st.number_input(
                        "Max RSI(14)", 0.0, 100.0, 100.0, step=5.0, key="cond_max_rsi"
                    )

                c_pe1, c_pe2 = st.columns(2)
                with c_pe1:
                    min_pe_in = st.number_input(
                        "Min P/E", 0.0, 200.0, 0.0, step=5.0, key="cond_min_pe"
                    )
                with c_pe2:
                    max_pe_in = st.number_input(
                        "Max P/E", 0.0, 200.0, 150.0, step=10.0, key="cond_max_pe"
                    )

            with c4:
                c_chg1, c_chg2 = st.columns(2)
                with c_chg1:
                    min_chg_in = st.number_input(
                        "Min 1D Change %", -100.0, 100.0, -100.0, step=1.0, key="cond_min_chg"
                    )
                with c_chg2:
                    max_chg_in = st.number_input(
                        "Max 1D Change %", -100.0, 100.0, 100.0, step=1.0, key="cond_max_chg"
                    )

                sort_choice = st.selectbox(
                    "Sort By",
                    [
                        "market_cap_basic",
                        "change",
                        "relative_volume_10d_calc",
                        "volume",
                        "RSI",
                        "price_earnings_ttm",
                        "close",
                    ],
                    index=0,
                    key="cond_sort_col",
                )

        # Bottom Bar: Sort direction & display limit
        b_col1, b_col2, b_col3 = st.columns([1.2, 1.2, 2.6])
        with b_col1:
            sort_dir = st.radio(
                "Sort Order", ["Descending", "Ascending"], horizontal=True, key="cond_sort_dir"
            )
            is_asc = sort_dir == "Ascending"
        with b_col2:
            display_limit = st.selectbox(
                "Result Limit", [25, 50, 100, 200], index=1, key="cond_limit"
            )

        # ---------------------------------------------------------------------
        # Execute Screener Query
        # ---------------------------------------------------------------------
        with st.spinner("Executing real-time institutional screener scan…"):
            total_matching, scr_df = run_tradingview_screener(
                market=market_val,
                exchange=ex_val,
                preset=preset_choice,
                sector=sector_input,
                min_mcap=min_mcap_val,
                min_pe=min_pe_in,
                max_pe=max_pe_in,
                min_rsi=min_rsi_in,
                max_rsi=max_rsi_in,
                min_rvol=min_rvol_in,
                min_change=min_chg_in,
                max_change=max_chg_in,
                ma_filter=ma_cond,
                sort_by=sort_choice,
                sort_asc=is_asc,
                limit=display_limit,
            )

        if scr_df.empty:
            st.info("No stocks match the exact filter criteria. Try relaxing your threshold inputs.")
        else:
            # Executive Summary KPI Badges
            k1, k2, k3, k4, k5 = st.columns(5)
            adv = int((scr_df["change"] > 0).sum())
            dec = int((scr_df["change"] < 0).sum())
            med_pe = (
                float(scr_df["price_earnings_ttm"].dropna().median())
                if "price_earnings_ttm" in scr_df.columns and len(scr_df["price_earnings_ttm"].dropna()) > 0
                else 0.0
            )
            top_gainer = scr_df.sort_values("change", ascending=False).iloc[0]
            max_rvol = (
                scr_df.sort_values("relative_volume_10d_calc", ascending=False).iloc[0]
                if "relative_volume_10d_calc" in scr_df.columns
                else top_gainer
            )

            k1.metric("Screen Matches", f"{len(scr_df):,}", f"Universe: {total_matching:,}")
            k2.metric("Market Breadth", f"▲ {adv} Up", f"▼ {dec} Down")
            k3.metric("Median P/E", f"{med_pe:.1f}x")
            k4.metric("Top Mover", f"{top_gainer['name']}", f"{top_gainer['change']:+.2f}%")
            k5.metric(
                "Max Volume Spike",
                f"{max_rvol['name']}",
                f"{max_rvol.get('relative_volume_10d_calc', 1.0):.2f}x RVOL",
            )

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            # Professional Formatted Table
            table_df = scr_df.copy()
            table_df["Price"] = table_df["close"].apply(lambda x: f"{curr_symbol}{x:,.2f}")
            table_df["Change %"] = table_df["change"].apply(lambda x: f"{x:+.2f}%")
            table_df["RSI"] = table_df["RSI"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
            table_df["RVOL"] = table_df["relative_volume_10d_calc"].apply(
                lambda x: f"{x:.2f}x" if pd.notna(x) else "—"
            )
            table_df["P/E"] = table_df["price_earnings_ttm"].apply(
                lambda x: f"{x:.1f}x" if (pd.notna(x) and x > 0) else "—"
            )
            table_df["Market Cap"] = table_df["market_cap_basic"].apply(
                lambda x: _fmt_num(x) if pd.notna(x) else "—"
            )
            table_df["Volume"] = table_df["volume"].apply(
                lambda x: _fmt_num(x) if pd.notna(x) else "—"
            )
            table_df["Technical Rating"] = table_df["Recommend.All"].apply(
                lambda x: (
                    "Strong Buy"
                    if x >= 0.5
                    else (
                        "Buy"
                        if x >= 0.1
                        else ("Neutral" if x >= -0.1 else ("Sell" if x >= -0.5 else "Strong Sell"))
                    )
                )
                if pd.notna(x)
                else "—"
            )

            view_cols = [
                "name",
                "exchange",
                "description",
                "Price",
                "Change %",
                "RSI",
                "RVOL",
                "P/E",
                "Market Cap",
                "Volume",
                "Technical Rating",
                "sector",
            ]
            renames = {
                "name": "Symbol",
                "exchange": "Exchange",
                "description": "Company Name",
                "sector": "Sector",
            }

            st.dataframe(
                table_df[view_cols].rename(columns=renames),
                hide_index=True,
                width="stretch",
            )

            # -----------------------------------------------------------------
            # 1-Click Send to Deep Comparison Action
            # -----------------------------------------------------------------
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            act_col1, act_col2 = st.columns([3, 1])

            with act_col1:
                available_tickers = []
                for _, r in scr_df.iterrows():
                    sym = r["name"]
                    ex_code = r.get("exchange", "NSE")
                    if ex_code == "NSE":
                        available_tickers.append(f"{sym}.NS")
                    elif ex_code == "BSE":
                        available_tickers.append(f"{sym}.BO")
                    else:
                        available_tickers.append(sym)
                st.markdown(
                    "<span style='color: #38BDF8; font-weight: 600; font-size: 0.88rem;'>⚖️ Quick Select for Deep-Dive Comparison (Tab 2)</span>",
                    unsafe_allow_html=True,
                )
                sel_pair = st.multiselect(
                    "Pick any 2 screened stocks to send directly to the Head-to-Head Comparison tab:",
                    options=available_tickers,
                    default=available_tickers[:2] if len(available_tickers) >= 2 else available_tickers,
                    max_selections=2,
                    key="h2h_quick_select",
                )
                if len(sel_pair) == 2:
                    if st.button(
                        f"🚀 Compare {sel_pair[0]} vs {sel_pair[1]} in Tab 2",
                        key="btn_send_compare",
                    ):
                        st.session_state["comp_stock_a"] = sel_pair[0]
                        st.session_state["comp_stock_b"] = sel_pair[1]
                        st.success(f"Configured comparison: {sel_pair[0]} vs {sel_pair[1]}! Click Tab 2 to view.")

            with act_col2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                csv_bytes = scr_df.to_csv(index=False)
                st.download_button(
                    "📥 Export Screener CSV",
                    csv_bytes,
                    file_name=f"screener_results_{market_val}.csv",
                    mime="text/csv",
                    key="btn_dl_scr_tab",
                )

    # =========================================================================
    # TAB 2: DEEP-DIVE 2-STOCK COMPARISON
    # =========================================================================
    with tab_compare:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        # Load indexed snapshot universe data
        snapshots = load_universe_snapshots()

        # Scope selector
        scope_col, period_col, freq_col = st.columns([2.2, 1.0, 1.0])
        with scope_col:
            comp_scope = st.radio(
                "Stock Universe Scope",
                ["All Markets (18,500+ India & US)", "India Stocks (7,600+ NSE & BSE)", "US Stocks (10,900+ NASDAQ/NYSE/AMEX)"],
                horizontal=True,
                key="h2h_univ_scope",
            )
        with period_col:
            comp_period = st.selectbox(
                "Historical Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
                index=3,
                key="h2h_period",
            )
        with freq_col:
            comp_interval = st.selectbox(
                "Frequency",
                ["1d", "1wk", "1mo"],
                index=0,
                key="h2h_freq",
            )

        # Select option pool
        if "India Stocks" in comp_scope:
            active_options = snapshots["india_labels"] if snapshots["india_labels"] else ["RELIANCE (RELIANCE.NS · NSE)"]
            default_a_sym = "RELIANCE.NS"
            default_b_sym = "TCS.NS"
        elif "US Stocks" in comp_scope:
            active_options = snapshots["us_labels"] if snapshots["us_labels"] else ["Apple Inc. (AAPL · NASDAQ)"]
            default_a_sym = "AAPL"
            default_b_sym = "MSFT"
        else:
            active_options = snapshots["all_labels"] if snapshots["all_labels"] else ["RELIANCE (RELIANCE.NS · NSE)"]
            default_a_sym = "RELIANCE.NS"
            default_b_sym = "TCS.NS"

        # Guard against stale session state across universe switches
        if st.session_state.get("prev_h2h_scope") != comp_scope:
            st.session_state["prev_h2h_scope"] = comp_scope
            if "select_label_a" in st.session_state and st.session_state["select_label_a"] not in active_options:
                st.session_state.pop("select_label_a", None)
            if "select_label_b" in st.session_state and st.session_state["select_label_b"] not in active_options:
                st.session_state.pop("select_label_b", None)

        # Determine index for Stock A
        stored_a = st.session_state.get("comp_stock_a", default_a_sym).upper()
        idx_a = 0
        for i, opt in enumerate(active_options):
            if f"({stored_a} ·" in opt.upper() or opt.upper().startswith(f"{stored_a} "):
                idx_a = i
                break

        # Determine index for Stock B
        stored_b = st.session_state.get("comp_stock_b", default_b_sym).upper()
        idx_b = min(1, len(active_options) - 1)
        for i, opt in enumerate(active_options):
            if f"({stored_b} ·" in opt.upper() or opt.upper().startswith(f"{stored_b} "):
                idx_b = i
                break

        # Dropdown selection
        sel_c1, sel_c2 = st.columns(2)
        with sel_c1:
            chosen_label_a = st.selectbox(
                "Candidate Stock A (Search by name or ticker)",
                options=active_options,
                index=idx_a,
                key="select_label_a",
            )
        with sel_c2:
            chosen_label_b = st.selectbox(
                "Candidate Stock B (Search by name or ticker)",
                options=active_options,
                index=idx_b,
                key="select_label_b",
            )

        # Optional manual symbol override
        with st.expander("⚙️ Advanced: Enter Custom Ticker Override (Optional)"):
            c_ov1, c_ov2 = st.columns(2)
            with c_ov1:
                override_a = st.text_input(
                    "Override Stock A (e.g. BTC-USD, SPY, ^NSEI)",
                    value="",
                    key="override_sym_a",
                ).strip().upper()
            with c_ov2:
                override_b = st.text_input(
                    "Override Stock B (e.g. ETH-USD, QQQ, ^GSPC)",
                    value="",
                    key="override_sym_b",
                ).strip().upper()

        # Resolve final tickers and snapshot dictionaries
        snap_a = snapshots["by_label"].get(chosen_label_a)
        if override_a:
            stock_a_resolved = override_a
            snap_a = snapshots["by_ticker"].get(override_a, snap_a)
        elif snap_a:
            stock_a_resolved = snap_a["yf_ticker"]
        else:
            stock_a_resolved = "RELIANCE.NS"

        snap_b = snapshots["by_label"].get(chosen_label_b)
        if override_b:
            stock_b_resolved = override_b
            snap_b = snapshots["by_ticker"].get(override_b, snap_b)
        elif snap_b:
            stock_b_resolved = snap_b["yf_ticker"]
        else:
            stock_b_resolved = "TCS.NS"

        st.session_state["comp_stock_a"] = stock_a_resolved
        st.session_state["comp_stock_b"] = stock_b_resolved

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        if not stock_a_resolved or not stock_b_resolved:
            st.info("Please select or enter both Stock A and Stock B above.")
        elif stock_a_resolved == stock_b_resolved:
            st.warning("Please choose two different stocks to perform a head-to-head comparative analysis.")
        else:
            with st.spinner(f"Fetching live market history and financial data for {stock_a_resolved} vs {stock_b_resolved}…"):
                raw_pair_data, info_a, info_b = fetch_comparison_pair_data(
                    stock_a_resolved, stock_b_resolved, period=comp_period, interval=comp_interval
                )

            if raw_pair_data.empty:
                st.error("Unable to load price data for the specified tickers. Please verify the symbols.")
            else:
                # Extract close series
                if isinstance(raw_pair_data.columns, pd.MultiIndex):
                    try:
                        ca = raw_pair_data["Close"][stock_a_resolved].dropna()
                        cb = raw_pair_data["Close"][stock_b_resolved].dropna()
                    except KeyError:
                        ca = raw_pair_data.xs("Close", axis=1, level=1)[stock_a_resolved].dropna()
                        cb = raw_pair_data.xs("Close", axis=1, level=1)[stock_b_resolved].dropna()
                else:
                    st.warning("Historical data format error.")
                    return

                # Align on common trading days
                common_df = pd.concat([ca, cb], axis=1).dropna()
                if common_df.empty or len(common_df) < 5:
                    st.error("Insufficient overlapping trading history between the selected assets.")
                    return

                ca_aligned = common_df[stock_a_resolved]
                cb_aligned = common_df[stock_b_resolved]

                name_a = (snap_a.get("name") if snap_a else None) or info_a.get("shortName") or stock_a_resolved
                name_b = (snap_b.get("name") if snap_b else None) or info_b.get("shortName") or stock_b_resolved
                curr_a = "₹" if (snap_a and snap_a.get("currency") == "INR") else CURRENCY_SYMBOLS.get(info_a.get("currency", "USD"), "$")
                curr_b = "₹" if (snap_b and snap_b.get("currency") == "INR") else CURRENCY_SYMBOLS.get(info_b.get("currency", "USD"), "$")

                last_price_a = float(ca_aligned.iloc[-1])
                last_price_b = float(cb_aligned.iloc[-1])
                ret_a_tot = float((ca_aligned.iloc[-1] / ca_aligned.iloc[0] - 1.0) * 100.0)
                ret_b_tot = float((cb_aligned.iloc[-1] / cb_aligned.iloc[0] - 1.0) * 100.0)

                chg_1d_a = snap_a.get("change_1d") if snap_a and snap_a.get("change_1d") is not None else float((ca_aligned.iloc[-1] / ca_aligned.iloc[-2] - 1.0) * 100.0) if len(ca_aligned) > 1 else 0.0
                chg_1d_b = snap_b.get("change_1d") if snap_b and snap_b.get("change_1d") is not None else float((cb_aligned.iloc[-1] / cb_aligned.iloc[-2] - 1.0) * 100.0) if len(cb_aligned) > 1 else 0.0

                mcap_disp_a = (snap_a.get("mcap") if snap_a else None) or info_a.get("marketCap", 0)
                mcap_disp_b = (snap_b.get("mcap") if snap_b else None) or info_b.get("marketCap", 0)
                pe_disp_a = (snap_a.get("pe") if snap_a else None) or info_a.get("trailingPE", 0)
                pe_disp_b = (snap_b.get("pe") if snap_b else None) or info_b.get("trailingPE", 0)
                eps_disp_a = (snap_a.get("eps") if snap_a else None) or info_a.get("trailingEps", 0)
                eps_disp_b = (snap_b.get("eps") if snap_b else None) or info_b.get("trailingEps", 0)
                div_disp_a = (snap_a.get("div_yield") if snap_a else None) or (info_a.get("dividendYield", 0) * 100 if info_a.get("dividendYield") else 0.0)
                div_disp_b = (snap_b.get("div_yield") if snap_b else None) or (info_b.get("dividendYield", 0) * 100 if info_b.get("dividendYield") else 0.0)

                pe_str_a = f"{pe_disp_a:.1f}x" if pe_disp_a else "—"
                pe_str_b = f"{pe_disp_b:.1f}x" if pe_disp_b else "—"
                div_str_a = f"{div_disp_a:.2f}%" if div_disp_a else "0.00%"
                div_str_b = f"{div_disp_b:.2f}%" if div_disp_b else "0.00%"
                eps_str_a = f"{curr_a}{eps_disp_a:,.2f}" if eps_disp_a else "—"
                eps_str_b = f"{curr_b}{eps_disp_b:,.2f}" if eps_disp_b else "—"
                eps_growth_disp_a = f"{snap_a.get('eps_growth', 0):+.1f}%" if (snap_a and snap_a.get('eps_growth') is not None) else "—"
                eps_growth_disp_b = f"{snap_b.get('eps_growth', 0):+.1f}%" if (snap_b and snap_b.get('eps_growth') is not None) else "—"

                vol_1d_disp_a = _fmt_num(snap_a.get("volume_1d", 0) if snap_a else 0)
                vol_1d_disp_b = _fmt_num(snap_b.get("volume_1d", 0) if snap_b else 0)

                # -------------------------------------------------------------
                # 1. Executive Side-by-Side Scorecards
                # -------------------------------------------------------------
                card_col1, card_col2 = st.columns(2)

                with card_col1:
                    st.markdown(
                        f"""
                        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 18px 22px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.76rem; font-weight: 700; color: #10B981; letter-spacing: 0.08em; text-transform: uppercase;">Stock A Candidate</span>
                                <span style="font-size: 0.74rem; background: rgba(16, 185, 129, 0.2); color: #10B981; padding: 2px 8px; border-radius: 6px; font-weight: 600;">{snap_a.get('exchange', 'NSE') if snap_a else 'EQUITY'}</span>
                            </div>
                            <div style="font-size: 1.55rem; font-weight: 700; color: #FFFFFF; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name_a}</div>
                            <div style="color: #38BDF8; font-size: 0.86rem; font-weight: 500;">{stock_a_resolved} · {snap_a.get('sector', info_a.get('sector', 'Equity')) if snap_a else info_a.get('sector', 'Equity')}</div>
                            <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 10px;">
                                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.75rem; font-weight: 700; color: #FFFFFF;">{curr_a}{last_price_a:,.2f}</span>
                                <span style="font-size: 1.05rem; font-weight: 600; color: {'#00E676' if ret_a_tot >= 0 else '#EF4444'};">{ret_a_tot:+.2f}% ({comp_period})</span>
                                <span style="font-size: 0.85rem; color: {'#00E676' if chg_1d_a >= 0 else '#EF4444'}; font-weight: 500;">1D: {chg_1d_a:+.2f}%</span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                                <div><span style="color: #94A3B8;">Market Cap:</span> <br><strong>{_fmt_num(mcap_disp_a)}</strong></div>
                                <div><span style="color: #94A3B8;">Trailing P/E:</span> <br><strong>{pe_str_a}</strong></div>
                                <div><span style="color: #94A3B8;">Div Yield:</span> <br><strong>{div_str_a}</strong></div>
                                <div><span style="color: #94A3B8;">Diluted EPS:</span> <br><strong>{eps_str_a}</strong></div>
                                <div><span style="color: #94A3B8;">EPS Growth:</span> <br><strong>{eps_growth_disp_a}</strong></div>
                                <div><span style="color: #94A3B8;">Volume 1D:</span> <br><strong>{vol_1d_disp_a}</strong></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with card_col2:
                    st.markdown(
                        f"""
                        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 18px 22px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.76rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.08em; text-transform: uppercase;">Stock B Candidate</span>
                                <span style="font-size: 0.74rem; background: rgba(56, 189, 248, 0.2); color: #38BDF8; padding: 2px 8px; border-radius: 6px; font-weight: 600;">{snap_b.get('exchange', 'NSE') if snap_b else 'EQUITY'}</span>
                            </div>
                            <div style="font-size: 1.55rem; font-weight: 700; color: #FFFFFF; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name_b}</div>
                            <div style="color: #38BDF8; font-size: 0.86rem; font-weight: 500;">{stock_b_resolved} · {snap_b.get('sector', info_b.get('sector', 'Equity')) if snap_b else info_b.get('sector', 'Equity')}</div>
                            <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 10px;">
                                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.75rem; font-weight: 700; color: #FFFFFF;">{curr_b}{last_price_b:,.2f}</span>
                                <span style="font-size: 1.05rem; font-weight: 600; color: {'#00E676' if ret_b_tot >= 0 else '#EF4444'};">{ret_b_tot:+.2f}% ({comp_period})</span>
                                <span style="font-size: 0.85rem; color: {'#00E676' if chg_1d_b >= 0 else '#EF4444'}; font-weight: 500;">1D: {chg_1d_b:+.2f}%</span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                                <div><span style="color: #94A3B8;">Market Cap:</span> <br><strong>{_fmt_num(mcap_disp_b)}</strong></div>
                                <div><span style="color: #94A3B8;">Trailing P/E:</span> <br><strong>{pe_str_b}</strong></div>
                                <div><span style="color: #94A3B8;">Div Yield:</span> <br><strong>{div_str_b}</strong></div>
                                <div><span style="color: #94A3B8;">Diluted EPS:</span> <br><strong>{eps_str_b}</strong></div>
                                <div><span style="color: #94A3B8;">EPS Growth:</span> <br><strong>{eps_growth_disp_b}</strong></div>
                                <div><span style="color: #94A3B8;">Volume 1D:</span> <br><strong>{vol_1d_disp_b}</strong></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                # -------------------------------------------------------------
                # 2. Deep-Dive Head-to-Head Comparison Matrix
                # -------------------------------------------------------------
                st.markdown(
                    """
                    <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                        📊 Deep-Dive Head-to-Head Comparison Matrix (Snapshot + Market Data)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                h2h_df = build_head_to_head_matrix(
                    ca_aligned,
                    cb_aligned,
                    info_a,
                    info_b,
                    snap_a,
                    snap_b,
                    stock_a_resolved,
                    stock_b_resolved,
                )

                # Pillar category selector & quick search
                cat_col1, cat_col2, cat_col3 = st.columns([1.6, 1.4, 1.0])
                with cat_col1:
                    filter_cat = st.selectbox(
                        "Comparison Dimension Filter",
                        [
                            "All Dimensions",
                            "Performance & Returns",
                            "Valuation & Fundamentals",
                            "Quality & Financials",
                            "Financial Health & Solvency",
                            "Risk & Volatility",
                            "Liquidity & Volume",
                        ],
                        index=0,
                        key="h2h_category_filter",
                    )
                with cat_col2:
                    h2h_search = st.text_input(
                        "🔍 Filter Metric",
                        placeholder="e.g., P/E, ROE, Sharpe, Drawdown...",
                        key="h2h_search_input",
                    )
                display_h2h = h2h_df
                if filter_cat != "All Dimensions":
                    display_h2h = display_h2h[display_h2h["Category"] == filter_cat]
                if h2h_search.strip():
                    q = h2h_search.strip().lower()
                    display_h2h = display_h2h[
                        display_h2h["Metric"].str.lower().str.contains(q, na=False)
                        | display_h2h["Institutional Benchmark"].str.lower().str.contains(q, na=False)
                    ]

                with cat_col3:
                    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                    csv_h2h = display_h2h.to_csv(index=False)
                    st.download_button(
                        "📥 Export CSV",
                        data=csv_h2h,
                        file_name=f"{stock_a_resolved}_vs_{stock_b_resolved}_fundamental_matrix.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_h2h_matrix",
                    )

                st.caption(f"Showing **{len(display_h2h)}** institutional fundamental metrics for **{name_a}** vs **{name_b}**")

                st.dataframe(
                    display_h2h,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Category": st.column_config.TextColumn("Dimension", width="small"),
                        "Metric": st.column_config.TextColumn("Comparison Metric", width="medium"),
                        stock_a_resolved: st.column_config.TextColumn(f"🟢 {stock_a_resolved}", width="small"),
                        stock_b_resolved: st.column_config.TextColumn(f"🔵 {stock_b_resolved}", width="small"),
                        "Spread / Delta": st.column_config.TextColumn("Delta (A − B)", width="small"),
                        "Advantage": st.column_config.TextColumn("Leading Asset", width="small"),
                        "Institutional Benchmark": st.column_config.TextColumn("Benchmark / Context", width="large"),
                    },
                )

                st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

                # -------------------------------------------------------------
                # 3. Chart 1: Normalized Relative Performance & Alpha Spread (%)
                # -------------------------------------------------------------
                st.markdown(
                    """
                    <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                        📈 Chart 1: Relative Alpha & Normalized Trajectory Spread (%)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                norm_a = (ca_aligned / ca_aligned.iloc[0] - 1.0) * 100.0
                norm_b = (cb_aligned / cb_aligned.iloc[0] - 1.0) * 100.0
                alpha_spread = norm_a - norm_b

                fig_h2h = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    row_heights=[0.68, 0.32],
                    subplot_titles=(
                        f"Cumulative Performance Trajectory (%) · Rebased from {ca_aligned.index[0].strftime('%b %d, %Y')}",
                        f"Alpha Spread ({stock_a_resolved} − {stock_b_resolved})",
                    ),
                )

                fig_h2h.add_trace(
                    go.Scatter(
                        x=norm_a.index,
                        y=norm_a,
                        mode="lines",
                        name=f"{stock_a_resolved} ({ret_a_tot:+.2f}%)",
                        line=dict(color="#10B981", width=2.5),
                        hovertemplate=f"<b>{stock_a_resolved}</b>: %{{y:+.2f}}%<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                fig_h2h.add_trace(
                    go.Scatter(
                        x=norm_b.index,
                        y=norm_b,
                        mode="lines",
                        name=f"{stock_b_resolved} ({ret_b_tot:+.2f}%)",
                        line=dict(color="#38BDF8", width=2.5),
                        hovertemplate=f"<b>{stock_b_resolved}</b>: %{{y:+.2f}}%<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                fig_h2h.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="rgba(255, 255, 255, 0.3)",
                    line_width=1,
                    row=1,
                    col=1,
                )

                spread_colors = ["#10B981" if s >= 0 else "#EF4444" for s in alpha_spread]
                fig_h2h.add_trace(
                    go.Bar(
                        x=alpha_spread.index,
                        y=alpha_spread,
                        name="Alpha Spread",
                        marker_color=spread_colors,
                        opacity=0.7,
                        hovertemplate="<b>Alpha Spread</b>: %{y:+.2f}%<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

                fig_h2h.update_layout(
                    template="plotly_dark",
                    height=520,
                    margin=dict(l=10, r=10, t=35, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0.0,
                        bgcolor="rgba(15, 23, 42, 0.7)",
                    ),
                    hovermode="x unified",
                )
                fig_h2h.update_yaxes(title_text="Performance (%)", row=1, col=1)
                fig_h2h.update_yaxes(title_text="Spread (%)", row=2, col=1)
                st.plotly_chart(fig_h2h, use_container_width=True)

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                # -------------------------------------------------------------
                # 4. Chart 2: Quantitative Pair Trading Ratio & Bollinger Bands
                # -------------------------------------------------------------
                st.markdown(
                    """
                    <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                        ⚖️ Chart 2: Quantitative Pair Trading Analysis (Price Ratio & Bollinger Bands)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                price_ratio = ca_aligned / cb_aligned
                sma_win = min(50, max(5, len(price_ratio) // 3))
                ratio_sma = price_ratio.rolling(sma_win, min_periods=5).mean()
                ratio_std = price_ratio.rolling(sma_win, min_periods=5).std()
                upper_2s = ratio_sma + 2.0 * ratio_std
                lower_2s = ratio_sma - 2.0 * ratio_std
                upper_1s = ratio_sma + 1.0 * ratio_std
                lower_1s = ratio_sma - 1.0 * ratio_std
                curr_ratio = float(price_ratio.iloc[-1])
                curr_sma = float(ratio_sma.iloc[-1]) if pd.notna(ratio_sma.iloc[-1]) else curr_ratio
                curr_std = float(ratio_std.iloc[-1]) if pd.notna(ratio_std.iloc[-1]) and ratio_std.iloc[-1] > 0 else 1e-6
                curr_z = (curr_ratio - curr_sma) / curr_std
                z_series = (price_ratio - ratio_sma) / (ratio_std + 1e-9)

                # Signal interpretation
                if curr_z > 2.0:
                    pair_signal = "🚨 Overbought Stock A · Mean Reversion Signal: Short A & Long B"
                    signal_color = "#EF4444"
                elif curr_z < -2.0:
                    pair_signal = "💎 Oversold Stock A · Mean Reversion Signal: Long A & Short B"
                    signal_color = "#10B981"
                elif abs(curr_z) <= 1.0:
                    pair_signal = "⚖️ Fair Value Zone · Ratio Oscillating Within 1-Sigma Band"
                    signal_color = "#38BDF8"
                else:
                    pair_signal = "⚠️ Moderate Divergence · Approaching Mean Reversion Threshold"
                    signal_color = "#F59E0B"

                # Ratio KPI tiles
                rk1, rk2, rk3, rk4 = st.columns(4)
                rk1.metric("Current Price Ratio (A/B)", f"{curr_ratio:.4f}")
                rk2.metric(f"{sma_win}-Day Rolling Mean", f"{curr_sma:.4f}")
                rk3.metric("Current Z-Score (Deviation)", f"{curr_z:+.2f}σ")
                with rk4:
                    st.markdown(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid {signal_color}; border-radius: 8px; padding: 8px 12px; font-size: 0.78rem; color: {signal_color}; font-weight: 600;">
                            {pair_signal}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                fig_ratio = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(
                        f"Pair Price Ratio ({stock_a_resolved} / {stock_b_resolved}) & 2-Sigma Bollinger Bands",
                        "Rolling Standardized Z-Score Oscillator",
                    ),
                )

                # Ratio + bands
                fig_ratio.add_trace(
                    go.Scatter(
                        x=price_ratio.index,
                        y=price_ratio,
                        mode="lines",
                        name=f"Ratio ({stock_a_resolved}/{stock_b_resolved})",
                        line=dict(color="#F59E0B", width=2.2),
                        hovertemplate="<b>Ratio</b>: %{y:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                fig_ratio.add_trace(
                    go.Scatter(
                        x=ratio_sma.index,
                        y=ratio_sma,
                        mode="lines",
                        name=f"{sma_win}-SMA Mean",
                        line=dict(color="rgba(255, 255, 255, 0.75)", dash="dash", width=1.5),
                        hovertemplate="<b>SMA Mean</b>: %{y:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                fig_ratio.add_trace(
                    go.Scatter(
                        x=upper_2s.index,
                        y=upper_2s,
                        mode="lines",
                        name="+2σ Upper Band",
                        line=dict(color="#EF4444", dash="dot", width=1.5),
                        hovertemplate="<b>+2σ Band</b>: %{y:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                fig_ratio.add_trace(
                    go.Scatter(
                        x=lower_2s.index,
                        y=lower_2s,
                        mode="lines",
                        name="-2σ Lower Band",
                        line=dict(color="#10B981", dash="dot", width=1.5),
                        hovertemplate="<b>-2σ Band</b>: %{y:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                # Z-score oscillator
                fig_ratio.add_trace(
                    go.Scatter(
                        x=z_series.index,
                        y=z_series,
                        mode="lines",
                        name="Z-Score",
                        line=dict(color="#38BDF8", width=1.8),
                        hovertemplate="<b>Z-Score</b>: %{y:+.2f}σ<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
                fig_ratio.add_hline(y=2.0, line_dash="dash", line_color="#EF4444", line_width=1, row=2, col=1)
                fig_ratio.add_hline(y=-2.0, line_dash="dash", line_color="#10B981", line_width=1, row=2, col=1)
                fig_ratio.add_hline(y=0.0, line_dash="solid", line_color="rgba(255,255,255,0.25)", line_width=1, row=2, col=1)

                fig_ratio.update_layout(
                    template="plotly_dark",
                    height=520,
                    margin=dict(l=10, r=10, t=35, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0.0,
                        bgcolor="rgba(15, 23, 42, 0.7)",
                    ),
                    hovermode="x unified",
                )
                fig_ratio.update_yaxes(title_text="Ratio", row=1, col=1)
                fig_ratio.update_yaxes(title_text="Z-Score (σ)", row=2, col=1)
                st.plotly_chart(fig_ratio, use_container_width=True)

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                # -------------------------------------------------------------
                # 5. Row of 2 Charts: Risk & Drawdown Profiles
                # -------------------------------------------------------------
                rc1, rc2 = st.columns(2)

                with rc1:
                    st.markdown(
                        """
                        <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                            ⚡ Chart 3: Rolling 30-Day Realized Volatility (%)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    rets_a = ca_aligned.pct_change().dropna()
                    rets_b = cb_aligned.pct_change().dropna()
                    roll_vol_a = rets_a.rolling(30, min_periods=5).std() * np.sqrt(252) * 100.0
                    roll_vol_b = rets_b.rolling(30, min_periods=5).std() * np.sqrt(252) * 100.0

                    fig_vol = go.Figure()
                    fig_vol.add_trace(
                        go.Scatter(
                            x=roll_vol_a.index,
                            y=roll_vol_a,
                            mode="lines",
                            name=f"{stock_a_resolved} Vol",
                            line=dict(color="#10B981", width=2),
                            hovertemplate=f"<b>{stock_a_resolved} Vol</b>: %{{y:.1f}}%<extra></extra>",
                        )
                    )
                    fig_vol.add_trace(
                        go.Scatter(
                            x=roll_vol_b.index,
                            y=roll_vol_b,
                            mode="lines",
                            name=f"{stock_b_resolved} Vol",
                            line=dict(color="#38BDF8", width=2),
                            hovertemplate=f"<b>{stock_b_resolved} Vol</b>: %{{y:.1f}}%<extra></extra>",
                        )
                    )
                    fig_vol.update_layout(
                        template="plotly_dark",
                        height=370,
                        margin=dict(l=10, r=10, t=25, b=20),
                        yaxis_title="Annualized Volatility (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

                with rc2:
                    st.markdown(
                        """
                        <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                            🌊 Chart 4: Underwater Peak-to-Trough Drawdown (%)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    dd_curve_a = ((ca_aligned - ca_aligned.cummax()) / ca_aligned.cummax()) * 100.0
                    dd_curve_b = ((cb_aligned - cb_aligned.cummax()) / cb_aligned.cummax()) * 100.0

                    fig_dd = go.Figure()
                    fig_dd.add_trace(
                        go.Scatter(
                            x=dd_curve_a.index,
                            y=dd_curve_a,
                            mode="lines",
                            fill="tozeroy",
                            name=f"{stock_a_resolved} Drawdown",
                            line=dict(color="#10B981", width=1.5),
                            fillcolor="rgba(16, 185, 129, 0.15)",
                            hovertemplate=f"<b>{stock_a_resolved} DD</b>: %{{y:.2f}}%<extra></extra>",
                        )
                    )
                    fig_dd.add_trace(
                        go.Scatter(
                            x=dd_curve_b.index,
                            y=dd_curve_b,
                            mode="lines",
                            fill="tozeroy",
                            name=f"{stock_b_resolved} Drawdown",
                            line=dict(color="#38BDF8", width=1.5),
                            fillcolor="rgba(56, 189, 248, 0.15)",
                            hovertemplate=f"<b>{stock_b_resolved} DD</b>: %{{y:.2f}}%<extra></extra>",
                        )
                    )
                    fig_dd.update_layout(
                        template="plotly_dark",
                        height=370,
                        margin=dict(l=10, r=10, t=25, b=20),
                        yaxis_title="Drawdown (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                # -------------------------------------------------------------
                # 6. Row of 2 Charts: Distributions & Correlation Regression
                # -------------------------------------------------------------
                dc1, dc2 = st.columns(2)

                with dc1:
                    st.markdown(
                        """
                        <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                            📊 Chart 5: Daily Return Density Distribution
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    fig_dist = go.Figure()
                    fig_dist.add_trace(
                        go.Histogram(
                            x=rets_a * 100.0,
                            name=f"{stock_a_resolved}",
                            opacity=0.6,
                            marker=dict(color="#10B981"),
                            nbinsx=35,
                            hovertemplate=f"<b>{stock_a_resolved}</b>: %{{x:.2f}}%<br>Count: %{{y}}<extra></extra>",
                        )
                    )
                    fig_dist.add_trace(
                        go.Histogram(
                            x=rets_b * 100.0,
                            name=f"{stock_b_resolved}",
                            opacity=0.6,
                            marker=dict(color="#38BDF8"),
                            nbinsx=35,
                            hovertemplate=f"<b>{stock_b_resolved}</b>: %{{x:.2f}}%<br>Count: %{{y}}<extra></extra>",
                        )
                    )
                    fig_dist.update_layout(
                        barmode="overlay",
                        template="plotly_dark",
                        height=370,
                        margin=dict(l=10, r=10, t=25, b=20),
                        xaxis_title="Daily Return (%)",
                        yaxis_title="Trading Days Frequency",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                with dc2:
                    st.markdown(
                        """
                        <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                            🎯 Chart 6: Return Scatter & OLS Linear Regression
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    scatter_df = pd.DataFrame({
                        stock_a_resolved: rets_a * 100.0,
                        stock_b_resolved: rets_b * 100.0,
                    }).dropna()

                    corr_val = float(rets_a.corr(rets_b)) if len(rets_a) > 2 else 0.0

                    # Calculate regression stats
                    if len(scatter_df) > 5:
                        slope, intercept = np.polyfit(scatter_df[stock_b_resolved], scatter_df[stock_a_resolved], 1)
                        r_squared = corr_val ** 2
                    else:
                        slope, intercept, r_squared = 1.0, 0.0, 0.0

                    fig_scatter = px.scatter(
                        scatter_df,
                        x=stock_b_resolved,
                        y=stock_a_resolved,
                        trendline="ols",
                        template="plotly_dark",
                        labels={
                            stock_b_resolved: f"{stock_b_resolved} Return (%)",
                            stock_a_resolved: f"{stock_a_resolved} Return (%)",
                        },
                        title=f"Pearson r = {corr_val:.3f} | Beta (Slope) = {slope:.2f} | R² = {r_squared:.3f}",
                    )
                    fig_scatter.update_traces(
                        marker=dict(size=6, color="rgba(56, 189, 248, 0.65)", line=dict(width=1, color="#38BDF8"))
                    )
                    fig_scatter.update_layout(
                        height=370,
                        margin=dict(l=10, r=10, t=35, b=20),
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                # -------------------------------------------------------------
                # 7. Chart 7: Relative Institutional Factor Radar Scorecard
                # -------------------------------------------------------------
                st.markdown(
                    """
                    <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                        🧭 Chart 7: Relative Institutional Factor Radar Scorecard
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Helper to compute normalized 0-100 scores for radar
                def score_factor(val_a, val_b, higher_is_better=True):
                    if val_a is None and val_b is None:
                        return 50.0, 50.0
                    va = float(val_a) if val_a is not None and not np.isnan(val_a) else 0.0
                    vb = float(val_b) if val_b is not None and not np.isnan(val_b) else 0.0
                    if abs(va - vb) < 1e-6:
                        return 50.0, 50.0
                    if higher_is_better:
                        diff = va - vb
                        base = max(abs(va), abs(vb), 1.0)
                        rel = (diff / base) * 35.0
                        sa = max(10.0, min(95.0, 50.0 + rel))
                        sb = max(10.0, min(95.0, 50.0 - rel))
                        return sa, sb
                    else:
                        diff = vb - va
                        base = max(abs(va), abs(vb), 1.0)
                        rel = (diff / base) * 35.0
                        sa = max(10.0, min(95.0, 50.0 + rel))
                        sb = max(10.0, min(95.0, 50.0 - rel))
                        return sa, sb

                # Factor 1: Valuation (Lower P/E is better)
                v_score_a, v_score_b = score_factor(pe_disp_a, pe_disp_b, higher_is_better=False)
                # Factor 2: Momentum (Period return)
                m_score_a, m_score_b = score_factor(ret_a_tot, ret_b_tot, higher_is_better=True)
                # Factor 3: Growth (EPS YoY growth)
                eps_g_a = snap_a.get("eps_growth") if snap_a else None
                eps_g_b = snap_b.get("eps_growth") if snap_b else None
                g_score_a, g_score_b = score_factor(eps_g_a, eps_g_b, higher_is_better=True)
                # Factor 4: Quality (Profit margin or ROE)
                pm_a = info_a.get("profitMargins")
                pm_b = info_b.get("profitMargins")
                q_score_a, q_score_b = score_factor(pm_a, pm_b, higher_is_better=True)
                # Factor 5: Low Volatility (Inverse volatility)
                vol_val_a = rets_a.std() * np.sqrt(252) if len(rets_a) > 2 else 0.2
                vol_val_b = rets_b.std() * np.sqrt(252) if len(rets_b) > 2 else 0.2
                lv_score_a, lv_score_b = score_factor(vol_val_a, vol_val_b, higher_is_better=False)
                # Factor 6: Market Scale (Market Cap)
                s_score_a, s_score_b = score_factor(mcap_disp_a, mcap_disp_b, higher_is_better=True)

                radar_categories = [
                    "Valuation Attractiveness",
                    "Return Momentum",
                    "Earnings Growth",
                    "Quality & Margins",
                    "Capital Protection (Low Vol)",
                    "Market Scale & Liquidity",
                ]

                fig_radar = go.Figure()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[v_score_a, m_score_a, g_score_a, q_score_a, lv_score_a, s_score_a],
                        theta=radar_categories,
                        fill="toself",
                        name=f"{stock_a_resolved}",
                        line=dict(color="#10B981", width=2),
                        fillcolor="rgba(16, 185, 129, 0.2)",
                    )
                )
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[v_score_b, m_score_b, g_score_b, q_score_b, lv_score_b, s_score_b],
                        theta=radar_categories,
                        fill="toself",
                        name=f"{stock_b_resolved}",
                        line=dict(color="#38BDF8", width=2),
                        fillcolor="rgba(56, 189, 248, 0.2)",
                    )
                )
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#94A3B8")),
                        angularaxis=dict(tickfont=dict(size=11, color="#E2E8F0")),
                        bgcolor="rgba(15, 23, 42, 0.5)",
                    ),
                    template="plotly_dark",
                    height=460,
                    margin=dict(l=40, r=40, t=30, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

    # =========================================================================
    # TAB 3: TECHNICALS SCREENER (MULTI-INDICATOR)
    # =========================================================================
    with tab_tech_screener:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="margin-bottom: 12px;">
                <span style="font-size: 0.86rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.06em; text-transform: uppercase;">
                    📈 Multi-Indicator Algorithmic Technical Screener
                </span>
                <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #94A3B8;">
                    Filter institutional universe across RSI momentum, MACD crossovers, Moving Average alignments, Bollinger Band squeezes, and volume breakouts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Technical Presets Chips / Radio
        tech_preset_options = [
            "All Stocks (Custom Technicals)",
            "🚀 Bullish Momentum Expansion (RSI 55-75 + SMA 50 Cross)",
            "📈 Golden Cross Breakout (50 SMA > 200 SMA)",
            "📉 Oversold Mean Reversion (RSI < 30 + Stoch < 25)",
            "⚡ Unusual Volume Surge & Breakout (RVOL > 1.5x + 1D Chg > 0)",
            "🎯 Bollinger Band Squeeze (Low Volatility Compression)",
            "🔥 Strong Technical Consensus (Recommend.All >= 0.3)",
            "🌊 Multi-SMA Bullish Alignment (Price > 20 > 50 > 200)",
            "💎 Oversold Bounce at Lower Bollinger Band",
        ]
        sel_tech_preset = st.radio(
            "Quantitative Technical Playbook Presets:",
            options=tech_preset_options,
            horizontal=True,
            key="tech_preset_radio",
        )

        # Interactive Technical Filter Controls
        with st.expander("🛠️ Advanced Technical Conditions & Indicator Thresholds", expanded=True):
            tc1, tc2, tc3 = st.columns([1.2, 1.2, 1.6])
            with tc1:
                tech_mkt_choice = st.selectbox(
                    "Market",
                    ["India (NSE/BSE)", "US (NASDAQ/NYSE/AMEX)"],
                    index=0,
                    key="tech_mkt_sel",
                )
                tech_market_val = "india" if "India" in tech_mkt_choice else "america"

            with tc2:
                if tech_market_val == "india":
                    tech_exchange_choice = st.selectbox(
                        "Primary Exchange",
                        ["All", "NSE", "BSE"],
                        index=1,
                        key="tech_ex_sel_in",
                    )
                else:
                    tech_exchange_choice = st.selectbox(
                        "Primary Exchange",
                        ["All", "NASDAQ", "NYSE", "AMEX"],
                        index=0,
                        key="tech_ex_sel_us",
                    )

            with tc3:
                tech_rating_filter = st.selectbox(
                    "Technical Rating Consensus",
                    [
                        "Any",
                        "Buy or Strong Buy (Rating > 0.1)",
                        "Strong Buy Only (Rating > 0.3)",
                        "Sell or Strong Sell (Rating < -0.1)",
                    ],
                    index=0,
                    key="tech_rating_sel",
                )

            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            # Oscillators Row
            osc_c1, osc_c2, osc_c3 = st.columns(3)
            with osc_c1:
                r_c1, r_c2 = st.columns(2)
                with r_c1:
                    min_rsi_in = float(st.number_input("Min RSI (14)", value=0.0, min_value=0.0, max_value=100.0, step=5.0, key="tech_rsi_min"))
                with r_c2:
                    max_rsi_in = float(st.number_input("Max RSI (14)", value=100.0, min_value=0.0, max_value=100.0, step=5.0, key="tech_rsi_max"))

            with osc_c2:
                macd_cond_in = st.selectbox(
                    "MACD (12, 26, 9) Condition",
                    [
                        "Any",
                        "Bullish Crossover (MACD > Signal)",
                        "Bearish Crossover (MACD < Signal)",
                        "MACD Positive (MACD > 0)",
                        "MACD Negative (MACD < 0)",
                    ],
                    index=0,
                    key="tech_macd_sel",
                )

            with osc_c3:
                stoch_cond_in = st.selectbox(
                    "Stochastic (14, 3) Condition",
                    [
                        "Any",
                        "Oversold (Stoch.K < 20)",
                        "Overbought (Stoch.K > 80)",
                        "Bullish Stoch (K > D)",
                    ],
                    index=0,
                    key="tech_stoch_sel",
                )

            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            # Trend, Bands & Volume Row
            tb_c1, tb_c2, tb_c3, tb_c4 = st.columns(4)
            with tb_c1:
                ma_cond_in = st.selectbox(
                    "Moving Average Structure",
                    [
                        "Any",
                        "Price > SMA 20",
                        "Price > SMA 50",
                        "Price > SMA 200",
                        "Price > SMA 50 & 200",
                        "Golden Cross (SMA 50 > SMA 200)",
                        "Multi-SMA Stack (Price > 20 > 50 > 200)",
                        "Price < SMA 200 (Discount / Bearish)",
                    ],
                    index=0,
                    key="tech_ma_sel",
                )

            with tb_c2:
                bb_cond_in = st.selectbox(
                    "Bollinger Bands (20, 2σ)",
                    [
                        "Any",
                        "Price Touching/Below Lower Band",
                        "Price Touching/Above Upper Band",
                    ],
                    index=0,
                    key="tech_bb_sel",
                )

            with tb_c3:
                min_rvol_in = float(st.number_input("Min Relative Volume (RVOL)", value=0.0, min_value=0.0, max_value=20.0, step=0.5, key="tech_rvol_in"))

            with tb_c4:
                min_chg_tech = float(st.number_input("Min 1-Day Return %", value=-100.0, min_value=-100.0, max_value=100.0, step=1.0, key="tech_min_chg"))

            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            # Sort & Limits
            so_c1, so_c2, so_c3 = st.columns([1.5, 1.0, 1.0])
            with so_c1:
                sort_choice_tech = st.selectbox(
                    "Rank Results By",
                    [
                        "RSI",
                        "change",
                        "relative_volume_10d_calc",
                        "Recommend.All",
                        "volume",
                        "close",
                    ],
                    format_func=lambda x: {
                        "RSI": "RSI (14) Momentum",
                        "change": "1-Day Price Change (%)",
                        "relative_volume_10d_calc": "Relative Volume (RVOL)",
                        "Recommend.All": "Overall Technical Rating",
                        "volume": "Trading Volume",
                        "close": "Close Price",
                    }.get(x, x),
                    index=0,
                    key="tech_sort_choice",
                )
            with so_c2:
                sort_dir_tech = st.radio("Sort Order", ["Descending", "Ascending"], horizontal=True, key="tech_sort_dir")
                is_asc_tech = sort_dir_tech == "Ascending"
            with so_c3:
                tech_limit = st.selectbox("Max Display Stocks", [25, 50, 100, 200], index=1, key="tech_limit_sel")

        # Execute Technical Query
        with st.spinner("Screening global technical universe via TradingView Engine…"):
            tech_count, tech_df = run_tradingview_technical_screener(
                market=tech_market_val,
                exchange=tech_exchange_choice,
                preset=sel_tech_preset,
                rsi_min=min_rsi_in,
                rsi_max=max_rsi_in,
                ma_filter=ma_cond_in,
                macd_filter=macd_cond_in,
                stoch_filter=stoch_cond_in,
                bb_filter=bb_cond_in,
                min_rvol=min_rvol_in,
                min_change=min_chg_tech,
                tech_rating=tech_rating_filter,
                sort_by=sort_choice_tech,
                sort_asc=is_asc_tech,
                limit=tech_limit,
            )

        if tech_df.empty:
            st.info("No stocks match the exact technical screening conditions. Try relaxing the RSI or Moving Average thresholds.")
        else:
            # Format Technical Rating Helper
            def _fmt_rating(val):
                if pd.isna(val):
                    return "—"
                v = float(val)
                if v >= 0.5:
                    return "🚀 Strong Buy"
                elif v >= 0.1:
                    return "🟢 Buy"
                elif v <= -0.5:
                    return "🚨 Strong Sell"
                elif v <= -0.1:
                    return "🔴 Sell"
                else:
                    return "⚖️ Neutral"

            # KPI Summary Badges
            tk1, tk2, tk3, tk4, tk5 = st.columns(5)
            bullish_count = int((tech_df["Recommend.All"] > 0.1).sum()) if "Recommend.All" in tech_df.columns else 0
            bearish_count = int((tech_df["Recommend.All"] < -0.1).sum()) if "Recommend.All" in tech_df.columns else 0
            med_rsi_val = float(tech_df["RSI"].dropna().median()) if "RSI" in tech_df.columns and len(tech_df["RSI"].dropna()) > 0 else 50.0
            top_mom_stock = tech_df.sort_values("change", ascending=False).iloc[0]
            max_rvol_stock = tech_df.sort_values("relative_volume_10d_calc", ascending=False).iloc[0] if "relative_volume_10d_calc" in tech_df.columns else top_mom_stock

            tk1.metric("Screened Matches", f"{tech_count:,}")
            tk2.metric("Bullish Consensus", f"{bullish_count:,}", f"{(bullish_count / max(len(tech_df), 1) * 100):.0f}% of view")
            tk3.metric("Bearish Consensus", f"{bearish_count:,}", f"{(bearish_count / max(len(tech_df), 1) * 100):.0f}% of view")
            tk4.metric("Median RSI (14)", f"{med_rsi_val:.1f}")
            tk5.metric("Top RVOL Surge", f"{max_rvol_stock['name']}", f"{max_rvol_stock.get('relative_volume_10d_calc', 0):.1f}x Vol")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Format DataFrame for Display
            disp_tech_df = pd.DataFrame()
            disp_tech_df["Symbol"] = tech_df["name"]
            disp_tech_df["Company Name"] = tech_df["description"]
            disp_tech_df["Price"] = tech_df["close"].map(lambda x: f"{x:,.2f}")
            disp_tech_df["1D Change %"] = tech_df["change"].map(lambda x: f"{x:+.2f}%")
            disp_tech_df["Technical Verdict"] = tech_df["Recommend.All"].map(_fmt_rating)
            disp_tech_df["RSI (14)"] = tech_df["RSI"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
            disp_tech_df["MACD Status"] = tech_df.apply(
                lambda r: "🟢 Bullish" if pd.notna(r.get("MACD.macd")) and pd.notna(r.get("MACD.signal")) and r["MACD.macd"] > r["MACD.signal"] else "🔴 Bearish",
                axis=1,
            )
            disp_tech_df["SMA 50 Status"] = tech_df.apply(
                lambda r: "Above" if pd.notna(r.get("SMA50")) and r["close"] > r["SMA50"] else "Below",
                axis=1,
            )
            disp_tech_df["SMA 200 Status"] = tech_df.apply(
                lambda r: "Above" if pd.notna(r.get("SMA200")) and r["close"] > r["SMA200"] else "Below",
                axis=1,
            )
            disp_tech_df["Relative Vol (RVOL)"] = tech_df["relative_volume_10d_calc"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
            disp_tech_df["ATR (14)"] = tech_df["ATR"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            disp_tech_df["Volume"] = tech_df["volume"].map(lambda x: _fmt_num(x) if pd.notna(x) else "—")
            disp_tech_df["Exchange"] = tech_df.get("exchange", "NSE")

            st.dataframe(
                disp_tech_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Company Name": st.column_config.TextColumn("Company", width="medium"),
                    "Price": st.column_config.TextColumn("Price", width="small"),
                    "1D Change %": st.column_config.TextColumn("1D %", width="small"),
                    "Technical Verdict": st.column_config.TextColumn("Technical Verdict", width="medium"),
                    "RSI (14)": st.column_config.TextColumn("RSI", width="small"),
                    "MACD Status": st.column_config.TextColumn("MACD", width="small"),
                    "Relative Vol (RVOL)": st.column_config.TextColumn("RVOL", width="small"),
                    "Volume": st.column_config.TextColumn("Volume", width="small"),
                },
            )

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            # Visual Analytics Plots for Screener
            tp_c1, tp_c2 = st.columns(2)
            with tp_c1:
                st.markdown(
                    """
                    <div style="font-size: 0.84rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 6px;">
                        📊 Market Breadth: RSI (14) Momentum Distribution
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_rsi_dist = px.histogram(
                    tech_df,
                    x="RSI",
                    nbins=25,
                    template="plotly_dark",
                    color_discrete_sequence=["#38BDF8"],
                    labels={"RSI": "RSI (14)", "count": "Stock Count"},
                )
                fig_rsi_dist.add_vline(x=70, line_dash="dash", line_color="#EF4444", annotation_text="Overbought 70")
                fig_rsi_dist.add_vline(x=30, line_dash="dash", line_color="#10B981", annotation_text="Oversold 30")
                fig_rsi_dist.add_vline(x=50, line_dash="dot", line_color="rgba(255,255,255,0.4)")
                fig_rsi_dist.update_layout(height=290, margin=dict(l=10, r=10, t=25, b=20))
                st.plotly_chart(fig_rsi_dist, use_container_width=True)

            with tp_c2:
                st.markdown(
                    """
                    <div style="font-size: 0.84rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 6px;">
                        🎯 Technical Rating Consensus Breakdown
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                verdict_counts = disp_tech_df["Technical Verdict"].value_counts().reset_index()
                verdict_counts.columns = ["Verdict", "Count"]
                color_map = {
                    "🚀 Strong Buy": "#10B981",
                    "🟢 Buy": "#34D399",
                    "⚖️ Neutral": "#94A3B8",
                    "🔴 Sell": "#F87171",
                    "🚨 Strong Sell": "#EF4444",
                }
                fig_rating_pie = px.pie(
                    verdict_counts,
                    names="Verdict",
                    values="Count",
                    hole=0.45,
                    template="plotly_dark",
                    color="Verdict",
                    color_discrete_map=color_map,
                )
                fig_rating_pie.update_layout(height=290, margin=dict(l=10, r=10, t=25, b=20))
                st.plotly_chart(fig_rating_pie, use_container_width=True)

            # Actions & Tab 4 Integration
            act_t1, act_t2 = st.columns([3, 1])
            with act_t1:
                available_tech_tickers = []
                for _, r in tech_df.iterrows():
                    sym = r["name"]
                    ex_code = r.get("exchange", "NSE")
                    if ex_code == "NSE":
                        available_tech_tickers.append(f"{sym}.NS")
                    elif ex_code == "BSE":
                        available_tech_tickers.append(f"{sym}.BO")
                    else:
                        available_tech_tickers.append(sym)

                st.markdown(
                    "<span style='color: #38BDF8; font-weight: 600; font-size: 0.88rem;'>⚡ Quick Select for Technicals Comparison (Tab 4)</span>",
                    unsafe_allow_html=True,
                )
                sel_pair_tech = st.multiselect(
                    "Pick any 2 screened stocks to send directly to the Technicals Comparison tab:",
                    options=available_tech_tickers,
                    default=available_tech_tickers[:2] if len(available_tech_tickers) >= 2 else available_tech_tickers,
                    max_selections=2,
                    key="tech_h2h_quick_select",
                )
                if len(sel_pair_tech) == 2:
                    if st.button(
                        f"🚀 Compare {sel_pair_tech[0]} vs {sel_pair_tech[1]} in Tab 4",
                        key="btn_send_tech_compare",
                    ):
                        st.session_state["tech_stock_a"] = sel_pair_tech[0]
                        st.session_state["tech_stock_b"] = sel_pair_tech[1]
                        st.success(f"Configured technical comparison: {sel_pair_tech[0]} vs {sel_pair_tech[1]}! Click Tab 4 to view.")

            with act_t2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                csv_bytes_tech = disp_tech_df.to_csv(index=False)
                st.download_button(
                    "📥 Export Technicals CSV",
                    csv_bytes_tech,
                    file_name=f"technicals_screener_{tech_market_val}.csv",
                    mime="text/csv",
                    key="btn_dl_tech_csv",
                )

    # =========================================================================
    # TAB 4: TECHNICALS COMPARISON (CHARTS & INDICATORS)
    # =========================================================================
    with tab_tech_compare:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="margin-bottom: 12px;">
                <span style="font-size: 0.86rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.06em; text-transform: uppercase;">
                    ⚡ Deep-Dive Technicals Comparison (Charts & Indicators)
                </span>
                <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #94A3B8;">
                    Head-to-head quantitative comparison of two assets across trend structures, momentum velocity, volatility envelopes, and institutional volume flows.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Technical Selection Controls
        t_scope_col, t_period_col, t_freq_col = st.columns([2.2, 1.0, 1.0])
        with t_scope_col:
            t_comp_scope = st.radio(
                "Stock Universe Scope",
                ["All Markets (18,500+ India & US)", "India Stocks (7,600+ NSE & BSE)", "US Stocks (10,900+ NASDAQ/NYSE/AMEX)"],
                horizontal=True,
                key="tech_h2h_univ_scope",
            )
        with t_period_col:
            t_comp_period = st.selectbox(
                "Historical Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "ytd"],
                index=2,
                key="tech_h2h_period",
            )
        with t_freq_col:
            t_comp_interval = st.selectbox(
                "Frequency",
                ["1d", "1wk"],
                index=0,
                key="tech_h2h_freq",
            )

        # Select option pool
        if "India Stocks" in t_comp_scope:
            t_active_options = snapshots["india_labels"] if snapshots["india_labels"] else ["RELIANCE (RELIANCE.NS · NSE)"]
            t_default_a = "RELIANCE.NS"
            t_default_b = "TCS.NS"
        elif "US Stocks" in t_comp_scope:
            t_active_options = snapshots["us_labels"] if snapshots["us_labels"] else ["Apple Inc. (AAPL · NASDAQ)"]
            t_default_a = "AAPL"
            t_default_b = "MSFT"
        else:
            t_active_options = snapshots["all_labels"] if snapshots["all_labels"] else ["RELIANCE (RELIANCE.NS · NSE)"]
            t_default_a = "RELIANCE.NS"
            t_default_b = "TCS.NS"

        # Guard against stale session state across universe switches
        if st.session_state.get("prev_tech_h2h_scope") != t_comp_scope:
            st.session_state["prev_tech_h2h_scope"] = t_comp_scope
            if "tech_select_label_a" in st.session_state and st.session_state["tech_select_label_a"] not in t_active_options:
                st.session_state.pop("tech_select_label_a", None)
            if "tech_select_label_b" in st.session_state and st.session_state["tech_select_label_b"] not in t_active_options:
                st.session_state.pop("tech_select_label_b", None)

        stored_ta = st.session_state.get("tech_stock_a", t_default_a).upper()
        t_idx_a = 0
        for i, opt in enumerate(t_active_options):
            if f"({stored_ta} ·" in opt.upper() or opt.upper().startswith(f"{stored_ta} "):
                t_idx_a = i
                break

        stored_tb = st.session_state.get("tech_stock_b", t_default_b).upper()
        t_idx_b = min(1, len(t_active_options) - 1)
        for i, opt in enumerate(t_active_options):
            if f"({stored_tb} ·" in opt.upper() or opt.upper().startswith(f"{stored_tb} "):
                t_idx_b = i
                break

        # Dropdowns
        tsel_c1, tsel_c2 = st.columns(2)
        with tsel_c1:
            t_chosen_label_a = st.selectbox(
                "Candidate Stock A (Search by name or ticker)",
                options=t_active_options,
                index=t_idx_a,
                key="tech_select_label_a",
            )
        with tsel_c2:
            t_chosen_label_b = st.selectbox(
                "Candidate Stock B (Search by name or ticker)",
                options=t_active_options,
                index=t_idx_b,
                key="tech_select_label_b",
            )

        with st.expander("⚙️ Advanced: Enter Custom Ticker Override (Optional)"):
            tc_ov1, tc_ov2 = st.columns(2)
            with tc_ov1:
                t_override_a = st.text_input(
                    "Override Stock A (e.g. BTC-USD, SPY, ^NSEI)",
                    value="",
                    key="tech_override_sym_a",
                ).strip().upper()
            with tc_ov2:
                t_override_b = st.text_input(
                    "Override Stock B (e.g. ETH-USD, QQQ, ^GSPC)",
                    value="",
                    key="tech_override_sym_b",
                ).strip().upper()

        snap_ta = snapshots["by_label"].get(t_chosen_label_a)
        if t_override_a:
            t_stock_a = t_override_a
            snap_ta = snapshots["by_ticker"].get(t_override_a, snap_ta)
        elif snap_ta:
            t_stock_a = snap_ta["yf_ticker"]
        else:
            t_stock_a = "RELIANCE.NS"

        snap_tb = snapshots["by_label"].get(t_chosen_label_b)
        if t_override_b:
            t_stock_b = t_override_b
            snap_tb = snapshots["by_ticker"].get(t_override_b, snap_tb)
        elif snap_tb:
            t_stock_b = snap_tb["yf_ticker"]
        else:
            t_stock_b = "TCS.NS"

        st.session_state["tech_stock_a"] = t_stock_a
        st.session_state["tech_stock_b"] = t_stock_b

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        if not t_stock_a or not t_stock_b:
            st.info("Please specify both Stock A and Stock B above.")
        elif t_stock_a == t_stock_b:
            st.warning("Please choose two distinct stocks to perform a technical comparison.")
        else:
            with st.spinner(f"Downloading high-resolution OHLCV series for {t_stock_a} and {t_stock_b}…"):
                t_raw = yf.download(
                    [t_stock_a, t_stock_b],
                    period=t_comp_period,
                    interval=t_comp_interval,
                    auto_adjust=True,
                    progress=False,
                )

            if t_raw.empty:
                st.error("Unable to load OHLCV price history for the selected tickers. Please verify the symbols.")
            else:
                # Extract individual OHLCVs
                try:
                    if isinstance(t_raw.columns, pd.MultiIndex):
                        df_ohlcv_a = pd.DataFrame({
                            "Open": t_raw["Open"][t_stock_a].dropna(),
                            "High": t_raw["High"][t_stock_a].dropna(),
                            "Low": t_raw["Low"][t_stock_a].dropna(),
                            "Close": t_raw["Close"][t_stock_a].dropna(),
                            "Volume": t_raw["Volume"][t_stock_a].dropna() if "Volume" in t_raw else pd.Series(1, index=t_raw.index),
                        }).dropna()

                        df_ohlcv_b = pd.DataFrame({
                            "Open": t_raw["Open"][t_stock_b].dropna(),
                            "High": t_raw["High"][t_stock_b].dropna(),
                            "Low": t_raw["Low"][t_stock_b].dropna(),
                            "Close": t_raw["Close"][t_stock_b].dropna(),
                            "Volume": t_raw["Volume"][t_stock_b].dropna() if "Volume" in t_raw else pd.Series(1, index=t_raw.index),
                        }).dropna()
                    else:
                        df_ohlcv_a = t_raw
                        df_ohlcv_b = t_raw
                except Exception as e:
                    st.error(f"Error structuring technical data: {e}")
                    return

                if len(df_ohlcv_a) < 5 or len(df_ohlcv_b) < 5:
                    st.error("Insufficient overlapping price bars to compute reliable technical indicators.")
                    return

                # Compute Technical Feature sets
                feat_a = compute_technical_features(df_ohlcv_a)
                feat_b = compute_technical_features(df_ohlcv_b)

                name_ta = (snap_ta.get("name") if snap_ta else None) or t_stock_a
                name_tb = (snap_tb.get("name") if snap_tb else None) or t_stock_b
                curr_ta = "₹" if (snap_ta and snap_ta.get("currency") == "INR") else "$"
                curr_tb = "₹" if (snap_tb and snap_tb.get("currency") == "INR") else "$"

                # -------------------------------------------------------------
                # 1. Executive Side-by-Side Technical Scorecards
                # -------------------------------------------------------------
                t_card1, t_card2 = st.columns(2)

                with t_card1:
                    st.markdown(
                        f"""
                        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 18px 22px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.76rem; font-weight: 700; color: #10B981; letter-spacing: 0.08em; text-transform: uppercase;">Technical Candidate A</span>
                                <span style="font-size: 0.78rem; background: rgba(16, 185, 129, 0.25); color: #10B981; padding: 3px 10px; border-radius: 6px; font-weight: 700;">{feat_a.get('overall_signal', 'Neutral')}</span>
                            </div>
                            <div style="font-size: 1.55rem; font-weight: 700; color: #FFFFFF; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name_ta}</div>
                            <div style="color: #38BDF8; font-size: 0.86rem; font-weight: 500;">{t_stock_a} · {snap_ta.get('exchange', 'NSE') if snap_ta else 'EQUITY'}</div>
                            <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 10px;">
                                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.75rem; font-weight: 700; color: #FFFFFF;">{curr_ta}{feat_a.get('last_close', 0):,.2f}</span>
                                <span style="font-size: 1.05rem; font-weight: 600; color: {'#00E676' if feat_a.get('ret_1d', 0) >= 0 else '#EF4444'};">1D: {feat_a.get('ret_1d', 0):+.2f}%</span>
                                <span style="font-size: 0.95rem; font-weight: 600; color: #38BDF8;">Tech Score: {feat_a.get('overall_score', 50):.1f}/100</span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                                <div><span style="color: #94A3B8;">Trend Regime:</span> <br><strong>{feat_a.get('trend_regime', '—')}</strong></div>
                                <div><span style="color: #94A3B8;">RSI (14):</span> <br><strong>{feat_a.get('rsi', 50):.1f}</strong></div>
                                <div><span style="color: #94A3B8;">MACD Hist:</span> <br><strong>{feat_a.get('macd_hist', 0):+.2f}</strong></div>
                                <div><span style="color: #94A3B8;">Bollinger Width:</span> <br><strong>{feat_a.get('bb_width', 0):.2f}%</strong></div>
                                <div><span style="color: #94A3B8;">ATR Volatility:</span> <br><strong>{feat_a.get('atr_pct', 0):.2f}%</strong></div>
                                <div><span style="color: #94A3B8;">Money Flow Index:</span> <br><strong>{feat_a.get('mfi', 50):.1f}</strong></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with t_card2:
                    st.markdown(
                        f"""
                        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 18px 22px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.76rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.08em; text-transform: uppercase;">Technical Candidate B</span>
                                <span style="font-size: 0.78rem; background: rgba(56, 189, 248, 0.25); color: #38BDF8; padding: 3px 10px; border-radius: 6px; font-weight: 700;">{feat_b.get('overall_signal', 'Neutral')}</span>
                            </div>
                            <div style="font-size: 1.55rem; font-weight: 700; color: #FFFFFF; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name_tb}</div>
                            <div style="color: #38BDF8; font-size: 0.86rem; font-weight: 500;">{t_stock_b} · {snap_tb.get('exchange', 'NSE') if snap_tb else 'EQUITY'}</div>
                            <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 10px;">
                                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.75rem; font-weight: 700; color: #FFFFFF;">{curr_tb}{feat_b.get('last_close', 0):,.2f}</span>
                                <span style="font-size: 1.05rem; font-weight: 600; color: {'#00E676' if feat_b.get('ret_1d', 0) >= 0 else '#EF4444'};">1D: {feat_b.get('ret_1d', 0):+.2f}%</span>
                                <span style="font-size: 0.95rem; font-weight: 600; color: #38BDF8;">Tech Score: {feat_b.get('overall_score', 50):.1f}/100</span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                                <div><span style="color: #94A3B8;">Trend Regime:</span> <br><strong>{feat_b.get('trend_regime', '—')}</strong></div>
                                <div><span style="color: #94A3B8;">RSI (14):</span> <br><strong>{feat_b.get('rsi', 50):.1f}</strong></div>
                                <div><span style="color: #94A3B8;">MACD Hist:</span> <br><strong>{feat_b.get('macd_hist', 0):+.2f}</strong></div>
                                <div><span style="color: #94A3B8;">Bollinger Width:</span> <br><strong>{feat_b.get('bb_width', 0):.2f}%</strong></div>
                                <div><span style="color: #94A3B8;">ATR Volatility:</span> <br><strong>{feat_b.get('atr_pct', 0):.2f}%</strong></div>
                                <div><span style="color: #94A3B8;">Money Flow Index:</span> <br><strong>{feat_b.get('mfi', 50):.1f}</strong></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                # -------------------------------------------------------------
                # 2. Detailed Technical Features Comparison Matrix Dataframe
                # -------------------------------------------------------------
                st.markdown(
                    """
                    <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                        📊 Deep-Dive Technical Features Head-to-Head Comparison Matrix
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                tech_matrix_df = build_technical_comparison_matrix(feat_a, feat_b, t_stock_a, t_stock_b)

                # Dimension Filter & Quick Search
                t_filt_col1, t_filt_col2, t_filt_col3 = st.columns([1.6, 1.4, 1.0])
                with t_filt_col1:
                    filter_dim = st.selectbox(
                        "Technical Dimension Pillar",
                        [
                            "All Dimensions",
                            "Trend & Moving Averages",
                            "Momentum & Oscillators",
                            "Volatility & Bands",
                            "Volume Flow & Pressure",
                            "Composite Technical Scores",
                        ],
                        index=0,
                        key="tech_dim_filter",
                    )
                with t_filt_col2:
                    tech_search = st.text_input(
                        "🔍 Filter Indicator",
                        placeholder="e.g. RSI, ADX, MACD, Squeeze, CMF...",
                        key="tech_search_input",
                    )
                display_tech_matrix = tech_matrix_df
                if filter_dim != "All Dimensions":
                    display_tech_matrix = display_tech_matrix[display_tech_matrix["Category"] == filter_dim]
                if tech_search.strip():
                    tq = tech_search.strip().lower()
                    display_tech_matrix = display_tech_matrix[
                        display_tech_matrix["Metric"].str.lower().str.contains(tq, na=False)
                        | display_tech_matrix["Signal / Context"].str.lower().str.contains(tq, na=False)
                    ]

                with t_filt_col3:
                    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                    csv_tech = display_tech_matrix.to_csv(index=False)
                    st.download_button(
                        "📥 Export CSV",
                        data=csv_tech,
                        file_name=f"{t_stock_a}_vs_{t_stock_b}_technical_matrix.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_tech_matrix",
                    )

                st.caption(f"Showing **{len(display_tech_matrix)}** technical indicators and quantitative signals for **{t_stock_a}** vs **{t_stock_b}**")

                st.dataframe(
                    display_tech_matrix,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Category": st.column_config.TextColumn("Pillar Dimension", width="small"),
                        "Metric": st.column_config.TextColumn("Technical Indicator / Feature", width="medium"),
                        t_stock_a: st.column_config.TextColumn(f"🟢 {t_stock_a}", width="small"),
                        t_stock_b: st.column_config.TextColumn(f"🔵 {t_stock_b}", width="small"),
                        "Spread / Delta": st.column_config.TextColumn("Spread (A − B)", width="small"),
                        "Advantage": st.column_config.TextColumn("Technical Advantage", width="small"),
                        "Signal / Context": st.column_config.TextColumn("Institutional Reading", width="large"),
                    },
                )

                st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

                # -------------------------------------------------------------
                # 3. Technical Plots Suite
                # -------------------------------------------------------------
                st.markdown(
                    """
                    <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                        🕯️ Plot 1: Dual Interactive Candlestick / Price Overlays with SMAs & Bollinger Bands
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # View mode toggle
                overlay_view = st.radio(
                    "Display Mode:",
                    ["Side-by-Side (Both Assets)", f"Focus {t_stock_a}", f"Focus {t_stock_b}"],
                    horizontal=True,
                    key="tech_overlay_view_radio",
                )

                def _build_single_tech_chart(f_dict, sym_label, line_color):
                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.75, 0.25],
                        subplot_titles=(
                            f"{sym_label} · Candlestick + SMA 20/50/200 & Bollinger Bands",
                            "Trading Volume",
                        ),
                    )
                    # Candlestick
                    fig.add_trace(
                        go.Candlestick(
                            x=f_dict["series_c"].index,
                            open=f_dict["series_o"],
                            high=f_dict["series_h"],
                            low=f_dict["series_l"],
                            close=f_dict["series_c"],
                            name=f"{sym_label} Price",
                        ),
                        row=1,
                        col=1,
                    )
                    # SMAs
                    fig.add_trace(go.Scatter(x=f_dict["series_sma20"].index, y=f_dict["series_sma20"], mode="lines", name="20 SMA", line=dict(color="#F59E0B", width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=f_dict["series_sma50"].index, y=f_dict["series_sma50"], mode="lines", name="50 SMA", line=dict(color="#06B6D4", width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=f_dict["series_sma200"].index, y=f_dict["series_sma200"], mode="lines", name="200 SMA", line=dict(color="#A855F7", width=1.8)), row=1, col=1)
                    # Bollinger
                    fig.add_trace(go.Scatter(x=f_dict["series_bbu"].index, y=f_dict["series_bbu"], mode="lines", name="Upper BB", line=dict(color="rgba(255,255,255,0.4)", dash="dot", width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=f_dict["series_bbl"].index, y=f_dict["series_bbl"], mode="lines", name="Lower BB", line=dict(color="rgba(255,255,255,0.4)", dash="dot", width=1)), row=1, col=1)

                    # Volume
                    vol_colors = ["#10B981" if c >= o else "#EF4444" for c, o in zip(f_dict["series_c"], f_dict["series_o"])]
                    fig.add_trace(go.Bar(x=f_dict["series_v"].index, y=f_dict["series_v"], name="Volume", marker_color=vol_colors, opacity=0.6), row=2, col=1)

                    fig.update_layout(
                        template="plotly_dark",
                        height=480,
                        margin=dict(l=10, r=10, t=35, b=20),
                        xaxis_rangeslider_visible=False,
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    )
                    return fig

                if overlay_view == "Side-by-Side (Both Assets)":
                    oc1, oc2 = st.columns(2)
                    with oc1:
                        fig_a = _build_single_tech_chart(feat_a, t_stock_a, "#10B981")
                        st.plotly_chart(fig_a, use_container_width=True)
                    with oc2:
                        fig_b = _build_single_tech_chart(feat_b, t_stock_b, "#38BDF8")
                        st.plotly_chart(fig_b, use_container_width=True)
                elif f"Focus {t_stock_a}" in overlay_view:
                    fig_a = _build_single_tech_chart(feat_a, t_stock_a, "#10B981")
                    st.plotly_chart(fig_a, use_container_width=True)
                else:
                    fig_b = _build_single_tech_chart(feat_b, t_stock_b, "#38BDF8")
                    st.plotly_chart(fig_b, use_container_width=True)

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                # Plot 2: Momentum Oscillators Head-to-Head
                st.markdown(
                    """
                    <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                        🎯 Plot 2: Momentum Oscillators Head-to-Head (RSI & MACD Velocity)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                fig_mom = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    row_heights=[0.55, 0.45],
                    subplot_titles=(
                        f"Relative Strength Index (RSI 14) Comparison · Overbought (70) & Oversold (30)",
                        f"MACD Histogram Comparison (Momentum Velocity)",
                    ),
                )

                fig_mom.add_trace(go.Scatter(x=feat_a["series_rsi"].index, y=feat_a["series_rsi"], mode="lines", name=f"{t_stock_a} RSI ({feat_a['rsi']:.1f})", line=dict(color="#10B981", width=2)), row=1, col=1)
                fig_mom.add_trace(go.Scatter(x=feat_b["series_rsi"].index, y=feat_b["series_rsi"], mode="lines", name=f"{t_stock_b} RSI ({feat_b['rsi']:.1f})", line=dict(color="#38BDF8", width=2)), row=1, col=1)
                fig_mom.add_hline(y=70, line_dash="dash", line_color="#EF4444", line_width=1, row=1, col=1)
                fig_mom.add_hline(y=30, line_dash="dash", line_color="#10B981", line_width=1, row=1, col=1)
                fig_mom.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=1, col=1)

                fig_mom.add_trace(go.Bar(x=feat_a["series_macd_hist"].index, y=feat_a["series_macd_hist"], name=f"{t_stock_a} MACD Hist", marker_color="#10B981", opacity=0.6), row=2, col=1)
                fig_mom.add_trace(go.Bar(x=feat_b["series_macd_hist"].index, y=feat_b["series_macd_hist"], name=f"{t_stock_b} MACD Hist", marker_color="#38BDF8", opacity=0.6), row=2, col=1)
                fig_mom.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.4)", row=2, col=1)

                fig_mom.update_layout(
                    template="plotly_dark",
                    height=460,
                    margin=dict(l=10, r=10, t=35, b=20),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                fig_mom.update_yaxes(title_text="RSI", row=1, col=1)
                fig_mom.update_yaxes(title_text="MACD Hist", row=2, col=1)
                st.plotly_chart(fig_mom, use_container_width=True)

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                # Row of 2 Charts: Volatility Squeeze & Money Flow Dynamics
                vt_c1, vt_c2 = st.columns(2)

                with vt_c1:
                    st.markdown(
                        """
                        <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                            🌪️ Plot 3: Volatility Squeeze Dynamics (Bollinger Bandwidth %)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    fig_bbw = go.Figure()
                    fig_bbw.add_trace(go.Scatter(x=feat_a["series_bb_width"].index, y=feat_a["series_bb_width"], mode="lines", name=f"{t_stock_a} Bandwidth", line=dict(color="#10B981", width=2)))
                    fig_bbw.add_trace(go.Scatter(x=feat_b["series_bb_width"].index, y=feat_b["series_bb_width"], mode="lines", name=f"{t_stock_b} Bandwidth", line=dict(color="#38BDF8", width=2)))
                    fig_bbw.add_hline(y=6.0, line_dash="dash", line_color="#EF4444", annotation_text="Active Squeeze Zone (< 6%)")
                    fig_bbw.update_layout(
                        template="plotly_dark",
                        height=360,
                        margin=dict(l=10, r=10, t=25, b=20),
                        yaxis_title="Bandwidth (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    )
                    st.plotly_chart(fig_bbw, use_container_width=True)

                with vt_c2:
                    st.markdown(
                        """
                        <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                            🌊 Plot 4: Institutional Money Flow Index (MFI 14)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    fig_mfi = go.Figure()
                    fig_mfi.add_trace(go.Scatter(x=feat_a["series_mfi"].index, y=feat_a["series_mfi"], mode="lines", name=f"{t_stock_a} MFI", line=dict(color="#10B981", width=2)))
                    fig_mfi.add_trace(go.Scatter(x=feat_b["series_mfi"].index, y=feat_b["series_mfi"], mode="lines", name=f"{t_stock_b} MFI", line=dict(color="#38BDF8", width=2)))
                    fig_mfi.add_hline(y=80, line_dash="dash", line_color="#EF4444", annotation_text="Overbought 80")
                    fig_mfi.add_hline(y=20, line_dash="dash", line_color="#10B981", annotation_text="Oversold 20")
                    fig_mfi.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.3)")
                    fig_mfi.update_layout(
                        template="plotly_dark",
                        height=360,
                        margin=dict(l=10, r=10, t=25, b=20),
                        yaxis_title="Money Flow Index",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    )
                    st.plotly_chart(fig_mfi, use_container_width=True)

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                # Plot 5: 5-Pillar Technical Radar / Spider Chart
                st.markdown(
                    """
                    <div style="font-size: 0.88rem; font-weight: 700; letter-spacing: 0.06em; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
                        🧭 Plot 5: 5-Pillar Quantitative Technical Radar Scorecard
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                t_categories = [
                    "Trend Alignment & Stack",
                    "Momentum Velocity",
                    "Mean-Reversion Potential",
                    "Capital Volatility Stability",
                    "Institutional Money Flow",
                ]

                r_vals_a = [
                    feat_a.get("trend_score", 50.0),
                    feat_a.get("mom_score", 50.0),
                    feat_a.get("reversion_score", 50.0),
                    feat_a.get("volat_score", 50.0),
                    feat_a.get("flow_score", 50.0),
                ]
                r_vals_b = [
                    feat_b.get("trend_score", 50.0),
                    feat_b.get("mom_score", 50.0),
                    feat_b.get("reversion_score", 50.0),
                    feat_b.get("volat_score", 50.0),
                    feat_b.get("flow_score", 50.0),
                ]

                fig_tech_radar = go.Figure()
                fig_tech_radar.add_trace(
                    go.Scatterpolar(
                        r=r_vals_a,
                        theta=t_categories,
                        fill="toself",
                        name=f"{t_stock_a}",
                        line=dict(color="#10B981", width=2),
                        fillcolor="rgba(16, 185, 129, 0.25)",
                    )
                )
                fig_tech_radar.add_trace(
                    go.Scatterpolar(
                        r=r_vals_b,
                        theta=t_categories,
                        fill="toself",
                        name=f"{t_stock_b}",
                        line=dict(color="#38BDF8", width=2),
                        fillcolor="rgba(56, 189, 248, 0.25)",
                    )
                )
                fig_tech_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#94A3B8")),
                        angularaxis=dict(tickfont=dict(size=11, color="#E2E8F0")),
                        bgcolor="rgba(15, 23, 42, 0.5)",
                    ),
                    template="plotly_dark",
                    height=460,
                    margin=dict(l=40, r=40, t=30, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_tech_radar, use_container_width=True)


if __name__ == "__main__":
    render_page()
