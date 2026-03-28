"""
backtest_engine.py  —  Senior Quant Researcher Edition  (Final)
=======================================================
Handles execution, professional performance metrics, and dual-mode
visualization (static Matplotlib + interactive Plotly HTML tear sheet).
"""

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import precision_score

# Plotly — graceful degradation if not installed
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None
    print("[WARN]  Plotly not installed. Install via `pip install plotly` for interactive tear sheets.")

from model_engine import (
    get_market_data,
    add_technical_features,
    prepare_data,
    train_and_predict,
    RISK_FREE_RATE,
)

warnings.filterwarnings("ignore")

TRADE_COST   = 0.0015           # 0.15% per trade (STT + brokerage)
TEARSHEET_DIR = "tearsheets"
os.makedirs(TEARSHEET_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# QUANT METRICS ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def compute_quant_metrics(returns: pd.Series, trading_days: int = 252) -> dict:
    """
    Computes institutional-grade risk/return metrics from a daily returns series.
    """
    daily_rf     = RISK_FREE_RATE / trading_days
    excess       = returns - daily_rf
    total_vol    = returns.std()

    # ── Sharpe & Sortino ──────────────────────────────────────────────────────
    sharpe = (excess.mean() / total_vol * np.sqrt(trading_days)
              if total_vol > 0 else np.nan)

    downside_returns = returns[returns < daily_rf]
    downside_std     = downside_returns.std()
    sortino = (excess.mean() * trading_days / (downside_std * np.sqrt(trading_days))
               if downside_std > 0 else np.nan)

    # ── Calmar ────────────────────────────────────────────────────────────────
    equity        = (1.0 + returns).cumprod()
    roll_max_eq   = equity.cummax()
    drawdown_ser  = (equity / roll_max_eq) - 1.0
    max_drawdown  = drawdown_ser.min()
    n_years       = len(returns) / trading_days
    cagr          = (equity.iloc[-1] ** (1.0 / n_years)) - 1.0 if n_years > 0 else np.nan
    calmar = (cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan)

    # ── Trade-Level Statistics (FIXED: active days only) ──────────────────────
    active_returns = returns[returns != 0]
    positive      = returns[returns > 0]
    negative      = returns[returns < 0]
    
    win_rate      = len(positive) / len(active_returns) if len(active_returns) > 0 else np.nan
    avg_win       = positive.mean() if len(positive) > 0 else 0.0
    avg_loss      = negative.mean() if len(negative) > 0 else 0.0
    profit_factor = (positive.sum() / abs(negative.sum())
                     if len(negative) > 0 and abs(negative.sum()) > 0 else np.nan)

    # Max consecutive losses (vectorised streak counter)
    loss_flag          = (returns < 0).astype(int)
    cumsum_reset       = loss_flag.groupby((loss_flag == 0).cumsum()).cumsum()
    max_consec_losses  = int(cumsum_reset.max())

    return {
        "sharpe":             round(sharpe, 3),
        "sortino":            round(sortino, 3),
        "calmar":             round(calmar, 3),
        "cagr_pct":           round(cagr * 100, 2),
        "max_drawdown_pct":   round(max_drawdown * 100, 2),
        "win_rate_pct":       round(win_rate * 100, 2),
        "profit_factor":      round(profit_factor, 3),
        "avg_win_pct":        round(avg_win * 100, 4),
        "avg_loss_pct":       round(avg_loss * 100, 4),
        "max_consec_losses":  max_consec_losses,
    }


def _build_backtest_frame(test_df: pd.DataFrame, custom_preds: np.ndarray) -> pd.DataFrame:
    """Attaches predictions, computes costs, and equity curves — fully vectorised."""
    df = test_df.copy()
    df["Prediction"]           = custom_preds

    position_changes           = df["Prediction"].diff().abs().fillna(0)
    df["Trade_Cost"]           = position_changes * TRADE_COST
    df["Gross_Strategy_Return"] = df["Tomorrow_Return"] * df["Prediction"]
    df["Net_Strategy_Return"]   = df["Gross_Strategy_Return"] - df["Trade_Cost"]

    df["Buy_and_Hold_Equity"]   = (1.0 + df["Tomorrow_Return"]).cumprod()
    df["Strategy_Equity"]       = (1.0 + df["Net_Strategy_Return"]).cumprod()

    roll_max                    = df["Strategy_Equity"].cummax()
    df["Drawdown"]              = (df["Strategy_Equity"] / roll_max) - 1.0

    return df, roll_max


# ──────────────────────────────────────────────────────────────────────────────
# CONSOLE OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def _print_console_report(
    ticker_name:    str,
    next_day_prob:  float,
    precision:      float,
    bt:             pd.DataFrame,
    metrics:        dict,
) -> None:
    bar  = "=" * 58
    dash = "-" * 58

    bh_ret  = (bt["Buy_and_Hold_Equity"].iloc[-2] - 1) * 100
    str_ret = (bt["Strategy_Equity"].iloc[-2] - 1) * 100

    print(f"\n{bar}")
    print(f"  📊  ADVANCED QUANT RESULTS  —  {ticker_name.upper()}")
    print(bar)
    print(f"  Tomorrow P(UP > 0.2%) : {next_day_prob*100:.2f}%")
    signal = "🟢  STRONG BUY" if next_day_prob > 0.52 else "⚪  HOLD / NO ACTION"
    print(f"  Model Signal          : {signal}")
    print(dash)
    print(f"  Precision (Buy calls) : {precision*100:.2f}%")
    print(f"  Win Rate              : {metrics['win_rate_pct']:.2f}%")
    print(f"  Profit Factor         : {metrics['profit_factor']:.3f}")
    print(f"  Avg Win / Avg Loss    : {metrics['avg_win_pct']:.3f}% / {metrics['avg_loss_pct']:.3f}%")
    print(f"  Max Consec. Losses    : {metrics['max_consec_losses']}")
    print(dash)
    print(f"  Sharpe  Ratio         : {metrics['sharpe']:.3f}")
    print(f"  Sortino Ratio         : {metrics['sortino']:.3f}")
    print(f"  Calmar  Ratio         : {metrics['calmar']:.3f}")
    print(f"  Strategy CAGR         : {metrics['cagr_pct']:.2f}%")
    print(f"  Max Drawdown          : {metrics['max_drawdown_pct']:.2f}%")
    print(dash)
    print(f"  Buy-&-Hold Return     : {bh_ret:.2f}%")
    print(f"  Net Strategy Return   : {str_ret:.2f}%")
    print(f"  Alpha (Strategy - B&H): {str_ret - bh_ret:.2f}%")
    print(f"{bar}\n")


# ──────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STATIC DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

def _plot_matplotlib(
    bt:           pd.DataFrame,
    roll_max:     pd.Series,
    model,
    predictors:   list,
    metrics:      dict,
    precision:    float,
    ticker_name:  str,
) -> None:
    """Four-panel static Matplotlib dashboard."""
    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(
        f"XGBoost Algo Strategy — {ticker_name}",
        fontsize=15, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[2, 2, 1, 1.2], hspace=0.45)

    dates = bt.index
    colors = {"up": "#2ecc71", "down": "#e74c3c", "neutral": "#3498db",
              "grey": "#95a5a6", "dark": "#2c3e50"}

    # ── Panel 1: Price + Signals ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, bt["Close"], color=colors["dark"], lw=1.1, label="Close Price", alpha=0.85)
    buys = bt[bt["Prediction"] == 1]
    ax1.scatter(buys.index, buys["Close"], marker="^", color=colors["up"],
                s=35, zorder=5, label="Active Long (≥52% prob)")
    ax1.set_title(f"Price Action & ML Signal — {ticker_name}", fontsize=11)
    ax1.set_ylabel("Price (INR)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Equity Curves + Drawdown ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, bt["Buy_and_Hold_Equity"], color=colors["grey"], lw=1.2,
             alpha=0.7, label="Buy & Hold")
    ax2.plot(dates, bt["Strategy_Equity"], color=colors["up"], lw=2.0,
             label="XGBoost Strategy (net costs)")
    ax2.fill_between(dates, bt["Strategy_Equity"], roll_max,
                     color=colors["down"], alpha=0.12, label="Drawdown")

    # Annotate key metrics in the equity panel
    metric_text = (
        f"Sharpe: {metrics['sharpe']:.2f}   "
        f"Sortino: {metrics['sortino']:.2f}   "
        f"Calmar: {metrics['calmar']:.2f}   "
        f"MaxDD: {metrics['max_drawdown_pct']:.1f}%"
    )
    ax2.set_title(
        f"Cumulative Equity (0.15% cost/trade)  |  {metric_text}",
        fontsize=9.5,
    )
    ax2.set_ylabel("Equity Multiplier")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: Drawdown Series ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(dates, bt["Drawdown"] * 100, 0,
                     color=colors["down"], alpha=0.55)
    ax3.set_title("Strategy Drawdown (%)", fontsize=10)
    ax3.set_ylabel("DD (%)")
    ax3.grid(True, alpha=0.25)

    # ── Panel 4: Feature Importances ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    bar_colors  = [colors["neutral"] if i >= len(predictors) - 4 else colors["dark"]
                   for i in range(len(indices))]
    bars = ax4.barh(range(len(indices)), importances[indices],
                    color=[bar_colors[i] for i in range(len(indices))], height=0.7)
    ax4.set_yticks(range(len(indices)))
    ax4.set_yticklabels([predictors[i] for i in indices], fontsize=8)
    ax4.set_title("XGBoost Feature Importances  (blue = new advanced features)", fontsize=10)
    ax4.set_xlabel("Relative Importance")
    ax4.grid(True, alpha=0.2, axis="x")

    plt.savefig(
        os.path.join(TEARSHEET_DIR, f"{ticker_name}_dashboard.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.show()
    print(f"[PLOT]   Static dashboard saved → {TEARSHEET_DIR}/{ticker_name}_dashboard.png")


# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY INTERACTIVE HTML TEAR SHEET
# ──────────────────────────────────────────────────────────────────────────────

def _plot_plotly(
    bt:           pd.DataFrame,
    roll_max:     pd.Series,
    model,
    predictors:   list,
    metrics:      dict,
    precision:    float,
    ticker_name:  str,
    next_day_prob: float,
) -> None:
    """
    Generates a professional, fully interactive Plotly HTML tear sheet.
    """
    if not PLOTLY_AVAILABLE:
        print("[SKIP]   Plotly not available — skipping interactive tear sheet.")
        return

    # ── Rolling 63-day Sharpe ─────────────────────────────────────────────────
    daily_rf          = RISK_FREE_RATE / 252
    excess_ret        = bt["Net_Strategy_Return"] - daily_rf
    rolling_sharpe    = (excess_ret.rolling(63).mean() /
                         bt["Net_Strategy_Return"].rolling(63).std()) * np.sqrt(252)

    # ── Buy signals ───────────────────────────────────────────────────────────
    buys   = bt[bt["Prediction"] == 1]
    signal_label = "🟢 STRONG BUY" if next_day_prob > 0.52 else "⚪ HOLD"

    # ── Subplot layout ────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        row_heights=[0.32, 0.24, 0.14, 0.14, 0.16],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{ticker_name} — Price & ML Signals",
            "Strategy vs Buy-&-Hold Equity",
            "Strategy Drawdown (%)",
            "Rolling 63-Day Sharpe Ratio",
            "Feature Importances",
        ],
    )

    dates = bt.index

    # ── Row 1: Candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates, open=bt["Open"], high=bt["High"],
        low=bt["Low"], close=bt["Close"],
        name="OHLC",
        increasing_line_color="#2ecc71",
        decreasing_line_color="#e74c3c",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Close"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=9, color="#2ecc71",
                    line=dict(color="white", width=0.8)),
        name="Long Signal",
        hovertemplate="<b>BUY</b><br>Date: %{x}<br>Close: ₹%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # ── Row 2: Equity Curves ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=bt["Buy_and_Hold_Equity"],
        name="Buy & Hold", line=dict(color="#95a5a6", width=1.5, dash="dot"),
        hovertemplate="B&H: %{y:.3f}x<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=bt["Strategy_Equity"],
        name="XGBoost Strategy", line=dict(color="#2ecc71", width=2.2),
        hovertemplate="Strategy: %{y:.3f}x<extra></extra>",
        fill="tonexty", fillcolor="rgba(46,204,113,0.08)",
    ), row=2, col=1)

    # ── Row 3: Drawdown ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=bt["Drawdown"] * 100,
        name="Drawdown", fill="tozeroy",
        fillcolor="rgba(231,76,60,0.30)",
        line=dict(color="#e74c3c", width=0.8),
        hovertemplate="DD: %{y:.2f}%<extra></extra>",
    ), row=3, col=1)

    # ── Row 4: Rolling Sharpe ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=rolling_sharpe,
        name="63-Day Sharpe", line=dict(color="#3498db", width=1.5),
        hovertemplate="Sharpe: %{y:.2f}<extra></extra>",
    ), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="grey", row=4, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color="#2ecc71",
                  annotation_text="Sharpe=1", row=4, col=1)

    # ── Row 5: Feature Importances ────────────────────────────────────────────
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    feat_names  = [predictors[i] for i in sorted_idx]
    feat_vals   = importances[sorted_idx]
    advanced    = {"GK_Volatility", "Vol_Regime_Ratio", "Dist_to_VWAP", "Return_AutoCorr"}
    bar_colors  = ["#3498db" if f in advanced else "#7f8c8d" for f in feat_names]

    fig.add_trace(go.Bar(
        x=feat_vals, y=feat_names, orientation="h",
        name="Feature Importance",
        marker_color=bar_colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ), row=5, col=1)

    # ── Metrics Annotation Box ─────────────────────────────────────────────────
    metrics_html = (
        f"<b>Signal:</b> {signal_label} ({next_day_prob*100:.1f}%)  │  "
        f"<b>Precision:</b> {precision*100:.1f}%  │  "
        f"<b>Sharpe:</b> {metrics['sharpe']:.2f}  │  "
        f"<b>Sortino:</b> {metrics['sortino']:.2f}  │  "
        f"<b>Calmar:</b> {metrics['calmar']:.2f}  │  "
        f"<b>CAGR:</b> {metrics['cagr_pct']:.1f}%  │  "
        f"<b>MaxDD:</b> {metrics['max_drawdown_pct']:.1f}%  │  "
        f"<b>Win Rate:</b> {metrics['win_rate_pct']:.1f}%  │  "
        f"<b>PF:</b> {metrics['profit_factor']:.2f}"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Quantitative Strategy Tear Sheet</b> — {ticker_name}<br>"
                f"<span style='font-size:11px;color:#7f8c8d'>{metrics_html}</span>"
            ),
            font=dict(size=14),
            x=0.01,
        ),
        template="plotly_dark",
        height=1100,
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.02, bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=110, b=40),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
    )

    # Style axes
    for row_idx in range(1, 6):
        fig.update_yaxes(gridcolor="#2d2d44", row=row_idx, col=1)
        fig.update_xaxes(gridcolor="#2d2d44", row=row_idx, col=1)

    out_path = os.path.join(TEARSHEET_DIR, f"{ticker_name}_tearsheet.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[PLOT]   Interactive tear sheet saved → {out_path}")
    fig.show()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_and_plot(
    test_df:      pd.DataFrame,
    custom_preds: np.ndarray,
    model,
    predictors:   list,
    latest_data:  pd.DataFrame,
    ticker_name:  str,
    plot_mode:    str = "both",   # "matplotlib" | "plotly" | "both"
) -> None:
    y_test    = test_df["Target"]
    precision = precision_score(y_test, custom_preds, zero_division=0)

    bt, roll_max = _build_backtest_frame(test_df, custom_preds)

    strat_returns  = bt["Net_Strategy_Return"].iloc[:-1]
    metrics        = compute_quant_metrics(strat_returns)

    next_day_prob  = model.predict_proba(latest_data)[:, 1][0]

    _print_console_report(ticker_name, next_day_prob, precision, bt, metrics)

    if plot_mode in ("matplotlib", "both"):
        _plot_matplotlib(bt, roll_max, model, predictors, metrics, precision, ticker_name)

    if plot_mode in ("plotly", "both"):
        _plot_plotly(bt, roll_max, model, predictors, metrics, precision,
                     ticker_name, next_day_prob)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  XGBoost Algo Trading System  —  Quant Edition")
    print("=" * 52)
    print("\nSelect an equity to analyse:")

    stock_map = {
        1: ("TATAMOTORS.NS",  "Tata Motors"),
        2: ("ADANIPOWER.NS",  "Adani Power"),
        3: ("RELIANCE.NS",    "Reliance Industries"),
        4: ("COALINDIA.NS",   "Coal India"),
        5: ("HDFCBANK.NS",    "HDFC Bank"),
    }
    for key, val in stock_map.items():
        print(f"  {key}. {val[1]}")

    try:
        n = int(input("\nEnter choice (1-5): "))
        ticker_symbol, company_name = stock_map[n]
    except (ValueError, KeyError):
        print("[WARN]  Invalid input — defaulting to Tata Motors.")
        ticker_symbol, company_name = stock_map[1]

    plot_choice = input("Plot mode — [1] Matplotlib  [2] Plotly  [3] Both (default): ").strip()
    mode_map    = {"1": "matplotlib", "2": "plotly", "3": "both"}
    plot_mode   = mode_map.get(plot_choice, "both")

    # ── Modular Pipeline ──────────────────────────────────────────────────────
    df = get_market_data(ticker_symbol)
    df = add_technical_features(df)

    X_train, y_train, X_test, y_test, predictors, test_df, latest_data = prepare_data(df)
    model, custom_preds = train_and_predict(X_train, y_train, X_test)

    evaluate_and_plot(
        test_df, custom_preds, model, predictors,
        latest_data, company_name, plot_mode=plot_mode,
    )