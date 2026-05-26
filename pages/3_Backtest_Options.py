"""
pages/3_Backtest_Options.py
Options Backtest — NIFTY Index options strategy backtest triggered by entry/exit signals.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.signal_builder import (
    SignalRule, Condition, STRATEGY_TEMPLATES,
    AVAILABLE_INDICATORS, AVAILABLE_OPERATORS,
)
from core.options_engine import (
    OptionsBacktestEngine, OPTION_STRATEGY_CONFIGS, NIFTY_LOT_SIZE
)
from core.indicators import IndicatorLibrary
from core.universe import UniverseManager

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Backtest | Option-Stock Scanner",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.options-header {
    background: linear-gradient(135deg, #0a0a1a, #1a0533, #0d1a4a);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.options-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.options-header p  { color: rgba(255,255,255,0.65); margin-top: 0.3rem; }

.metric-card {
    background: linear-gradient(135deg, rgba(30,30,60,0.9), rgba(20,20,40,0.95));
    border: 1px solid rgba(100,120,255,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: white;
}
.metric-value { font-size: 1.6rem; font-weight: 700; }
.metric-label { font-size: 0.8rem; color: rgba(255,255,255,0.6); margin-top: 0.2rem; }

.trade-win  { color: #00ff88; }
.trade-loss { color: #ff4444; }

.strategy-badge {
    display: inline-block;
    background: linear-gradient(135deg, #4a90e2, #7b2fbe);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="options-header">
    <h1>🎯 NIFTY Options Backtest</h1>
    <p>Simulate option strategies triggered by entry/exit signals on the NIFTY Index</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Strategy Config
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("⚙️ Strategy Configuration")

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    strategy_name = st.selectbox(
        "Option Strategy",
        list(OPTION_STRATEGY_CONFIGS.keys()) + ["Custom"],
        index=0,
        help="Pre-built multi-leg strategy templates",
        key="opt_strategy",
    )
    st.caption({
        "Buy ATM Call":      "🟢 Single leg — bullish directional bet. Max loss = premium paid.",
        "Buy ATM Put":       "🔴 Single leg — bearish directional bet. Max loss = premium paid.",
        "Bull Call Spread":  "📈 Buy ATM Call + Sell OTM Call. Capped profit, lower cost.",
        "Bear Put Spread":   "📉 Buy ATM Put + Sell OTM Put. Capped profit, lower cost.",
        "Long Straddle":     "⚡ Buy ATM Call + ATM Put. Profits from big moves either way.",
        "Long Strangle":     "⚡ Buy OTM Call + OTM Put. Cheaper straddle, needs bigger move.",
        "Custom":            "🔧 Define your own legs below.",
    }.get(strategy_name, ""))

with col_cfg2:
    lot_size   = st.number_input("NIFTY Lot Size", value=NIFTY_LOT_SIZE, min_value=1, key="lot_size")
    lots       = st.number_input("Number of Lots", value=1, min_value=1, max_value=50, key="lots")
    init_cap   = st.number_input("Initial Capital (₹)", value=500_000, step=50_000, key="init_cap")

# Custom legs UI
if strategy_name == "Custom":
    st.markdown("**Define Custom Legs:**")
    num_legs = st.number_input("Number of Legs", min_value=1, max_value=4, value=2, key="num_custom_legs")
    custom_legs_data = []
    for leg_i in range(int(num_legs)):
        st.markdown(f"*Leg {leg_i + 1}*")
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            opt_type = st.selectbox("Type", ["CE", "PE"], key=f"custom_leg_type_{leg_i}")
        with lc2:
            direction = st.selectbox("Direction", ["BUY", "SELL"], key=f"custom_leg_dir_{leg_i}")
        with lc3:
            offset = st.number_input(
                "Strike Offset (× 50 pts)",
                min_value=-10, max_value=10, value=0, key=f"custom_leg_offset_{leg_i}"
            )
        with lc4:
            leg_lots = st.number_input("Lots", min_value=1, max_value=20, value=1, key=f"custom_leg_lots_{leg_i}")
        from core.options_engine import OptionLeg
        custom_legs_data.append(OptionLeg(opt_type, direction, int(offset), int(leg_lots)))
else:
    custom_legs_data = None

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Entry / Exit Signals
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📡 Signal Rules")

# Quick template selector
tmpl_names = ["(Custom)"] + list(STRATEGY_TEMPLATES.keys())
tmpl_col, _ = st.columns([1, 2])
with tmpl_col:
    tmpl_choice = st.selectbox("Signal Template", tmpl_names, key="opt_tmpl_choice")

def _build_rule_from_tmpl(tmpl_name: str, direction: str) -> SignalRule:
    if tmpl_name == "(Custom)":
        return SignalRule()
    tmpl = STRATEGY_TEMPLATES.get(tmpl_name, {})
    return tmpl.get(direction, SignalRule())

if "opt_entry_rule" not in st.session_state:
    st.session_state.opt_entry_rule = _build_rule_from_tmpl(tmpl_choice, "entry")
    st.session_state.opt_exit_rule  = _build_rule_from_tmpl(tmpl_choice, "exit")

if tmpl_choice != "(Custom)":
    entry_rule = _build_rule_from_tmpl(tmpl_choice, "entry")
    exit_rule  = _build_rule_from_tmpl(tmpl_choice, "exit")
    st.session_state.opt_entry_rule = entry_rule
    st.session_state.opt_exit_rule  = exit_rule

# Show current rules
col_er, col_xr = st.columns(2)
with col_er:
    st.markdown("**Entry Conditions**")
    entry_rule = st.session_state.opt_entry_rule
    for c in entry_rule.conditions:
        st.code(f"{c.left}  {c.operator}  {c.right}", language=None)
    if entry_rule.is_empty():
        st.caption("_(No conditions — uses Scanner entry signals if available)_")

with col_xr:
    st.markdown("**Exit Conditions**")
    exit_rule = st.session_state.opt_exit_rule
    for c in exit_rule.conditions:
        st.code(f"{c.left}  {c.operator}  {c.right}", language=None)
    if exit_rule.is_empty():
        st.caption("_(No conditions — uses Scanner exit signals if available)_")

st.caption("💡 Use the **Scanner** page to build detailed rules, then come back here to backtest.")

# Pull from Scanner session if available
if "entry_conditions" in st.session_state and st.session_state.entry_conditions:
    if st.button("📥 Import rules from Scanner page", key="import_scanner_rules"):
        def _rebuild_rule(conds_data, logic_key):
            conditions = []
            for c in conds_data:
                r = c["right"]
                try:
                    r = float(r)
                except (ValueError, TypeError):
                    pass
                conditions.append(Condition(c["left"], c["operator"], r))
            return SignalRule(conditions=conditions, logic=st.session_state.get(logic_key, "AND"))

        entry_rule = _rebuild_rule(st.session_state.entry_conditions, "entry_logic")
        exit_rule  = _rebuild_rule(st.session_state.exit_conditions, "exit_logic")
        st.session_state.opt_entry_rule = entry_rule
        st.session_state.opt_exit_rule  = exit_rule
        st.success("✅ Rules imported from Scanner!")
        st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Date Range & Options Settings
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📅 Backtest Settings")

col_b1, col_b2, col_b3 = st.columns(3)
with col_b1:
    bt_start = st.date_input("Start Date", value=pd.Timestamp("2022-01-01"), key="bt_start")
with col_b2:
    bt_end   = st.date_input("End Date",   value=pd.Timestamp.today(),       key="bt_end")
with col_b3:
    use_vix  = st.toggle("Use India VIX for IV (recommended)", value=True, key="use_vix")

st.caption(
    "📌 **Data source**: Dhan Rolling Options API (historical option prices). "
    "If Dhan is unavailable, Black-Scholes with India VIX is used as fallback."
)

# Dhan client check
dhan_client = None
try:
    import streamlit as st_inner
    if hasattr(st_inner, "session_state") and "dhan_client" in st_inner.session_state:
        dhan_client = st_inner.session_state["dhan_client"]
except Exception:
    pass

if dhan_client is None:
    st.warning("⚠️ Dhan client not authenticated — will use Black-Scholes fallback. "
               "Authenticate via the **Dhan Auth** tab for live option prices.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# RUN BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
run_backtest = st.button("🚀 Run Options Backtest", type="primary", use_container_width=False, key="run_opt_bt")

if run_backtest:
    entry_rule = st.session_state.opt_entry_rule
    exit_rule  = st.session_state.opt_exit_rule

    if entry_rule.is_empty():
        st.error("Please select a signal template or import rules from the Scanner page.")
    else:
        with st.spinner("Running options backtest..."):
            engine = OptionsBacktestEngine(
                entry_rule=entry_rule,
                exit_rule=exit_rule,
                strategy_name=strategy_name,
                custom_legs=custom_legs_data,
                start_date=str(bt_start),
                end_date=str(bt_end),
                initial_capital=float(init_cap),
                dhan_client=dhan_client,
                lot_size=int(lot_size),
                use_vix=use_vix,
            )
            result = engine.run()

        st.session_state.opt_backtest_result = result
        st.success("✅ Options backtest complete!")

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if "opt_backtest_result" in st.session_state and st.session_state.opt_backtest_result:
    result = st.session_state.opt_backtest_result
    metrics    = result.get("metrics", {})
    trades_df  = result.get("trades_df", pd.DataFrame())
    equity_df  = result.get("equity_df", pd.DataFrame())
    underlying_df = result.get("underlying_df", pd.DataFrame())

    st.divider()
    st.subheader("📊 Backtest Results")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    m_cols = st.columns(4)
    kpi_items = [
        ("Total P&L", f"₹{metrics.get('Total P&L (INR)', 0):,.0f}",
         None, "Total P&L (INR)"),
        ("Total Return", f"{metrics.get('Total Return (%)', 0):.1f}%",
         None, "Total Return (%)"),
        ("Win Rate", f"{metrics.get('Win Rate (%)', 0):.0f}%",
         None, "Win Rate (%)"),
        ("Max Drawdown", f"{metrics.get('Max Drawdown (%)', 0):.1f}%",
         None, "Max Drawdown (%)"),
    ]
    for i, (label, val, _, _) in enumerate(kpi_items):
        with m_cols[i]:
            pnl = metrics.get('Total P&L (INR)', 0)
            color = "#00ff88" if pnl >= 0 else "#ff4444"
            st.metric(label, val)

    m2_cols = st.columns(4)
    with m2_cols[0]:
        st.metric("Total Trades",      metrics.get("Total Trades", 0))
    with m2_cols[1]:
        st.metric("Sharpe Ratio",      f"{metrics.get('Sharpe Ratio', 0):.2f}")
    with m2_cols[2]:
        st.metric("Best Trade",        f"₹{metrics.get('Best Trade (INR)', 0):,.0f}")
    with m2_cols[3]:
        st.metric("Worst Trade",       f"₹{metrics.get('Worst Trade (INR)', 0):,.0f}")

    # ── Equity Curve ──────────────────────────────────────────────────────────
    if not equity_df.empty:
        st.markdown("#### 📈 Portfolio Equity Curve")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=equity_df.index, y=equity_df["Equity"],
            mode="lines", name="Portfolio Value",
            line=dict(color="#4a90e2", width=2),
            fill="tozeroy",
            fillcolor="rgba(74,144,226,0.08)",
        ))
        fig_eq.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            height=400,
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── Trade Log ─────────────────────────────────────────────────────────────
    if not trades_df.empty:
        st.markdown("#### 🗒️ Trade Log")

        display_cols = ["entry_date", "exit_date", "entry_spot", "exit_spot",
                        "strategy", "legs_detail", "total_pnl_inr", "pnl_pct", "exit_reason"]
        disp = trades_df[[c for c in display_cols if c in trades_df.columns]].copy()
        disp.columns = [c.replace("_", " ").title() for c in disp.columns]

        # Color P&L column
        def _style_pnl(val):
            try:
                v = float(val)
                return f"color: {'#00ff88' if v >= 0 else '#ff4444'}"
            except Exception:
                return ""

        st.dataframe(disp, use_container_width=True, hide_index=True)

        # ── P&L by Trade Chart ────────────────────────────────────────────────
        if "total_pnl_inr" in trades_df.columns:
            fig_pnl = go.Figure(go.Bar(
                x=[str(d)[:10] for d in trades_df.get("entry_date", [])],
                y=trades_df["total_pnl_inr"],
                marker_color=["#00ff88" if p >= 0 else "#ff4444"
                              for p in trades_df["total_pnl_inr"]],
                name="P&L per Trade",
            ))
            fig_pnl.update_layout(
                title="P&L Per Trade",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                xaxis_title="Entry Date",
                yaxis_title="P&L (₹)",
                height=350,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        # Download
        csv = trades_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Trade Log (CSV)",
            data=csv,
            file_name="nifty_options_trades.csv",
            mime="text/csv",
            key="dl_opt_trades",
        )

    # ── NIFTY Chart with signals ───────────────────────────────────────────────
    if not underlying_df.empty:
        signals_df = result.get("signals_df", pd.DataFrame())
        with st.expander("📉 NIFTY Price Chart with Trade Signals"):
            fig_nifty = go.Figure()
            fig_nifty.add_trace(go.Candlestick(
                x=underlying_df.index,
                open=underlying_df["Open"],
                high=underlying_df["High"],
                low=underlying_df["Low"],
                close=underlying_df["Close"],
                name="NIFTY",
            ))

            if not trades_df.empty and "entry_date" in trades_df.columns:
                entry_prices = underlying_df.loc[
                    underlying_df.index.isin(trades_df["entry_date"]), "Low"
                ] * 0.995
                fig_nifty.add_trace(go.Scatter(
                    x=trades_df["entry_date"].tolist(),
                    y=entry_prices.tolist(),
                    mode="markers", name="Trade Entry",
                    marker=dict(symbol="triangle-up", color="#00ff88", size=14),
                ))
                if "exit_date" in trades_df.columns:
                    exit_prices = underlying_df.loc[
                        underlying_df.index.isin(trades_df["exit_date"]), "High"
                    ] * 1.005
                    fig_nifty.add_trace(go.Scatter(
                        x=trades_df["exit_date"].tolist(),
                        y=exit_prices.tolist(),
                        mode="markers", name="Trade Exit",
                        marker=dict(symbol="triangle-down", color="#ff4444", size=14),
                    ))

            fig_nifty.update_layout(
                title="NIFTY50 with Options Trade Entries/Exits",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                xaxis_rangeslider_visible=False,
                height=500,
            )
            st.plotly_chart(fig_nifty, use_container_width=True)
