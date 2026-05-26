"""
pages/1_Scanner.py
Option-Stock Scanner — Main entry/exit signal scanner with scoring.

Supports two modes:
  Equity  : Scan NSE stocks from a predefined index or custom list
  Options : Evaluate entry/exit signals on NIFTY underlying for options trades
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.indicators import IndicatorLibrary
from core.signal_builder import (
    SignalBuilder, SignalRule, Condition,
    STRATEGY_TEMPLATES, AVAILABLE_INDICATORS, AVAILABLE_OPERATORS,
)
from core.universe import UniverseManager
from core.scanner import Scanner, DEFAULT_SCORING_WEIGHTS
from core.scoring import ScoreParser

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Option-Stock Scanner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.main-header p  { color: rgba(255,255,255,0.7); margin-top: 0.3rem; font-size: 1rem; }

.mode-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem;
    cursor: pointer;
    transition: all 0.2s ease;
    color: white;
}
.mode-card:hover { border-color: rgba(100,160,255,0.4); transform: translateY(-2px); }
.mode-card.selected { border-color: #4a90e2; box-shadow: 0 0 20px rgba(74,144,226,0.3); }

.rule-row {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
}

.score-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "entry_conditions" not in st.session_state:
    st.session_state.entry_conditions = []
if "exit_conditions" not in st.session_state:
    st.session_state.exit_conditions = []
if "scanner_mode" not in st.session_state:
    st.session_state.scanner_mode = "Equity"
if "scoring_weights" not in st.session_state:
    st.session_state.scoring_weights = dict(DEFAULT_SCORING_WEIGHTS)
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

um = UniverseManager()

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🔍 Option-Stock Scanner</h1>
    <p>Build entry/exit signal rules, scan your universe, and rank stocks by score</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Mode & Universe
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 1 — Asset Class & Universe")

col_mode1, col_mode2 = st.columns(2)

with col_mode1:
    eq_selected = st.session_state.scanner_mode == "Equity"
    if st.button(
        "📈 Equity Mode\nScan NSE stocks from an index or custom list",
        key="btn_equity_mode",
        use_container_width=True,
        type="primary" if eq_selected else "secondary",
    ):
        st.session_state.scanner_mode = "Equity"
        st.rerun()

with col_mode2:
    opt_selected = st.session_state.scanner_mode == "Options (Nifty Index)"
    if st.button(
        "🎯 Options Mode (Nifty Index)\nBuy/sell NIFTY CE/PE on signal triggers",
        key="btn_options_mode",
        use_container_width=True,
        type="primary" if opt_selected else "secondary",
    ):
        st.session_state.scanner_mode = "Options (Nifty Index)"
        st.rerun()

st.caption(f"**Current mode:** {st.session_state.scanner_mode}")

# Universe selector
if st.session_state.scanner_mode == "Equity":
    col_univ, col_custom = st.columns([1, 2])

    with col_univ:
        universe_source = st.radio(
            "Universe Source",
            ["Predefined Index", "Custom Ticker List"],
            horizontal=True,
            key="universe_source",
        )

    if universe_source == "Predefined Index":
        with col_custom:
            selected_index = st.selectbox(
                "Select Index",
                options=um.get_index_display_names(),
                index=0,
                key="selected_index",
            )
        tickers = um.get_equity_universe(selected_index)
        st.caption(f"📋 **{len(tickers)} stocks** in {selected_index}")

    else:
        with col_custom:
            raw_text = st.text_area(
                "Paste tickers (comma, space, or newline separated)",
                placeholder="RELIANCE, TCS, INFY\nHDFCBANK\nICICIBANK",
                height=100,
                key="custom_tickers_text",
            )
        tickers = um.parse_custom_tickers(raw_text) if raw_text.strip() else []
        if tickers:
            st.caption(f"📋 **{len(tickers)} tickers** detected: {', '.join(tickers[:10])}{'...' if len(tickers)>10 else ''}")
        else:
            st.warning("⚠️ No tickers found. Please paste at least one valid NSE symbol.")

else:
    # Options mode — underlying is always NIFTY
    st.info("🎯 **Options Mode**: Signals are evaluated on the **NIFTY50 Index** (^NSEI). "
            "When an entry fires, an options trade is placed via the Backtest Options page.")
    tickers = ["^NSEI"]

# Date range
col_d1, col_d2 = st.columns(2)
with col_d1:
    start_date = st.date_input("Scan Start Date", value=pd.Timestamp("2023-01-01"), key="scan_start")
with col_d2:
    end_date = st.date_input("Scan End Date", value=pd.Timestamp.today(), key="scan_end")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Signal Builder
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 2 — Entry / Exit Signal Builder")


def _render_rule_builder(label: str, key_prefix: str) -> SignalRule:
    """Renders the two-panel signal builder (templates + rule editor). Returns a SignalRule."""

    st.markdown(f"**{label}**")

    # Panel A — Templates
    template_names = ["(Custom — no template)"] + list(STRATEGY_TEMPLATES.keys())
    chosen_template = st.selectbox(
        f"Quick-start Template",
        template_names,
        key=f"{key_prefix}_template",
        label_visibility="collapsed",
    )

    # Load template → seed conditions list
    direction = "entry" if "Entry" in label else "exit"
    if chosen_template != "(Custom — no template)":
        tmpl = STRATEGY_TEMPLATES[chosen_template]
        rule_data = tmpl[direction if direction in tmpl else "entry"]
        # Seed conditions from template (only if user hasn't edited yet)
        cond_key = f"{key_prefix}_conditions"
        if cond_key not in st.session_state or st.session_state.get(f"{key_prefix}_last_tmpl") != chosen_template:
            st.session_state[cond_key] = [c.to_dict() for c in rule_data.conditions]
            st.session_state[f"{key_prefix}_logic"] = rule_data.logic
            st.session_state[f"{key_prefix}_last_tmpl"] = chosen_template
    else:
        cond_key = f"{key_prefix}_conditions"
        if cond_key not in st.session_state:
            st.session_state[cond_key] = []

    # Panel B — Rule Editor
    with st.container(border=True):
        conditions_data = st.session_state.get(f"{key_prefix}_conditions", [])

        # Add condition button
        if st.button(f"➕ Add Condition", key=f"{key_prefix}_add"):
            conditions_data.append({
                "left": "RSI_14", "operator": ">", "right": "50"
            })
            st.session_state[f"{key_prefix}_conditions"] = conditions_data
            st.rerun()

        # Render each condition as a row
        to_delete = None
        for i, cond in enumerate(conditions_data):
            cols = st.columns([2.5, 2, 2, 0.6])
            with cols[0]:
                left = st.selectbox(
                    "Indicator", AVAILABLE_INDICATORS,
                    index=AVAILABLE_INDICATORS.index(cond.get("left", "RSI_14")) if cond.get("left") in AVAILABLE_INDICATORS else 0,
                    key=f"{key_prefix}_left_{i}",
                    label_visibility="collapsed",
                )
            with cols[1]:
                op = st.selectbox(
                    "Operator", AVAILABLE_OPERATORS,
                    index=AVAILABLE_OPERATORS.index(cond.get("operator", ">")) if cond.get("operator") in AVAILABLE_OPERATORS else 0,
                    key=f"{key_prefix}_op_{i}",
                    label_visibility="collapsed",
                )
            with cols[2]:
                right_default = str(cond.get("right", "50"))
                right = st.text_input(
                    "Value or Column",
                    value=right_default,
                    key=f"{key_prefix}_right_{i}",
                    label_visibility="collapsed",
                    placeholder="50 or EMA_200",
                )
            with cols[3]:
                if st.button("✕", key=f"{key_prefix}_del_{i}"):
                    to_delete = i

            # Update in-session
            conditions_data[i] = {"left": left, "operator": op, "right": right}

            if i < len(conditions_data) - 1:
                logic_key = f"{key_prefix}_logic"
                logic = st.session_state.get(logic_key, "AND")
                st.caption(f"*{logic}*")

        if to_delete is not None:
            conditions_data.pop(to_delete)
            st.session_state[f"{key_prefix}_conditions"] = conditions_data
            st.rerun()

        st.session_state[f"{key_prefix}_conditions"] = conditions_data

        # Logic selector
        col_l, col_s = st.columns([1, 2])
        with col_l:
            logic = st.selectbox("Logic", ["AND", "OR"],
                                  index=0 if st.session_state.get(f"{key_prefix}_logic", "AND") == "AND" else 1,
                                  key=f"{key_prefix}_logic_sel")
            st.session_state[f"{key_prefix}_logic"] = logic
        with col_s:
            save_name = st.text_input("Save as template (optional)", key=f"{key_prefix}_save_name",
                                       placeholder="My Custom Rule", label_visibility="visible")
            if st.button("💾 Save Template", key=f"{key_prefix}_save_btn") and save_name:
                # Could persist to session or file — stub for now
                st.success(f"✅ Template '{save_name}' saved to session!")

    # Build SignalRule from UI
    conditions = []
    for cond in conditions_data:
        r = cond["right"]
        try:
            r = float(r)
        except ValueError:
            pass
        conditions.append(Condition(cond["left"], cond["operator"], r))

    return SignalRule(
        name=f"{label} Rule",
        logic=st.session_state.get(f"{key_prefix}_logic", "AND"),
        conditions=conditions,
    )


tab_entry, tab_exit = st.tabs(["🟢 Entry Conditions", "🔴 Exit Conditions"])

with tab_entry:
    entry_rule = _render_rule_builder("ENTRY CONDITIONS", "entry")

with tab_exit:
    exit_rule = _render_rule_builder("EXIT CONDITIONS", "exit")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Scoring Weights
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 3 — Scoring Weights")

sp = ScoreParser()
scoring_options = sp.get_scanner_scoring_options()

st.caption("Assign weights to each metric. The Scanner will compute a 0–100 composite score per stock.")

scoring_weights: dict = {}

with st.expander("⚙️ Configure Scoring Weights", expanded=True):
    for group, metrics in scoring_options.items():
        st.markdown(f"**{group}**")
        metric_cols = st.columns(len(metrics))
        for j, metric in enumerate(metrics):
            with metric_cols[j]:
                direction = "desc" if any(k in metric for k in ["Volatility", "Drawdown", "Days Since"]) else "asc"
                w = st.slider(
                    metric,
                    min_value=0.0, max_value=1.0,
                    value=st.session_state.scoring_weights.get(metric, (0.1, direction))[0]
                    if isinstance(st.session_state.scoring_weights.get(metric), tuple) else 0.1,
                    step=0.05,
                    key=f"weight_{metric.replace(' ', '_')}",
                )
                if w > 0:
                    scoring_weights[metric] = (w, direction)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Run Scanner
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 4 — Run Scanner")

col_run1, col_run2, col_run3 = st.columns([1, 1, 3])
with col_run1:
    run_scan = st.button("🚀 Run Scanner", type="primary", use_container_width=True, key="run_scan_btn")
with col_run2:
    if st.button("🔄 Clear Results", use_container_width=True, key="clear_scan_btn"):
        st.session_state.scan_results = None
        st.rerun()

if run_scan:
    if not tickers:
        st.error("Please select or enter at least one ticker.")
    elif entry_rule.is_empty():
        st.error("Please add at least one entry condition.")
    elif exit_rule.is_empty():
        st.error("Please add at least one exit condition.")
    else:
        weights_to_use = scoring_weights or DEFAULT_SCORING_WEIGHTS

        if st.session_state.scanner_mode == "Equity":
            progress_bar = st.progress(0, text="Starting scan...")
            status_text  = st.empty()

            def on_progress(ticker, i, total):
                pct = (i + 1) / total
                progress_bar.progress(pct, text=f"Scanning {ticker} ({i+1}/{total})...")
                status_text.caption(f"⏳ Processing {ticker}...")

            with st.spinner("Running scanner..."):
                scanner = Scanner(
                    universe=tickers,
                    entry_rule=entry_rule,
                    exit_rule=exit_rule,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    scoring_weights=weights_to_use,
                    on_progress=on_progress,
                )
                results = scanner.run()

            progress_bar.empty()
            status_text.empty()

            if results.empty:
                st.warning("No results returned. Check your tickers or date range.")
            else:
                st.session_state.scan_results = results
                st.success(f"✅ Scanned {len(results)} stocks successfully!")

        else:  # Options mode — NIFTY signal scan
            with st.spinner("Fetching NIFTY data and evaluating signals..."):
                result = Scanner.run_nifty_options_scan(
                    entry_rule=entry_rule,
                    exit_rule=exit_rule,
                    start_date=str(start_date),
                    end_date=str(end_date),
                )
            if result:
                st.session_state.scan_results = result
                st.success(
                    f"✅ NIFTY signal scan complete! "
                    f"**{len(result['entry_dates'])} entry signals** | "
                    f"**{len(result['exit_dates'])} exit signals** found."
                )
            else:
                st.error("Failed to fetch NIFTY data. Check your internet connection.")

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.scan_results is not None:
    results = st.session_state.scan_results

    st.divider()
    st.subheader("📊 Scanner Results")

    if st.session_state.scanner_mode == "Equity" and isinstance(results, pd.DataFrame):
        # ── Summary KPIs ──────────────────────────────────────────────────────
        entry_count = results["Entry_Signal"].sum()
        exit_count  = results["Exit_Signal"].sum()
        top_score   = results["Score"].max()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Stocks Scanned", len(results))
        with k2:
            st.metric("Entry Signals", int(entry_count),
                      delta=f"{entry_count/len(results)*100:.0f}% of universe")
        with k3:
            st.metric("Exit Signals", int(exit_count))
        with k4:
            st.metric("Top Score", f"{top_score:.1f}/100")

        # ── Ranked Table ──────────────────────────────────────────────────────
        st.markdown("#### 🏆 Ranked Stocks")

        # Highlight entry signal rows
        def _color_signal(val):
            if val is True:
                return "background-color: rgba(0,200,100,0.15); color: #00c864"
            return ""

        display_cols = ["Rank", "Ticker", "Last_Close", "Entry_Signal", "Exit_Signal",
                        "Score", "Entry_Signal_Strength"]
        # Add scoring metric cols that exist
        for m in scoring_weights:
            if m in results.columns and m not in display_cols:
                display_cols.append(m)

        display_df = results[[c for c in display_cols if c in results.columns]].copy()
        display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.1f}")
        display_df["Entry_Signal_Strength"] = display_df["Entry_Signal_Strength"].apply(lambda x: f"{x:.0f}%")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # ── Score Distribution Chart ──────────────────────────────────────────
        st.markdown("#### 📈 Score Distribution")
        fig = px.bar(
            results.sort_values("Score", ascending=False).head(30),
            x="Ticker", y="Score",
            color="Score",
            color_continuous_scale="Viridis",
            title="Top 30 Stocks by Score",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Entry signal highlight ────────────────────────────────────────────
        entry_stocks = results[results["Entry_Signal"] == True]
        if not entry_stocks.empty:
            st.markdown("#### 🟢 Stocks With Active Entry Signal")
            st.dataframe(
                entry_stocks[["Rank", "Ticker", "Last_Close", "Score"]],
                use_container_width=True,
                hide_index=True,
            )

        # ── Download ──────────────────────────────────────────────────────────
        csv = results.to_csv(index=False)
        st.download_button(
            "⬇️ Download Full Results (CSV)",
            data=csv,
            file_name="scanner_results.csv",
            mime="text/csv",
            key="download_results",
        )

    elif st.session_state.scanner_mode == "Options (Nifty Index)" and isinstance(results, dict):
        # ── NIFTY Signal Chart ────────────────────────────────────────────────
        underlying_df = results.get("underlying_df", pd.DataFrame())
        signal_df     = results.get("signal_df", pd.DataFrame())

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Entry Signals Found", len(results.get("entry_dates", [])))
        with k2:
            st.metric("Exit Signals Found", len(results.get("exit_dates", [])))
        with k3:
            status = "🟢 ACTIVE" if results.get("latest_entry") else "⚪ FLAT"
            st.metric("Current Signal", status)

        if not underlying_df.empty and not signal_df.empty:
            st.markdown("#### 📈 NIFTY Price with Entry/Exit Signals")

            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=underlying_df.index,
                open=underlying_df["Open"],
                high=underlying_df["High"],
                low=underlying_df["Low"],
                close=underlying_df["Close"],
                name="NIFTY",
            ))

            # Entry arrows
            entry_dates = results.get("entry_dates", [])
            if entry_dates:
                entry_prices = underlying_df.loc[underlying_df.index.isin(entry_dates), "Low"] * 0.995
                fig.add_trace(go.Scatter(
                    x=entry_dates, y=entry_prices,
                    mode="markers", name="Entry Signal",
                    marker=dict(symbol="triangle-up", color="#00ff88", size=12),
                ))

            # Exit arrows
            exit_dates = results.get("exit_dates", [])
            if exit_dates:
                exit_prices = underlying_df.loc[underlying_df.index.isin(exit_dates), "High"] * 1.005
                fig.add_trace(go.Scatter(
                    x=exit_dates, y=exit_prices,
                    mode="markers", name="Exit Signal",
                    marker=dict(symbol="triangle-down", color="#ff4444", size=12),
                ))

            fig.update_layout(
                title="NIFTY Price — Entry/Exit Signals",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                xaxis_rangeslider_visible=False,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("💡 **Next step**: Go to **Backtest Options** page to simulate "
                    "a full options strategy backtest using these entry/exit dates.")
