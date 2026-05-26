---
title: Option-Stock Scanner
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.41.1"
app_file: app.py
pinned: true
---

# 📊 Option-Stock Scanner

An advanced **entry/exit signal scanner and backtesting platform** for Indian markets, built on top of the original Put Hedge Scanner. Now supports both **Equity** and **NIFTY Index Options** modes.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Dhan](https://img.shields.io/badge/Dhan_API-Integrated-green?style=for-the-badge)

---

## 🚀 Features

### 🔍 Scanner (New)
- **Asset Class Switch** — toggle between Equity and NIFTY Index Options mode
- **Entry/Exit Signal Builder** — visual drag-and-drop condition composer
  - 8 pre-built strategy templates (EMA Crossover, RSI Reversal, MACD, SuperTrend, Bollinger, Breakout, Nifty Call/Put)
  - Custom rule builder: add any indicator + operator + value, AND/OR logic
  - Save custom rules as named templates
- **Scoring Formula** — weighted percentile rank across universe
- **Ranked Output** — table + score distribution chart

### 📈 Equity Backtest (Existing)
- Nifty50/100/500 and sectoral universe support
- Custom ticker list or CSV upload
- Flexible rebalancing, regime filters, Monte Carlo simulation

### 🎯 Options Backtest (New)
- **Underlying**: NIFTY Index (^NSEI)
- **Strategy Templates**:
  - Single-leg: Buy ATM Call / Buy ATM Put
  - Multi-leg: Bull Call Spread, Bear Put Spread, Long Straddle, Long Strangle
  - Custom: define your own legs
- **Data**: Dhan Rolling Options API (historical premiums) + Black-Scholes + India VIX fallback
- **Outputs**: Equity curve, trade log, P&L bar chart, NIFTY price chart with signals

### 🛡️ Nifty Put Hedge (Original)
- Automatically buys NIFTY ATM Weekly Puts when regime filter triggers
- Delta-neutral lot sizing

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone https://github.com/Hari-sh-S/Option-stock.git
cd Option-stock
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
cp .env.example .env
```

Fill in:
```
DHAN_CLIENT_ID=your_client_id
DHAN_PIN=your_5_digit_pin
```

### 3. Run Locally

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Option-stock/
├── app.py                    # Main Streamlit entry point
├── core/                     # Framework-agnostic business logic
│   ├── indicators.py         # Extended IndicatorLibrary (SMA/EMA/RSI/MACD/BB/SuperTrend/ATR/ADX...)
│   ├── scoring.py            # Extended ScoreParser + weighted percentile scorer
│   ├── signal_builder.py     # Entry/Exit rule engine + 8 templates
│   ├── options_engine.py     # NIFTY options backtest (multi-leg, BS fallback)
│   ├── scanner.py            # Universe scanner + ranked output
│   └── universe.py           # Equity + Nifty Index options universes
├── pages/
│   ├── 1_Scanner.py          # ⭐ NEW: Signal scanner + ranking
│   ├── 3_Backtest_Options.py # ⭐ NEW: Options backtest
│   └── ...                   # Existing pages
├── indicators.py             # Original (kept for backward compatibility)
├── scoring.py                # Original (kept for backward compatibility)
├── portfolio_engine.py       # Portfolio backtesting
└── nifty_put_hedge.py        # Put hedge module
```

---

## 🔐 Dhan Authentication

1. Open the app → click **🔐 Dhan Auth** tab
2. Enter Client ID + PIN → click **💾 Save**
3. Enter TOTP from Google Authenticator → click **🔑 Authenticate**

---

## 📜 Changelog

### v2.0 (Current — Option-stock repo)
- ⭐ Asset class mode switch (Equity / NIFTY Options)
- ⭐ Entry/Exit signal builder with 8 pre-built templates
- ⭐ Visual rule-builder UI (drag-and-drop conditions)
- ⭐ Weighted percentile scoring across universe
- ⭐ NIFTY options backtest (single + multi-leg)
- ⭐ Black-Scholes + India VIX fallback pricer

### v1.0 (investing-scanner-put-hedge)
- Nifty50/500 universe backtesting
- Portfolio engine with regime filters
- Nifty Put Hedge via Dhan API
- Monte Carlo simulation
