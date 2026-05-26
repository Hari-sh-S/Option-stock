"""
core/indicators.py
Extended IndicatorLibrary — includes all existing indicators from the original
indicators.py plus new options-specific and signal-generation helpers.

Fully backward-compatible with portfolio_engine.py and engine.py.
"""
import pandas as pd
import numpy as np
import ta

# ─── SuperTrend helper (module-level, shared by all methods) ─────────────────

def _compute_supertrend_fast(close, upper, lower):
    """Optimized SuperTrend using NumPy (no Python loops where possible)."""
    n = len(close)
    supertrend = np.empty(n)
    supertrend[0] = upper[0]

    for i in range(1, n):
        if close[i] > upper[i - 1]:
            supertrend[i] = lower[i]
        elif close[i] < lower[i - 1]:
            supertrend[i] = upper[i]
        else:
            supertrend[i] = supertrend[i - 1]

    return supertrend


# ─── Main Library ─────────────────────────────────────────────────────────────

class IndicatorLibrary:
    """Static library of technical indicators applied to an OHLCV DataFrame."""

    # ── Basic Indicators ───────────────────────────────────────────────────────

    @staticmethod
    def add_sma(df, window, column='Close'):
        col_data = df[column] if not isinstance(df[column], pd.DataFrame) else df[column].squeeze()
        df[f'SMA_{window}'] = ta.trend.sma_indicator(col_data, window=window)
        return df

    @staticmethod
    def add_ema(df, window, column='Close'):
        col_data = df[column] if not isinstance(df[column], pd.DataFrame) else df[column].squeeze()
        df[f'EMA_{window}'] = ta.trend.ema_indicator(col_data, window=window)
        return df

    @staticmethod
    def add_rsi(df, window=14, column='Close'):
        col_data = df[column] if not isinstance(df[column], pd.DataFrame) else df[column].squeeze()
        df[f'RSI_{window}'] = ta.momentum.rsi(col_data, window=window)
        return df

    @staticmethod
    def add_macd(df, window_slow=26, window_fast=12, window_sign=9, column='Close'):
        col_data = df[column] if not isinstance(df[column], pd.DataFrame) else df[column].squeeze()
        macd = ta.trend.MACD(col_data, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        return df

    @staticmethod
    def add_bollinger_bands(df, window=20, window_dev=2, column='Close'):
        col_data = df[column] if not isinstance(df[column], pd.DataFrame) else df[column].squeeze()
        indicator_bb = ta.volatility.BollingerBands(close=col_data, window=window, window_dev=window_dev)
        df['BB_High'] = indicator_bb.bollinger_hband()
        df['BB_Low'] = indicator_bb.bollinger_lband()
        df['BB_Mid'] = indicator_bb.bollinger_mavg()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']  # New: bandwidth
        df['BB_Pct'] = (col_data - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])  # %B
        return df

    @staticmethod
    def add_atr(df, period=14):
        """Average True Range."""
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        tr = np.maximum.reduce([
            (high - low).values,
            np.abs((high - close.shift(1)).fillna(0).values),
            np.abs((low - close.shift(1)).fillna(0).values)
        ])
        df[f'ATR_{period}'] = pd.Series(tr, index=df.index).ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_stochastic(df, window=14, smooth=3):
        """Stochastic Oscillator %K and %D."""
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        low_min = low.rolling(window).min()
        high_max = high.rolling(window).max()
        df['Stoch_K'] = 100 * (close - low_min) / (high_max - low_min + 1e-9)
        df['Stoch_D'] = df['Stoch_K'].rolling(smooth).mean()
        return df

    @staticmethod
    def add_adx(df, period=14):
        """Average Directional Index (trend strength)."""
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=period)
        df[f'ADX_{period}'] = indicator.adx()
        df['DI_Plus'] = indicator.adx_pos()
        df['DI_Minus'] = indicator.adx_neg()
        return df

    @staticmethod
    def add_supertrend(df, period=7, multiplier=3):
        """Optimized SuperTrend using vectorized NumPy operations."""
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

        tr = np.maximum.reduce([
            (high - low).values,
            np.abs((high - close.shift(1)).values),
            np.abs((low - close.shift(1)).values)
        ])
        atr = pd.Series(tr, index=df.index).ewm(span=period).mean()
        hl2 = (high + low) / 2
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        supertrend = _compute_supertrend_fast(close.values, upper.values, lower.values)
        df['Supertrend'] = supertrend
        df['Supertrend_Signal'] = np.where(close.values > supertrend, 1, -1)
        return df

    @staticmethod
    def add_volume_indicators(df):
        """Volume-based indicators: OBV, Volume MA, Volume Ratio."""
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
        df['OBV'] = ta.volume.on_balance_volume(close, volume)
        df['Volume_MA20'] = volume.rolling(20).mean()
        df['Volume_Ratio'] = volume / df['Volume_MA20']
        return df

    # ── Momentum & Volatility Metrics (existing, unchanged) ───────────────────

    @staticmethod
    def add_momentum_volatility_metrics(df, required_periods=None):
        """
        FAST vectorized Performance and Risk metrics for multiple timeframes.
        Uses pure NumPy operations for maximum speed.
        """
        if isinstance(df, pd.Series):
            raise ValueError("Input must be a DataFrame, not a Series")

        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have a 'Close' column")

        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

        active_periods = {(1, 'Month'), (3, 'Month'), (6, 'Month'), (9, 'Month'), (12, 'Month')}

        if required_periods:
            for item in required_periods:
                if len(item) == 3:
                    val, unit, _ = item
                    active_periods.add((val, unit))
                elif len(item) == 2:
                    val, _ = item
                    active_periods.add((val, 'Month'))

        periods = {}
        for val, unit in active_periods:
            if unit == 'Month':
                name = '1 Year' if val == 12 else f'{val} Month'
                window = val * 21
            elif unit == 'Week':
                name = f'{val} Week'
                window = val * 5
            else:
                continue
            periods[name] = window

        daily_returns = close.pct_change()
        if 'Daily_Returns' not in df.columns:
            df['Daily_Returns'] = daily_returns

        for name, window in periods.items():
            if f'{name} Performance' not in df.columns:
                df[f'{name} Performance'] = close.pct_change(periods=window)
            if f'{name} Volatility' not in df.columns:
                df[f'{name} Volatility'] = daily_returns.rolling(window).std() * np.sqrt(252)
            if f'{name} Downside Volatility' not in df.columns:
                downside = daily_returns.clip(upper=0)
                df[f'{name} Downside Volatility'] = downside.rolling(window).std() * np.sqrt(252)
            if f'{name} Max Drawdown' not in df.columns:
                rolling_max = close.rolling(window).max()
                drawdown = (close - rolling_max) / rolling_max
                df[f'{name} Max Drawdown'] = drawdown.rolling(window).min()
            if f'{name} Sharpe' not in df.columns:
                mean_ret = daily_returns.rolling(window).mean()
                std_ret = daily_returns.rolling(window).std()
                df[f'{name} Sharpe'] = ((mean_ret / std_ret) * np.sqrt(252)).replace([np.inf, -np.inf], 0)
            if f'{name} Sortino' not in df.columns:
                downside = daily_returns.clip(upper=0)
                mean_ret = daily_returns.rolling(window).mean()
                downside_std = downside.rolling(window).std()
                df[f'{name} Sortino'] = ((mean_ret / downside_std) * np.sqrt(252)).replace([np.inf, -np.inf], 0)
            if f'{name} Calmar' not in df.columns:
                calmar = df[f'{name} Performance'] / df[f'{name} Max Drawdown'].abs()
                df[f'{name} Calmar'] = calmar.replace([np.inf, -np.inf], 0)
            if f'{name} Positive Days' not in df.columns:
                df[f'{name} Positive Days'] = (daily_returns > 0).astype(float).rolling(window).mean()
            if f'{name} Negative Days' not in df.columns:
                df[f'{name} Negative Days'] = (daily_returns < 0).astype(float).rolling(window).mean()
            if f'{name} Distance From High' not in df.columns:
                rolling_high = close.rolling(window).max()
                df[f'{name} Distance From High'] = (rolling_high - close) / close
            if f'{name} Distance From Low' not in df.columns:
                rolling_low = close.rolling(window).min()
                df[f'{name} Distance From Low'] = ((close - rolling_low) / rolling_low).replace([np.inf, -np.inf], 0)

        df.fillna(0, inplace=True)
        return df

    # ── Regime Filters (existing, unchanged) ──────────────────────────────────

    @staticmethod
    def add_regime_filters(df, supertrend_period=7, supertrend_multiplier=3, sma_period=50, ema_period=68):
        """Optimized regime indicators — SMA/EMA/MACD/SuperTrend across 1D/1W/1M."""
        if isinstance(df, pd.Series):
            raise ValueError("Input must be a DataFrame, not a Series")

        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']

        if df.index.name != 'Date':
            df_temp = df.copy()
            if 'Date' in df.columns:
                df_temp = df_temp.set_index('Date')
        else:
            df_temp = df

        # SMA
        df[f'SMA_{sma_period}'] = close.rolling(sma_period).mean()
        df['SMA_1D'] = df[f'SMA_{sma_period}']
        df['SMA_1D_Direction'] = np.where(close > df['SMA_1D'], 'BUY', 'SELL')

        try:
            weekly_close = df_temp['Close'].resample('W').last().dropna()
            if len(weekly_close) >= sma_period:
                weekly_sma = weekly_close.rolling(sma_period).mean()
                df['SMA_1W'] = weekly_sma.reindex(df_temp.index, method='ffill')
                df['SMA_1W_Direction'] = np.where(close > df['SMA_1W'], 'BUY', 'SELL')
            else:
                df['SMA_1W'] = df['SMA_1D']
                df['SMA_1W_Direction'] = df['SMA_1D_Direction']
        except Exception:
            df['SMA_1W'] = df['SMA_1D']
            df['SMA_1W_Direction'] = df['SMA_1D_Direction']

        try:
            monthly_close = df_temp['Close'].resample('ME').last().dropna()
            if len(monthly_close) >= min(sma_period, 12):
                monthly_sma = monthly_close.rolling(min(sma_period, 12)).mean()
                df['SMA_1M'] = monthly_sma.reindex(df_temp.index, method='ffill')
                df['SMA_1M_Direction'] = np.where(close > df['SMA_1M'], 'BUY', 'SELL')
            else:
                df['SMA_1M'] = df['SMA_1D']
                df['SMA_1M_Direction'] = df['SMA_1D_Direction']
        except Exception:
            df['SMA_1M'] = df['SMA_1D']
            df['SMA_1M_Direction'] = df['SMA_1D_Direction']

        # EMA
        for period in [34, 68, 100, 150, 200]:
            df[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
        df['EMA_1D'] = close.ewm(span=ema_period, adjust=False).mean()
        df['EMA_1D_Direction'] = np.where(close > df['EMA_1D'], 'BUY', 'SELL')

        try:
            weekly_close = df_temp['Close'].resample('W').last().dropna()
            if len(weekly_close) >= ema_period:
                weekly_ema = weekly_close.ewm(span=ema_period, adjust=False).mean()
                df['EMA_1W'] = weekly_ema.reindex(df_temp.index, method='ffill')
                df['EMA_1W_Direction'] = np.where(close > df['EMA_1W'], 'BUY', 'SELL')
            else:
                df['EMA_1W'] = df['EMA_1D']
                df['EMA_1W_Direction'] = df['EMA_1D_Direction']
        except Exception:
            df['EMA_1W'] = df['EMA_1D']
            df['EMA_1W_Direction'] = df['EMA_1D_Direction']

        try:
            monthly_close = df_temp['Close'].resample('ME').last().dropna()
            if len(monthly_close) >= min(ema_period, 12):
                monthly_ema = monthly_close.ewm(span=min(ema_period, 12), adjust=False).mean()
                df['EMA_1M'] = monthly_ema.reindex(df_temp.index, method='ffill')
                df['EMA_1M_Direction'] = np.where(close > df['EMA_1M'], 'BUY', 'SELL')
            else:
                df['EMA_1M'] = df['EMA_1D']
                df['EMA_1M_Direction'] = df['EMA_1D_Direction']
        except Exception:
            df['EMA_1M'] = df['EMA_1D']
            df['EMA_1M_Direction'] = df['EMA_1D_Direction']

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']

        # SuperTrend 1D
        st_period = supertrend_period
        st_mult = supertrend_multiplier
        tr = np.maximum.reduce([
            (high - low).values,
            np.abs((high - close.shift(1)).fillna(0).values),
            np.abs((low - close.shift(1)).fillna(0).values)
        ])
        atr = pd.Series(tr, index=df.index).ewm(span=st_period).mean()
        hl2 = (high + low) / 2
        upper = hl2 + (st_mult * atr)
        lower = hl2 - (st_mult * atr)
        supertrend = _compute_supertrend_fast(close.values, upper.values, lower.values)
        df['Supertrend'] = supertrend
        df['Supertrend_Direction'] = np.where(close.values > supertrend, 'BUY', 'SELL')
        df['Supertrend_1D'] = supertrend
        df['Supertrend_1D_Direction'] = df['Supertrend_Direction']

        # SuperTrend 1W
        try:
            weekly_ohlc = df_temp[['Open', 'High', 'Low', 'Close']].resample('W').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            if len(weekly_ohlc) >= 10:
                w_h = weekly_ohlc['High']
                w_l = weekly_ohlc['Low']
                w_c = weekly_ohlc['Close']
                w_tr = np.maximum.reduce([
                    (w_h - w_l).values,
                    np.abs((w_h - w_c.shift(1)).fillna(0).values),
                    np.abs((w_l - w_c.shift(1)).fillna(0).values)
                ])
                w_atr = pd.Series(w_tr, index=weekly_ohlc.index).ewm(span=st_period).mean()
                w_hl2 = (w_h + w_l) / 2
                w_upper = w_hl2 + (st_mult * w_atr)
                w_lower = w_hl2 - (st_mult * w_atr)
                w_st = _compute_supertrend_fast(w_c.values, w_upper.values, w_lower.values)
                weekly_ohlc['Supertrend_1W'] = w_st
                weekly_ohlc['Supertrend_1W_Direction'] = np.where(w_c.values > w_st, 'BUY', 'SELL')
                df['Supertrend_1W'] = weekly_ohlc['Supertrend_1W'].reindex(df_temp.index, method='ffill')
                df['Supertrend_1W_Direction'] = weekly_ohlc['Supertrend_1W_Direction'].reindex(df_temp.index, method='ffill')
            else:
                df['Supertrend_1W'] = df['Supertrend']
                df['Supertrend_1W_Direction'] = df['Supertrend_Direction']
        except Exception:
            df['Supertrend_1W'] = df['Supertrend']
            df['Supertrend_1W_Direction'] = df['Supertrend_Direction']

        # SuperTrend 1M
        try:
            monthly_ohlc = df_temp[['Open', 'High', 'Low', 'Close']].resample('ME').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            if len(monthly_ohlc) >= 10:
                m_h = monthly_ohlc['High']
                m_l = monthly_ohlc['Low']
                m_c = monthly_ohlc['Close']
                m_tr = np.maximum.reduce([
                    (m_h - m_l).values,
                    np.abs((m_h - m_c.shift(1)).fillna(0).values),
                    np.abs((m_l - m_c.shift(1)).fillna(0).values)
                ])
                m_atr = pd.Series(m_tr, index=monthly_ohlc.index).ewm(span=st_period).mean()
                m_hl2 = (m_h + m_l) / 2
                m_upper = m_hl2 + (st_mult * m_atr)
                m_lower = m_hl2 - (st_mult * m_atr)
                m_st = _compute_supertrend_fast(m_c.values, m_upper.values, m_lower.values)
                monthly_ohlc['Supertrend_1M'] = m_st
                monthly_ohlc['Supertrend_1M_Direction'] = np.where(m_c.values > m_st, 'BUY', 'SELL')
                df['Supertrend_1M'] = monthly_ohlc['Supertrend_1M'].reindex(df_temp.index, method='ffill')
                df['Supertrend_1M_Direction'] = monthly_ohlc['Supertrend_1M_Direction'].reindex(df_temp.index, method='ffill')
            else:
                df['Supertrend_1M'] = df['Supertrend']
                df['Supertrend_1M_Direction'] = df['Supertrend_Direction']
        except Exception:
            df['Supertrend_1M'] = df['Supertrend']
            df['Supertrend_1M_Direction'] = df['Supertrend_Direction']

        # Key levels
        df['SMA_200'] = close.rolling(200).mean()
        df['Above_SMA_200'] = (close > df['SMA_200']).astype(int)
        df['52W_High'] = close.rolling(252).max()
        df['52W_Low'] = close.rolling(252).min()
        df['Near_52W_High'] = ((close / df['52W_High']) > 0.95).astype(int)
        df['Near_52W_Low'] = ((close / df['52W_Low']) < 1.05).astype(int)
        df['SMA_63'] = close.rolling(63).mean()
        df['SMA_126'] = close.rolling(126).mean()
        df['Bullish_Trend'] = (df['SMA_63'] > df['SMA_126']).astype(int)

        return df

    # ── Donchian / Swing (existing, unchanged) ────────────────────────────────

    @staticmethod
    def add_donchian_channels(df, exit_period=55, recovery_period=20):
        """Donchian channels for regime filter (Turtle Trading rules)."""
        if isinstance(df, pd.Series):
            raise ValueError("Input must be a DataFrame, not a Series")
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        df[f'Donchian_Low_{exit_period}'] = low.rolling(exit_period).min()
        df[f'Donchian_High_{recovery_period}'] = high.rolling(recovery_period).max()
        df['Donchian_Close'] = close
        return df

    @staticmethod
    def add_swing_atr(df, swing_period=20, atr_period=14):
        """Swing pivot levels with ATR buffer."""
        if isinstance(df, pd.Series):
            raise ValueError("Input must be a DataFrame, not a Series")
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        df[f'Swing_Low_{swing_period}'] = low.rolling(swing_period).min()
        df[f'Swing_High_{swing_period}'] = high.rolling(swing_period).max()
        if f'ATR_{atr_period}' not in df.columns:
            tr = np.maximum.reduce([
                (high - low).values,
                np.abs((high - close.shift(1)).fillna(0).values),
                np.abs((low - close.shift(1)).fillna(0).values)
            ])
            df[f'ATR_{atr_period}'] = pd.Series(tr, index=df.index).ewm(span=atr_period, adjust=False).mean()
        df['Swing_Close'] = close
        return df

    @staticmethod
    def _add_supertrend_basic(df, period, multiplier, suffix=""):
        """Simplified supertrend for regime filter."""
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        tr = np.maximum.reduce([
            (high - low).values,
            np.abs((high - close.shift(1)).fillna(0).values),
            np.abs((low - close.shift(1)).fillna(0).values)
        ])
        atr = pd.Series(tr, index=df.index).ewm(span=period, adjust=False).mean()
        hl2 = (high + low) / 2
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        supertrend = _compute_supertrend_fast(close.values, upper.values, lower.values)
        df[f'Supertrend{suffix}'] = supertrend
        return df

    # ── Signal-generation helpers (NEW) ───────────────────────────────────────

    @staticmethod
    def add_crossover_signals(df, fast_col, slow_col, signal_name=None):
        """
        Detect crossover/crossunder events between two series.
        Adds columns:
          <signal_name>_Cross_Above  : 1 on the day fast crosses above slow
          <signal_name>_Cross_Below  : 1 on the day fast crosses below slow
        """
        name = signal_name or f"{fast_col}_vs_{slow_col}"
        fast = df[fast_col]
        slow = df[slow_col]
        df[f'{name}_Cross_Above'] = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).astype(int)
        df[f'{name}_Cross_Below'] = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).astype(int)
        return df

    @staticmethod
    def add_all_for_signal_builder(df, ema_periods=(9, 20, 50, 100, 200),
                                   rsi_period=14, atr_period=14):
        """
        One-shot helper: adds all indicators needed by the Signal Builder.
        Designed to be called once before evaluating any rule combination.
        """
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

        # EMAs
        for p in ema_periods:
            if f'EMA_{p}' not in df.columns:
                df[f'EMA_{p}'] = close.ewm(span=p, adjust=False).mean()

        # SMAs (same periods for flexibility)
        for p in ema_periods:
            if f'SMA_{p}' not in df.columns:
                df = IndicatorLibrary.add_sma(df, p)

        # RSI
        if f'RSI_{rsi_period}' not in df.columns:
            df = IndicatorLibrary.add_rsi(df, rsi_period)

        # MACD
        if 'MACD' not in df.columns:
            df = IndicatorLibrary.add_macd(df)

        # Bollinger Bands
        if 'BB_High' not in df.columns:
            df = IndicatorLibrary.add_bollinger_bands(df)

        # SuperTrend
        if 'Supertrend' not in df.columns:
            df = IndicatorLibrary.add_supertrend(df)

        # ATR
        if f'ATR_{atr_period}' not in df.columns:
            df = IndicatorLibrary.add_atr(df, atr_period)

        # Volume
        if 'Volume' in df.columns and 'Volume_Ratio' not in df.columns:
            df = IndicatorLibrary.add_volume_indicators(df)

        # 52W levels
        if '52W_High' not in df.columns:
            df['52W_High'] = close.rolling(252).max()
            df['52W_Low'] = close.rolling(252).min()

        # EMA crossover signals for common pairs
        if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
            df = IndicatorLibrary.add_crossover_signals(df, 'EMA_20', 'EMA_50', 'EMA_20_50')
        if 'EMA_50' in df.columns and 'EMA_200' in df.columns:
            df = IndicatorLibrary.add_crossover_signals(df, 'EMA_50', 'EMA_200', 'EMA_50_200')
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            df = IndicatorLibrary.add_crossover_signals(df, 'MACD', 'MACD_Signal', 'MACD')

        return df
