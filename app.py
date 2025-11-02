# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
from typing import List, Tuple, Optional

# ==================== 初始化 ====================
st.set_page_config(page_title="多股票突破監控", layout="wide")
st.title("多股票支撐 / 阻力突破監控系統")

# session_state
for key in ["last_update", "last_signal_keys", "signal_history"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key == "last_update" else {} if key == "last_signal_keys" else []

# ==================== 側邊欄 ====================
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "AAPL, TSLA, NVDA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval = st.sidebar.selectbox("時間週期", ["1m", "5m", "15m", "30m", "1h", "1d"], index=1)
lookback = st.sidebar.slider("觀察根數", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "3分鐘", "5分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", True)
use_volume_filter = st.sidebar.checkbox("成交量確認 (>1.5x)", True)
buffer_pct = st.sidebar.slider("緩衝區 (%)", 0.05, 1.0, 0.1, 0.05) / 100
sound_alert = st.sidebar.checkbox("聲音提醒", True)

# ==================== Telegram ====================
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
except Exception:
    BOT_TOKEN = CHAT_ID = None
    st.sidebar.error("Telegram 設定錯誤")

def send_telegram_alert(msg: str) -> bool:
    if not (BOT_TOKEN and CHAT_ID):
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
        st.toast("Telegram 已發送", icon="success")
        return True
    except Exception as e:
        st.toast(f"Telegram 失敗: {e}", icon="error")
        return False

# ==================== 聲音提醒 ====================
def play_alert_sound():
    if sound_alert:
        st.markdown("""
        <audio autoplay style="display:none;">
            <source src="https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

# ==================== 支撐阻力（終極防呆版） ====================
def find_support_resistance_fractal(df: pd.DataFrame, window: int = 5, min_touches: int = 2):
    if len(df) < window * 2 + 1:
        return float(df["Low"].min(skipna=True)), float(df["High"].max(skipna=True))

    high, low = df["High"], df["Low"]
    res_pts, sup_pts = [], []

    for i in range(window, len(df) - window):
        sl = slice(i - window, i + window + 1)
        segment_high = high.iloc[sl]
        segment_low = low.iloc[sl]

        if segment_high.empty or segment_low.empty:
            continue

        # 強制 skipna + float
        max_high = float(segment_high.max(skipna=True))
        min_low = float(segment_low.min(skipna=True))

        if not (np.isfinite(max_high) and np.isfinite(min_low)):
            continue

        # 使用 np.isclose 避免浮點誤差
        if np.isclose(high.iloc[i], max_high, atol=1e-6):
            res_pts.append((i, max_high))
        if np.isclose(low.iloc[i], min_low, atol=1e-6):
            sup_pts.append((i, min_low))

    def cluster_points(points, tol=0.005):
        if not points:
            return []
        points = sorted(points, key=lambda x: x[1])
        clusters = []
        current = [points[0]]
        for pt in points[1:]:
            if abs(pt[1] - current[-1][1]) / current[-1][1] < tol:
                current.append(pt)
            else:
                if len(current) >= min_touches:
                    clusters.append(np.mean([p[1] for p in current]))
                current = [pt]
        if len(current) >= min_touches:
            clusters.append(np.mean([p[1] for p in current]))
        return clusters

    res_lv = cluster_points(res_pts)
    sup_lv = cluster_points(sup_pts)

    cur = df["Close"].iloc[-1] if len(df) > 0 and np.isfinite(df["Close"].iloc[-1]) else 0
    resistance = max(res_lv, key=lambda x: (-abs(x - cur), x)) if res_lv else float(df["High"].max(skipna=True))
    support = min(sup_lv, key=lambda x: (-abs(x - cur), -x)) if sup_lv else float(df["Low"].min(skipna=True))

    return float(support), float(resistance)

# ==================== 突破偵測 ====================
def detect_breakout(df: pd.DataFrame, support: float, resistance: float,
                    buffer_pct: float, use_volume: bool, vol_mult: float, lookback: int, symbol: str):
    if len(df) < 4:
        return None, None

    try:
        last_close = float(df["Close"].iloc[-2])
        prev_close = float(df["Close"].iloc[-3])
        prev2_close = float(df["Close"].iloc[-4]) if len(df) >= 4 else prev_close
        last_volume = float(df["Volume"].iloc[-2])
    except Exception:
        return None, None

    vol_tail = df["Volume"].iloc[-(lookback + 2):-2]
    avg_volume = vol_tail.mean(skipna=True)
    if pd.isna(avg_volume) or avg_volume <= 0:
        avg_volume = 1
    vol_ratio = last_volume / avg_volume
    vol_ok = (not use_volume) or (vol_ratio > vol_mult)

    buffer = max(support, resistance) * buffer_pct

    # 突破阻力
    if (np.isclose(prev2_close, resistance - buffer, atol=1e-4) or prev2_close <= (resistance - buffer)) and \
       (np.isclose(prev_close, resistance - buffer, atol=1e-4) or prev_close <= (resistance - buffer)) and \
       last_close > resistance and vol_ok:
        msg = (f"突破阻力！\n"
               f"股票: <b>{symbol}</b>\n"
               f"現價: <b>{last_close:.2f}</b>\n"
               f"阻力: {resistance:.2f}\n"
               f"成交量: {last_volume/1e6:.1f}M ({vol_ratio:.1f}x)")
        key = f"{symbol}_UP_{resistance:.2f}"
        return msg, key

    # 跌破支撐
    if (np.isclose(prev2_close, support + buffer, atol=1e-4) or prev2_close >= (support + buffer)) and \
       (np.isclose(prev_close, support + buffer, atol=1e-4) or prev_close >= (support + buffer)) and \
       last_close < support and vol_ok:
        msg = (f"跌破支撐！\n"
               f"股票: <b>{symbol}</b>\n"
               f"現價: <b>{last_close:.2f}</b>\n"
               f"支撐: {support:.2f}\n"
               f"成交量: {last_volume/1e6:.1f}M ({vol_ratio:.1f}x)")
        key = f"{symbol}_DN_{support:.2f}"
        return msg, key

    return None, None

# ==================== 資料快取 ====================
@st.cache_data(ttl=60, show_spinner=False, hash_funcs={pd.DataFrame: lambda df: hash(df.to_json())})
def fetch_data(_symbol: str, _interval: str) -> Optional[pd.DataFrame]:
    period_map = {
        "1m": "2d", "5m": "5d", "15m": "10d", "30m": "20d",
        "1h": "1mo", "1d": "3mo"
    }
    period = period_map.get(_interval, "5d")
    try:
        df = yf.download(_symbol, period=period, interval=_interval,
                         progress=False, auto_adjust=True, threads=True)
        if df.empty or df.isna().all().all():
            return None
        df = df[~df.index.duplicated(keep='last')].copy()
        df = df.dropna(how='all')  # 移除全 NaN 列
        return df
    except Exception as e:
        st.warning(f"{_symbol} 下載失敗: {e}")
        return None

# ==================== 主程式 ====================
def process_symbol(symbol: str):
    df = fetch_data(symbol, interval)
    if df is None or len(df) < 15:
        return None, None, None, None, None, None

    df_display = df.copy()
    df = df.iloc[:-1]  # 排除未完成棒
    if len(df) < 10:
        return None, None, None, None, None, None

    window = max(5, lookback // 15)
    support, resistance = find_support_resistance_fractal(df, window=window, min_touches=2)
    signal, signal_key = detect_breakout(df, support, resistance, buffer_pct,
                                         use_volume_filter, 1.5, lookback, symbol)

    # 圖表
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_display.index, open=df_display["Open"], high=df_display["High"],
        low=df_display["Low"], close=df_display["Close"], name="K線"
    ))
    fig.add_hline(y=support, line_dash="dot", line_color="green",
                  annotation_text=f"支撐 {support:.2f}")
    fig.add_hline(y=resistance, line_dash="dot", line_color="red",
                  annotation_text=f"阻力 {resistance:.2f}")
    fig.add_trace(go.Bar(x=df_display.index, y=df_display["Volume"],
                         name="成交量", marker_color="lightblue", yaxis="y2"))

    if signal:
        last_time = df_display.index[-2]
        last_close = df_display["Close"].iloc[-2]
        fig.add_scatter(x=[last_time], y=[last_close], mode="markers",
                        marker=dict(color="yellow", size=12, symbol="star"),
                        name="突破點")

    fig.update_layout(
        title=f"{symbol} {interval}",
        height=600, xaxis_rangeslider_visible=False,
        yaxis=dict(title="價格"), yaxis2=dict(title="成交量", overlaying="y", side="right")
    )

    current_price = df_display["Close"].iloc[-1] if np.isfinite(df_display["Close"].iloc[-1]) else 0
    return fig, current_price, support, resistance, signal, signal_key

# ==================== 執行 ====================
interval_map = {"30秒": 30, "60秒": 60, "3分鐘": 180, "5分鐘": 300}
refresh_seconds = interval_map[update_freq]

# 自動更新
if auto_update:
    now = time.time()
    if now - st.session_state.last_update >= refresh_seconds:
        st.session_state.last_update = now
        time.sleep(1.5)
        st.rerun()
    else:
        remaining = int(refresh_seconds - (now - st.session_state.last_update))
        st.sidebar.caption(f"下次更新：{max(0, remaining)} 秒")
else:
    if st.sidebar.button("手動更新", type="primary"):
        st.rerun()

# 主流程
if not symbols:
    st.warning("請輸入至少一檔股票代號")
    st.stop()

st.header(f"監控中：{', '.join(symbols)}")

# 突破總表
breakout_signals = []
results = {}

with st.spinner("下載資料與分析中…"):
    for symbol in symbols:
        fig, price, support, resistance, signal, key = process_symbol(symbol)
        results[symbol] = {
            "fig": fig, "price": price, "support": support,
            "resistance": resistance, "signal": signal, "key": key
        }
        if signal:
            breakout_signals.append((symbol, signal, key))

# 顯示即時突破
if breakout_signals:
    st.success("即時突破訊號！")
    for sym, sig, key in breakout_signals:
        if st.session_state.last_signal_keys.get(key) != key:
            st.session_state.last_signal_keys[key] = key
            st.session_state.signal_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "symbol": sym,
                "signal": sig
            })
            if len(st.session_state.signal_history) > 20:
                st.session_state.signal_history.pop(0)
            st.markdown(f"**{sym}** → {sig}")
            send_telegram_alert(sig)
            play_alert_sound()
else:
    st.info("無突破訊號")

# 分頁顯示
tabs = st.tabs(symbols)
for tab, symbol in zip(tabs, symbols):
    with tab:
        data = results.get(symbol)
        if not data or data["fig"] is None:
            st.error(f"{symbol} 無資料")
            continue

        st.plotly_chart(data["fig"], use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("現價", f"{data['price']:.2f}")
        with c2: st.metric("支撐", f"{data['support']:.2f}", f"{data['price']-data['support']:+.2f}")
        with c3: st.metric("阻力", f"{data['resistance']:.2f}", f"{data['resistance']-data['price']:+.2f}")

        if data["signal"]:
            st.success("突破！")
            st.markdown(data["signal"])
        else:
            st.info("無訊號")

# 歷史訊號
if st.session_state.signal_history:
    st.subheader("歷史訊號（最近20筆）")
    for s in reversed(st.session_state.signal_history):
        st.markdown(f"**{s['time']} | {s['symbol']}** → {s['signal']}")
