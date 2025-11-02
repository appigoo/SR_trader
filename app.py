# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime

# ==================== 設定 ====================
st.set_page_config(page_title="股票突破監控", layout="wide")
st.title("股票支撐 / 阻力突破監控系統")

# session_state
for k, v in {"last_update": 0, "last_signal": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== 側邊欄 ====================
symbol = st.sidebar.text_input("股票代號", "TSLA").upper().strip()
interval = st.sidebar.selectbox("時間週期", ["1m", "5m", "15m", "30m", "1h", "1d"], index=1)
lookback = st.sidebar.slider("觀察根數", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "3分鐘", "5分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", True)
use_volume_filter = st.sidebar.checkbox("成交量確認 (>1.5x)", True)
buffer_pct = st.sidebar.slider("緩衝區 (%)", 0.05, 1.0, 0.1, 0.05) / 100

# ==================== Telegram ====================
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
except Exception:
    BOT_TOKEN = CHAT_ID = None
    st.sidebar.error("Telegram 設定錯誤")

def send_telegram_alert(msg: str) -> bool:
    if not (BOT_TOKEN and CHAT_ID) or st.session_state.last_signal == msg:
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
        st.session_state.last_signal = msg
        st.toast("Telegram 已發送", icon="success")
        return True
    except:
        return False

# ==================== 支撐阻力 ====================
def find_support_resistance_fractal(df: pd.DataFrame, window: int = 5):
    if len(df) < window * 2 + 1:
        return df["Low"].min(), df["High"].max()

    high, low = df["High"], df["Low"]
    res_pts, sup_pts = [], []

    for i in range(window, len(df) - window):
        sl = slice(i - window, i + window + 1)
        if high.iloc[i] == high.iloc[sl].max():
            res_pts.append(high.iloc[i])
        if low.iloc[i] == low.iloc[sl].min():
            sup_pts.append(low.iloc[i])

    def cluster(prices, tol=0.005):
        if not prices: return []
        prices = sorted(prices)
        clusters, cur = [], [prices[0]]
        for p in prices[1:]:
            if abs(p - cur[-1]) / cur[-1] < tol:
                cur.append(p)
            else:
                clusters.append(np.mean(cur))
                cur = [p]
        clusters.append(np.mean(cur))
        return clusters

    res_lv = cluster(res_pts) or [high.max()]
    sup_lv = cluster(sup_pts) or [low.min()]
    cur = df["Close"].iloc[-1]
    resistance = max(res_lv, key=lambda x: (-abs(x - cur), x))
    support = min(sup_lv, key=lambda x: (-abs(x - cur), -x))
    return float(support), float(resistance)

# ==================== 突破偵測 ====================
def detect_breakout(df: pd.DataFrame, support: float, resistance: float,
                    buffer_pct: float, use_volume: bool, vol_mult: float, lookback: int):
    if len(df) < 2:
        return None

    # 關鍵：強制取單一值，防 Series
    try:
        last_close = float(df["Close"].iat[-1])   # .iat 比 .iloc 更安全
        prev_close = float(df["Close"].iat[-2])
        last_volume = float(df["Volume"].iat[-1])
    except Exception:
        return None

    vol_tail = df["Volume"].tail(lookback)
    avg_volume = float(vol_tail.mean()) if not vol_tail.empty else last_volume
    if avg_volume == 0:
        avg_volume = 1

    buffer = max(support, resistance) * buffer_pct

    up   = prev_close <= resistance - buffer and last_close > resistance
    down = prev_close >= support + buffer and last_close < support
    vol_ok = (not use_volume) or (last_volume > avg_volume * vol_mult)

    ratio = last_volume / avg_volume

    if up and vol_ok:
        return f"突破阻力！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n阻力: {resistance:.2f}\n成交量: {last_volume/1e6:.1f}M ({ratio:.1f}x)"
    if down and vol_ok:
        return f"跌破支撐！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n支撐: {support:.2f}\n成交量: {last_volume/1e6:.1f}M ({ratio:.1f}x)"
    return None

# ==================== 主程式 ====================
def load_and_update_data():
    try:
        period_map = {"1m": "2d", "5m": "5d", "15m": "10d", "30m": "20d", "1h": "1mo", "1d": "3mo"}
        period = period_map.get(interval, "5d")

        with st.spinner(f"下載 {symbol}..."):
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty:
                st.error("無資料")
                return

            # 關鍵：移除重複索引
            df = df[~df.index.duplicated(keep='last')].copy()
            df.dropna(inplace=True)

            if len(df) < 10:
                st.warning("資料不足")
                return

        support, resistance = find_support_resistance_fractal(df, window=max(3, lookback // 20))
        signal = detect_breakout(df, support, resistance, buffer_pct, use_volume_filter, 1.5, lookback)

        # 繪圖
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K線"))
        fig.add_hline(y=support, line_dash="dot", line_color="green", annotation_text=f"支撐 {support:.2f}")
        fig.add_hline(y=resistance, line_dash="dot", line_color="red", annotation_text=f"阻力 {resistance:.2f}")
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="成交量", marker_color="lightblue", yaxis="y2"))
        fig.update_layout(
            title=f"{symbol} {interval} | {datetime.now():%H:%M:%S}",
            height=700, xaxis_rangeslider_visible=False,
            yaxis=dict(title="價格"), yaxis2=dict(title="成交量", overlaying="y", side="right")
        )
        st.plotly_chart(fig, use_container_width=True)

        # 資訊
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("現價", f"{df['Close'].iat[-1]:.2f}")
        with c2: st.metric("支撐", f"{support:.2f}", f"{df['Close'].iat[-1]-support:+.2f}")
        with c3: st.metric("阻力", f"{resistance:.2f}", f"{resistance-df['Close'].iat[-1]:+.2f}")

        # 訊號
        if signal:
            st.success("突破！")
            st.markdown(signal)
            send_telegram_alert(signal)
        else:
            st.info("無訊號")

    except Exception as e:
        st.error(f"錯誤：{e}")

# ==================== 自動更新 ====================
interval_map = {"30秒": 30, "60秒": 60, "3分鐘": 180, "5分鐘": 300}
refresh_seconds = interval_map[update_freq]

if auto_update:
    now = time.time()
    remaining = int(refresh_seconds - (now - st.session_state.last_update))
    if remaining <= 0:
        st.session_state.last_update = now
        st.rerun()
    else:
        st.sidebar.caption(f"下次更新：{max(0, remaining)} 秒")
else:
    st.sidebar.caption("手動模式")

# 執行
load_and_update_data()
