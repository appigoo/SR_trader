import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime

# ========== 基本設定 ==========
st.set_page_config(page_title="股票支撐/阻力突破監控系統", layout="wide")
st.title("股票支撐 / 阻力突破交易監控系統")

# 初始化 session_state
if "last_update" not in st.session_state:
    st.session_state.last_update = 0
if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

# ========== 側邊欄設定 ==========
symbol = st.sidebar.text_input("股票代號", value="TSLA").upper().strip()
interval = st.sidebar.selectbox("K線時間週期", ["1m", "5m", "15m", "30m", "1h", "1d"], index=1)
lookback = st.sidebar.slider("觀察K線根數", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "3分鐘", "5分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", value=True)
use_volume_filter = st.sidebar.checkbox("啟用成交量放大確認（> 均量 1.5 倍）", value=True)
buffer_pct = st.sidebar.slider("突破緩衝區（%）", 0.05, 1.0, 0.1, 0.05) / 100

# ========== Telegram 設定 ==========
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
except Exception:
    BOT_TOKEN = None
    CHAT_ID = None
    st.sidebar.error("無法讀取 Telegram 設定，請檢查 `.streamlit/secrets.toml`")

def send_telegram_alert(message):
    """安全推播 + 防重複"""
    if not BOT_TOKEN or not CHAT_ID:
        return False
    if st.session_state.last_signal == message:
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        params = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            st.session_state.last_signal = message
            st.toast("Telegram 通知已發送", icon="success")
            return True
    except Exception as e:
        st.error(f"推播失敗：{e}")
    return False

# ========== 改良版：分型 + 群聚平均 計算支撐/阻力 ==========
def find_support_resistance_fractal(df, window=5):
    if len(df) < window * 2 + 1:
        return df["Low"].min(), df["High"].max()

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    resistance_points = []
    support_points = []

    for i in range(window, len(df) - window):
        window_slice = slice(i - window, i + window + 1)
        # 阻力：中間最高
        if high.iloc[i] == high.iloc[window_slice].max():
            resistance_points.append(high.iloc[i])
        # 支撐：中間最低
        if low.iloc[i] == low.iloc[window_slice].min():
            support_points.append(low.iloc[i])

    # 群聚平均
    def cluster_levels(prices, tol=0.005):
        if not prices:
            return []
        prices = sorted(prices)
        clusters = []
        current = [prices[0]]
        for p in prices[1:]:
            if abs(p - current[-1]) / current[-1] < tol:
                current.append(p)
            else:
                clusters.append(np.mean(current))
                current = [p]
        clusters.append(np.mean(current))
        return clusters

    res_levels = cluster_levels(resistance_points) if resistance_points else [high.max()]
    sup_levels = cluster_levels(support_points) if support_points else [low.min()]

    # 選最近且強勢的水平
    current_price = close.iloc[-1]
    final_resistance = max(res_levels, key=lambda x: (-abs(x - current_price), x))
    final_support = min(sup_levels, key=lambda x: (-abs(x - current_price), -x))

    return float(final_support), float(final_resistance)

# ========== 突破偵測 ==========
def detect_breakout(df, support, resistance, buffer_pct=0.001, use_volume=False, vol_mult=1.5, lookback=100):
    if len(df) < 2:
        return None

    # 強制取單一值，避免 Series 錯誤
    try:
        last_close = df["Close"].iloc[-1].item()
        prev_close = df["Close"].iloc[-2].item()
        last_volume = df["Volume"].iloc[-1].item()
    except (IndexError, AttributeError):
        return None

    # 計算均量
    vol_tail = df["Volume"].tail(lookback)
    avg_volume = vol_tail.mean() if not vol_tail.empty else last_volume

    buffer = max(support, resistance) * buffer_pct

    breakout_up = prev_close <= resistance - buffer and last_close > resistance
    breakout_down = prev_close >= support + buffer and last_close < support
    volume_condition = (not use_volume) or (last_volume > avg_volume * vol_mult)

    if breakout_up and volume_condition:
        vol_ratio = last_volume / avg_volume if avg_volume > 0 else 1
        return f"突破阻力！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n阻力: {resistance:.2f}\n成交量: {last_volume/1e6:.1f}M ({vol_ratio:.1f}x 均量)"
    elif breakout_down and volume_condition:
        vol_ratio = last_volume / avg_volume if avg_volume > 0 else 1
        return f"跌破支撐！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n支撐: {support:.2f}\n成交量: {last_volume/1e6:.1f}M ({vol_ratio:.1f}x 均量)"
    return None

# ========== 主程式 ==========
def load_and_update_data():
    try:
        # 動態 period
        period_map = {
            "1m": "2d", "5m": "5d", "15m": "10d", "30m": "20d",
            "1h": "1mo", "1d": "3mo"
        }
        period = period_map.get(interval, "5d")

        with st.spinner(f"正在下載 {symbol} 資料 ({interval})..."):
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)

            if df.empty:
                st.error(f"無法取得 {symbol} 資料，請確認代號或網路。")
                return

            # 關鍵：移除重複索引
            df = df[~df.index.duplicated(keep='last')]
            df.dropna(inplace=True)

            if len(df) < 10:
                st.warning("資料不足，無法分析。")
                return

        # 計算支撐阻力
        support, resistance = find_support_resistance_fractal(df, window=max(3, lookback // 20))

        # 突破偵測
        signal = detect_breakout(
            df, support, resistance,
            buffer_pct=buffer_pct,
            use_volume=use_volume_filter,
            vol_mult=1.5,
            lookback=lookback
        )

        # === 繪圖 ===
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="K線"
        ))

        fig.add_hline(y=support, line_dash="dot", line_color="green",
                      annotation_text=f"支撐 {support:.2f}", annotation_position="bottom right")
        fig.add_hline(y=resistance, line_dash="dot", line_color="red",
                      annotation_text=f"阻力 {resistance:.2f}", annotation_position="top right")

        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="成交量", marker_color="rgba(100,150,255,0.3)", yaxis="y2"
        ))

        fig.update_layout(
            title=f"{symbol} {interval} 走勢圖 | 更新: {datetime.now().strftime('%H:%M:%S')}",
            height=700,
            xaxis_rangeslider_visible=False,
            yaxis=dict(title="價格"),
            yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False),
            legend=dict(x=0, y=1.1, orientation="h")
        )

        st.plotly_chart(fig, use_container_width=True)

        # === 資訊欄 ===
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("現價", f"{df['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("支撐位", f"{support:.2f}", delta=f"{df['Close'].iloc[-1]-support:+.2f}")
        with col3:
            st.metric("阻力位", f"{resistance:.2f}", delta=f"{resistance-df['Close'].iloc[-1]:+.2f}")

        # === 訊號 ===
        if signal:
            st.success("突破訊號觸發！")
            st.markdown(signal)
            send_telegram_alert(signal)
        else:
            st.info("尚未出現突破訊號")

        st.caption(f"觀察根數: {lookback} | 緩衝: {buffer_pct*100:.2f}% | "
                   f"成交量條件: {'開啟' if use_volume_filter else '關閉'}")

    except Exception as e:
        st.error(f"程式錯誤：{e}")
        st.code(f"除錯資訊：\n{e.__class__.__name__}: {e}")

# ========== 自動更新控制 ==========
interval_map = {"30秒": 30, "60秒": 60, "3分鐘": 180, "5分鐘": 300}
refresh_seconds = interval_map[update_freq]

# 主流程
if auto_update:
    current_time = time.time()
    remaining = int(refresh_seconds - (current_time - st.session_state.last_update))
    if remaining <= 0:
        st.session_state.last_update = current_time
        st.rerun()
    else:
        st.sidebar.caption(f"下次更新：{max(0, remaining)} 秒")
else:
    st.sidebar.caption("自動更新已關閉")

# 執行主程式
load_and_update_data()
