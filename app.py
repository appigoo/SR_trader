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
symbol = st.sidebar.text_input("股票代號", value="TSLA").upper()
interval = st.sidebar.selectbox("K線時間週期", ["1m", "5m", "15m", "30m", "1h", "1d"], index=1)
lookback = st.sidebar.slider("觀察K線根數", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "3分鐘", "5分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", value=True)
use_volume_filter = st.sidebar.checkbox("啟用成交量放大確認（> 均量 1.5 倍）", value=True)

# 動態 buffer（百分比）
buffer_pct = st.sidebar.slider("突破緩衝區（%）", 0.05, 1.0, 0.1, 0.05) / 100

# ========== 推播設定 ==========
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
except Exception:
    BOT_TOKEN = None
    CHAT_ID = None
    st.sidebar.error("無法讀取 Telegram 設定，請檢查 secrets.toml")

def send_telegram_alert(message):
    """安全推播 + 避免重複"""
    if not BOT_TOKEN or not CHAT_ID:
        return False
    if st.session_state.last_signal == message:
        return False  # 防止重複推播
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        params = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            st.session_state.last_signal = message
            st.toast("已發送 Telegram 通知", icon="success")
            return True
    except Exception as e:
        st.error(f"推播失敗：{e}")
    return False

# ========== 改良版：分型 + 群聚平均 計算支撐/阻力 ==========
def find_support_resistance_fractal(df, window=20, min_touches=2):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # 找分型高點（中間最高）
    resistance_points = []
    support_points = []

    for i in range(window, len(df) - window):
        # 阻力：中間最高
        if high.iloc[i] == high.iloc[i-window:i+window+1].max():
            resistance_points.append((df.index[i], high.iloc[i]))
        # 支撐：中間最低
        if low.iloc[i] == low.iloc[i-window:i+window+1].min():
            support_points.append((df.index[i], low.iloc[i]))

    # 轉為 DataFrame
    res_df = pd.DataFrame(resistance_points, columns=["date", "price"]) if resistance_points else pd.DataFrame()
    sup_df = pd.DataFrame(support_points, columns=["date", "price"]) if support_points else pd.DataFrame()

    # 群聚平均（價格相近視為同一水平）
    def cluster_levels(levels_df, tol=0.005):
        if levels_df.empty:
            return []
        prices = levels_df["price"].sort_values()
        clusters = []
        current = [prices.iloc[0]]
        for p in prices.iloc[1:]:
            if abs(p - current[-1]) / current[-1] < tol:
                current.append(p)
            else:
                clusters.append(np.mean(current))
                current = [p]
        clusters.append(np.mean(current))
        return clusters

    res_levels = cluster_levels(res_df) if not res_df.empty else [high.max()]
    sup_levels = cluster_levels(sup_df) if not sup_df.empty else [low.min()]

    # 取最近且最強的水平線（出現次數多或最近）
    final_resistance = max(res_levels, key=lambda x: (abs(close.iloc[-1] - x), -abs(x - high.tail(50).mean())))
    final_support = min(sup_levels, key=lambda x: (abs(close.iloc[-1] - x), abs(x - low.tail(50).mean())))

    return final_support, final_resistance

# ========== 突破偵測 ==========
def detect_breakout(df, support, resistance, buffer_pct=0.001, use_volume=False, vol_mult=1.5, lookback=100):
    if len(df) < 2:
        return None

    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    last_volume = df["Volume"].iloc[-1]
    avg_volume = df["Volume"].tail(lookback).mean() if len(df) >= lookback else df["Volume"].mean()

    buffer = max(resistance, support) * buffer_pct

    breakout_up = prev_close <= resistance - buffer and last_close > resistance
    breakout_down = prev_close >= support + buffer and last_close < support
    volume_condition = (not use_volume) or (last_volume > avg_volume * vol_mult)

    if breakout_up and volume_condition:
        return f"突破阻力！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n阻力: {resistance:.2f}\n成交量: {last_volume/1e6:.1f}M ({last_volume/avg_volume:.1f}x 均量)"
    elif breakout_down and volume_condition:
        return f"跌破支撐！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n支撐: {support:.2f}\n成交量: {last_volume/1e6:.1f}M ({last_volume/avg_volume:.1f}x 均量)"
    return None

# ========== 主程式 ==========
def load_and_update_data():
    with st.container():
        try:
            # 根據 interval 調整 period
            period_map = {
                "1m": "2d", "5m": "5d", "15m": "10d", "30m": "20d",
                "1h": "1mo", "1d": "3mo"
            }
            period = period_map.get(interval, "5d")

            with st.spinner(f"正在下載 {symbol} 資料..."):
                df = yf.download(symbol, period=period, interval=interval, progress=False)
                if df.empty:
                    st.error(f"無法取得 {symbol} 資料，請確認代號或網路連線。")
                    return

            df.dropna(inplace=True)
            if len(df) < 10:
                st.warning("資料不足，無法分析。")
                return

            # 計算支撐阻力
            support, resistance = find_support_resistance_fractal(df, window=min(5, lookback//10))

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

            # K線
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="K線"
            ))

            # 支撐阻力線
            fig.add_hline(y=support, line_dash="dot", line_color="green",
                          annotation_text=f"支撐 {support:.2f}", annotation_position="bottom right")
            fig.add_hline(y=resistance, line_dash="dot", line_color="red",
                          annotation_text=f"阻力 {resistance:.2f}", annotation_position="top right")

            # 成交量
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"],
                name="成交量", marker_color="rgba(100,150,255,0.3)", yaxis="y2"
            ))

            # 佈局
            fig.update_layout(
                title=f"{symbol} {interval} 走勢圖 (最後更新: {datetime.now().strftime('%H:%M:%S')})",
                height=700,
                xaxis_rangeslider_visible=False,
                yaxis=dict(title="價格"),
                yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False),
                legend=dict(x=0, y=1.1, orientation="h")
            )

            st.plotly_chart(fig, use_container_width=True)

            # === 資訊顯示 ===
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("現價", f"{df['Close'].iloc[-1]:.2f}")
            with col2:
                st.metric("支撐位", f"{support:.2f}", delta=f"{df['Close'].iloc[-1]-support:+.2f}")
            with col3:
                st.metric("阻力位", f"{resistance:.2f}", delta=f"{resistance-df['Close'].iloc[-1]:+.2f}")

            # 訊號
            if signal:
                st.success("突破訊號！")
                st.markdown(signal)
                send_telegram_alert(signal)
            else:
                st.info("尚未出現突破訊號")

            # 狀態
            st.caption(f"觀察根數: {lookback} | 緩衝: {buffer_pct*100:.2f}% | "
                       f"成交量條件: {'開啟' if use_volume_filter else '關閉'}")

        except Exception as e:
            st.error(f"載入資料失敗：{e}")

# ========== 自動更新控制（穩定版）==========
interval_map = {"30秒": 30, "60秒": 60, "3分鐘": 180, "5分鐘": 300}
refresh_seconds = interval_map[update_freq]

# 主流程
if auto_update:
    current_time = time.time()
    if current_time - st.session_state.last_update >= refresh_seconds:
        st.session_state.last_update = current_time
        st.rerun()
    else:
        # 顯示倒數
        remaining = int(refresh_seconds - (current_time - st.session_state.last_update))
        st.sidebar.caption(f"下次更新：{remaining} 秒")
else:
    st.sidebar.caption("自動更新已關閉")

# 執行主程式
load_and_update_data()
