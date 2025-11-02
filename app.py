import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time

# ========== 基本設定 ==========
st.set_page_config(page_title="TSLA 支撐/阻力突破監控系統", layout="wide")
st.title("🚀 TSLA 支撐 / 阻力突破交易監控系統")

# ========== 側邊欄設定 ==========
symbol = st.sidebar.text_input("股票代號", value="TSLA")
interval = st.sidebar.selectbox("K線時間週期", ["5m", "10m", "15m", "30m", "1h", "1d"])
lookback = st.sidebar.slider("觀察K線根數", 50, 500, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "5分鐘"])
auto_update = st.sidebar.checkbox("🔄 自動更新", value=True)

# 新增成交量放大條件選項
use_volume_filter = st.sidebar.checkbox("📊 啟用成交量放大確認（> 均量 1.5 倍）", value=True)

# ========== 推播設定（從 st.secrets 讀取） ==========
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
except Exception:
    BOT_TOKEN = None
    CHAT_ID = None
    st.error("❌ 無法從 st.secrets 讀取 Telegram BOT_TOKEN 或 CHAT_ID，請確認 secrets.toml 設定正確。")

def send_telegram_alert(message):
    """安全的 Telegram 推播函式"""
    if not BOT_TOKEN or not CHAT_ID:
        st.warning("⚠️ 尚未設定 Telegram Token 或 Chat ID，無法推播。")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        params = {"chat_id": CHAT_ID, "text": message}
        requests.get(url, params=params, timeout=5)
        st.toast("📨 已發出 Telegram 通知", icon="📬")
    except Exception as e:
        st.error(f"⚠️ 推播失敗：{e}")

# ========== 支撐/阻力計算 ==========
def find_support_resistance(df, window=50):
    highs = df["High"].tail(window)
    lows = df["Low"].tail(window)

    resistance = highs[highs == highs.rolling(3, center=True).max()]
    support = lows[lows == lows.rolling(3, center=True).min()]

    resistance_level = np.mean(resistance.tail(3)) if len(resistance) >= 3 else highs.max()
    support_level = np.mean(support.tail(3)) if len(support) >= 3 else lows.min()

    return support_level, resistance_level

# ========== Breakout 偵測 ==========
def detect_breakout(df, support, resistance, buffer=0.2, use_volume=False, vol_mult=1.5):
    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    last_volume = df["Volume"].iloc[-1]
    avg_volume = df["Volume"].tail(lookback).mean()
    signal = None

    breakout_up = prev_close < resistance - buffer and last_close >= resistance
    breakout_down = prev_close > support + buffer and last_close <= support
    volume_condition = (not use_volume) or (last_volume > avg_volume * vol_mult)

    if breakout_up and volume_condition:
        signal = f"🚀 {symbol} 突破阻力線！現價 {last_close:.2f}，成交量 {last_volume/avg_volume:.1f} 倍均量"
    elif breakout_down and volume_condition:
        signal = f"⚠️ {symbol} 跌破支撐線！現價 {last_close:.2f}，成交量 {last_volume/avg_volume:.1f} 倍均量"
    return signal

# ========== 主程式邏輯 ==========
def load_and_update_data():
    df = yf.download(symbol, period="2d", interval=interval)
    df.dropna(inplace=True)
    support, resistance = find_support_resistance(df, lookback)
    signal = detect_breakout(df, support, resistance, use_volume=use_volume_filter)

    # --- 畫圖 ---
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Candlestick"
    )])
    fig.add_hline(y=support, line_dash="dot", line_color="green", annotation_text="Support")
    fig.add_hline(y=resistance, line_dash="dot", line_color="red", annotation_text="Resistance")

    # 加上成交量子圖
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume", marker_opacity=0.3, yaxis="y2"
    ))
    fig.update_layout(
        title=f"{symbol} {interval} K 線",
        height=700,
        yaxis=dict(title="價格"),
        yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- 顯示資訊 ---
    st.info(f"📉 支撐位: {support:.2f}  |  📈 阻力位: {resistance:.2f}")
    if use_volume_filter:
        st.write("✅ 已啟用成交量放大確認條件（1.5 倍均量）")
    else:
        st.write("📊 未啟用成交量條件（僅以價格突破判斷）")

    if signal:
        st.success(signal)
        send_telegram_alert(signal)
    else:
        st.write("⌛ 尚未出現突破訊號")

# ========== 自動更新控制 ==========
interval_map = {"30秒": 30, "60秒": 60, "5分鐘": 300}
refresh_seconds = interval_map[update_freq]

if auto_update:
    while True:
        st.empty()
        load_and_update_data()
        time.sleep(refresh_seconds)
        st.rerun()
else:
    load_and_update_data()
