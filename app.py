# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple

# (新增) 引入 Autorefresh 組件
from streamlit_autorefresh import st_autorefresh

# ==================== 初始化 ====================
st.set_page_config(page_title="多股票即時監控面板", layout="wide")
st.title("多股票支撐/阻力突破監控面板")

# session_state
for key in ["last_signal_keys", "signal_history"]:
    if key not in st.session_state:
        st.session_state[key] = ({} if key == "last_signal_keys" else [])

# ==================== 側邊欄選項 ====================
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "AAPL, TSLA, NVDA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval_options = {"1分鐘": "1m", "5分鐘": "5m", "15分鐘": "15m", "1小時": "60m", "日線": "1d"}
interval_label = st.sidebar.selectbox("K線週期", options=list(interval_options.keys()), index=1)
interval = interval_options[interval_label]

period_options = {"1天": "1d", "5天": "5d", "10天": "10d", "1個月": "1mo", "3個月": "3mo", "1年": "1y"}
period_label = st.sidebar.selectbox("資料範圍", options=list(period_options.keys()), index=1)
period = period_options[period_label]

# (!!!) 健壮性: 检查 yfinance 的数据限制
if interval == "1m" and period not in ["1d", "5d"]:
    st.sidebar.warning("警告：1分鐘 K 線最多只能回溯 7 天資料。")
if interval in ["5m", "15m", "60m"] and period not in ["1d", "5d", "10d", "1mo"]:
    st.sidebar.warning(f"警告：{interval_label} K 線最多只能回溯 60 天資料。")
# (!!!) 检查结束

lookback = st.sidebar.slider("觀察根數", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "3分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", True)
buffer_pct = st.sidebar.slider("緩衝區 (%)", 0.01, 1.0, 0.1, 0.01) / 100
sound_alert = st.sidebar.checkbox("聲音提醒", True)
show_touches = st.sidebar.checkbox("顯示價位觸碰分析", True)

st.sidebar.markdown("---")
st.sidebar.caption(f"**K線**：{interval_label} | **範圍**：{period_label}")

# ==================== 警報設定 ====================
st.sidebar.markdown("### 警報設定")
use_auto_sr_alerts = st.sidebar.checkbox("啟用自動 S/R 突破警報", True)
use_volume_filter = st.sidebar.checkbox("自動 S/R 需成交量確認 (>1.5x)", True)
st.sidebar.markdown("#### 獨立成交量警報")
use_volume_alert = st.sidebar.checkbox("啟用獨立成交量警報", True)
volume_alert_multiplier = st.sidebar.slider("成交量警報倍數", 1.5, 5.0, 2.5, 0.1)
st.sidebar.markdown("#### 自訂價位警報")
custom_alert_input = st.sidebar.text_area(
    "自訂警報價位 (每行格式: SYMBOL,價位1,價位2...)",
    "AAPL,180.5,190\nNVDA,850,900.5"
)

# 解析自訂價位
def parse_custom_alerts(text_input: str) -> Dict[str, List[float]]:
    alerts = {}
    for line in text_input.split("\n"):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            symbol = parts[0].upper()
            try:
                prices = [float(p) for p in parts[1:]]
                if symbol not in alerts:
                    alerts[symbol] = []
                alerts[symbol].extend(prices)
            except ValueError:
                continue # Skip invalid lines
    return alerts

custom_alert_levels = parse_custom_alerts(custom_alert_input)
st.sidebar.caption(f"已載入 {len(custom_alert_levels)} 檔股票的自訂價位")


# ==================== Telegram 設定與函數 (保持不變) ====================
try:
    # 假設 secrets.toml 已經設定
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except Exception:
    BOT_TOKEN = CHAT_ID = None
    telegram_ready = False
    # st.sidebar.error("Telegram 設定錯誤，請檢查 secrets.toml") # 避免過度提醒

def send_telegram_alert(msg: str) -> bool:
    if not (BOT_TOKEN and CHAT_ID):
        return False
    # ... (Telegram 發送邏輯，保持不變)
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        response = requests.get(url, params=payload, timeout=10)
        if response.status_code == 200 and response.json().get("ok"):
            return True
        else:
            # st.warning(f"Telegram API 錯誤: {response.json()}")
            return False
    except Exception as e:
        # st.warning(f"Telegram 發送失敗: {e}")
        return False

# ==================== 聲音提醒 (保持不變) ====================
def play_alert_sound():
    if sound_alert:
        st.markdown("""
        <audio autoplay style="display:none;">
            <source src="https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

# ==================== (修正) 資料獲取與快取 ====================
@st.cache_data(ttl=60) # 設置 60 秒的快取壽命 (TTL)，確保每次 Autorefresh 後數據強制過期
def fetch_data_cache(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """
    使用 st.cache_data 確保資料在 ttl 時間後強制重新下載。
    """
    try:
        # 強制重新下載數據
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=True)
        if df.empty or df.isna().all().all():
            return None
        df = df[~df.index.duplicated(keep='last')].copy()
        df = df.dropna(how='all')
        return df
    except Exception as e:
        return None

# ==================== 價位觸碰分析 (保持不變) ====================
def analyze_price_touches(df: pd.DataFrame, levels: List[float], tolerance: float = 0.005) -> List[dict]:
    # ... (函數邏輯，保持不變)
    touches = []
    high, low = df["High"], df["Low"]
    for level in levels:
        if not np.isfinite(level):
            continue
        sup_touch = int(((low <= level * (1 + tolerance)) & (low >= level * (1 - tolerance))).sum())
        res_touch = int(((high >= level * (1 - tolerance)) & (high <= level * (1 + tolerance))).sum())
        total_touch = sup_touch + res_touch
        if total_touch == 0:
            continue
        strength = "強" if total_touch >= 3 else "次"
        role = "支撐" if sup_touch > res_touch else "阻力" if res_touch > sup_touch else "支阻"
        meaning = f"每次{'止跌反彈' if role=='支撐' else '遇壓下跌'}"
        if total_touch == 2:
            meaning = "無法突破" if role == "阻力" else "小幅反彈"
        touches.append({
            "價位": f"${level:.2f}",
            "觸碰次數": f"{total_touch} 次",
            "結果": meaning,
            "意義": f"{strength}{role}"
        })
    return sorted(touches, key=lambda x: float(x["價位"][1:]), reverse=True)


# ==================== 支撐阻力 (保持不變) ====================
def find_support_resistance_fractal(df_full: pd.DataFrame, window: int = 5, min_touches: int = 2):
    # ... (函數邏輯，保持不變)
    df = df_full.iloc[:-1]
    if len(df)
