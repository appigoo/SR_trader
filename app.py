# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
from typing import Optional, List

# ==================== 初始化 ====================
st.set_page_config(page_title="專業多股票突破監控", layout="wide")
st.title("專業多股票支撐 / 阻力突破監控系統")

# session_state
for key in ["last_update", "last_signal_keys", "signal_history"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key == "last_update" else {} if key == "last_signal_keys" else []

# ==================== 側邊欄選項 ====================
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "AAPL, TSLA, NVDA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval_options = {
    "1分鐘": "1m", "2分鐘": "2m", "3分鐘": "3m", "5分鐘": "5m", "10分鐘": "10m",
    "15分鐘": "15m", "30分鐘": "30m", "1小時": "60m", "日線": "1d", "週線": "1wk", "月線": "1mo"
}
interval_label = st.sidebar.selectbox("K線週期", options=list(interval_options.keys()), index=3)
interval = interval_options[interval_label]

period_options = {
    "1天": "1d", "5天": "5d", "10天": "10d",
    "1個月": "1mo", "3個月": "3mo", "6個月": "6mo",
    "1年": "1y", "2年": "2y", "5年": "5y", "10年": "10y",
    "今年至今": "ytd", "全部": "max"
}
period_label = st.sidebar.selectbox("資料範圍", options=list(period_options.keys()), index=2)
period = period_options[period_label]

lookback = st.sidebar.slider("觀察根數", 20, 500, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "3分鐘", "5分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", True)
use_volume_filter = st.sidebar.checkbox("成交量確認 (>1.5x)", True)
buffer_pct = st.sidebar.slider("緩衝區 (%)", 0.01, 2.0, 0.1, 0.01) / 100
sound_alert = st.sidebar.checkbox("聲音提醒", True)
show_touches = st.sidebar.checkbox("顯示價位觸碰分析", True)

st.sidebar.markdown("---")
st.sidebar.caption(f"**K線**：{interval_label} | **範圍**：{period_label}")

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

# ==================== 價位觸碰分析 ====================
def analyze_price_touches(df: pd.DataFrame, levels: List[float], tolerance: float = 0.005) -> List[dict]:
    touches = []
    high, low = df["High"], df["Low"]
    for level in levels:
        if not np.isfinite(level):
            continue
        sup_touch = ((low <= level * (1 + tolerance)) & (low >= level * (1 - tolerance))).sum()
        res_touch = ((high >= level * (1 - tolerance)) & (high <= level * (1 + tolerance))).sum()
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

# ==================== 支撐阻力 ====================
def find_support_resistance_fractal(df: pd.DataFrame, window: int = 5, min_touches: int = 2):
    if len(df) < window * 2 + 1:
        try:
            low_min = float(df["Low"].min(skipna=True).item())
            high_max = float(df["High"].max(skipna=True).item())
        except:
            low_min = high_max = 0.0
        return low_min, high_max, []

    high, low = df["High"], df["Low"]
    res_pts, sup_pts = [], []

    for i in range(window, len(df) - window):
        sl = slice(i - window, i + window + 1)
        segment_high = high.iloc[sl]
        segment_low = low.iloc[sl]

        if segment_high.empty or segment_low.empty:
            continue

        try:
            max_high = float(segment_high.max(skipna=True).item())
            min_low = float(segment_low.min(skipna=True).item())
        except:
            continue

        if not (np.isfinite(max_high) and np.isfinite(min_low)):
            continue

        if np.isclose(high.iloc[i], max_high, atol=1e-6):
            res_pts.append((i, max_high))
        if np.isclose(low.iloc[i], min_low, atol=1e-6):
            sup_pts.append((i, min_low))

    def cluster_points(points, tol=0.005):
        if not points: return []
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

    try:
        cur = float(df["Close"].iloc[-1].item())
    except:
        cur = 0.0

    try:
        high_max = float(df["High"].max(skipna=True).item())
        low_min = float(df["Low"].min(skipna=True).item())
    except:
        high_max = low_min = cur

    resistance = max(res_lv, key=lambda x: (-abs(x - cur), x)) if res_lv else high_max
    support = min(sup_lv, key=lambda x: (-abs(x - cur), -x)) if sup_lv else low_min

    all_levels = list(set(res_lv + sup_lv))
    return support, resistance, all_levels

# ==================== 突破偵測 ====================
def detect_breakout(df: pd.DataFrame, support: float, resistance: float,
                    buffer_pct: float, use_volume: bool, vol_mult: float, lookback: int, symbol: str):
    if len(df) < 4:
        return None, None

    try:
        last_close = float(df["Close"].iloc[-2].item())
        prev_close = float(df["Close"].iloc[-3].item())
        prev2_close = float(df["Close"].iloc[-4].item()) if len(df) >= 4 else prev_close
        last_volume = float(df["Volume"].iloc[-2].item())
    except:
        return None, None

    vol_tail = df["Volume"].iloc[-(lookback + 2):-2]
    try:
        avg_volume = float(vol_tail.mean(skipna=True).item())
    except:
        avg_volume = 1.0
    vol_ratio = last_volume / avg_volume if avg_volume > 0 else 0
    vol_ok = (not use_volume) or (vol_ratio > vol_mult)

    buffer = max(support, resistance) * buffer_pct

    if (prev2_close <= (resistance - buffer)) and \
       (prev_close <= (resistance - buffer)) and \
       (last_close > resistance) and vol_ok:
        msg = (f"突破阻力！\n"
               f"股票: <b>{symbol}</b>\n"
               f"現價: <b>{last_close:.2f}</b>\n"
               f"阻力: {resistance:.2f}\n"
               f"成交量: {last_volume/1e6:.1f}M ({vol_ratio:.1f}x)")
        key = f"{symbol}_UP_{resistance:.2f}"
        return msg, key

    if (prev2_close >= (support + buffer)) and \
       (prev_close >= (support + buffer)) and \
       (last_close < support) and vol_ok:
        msg = (f"跌破支撐！\n"
               f"股票: <b>{symbol}</b>\n"
               f"現價: <b>{last_close:.2f}</b>\n"
               f"支撐: {support:.2f}\n"
               f"成交量: {last_volume/1e6:.1f}M ({vol_ratio:.1f}x)")
        key = f"{symbol}_DN_{support:.2f}"
        return msg, key

    return None, None

# ==================== 資料快取 ====================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(_symbol: str, _interval: str, _period: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(_symbol, period=_period, interval=_interval,
                         progress=False, auto_adjust=True, threads=True)
        if df.empty or df.isna().all().all():
            return None
        df = df[~df.index.duplicated(keep='last')].copy()
        df = df.dropna(how='all')
        return df
    except Exception as e:
        st.warning(f"{_symbol} 下載失敗: {e}")
        return None

# ==================== 主程式 ====================
def process_symbol(symbol: str):
    df = fetch_data(symbol, interval, period)
    if df is None or len(df) < 15:
        return None, None, None, None, None, None, [], None

    df_display = df.copy()
    df = df.iloc[:-1]
    if len(df) < 10:
        return None, None, None, None, None, None, [], None

    window = max(5, lookback // 15)
    support, resistance, all_levels = find_support_resistance_fractal(df, window=window, min_touches=2)
    signal, signal_key = detect_breakout(df, support, resistance, buffer_pct,
                                         use_volume_filter, 1.5, lookback, symbol)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_display.index, open=df_display["Open"], high=df_display["High"],
        low=df_display["Low"], close=df_display["Close"], name="K線"
    ))
    fig.add_hline(y=support, line_dash="dot", line_color="green",
                  annotation_text=f"支撐 {support:.2f}")
    fig.add_hline(y=resistance, line_dash="dot", line_color="red",
                  annotation_text=f"阻力 {resistance:.2f}")

    for level in all_levels:
        color = "green" if abs(level - support) < 1e-6 else "red" if abs(level - resistance) < 1e-6 else "gray"
        fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1)

    fig.add_trace(go.Bar(x=df_display.index, y=df_display["Volume"],
                         name="成交量", marker_color="lightblue", yaxis="y2"))

    if signal:
        last_time = df_display.index[-2]
        last_close = df_display["Close"].iloc[-2]
        fig.add_scatter(x=[last_time], y=[last_close], mode="markers",
                        marker=dict(color="yellow", size=12, symbol="star"),
                        name="突破點")

    fig.update_layout(
        title=f"{symbol} {interval_label} ({period_label})",
        height=600, xaxis_rangeslider_visible=False,
        yaxis=dict(title="價格"), yaxis2=dict(title="成交量", overlaying="y", side="right")
    )

    try:
        current_price = float(df_display["Close"].iloc[-1].item())
    except:
        current_price = 0.0

    return fig, current_price, support, resistance, signal, signal_key, all_levels, df_display

# ==================== 執行 ====================
interval_map = {"30秒": 30, "60秒": 60, "3分鐘": 180, "5分鐘": 300}
refresh_seconds = interval_map[update_freq]

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

if not symbols:
    st.warning("請輸入至少一檔股票代號")
    st.stop()

st.header(f"監控中：{', '.join(symbols)} | {interval_label} | {period_label}")

breakout_signals = []
results = {}

with st.spinner("下載資料與分析中…"):
    for symbol in symbols:
        fig, price, support, resistance, signal, key, levels, df_display = process_symbol(symbol)
        results[symbol] = {
            "fig": fig, "price": price, "support": support,
            "resistance": resistance, "signal": signal, "key": key,
            "levels": levels, "df_display": df_display
        }
        if signal:
            breakout_signals.append((symbol, signal, key))

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

        # === 交易建議表 ===
        if show_touches and data["levels"] and data["df_display"] is not None:
            st.subheader(f"**{symbol} 價位觸碰分析**")
            touches = analyze_price_touches(data["df_display"], data["levels"])
            if touches:
                df_touches = pd.DataFrame(touches)
                st.table(df_touches.style.set_properties(**{'text-align': 'center'}))
            else:
                st.info("無明顯觸碰價位")

# 歷史訊號
if st.session_state.signal_history:
    st.subheader("歷史訊號（最近20筆）")
    for s in reversed(st.session_state.signal_history):
        st.markdown(f"**{s['time']} | {s['symbol']}** → {s['signal']}")
