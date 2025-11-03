# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
from typing import Optional, List, Tuple

# ==================== 初始化 ====================
st.set_page_config(page_title="多股票即時監控面板", layout="wide")
st.title("多股票支撐/阻力突破監控面板")

# session_state
for key in ["last_update", "last_signal_keys", "signal_history", "data_cache", "custom_levels"]:
    if key not in st.session_state:
        st.session_state[key] = (0 if key == "last_update" else 
                                {} if key in ["last_signal_keys", "data_cache", "custom_levels"] else [])

# ==================== 側邊欄選項 ====================
symbols_input = st.sidebar.text_input("股票代號（逗號分隔）", "AAPL, TSLA, NVDA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

interval_options = {"1分鐘": "1m", "5分鐘": "5m", "15分鐘": "15m", "1小時": "60m", "日線": "1d"}
interval_label = st.sidebar.selectbox("K線週期", options=list(interval_options.keys()), index=1)
interval = interval_options[interval_label]

period_options = {"1天": "1d", "5天": "5d", "10天": "10d", "1個月": "1mo", "3個月": "3mo", "1年": "1y"}
period_label = st.sidebar.selectbox("資料範圍", options=list(period_options.keys()), index=1)
period = period_options[period_label]

lookback = st.sidebar.slider("觀察根數", 20, 300, 100, 10)
update_freq = st.sidebar.selectbox("更新頻率", ["30秒", "60秒", "3分鐘"], index=1)
auto_update = st.sidebar.checkbox("自動更新", True)
use_volume_filter = st.sidebar.checkbox("成交量確認 (>1.5x)", True)
buffer_pct = st.sidebar.slider("緩衝區 (%)", 0.01, 1.0, 0.1, 0.01) / 100
sound_alert = st.sidebar.checkbox("聲音提醒", True)
show_touches = st.sidebar.checkbox("顯示價位觸碰分析", True)

st.sidebar.markdown("---")
st.sidebar.caption(f"**K線**：{interval_label} | **範圍**：{period_label}")

# ==================== 自訂價位監控 ====================
st.sidebar.markdown("---")
st.sidebar.subheader("自訂價位監控")
for symbol in symbols:
    with st.sidebar.expander(f"{symbol} 自訂價位", expanded=False):
        key_prefix = f"custom_{symbol}"
        levels_input = st.text_input(
            "價位（逗號分隔）", placeholder="e.g. 150.0, 155.5",
            key=f"{key_prefix}_input"
        )
        vol_filter = st.checkbox("成交量放大 (>1.5x)", value=True, key=f"{key_prefix}_vol")
        if levels_input.strip():
            try:
                levels = [float(x.strip()) for x in levels_input.split(",") if x.strip()]
                st.session_state.custom_levels[symbol] = {
                    "levels": levels,
                    "use_volume": vol_filter
                }
            except:
                st.error("請輸入有效數字")

# ==================== Telegram 設定 ====================
try:
    BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except Exception:
    BOT_TOKEN = CHAT_ID = None
    telegram_ready = False
    st.sidebar.error("Telegram 設定錯誤，請檢查 secrets.toml")

# ==================== Telegram 發送函數 ====================
def send_telegram_alert(msg: str) -> bool:
    if not (BOT_TOKEN and CHAT_ID):
        return False
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
            st.warning(f"Telegram API 錯誤: {response.json()}")
            return False
    except Exception as e:
        st.warning(f"Telegram 發送失敗: {e}")
        return False

# ==================== 測試按鈕 ====================
st.sidebar.markdown("### Telegram 通知測試")
if st.sidebar.button("發送測試訊息", type="secondary", use_container_width=True):
    if not telegram_ready:
        st.error("Telegram 設定錯誤，請檢查 secrets.toml")
    else:
        test_msg = (
            "<b>Telegram 通知測試成功！</b>\n"
            "這是一條來自 <i>多股票監控系統</i> 的測試訊息。\n"
            "時間: <code>" + datetime.now().strftime("%H:%M:%S") + "</code>"
        )
        with st.spinner("發送中…"):
            if send_telegram_alert(test_msg):
                st.success("Telegram 發送成功！請檢查您的 Telegram")
            else:
                st.error("Telegram 發送失敗，請檢查 Token / Chat ID")

# ==================== 聲音提醒 ====================
def play_alert_sound():
    if sound_alert:
        st.markdown("""
        <audio autoplay style="display:none;">
            <source src="https://cdn.freesound.org/previews/612/612612_5674468-lq.mp3" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

# ==================== 手動快取 ====================
def fetch_data_manual(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    cache_key = f"{symbol}_{interval}_{period}"
    if cache_key in st.session_state.data_cache:
        return st.session_state.data_cache[cache_key]
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=True)
        if df.empty or df.isna().all().all():
            return None
        df = df[~df.index.duplicated(keep='last')].copy()
        df = df.dropna(how='all')
        # 確保 Volume 是數值
        df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce').fillna(0)
        st.session_state.data_cache[cache_key] = df
        return df
    except Exception as e:
        st.warning(f"{symbol} 下載失敗: {e}")
        return None

# ==================== 價位觸碰分析 ====================
def analyze_price_touches(df: pd.DataFrame, levels: List[float Assy, tolerance: float = 0.005) -> List[dict]:
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

# ==================== 支撐阻力 ====================
def find_support_resistance_fractal(df_full: pd.DataFrame, window: int = 5, min_touches: int = 2):
    df = df_full.iloc[:-1]
    if len(df) < window * 2 + 1:
        try:
            low_min = float(df_full["Low"].min(skipna=True).item())
            high_max = float(df_full["High"].max(skipna=True).item())
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
            res_pts.append(max_high)
        if np.isclose(low.iloc[i], min_low, atol=1e-6):
            sup_pts.append(min_low)
    def cluster_points(points, tol=0.005):
        if not points: return []
        points = sorted(points)
        clusters = []
        current = [points[0]]
        for pt in points[1:]:
            if abs(pt - current[-1]) / current[-1] < tol:
                current.append(pt)
            else:
                if len(current) >= min_touches:
                    clusters.append(np.mean(current))
                current = [pt]
        if len(current) >= min_touches:
            clusters.append(np.mean(current))
        return clusters
    res_lv = cluster_points(res_pts)
    sup_lv = cluster_points(sup_pts)
    try:
        cur = float(df_full["Close"].iloc[-1].item())
    except:
        cur = 0.0
    resistance = max(res_lv, key=lambda x: (-abs(x - cur), x)) if res_lv else float(df_full["High"].max(skipna=True).item())
    support = min(sup_lv, key=lambda x: (-abs(x - cur), -x)) if sup_lv else float(df_full["Low"].min(skipna=True).item())
    all_levels = list(set(res_lv + sup_lv))
    return support, resistance, all_levels

# ==================== 突破偵測 ====================
def detect_breakout(df_full: pd.DataFrame, support: float, resistance: float,
                    buffer_pct: float, use_volume: bool, vol_mult: float, lookback: int, symbol: str):
    df = df_full.iloc[:-1]
    if len(df) < 4:
        return None, None
    try:
        last_close = float(df["Close"].iloc[-1].item())
        prev_close = float(df["Close"].iloc[-2].item())
        prev2_close = float(df["Close"].iloc[-3].item()) if len(df) >= 3 else prev_close
        last_volume = float(df["Volume"].iloc[-1].item())
    except:
        return None, None
    vol_tail = df["Volume"].iloc[-(lookback + 1):-1]
    try:
        avg_volume = float(vol_tail.mean(skipna=True).item())
    except:
        avg_volume = 1.0
    vol_ratio = last_volume / avg_volume if avg_volume > 0 else 0
    vol_ok = (not use_volume) or (vol_ratio > vol_mult)
    buffer = max(support, resistance) * buffer_pct
    if (prev2_close <= (resistance - buffer)) and (prev_close <= (resistance - buffer)) and (last_close > resistance) and vol_ok:
        msg = f"突破阻力！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n阻力: {resistance:.2f}"
        key = f"{symbol}_UP_{resistance:.2f}"
        return msg, key
    if (prev2_close >= (support + buffer)) and (prev_close >= (support + buffer)) and (last_close < support) and vol_ok:
        msg = f"跌破支撐！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n支撐: {support:.2f}"
        key = f"{symbol}_DN_{support:.2f}"
        return msg, key
    return None, None

# ==================== 自訂價位突破偵測 ====================
def detect_custom_breakout(df_full: pd.DataFrame, levels: List[float], 
                          use_volume: bool, lookback: int, symbol: str):
    if len(df_full) < 2:
        return None, None
    try:
        last_close = float(df_full["Close"].iloc[-1].item())
        prev_close = float(df_full["Close"].iloc[-2].item())
        last_volume = float(df_full["Volume"].iloc[-1].item())
    except:
        return None, None
    vol_tail = df_full["Volume"].iloc[-(lookback + 1):-1]
    avg_volume = float(vol_tail.mean()) if len(vol_tail) > 0 else 1
    vol_ratio = last_volume / avg_volume if avg_volume > 0 else 0
    vol_ok = (not use_volume) or (vol_ratio > 1.5)
    for level in levels:
        key = f"CUSTOM_{symbol}_{level:.2f}"
        if prev_close <= level and last_close > level and vol_ok:
            msg = f"自訂突破！\n股票: <b>{symbol}</b>\n目標: <b>{level:.2f}</b>\n現價: <b>{last_close:.2f}</b>"
            return msg, key
        elif prev_close >= level and last_close < level and vol_ok:
            msg = f"自訂跌破！\n股票: <b>{symbol}</b>\n目標: <b>{level:.2f}</b>\n現價: <b>{last_close:.2f}</b>"
            return msg, key
    return None, None

# ==================== 主程式 ====================
def process_symbol(symbol: str):
    df_full = fetch_data_manual(symbol, interval, period)
    if df_full is None or len(df_full) < 15:
        return None
    df = df_full.iloc[:-1]
    if len(df) < 10:
        return None

    # 動態 window
    if interval in ["1m", "5m", "15m"]:
        window = max(3, lookback // 20)
    elif interval in ["60m", "1d"]:
        window = max(5, lookback // 10)
    else:
        window = 5

    support, resistance, all_levels = find_support_resistance_fractal(df_full, window=window, min_touches=2)
    signal, signal_key = detect_breakout(df_full, support, resistance, buffer_pct,
                                         use_volume_filter, 1.5, lookback, symbol)

    # 自訂訊號
    custom_signal, custom_key = None, None
    if symbol in st.session_state.custom_levels:
        config = st.session_state.custom_levels[symbol]
        custom_signal, custom_key = detect_custom_breakout(
            df_full, config["levels"], config["use_volume"], lookback, symbol
        )

    # ==================== 改良圖表 ====================
    fig = go.Figure()

    # K線
    fig.add_trace(go.Candlestick(
        x=df_full.index,
        open=df_full["Open"], high=df_full["High"],
        low=df_full["Low"], close=df_full["Close"],
        name="K線",
        increasing_line_color='red', increasing_fillcolor='rgba(255,100,100,0.6)',
        decreasing_line_color='green', decreasing_fillcolor='rgba(100,255,100,0.6)',
        line_width=1.5, whiskerwidth=0.8, showlegend=False
    ))

    # 收盤走勢線
    fig.add_trace(go.Scatter(
        x=df_full.index, y=df_full["Close"],
        mode='lines', line=dict(color='black', width=1.2),
        name='收盤價', hovertemplate='收盤: $%{y:.2f}'
    ))

    # 支撐/阻力色帶
    fig.add_hrect(y0=support * (1 - buffer_pct), y1=support * (1 + buffer_pct),
                  fillcolor="green", opacity=0.15, line_width=0,
                  annotation_text=f"支撐區 {support:.2f}", annotation_position="top left")
    fig.add_hrect(y0=resistance * (1 - buffer_pct), y1=resistance * (1 + buffer_pct),
                  fillcolor="red", opacity=0.15, line_width=0,
                  annotation_text=f"阻力區 {resistance:.2f}", annotation_position="bottom left")

    # 其他支阻線
    for level in all_levels:
        if abs(level - support) < 1e-6 or abs(level - resistance) < 1e-6:
            continue
        fig.add_hline(y=level, line_dash="dot", line_color="gray", line_width=1)

    # 自訂價位
    if symbol in st.session_state.custom_levels:
        for level in st.session_state.custom_levels[symbol]["levels"]:
            fig.add_hline(y=level, line_dash="dash", line_color="orange", line_width=2,
                          annotation_text=f"自訂 {level:.2f}", annotation_position="top right")

    # ==================== 成交量柱 - 安全版 ====================
    vol_colors = ['lightblue'] * len(df_full)
    has_signal = signal or custom_signal

    # 安全計算平均成交量（不含最新一根）
    vol_tail = df_full["Volume"].iloc[-(lookback + 1):-1]
    avg_volume = vol_tail.mean() if len(vol_tail) > 0 and not vol_tail.empty else 0
    current_vol = df_full["Volume"].iloc[-1]

    if has_signal:
        vol_colors[-1] = 'gold'
    elif avg_volume > 0 and current_vol > avg_volume * 1.5:
        vol_colors[-1] = 'yellow'

    fig.add_trace(go.Bar(
        x=df_full.index, y=df_full["Volume"],
        name="成交量", marker_color=vol_colors,
        yaxis="y2", opacity=0.7
    ))

    # 突破標記
    if has_signal:
        last_time = df_full.index[-1]
        last_close = df_full["Close"].iloc[-1]
        fig.add_scatter(
            x=[last_time], y=[last_close],
            mode="markers+text",
            marker=dict(color="yellow", size=16, symbol="star", line=dict(width=2, color='black')),
            text=["突破" if "突破" in (signal or custom_signal or "") else "跌破"],
            textposition="top center",
            textfont=dict(size=12, color="black", family="Arial Black"),
            name="突破點"
        )
        fig.add_annotation(
            x=last_time, y=last_close,
            ax=last_time, ay=last_close + (resistance - support) * 0.3,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3,
            arrowcolor="gold"
        )

    # 佈局
    fig.update_layout(
        title=f"{symbol}｜支撐 {support:.2f} → 阻力 {resistance:.2f}",
        height=500,
        margin=dict(l=40, r=40, t=60, b=20),
        xaxis_rangeslider_visible=False,
        xaxis_title="時間",
        yaxis=dict(title="價格 (USD)", side="left", showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)'),
        yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0, y=1.0, bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1),
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    if interval in ["1m", "5m", "15m"]:
        fig.update_xaxes(tickformat="%H:%M")
    elif interval == "60m":
        fig.update_xaxes(tickformat="%m/%d %H:%M")
    else:
        fig.update_xaxes(tickformat="%Y/%m/%d")

    try:
        current_price = float(df_full["Close"].iloc[-1].item())
    except:
        current_price = 0.0

    return (fig, current_price, support, resistance, signal, signal_key,
            all_levels, df_full, custom_signal, custom_key)

# ==================== 自動更新 ====================
interval_map = {"30秒": 30, "60秒": 60, "3分鐘": 180}
refresh_seconds = interval_map[update_freq]

if auto_update:
    now = time.time()
    elapsed = now - st.session_state.last_update
    if elapsed >= refresh_seconds:
        st.session_state.last_update = now
        time.sleep(0.5)
        st.rerun()
    else:
        remaining = int(refresh_seconds - elapsed)
        st.sidebar.caption(f"下次更新：{max(0, remaining)} 秒")
else:
    if st.sidebar.button("手動更新", type="primary"):
        st.rerun()

if not symbols:
    st.warning("請輸入至少一檔股票代號")
    st.stop()

st.header(f"即時監控中：{', '.join(symbols)} | {interval_label} | {period_label}")

# ==================== 顯示所有股票 ====================
results = {}
breakout_signals = []

progress_container = st.container()
progress_bar = progress_container.progress(0)
status_text = progress_container.empty()

with st.spinner("下載資料與分析中…"):
    total_symbols = len(symbols)
    for idx, symbol in enumerate(symbols):
        progress = (idx + 1) / total_symbols
        progress_bar.progress(progress)
        status_text.text(f"正在處理：{symbol} ({idx + 1}/{total_symbols})")

        result = process_symbol(symbol)
        if result is None:
            continue
        fig, price, support, resistance, signal, key, levels, df_full, custom_signal, custom_key = result
        
        results[symbol] = {
            "fig": fig, "price": price, "support": support,
            "resistance": resistance, "signal": signal, "key": key,
            "levels": levels, "df_full": df_full,
            "custom_signal": custom_signal, "custom_key": custom_key
        }
        if signal:
            breakout_signals.append((symbol, signal, key))
        if custom_signal:
            breakout_signals.append((symbol, custom_signal, custom_key))

    progress_bar.empty()
    status_text.empty()

# ==================== 顯示結果 ====================
for symbol in symbols:
    data = results.get(symbol)
    if not data or data["fig"] is None:
        st.error(f"**{symbol}** 無資料")
        continue

    # 訊號顯示
    if data["signal"]:
        st.markdown(f"### **{symbol}**")
        st.success(data["signal"])
        if st.session_state.last_signal_keys.get(data["key"]) != data["key"]:
            st.session_state.last_signal_keys[data["key"]] = data["key"]
            st.session_state.signal_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "symbol": symbol,
                "signal": data["signal"]
            })
            if len(st.session_state.signal_history) > 20:
                st.session_state.signal_history.pop(0)
            if send_telegram_alert(data["signal"]):
                st.success("Telegram 訊號已發送")
            play_alert_sound()

    if data["custom_signal"]:
        st.markdown(f"### **{symbol}**")
        st.success(data["custom_signal"])
        if st.session_state.last_signal_keys.get(data["custom_key"]) != data["custom_key"]:
            st.session_state.last_signal_keys[data["custom_key"]] = data["custom_key"]
            st.session_state.signal_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "symbol": symbol,
                "signal": data["custom_signal"]
            })
            if len(st.session_state.signal_history) > 20:
                st.session_state.signal_history.pop(0)
            if send_telegram_alert(data["custom_signal"]):
                st.success("Telegram 自訂訊號已發送")
            play_alert_sound()

    if not (data["signal"] or data["custom_signal"]):
        st.markdown(f"### {symbol}")

    st.plotly_chart(data["fig"], use_container_width=True, config={'displayModeBar': True})

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("現價", f"{data['price']:.2f}")
    with c2: st.metric("支撐", f"{data['support']:.2f}", f"{data['price']-data['support']:+.2f}")
    with c3: st.metric("阻力", f"{data['resistance']:.2f}", f"{data['resistance']-data['price']:+.2f}")

    if show_touches and data["levels"] and data["df_full"] is not None:
        touches = analyze_price_touches(data["df_full"], data["levels"])
        if touches:
            df_touches = pd.DataFrame(touches)
            st.table(df_touches.style.set_properties(**{'text-align': 'center'}))
        else:
            st.info("無明顯觸碰價位")

    st.markdown("---")

# 歷史訊號
if st.session_state.signal_history:
    st.subheader("歷史訊號（最近20筆）")
    for s in reversed(st.session_state.signal_history[-20:]):
        st.markdown(f"**{s['time']} | {s['symbol']}** → {s['signal']}")
