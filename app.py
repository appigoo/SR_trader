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

# ==================== 初始化 ====================
st.set_page_config(page_title="多股票即時監控面板", layout="wide")
st.title("多股票支撐/阻力突破監控面板")

# session_state
for key in ["last_update", "last_signal_keys", "signal_history", "data_cache"]:
    if key not in st.session_state:
        st.session_state[key] = (0 if key == "last_update" else
                                 {} if key in ["last_signal_keys", "data_cache"] else [])

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

# ==================== (新) 警報設定 ====================
st.sidebar.markdown("### 警報設定")

# 1. 自動 S/R 警報
st.sidebar.markdown("#### 1. 自動 S/R 突破警報")
use_auto_sr_alerts = st.sidebar.checkbox("啟用自動 S/R 突破警報", True)
use_volume_filter = st.sidebar.checkbox("自動 S/R 需成交量確認 (>1.5x)", True)

# 2. 獨立成交量警報
st.sidebar.markdown("#### 2. 獨立成交量警報")
use_volume_alert = st.sidebar.checkbox("啟用獨立成交量警報", True)
volume_alert_multiplier = st.sidebar.slider("成交量警報倍數", 1.5, 5.0, 2.5, 0.1)

# 3. 自訂價位警報
st.sidebar.markdown("#### 3. 自訂價位警報")
custom_alert_input = st.sidebar.text_area(
    "自訂警報價位 (每行格式: SYMBOL,價位1,價位2...)",
    "AAPL,180.5,190\nNVDA,850,900.5"
)

# (新) 解析自訂價位
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

# ==================== 測試按鈕（純文字提示） ====================
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
                st.error("Telegram 發送失敗，请检查 Token / Chat ID")

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
        st.session_state.data_cache[cache_key] = df
        return df
    except Exception as e:
        st.warning(f"{symbol} 下載失敗: {e}")
        return None

# ==================== 價位觸碰分析 ====================
def analyze_price_touches(df: pd.DataFrame, levels: List[float], tolerance: float = 0.005) -> List[dict]:
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
            low_min = float(df_full["Low"].min(skipna=True))
            high_max = float(df_full["High"].max(skipna=True))
        except (ValueError, TypeError):
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
            max_high = float(segment_high.max(skipna=True))
            min_low = float(segment_low.min(skipna=True))
        except (ValueError, TypeError):
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
        cur = float(df_full["Close"].iloc[-1])
        df_high_max = float(df_full["High"].max(skipna=True))
        df_low_min = float(df_full["Low"].min(skipna=True))
    except (IndexError, ValueError, TypeError):
        cur = 0.0
        df_high_max = 0.0
        df_low_min = 0.0

    resistance = max(res_lv, key=lambda x: (-abs(x - cur), x)) if res_lv else df_high_max
    support = min(sup_lv, key=lambda x: (-abs(x - cur), -x)) if sup_lv else df_low_min
    
    all_levels = list(set(res_lv + sup_lv))
    return support, resistance, all_levels

# ==================== (修改) 警報 1: 自動突破偵測 ====================
def check_auto_breakout(df_full: pd.DataFrame, support: float, resistance: float,
                        buffer_pct: float, use_volume: bool, vol_mult: float, lookback: int, symbol: str) -> Optional[Tuple[str, str, str]]:
    df = df_full.iloc[:-1] # 使用已完成的 K 棒
    if len(df) < 4:
        return None
        
    try:
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        prev2_close = float(df["Close"].iloc[-3])
        last_volume = float(df["Volume"].iloc[-1])
    except (IndexError, ValueError, TypeError):
        return None

    vol_tail = df["Volume"].iloc[-(lookback + 1):-1]
    
    try:
        avg_volume = float(vol_tail.mean(skipna=True))
    except (ValueError, TypeError):
        avg_volume = 1.0

    vol_ratio = last_volume / avg_volume if avg_volume > 0 else 0
    vol_ok = (not use_volume) or (vol_ratio > vol_mult)
    
    buffer = max(support, resistance) * buffer_pct
    
    if (prev2_close <= (resistance - buffer)) and (prev_close <= (resistance - buffer)) and (last_close > resistance) and vol_ok:
        msg = f"突破阻力！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n阻力: {resistance:.2f}"
        key = f"{symbol}_AUTO_UP_{resistance:.2f}"
        return (symbol, msg, key)
        
    if (prev2_close >= (support + buffer)) and (prev_close >= (support + buffer)) and (last_close < support) and vol_ok:
        msg = f"跌破支撐！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n支撐: {support:.2f}"
        key = f"{symbol}_AUTO_DN_{support:.2f}"
        return (symbol, msg, key)
        
    return None

# ==================== (新) 警報 2: 自訂價位偵測 ====================
def check_custom_price_alerts(symbol: str, df_full: pd.DataFrame, 
                              custom_levels: List[float]) -> List[Tuple[str, str, str]]:
    if not custom_levels or len(df_full) < 2:
        return []
    
    try:
        # 比較最新（可能未完成）的 K 棒
        last_close = float(df_full["Close"].iloc[-1])
        prev_close = float(df_full["Close"].iloc[-2])
    except (IndexError, ValueError):
        return []

    signals = []
    for level in custom_levels:
        # 檢查向上穿越
        if (prev_close <= level) and (last_close > level):
            msg = f"觸及自訂價位 (向上)！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n自訂價位: {level:.2f}"
            key = f"{symbol}_CUSTOM_UP_{level:.2f}"
            signals.append((symbol, msg, key))
        # 檢查向下穿越
        elif (prev_close >= level) and (last_close < level):
            msg = f"觸及自訂價位 (向下)！\n股票: <b>{symbol}</b>\n現價: <b>{last_close:.2f}</b>\n自訂價位: {level:.2f}"
            key = f"{symbol}_CUSTOM_DN_{level:.2f}"
            signals.append((symbol, msg, key))
    return signals

# ==================== (新) 警報 3: 獨立成交量偵測 ====================
def check_volume_alert(symbol: str, df_full: pd.DataFrame, 
                       vol_multiplier: float, lookback: int) -> Optional[Tuple[str, str, str]]:
    df = df_full.iloc[:-1] # 使用已完成的 K 棒
    if len(df) < lookback:
        return None
    
    try:
        last_volume = float(df["Volume"].iloc[-1])
    except (IndexError, ValueError):
        return None

    vol_tail = df["Volume"].iloc[-(lookback + 1):-1]
    if vol_tail.empty:
        return None
        
    try:
        avg_volume = float(vol_tail.mean(skipna=True))
    except (ValueError, TypeError):
        avg_volume = 1.0

    if avg_volume == 0:
        return None
        
    vol_ratio = last_volume / avg_volume
    
    if vol_ratio > vol_multiplier:
        msg = f"成交量激增！\n股票: <b>{symbol}</b>\n現量: {last_volume:,.0f}\n均量: {avg_volume:,.0f} (<b>{vol_ratio:.1f}x</b>)"
        key = f"{symbol}_VOL_{vol_ratio:.1f}x_{pd.Timestamp.now().floor('T')}" # 加上時間戳，確保獨一無二
        return (symbol, msg, key)
    return None

# ==================== 主程式 (重構) ====================
def process_symbol(symbol: str, custom_levels: List[float]):
    df_full = fetch_data_manual(symbol, interval, period)
    
    if df_full is None or len(df_full) < 15:
        return None, None, None, None, [], None
        
    df = df_full.iloc[:-1]
    if len(df) < 10:
        return None, None, None, None, [], None
        
    window = max(5, lookback // 15)
    support, resistance, all_levels = find_support_resistance_fractal(df_full, window=window, min_touches=2)
                                         
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_full.index, open=df_full["Open"], high=df_full["High"],
                                 low=df_full["Low"], close=df_full["Close"], name="K線"))
                                 
    # (新) 需求 1: 添加 SMA 均線
    sma_period = 20
    if len(df_full) > sma_period:
        df_full[f'SMA_{sma_period}'] = df_full['Close'].rolling(window=sma_period).mean()
        fig.add_trace(go.Scatter(x=df_full.index, y=df_full[f'SMA_{sma_period}'], 
                                 name=f'SMA {sma_period}', line=dict(color='orange', width=1), 
                                 opacity=0.7))

    # (新) 需求 1: 添加 S/R 區間背景
    fig.add_hrect(y0=support, y1=resistance, 
                  fillcolor="rgba(100, 100, 100, 0.1)", 
                  layer="below", line_width=0,
                  annotation_text="S/R Range", annotation_position="right")

    # (新) 需求 1: 強化主要 S/R 線條
    fig.add_hline(y=support, line_dash="dash", line_color="green", line_width=2, annotation_text=f"支撐 {support:.2f}")
    fig.add_hline(y=resistance, line_dash="dash", line_color="red", line_width=2, annotation_text=f"阻力 {resistance:.2f}")
    
    # (新) 需求 1: 弱化其他價位
    for level in all_levels:
        if not (np.isclose(level, support) or np.isclose(level, resistance)):
            fig.add_hline(y=level, line_dash="dot", line_color="grey", line_width=1, opacity=0.5)

    # (新) 需求 2: 添加自訂價位線條到圖表
    for level in custom_levels:
        fig.add_hline(y=level, line_dash="longdash", line_color="blue", line_width=1.5, 
                      annotation_text=f"自訂 {level:.2f}", annotation_position="right")

    fig.add_trace(go.Bar(x=df_full.index, y=df_full["Volume"], name="成交量", marker_color="lightblue", yaxis="y2"))
                        
    fig.update_layout(title=f"{symbol}", height=400, margin=dict(l=20, r=20, t=40, b=20),
                      xaxis_rangeslider_visible=False, yaxis=dict(title="價格"), yaxis2=dict(title="成交量", overlaying="y", side="right"))
                      
    try:
        current_price = float(df_full["Close"].iloc[-1])
    except (IndexError, ValueError, TypeError):
        current_price = 0.0
        
    # (修改) 不再返回 signal/key
    return fig, current_price, support, resistance, all_levels, df_full

# ==================== 自動更新 ====================
interval_map = {"30秒": 30, "60秒": 60, "3分鐘": 180}
refresh_seconds = interval_map[update_freq]

if auto_update:
    now = time.time()
    if now - st.session_state.last_update >= refresh_seconds:
        st.session_state.data_cache = {}
        st.session_state.last_update = now
        time.sleep(1.5)
        st.rerun()
    else:
        remaining = int(refresh_seconds - (now - st.session_state.last_update))
        st.sidebar.caption(f"下次更新：{max(0, remaining)} 秒")
else:
    if st.sidebar.button("手動更新", type="primary"):
        st.session_state.data_cache = {}
        st.rerun()

if not symbols:
    st.warning("請輸入至少一檔股票代號")
    st.stop()

st.header(f"即時監控中：{', '.join(symbols)} | {interval_label} | {period_label}")

# ==================== 顯示所有股票 (重構警報邏輯) ====================
results = {}
all_generated_signals = [] # (新) 儲存所有警報

# 進度條容器
progress_container = st.container()
progress_bar = progress_container.progress(0)
status_text = progress_container.empty()

with st.spinner("下載資料与分析中…"):
    total_symbols = len(symbols)
    for idx, symbol in enumerate(symbols):
        progress = (idx + 1) / total_symbols
        progress_bar.progress(progress)
        status_text.text(f"正在處理：{symbol} ({idx + 1}/{total_symbols})")

        # (新) 獲取該股票的自訂價位
        symbol_custom_levels = custom_alert_levels.get(symbol, [])

        # (修改) process_symbol 不再返回 signal/key
        fig, price, support, resistance, levels, df_full = process_symbol(symbol, symbol_custom_levels)
        
        results[symbol] = {
            "fig": fig, "price": price, "support": support,
            "resistance": resistance, "levels": levels, "df_full": df_full
        }
        
        # --- (新) 警報生成區 ---
        if df_full is not None and len(df_full) > 5:
            
            # 警報 1: 自動 S/R 突破
            if use_auto_sr_alerts:
                auto_signal = check_auto_breakout(df_full, support, resistance, buffer_pct,
                                                    use_volume_filter, 1.5, lookback, symbol)
                if auto_signal:
                    all_generated_signals.append(auto_signal)

            # 警報 2: 自訂價位
            custom_signals = check_custom_price_alerts(symbol, df_full, symbol_custom_levels)
            all_generated_signals.extend(custom_signals)
            
            # 警報 3: 獨立成交量
            if use_volume_alert:
                vol_signal = check_volume_alert(symbol, df_full, volume_alert_multiplier, lookback)
                if vol_signal:
                    all_generated_signals.append(vol_signal)

    # 完成後清除進度條
    progress_bar.empty()
    status_text.empty()

# ==================== 顯示結果 (重構) ====================
for symbol in symbols:
    data = results.get(symbol)
    if not data or data["fig"] is None:
        st.error(f"**{symbol}** 無資料")
        continue

    # (新) 找出這檔股票的所有警報
    symbol_signals = [s for s in all_generated_signals if s[0] == symbol]

    if symbol_signals:
        st.markdown(f"### **{symbol}**")
        
        # (新) 循環顯示所有警報
        for (sym, signal_msg, signal_key) in symbol_signals:
            st.success(signal_msg) # 顯示警報訊息
            
            if signal_key: 
                # 檢查是否為新訊號
                if st.session_state.last_signal_keys.get(signal_key) != signal_key:
                    st.session_state.last_signal_keys[signal_key] = signal_key
                    st.session_state.signal_history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "symbol": symbol,
                        "signal": signal_msg
                    })
                    if len(st.session_state.signal_history) > 20:
                        st.session_state.signal_history.pop(0)
                    
                    if send_telegram_alert(signal_msg):
                        st.success("Telegram 訊號已發送")
                    play_alert_sound()
            
    else:
        st.markdown(f"### {symbol}")

    # 顯示圖表
    st.plotly_chart(data["fig"], use_container_width=True)

    # 顯示指標
    c1, c2, c3 = st.columns(3)
    if data["price"] is not None and data["support"] is not None and data["resistance"] is not None:
        try:
            with c1: st.metric("現價", f"{data['price']:.2f}")
            with c2: st.metric("支撐", f"{data['support']:.2f}", f"{data['price']-data['support']:+.2f}")
            with c3: st.metric("阻力", f"{data['resistance']:.2f}", f"{data['resistance']-data['price']:+.2f}")
        except (ValueError, TypeError):
             with c1: st.metric("現價", "N/A")
             with c2: st.metric("支撐", "N/A")
             with c3: st.metric("阻力", "N/A")
    else:
        with c1: st.metric("現價", "N/A")

    # 顯示價位觸碰分析
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
        signal_text = s['signal'].replace('\n', ' | ')
        st.markdown(f"**{s['time']} | {s['symbol']}** → {signal_text}")
