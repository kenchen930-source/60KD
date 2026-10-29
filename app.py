import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

TZ = ZoneInfo('Asia/Taipei')
st.set_page_config(page_title='台股全市場 60 分 KD', layout='wide')
st.title('台股全市場 60 分 KD 選股器 V1.4')
st.caption('實驗版｜上市＋上櫃普通股｜KD(9,3,3)｜Yahoo 60m 行情，非保證即時')

@st.cache_data(ttl=86400, show_spinner=False)
def roster():
    sources = [
        ('上市', 'https://openapi.twse.com.tw/v1/opendata/t187ap03_L', '.TW'),
        ('上櫃', 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O', '.TWO'),
    ]
    out, errors = [], []
    for market, url, suffix in sources:
        try:
            response = requests.get(url, timeout=25, headers={'User-Agent':'Mozilla/5.0'})
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError('非預期的清單格式')
            count = 0
            for item in payload:
                code = str(next((item[k] for k in ('公司代號', 'SecuritiesCompanyCode', 'CompanyCode', 'Code') if k in item and item[k]), '')).strip()
                name = str(next((item[k] for k in ('公司名稱', '公司簡稱', 'CompanyName', 'SecuritiesCompanyName', 'Name') if k in item and item[k]), code)).strip()
                if re.fullmatch(r'\d{4}', code):
                    out.append({'代號':code,'名稱':name,'市場':market,'ticker':code+suffix})
                    count += 1
            if count < 100:
                raise ValueError(f'僅解析出 {count} 檔，需確認 API 欄位')
        except Exception as exc:
            errors.append(f'{market}清單失敗：{exc}')
    unique = {x['ticker']:x for x in out}
    return list(unique.values()), errors

def kd(frame):
    lo = frame['Low'].rolling(9, min_periods=9).min()
    hi = frame['High'].rolling(9, min_periods=9).max()
    rsv = ((frame['Close'] - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50)
    k, d, ks, ds = 50., 50., [], []
    for value in rsv:
        k = (2*k + float(value))/3
        d = (2*d + k)/3
        ks.append(k); ds.append(d)
    return pd.DataFrame({'K':ks,'D':ds}, index=frame.index)

def is_new_cross(ind):
    """Only the latest completed bar may trigger an A signal."""
    if len(ind) < 20:
        return False
    k, d = ind['K'], ind['D']
    return bool(
        pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1])
        and pd.notna(k.iloc[-2]) and pd.notna(d.iloc[-2])
        and k.iloc[-1] < 20
        and k.iloc[-2] <= d.iloc[-2]
        and k.iloc[-1] > d.iloc[-1]
    )

@st.cache_data(ttl=900, show_spinner=False)
def fetch_batch(tickers):
    # Separate cache entries for each small batch. Failure is returned rather than hidden.
    try:
        data = yf.download(list(tickers), period='30d', interval='60m', group_by='ticker',
                           auto_adjust=False, progress=False, threads=4, timeout=12)
        return data, ''
    except Exception as exc:
        return pd.DataFrame(), str(exc)

def extract(data, ticker, total):
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            df = data[ticker].copy()
        elif ticker in data.columns.get_level_values(1):
            df = data.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    elif total == 1:
        df = data.copy()
    else:
        return pd.DataFrame()
    if not {'High','Low','Close'}.issubset(df.columns):
        return pd.DataFrame()
    return df.dropna(subset=['High','Low','Close']).sort_index()

def only_completed(df, include_partial):
    if include_partial or df.empty:
        return df
    # Yahoo 60m timestamps normally denote bar START. Conservative handling:
    # During trading hours discard the most recent bar if it is from today,
    # because Yahoo 60m bar boundaries may differ from brokerage platforms.
    now = datetime.now(TZ)
    if now.weekday() < 5 and 9 <= now.hour < 14:
        last = pd.Timestamp(df.index[-1])
        if last.tzinfo is None:
            last = last.tz_localize(TZ)
        else:
            last = last.tz_convert(TZ)
        if last.date() == now.date():
            return df.iloc[:-1]
    return df

universe, roster_errors = roster()
if roster_errors:
    st.warning('；'.join(roster_errors))
if not universe:
    st.error('無法取得股票清單，暫停掃描，避免誤報全市場掃描完成。')
    st.stop()

st.write(f'目前取得 **{len(universe):,}** 檔普通股代號（依官方公司清單的四位數代號篩選）')
include_partial = st.checkbox('包含尚未收完的最新 60 分 K（訊號可能消失）', value=False)
use_slope = st.checkbox('啟用 K 值增量門檻', value=True)
min_slope = st.number_input('K 值至少上升（每根 60 分 K）', min_value=0.0, max_value=100.0, value=2.0, step=0.5, disabled=not use_slope)
st.caption('K 值增量 = 最新完成 K 值 − 前一根 K 值；關閉門檻後仍按增量由大到小排序。')
limit = st.selectbox('本次掃描範圍', ['全市場', '先測試 30 檔', '先測試 100 檔'])
st.caption('全市場首次掃描可能需數分鐘以上；免費 Yahoo 來源可能限流，結果一定要看成功率。')

if st.button('開始掃描', type='primary'):
    selected = universe if limit == '全市場' else universe[:30 if '30' in limit else 100]
    rows, failures = [], []
    bar = st.progress(0)
    status = st.empty()
    batch_size = 15
    for start in range(0, len(selected), batch_size):
        group = selected[start:start+batch_size]
        data, error = fetch_batch(tuple(x['ticker'] for x in group))
        for item in group:
            ticker = item['ticker']
            try:
                df = only_completed(extract(data, ticker, len(group)), include_partial)
                if len(df) < 25:
                    raise ValueError('60 分 K 資料不足或下載失敗')
                ind = kd(df)
                if is_new_cross(ind):
                    slope = float(ind['K'].iloc[-1] - ind['K'].iloc[-2])
                    if not use_slope or slope >= min_slope:
                        rows.append({'代號':item['代號'], '名稱':item['名稱'],
                                     '市場':item['市場'], 'K值增量':round(slope, 2),
                                     'K':round(float(ind['K'].iloc[-1]),2),
                                     'D':round(float(ind['D'].iloc[-1]),2),
                                     '收盤價':round(float(df['Close'].iloc[-1]),2),
                                     '訊號K棒時間':str(df.index[-1])})
            except Exception as exc:
                failures.append({'股票':ticker, '原因':str(exc)[:120] or error})
        done = min(start+len(group), len(selected))
        bar.progress(done/len(selected))
        status.caption(f'已處理 {done:,}/{len(selected):,} 檔｜資料成功 {done-len(failures):,}｜失敗 {len(failures):,}')
        if start + batch_size < len(selected):
            time.sleep(0.4)
    success = len(selected) - len(failures)
    st.success(f'掃描結束｜成功 {success:,}/{len(selected):,}（{success/len(selected):.1%}）｜失敗 {len(failures):,}')
    if roster_errors or failures:
        st.warning('資料未完整覆蓋全市場；未掃描成功的股票不能視為不符合條件。')
    results = pd.DataFrame(rows)
    st.subheader(f'A 新黃金交叉（符合目前斜率設定）：{len(results)} 檔')
    if not results.empty:
        results = results.sort_values('K值增量', ascending=False, kind='stable').reset_index(drop=True)
        st.dataframe(results, hide_index=True, use_container_width=True)
        st.download_button('下載符合條件 CSV', results.to_csv(index=False).encode('utf-8-sig'),
                           'kd60_A_new_cross_slope.csv', mime='text/csv')
    else:
        st.info('本次成功取得行情的股票中，沒有符合目前條件的 A 類訊號。')
    if failures:
        with st.expander(f'查看失敗股票（{len(failures)} 檔）'):
            st.dataframe(pd.DataFrame(failures), hide_index=True)

st.info('注意：Yahoo 的 60m 切棒時間、行情延遲與券商可能不同。請先拿至少 3 檔比對；此版為免費資料可行性測試，非保證全市場即時監控。')
