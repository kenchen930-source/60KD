import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title='台股60分KD選股',layout='wide')
st.title('台股 60 分 KD 選股器')
st.caption('實驗版：Yahoo Finance 60m 行情，可能延遲或缺漏。僅掃描輸入股票，不代表全市場。')

def kd(df):
    low=df['Low'].rolling(9,min_periods=9).min()
    high=df['High'].rolling(9,min_periods=9).max()
    rsv=((df['Close']-low)/(high-low).replace(0,np.nan)*100).fillna(50)
    k=[];d=[];pk=50.;pd_=50.
    for r in rsv:
        pk=pk*2/3+r/3
        pd_=pd_*2/3+pk/3
        k.append(pk);d.append(pd_)
    return pd.DataFrame({'K':k,'D':d},index=df.index)

def classify(ind):
    if len(ind)<3:return None
    k,d=ind['K'],ind['D']
    if k.iloc[-1]>=20 or k.iloc[-1]<=d.iloc[-1]:return None
    crosses=((k.shift(1)<=d.shift(1))&(k>d)&(k<20))
    if crosses.iloc[-1]:return 'A 新黃金交叉'
    prior=np.flatnonzero(crosses.to_numpy()[:-1])
    if len(prior)==0:return None
    i=prior[-1]
    if ((k.iloc[i+1:]<=d.iloc[i+1:]).any()):return None
    return 'B 交叉後低檔'

codes=st.text_area('股票代號（逗號、空白或換行分隔）','2330 2317 2454 2382 2303 2603')
include_partial=st.checkbox('包含最新尚未完成的60分K（訊號可能改變）',value=False)
if st.button('開始掃描',type='primary'):
    ids=list(dict.fromkeys(c.strip().upper() for c in codes.replace(',',' ').split() if c.strip()))
    if len(ids)>100:
        st.error('測試版一次最多100檔，請分批掃描。')
    else:
        results=[]; errors=[]
        bar=st.progress(0)
        now=datetime.now(ZoneInfo('Asia/Taipei'))
        for n,code in enumerate(ids):
            symbol=code if code.endswith(('.TW','.TWO')) else code+'.TW'
            try:
                df=yf.download(symbol,period='30d',interval='60m',auto_adjust=False,progress=False,threads=False)
                if df.empty and not code.endswith(('.TW','.TWO')):
                    symbol=code+'.TWO'
                    df=yf.download(symbol,period='30d',interval='60m',auto_adjust=False,progress=False,threads=False)
                if df.empty:raise ValueError('無60分K資料')
                if isinstance(df.columns,pd.MultiIndex):df.columns=df.columns.get_level_values(0)
                df=df.dropna(subset=['High','Low','Close']).sort_index()
                if len(df)<20:raise ValueError('資料不足')
                # Conservative approach: if current trading session, omit latest bar unless explicitly allowed.
                # Yahoo bar timestamp conventions vary; this avoids claiming unverified live-bar confirmation.
                if not include_partial and now.weekday()<5 and 9<=now.hour<14 and len(df)>1:
                    df=df.iloc[:-1]
                ind=kd(df)
                group=classify(ind)
                if group:
                    results.append({'股票':symbol,'分類':group,'K':round(ind['K'].iloc[-1],2),'D':round(ind['D'].iloc[-1],2),'收盤價':round(float(df['Close'].iloc[-1]),2),'最後K棒時間':str(df.index[-1]),'狀態':'可能未收棒' if include_partial else '以已取得資料計算'})
            except Exception as e:errors.append(f'{symbol}: {str(e)[:90]}')
            bar.progress((n+1)/max(len(ids),1))
        st.caption(f'掃描時間（台北）：{now:%Y-%m-%d %H:%M:%S}；已嘗試 {len(ids)} 檔；資料失敗 {len(errors)} 檔')
        for group in ('A 新黃金交叉','B 交叉後低檔'):
            subset=[r for r in results if r['分類']==group]
            st.subheader(f'{group}：{len(subset)} 檔')
            st.dataframe(pd.DataFrame(subset),use_container_width=True,hide_index=True)
        if results:
            st.download_button('下載CSV',pd.DataFrame(results).to_csv(index=False).encode('utf-8-sig'),'kd60_results.csv','text/csv')
        if errors:
            with st.expander('資料取得失敗清單'):
                st.write('\n'.join(errors))
st.info('注意：台股60分鐘K的切棒時點因資料商而異。Yahoo直接提供的60m K與券商看盤軟體可能不同，請先比對至少3檔股票。')
