import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.regression import prepare, fit_model, inference, diagnostics, METHODS

@st.cache_resource(show_spinner=False, max_entries=6, ttl=1800)
def cached_context(df,target,chronological):
    return prepare(df,target,chronological)

@st.cache_data(show_spinner=False,max_entries=12,ttl=1800)
def sweep(df,target,chronological,method):
    ctx = cached_context(df,target,chronological)
    p = len(ctx['names'])
    ks = list(range(1,p+1)) if p <= 8 else sorted(set([1,3,5,10,15,20,25,p]) & set(range(1,p+1)))
    results = [fit_model(ctx,method,k) for k in ks]
    return pd.DataFrame([dict(K=k,**r['metrics']) for k,r in zip(ks,results)]), results

def render_explorer(df,target,chronological=False,key='explorer',unit=''):
    st.dataframe(df.head(10),hide_index=True)
    st.caption(f'{len(df):,} 列 · {len(df.columns)} 欄 · 目標：{target} {unit}')
    try:
        ctx = prepare(df,target,chronological) if key.startswith('custom') else cached_context(df,target,chronological)
        p = len(ctx['names'])
        methods = METHODS if len(ctx['train'])<=1000 and p<=8 else ['Lasso','RFE','ElasticNet','F-test']
        with st.form(key+'_fit'):
            method = st.selectbox('特徵選擇演算法',methods,key=key+'_method')
            st.caption('Lasso／ElasticNet 使用固定正則化強度排名後取 Top K，再擬合 OLS。K 指編碼後的特徵數，State 可拆成數個 dummy。')
            run = st.form_submit_button('計算／更新 K 分析',type='primary')
        state_key = key+'_result'
        if run:
            with st.spinner('僅以訓練資料選特徵，計算驗證集表現…'):
                if key.startswith('custom'):
                    ks = list(range(1,p+1)) if p<=8 else sorted(set([1,3,5,10,15,20,25,p]) & set(range(1,p+1)))
                    fits = [fit_model(ctx,method,k) for k in ks]
                    table = pd.DataFrame([dict(K=k,**r['metrics']) for k,r in zip(ks,fits)])
                else:
                    table, fits = sweep(df,target,chronological,method)
                st.session_state[state_key] = (table,fits,method)
        if state_key not in st.session_state:
            st.info('按「計算／更新 K 分析」開始。輸入修改後須重新計算，避免每次互動重訓。')
            return
        table,fits,used_method = st.session_state[state_key]
        best = int(table.loc[table.RMSE.idxmin(),'K'])
        k = st.select_slider('特徵數 K',options=table.K.tolist(),value=best,key=key+'_k')
        result = fits[table.K.tolist().index(k)]
        st.caption(f'已計算方法：{used_method}｜驗證 RMSE 最小的 K={best}。測試集未參與選擇。')
        columns = st.columns(4)
        for col,(label,value) in zip(columns,result['metrics'].items()):
            col.metric('Validation '+label,'N/A' if not np.isfinite(value) else f'{value:,.4f}')
        st.caption('Validation Adjusted R² 為依驗證樣本數與選中特徵數計算的描述性調整指標；不作為訓練集整體 F 檢定。')
        st.write('**選中特徵：** '+', '.join(result['names']))
        a,b = st.columns(2)
        a.plotly_chart(px.line(table,x='K',y='RMSE',markers=True,title='Validation RMSE vs. K'))
        b.plotly_chart(px.line(table,x='K',y='R²',markers=True,title='Validation R² vs. K'))
        with st.expander('檢視實際 K 分析數據'):
            st.dataframe(table,hide_index=True)
        cols = result['cols']
        model = result['model']
        st.subheader('OLS 方程式與統計推論')
        st.code('ŷ = '+f'{model.intercept_:.6g}'+' '.join(f'{b:+.6g} × [{name}]' for name,b in zip(result['names'],model.coef_)))
        st.caption('方程式使用前處理後的欄位：數值經訓練中位數補值與標準化，類別以第一類為基準做 One-hot encoding。')
        try:
            coef,f,pvalue = inference(ctx['train'][:,cols],ctx['y'][ctx['ti']],result['names'],model)
            st.write(f'訓練集整體 F-statistic = {f:.4g}；p-value = {pvalue:.4g}')
            st.dataframe(coef,hide_index=True)
        except ValueError as exc:
            st.warning(str(exc))
            st.dataframe(pd.DataFrame({'特徵':result['names'],'係數':model.coef_}))
        st.warning('經典 OLS 推論假設線性、獨立且同質變異誤差；特徵選擇後的 p-value／信賴區間未校正選擇偏差，不代表因果。時間序列尤其可能違反獨立性。係數信賴區間不是預測區間。')
        st.subheader('即時預測')
        train_raw = ctx['X'].iloc[ctx['ti']]
        defaults = {}
        with st.form(key+'_predict'):
            widgets = st.columns(3)
            for i,name in enumerate(ctx['X'].columns):
                series = train_raw[name]
                if pd.api.types.is_numeric_dtype(series):
                    default = float(series.median()) if series.notna().any() else 0.
                    defaults[name] = widgets[i%3].number_input(str(name),value=default,key=key+'_input_'+str(name))
                else:
                    options = series.dropna().unique().tolist()
                    mode = series.mode()
                    defaults[name] = widgets[i%3].selectbox(str(name),options,index=options.index(mode.iloc[0]) if len(mode) else 0,key=key+'_input_'+str(name))
            predict = st.form_submit_button('計算預測')
        if predict:
            row = pd.DataFrame([defaults],columns=ctx['X'].columns)
            pred = model.predict(ctx['pre'].transform(row)[:,cols])[0]
            st.metric('預測值',f'{pred:,.2f} {unit}')
        st.caption('預設值使用訓練資料中位數／眾數。模型與前處理已快取；修改輸入不重訓。新探索模型未經 Conformal 校準，不沿用原模型區間。')
        st.subheader('Validation 誤差診斷')
        d = diagnostics(ctx['y'][ctx['vi']],result['predictions'])
        d.insert(0,'資料列位置',ctx['vi'])
        a,b = st.columns(2)
        a.plotly_chart(px.scatter(d,x='實際值',y='預測值',title='Actual vs. Predicted'))
        b.plotly_chart(px.scatter(d,x='預測值',y='殘差',title='Residual vs. Predicted'))
        st.plotly_chart(px.histogram(d,x='殘差',nbins=40,title='殘差分布'))
        st.dataframe(d.sort_values('絕對誤差',ascending=False).head(25),hide_index=True)
        st.caption('實際值為 0 時，百分比誤差顯示為空值。')
        with st.expander('獨立 Test 評估（不參與 K 選擇）'):
            st.caption('僅對驗證 RMSE 最佳 K 評估；模型保持訓練集擬合狀態，未以測試資料調整。')
            if st.button('評估驗證選出的模型',key=key+'_test'):
                from utils.regression import metrics
                winner = fits[table.K.tolist().index(best)]
                test_pred = winner['model'].predict(ctx['pre'].transform(ctx['X'].iloc[ctx['test']])[:,winner['cols']])
                st.json(metrics(ctx['y'][ctx['test']],test_pred,best))
    except (ValueError,TypeError,KeyError) as exc:
        st.error(f'無法完成分析：{exc}')
