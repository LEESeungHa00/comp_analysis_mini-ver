import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import statsmodels.api as sm

# --------------------------------#
# 데이터 전처리 및 분석 함수 #
# --------------------------------#

def remove_outliers_iqr(df, column_name):
    """IQR 방식을 사용하여 이상치를 제거하는 함수"""
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    initial_rows = len(df)
    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    removed_rows = initial_rows - len(df_filtered)
    if removed_rows > 0:
        st.warning(f"분석의 정확도를 위해 시장 데이터의 단가(Unit Price) 이상치 {removed_rows}건을 제거했습니다.")
    return df_filtered

def reset_market_analysis_states():
    """분석 상태를 초기화하는 함수"""
    st.session_state.market_analysis_done = False
    keys_to_reset = ['market_df', 'analyzed_product_name', 'selected_customer', 
                     'market_contract_date', 'top_competitors_list',
                     'all_competitors_ranked']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# --------------------------#
# 메인 애플리케이션 UI 및 로직 #
# --------------------------#

st.set_page_config(layout="wide")

# --- 세션 상태 초기화 ---
if 'market_analysis_done' not in st.session_state:
    st.session_state.market_analysis_done = False

# ==============================================================================
# 페이지: 시장 경쟁력 분석
# ==============================================================================
st.title('🏆 시장 경쟁력 상세 분석 (Demo Version)')

if st.session_state.get('market_analysis_done', False):
    st.button("새로운 시장 분석 시작 (다시하기)", on_click=reset_market_analysis_states)

if not st.session_state.get('market_analysis_done', False):
    st.write("특정 품목에 대한 전체 시장 데이터를 업로드하여, 고객사의 시장 내 경쟁력을 심층 분석합니다.")
    market_file = st.file_uploader(f"분석할 품목의 전체 시장 데이터를 업로드하세요.", type=['csv', 'xlsx'], key="market_uploader")
    st.caption("※ 하나의 품목에 대한 여러 회사의 정보가 포함된 TDS raw file을 업로드해주세요.")
    
    if market_file:
        with st.form("market_analysis_form"):
            try:
                market_df_for_importers = pd.read_csv(market_file) if market_file.name.endswith('.csv') else pd.read_excel(market_file)
                
                if 'Raw Importer Name' in market_df_for_importers.columns:
                    importer_list = sorted(market_df_for_importers['Raw Importer Name'].unique())
                    customer_name_selection = st.selectbox("분석할 기준 업체를 선택해주세요.", options=importer_list)
                else:
                    st.warning("업로드된 파일에 'Raw Importer Name' 컬럼이 없습니다. 아래에 직접 입력해주세요.")
                    customer_name_selection = st.text_input("분석할 수입 업체 이름을 입력해주세요.")
            
            except Exception as e:
                st.error("파일을 읽는 중 오류가 발생했습니다. 컬럼명을 확인해주세요.")
                customer_name_selection = None
            
            analyzed_product_name_input = st.text_input("분석할 품목명을 입력하세요 (예: 건면)")
            contract_date_input = st.date_input("분석 기준이 될 계약 시작일을 선택하세요.")
            market_submitted = st.form_submit_button("시장 경쟁력 분석 시작")

        if 'market_submitted' in locals() and market_submitted and customer_name_selection and analyzed_product_name_input:
            with st.spinner('시장 데이터를 분석 중입니다. 파일 크기에 따라 시간이 걸릴 수 있습니다...'):
                market_df = market_df_for_importers.copy()
                
                rename_dict = {'Date': 'date', 'Reported Product Name': 'product_name', 'Volume': 'volume', 'Unit Price': 'unit_price', 'Origin Country': 'origin_country'}
                if 'Raw Importer Name' in market_df.columns:
                    rename_dict['Raw Importer Name'] = 'importer_name'
                
                market_df.rename(columns=rename_dict, inplace=True)
                
                if 'importer_name' not in market_df.columns:
                    market_df['importer_name'] = customer_name_selection

                market_df['date'] = pd.to_datetime(market_df['date'])
                market_df['year_month'] = market_df['date'].dt.to_period('M')
                market_df['year'] = market_df['date'].dt.year
                market_df['quarter'] = market_df['date'].dt.quarter
                
                required_market_cols = ['importer_name', 'product_name', 'volume', 'unit_price']
                if 'Exporter' in market_df.columns: required_market_cols.append('Exporter')
                if 'origin_country' in market_df.columns: required_market_cols.append('origin_country')
                market_df = market_df.dropna(subset=required_market_cols)
                market_df = remove_outliers_iqr(market_df, 'unit_price')
                
                # --- 마스킹 로직 ---
                all_importers = sorted(market_df['importer_name'].unique())
                competitors = [name for name in all_importers if name != customer_name_selection]
                masking_map = {name: f'{chr(65+i)}사' for i, name in enumerate(competitors)}
                masking_map[customer_name_selection] = customer_name_selection
                market_df['masked_name'] = market_df['importer_name'].map(masking_map)

                lowess_results = sm.nonparametric.lowess(market_df['unit_price'], market_df['volume'], frac=0.5)
                market_df['expected_price'] = np.interp(market_df['volume'], lowess_results[:, 0], lowess_results[:, 1])
                market_df['competitiveness_index'] = market_df['expected_price'] - market_df['unit_price']
                
                all_competitors_ranked = market_df.groupby('masked_name')['competitiveness_index'].mean().sort_values(ascending=False).reset_index()
                
                customer_rank_info = all_competitors_ranked[all_competitors_ranked['masked_name'] == customer_name_selection]
                customer_rank = customer_rank_info.index[0] if not customer_rank_info.empty else len(all_competitors_ranked)
                top_competitors_list = all_competitors_ranked.iloc[:customer_rank]['masked_name'].tolist()
                if customer_name_selection in top_competitors_list:
                    top_competitors_list.remove(customer_name_selection)
                
                st.session_state.market_df = market_df
                st.session_state.analyzed_product_name = analyzed_product_name_input
                st.session_state.selected_customer = customer_name_selection
                st.session_state.market_contract_date = pd.to_datetime(contract_date_input)
                st.session_state.top_competitors_list = top_competitors_list
                st.session_state.all_competitors_ranked = all_competitors_ranked
                st.session_state.market_analysis_done = True
            st.rerun()

if st.session_state.get('market_analysis_done', False):
    customer_name = st.session_state.selected_customer
    market_df = st.session_state.market_df
    analyzed_product_name = st.session_state.analyzed_product_name
    contract_date = st.session_state.market_contract_date
    top_competitors_list = st.session_state.top_competitors_list
    all_competitors_ranked = st.session_state.all_competitors_ranked
    
    st.subheader(f"'{analyzed_product_name}' 품목 시장 분석 결과 (기준 업체: {customer_name})")

    with st.expander(f"1. [{analyzed_product_name}] 구매 경쟁력 분석", expanded=True):
        st.markdown("##### Volume 대비 Unit Price 분포 및 시장 추세")
        fig_comp = px.scatter(market_df, x='volume', y='unit_price', trendline="lowess", trendline_color_override="red", hover_data=['masked_name', 'date'], 
                              title="<b>시장 내 거래 분포 및 평균 가격 추세선</b><br><span style='font-size: 0.8em; color:grey;'>LOWESS 회귀분석 기반</span>",
                              labels={'volume': '수입량(KG)', 'unit_price': '단가(USD/KG)'})
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("##### 구매 경쟁력 상위 10개사")
        top_10_competitors = all_competitors_ranked.head(10)
        
        def highlight_customer(row):
            color = 'background-color: lightblue' if row.masked_name == customer_name else ''
            return [color] * len(row)
        
        st.dataframe(top_10_competitors.style.apply(highlight_customer, axis=1).format({'competitiveness_index': '{:,.2f}'}))
        
        customer_rank_info = all_competitors_ranked[all_competitors_ranked['masked_name'] == customer_name]
        if not customer_rank_info.empty:
            customer_rank = customer_rank_info.index[0] + 1
            if customer_rank > 10:
                st.info(f"참고: **{customer_name}**의 구매 경쟁력 순위는 전체 {len(all_competitors_ranked)}개사 중 **{customer_rank}위**입니다.")

    with st.expander(f"2. [{analyzed_product_name}] 단가 추세 및 경쟁 우위 그룹 벤치마킹", expanded=True):
        st.markdown("##### 월별 평균 단가 추세")
        market_avg_price = market_df.groupby('year_month')['unit_price'].mean().rename('market_avg_price')
        customer_market_df = market_df[market_df['masked_name'] == customer_name]
        customer_avg_price = customer_market_df.groupby('year_month')['unit_price'].mean().rename('customer_avg_price')
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=market_avg_price.index.to_timestamp(), y=market_avg_price, mode='lines+markers', name='시장 전체 평균 단가', line=dict(width=3)))
        fig4.add_trace(go.Scatter(x=customer_avg_price.index.to_timestamp(), y=customer_avg_price, mode='lines+markers', name=f'{customer_name} 평균 단가', line=dict(color='red')))
        
        if top_competitors_list:
            st.info(f"**벤치마크: 경쟁 우위 그룹 평균**")
            st.caption("※ '경쟁 우위 그룹'은 '구매 경쟁력 분석'의 순위에서 현재 선택된 고객사보다 높은 순위를 기록한 모든 기업들의 평균입니다.")
            top_competitors_df = market_df[market_df['masked_name'].isin(top_competitors_list)]
            top_competitors_avg_price = top_competitors_df.groupby('year_month')['unit_price'].mean().rename('top_competitors_avg_price')
            fig4.add_trace(go.Scatter(x=top_competitors_avg_price.index.to_timestamp(), y=top_competitors_avg_price, mode='lines+markers', name='경쟁 우위 그룹 평균', line=dict(color='green', dash='dash')))
        else:
            st.success(f"**벤치마크 분석:** `{customer_name}`님이 현재 시장에서 가장 우수한 구매 경쟁력을 보이고 있습니다!")

        fig4.update_layout(title=f'<b>[{analyzed_product_name}] 단가 추세</b>', xaxis_title='연-월', yaxis_title='평균 단가(USD/KG)')
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("##### 전체 기간 평균 단가 비교")
        col1, col2, col3 = st.columns(3)
        col1.metric("시장 전체 평균", f"${market_df['unit_price'].mean():.2f}")
        col2.metric(f"{customer_name} 평균", f"${customer_market_df['unit_price'].mean():.2f}")
        if top_competitors_list:
            col3.metric("경쟁 우위 그룹 평균", f"${top_competitors_df['unit_price'].mean():.2f}")

        if top_competitors_list:
            st.subheader("경쟁 우위 그룹 벤치마킹 시뮬레이션")
            with st.form("simulation_form"):
                sim_start_date = st.date_input("시뮬레이션 시작일", contract_date)
                sim_end_date = st.date_input("시뮬레이션 종료일")
                run_simulation = st.form_submit_button("예상 절감액 계산")
            
            if run_simulation:
                sim_df = pd.merge(customer_avg_price, top_competitors_avg_price, left_index=True, right_index=True, how='inner')
                customer_volume_monthly = customer_market_df.groupby('year_month')['volume'].sum()
                sim_df = pd.merge(sim_df, customer_volume_monthly, left_index=True, right_index=True, how='inner')
                
                sim_period_start = pd.to_datetime(sim_start_date).to_period('M')
                sim_period_end = pd.to_datetime(sim_end_date).to_period('M')
                sim_df = sim_df[(sim_df.index >= sim_period_start) & (sim_df.index <= sim_period_end)]
                
                if not sim_df.empty:
                    sim_df['potential_savings'] = (sim_df['customer_avg_price'] - sim_df['top_competitors_avg_price']) * sim_df['volume']
                    total_potential_savings = sim_df[sim_df['potential_savings'] > 0]['potential_savings'].sum()
                    st.success(f"해당 기간 동안 **경쟁 우위 그룹**의 평균 단가를 따랐다면 **${total_potential_savings:,.2f}**를 추가로 절감할 수 있었습니다.")
                    st.caption("※ 이 금액은 고객사의 월평균 단가가 경쟁 우위 그룹보다 높았던 달의 절감 가능액만을 합산한 값입니다.")
                else:
                    st.warning("해당 기간에 비교할 데이터가 없습니다.")
