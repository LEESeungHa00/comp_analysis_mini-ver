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
    st.write("특정 품목에 대한 전체 시장 데이터를 업로드하여, 기준 업체의 시장 내 경쟁력을 심층 분석합니다.")
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
    
    masked_customer_name = market_df[market_df['importer_name'] == customer_name]['masked_name'].iloc[0]
    
    st.subheader(f"'{analyzed_product_name}' 품목 시장 분석 결과 (기준 업체: {customer_name})")

    with st.expander(f"1. [{analyzed_product_name}] 구매 경쟁력 분석", expanded=True):
        st.markdown("##### Volume 대비 Unit Price 분포 및 시장 추세")
        fig_comp = px.scatter(
            market_df, x='volume', y='unit_price', trendline="lowess",
            trendline_color_override="red", hover_data=['masked_name', 'date'], 
            title="<b>시장 내 거래 분포 및 평균 가격 추세선</b><br><span style='font-size: 0.8em; color:grey;'>LOWESS 회귀분석 기반</span>",
            labels={'volume': '수입량(KG)', 'unit_price': '단가(USD/KG)'}
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("##### 구매 경쟁력 상위 10개사")
        top_10_competitors = all_competitors_ranked.head(10)
        
        def highlight_customer(row):
            color = 'background-color: lightblue' if row.masked_name == masked_customer_name else ''
            return [color] * len(row)
        
        st.dataframe(top_10_competitors.style.apply(highlight_customer, axis=1).format({'competitiveness_index': '{:,.2f}'}))
        
        customer_rank_info = all_competitors_ranked[all_competitors_ranked['masked_name'] == masked_customer_name]
        if not customer_rank_info.empty:
            customer_rank = customer_rank_info.index[0] + 1
            if customer_rank > 10:
                st.info(f"참고: **{customer_name}**의 구매 경쟁력 순위는 전체 {len(all_competitors_ranked)}개사 중 **{customer_rank}위**입니다.")

    with st.expander(f"2. [{analyzed_product_name}] 단가 추세 및 경쟁 우위 그룹 벤치마킹", expanded=True):
        st.markdown("##### 월별 평균 단가 추세")
        market_avg_price = market_df.groupby('year_month')['unit_price'].mean().rename('market_avg_price')
        customer_market_df = market_df[market_df['masked_name'] == masked_customer_name]
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
                customer_volume_monthly = customer_market_df.groupby('year_month')['volumimport streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import time
import altair as alt

# ---------------------------------
# 페이지 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="수입 실적 분석 대시보드",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------
# 상수 정의
# ---------------------------------
# 분석 기준 컬럼 정의
PRIMARY_WEIGHT_COL = '적합 중량(KG)'
PRIMARY_AMOUNT_COL = '적합 금액($)'

# 구글 시트에서 불러올 헤더 정의
DESIRED_HEADER = [
    'NO', 'Year', 'Month', '제품구분별', '제조국(원산지)별', '수출국별',
    '수입용도별', '대표품목별', '총 중량(KG)', '총 금액($)', '적합 중량(KG)',
    '적합 금액($)', '부적합 중량(KG)', '부적합 금액($)'
]
GOOGLE_SHEET_NAME = "수입실적_DB"
WORKSHEET_NAME = "월별통합"

# ---------------------------------
# 구글 시트 연동 설정
# ---------------------------------
def get_google_sheet_client():
    """Streamlit의 Secrets를 사용하여 구글 시트 API에 연결하고 클라이언트 객체를 반환합니다."""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        scopes = [
            "https.www.googleapis.com/auth/spreadsheets",
            "https.www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"🚨 구글 시트 인증 중 오류가 발생했습니다: {e}")
        return None

# ---------------------------------
# 데이터 로딩 및 전처리
# ---------------------------------
def preprocess_dataframe(df):
    """데이터프레임의 숫자 컬럼을 정리하고 날짜 관련 파생 변수를 생성합니다."""
    df_copy = df.copy()
    numeric_cols = [
        '총 중량(KG)', '총 금액($)', '적합 중량(KG)', '적합 금액($)',
        '부적합 중량(KG)', '부적합 금액($)'
    ]
    for col in numeric_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(
                df_copy[col].astype(str).str.replace(',', ''),
                errors='coerce'
            ).fillna(0)

    if 'Year' in df_copy.columns and 'Month' in df_copy.columns:
        df_copy['Year'] = pd.to_numeric(df_copy['Year'], errors='coerce')
        df_copy['Month'] = pd.to_numeric(df_copy['Month'], errors='coerce')
        df_copy['날짜'] = pd.to_datetime(
            df_copy['Year'].astype('Int64').astype(str) + '-' + df_copy['Month'].astype('Int64').astype(str) + '-01',
            errors='coerce'
        )
        valid_dates = df_copy['날짜'].notna()
        df_copy.loc[valid_dates, '연도'] = df_copy.loc[valid_dates, '날짜'].dt.year
        df_copy.loc[valid_dates, '월'] = df_copy.loc[valid_dates, '날짜'].dt.month
        df_copy.loc[valid_dates, '분기'] = df_copy.loc[valid_dates, '날짜'].dt.quarter
        df_copy.loc[valid_dates, '반기'] = (df_copy.loc[valid_dates, '날짜'].dt.month - 1) // 6 + 1
    return df_copy

@st.cache_data(ttl=3600)
def load_data():
    """구글 시트에서 데이터를 로드하고, 실패 시 샘플 데이터를 생성합니다."""
    client = get_google_sheet_client()
    if client is None:
        st.warning("구글 시트 연동에 실패하여 샘플 데이터로 앱을 실행합니다.")
        return create_sample_data()
    try:
        sheet = client.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)
        all_data = sheet.get_all_values()
        if not all_data or len(all_data) < 2:
            st.info("시트에 헤더 또는 데이터가 없습니다.")
            return pd.DataFrame()

        header = all_data[0]
        data = all_data[1:]
        
        desired_set = set(DESIRED_HEADER)
        header_set = set(header)
        if desired_set != header_set:
            missing = desired_set - header_set
            extra = header_set - desired_set
            error_message = "🚨 구글 시트의 컬럼 구성이 올바르지 않습니다.\n"
            if missing: error_message += f"\n**- 누락된 컬럼:** `{', '.join(missing)}`"
            if extra: error_message += f"\n**- 불필요한 컬럼:** `{', '.join(extra)}`"
            st.error(error_message)
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=header)
        df.dropna(how='all', inplace=True)
        if not df.empty:
            df = preprocess_dataframe(df)
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"🚨 구글 시트 파일을 찾을 수 없습니다. 이름이 '{GOOGLE_SHEET_NAME}'인지, 서비스 계정에 공유되었는지 확인해주세요.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"데이터 로딩 중 예상치 못한 오류 발생: {e}")
        return create_sample_data()

def create_sample_data():
    """분석용 샘플 데이터를 생성합니다."""
    items = ['소고기(냉장)', '바지락(활)', '김치', '과자', '맥주', '새우(냉동)', '오렌지', '바나나', '커피원두', '치즈']
    categories = {
        '소고기(냉장)': '축산물', '바지락(활)': '수산물', '김치': '가공식품', 
        '과자': '가공식품', '맥주': '가공식품', '새우(냉동)': '수산물', 
        '오렌지': '농산물', '바나나': '농산물', '커피원두': '농산물', '치즈': '축산물'
    }
    daterange = pd.date_range(start='2021-01-01', end='2025-07-31', freq='M')
    data = []
    no_counter = 1
    for date in daterange:
        for item in items:
            weight = (10000 + items.index(item) * 5000) * np.random.uniform(0.8, 1.2)
            price = weight * np.random.uniform(5, 10)
            data.append([
                no_counter, date.year, date.month, categories[item], '미국', '미국', '판매용',
                item, weight, price, weight*0.95, price*0.95, weight*0.05, price*0.05
            ])
            no_counter += 1
    df = pd.DataFrame(data, columns=DESIRED_HEADER)
    df = preprocess_dataframe(df)
    return df

def update_sheet_in_batches(worksheet, dataframe, batch_size=10000):
    """데이터프레임을 작은 배치로 나누어 구글 시트에 업로드합니다."""
    worksheet.clear()
    worksheet.append_row(dataframe.columns.values.tolist())
    
    data = dataframe.fillna('').values.tolist()
    total_rows = len(data)
    
    if total_rows == 0:
        st.success("✅ 업로드 완료! (업로드할 데이터 없음)")
        return

    progress_bar = st.progress(0, text="데이터 업로드를 시작합니다...")
    
    for i in range(0, total_rows, batch_size):
        batch = data[i:i+batch_size]
        worksheet.append_rows(batch, value_input_option='USER_ENTERED')
        
        progress_percentage = min((i + batch_size) / total_rows, 1.0)
        progress_text = f"{min(i + batch_size, total_rows)} / {total_rows} 행 업로드 중..."
        progress_bar.progress(progress_percentage, text=progress_text)
        
        time.sleep(1)
        
    progress_bar.progress(1.0, text="✅ 업로드 완료!")

# ---------------------------------
# ---- 메인 애플리케이션 로직 ----
# ---------------------------------

# 1. 사이드바 메뉴 및 분석 모드 선택
st.sidebar.title("메뉴")
analysis_mode = st.sidebar.radio(
    "분석 기준",
    ('중량 모드', '금액 모드'),
    horizontal=True
)
menu = st.sidebar.radio(
    "원하는 기능을 선택하세요.",
    ("수입 현황 대시보드", "시계열 추세 분석", "기간별 추이 분석", "데이터 추가")
)
if st.sidebar.button("🔄 데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

# 2. 선택된 모드에 따라 동적 변수 설정
if analysis_mode == '중량 모드':
    primary_col = PRIMARY_WEIGHT_COL
    unit = '(KG)'
    value_name = '수입량'
    change_name = '증감량'
    format_str = '{:,.0f}'
    axis_format = '~s'
    label_expr = """
    datum.value == 0 ? '0' : 
    (abs(datum.value) >= 1000000 ? format(abs(datum.value) / 1000000, ',.0f') + 'M' : 
    (abs(datum.value) >= 1000 ? format(abs(datum.value) / 1000, ',.0f') + 'K' : format(abs(datum.value), ',.0f')))
    """
else: # 금액 모드
    primary_col = PRIMARY_AMOUNT_COL
    unit = '($)'
    value_name = '수입액'
    change_name = '증감액'
    format_str = '${:,.0f}'
    axis_format = '$,.0s'
    label_expr = """
    datum.value == 0 ? '$0' : 
    (abs(datum.value) >= 1000000 ? '$' + format(abs(datum.value) / 1000000, ',.0f') + 'M' : 
    (abs(datum.value) >= 1000 ? '$' + format(abs(datum.value) / 1000, ',.0f') + 'K' : '$' + format(abs(datum.value), ',.0f')))
    """

df = load_data()

if df.empty and menu != "데이터 추가":
    st.warning("데이터가 없습니다. '데이터 추가' 탭으로 이동하여 데이터를 업로드해주세요.")
    st.stop()

# ----- [수정] 줌/팬 동작 재정의 -----
# 1. 확대 (그냥 드래그)
zoom_on_drag = alt.selection_interval(
    bind='scales',
    on="[mousedown[!event.shiftKey], mouseup] > mousemove", # Shift 키가 눌리지 않은 상태의 드래그
    empty='all'
)
# 2. 이동 (Shift + 드래그)
pan_on_shift_drag = alt.selection_interval(
    bind='scales',
    on="[mousedown[event.shiftKey], mouseup] > mousemove", # Shift 키가 눌린 상태의 드래그
    empty='all'
)
# -----------------------------------


# --- 대시보드 페이지 ---
if menu == "수입 현황 대시보드":
    st.title(f"📊 수입 현황 대시보드")
    st.info(f"(기준: {primary_col})")

    analysis_df_raw = df.dropna(subset=['날짜', primary_col, '연도', '분기', '반기'])
    if analysis_df_raw.empty:
        st.warning("분석할 유효한 데이터가 없습니다. 'Year', 'Month' 데이터가 올바른지 확인해주세요.")
        st.stop()
    
    available_years = sorted(analysis_df_raw['연도'].unique().astype(int), reverse=True)
    available_months = sorted(analysis_df_raw['월'].unique().astype(int))
    latest_date = analysis_df_raw['날짜'].max()

    def create_butterfly_chart_altair(df_agg, base_col, prev_col, base_label, prev_label):
        """증감 상위/하위 품목에 대한 나비 차트를 생성합니다."""
        change_col = f'{change_name}{unit}'
        top_items = df_agg.nlargest(5, change_col)
        bottom_items = df_agg.nsmallest(5, change_col)
        chart_data = pd.concat([top_items, bottom_items])
        
        if chart_data.empty:
            st.info("비교할 증감 내역이 있는 품목이 없습니다.")
            return

        chart_data = chart_data.reset_index()
        df_melted = chart_data.melt(
            id_vars='대표품목별', value_vars=[prev_col, base_col],
            var_name='시점_컬럼명', value_name=f'{value_name}{unit}'
        )
        df_melted['차트_값'] = df_melted.apply(
            lambda row: -row[f'{value_name}{unit}'] if row['시점_컬럼명'] == prev_col else row[f'{value_name}{unit}'],
            axis=1
        )
        df_melted['시점'] = df_melted['시점_컬럼명'].map({prev_col: prev_label, base_col: base_label})
        sort_order = chart_data.sort_values(change_col, ascending=False)['대표품목별'].tolist()
        
        final_chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X('차트_값:Q', title=f'{value_name} {unit}', axis=alt.Axis(labelExpr=label_expr)),
            y=alt.Y('대표품목별:N', sort=sort_order, title=None),
            color=alt.Color('시점:N',
                scale=alt.Scale(domain=[prev_label, base_label], range=['#5f8ad6', '#d65f5f']),
                legend=alt.Legend(title="시점 구분", orient='top')
            ),
            tooltip=[
                alt.Tooltip('대표품목별', title='품목'),
                alt.Tooltip('시점', title='기간'),
                alt.Tooltip(f'{value_name}{unit}', title=value_name, format=',.0f')
            ]
        ).properties(
            title=alt.TitleParams(text=f'{prev_label} vs {base_label} {value_name} 비교', anchor='middle')
        ).add_params( # [수정] .interactive() 대신 add_params 사용
            zoom_on_drag,
            pan_on_shift_drag
        )
        
        st.altair_chart(final_chart, use_container_width=True)

    def display_comparison_tab(title, current_data, prev_data, base_label, prev_label):
        """비교 분석 탭의 UI와 로직을 표시하는 함수."""
        st.subheader(f"🆚 {title}")
        current_agg = current_data.groupby('대표품목별')[primary_col].sum()
        prev_agg = prev_data.groupby('대표품목별')[primary_col].sum()

        base_col_name = f'기준_{value_name}{unit}'
        prev_col_name = f'이전_{value_name}{unit}'
        change_col_name = f'{change_name}{unit}'
        rate_col_name = '증감률'

        df_agg = pd.DataFrame(current_agg).rename(columns={primary_col: base_col_name})
        df_agg = df_agg.join(prev_agg.rename(prev_col_name), how='outer').fillna(0)
        df_agg[change_col_name] = df_agg[base_col_name] - df_agg[prev_col_name]
        df_agg[rate_col_name] = df_agg[change_col_name] / df_agg[prev_col_name].replace(0, np.nan)
        
        with st.expander("📊 Before & After (증감 상위/하위 5개 품목)"):
            create_butterfly_chart_altair(df_agg, base_col_name, prev_col_name, base_label, prev_label)
        
        formatter = {
            base_col_name: format_str,
            prev_col_name: format_str,
            change_col_name: f'{{:+,.0f}}' if analysis_mode == '중량 모드' else f'${{:+,.0f}}',
            rate_col_name: '{:+.2%}'
        }
        
        st.markdown(f'<p style="color:red; font-weight:bold;">🔼 {value_name} 증가 TOP 5 ({change_name} 많은 순)</p>', unsafe_allow_html=True)
        st.dataframe(df_agg.nlargest(5, change_col_name).reset_index().style.format(formatter, na_rep="-"), hide_index=True, use_container_width=True)
        
        st.markdown(f'<p style="color:blue; font-weight:bold;">🔽 {value_name} 감소 TOP 5 ({change_name} 많은 순)</p>', unsafe_allow_html=True)
        st.dataframe(df_agg.nsmallest(5, change_col_name).reset_index().style.format(formatter, na_rep="-"), hide_index=True, use_container_width=True)

        st.markdown(f'<p style="color:green; font-weight:bold;">❇️ 신규 수입 품목 TOP 10 (이전 기간 0)</p>', unsafe_allow_html=True)
        
        new_items_df = df_agg[
            (df_agg[base_col_name] > 0) & (df_agg[prev_col_name] == 0)
        ]
        
        if new_items_df.empty:
            st.info("해당 기간에 신규로 수입된 품목이 없습니다.")
        else:
            new_items_top10 = new_items_df.sort_values(
                by=base_col_name, ascending=False
            ).head(10).reset_index()
            
            final_new_items_df = new_items_top10.rename(
                columns={'대표품목별': '품목명'}
            )[['품목명', base_col_name, prev_col_name]]
            
            st.dataframe(
                final_new_items_df.style.format(formatter, na_rep="-"), 
                hide_index=True,
                use_container_width=True
            )

    tab_yy, tab_mom, tab_yoy, tab_qoq, tab_hoh = st.tabs([
        "전년 대비", "전월 대비", "전년 동월 대비", "전년 동분기 대비", "전년 동반기 대비"
    ])

    with tab_yy:
        yy_year = st.selectbox("기준 연도", available_years, key="yy_year", index=0)
        current_yy_data = analysis_df_raw[analysis_df_raw['연도'] == yy_year]
        prev_yy_data = analysis_df_raw[analysis_df_raw['연도'] == yy_year - 1]
        display_comparison_tab(f"전년 대비 {value_name} 분석", current_yy_data, prev_yy_data, f'{yy_year}년', f'{yy_year-1}년')

    with tab_mom:
        mom_col1, mom_col2 = st.columns(2)
        with mom_col1:
            mom_year = st.selectbox("기준 연도", available_years, key="mom_year", index=0)
        with mom_col2:
            mom_month = st.selectbox("기준 월", available_months, key="mom_month", index=available_months.index(latest_date.month))
        current_date = datetime(mom_year, mom_month, 1)
        prev_month_date = current_date - pd.DateOffset(months=1)
        current_data = analysis_df_raw[(analysis_df_raw['연도'] == mom_year) & (analysis_df_raw['월'] == mom_month)]
        prev_data = analysis_df_raw[(analysis_df_raw['연도'] == prev_month_date.year) & (analysis_df_raw['월'] == prev_month_date.month)]
        display_comparison_tab(f"전월 대비 {value_name} 분석", current_data, prev_data, f'{mom_year}년 {mom_month}월', f'{prev_month_date.year}년 {prev_month_date.month}월')

    with tab_yoy:
        yoy_col1, yoy_col2 = st.columns(2)
        with yoy_col1:
            yoy_year = st.selectbox("기준 연도", available_years, key="yoy_year", index=0)
        with yoy_col2:
            yoy_month = st.selectbox("기준 월", available_months, key="yoy_month", index=available_months.index(latest_date.month))
        current_data_yoy = analysis_df_raw[(analysis_df_raw['연도'] == yoy_year) & (analysis_df_raw['월'] == yoy_month)]
        prev_year_data = analysis_df_raw[(analysis_df_raw['연도'] == yoy_year - 1) & (analysis_df_raw['월'] == yoy_month)]
        display_comparison_tab(f"전년 동월 대비 {value_name} 분석", current_data_yoy, prev_year_data, f'{yoy_year}년 {yoy_month}월', f'{yoy_year-1}년 {yoy_month}월')

    with tab_qoq:
        q_col1, q_col2 = st.columns(2)
        default_quarter = (latest_date.month - 1) // 3 + 1
        with q_col1:
            q_year = st.selectbox("기준 연도", available_years, key="q_year", index=0)
        with q_col2:
            q_quarter = st.selectbox("기준 분기", [1, 2, 3, 4], key="q_quarter", index=int(default_quarter - 1))
        current_q_data = analysis_df_raw[(analysis_df_raw['연도'] == q_year) & (analysis_df_raw['분기'] == q_quarter)]
        prev_q_data = analysis_df_raw[(analysis_df_raw['연도'] == q_year - 1) & (analysis_df_raw['분기'] == q_quarter)]
        display_comparison_tab(f"전년 동분기 대비 {value_name} 분석", current_q_data, prev_q_data, f'{q_year}년 {q_quarter}분기', f'{q_year-1}년 {q_quarter}분기')

    with tab_hoh:
        h_col1, h_col2 = st.columns(2)
        default_half = (latest_date.month - 1) // 6 + 1
        half_display = lambda x: f"{'상반기' if x == 1 else '하반기'}"
        with h_col1:
            h_year = st.selectbox("기준 연도", available_years, key="h_year", index=0)
        with h_col2:
            h_half = st.selectbox("기준 반기", [1, 2], key="h_half", index=int(default_half - 1), format_func=half_display)
        current_h_data = analysis_df_raw[(analysis_df_raw['연도'] == h_year) & (analysis_df_raw['반기'] == h_half)]
        prev_h_data = analysis_df_raw[(analysis_df_raw['연도'] == h_year - 1) & (analysis_df_raw['반기'] == h_half)]
        display_comparison_tab(f"전년 동반기 대비 {value_name} 분석", current_h_data, prev_h_data, f'{h_year}년 {half_display(h_half)}', f'{h_year-1}년 {half_display(h_half)}')

# --- 시계열 추세 분석 페이지 ---
elif menu == "시계열 추세 분석":
    st.title(f"📈 시계열 추세 분석 (기준: {primary_col})")
    st.info("선택한 기간 동안 꾸준한 증가 또는 감소 추세를 보이는 품목을 식별합니다.")
    
    trend_df = df.dropna(subset=['날짜', primary_col, '연도', '월'])
    if trend_df.empty:
        st.warning("분석할 유효한 데이터가 없습니다.")
        st.stop()

    # --- 연도별 - 장기 추세 분석 ---
    st.markdown("---")
    st.subheader("연도별 - 장기 추세 분석")

    yearly_agg = trend_df.groupby(['연도', '대표품목별'])[primary_col].sum().reset_index()
    available_years_trend = sorted(yearly_agg['연도'].unique().astype(int))

    if len(available_years_trend) >= 2:
        start_y, end_y = st.select_slider(
            '분석 기간 (년)',
            options=available_years_trend,
            value=(available_years_trend[0], available_years_trend[-1]),
            key='yearly_slider'
        )
        duration_years = end_y - start_y + 1
        st.caption(f"선택된 기간 : **{duration_years}년** ({start_y}년 ~ {end_y}년)")
        
        trend_type_years = st.radio("추세 선택", ("지속 증가 📈", "지속 감소 📉"), horizontal=True, key="trend_type_years")

        period_df_yearly = yearly_agg[(yearly_agg['연도'] >= start_y) & (yearly_agg['연도'] <= end_y)]
        results_yearly = []
        for item, group in period_df_yearly.groupby('대표품목별'):
            if len(group['연도'].unique()) == duration_years:
                group = group.sort_values('연도')
                diffs = group[primary_col].diff().dropna()
                if (trend_type_years == "지속 증가 📈" and (diffs > 0).all()) or \
                   (trend_type_years == "지속 감소 📉" and (diffs < 0).all()):
                    
                    start_val = group.iloc[0][primary_col]
                    end_val = group.iloc[-1][primary_col]
                    growth_rate = (end_val - start_val) / start_val if start_val > 0 else (np.inf if end_val > 0 else 0)
                    results_yearly.append({
                        '대표품목별': item,
                        f'{start_y}년_{value_name}{unit}': start_val, f'{end_y}년_{value_name}{unit}': end_val,
                        '기간내_증감률': growth_rate
                    })
        
        if results_yearly:
            result_df_yearly = pd.DataFrame(results_yearly)
            sort_col = '기간내_증감률'
            result_df_yearly = result_df_yearly.nlargest(10, sort_col) if trend_type_years == "지속 증가 📈" else result_df_yearly.nsmallest(10, sort_col)
            
            st.markdown(f"**선택 기간 동안 `{trend_type_years}` 품목 TOP 10**")
            st.dataframe(result_df_yearly.style.format({
                f'{start_y}년_{value_name}{unit}': format_str, f'{end_y}년_{value_name}{unit}': format_str,
                '기간내_증감률': '{:+.2%}'
            }, na_rep="-"), hide_index=True)

            if not result_df_yearly.empty:
                st.markdown("---")
                st.subheader("개별 품목 연도별 추이 그래프")
                selected_item_y = st.selectbox("그래프로 확인할 품목을 선택하세요", options=result_df_yearly['대표품목별'], key="selected_item_y")
                if selected_item_y:
                    item_trend_df_y = period_df_yearly[period_df_yearly['대표품목별'] == selected_item_y]
                    chart_y = alt.Chart(item_trend_df_y).mark_line(point=True).encode(
                        x=alt.X('연도:O', title='연도'),
                        y=alt.Y(f'{primary_col}:Q', title=f'{value_name} {unit}', axis=alt.Axis(format=axis_format)),
                        tooltip=['연도', alt.Tooltip(f'{primary_col}', title=value_name, format=',.0f')]
                    ).properties(title=f"'{selected_item_y}'의 {start_y}년 ~ {end_y}년 {value_name} 추이"
                    ).add_params( # [수정] .interactive() 대신 add_params 사용
                        zoom_on_drag,
                        pan_on_shift_drag
                    )
                    st.altair_chart(chart_y, use_container_width=True)
    else:
        st.warning("연도별 추세를 분석하려면 최소 2년 이상의 데이터가 필요합니다.")

    # --- 월별 - 단기 추세 분석 ---
    st.markdown("---")
    st.subheader("월별 - 단기 추세 분석")
    monthly_periods = sorted(trend_df['날짜'].dt.to_period('M').unique().astype(str))
    if len(monthly_periods) >= 3:
        start_m, end_m = st.select_slider(
            '분석 기간 (월)',
            options=monthly_periods,
            value=(monthly_periods[0], monthly_periods[-1]),
            key='monthly_slider'
        )
        start_date = pd.to_datetime(start_m).to_pydatetime()
        end_date = pd.to_datetime(end_m).to_pydatetime()
        duration_months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
        st.caption(f"선택된 기간: **{duration_months}개월** ({start_m} ~ {end_m})")
        
        trend_type_months = st.radio("추세 선택", ("지속 증가 📈", "지속 감소 📉"), horizontal=True, key="trend_type_months")
        
        period_df_monthly = trend_df[(trend_df['날짜'] >= start_date) & (trend_df['날짜'] <= end_date)]
        results_monthly = []
        for item, group in period_df_monthly.groupby('대표품목별'):
            if len(group['날짜'].dt.to_period('M').unique()) == duration_months:
                monthly_agg = group.groupby(pd.Grouper(key='날짜', freq='M'))[primary_col].sum()
                diffs = monthly_agg.diff().dropna()
                if (trend_type_months == "지속 증가 📈" and (diffs > 0).all()) or \
                   (trend_type_months == "지속 감소 📉" and (diffs < 0).all()):
                    start_val = monthly_agg.iloc[0]
                    end_val = monthly_agg.iloc[-1]
                    growth_rate = (end_val - start_val) / start_val if start_val > 0 else (np.inf if end_val > 0 else 0)
                    results_monthly.append({
                        '대표품목별': item,
                        f'시작월_{value_name}{unit}': start_val, f'종료월_{value_name}{unit}': end_val,
                        '기간내_증감률': growth_rate
                    })
        
        if results_monthly:
            result_df_monthly = pd.DataFrame(results_monthly)
            sort_col = '기간내_증감률'
            result_df_monthly = result_df_monthly.nlargest(10, sort_col) if trend_type_months == "지속 증가 📈" else result_df_monthly.nsmallest(10, sort_col)
            
            st.markdown(f"**선택 기간 동안 `{trend_type_months}` 품목 TOP 10**")
            st.dataframe(result_df_monthly.style.format({
                f'시작월_{value_name}{unit}': format_str, f'종료월_{value_name}{unit}': format_str,
                '기간내_증감률': '{:+.2%}'
            }, na_rep="-"), hide_index=True)

            if not result_df_monthly.empty:
                st.markdown("---")
                st.subheader("개별 품목 월별 추이 그래프")
                selected_item_m = st.selectbox("그래프로 확인할 품목을 선택하세요", options=result_df_monthly['대표품목별'], key="selected_item_m")
                if selected_item_m:
                    item_trend_df_m = period_df_monthly[period_df_monthly['대표품목별'] == selected_item_m]
                    monthly_item_agg = item_trend_df_m.groupby(pd.Grouper(key='날짜', freq='M'))[primary_col].sum().reset_index()
                    monthly_item_agg['기간'] = monthly_item_agg['날짜'].dt.strftime('%Y-%m')
                    chart_m = alt.Chart(monthly_item_agg).mark_line(point=True).encode(
                        x=alt.X('기간:N', sort=None, title='월'),
                        y=alt.Y(f'{primary_col}:Q', title=f'{value_name} {unit}', axis=alt.Axis(format=axis_format)),
                        tooltip=['기간', alt.Tooltip(f'{primary_col}', title=value_name, format=',.0f')]
                    ).properties(title=f"'{selected_item_m}'의 {start_m} ~ {end_m} {value_name} 추이"
                    ).add_params( # [수정] .interactive() 대신 add_params 사용
                        zoom_on_drag,
                        pan_on_shift_drag
                    )
                    st.altair_chart(chart_m, use_container_width=True)
    else:
        st.warning("월별 추세를 분석하려면 최소 3개월 이상의 데이터가 필요합니다.")

# --- 기간별 추이 분석 페이지 ---
elif menu == "기간별 추이 분석":
    st.title(f"📆 기간별 {value_name} 추이 분석 (기준: {primary_col})")
    st.markdown("---")
    analysis_df = df.dropna(subset=['날짜', primary_col, '연도', '월', '분기', '반기', '제품구분별', '대표품목별'])
    if analysis_df.empty:
        st.warning("분석할 유효한 데이터가 없습니다.")
        st.stop()
    
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        period_type = st.radio("분석 기간 단위", ('월별', '분기별', '반기별'))
    
    all_categories = sorted(analysis_df['제품구분별'].unique())
    all_items = sorted(analysis_df['대표품목별'].unique())

    with col2:
        # [수정] UI 텍스트 명확화
        st.markdown("##### 1. 제품구분별 선택 (최대 5개)")
        st.info("기본적으로 '카테고리' 그래프가 그려집니다. 2번에서 품목 선택 시 '필터'로 동작합니다.")
        selected_categories = st.multiselect(
            "제품구분별 선택",
            options=all_categories,
            placeholder="카테고리를 선택하세요 (최대 5개)",
            label_visibility="collapsed",
            max_selections=5,
            key='cat_select'
        )
        
        if selected_categories:
            filtered_items_df = analysis_df[analysis_df['제품구분별'].isin(selected_categories)]
            available_items = sorted(filtered_items_df['대표품목별'].unique())
            item_placeholder = "선택한 카테고리 내 개별 품목 (최대 5개)"
        else:
            available_items = all_items
            item_placeholder = "전체 개별 품목 (최대 5개)"

        # [수정] UI 텍스트 명확화
        st.markdown("##### 2. 대표품목별 선택 (최대 5개)")
        st.info("여기에 품목을 선택하면, 그래프는 '품목' 기준으로 그려집니다.")
        selected_items = st.multiselect(
            "대표품목별 선택",
            options=available_items,
            placeholder=f"{item_placeholder}",
            label_visibility="collapsed",
            max_selections=5,
            key='item_select'
        )

    agg_df = pd.DataFrame()
    
    if selected_items:
        graph_title = "대표품목별 추이"
        agg_by_col = '대표품목별'
        filtered_df = analysis_df[analysis_df['대표품목별'].isin(selected_items)]
        if selected_categories:
             filtered_df = filtered_df[filtered_df['제품구분별'].isin(selected_categories)]
    
    elif selected_categories:
        graph_title = "제품구분별 추이"
        agg_by_col = '제품구분별'
        filtered_df = analysis_df[analysis_df['제품구분별'].isin(selected_categories)]
    
    else:
        st.info("그래프를 보려면 '제품구분별' 또는 '대표품목별'을 선택해주세요.")
        filtered_df = pd.DataFrame()
        agg_by_col = None

    if not filtered_df.empty and agg_by_col:
        agg_cols, title_suffix = [], ""
        if period_type == '월별':
            agg_cols, title_suffix = ['연도', '월'], f"월별 {value_name} 추이"
        elif period_type == '분기별':
            agg_cols, title_suffix = ['연도', '분기'], f"분기별 {value_name} 추이"
        elif period_type == '반기별':
            agg_cols, title_suffix = ['연도', '반기'], f"반기별 {value_name} 추이"
        
        agg_df = filtered_df.groupby(agg_cols + [agg_by_col])[primary_col].sum().unstack(fill_value=0)
        
        if agg_df.empty:
            st.info("선택한 항목에 대한 데이터가 없습니다.")
        else:
            if period_type == '월별':
                agg_df.index = agg_df.index.map(lambda x: f"{int(x[0])}-{int(x[1]):02d}")
            elif period_type == '분기별':
                agg_df.index = agg_df.index.map(lambda x: f"{int(x[0])}-{int(x[1])}분기")
            elif period_type == '반기별':
                agg_df.index = agg_df.index.map(lambda x: f"{int(x[0])}-{'상반기' if x[1] == 1 else '하반기'}")
            
            st.header(f"📈 {graph_title} - {title_suffix}")
            
            df_melted = agg_df.reset_index().melt(id_vars='index', var_name=agg_by_col, value_name=f'{value_name}{unit}')
            df_melted.rename(columns={'index': '기간'}, inplace=True)
            
            chart_type = st.radio("차트 종류", ('선 그래프', '막대 그래프'), horizontal=True, key="chart_type_trends")
            
            # [수정] .interactive() 대신 add_params를 base_chart에 적용
            base_chart = alt.Chart(df_melted).encode(
                x=alt.X('기간:N', sort=None, title='기간'),
                y=alt.Y(f'{value_name}{unit}:Q', title=f'{value_name} {unit}', axis=alt.Axis(format=axis_format)),
                color=alt.Color(f'{agg_by_col}:N', title='선택 항목'),
                tooltip=['기간', alt.Tooltip(f'{agg_by_col}', title='선택 항목'), alt.Tooltip(f'{value_name}{unit}', title=value_name, format=',.0f')]
            ).add_params(
                zoom_on_drag,
                pan_on_shift_drag
            )
            
            chart = base_chart.mark_line(point=True) if chart_type == '선 그래프' else base_chart.mark_bar()
            st.altair_chart(chart, use_container_width=True)
                
            with st.expander("데이터 상세 보기"):
                st.subheader(f"기간별 {value_name} {unit}")
                st.dataframe(agg_df.style.format(format_str))
                st.subheader("이전 기간 대비 증감률 (%)")
                growth_rate_df = agg_df.pct_change()
                st.dataframe(growth_rate_df.style.format("{:+.2%}", na_rep="-"))
    elif not selected_categories and not selected_items:
        pass
    else:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")

# --- 데이터 추가 페이지 ---
elif menu == "데이터 추가":
    st.title("📤 데이터 추가")
    st.info(f"다음 컬럼을 포함한 엑셀/CSV 파일을 업로드해주세요:\n`{', '.join(DESIRED_HEADER)}`")
    uploaded_file = st.file_uploader("파일 선택", type=['xlsx', 'csv'])
    password = st.text_input("업로드 비밀번호", type="password")
    if st.button("데이터베이스에 추가"):
        if uploaded_file and password == "1004":
            try:
                st.info("파일을 읽고 처리하는 중입니다...")
                new_df = pd.read_csv(uploaded_file, dtype=str) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, dtype=str)
                
                desired_set = set(DESIRED_HEADER)
                new_df_set = set(new_df.columns)
                if desired_set != new_df_set:
                    missing = desired_set - new_df_set
                    extra = new_df_set - desired_set
                    error_message = "🚨 업로드한 파일의 컬럼 구성이 올바르지 않습니다.\n"
                    if missing: error_message += f"\n**- 누락된 컬럼:** `{', '.join(missing)}`"
                    if extra: error_message += f"\n**- 불필요한 컬럼:** `{', '.join(extra)}`"
                    st.error(error_message)
                    st.stop()

                new_df_processed = preprocess_dataframe(new_df)
                client = get_google_sheet_client()
                if client:
                    sheet = client.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)
                    
                    unique_periods = new_df_processed.dropna(subset=['연도', '월'])[['연도', '월']].drop_duplicates()
                    df_filtered = df.copy()
                    if not df_filtered.empty and not unique_periods.empty:
                        df_filtered['연도'] = pd.to_numeric(df_filtered['연도'], errors='coerce')
                        df_filtered['월'] = pd.to_numeric(df_filtered['월'], errors='coerce')
                        
                        merged = df_filtered.merge(unique_periods, on=['연도', '월'], how='left', indicator=True)
                        df_filtered = df_filtered[merged['_merge'] == 'left_only']

                    combined_df = pd.concat([df_filtered, new_df_processed], ignore_index=True)
                    combined_df.sort_values(by=['Year', 'Month', 'NO'], inplace=True, na_position='last')
                    
                    df_to_write = combined_df.reindex(columns=DESIRED_HEADER)
                    
                    update_sheet_in_batches(sheet, df_to_write)
                    st.cache_data.clear()
                else:
                    st.error("🚨 구글 시트 연결에 실패했습니다.")
            except Exception as e:
                st.error(f"데이터 처리/업로드 중 오류 발생: {e}")
        else:
            if not uploaded_file:
                st.warning("⚠️ 파일을 먼저 업로드해주세요.")
            else:
                st.error("🚨 비밀번호가 틀렸습니다.")
    if 'Exporter' in market_df.columns and 'origin_country' in market_df.columns:
        with st.expander(f"4. [{analyzed_product_name}] 공급망(공급사/원산지) 분석", expanded=True):
            years_with_data_exporter = sorted(market_df['year'].unique(), reverse=True)
            if years_with_data_exporter:
                selected_year_exporter = st.selectbox("공급망 분석 연도 선택", options=years_with_data_exporter, key=f"exporter_year_{analyzed_product_name}")
                exporter_analysis_df = market_df[market_df['year'] == selected_year_exporter]
                
                top_10_exporters_by_vol = exporter_analysis_df.groupby('Exporter')['volume'].sum().nlargest(10).index
                exporter_analysis_df_top10 = exporter_analysis_df[exporter_analysis_df['Exporter'].isin(top_10_exporters_by_vol)]

                st.subheader(f"{selected_year_exporter}년 분기별 공급사 단가 분포")
                fig9 = px.box(
                    exporter_analysis_df_top10, x='quarter', y='unit_price', color='Exporter', 
                    title=f"<b>{selected_year_exporter}년 분기별 공급사 단가 분포</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준 상위 10개 공급사</span>", 
                    labels={'quarter': '분기', 'unit_price': '단가(USD/KG)'}
                )
                st.plotly_chart(fig9, use_container_width=True)
                
                customer_exporters_in_year = exporter_analysis_df[exporter_analysis_df['masked_name'] == masked_customer_name]['Exporter'].unique()
                st.info(f"**{customer_name}**가 {selected_year_exporter}년에 거래한 공급사: **{', '.join(customer_exporters_in_year)}**")

                # ✅ 들여쓰기 오류 수정: for 루프의 시작 위치 바로잡음
                for exporter in customer_exporters_in_year:
                    st.markdown(f"--- \n #### 공급사 '{exporter}' 비교 분석")
                    single_exporter_df = exporter_analysis_df[exporter_analysis_df['Exporter'] == exporter]
                    
                    st.subheader(f"Volume 및 평균 단가 비교")
                    importer_summary = single_exporter_df.groupby('importer_name').agg(
                        total_volume=('volume', 'sum'),
                        avg_unit_price=('unit_price', 'mean')
                    ).sort_values('total_volume', ascending=False).reset_index()

                    fig8 = go.Figure()
                    fig8.add_trace(go.Bar(
                        x=importer_summary['importer_name'],
                        y=importer_summary['total_volume'],
                        name='총 수입량(KG)',
                        marker_color=['red' if imp == customer_name else 'lightskyblue' for imp in importer_summary['importer_name']]
                    ))
                    fig8.add_trace(go.Scatter(
                        x=importer_summary['importer_name'],
                        y=importer_summary['avg_unit_price'],
                        name='평균 수입단가(USD/KG)',
                        yaxis='y2',
                        mode='lines+markers',
                        line=dict(color='orange')
                    ))
                    fig8.update_layout(
                        title=f"<b>'{exporter}' 거래 업체별 Volume 및 평균 단가</b>",
                        xaxis_title='수입사',
                        yaxis=dict(title='총 수입량(KG)'),
                        yaxis2=dict(title='평균 수입단가(USD/KG)', overlaying='y', side='right'),
                        legend=dict(x=0, y=1.1, orientation='h')
                    )
                    st.plotly_chart(fig8, use_container_width=True)

                    st.subheader(f"단가 분포 비교")
                    top_10_importers_by_vol = single_exporter_df.groupby('importer_name')['volume'].sum().nlargest(10).index
                    single_exporter_df_top10 = single_exporter_df[single_exporter_df['importer_name'].isin(top_10_importers_by_vol)]
                    
                    importers_in_plot = single_exporter_df_top10['importer_name'].unique()
                    competitors = [imp for imp in importers_in_plot if imp != customer_name]
                    blue_shades = px.colors.sequential.Blues_r
                    color_map_box = {comp: blue_shades[i % len(blue_shades)] for i, comp in enumerate(competitors)}
                    color_map_box[customer_name] = 'red'

                    fig10 = px.box(
                        single_exporter_df_top10, x='importer_name', y='unit_price', 
                        title=f"<b>'{exporter}' 거래 업체별 단가 분포</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준 상위 10개 수입사</span>", 
                        labels={'importer_name': '수입사', 'unit_price': '단가(USD/KG)'},
                        color='importer_name', color_discrete_map=color_map_box
                    )
                    st.plotly_chart(fig10, use_container_width=True)
                    with st.expander("상세 데이터 보기"):
                        summary_df_imp = single_exporter_df_top10.groupby('importer_name')['unit_price'].agg(['max', 'mean', 'min']).reset_index()
                        summary_df_imp.columns = ['수입사', '최대 단가(USD/KG)', '평균 단가(USD/KG)', '최소 단가(USD/KG)']
                        st.dataframe(summary_df_imp.style.format({
                            '최대 단가(USD/KG)': '${:,.2f}', 
                            '평균 단가(USD/KG)': '${:,.2f}', 
                            '최소 단가(USD/KG)': '${:,.2f}'
                        }))

                # ✅ for 루프 바깥으로 정상 복귀
                st.subheader(f"{selected_year_exporter}년 분기별 대안 소싱 옵션")
                customer_origins = exporter_analysis_df[exporter_analysis_df['importer_name'] == customer_name]['origin_country'].unique()
                avg_prices = exporter_analysis_df.groupby(['quarter', 'Exporter', 'origin_country']).agg(
                    avg_price=('unit_price', 'mean'), 
                    representative_product=('product_name', 'first')
                ).reset_index()
                
                for q in range(1, 5):
                    st.markdown(f"--- \n #### {q}분기")
                    q_df = avg_prices[avg_prices['quarter'] == q]
                    if q_df.empty:
                        st.write("- 해당 분기에 거래 데이터가 없습니다.")
                        continue
                    
                    st.markdown("**현재 소싱 옵션**")
                    customer_exporters_q_df = q_df[q_df['Exporter'].isin(customer_exporters_in_year)].sort_values('avg_price')
                    if not customer_exporters_q_df.empty:
                        st.dataframe(
                            customer_exporters_q_df[['Exporter', 'avg_price']]
                            .rename(columns={'Exporter': '공급사', 'avg_price': '평균 단가(USD/KG)'})
                            .style.format({'평균 단가(USD/KG)': '${:,.2f}'})
                        )
                    else:
                        st.write("- 공급사 거래 없음")

                    customer_origins_q_df = (
                        q_df[q_df['origin_country'].isin(customer_origins)]
                        .groupby('origin_country')['avg_price']
                        .mean().reset_index().sort_values('avg_price')
                    )
                    if not customer_origins_q_df.empty:
                        st.dataframe(
                            customer_origins_q_df
                            .rename(columns={'origin_country': '원산지', 'avg_price': '평균 단가(USD/KG)'})
                            .style.format({'평균 단가(USD/KG)': '${:,.2f}'})
                        )
                    else:
                        st.write("- 원산지 거래 없음")

                    st.markdown("**대안 추천 옵션**")
                    customer_avg_price_q = q_df[q_df['Exporter'].isin(customer_exporters_in_year)]['avg_price'].mean()
                    if not pd.isna(customer_avg_price_q):
                        cheaper_exporters = q_df[(~q_df['Exporter'].isin(customer_exporters_in_year)) & (q_df['avg_price'] < customer_avg_price_q)].sort_values('avg_price')
                        if not cheaper_exporters.empty:
                            st.dataframe(
                                cheaper_exporters[['Exporter', 'representative_product', 'avg_price']]
                                .rename(columns={'Exporter': '추천 공급사', 'representative_product': '대표 품목', 'avg_price': '평균 단가(USD/KG)'})
                                .style.format({'평균 단가(USD/KG)': '${:,.2f}'})
                            )
                        else:
                            st.write("- 더 저렴한 공급사 없음")
                    
                    customer_origin_avg_price_q = (
                        q_df[q_df['origin_country'].isin(customer_origins)]
                        .groupby('origin_country')['avg_price'].mean().mean()
                    )
                    if not pd.isna(customer_origin_avg_price_q):
                        cheaper_origins = q_df.groupby('origin_country')['avg_price'].mean().reset_index()
                        cheaper_origins = cheaper_origins[(~cheaper_origins['origin_country'].isin(customer_origins)) & (cheaper_origins['avg_price'] < customer_origin_avg_price_q)].sort_values('avg_price')
                        if not cheaper_origins.empty:
                            st.dataframe(
                                cheaper_origins
                                .rename(columns={'origin_country': '추천 원산지', 'avg_price': '평균 단가(USD/KG)'})
                                .style.format({'평균 단가(USD/KG)': '${:,.2f}'})
                            )
                        else:
                            st.write("- 더 저렴한 원산지 없음")
    else:
        st.warning("'Exporter' 또는 'Origin Country' 컬럼이 없어 공급망 분석을 수행할 수 없습니다.")
