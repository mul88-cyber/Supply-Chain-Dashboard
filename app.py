import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import gspread
from google.oauth2.service_account import Credentials
from dateutil.relativedelta import relativedelta
from streamlit_echarts import st_echarts
import warnings
from tenacity import retry, stop_after_attempt, wait_exponential
import math
warnings.filterwarnings('ignore')

# --- AI FORECASTING IMPORTS ---
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    import logging
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# --- TAMBAHAN IMPORT BARU (VIZZU & ECHARTS) ---
try:
    from ipyvizzu import Data, Config, Style
    from ipyvizzustory import Story, Slide, Step
    VIZZU_AVAILABLE = True
except ImportError:
    VIZZU_AVAILABLE = False

try:
    from streamlit_echarts import st_echarts
    ECHARTS_AVAILABLE = True
except ImportError:
    ECHARTS_AVAILABLE = False
    # Buat dummy function untuk mencegah error
    def st_echarts(options=None, height="400px", width="100%", key=None, **kwargs):
        import streamlit as st
        st.warning("⚠️ streamlit-echarts tidak terinstall. Jalankan: pip install streamlit-echarts")
        return None
    
# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Inventory Intelligence Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS KHUSUS PRINT PDF (FIX BLANK PAGE) ---
st.markdown("""
<style>
    @media print {
        /* FIX UTAMA: Reset SEMUA element ke block/visible */
        * {
            overflow: visible !important;
            position: static !important;
            display: block !important;
            float: none !important;
            height: auto !important;
            max-height: none !important;
            width: auto !important;
            max-width: none !important;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
            break-inside: avoid !important;
        }

        /* Hide unnecessary elements */
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        .stButton,
        .stDeployButton,
        footer,
        .stDownloadButton,
        .stActionButton,
        button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"],
        .stAlert,
        .stMarkdown:has(button) {
            display: none !important;
            height: 0 !important;
            width: 0 !important;
            opacity: 0 !important;
            visibility: hidden !important;
        }

        /* Force main container to be visible */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            position: static !important;
            width: 100vw !important;
            height: auto !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
            display: block !important;
        }

        /* Force all content to be visible */
        section[data-testid="stMain"] > div,
        [data-testid="block-container"] {
            overflow: visible !important;
            height: auto !important;
            max-height: none !important;
            display: block !important;
            position: static !important;
            break-inside: avoid;
        }

        /* Charts and tables - force visibility */
        .element-container,
        .stDataFrame,
        .stPlotlyChart,
        .stAltairChart,
        [data-testid="stHorizontalBlock"] {
            break-inside: avoid-page !important;
            page-break-inside: avoid !important;
            overflow: visible !important;
        }

        /* Ensure text is black for printing */
        body, h1, h2, h3, h4, h5, h6, p, div, span {
            color: #000000 !important;
            background-color: white !important;
        }

        /* Remove shadows and gradients for print */
        .status-indicator,
        .inventory-card,
        .metric-highlight {
            box-shadow: none !important;
            background: white !important;
            border: 1px solid #ccc !important;
        }

        /* Fix for Plotly charts */
        .js-plotly-plot,
        .plotly,
        .plot-container {
            width: 100% !important;
            height: auto !important;
        }

        /* Add page breaks between major sections */
        .stTabs {
            break-after: page !important;
        }

        /* Ensure all content fits page width */
        .row {
            display: block !important;
        }

        .column {
            width: 100% !important;
            float: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Custom CSS Premium ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .status-indicator {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .status-indicator:hover {
        transform: translateY(-5px);
    }
    .status-under { 
        background: linear-gradient(135deg, #FF5252 0%, #FF1744 100%);
        color: white;
        border-left: 5px solid #D32F2F;
    }
    .status-accurate { 
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        border-left: 5px solid #1B5E20;
    }
    .status-over { 
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        border-left: 5px solid #E65100;
    }
    
    .inventory-card {
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .inventory-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    .card-replenish { 
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        color: #EF6C00;
        border: 2px solid #FF9800;
    }
    .card-ideal { 
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        color: #2E7D32;
        border: 2px solid #4CAF50;
    }
    .card-high { 
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        color: #C62828;
        border: 2px solid #F44336;
    }
    
    .metric-highlight {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
        border-top: 5px solid #667eea;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 1rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid #5a67d8 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .sankey-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* New CSS */
    .monthly-performance-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid;
    }
    
    .performance-under { border-left-color: #F44336; }
    .performance-accurate { border-left-color: #4CAF50; }
    .performance-over { border-left-color: #FF9800; }
    
    .highlight-row {
        background-color: #FFF9C4 !important;
        font-weight: bold !important;
    }
    
    .warning-badge {
        background: #FF5252;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .success-badge {
        background: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Compact metrics */
    .compact-metric {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    
    /* Brand performance */
    .brand-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
    }
    
    /* Financial cards */
    .financial-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-top: 4px solid;
        transition: all 0.3s ease;
    }
    .financial-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .card-revenue { border-top-color: #667eea; }
    .card-margin { border-top-color: #4CAF50; }
    .card-cost { border-top-color: #FF9800; }
    .card-inventory { border-top-color: #9C27B0; }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0E1117;
            color: #FFFFFF;
        }
        .financial-card, .brand-card, .compact-metric {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
    }
    
    /* Progress bar animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- Judul Dashboard ---
st.markdown('<h1 class="main-header">💰 FORECAST & INVENTORY CONTROL PRO DASHBOARD</h1>', unsafe_allow_html=True)
st.caption(f"🚀 Inventory Control & Forecast Analytics - Mulyanto | Real-time Insights | Updated: {datetime.now().strftime('%d %B %Y %H:%M')}")

# --- ====================================================== ---
# ---                KONEKSI & LOAD DATA                    ---
# --- ====================================================== ---

@st.cache_resource(show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def init_gsheet_connection():
    """Inisialisasi koneksi ke Google Sheets dengan retry mechanism"""
    try:
        skey = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(skey, scopes=scopes)
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"❌ Koneksi Gagal: {str(e)}")
        return None

def validate_month_format(month_str):
    """Validate and standardize month formats"""
    if pd.isna(month_str):
        return datetime.now()
    
    month_str = str(month_str).strip().upper()
    
    # Mapping bulan
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    
    formats_to_try = ['%b-%Y', '%b-%y', '%B %Y', '%m/%Y', '%Y-%m']
    
    for fmt in formats_to_try:
        try:
            return datetime.strptime(month_str, fmt)
        except:
            continue
    
    # Fallback: cari bulan dalam string
    for month_name, month_num in month_map.items():
        if month_name in month_str:
            # Cari tahun
            year_part = month_str.replace(month_name, '').replace('-', '').replace(' ', '').strip()
            if year_part and year_part.isdigit():
                year = int('20' + year_part) if len(year_part) == 2 else int(year_part)
            else:
                year = datetime.now().year
            
            return datetime(year, month_num, 1)
    
    return datetime.now()

def add_product_info_to_data(df, df_product):
    """Add Product_Name, Brand, SKU_Tier, Prices from Product_Master to any dataframe"""
    if df.empty or df_product.empty or 'SKU_ID' not in df.columns:
        return df
    
    # Get product info from Product_Master (including prices)
    price_cols = ['Floor_Price', 'Net_Order_Price'] if 'Floor_Price' in df_product.columns and 'Net_Order_Price' in df_product.columns else []
    
    product_info_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Status'] + price_cols
    product_info_cols = [col for col in product_info_cols if col in df_product.columns]
    
    product_info = df_product[product_info_cols].copy()
    product_info = product_info.drop_duplicates(subset=['SKU_ID'])
    
    # Remove existing columns if they exist (except SKU_ID)
    cols_to_remove = []
    for col in ['Product_Name', 'Brand', 'SKU_Tier', 'Status', 'Floor_Price', 'Net_Order_Price']:
        if col in df.columns and col != 'SKU_ID':
            cols_to_remove.append(col)
    
    if cols_to_remove:
        df_temp = df.drop(columns=cols_to_remove)
    else:
        df_temp = df.copy()
    
    # Merge with product info
    df_result = pd.merge(df_temp, product_info, on='SKU_ID', how='left')
    return df_result

@st.cache_data(ttl=300, max_entries=3, show_spinner=False)
def load_and_process_data(_client):
    """
    Load semua data termasuk sheet baru: BS_Fullfilment_Cost
    """
    
    gsheet_url = st.secrets["gsheet_url"]  # Ambil dari secrets
    data = {}

    # --- HELPER: Baca Sheet Manual ---
    def safe_read_stock_sheet(sheet_name):
        try:
            ws = _client.open_by_url(gsheet_url).worksheet(sheet_name)
            raw_data = ws.get_all_values()
            if len(raw_data) < 2: return pd.DataFrame()
            headers = [str(h).strip() for h in raw_data[0]]
            df = pd.DataFrame(raw_data[1:], columns=headers)
            df = df.loc[:, df.columns != '']
            return df
        except: return pd.DataFrame()

    try:
        # 1. PRODUCT MASTER
        ws_prod = _client.open_by_url(gsheet_url).worksheet("Product_Master")
        df_product = pd.DataFrame(ws_prod.get_all_records())
        df_product.columns = [col.strip().replace(' ', '_') for col in df_product.columns]
        
        for col in ['Floor_Price', 'Net_Order_Price']:
            if col in df_product.columns:
                df_product[col] = pd.to_numeric(df_product[col], errors='coerce').fillna(0)
        
        if 'Status' not in df_product.columns: df_product['Status'] = 'Active'
        df_product_active = df_product[df_product['Status'].str.upper() == 'ACTIVE'].copy()
        active_skus = df_product_active['SKU_ID'].tolist()
        
        data['product'] = df_product
        data['product_active'] = df_product_active

        # 2. SALES DATA
        ws_sales = _client.open_by_url(gsheet_url).worksheet("Sales")
        df_sales_raw = pd.DataFrame(ws_sales.get_all_records())
        df_sales_raw.columns = [col.strip() for col in df_sales_raw.columns]
        month_cols = [c for c in df_sales_raw.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        if month_cols and 'SKU_ID' in df_sales_raw.columns:
            id_cols = ['SKU_ID']
            for col in ['SKU_Name', 'Product_Name', 'Brand', 'SKU_Tier']:
                if col in df_sales_raw.columns: id_cols.append(col)
            df_sales_long = df_sales_raw.melt(id_vars=id_cols, value_vars=month_cols, var_name='Month_Label', value_name='Sales_Qty')
            df_sales_long['Sales_Qty'] = pd.to_numeric(df_sales_long['Sales_Qty'], errors='coerce').fillna(0)
            df_sales_long['Month'] = df_sales_long['Month_Label'].apply(validate_month_format)
            df_sales_long = df_sales_long[df_sales_long['SKU_ID'].isin(active_skus)]
            df_sales_long = add_product_info_to_data(df_sales_long, df_product)
            data['sales'] = df_sales_long.sort_values('Month')

        # 3. ROFO DATA
        ws_rofo = _client.open_by_url(gsheet_url).worksheet("Rofo")
        df_rofo_raw = pd.DataFrame(ws_rofo.get_all_records())
        df_rofo_raw.columns = [col.strip() for col in df_rofo_raw.columns]
        month_cols_rofo = [c for c in df_rofo_raw.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        if month_cols_rofo:
            id_cols_rofo = ['SKU_ID']
            for col in ['Product_Name', 'Brand']:
                if col in df_rofo_raw.columns: id_cols_rofo.append(col)
            df_rofo_long = df_rofo_raw.melt(id_vars=id_cols_rofo, value_vars=month_cols_rofo, var_name='Month_Label', value_name='Forecast_Qty')
            df_rofo_long['Forecast_Qty'] = pd.to_numeric(df_rofo_long['Forecast_Qty'], errors='coerce').fillna(0)
            df_rofo_long['Month'] = df_rofo_long['Month_Label'].apply(validate_month_format)
            df_rofo_long = df_rofo_long[df_rofo_long['SKU_ID'].isin(active_skus)]
            df_rofo_long = add_product_info_to_data(df_rofo_long, df_product)
            data['forecast'] = df_rofo_long

        # 4. PO DATA
        ws_po = _client.open_by_url(gsheet_url).worksheet("PO")
        df_po_raw = pd.DataFrame(ws_po.get_all_records())
        df_po_raw.columns = [col.strip() for col in df_po_raw.columns]
        month_cols_po = [c for c in df_po_raw.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        if month_cols_po and 'SKU_ID' in df_po_raw.columns:
            df_po_long = df_po_raw.melt(id_vars=['SKU_ID'], value_vars=month_cols_po, var_name='Month_Label', value_name='PO_Qty')
            df_po_long['PO_Qty'] = pd.to_numeric(df_po_long['PO_Qty'], errors='coerce').fillna(0)
            df_po_long['Month'] = df_po_long['Month_Label'].apply(validate_month_format)
            df_po_long = df_po_long[df_po_long['SKU_ID'].isin(active_skus)]
            df_po_long = add_product_info_to_data(df_po_long, df_product)
            data['po'] = df_po_long

        # 5. STOCK DATA
        df_stock_raw = safe_read_stock_sheet("Stock_Onhand")
        if not df_stock_raw.empty:
            col_mapping = {
                'SKU_ID': 'SKU_ID', 'Qty_Available': 'Stock_Qty', 'Product_Code': 'Anchanto_Code',
                'Stock_Category': 'Stock_Category', 'Expiry_Date': 'Expiry_Date', 'Product_Name': 'Product_Name'
            }
            if 'SKU_ID' in df_stock_raw.columns and 'Qty_Available' in df_stock_raw.columns:
                cols_to_use = [c for c in col_mapping.keys() if c in df_stock_raw.columns]
                df_stock = df_stock_raw[cols_to_use].copy()
                df_stock = df_stock.rename(columns=col_mapping)
                df_stock['Stock_Qty'] = pd.to_numeric(df_stock['Stock_Qty'], errors='coerce').fillna(0)
                df_stock['SKU_ID'] = df_stock['SKU_ID'].astype(str).str.strip()
                if 'Floor_Price' in df_product.columns:
                    df_stock = pd.merge(df_stock, df_product[['SKU_ID', 'Floor_Price', 'Net_Order_Price']], on='SKU_ID', how='left')
                data['stock'] = df_stock
            else:
                data['stock'] = pd.DataFrame(columns=['SKU_ID', 'Stock_Qty'])
        else:
            data['stock'] = pd.DataFrame(columns=['SKU_ID', 'Stock_Qty'])

        # 6. FORECAST 2026 ECOMM
        try:
            ws_ecomm = _client.open_by_url(gsheet_url).worksheet("Forecast_2026_Ecomm")
            df_ecomm_raw = pd.DataFrame(ws_ecomm.get_all_records())
            df_ecomm_raw.columns = [col.strip().replace(' ', '_') for col in df_ecomm_raw.columns]
            month_cols_ecomm = [c for c in df_ecomm_raw.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            for col in month_cols_ecomm:
                df_ecomm_raw[col] = pd.to_numeric(df_ecomm_raw[col], errors='coerce').fillna(0)
            data['ecomm_forecast'] = df_ecomm_raw
            data['ecomm_forecast_month_cols'] = month_cols_ecomm
        except:
            data['ecomm_forecast'] = pd.DataFrame()
            data['ecomm_forecast_month_cols'] = []
        
        # 7. FORECAST 2026 RESELLER
        try:
            ws_reseller = _client.open_by_url(gsheet_url).worksheet("Forecast_2026_Reseller")
            df_reseller_raw = pd.DataFrame(ws_reseller.get_all_records())
            df_reseller_raw.columns = [col.strip().replace(' ', '_') for col in df_reseller_raw.columns]
            all_month_cols_res = [c for c in df_reseller_raw.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            for col in all_month_cols_res:
                df_reseller_raw[col] = pd.to_numeric(df_reseller_raw[col], errors='coerce').fillna(0)
            
            forecast_start_date = datetime(2026, 1, 1)
            def is_forecast_month(month_str):
                try:
                    month_str = str(month_str).upper().replace('_', ' ').replace('-', ' ')
                    if ' ' in month_str:
                        month_part, year_part = month_str.split(' ')
                        month_num = datetime.strptime(month_part[:3], '%b').month
                        year_clean = ''.join(filter(str.isdigit, year_part))
                        year = 2000 + int(year_clean) if len(year_clean) == 2 else int(year_clean)
                        return datetime(year, month_num, 1) >= forecast_start_date
                except: return False
                return False
            
            hist_cols = [c for c in all_month_cols_res if not is_forecast_month(c)]
            fcst_cols = [c for c in all_month_cols_res if is_forecast_month(c)]
            data['reseller_forecast'] = df_reseller_raw
            data['reseller_all_month_cols'] = all_month_cols_res
            data['reseller_historical_cols'] = hist_cols
            data['reseller_forecast_cols'] = fcst_cols
        except:
            data['reseller_forecast'] = pd.DataFrame()
            data['reseller_all_month_cols'] = []
            data['reseller_historical_cols'] = []
            data['reseller_forecast_cols'] = []

        # ==============================================================================
        # 8. BS FULLFILMENT COST (NEW SHEET)
        # ==============================================================================
        try:
            ws_bs = _client.open_by_url(gsheet_url).worksheet("BS_Fullfilment_Cost")
            df_bs = pd.DataFrame(ws_bs.get_all_records())
            
            # Cleaning Headers & Data
            # Hapus spasi di nama kolom
            df_bs.columns = [c.strip() for c in df_bs.columns]
            
            # Helper untuk bersihkan angka (hapus koma dan persen)
            def clean_currency(x):
                if isinstance(x, str):
                    return pd.to_numeric(x.replace(',', '').replace('%', ''), errors='coerce')
                return x

            # List kolom angka yang perlu dibersihkan
            numeric_cols = ['Total Order(BS)', 'GMV (Fullfil By BS)', 'GMV Total (MP)', 'Total Cost', 'BSA', '%Cost']
            
            for col in numeric_cols:
                if col in df_bs.columns:
                    df_bs[col] = df_bs[col].apply(clean_currency).fillna(0)
            
            # Convert Percentages (karena 3.14% jadi 3.14, mungkin perlu dibagi 100 utk kalkulasi, tapi utk display biar saja)
            # Kita tandai kolom ini
            
            # Parse Date (Apr-25)
            df_bs['Month_Date'] = pd.to_datetime(df_bs['Month'], format='%b-%y', errors='coerce')
            df_bs = df_bs.sort_values('Month_Date')
            
            data['fulfillment'] = df_bs
            
        except Exception as e:
            st.warning(f"Gagal load BS_Fullfilment_Cost: {e}")
            data['fulfillment'] = pd.DataFrame()

        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}

# --- FUNGSI BARU: LOAD DATA RESELLER LENGKAP ---
@st.cache_data(ttl=300, show_spinner=False)
def load_reseller_complete_data(_client):
    """
    Load SEMUA data reseller: forecast, sales, past rofo, past PO
    """
    # Gunakan url yang sudah ada
    gsheet_url = st.secrets["gsheet_url"]  # Ambil dari secrets
    reseller_data = {}
    
    try:
        # 1. FORECAST 2026 RESELLER
        ws_fcst = _client.open_by_url(gsheet_url).worksheet("Forecast_2026_Reseller")
        df_fcst_raw = pd.DataFrame(ws_fcst.get_all_records())
        df_fcst_raw.columns = [col.strip() for col in df_fcst_raw.columns]
        
        # Identifikasi kolom bulan
        all_month_cols = [c for c in df_fcst_raw.columns if any(m in c.upper() for m in 
                      ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        
        # Pisahkan 2025 (history) vs 2026+ (forecast)
        hist_cols = []
        fcst_cols = []
        
        for col in all_month_cols:
            col_str = str(col).upper()
            if '25' in col_str or '2025' in col_str:
                hist_cols.append(col)
            else:
                fcst_cols.append(col)  # 2026, 2027, dll
        
        # Convert numeric
        for col in all_month_cols:
            df_fcst_raw[col] = pd.to_numeric(df_fcst_raw[col], errors='coerce').fillna(0)
        
        reseller_data['forecast'] = df_fcst_raw
        reseller_data['forecast_month_cols'] = fcst_cols
        reseller_data['historical_month_cols'] = hist_cols
        
        # 2. SALES RESELLER
        try:
            ws_sales = _client.open_by_url(gsheet_url).worksheet("Sales_Reseller")
            df_sales_raw = pd.DataFrame(ws_sales.get_all_records())
            df_sales_raw.columns = [col.strip() for col in df_sales_raw.columns]
            
            # Transform ke long format
            month_cols_sales = [c for c in df_sales_raw.columns if any(m in c.upper() for m in 
                          ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            
            if month_cols_sales and 'SKU_ID' in df_sales_raw.columns:
                id_cols_sales = ['SKU_ID', 'Brand', 'Product_Name', 'SKU_Tier', 'Floor_Price']
                id_cols_sales = [c for c in id_cols_sales if c in df_sales_raw.columns]
                
                df_sales_long = df_sales_raw.melt(
                    id_vars=id_cols_sales,
                    value_vars=month_cols_sales,
                    var_name='Month_Label',
                    value_name='Sales_Qty'
                )
                df_sales_long['Sales_Qty'] = pd.to_numeric(df_sales_long['Sales_Qty'], errors='coerce').fillna(0)
                df_sales_long['Month'] = df_sales_long['Month_Label'].apply(validate_month_format)
                reseller_data['sales'] = df_sales_long
        except Exception as e:
            st.warning(f"⚠️ Sales_Reseller sheet not accessible: {str(e)}")
        
        # 3. PAST ROFO RESELLER
        try:
            ws_rofo = _client.open_by_url(gsheet_url).worksheet("Past_Rofo_Reseller")
            df_rofo_raw = pd.DataFrame(ws_rofo.get_all_records())
            df_rofo_raw.columns = [col.strip() for col in df_rofo_raw.columns]
            
            month_cols_rofo = [c for c in df_rofo_raw.columns if any(m in c.upper() for m in 
                          ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            
            if month_cols_rofo and 'SKU_ID' in df_rofo_raw.columns:
                id_cols_rofo = ['SKU_ID', 'Brand', 'Product_Name', 'SKU_Tier', 'Floor_Price']
                id_cols_rofo = [c for c in id_cols_rofo if c in df_rofo_raw.columns]
                
                df_rofo_long = df_rofo_raw.melt(
                    id_vars=id_cols_rofo,
                    value_vars=month_cols_rofo,
                    var_name='Month_Label',
                    value_name='Forecast_Qty'
                )
                df_rofo_long['Forecast_Qty'] = pd.to_numeric(df_rofo_long['Forecast_Qty'], errors='coerce').fillna(0)
                df_rofo_long['Month'] = df_rofo_long['Month_Label'].apply(validate_month_format)
                reseller_data['past_rofo'] = df_rofo_long
        except Exception as e:
            st.warning(f"⚠️ Past_Rofo_Reseller sheet not accessible: {str(e)}")
        
        # 4. PAST PO RESELLER
        try:
            ws_po = _client.open_by_url(gsheet_url).worksheet("Past_PO_Reseller")
            df_po_raw = pd.DataFrame(ws_po.get_all_records())
            df_po_raw.columns = [col.strip() for col in df_po_raw.columns]
            
            month_cols_po = [c for c in df_po_raw.columns if any(m in c.upper() for m in 
                          ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            
            if month_cols_po and 'SKU_ID' in df_po_raw.columns:
                id_cols_po = ['SKU_ID', 'Brand', 'Product_Name', 'SKU_Tier', 'Floor_Price']
                id_cols_po = [c for c in id_cols_po if c in df_po_raw.columns]
                
                df_po_long = df_po_raw.melt(
                    id_vars=id_cols_po,
                    value_vars=month_cols_po,
                    var_name='Month_Label',
                    value_name='PO_Qty'
                )
                df_po_long['PO_Qty'] = pd.to_numeric(df_po_long['PO_Qty'], errors='coerce').fillna(0)
                df_po_long['Month'] = df_po_long['Month_Label'].apply(validate_month_format)
                reseller_data['past_po'] = df_po_long
        except Exception as e:
            st.warning(f"⚠️ Past_PO_Reseller sheet not accessible: {str(e)}")
        
        return reseller_data
        
    except Exception as e:
        st.error(f"❌ Error loading reseller data: {str(e)}")
        return {}

# --- ====================================================== ---
# ---                FINANCIAL FUNCTIONS                    ---
# --- ====================================================== ---

@st.cache_data(ttl=300)
def calculate_financial_metrics_all(df_sales, df_product):
    """Calculate all financial metrics from sales data"""
    
    if df_sales.empty or df_product.empty:
        return pd.DataFrame()
    
    try:
        # Check if price columns exist
        required_price_cols = ['Floor_Price', 'Net_Order_Price']
        price_cols_exist = all(col in df_product.columns for col in required_price_cols)
        
        if not price_cols_exist:
            st.warning("⚠️ Price columns missing in Product Master")
            return pd.DataFrame()
        
        # Ensure sales data has product info with prices
        if 'Floor_Price' not in df_sales.columns or 'Net_Order_Price' not in df_sales.columns:
            df_sales = add_product_info_to_data(df_sales, df_product)
        
        # Fill missing prices
        df_sales['Floor_Price'] = df_sales['Floor_Price'].fillna(0)
        df_sales['Net_Order_Price'] = df_sales['Net_Order_Price'].fillna(0)
        
        # Calculate financial metrics
        df_sales['Revenue'] = df_sales['Sales_Qty'] * df_sales['Floor_Price']
        df_sales['Cost'] = df_sales['Sales_Qty'] * df_sales['Net_Order_Price']
        df_sales['Gross_Margin'] = df_sales['Revenue'] - df_sales['Cost']
        df_sales['Margin_Percentage'] = np.where(
            df_sales['Revenue'] > 0,
            (df_sales['Gross_Margin'] / df_sales['Revenue'] * 100),
            0
        )
        
        # Add additional metrics
        df_sales['Avg_Selling_Price'] = np.where(
            df_sales['Sales_Qty'] > 0,
            df_sales['Revenue'] / df_sales['Sales_Qty'],
            0
        )
        
        return df_sales
        
    except Exception as e:
        st.error(f"Financial metrics calculation error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def calculate_inventory_financial(df_stock, df_product):
    """Calculate inventory financial value"""
    
    if df_stock.empty or df_product.empty:
        return pd.DataFrame()
    
    try:
        # Check price columns
        if 'Floor_Price' not in df_product.columns or 'Net_Order_Price' not in df_product.columns:
            return pd.DataFrame()
        
        # Ensure stock data has prices
        if 'Floor_Price' not in df_stock.columns or 'Net_Order_Price' not in df_stock.columns:
            df_stock = add_product_info_to_data(df_stock, df_product)
        
        # Fill missing prices
        df_stock['Floor_Price'] = df_stock['Floor_Price'].fillna(0)
        df_stock['Net_Order_Price'] = df_stock['Net_Order_Price'].fillna(0)
        
        # Calculate inventory values
        df_stock['Value_at_Cost'] = df_stock['Stock_Qty'] * df_stock['Net_Order_Price']
        df_stock['Value_at_Retail'] = df_stock['Stock_Qty'] * df_stock['Floor_Price']
        df_stock['Potential_Margin'] = df_stock['Value_at_Retail'] - df_stock['Value_at_Cost']
        df_stock['Margin_Percentage'] = np.where(
            df_stock['Value_at_Retail'] > 0,
            (df_stock['Potential_Margin'] / df_stock['Value_at_Retail'] * 100),
            0
        )
        
        return df_stock
        
    except Exception as e:
        st.error(f"Inventory financial calculation error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def calculate_seasonality(df_financial):
    """Calculate seasonal patterns from financial data"""
    
    if df_financial.empty:
        return pd.DataFrame()
    
    try:
        # Add month and year columns
        df_financial['Year'] = df_financial['Month'].dt.year
        df_financial['Month_Num'] = df_financial['Month'].dt.month
        df_financial['Month_Name'] = df_financial['Month'].dt.strftime('%b')
        
        # Group by month across years
        seasonal_pattern = df_financial.groupby(['Month_Num', 'Month_Name']).agg({
            'Revenue': 'mean',
            'Gross_Margin': 'mean',
            'Sales_Qty': 'mean'
        }).reset_index()
        
        # Calculate seasonal indices
        overall_avg_revenue = seasonal_pattern['Revenue'].mean()
        seasonal_pattern['Seasonal_Index_Revenue'] = seasonal_pattern['Revenue'] / overall_avg_revenue
        
        overall_avg_margin = seasonal_pattern['Gross_Margin'].mean()
        seasonal_pattern['Seasonal_Index_Margin'] = seasonal_pattern['Gross_Margin'] / overall_avg_margin
        
        # Classify seasons
        conditions = [
            seasonal_pattern['Seasonal_Index_Revenue'] >= 1.2,
            (seasonal_pattern['Seasonal_Index_Revenue'] >= 0.9) & (seasonal_pattern['Seasonal_Index_Revenue'] < 1.2),
            seasonal_pattern['Seasonal_Index_Revenue'] < 0.9
        ]
        choices = ['Peak Season', 'Normal Season', 'Low Season']
        
        seasonal_pattern['Season_Type'] = np.select(conditions, choices, default='Normal Season')
        
        return seasonal_pattern.sort_values('Month_Num')
        
    except Exception as e:
        st.error(f"Seasonality calculation error: {str(e)}")
        return pd.DataFrame()

def calculate_eoq(demand, order_cost, holding_cost_per_unit):
    """Calculate Economic Order Quantity"""
    if demand <= 0 or order_cost <= 0 or holding_cost_per_unit <= 0:
        return 0
    
    eoq = math.sqrt((2 * demand * order_cost) / holding_cost_per_unit)
    return round(eoq)

def calculate_forecast_bias(df_forecast, df_po):
    """Calculate forecast bias (systematic over/under forecasting)"""
    
    if df_forecast.empty or df_po.empty:
        return {}
    
    try:
        # Get common months
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        common_months = sorted(set(forecast_months) & set(po_months))
        
        if not common_months:
            return {}
        
        bias_results = []
        
        for month in common_months:
            df_f_month = df_forecast[df_forecast['Month'] == month]
            df_p_month = df_po[df_po['Month'] == month]
            
            # Merge forecast and PO
            df_merged = pd.merge(
                df_f_month[['SKU_ID', 'Forecast_Qty']],
                df_p_month[['SKU_ID', 'PO_Qty']],
                on='SKU_ID',
                how='inner'
            )
            
            # Calculate bias
            df_merged['Bias'] = df_merged['PO_Qty'] - df_merged['Forecast_Qty']
            df_merged['Bias_Percentage'] = np.where(
                df_merged['Forecast_Qty'] > 0,
                (df_merged['Bias'] / df_merged['Forecast_Qty'] * 100),
                0
            )
            
            avg_bias = df_merged['Bias'].mean()
            avg_bias_pct = df_merged['Bias_Percentage'].mean()
            
            bias_results.append({
                'Month': month,
                'Avg_Bias': avg_bias,
                'Avg_Bias_Percentage': avg_bias_pct,
                'Over_Forecast_SKUs': len(df_merged[df_merged['Bias'] > 0]),
                'Under_Forecast_SKUs': len(df_merged[df_merged['Bias'] < 0])
            })
        
        return pd.DataFrame(bias_results)
        
    except Exception as e:
        st.error(f"Forecast bias calculation error: {str(e)}")
        return pd.DataFrame()

# --- ====================================================== ---
# ---                ANALYTICS FUNCTIONS                    ---
# --- ====================================================== ---

def calculate_monthly_performance(df_forecast, df_po, df_product):
    """Calculate performance using Hit Rate for Accuracy and WMAPE for Error Rate"""
    
    monthly_performance = {}
    
    if df_forecast.empty or df_po.empty:
        return monthly_performance
    
    try:
        # ADD PRODUCT INFO jika belum ada
        df_forecast = add_product_info_to_data(df_forecast, df_product)
        df_po = add_product_info_to_data(df_po, df_product)
        
        # Get unique months from both datasets
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        all_months = sorted(set(list(forecast_months) + list(po_months)))
        
        for month in all_months:
            # Get data for this month - FILTER HANYA Forecast_Qty > 0 (Abaikan Unplanned)
            df_forecast_month = df_forecast[
                (df_forecast['Month'] == month) & 
                (df_forecast['Forecast_Qty'] > 0)
            ].copy()
            
            df_po_month = df_po[df_po['Month'] == month].copy()
            
            if df_forecast_month.empty or df_po_month.empty:
                continue
            
            # Merge forecast and PO for this month
            df_merged = pd.merge(
                df_forecast_month, df_po_month,
                on=['SKU_ID'], how='inner', suffixes=('_forecast', '_po')
            )
            
            if not df_merged.empty:
                # Add product info (jika belum ada dari merge)
                if 'Product_Name' not in df_merged.columns or 'Brand' not in df_merged.columns:
                    df_merged = add_product_info_to_data(df_merged, df_product)
                
                # 1. PERHITUNGAN UNTUK HIT RATE (SKU LEVEL)
                df_merged['PO_Rofo_Ratio'] = (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100
                
                # Categorize Status
                conditions = [
                    df_merged['PO_Rofo_Ratio'] < 80,
                    (df_merged['PO_Rofo_Ratio'] >= 80) & (df_merged['PO_Rofo_Ratio'] <= 120),
                    df_merged['PO_Rofo_Ratio'] > 120
                ]
                choices = ['Under', 'Accurate', 'Over']
                df_merged['Accuracy_Status'] = np.select(conditions, choices, default='Unknown')
                
                status_counts = df_merged['Accuracy_Status'].value_counts().to_dict()
                total_records = len(df_merged)
                status_percentages = {k: (v/total_records*100) for k, v in status_counts.items()}
                
                # >>> ACCURACY = HIT RATE (SKU yang Accurate / Total SKU) <<<
                accurate_count = status_counts.get('Accurate', 0)
                monthly_accuracy = (accurate_count / total_records * 100) if total_records > 0 else 0
                
                # 2. PERHITUNGAN UNTUK MAPE (MENGGUNAKAN WMAPE AGAR AMAN)
                total_rofo = df_merged['Forecast_Qty'].sum()
                total_abs_error = abs(df_merged['PO_Qty'] - df_merged['Forecast_Qty']).sum()
                
                # >>> WMAPE = Total Selisih Absolut / Total Rofo <<<
                if total_rofo > 0:
                    wmape = (total_abs_error / total_rofo) * 100
                else:
                    wmape = 0
                
                # Store results
                monthly_performance[month] = {
                    'accuracy': monthly_accuracy, # Hit Rate (Sesuai strategi Bapak)
                    'mape': wmape,                # WMAPE (Aman dari outlier)
                    'status_counts': status_counts,
                    'status_percentages': status_percentages,
                    'total_records': total_records,
                    'data': df_merged,
                    'under_skus': df_merged[df_merged['Accuracy_Status'] == 'Under'].copy(),
                    'over_skus': df_merged[df_merged['Accuracy_Status'] == 'Over'].copy(),
                    'accurate_skus': df_merged[df_merged['Accuracy_Status'] == 'Accurate'].copy()
                }
        
        return monthly_performance
        
    except Exception as e:
        st.error(f"Monthly performance calculation error: {str(e)}")
        return monthly_performance

def get_last_3_months_performance(monthly_performance):
    """Get performance for last 3 months"""
    
    if not monthly_performance:
        return {}
    
    # Get last 3 months
    sorted_months = sorted(monthly_performance.keys())
    if len(sorted_months) >= 3:
        last_3_months = sorted_months[-3:]
    else:
        last_3_months = sorted_months
    
    last_3_data = {}
    for month in last_3_months:
        last_3_data[month] = monthly_performance[month]
    
    return last_3_data

@st.cache_data(ttl=300)
def calculate_inventory_metrics_with_3month_avg(df_stock, df_sales, df_product):
    """Calculate inventory metrics using 3-month average sales (FIXED: AGGREGATE STOCK FIRST)"""
    
    metrics = {}
    
    if df_stock.empty:
        return metrics
    
    try:
        # --- FIX UTAMA: Agregasi Stok dari Level Batch ke Level SKU ---
        # Kita jumlahkan dulu Stock_Qty berdasarkan SKU_ID agar 1 SKU = 1 Baris
        df_stock_agg = df_stock.groupby('SKU_ID').agg({
            'Stock_Qty': 'sum'
        }).reset_index()
        
        # ADD PRODUCT INFO ke data yang sudah di-agregasi
        df_stock_agg = add_product_info_to_data(df_stock_agg, df_product)
        
        # Siapkan Sales Data
        df_sales = add_product_info_to_data(df_sales, df_product)
        
        # Get last 3 months sales data
        if not df_sales.empty:
            sales_months = sorted(df_sales['Month'].unique())
            if len(sales_months) >= 3:
                last_3_sales_months = sales_months[-3:]
                df_sales_last_3 = df_sales[df_sales['Month'].isin(last_3_sales_months)].copy()
            else:
                df_sales_last_3 = df_sales.copy()
        
        # Calculate average monthly sales per SKU
        if not df_sales.empty and not df_sales_last_3.empty:
            avg_monthly_sales = df_sales_last_3.groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
            avg_monthly_sales.columns = ['SKU_ID', 'Avg_Monthly_Sales_3M']
        else:
            avg_monthly_sales = pd.DataFrame(columns=['SKU_ID', 'Avg_Monthly_Sales_3M'])
        
        # Merge Stock Aggregated dengan Product Info (redundant check but safe)
        df_inventory = pd.merge(
            df_stock_agg,
            df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']],
            on='SKU_ID',
            how='left',
            suffixes=('', '_master')
        )
        
        # Bersihkan kolom duplikat jika ada setelah merge
        df_inventory = df_inventory.loc[:,~df_inventory.columns.duplicated()]
        
        # Merge dengan Average Sales
        df_inventory = pd.merge(df_inventory, avg_monthly_sales, on='SKU_ID', how='left')
        df_inventory['Avg_Monthly_Sales_3M'] = df_inventory['Avg_Monthly_Sales_3M'].fillna(0)
        
        # Calculate cover months
        df_inventory['Cover_Months'] = np.where(
            df_inventory['Avg_Monthly_Sales_3M'] > 0,
            df_inventory['Stock_Qty'] / df_inventory['Avg_Monthly_Sales_3M'],
            999  # For SKUs with no sales
        )
        
        # Categorize inventory status
        conditions = [
            df_inventory['Cover_Months'] < 0.8,
            (df_inventory['Cover_Months'] >= 0.8) & (df_inventory['Cover_Months'] <= 2),
            df_inventory['Cover_Months'] > 1.5
        ]
        choices = ['Need Replenishment', 'Ideal/Healthy', 'High Stock']
        df_inventory['Inventory_Status'] = np.select(conditions, choices, default='Unknown')
        
        # Get high/low stock items
        high_stock_df = df_inventory[df_inventory['Inventory_Status'] == 'High Stock'].copy().sort_values('Cover_Months', ascending=False)
        low_stock_df = df_inventory[df_inventory['Inventory_Status'] == 'Need Replenishment'].copy().sort_values('Cover_Months', ascending=True)
        
        # Tier analysis
        if 'SKU_Tier' in df_inventory.columns:
            tier_analysis = df_inventory.groupby('SKU_Tier').agg({
                'SKU_ID': 'count',
                'Stock_Qty': 'sum',
                'Avg_Monthly_Sales_3M': 'sum',
                'Cover_Months': 'mean'
            }).reset_index()
            tier_analysis.columns = ['Tier', 'SKU_Count', 'Total_Stock', 'Total_Sales_3M_Avg', 'Avg_Cover_Months']
            tier_analysis['Turnover'] = tier_analysis['Total_Sales_3M_Avg'] / tier_analysis['Total_Stock']
            metrics['tier_analysis'] = tier_analysis
        
        metrics['inventory_df'] = df_inventory
        metrics['high_stock'] = high_stock_df
        metrics['low_stock'] = low_stock_df
        metrics['total_stock'] = df_inventory['Stock_Qty'].sum()
        metrics['total_skus'] = len(df_inventory)
        metrics['avg_cover'] = df_inventory[df_inventory['Cover_Months'] < 999]['Cover_Months'].mean()
        
        metrics['inventory_value_score'] = (len(df_inventory[df_inventory['Inventory_Status'] == 'Ideal/Healthy']) / 
                                            len(df_inventory) * 100) if len(df_inventory) > 0 else 0
        
        return metrics
        
    except Exception as e:
        st.error(f"Inventory metrics error: {str(e)}")
        return metrics

def calculate_sales_vs_forecast_po(df_sales, df_forecast, df_po, df_product):
    """Calculate sales vs forecast and PO comparison - HANYA ACTIVE SKUS"""
    
    results = {}
    
    if df_sales.empty or df_forecast.empty:
        return results
    
    try:
        # ADD PRODUCT INFO jika belum ada
        df_sales = add_product_info_to_data(df_sales, df_product)
        df_forecast = add_product_info_to_data(df_forecast, df_product)
        df_po = add_product_info_to_data(df_po, df_product)
        
        # FILTER HANYA ACTIVE SKUS
        if 'Status' in df_product.columns:
            active_skus = df_product[df_product['Status'].str.upper() == 'ACTIVE']['SKU_ID'].tolist()
            
            # Filter semua dataset untuk hanya active SKUs
            df_sales = df_sales[df_sales['SKU_ID'].isin(active_skus)]
            df_forecast = df_forecast[df_forecast['SKU_ID'].isin(active_skus)]
            if not df_po.empty:
                df_po = df_po[df_po['SKU_ID'].isin(active_skus)]
        
        # Get last 3 months for comparison
        sales_months = sorted(df_sales['Month'].unique())
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        
        # Find common months
        common_months = sorted(set(sales_months) & set(forecast_months) & set(po_months))
        
        if not common_months:
            return results
        
        # Use last common month
        last_month = common_months[-1]
        
        # Get data for last month
        df_sales_month = df_sales[df_sales['Month'] == last_month].copy()
        df_forecast_month = df_forecast[df_forecast['Month'] == last_month].copy()
        df_po_month = df_po[df_po['Month'] == last_month].copy()
        
        # Filter hanya SKU dengan Forecast_Qty > 0
        df_forecast_month = df_forecast_month[df_forecast_month['Forecast_Qty'] > 0]
        
        # Merge all data
        df_merged = pd.merge(
            df_sales_month[['SKU_ID', 'Sales_Qty']],
            df_forecast_month[['SKU_ID', 'Forecast_Qty']],
            on='SKU_ID',
            how='inner'
        )
        
        df_merged = pd.merge(
            df_merged,
            df_po_month[['SKU_ID', 'PO_Qty']],
            on='SKU_ID',
            how='left'
        )
        
        # Add product info
        df_merged = add_product_info_to_data(df_merged, df_product)
        
        # Filter out SKU dengan PO_Qty = 0 (tidak ada PO) jika mau
        # df_merged = df_merged[df_merged['PO_Qty'] > 0]
        
        # Calculate ratios
        df_merged['Sales_vs_Forecast_Ratio'] = np.where(
            df_merged['Forecast_Qty'] > 0,
            (df_merged['Sales_Qty'] / df_merged['Forecast_Qty']) * 100,
            0
        )
        
        df_merged['Sales_vs_PO_Ratio'] = np.where(
            df_merged['PO_Qty'] > 0,
            (df_merged['Sales_Qty'] / df_merged['PO_Qty']) * 100,
            0
        )
        
        # Calculate deviations
        df_merged['Forecast_Deviation'] = abs(df_merged['Sales_vs_Forecast_Ratio'] - 100)
        df_merged['PO_Deviation'] = abs(df_merged['Sales_vs_PO_Ratio'] - 100)
        
        # Identify SKUs with high deviation (> 30%) - HANYA ACTIVE SKUS
        high_deviation_skus = df_merged[
            (df_merged['Forecast_Deviation'] > 30) | 
            (df_merged['PO_Deviation'] > 30)
        ].copy()
        
        high_deviation_skus = high_deviation_skus.sort_values('Forecast_Deviation', ascending=False)
        
        # Calculate overall metrics
        avg_forecast_deviation = df_merged['Forecast_Deviation'].mean()
        avg_po_deviation = df_merged['PO_Deviation'].mean()
        
        results = {
            'last_month': last_month,
            'comparison_data': df_merged,
            'high_deviation_skus': high_deviation_skus,
            'avg_forecast_deviation': avg_forecast_deviation,
            'avg_po_deviation': avg_po_deviation,
            'total_skus_compared': len(df_merged),
            'active_skus_only': True
        }
        
        return results
        
    except Exception as e:
        st.error(f"Sales vs forecast calculation error: {str(e)}")
        return results

def calculate_brand_performance(df_forecast, df_po, df_product):
    """Calculate forecast accuracy performance by brand"""
    
    if df_forecast.empty or df_po.empty or df_product.empty:
        return pd.DataFrame()
    
    try:
        # ADD PRODUCT INFO jika belum ada
        df_forecast = add_product_info_to_data(df_forecast, df_product)
        df_po = add_product_info_to_data(df_po, df_product)
        
        # Get last month data
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        common_months = sorted(set(forecast_months) & set(po_months))
        
        if not common_months:
            return pd.DataFrame()
        
        last_month = common_months[-1]
        
        # Get data for last month
        df_forecast_month = df_forecast[df_forecast['Month'] == last_month].copy()
        df_po_month = df_po[df_po['Month'] == last_month].copy()
        
        # Merge forecast and PO
        df_merged = pd.merge(
            df_forecast_month,
            df_po_month,
            on=['SKU_ID'],
            how='inner'
        )
        
        # Add brand info jika belum ada
        if 'Brand' not in df_merged.columns:
            df_merged = add_product_info_to_data(df_merged, df_product)
        
        if 'Brand' not in df_merged.columns:
            return pd.DataFrame()
        
        # Calculate ratio and accuracy
        df_merged['PO_Rofo_Ratio'] = np.where(
            df_merged['Forecast_Qty'] > 0,
            (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100,
            0
        )
        
        # Categorize
        conditions = [
            df_merged['PO_Rofo_Ratio'] < 80,
            (df_merged['PO_Rofo_Ratio'] >= 80) & (df_merged['PO_Rofo_Ratio'] <= 120),
            df_merged['PO_Rofo_Ratio'] > 120
        ]
        choices = ['Under', 'Accurate', 'Over']
        df_merged['Accuracy_Status'] = np.select(conditions, choices, default='Unknown')
        
        # Calculate brand performance
        brand_performance = df_merged.groupby('Brand').agg({
            'SKU_ID': 'count',
            'Forecast_Qty': 'sum',
            'PO_Qty': 'sum',
            'PO_Rofo_Ratio': lambda x: 100 - abs(x - 100).mean()  # Accuracy
        }).reset_index()
        
        brand_performance.columns = ['Brand', 'SKU_Count', 'Total_Forecast', 'Total_PO', 'Accuracy']
        
        # Calculate additional metrics
        brand_performance['PO_vs_Forecast_Ratio'] = (brand_performance['Total_PO'] / brand_performance['Total_Forecast'] * 100)
        brand_performance['Qty_Difference'] = brand_performance['Total_PO'] - brand_performance['Total_Forecast']
        
        # Get status counts
        status_counts = df_merged.groupby(['Brand', 'Accuracy_Status']).size().unstack(fill_value=0).reset_index()
        
        # Merge with performance data
        brand_performance = pd.merge(brand_performance, status_counts, on='Brand', how='left')
        
        # Fill NaN with 0 for status columns
        for status in ['Under', 'Accurate', 'Over']:
            if status not in brand_performance.columns:
                brand_performance[status] = 0
        
        # Sort by accuracy
        brand_performance = brand_performance.sort_values('Accuracy', ascending=False)
        
        return brand_performance
        
    except Exception as e:
        st.error(f"Brand performance calculation error: {str(e)}")
        return pd.DataFrame()

def identify_profitability_segments(df_financial):
    """Segment SKUs by profitability"""
    
    if df_financial.empty:
        return pd.DataFrame()
    
    try:
        sku_profitability = df_financial.groupby(['SKU_ID', 'Product_Name', 'Brand']).agg({
            'Revenue': 'sum',
            'Gross_Margin': 'sum',
            'Sales_Qty': 'sum'
        }).reset_index()
        
        # Calculate metrics
        sku_profitability['Avg_Margin_Per_SKU'] = sku_profitability['Gross_Margin'] / sku_profitability['Sales_Qty']
        sku_profitability['Margin_Percentage'] = np.where(
            sku_profitability['Revenue'] > 0,
            (sku_profitability['Gross_Margin'] / sku_profitability['Revenue'] * 100),
            0
        )
        
        # Segment by margin percentage
        conditions = [
            (sku_profitability['Margin_Percentage'] >= 40),
            (sku_profitability['Margin_Percentage'] >= 20) & (sku_profitability['Margin_Percentage'] < 40),
            (sku_profitability['Margin_Percentage'] < 20) & (sku_profitability['Margin_Percentage'] > 0),
            (sku_profitability['Margin_Percentage'] <= 0)
        ]
        choices = ['High Margin (>40%)', 'Medium Margin (20-40%)', 'Low Margin (<20%)', 'Negative Margin']
        
        sku_profitability['Margin_Segment'] = np.select(conditions, choices, default='Unknown')
        
        return sku_profitability.sort_values('Gross_Margin', ascending=False)
        
    except Exception as e:
        st.error(f"Profitability segmentation error: {str(e)}")
        return pd.DataFrame()

def validate_data_quality(df, df_name):
    """Comprehensive data quality validation"""
    
    checks = {}
    
    if df.empty:
        checks['Empty Dataset'] = '❌ Dataset kosong'
        return checks
    
    # Basic checks
    checks['Total Rows'] = f"📊 {len(df):,} rows"
    checks['Total Columns'] = f"📋 {len(df.columns)} columns"
    
    # Missing values
    missing_values = df.isnull().sum().sum()
    missing_pct = (missing_values / (len(df) * len(df.columns)) * 100)
    checks['Missing Values'] = f"⚠️ {missing_values:,} ({missing_pct:.1f}%)" if missing_values > 0 else f"✅ {missing_values:,}"
    
    # Duplicates
    duplicates = df.duplicated().sum()
    checks['Duplicate Rows'] = f"⚠️ {duplicates:,}" if duplicates > 0 else f"✅ {duplicates:,}"
    
    # Zero values (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        zero_values = (df[numeric_cols] == 0).sum().sum()
        zero_pct = (zero_values / (len(df) * len(numeric_cols)) * 100)
        checks['Zero Values'] = f"📉 {zero_values:,} ({zero_pct:.1f}%)"
    
    # Negative values
    if len(numeric_cols) > 0:
        negative_values = (df[numeric_cols] < 0).sum().sum()
        if negative_values > 0:
            checks['Negative Values'] = f"❌ {negative_values:,}"
    
    # Date range (if Month column exists)
    if 'Month' in df.columns:
        try:
            min_date = df['Month'].min()
            max_date = df['Month'].max()
            checks['Date Range'] = f"📅 {min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}"
        except:
            pass
    
    return checks

# =============================================================================
# AI FORECASTING ENGINE - HELPER FUNCTIONS
# =============================================================================

def ai_prepare_ts(df_sales, entity_type, entity_value):
    """Prepare monthly time series for a SKU or Brand"""
    if df_sales.empty:
        return pd.Series(dtype=float)
    
    if entity_type == 'SKU':
        df_filt = df_sales[df_sales['SKU_ID'] == entity_value]
    else:
        df_filt = df_sales[df_sales['Brand'] == entity_value]
    
    ts = df_filt.groupby('Month')['Sales_Qty'].sum().sort_index()
    
    if len(ts) >= 2:
        full_idx = pd.date_range(ts.index.min(), ts.index.max(), freq='MS')
        ts = ts.reindex(full_idx, fill_value=0)
    
    return ts


def ai_forecast_ma(ts, n=12, window=3, weighted=False):
    """Moving Average Forecast (Simple or Weighted) with trend adjustment"""
    vals = ts.values.astype(float)
    w = min(window, len(vals))
    if w == 0:
        return None, None, None

    if weighted:
        wts = np.arange(1, w + 1, dtype=float)
        wts /= wts.sum()
        base = float(np.dot(vals[-w:], wts))
    else:
        base = float(np.mean(vals[-w:]))

    # Trend from last 6 months (or available)
    n_trend = min(6, len(vals))
    trend = float(np.polyfit(range(n_trend), vals[-n_trend:], 1)[0]) if n_trend >= 2 else 0.0

    preds = [max(0.0, base + trend * (i + 1)) for i in range(n)]
    hist_std = float(np.std(vals)) if len(vals) > 1 else base * 0.2
    ci_lo = np.maximum(0, np.array(preds) - 1.5 * hist_std)
    ci_hi = np.array(preds) + 1.5 * hist_std

    future = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=n, freq='MS')
    return pd.Series(preds, index=future), ci_lo, ci_hi


def ai_forecast_holt_winters(ts, n=12):
    """Holt-Winters Exponential Smoothing"""
    if not STATSMODELS_AVAILABLE:
        return None, None, None

    vals = ts.values.astype(float)
    if np.all(vals == 0) or len(vals) < 3:
        future = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=n, freq='MS')
        return pd.Series([0.0] * n, index=future), np.zeros(n), np.zeros(n)

    try:
        if len(vals) >= 24:
            model = ExponentialSmoothing(vals, trend='add', seasonal='add',
                                         seasonal_periods=12, damped_trend=True)
        elif len(vals) >= 12:
            model = ExponentialSmoothing(vals, trend='add', seasonal='add',
                                         seasonal_periods=12)
        elif len(vals) >= 4:
            model = ExponentialSmoothing(vals, trend='add', seasonal=None)
        else:
            model = ExponentialSmoothing(vals, trend=None, seasonal=None)

        fit = model.fit(optimized=True, remove_bias=True)
        preds = np.maximum(0, fit.forecast(n))

        resid_std = float(np.std(fit.resid)) if len(fit.resid) > 0 else float(np.std(vals)) * 0.3
        ci_lo = np.maximum(0, preds - 1.96 * resid_std)
        ci_hi = preds + 1.96 * resid_std

        future = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=n, freq='MS')
        return pd.Series(preds, index=future), ci_lo, ci_hi
    except Exception:
        return None, None, None


def ai_forecast_linear_seasonal(ts, n=12):
    """Linear Trend + Monthly Seasonal Factors"""
    vals = ts.values.astype(float)
    months_arr = ts.index.month
    x = np.arange(len(vals))

    if len(vals) < 2:
        return None, None, None

    slope, intercept = np.polyfit(x, vals, 1)
    trend_vals = slope * x + intercept
    detrended = vals - trend_vals

    # Seasonal factors per month
    sf = {}
    for m in range(1, 13):
        idx = np.where(months_arr == m)[0]
        sf[m] = float(np.mean(detrended[idx])) if len(idx) > 0 else 0.0

    future = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=n, freq='MS')
    preds = []
    for i, d in enumerate(future):
        val = max(0.0, slope * (len(vals) + i) + intercept + sf.get(d.month, 0.0))
        preds.append(val)

    residuals = vals - (trend_vals + np.array([sf.get(m, 0.0) for m in months_arr]))
    resid_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0
    ci_lo = np.maximum(0, np.array(preds) - 1.96 * resid_std)
    ci_hi = np.array(preds) + 1.96 * resid_std

    return pd.Series(preds, index=future), ci_lo, ci_hi


def ai_forecast_prophet(ts, n=12):
    """Meta Prophet Forecasting"""
    if not PROPHET_AVAILABLE:
        return None, None, None

    df_p = pd.DataFrame({'ds': ts.index, 'y': ts.values.astype(float)})
    df_p = df_p[df_p['y'] >= 0].copy()

    if len(df_p) < 4:
        return None, None, None

    try:
        mode = 'multiplicative' if df_p['y'].mean() > 5 and df_p['y'].min() > 0 else 'additive'
        m = Prophet(
            seasonality_mode=mode,
            yearly_seasonality=(len(ts) >= 12),
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1
        )
        m.fit(df_p)

        future = m.make_future_dataframe(periods=n, freq='MS')
        fc = m.predict(future)
        fc_future = fc[fc['ds'] > ts.index[-1]].head(n)

        preds = np.maximum(0, fc_future['yhat'].values)
        ci_lo = np.maximum(0, fc_future['yhat_lower'].values)
        ci_hi = fc_future['yhat_upper'].values

        future_idx = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=n, freq='MS')
        return pd.Series(preds, index=future_idx), ci_lo, ci_hi
    except Exception:
        return None, None, None

def ai_forecast_rolling_baseline_mtm(ts, n=12, baseline_window=3, min_history_for_mtm=6):
    """
    Rolling Baseline × MtM Seasonal Method
    =========================================
    Metode S&OP Praktisi:
    - Baseline  = rata-rata actual N bulan terakhir (default 3)
    - Seasonal  = rata-rata MtM growth ratio dari semua history tersedia
    - Flag      = deteksi anomali pada baseline (spike/drop > 2x avg)
    - Output    = 12 bulan forecast + anomaly warnings
    """
    vals = ts.values.astype(float)
    months = ts.index.month  # array of month numbers

    warnings_list = []

    # ----------------------------------------------------------------
    # STEP 1: BUILD BASELINE
    # ----------------------------------------------------------------
    w = min(baseline_window, len(vals))
    baseline_vals = vals[-w:]
    baseline = float(np.mean(baseline_vals))

    # Anomaly check: apakah salah satu bulan di baseline > 2x overall avg?
    overall_avg = float(np.mean(vals)) if len(vals) > 0 else baseline
    if overall_avg > 0:
        for i, v in enumerate(baseline_vals):
            month_label = ts.index[-(w - i)].strftime('%b-%Y')
            if v > 2.0 * overall_avg:
                warnings_list.append({
                    'type': 'spike',
                    'month': month_label,
                    'value': v,
                    'avg': overall_avg,
                    'ratio': v / overall_avg
                })
            elif v < 0.3 * overall_avg and overall_avg > 5:
                warnings_list.append({
                    'type': 'drop',
                    'month': month_label,
                    'value': v,
                    'avg': overall_avg,
                    'ratio': v / overall_avg
                })

    # ----------------------------------------------------------------
    # STEP 2: BUILD MtM GROWTH RATIO TABLE
    # Per transition: Jan→Feb, Feb→Mar, ..., Dec→Jan
    # Ambil rata-rata dari semua pasangan bulan di history
    # ----------------------------------------------------------------
    # Dict: key = (from_month, to_month), value = list of ratios
    mtm_ratios = {}
    for i in range(1, len(vals)):
        from_m = ts.index[i - 1].month
        to_m = ts.index[i].month
        key = (from_m, to_m)
        if vals[i - 1] > 0:
            ratio = vals[i] / vals[i - 1]
            # Cap extreme ratios (>5x atau <0.1x) agar tidak meledak
            ratio = np.clip(ratio, 0.1, 5.0)
            mtm_ratios.setdefault(key, []).append(ratio)

    # Hitung rata-rata ratio per transisi bulan
    avg_mtm = {}
    for key, ratio_list in mtm_ratios.items():
        avg_mtm[key] = float(np.median(ratio_list))  # median lebih robust vs mean

    # ----------------------------------------------------------------
    # STEP 3: GENERATE 12-MONTH FORECAST
    # ----------------------------------------------------------------
    preds = []
    current_val = baseline
    last_month_num = ts.index[-1].month
    future_dates = pd.date_range(
        ts.index[-1] + pd.DateOffset(months=1),
        periods=n, freq='MS'
    )

    for i, d in enumerate(future_dates):
        from_m = last_month_num if i == 0 else future_dates[i - 1].month
        to_m = d.month
        key = (from_m, to_m)

        if key in avg_mtm:
            ratio = avg_mtm[key]
        else:
            # Fallback: gunakan overall average ratio
            all_ratios = [r for rlist in mtm_ratios.values() for r in rlist]
            ratio = float(np.median(all_ratios)) if all_ratios else 1.0

        current_val = max(0.0, current_val * ratio)
        preds.append(current_val)

    # ----------------------------------------------------------------
    # STEP 4: CONFIDENCE INTERVAL
    # Gunakan std dev dari semua historical MtM ratios sebagai proxy volatilitas
    # ----------------------------------------------------------------
    all_ratios_flat = [r for rlist in mtm_ratios.values() for r in rlist]
    ratio_std = float(np.std(all_ratios_flat)) if len(all_ratios_flat) > 1 else 0.2

    ci_lo = np.array(preds) * max(0, 1 - 1.5 * ratio_std)
    ci_hi = np.array(preds) * (1 + 1.5 * ratio_std)
    ci_lo = np.maximum(0, ci_lo)

    fc_series = pd.Series(preds, index=future_dates)

    # Attach warnings ke dalam fc_series sebagai attribute (via custom dict return)
    return fc_series, ci_lo, ci_hi, warnings_list, avg_mtm


def ai_build_mtm_heatmap_data(avg_mtm):
    """
    Build MtM ratio matrix untuk visualisasi heatmap
    Rows = from_month, Cols = to_month
    """
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    matrix = np.full((12, 12), np.nan)

    for (from_m, to_m), ratio in avg_mtm.items():
        matrix[from_m - 1][to_m - 1] = ratio

    return pd.DataFrame(matrix, index=month_names, columns=month_names)

def ai_backtest(ts, method, params, holdout=3):
    """Backtest: hold out last N months, measure MAPE"""
    if len(ts) <= holdout + 2:
        return None, None

    train = ts.iloc[:-holdout]
    actual = ts.iloc[-holdout:]

    try:
        if method == 'ma_simple':
            fc, _, _ = ai_forecast_ma(train, holdout, params.get('window', 3), False)
        elif method == 'ma_weighted':
            fc, _, _ = ai_forecast_ma(train, holdout, params.get('window', 3), True)
        elif method == 'holt_winters':
            fc, _, _ = ai_forecast_holt_winters(train, holdout)
        elif method == 'linear_seasonal':
            fc, _, _ = ai_forecast_linear_seasonal(train, holdout)
        elif method == 'prophet':
            fc, _, _ = ai_forecast_prophet(train, holdout)
        elif method == 'rolling_mtm':
            # Backtest untuk rolling_mtm — ambil fc saja, abaikan warnings & avg_mtm
            result = ai_forecast_rolling_baseline_mtm(
                train, holdout,
                baseline_window=params.get('baseline_window', 3)
            )
            fc = result[0]  # fc series saja
        else:
            return None, None

        fc_vals = fc.values[:holdout]
        act_vals = actual.values
        mask = act_vals > 0

        mape = float(np.mean(np.abs(
            (act_vals[mask] - fc_vals[mask]) / act_vals[mask]
        )) * 100) if mask.sum() > 0 else None

        comp_df = pd.DataFrame({
            'Month': actual.index.strftime('%b-%y'),
            'Actual': act_vals.astype(int),
            'Predicted': fc_vals.round().astype(int),
            'Error': (fc_vals - act_vals).round(1),
            'APE (%)': np.where(act_vals > 0,
                                np.abs((fc_vals - act_vals) / act_vals) * 100, 0).round(1)
        })

        return mape, comp_df
    except Exception:
        return None, None


def ai_get_existing_plan(df_ecomm_forecast, entity_type, entity_value, month_cols):
    """Extract existing plan from ecomm forecast for comparison"""
    if df_ecomm_forecast.empty or not month_cols:
        return pd.Series(dtype=float)

    if entity_type == 'SKU':
        mask = df_ecomm_forecast['SKU_ID'] == entity_value
    else:
        mask = df_ecomm_forecast['Brand'] == entity_value

    subset = df_ecomm_forecast[mask]
    if subset.empty:
        return pd.Series(dtype=float)

    plan_vals = {}
    for col in month_cols:
        try:
            dt = None
            for fmt in ['%b-%y', '%b %y', '%b_%y', '%B %Y']:
                try:
                    dt = datetime.strptime(str(col).strip(), fmt)
                    break
                except:
                    continue
            if dt:
                plan_vals[dt] = float(subset[col].sum())
        except:
            continue

    if not plan_vals:
        return pd.Series(dtype=float)

    return pd.Series(plan_vals).sort_index()
    

# --- ====================================================== ---
# ---                DASHBOARD INITIALIZATION               ---
# --- ====================================================== ---

# --- DASHBOARD INITIALIZATION ---
# Initialize connection
client = init_gsheet_connection()

if client is None:
    st.error("❌ Tidak dapat terhubung ke Google Sheets")
    st.stop()

# Load and process data
with st.spinner('🔄 Loading and processing data from Google Sheets...'):
    all_data = load_and_process_data(client)
    
    df_product = all_data.get('product', pd.DataFrame())
    df_product_active = all_data.get('product_active', pd.DataFrame())
    df_sales = all_data.get('sales', pd.DataFrame())
    df_forecast = all_data.get('forecast', pd.DataFrame())
    df_po = all_data.get('po', pd.DataFrame())
    df_stock = all_data.get('stock', pd.DataFrame())
    
    # Ganti rofo_onwards dengan ecomm_forecast (untuk Tab 7)
    df_ecomm_forecast = all_data.get('ecomm_forecast', pd.DataFrame())
    ecomm_forecast_month_cols = all_data.get('ecomm_forecast_month_cols', [])
    
    # Tambah data reseller (untuk Tab 9) - DARI all_data LAMA
    df_reseller_forecast = all_data.get('reseller_forecast', pd.DataFrame())
    reseller_all_month_cols = all_data.get('reseller_all_month_cols', [])
    reseller_historical_cols = all_data.get('reseller_historical_cols', [])
    reseller_forecast_cols = all_data.get('reseller_forecast_cols', [])
    
    # Untuk backward compatibility (jika ada script yang masih pakai nama lama)
    df_rofo_onwards = df_ecomm_forecast  # Alias untuk Tab 7
    rofo_onwards_month_cols = ecomm_forecast_month_cols  # Alias untuk Tab 7
    
    # --- LOAD DATA RESELLER LENGKAP (BARU) ---
    with st.spinner('🔄 Loading Reseller Data...'):
        reseller_complete_data = load_reseller_complete_data(client)
        
        # Data Reseller yang sudah ada (tetap pakai untuk kompatibilitas)
        if df_reseller_forecast.empty and 'forecast' in reseller_complete_data:
            df_reseller_forecast = reseller_complete_data.get('forecast', pd.DataFrame())
        
        if not reseller_forecast_cols and 'forecast_month_cols' in reseller_complete_data:
            reseller_forecast_cols = reseller_complete_data.get('forecast_month_cols', [])
        
        # Data Reseller BARU
        df_sales_reseller = reseller_complete_data.get('sales', pd.DataFrame())
        df_past_rofo_reseller = reseller_complete_data.get('past_rofo', pd.DataFrame())
        df_past_po_reseller = reseller_complete_data.get('past_po', pd.DataFrame())

# Calculate metrics
monthly_performance = calculate_monthly_performance(df_forecast, df_po, df_product)
last_3_months_performance = get_last_3_months_performance(monthly_performance)
inventory_metrics = calculate_inventory_metrics_with_3month_avg(df_stock, df_sales, df_product)
sales_vs_forecast = calculate_sales_vs_forecast_po(df_sales, df_forecast, df_po, df_product)

# Calculate financial metrics
df_financial = calculate_financial_metrics_all(df_sales, df_product)
df_inventory_financial = calculate_inventory_financial(df_stock, df_product)
seasonal_pattern = calculate_seasonality(df_financial) if not df_financial.empty else pd.DataFrame()
forecast_bias = calculate_forecast_bias(df_forecast, df_po)
profitability_segments = identify_profitability_segments(df_financial) if not df_financial.empty else pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    with col_sb2:
        if st.button("📊 Show Data Stats", use_container_width=True):
            st.session_state.show_stats = True
            
    # --- TAMBAHAN: TOMBOL CETAK PDF ---
    st.markdown("---")
    import streamlit.components.v1 as components
    
    if st.button("🖨️ Save as PDF", use_container_width=True):
        # Script JavaScript untuk memicu dialog print browser
        components.html(
            """
            <script>
            window.print();
            </script>
            """,
            height=0,
            width=0
        )
    st.caption("Tip: Pilih Destination **'Save as PDF'** & centang **'Background graphics'** di settings print.")
    # ----------------------------------

    # --- 🔒 SECURITY (SESSION STATE PIN PROTECTION) ---
    st.markdown("---")
    st.markdown("### 🔒 Data Security")
    
    # Inisialisasi state kunci di memori Streamlit
    if "unlocked" not in st.session_state:
        st.session_state.unlocked = False

    # Jika masih terkunci, tampilkan kolom PIN
    if not st.session_state.unlocked:
        admin_pin = st.text_input("Enter Admin PIN to Unmask:", type="password", help="Hanya untuk presentasi internal.")
        if admin_pin == "FOOM2026":
            st.session_state.unlocked = True
            st.rerun() # Refresh seketika untuk menghilangkan kolom password
        elif admin_pin != "":
            st.error("❌ PIN Salah. Akses Ditolak.")
    else:
        # Jika sudah terbuka, hilangkan kolom PIN, munculkan tombol Lock
        st.success("🔓 Mode Terbuka (Admin)")
        if st.button("🔒 Lock Dashboard", use_container_width=True):
            st.session_state.unlocked = False
            st.rerun()

    # FUNGSI MASTER FORMAT RUPIAH (DYNAMIC ASTERISK MASKING)
    def format_rupiah(value):
        """Format angka menjadi Rupiah. Jika Mask=True, bintang menyesuaikan jumlah digit."""
        if pd.isna(value) or value == 0: return "Rp 0"
        
        abs_val = abs(value)
        sign = "-" if value < 0 else ""
        
        # Helper: Mengubah angka menjadi bintang, tapi membiarkan titik/koma
        def mask_digits(text_num):
            return "".join(["*" if char.isdigit() else char for char in text_num])
        
        if not st.session_state.unlocked: # JIKA TERKUNCI (SENSOR)
            if abs_val >= 1_000_000_000: 
                base_str = f"{abs_val/1e9:,.1f}"
                return f"{sign}Rp {mask_digits(base_str)} M"
            elif abs_val >= 1_000_000: 
                base_str = f"{abs_val/1e6:,.1f}"
                return f"{sign}Rp {mask_digits(base_str)} Jt"
            else: 
                base_str = f"{abs_val:,.0f}"
                return f"{sign}Rp {mask_digits(base_str)}"
        else: # JIKA TERBUKA (ASLI)
            if abs_val >= 1_000_000_000: return f"{sign}Rp {abs_val/1e9:,.1f} M"
            elif abs_val >= 1_000_000: return f"{sign}Rp {abs_val/1e6:,.1f} Jt"
            else: return f"{sign}Rp {abs_val:,.0f}"

    st.markdown("---")
    st.markdown("### 📈 Data Overview")
    
    
    if not df_product_active.empty:
        st.metric("Active SKUs", len(df_product_active))
    
    if not df_stock.empty:
        total_stock = df_stock['Stock_Qty'].sum()
        st.metric("Total Stock", f"{total_stock:,.0f}")
    
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        accuracy = monthly_performance[last_month]['accuracy']
        st.metric("Latest Accuracy", f"{accuracy:.1f}%")
    
        
    st.markdown("---")
    
    # Threshold Settings
    st.markdown("### ⚙️ Threshold Settings")
    under_threshold = st.slider("Under Forecast Threshold (%)", 0, 100, 80)
    over_threshold = st.slider("Over Forecast Threshold (%)", 100, 200, 120)
    
    st.markdown("---")
    
    # Inventory Thresholds
    st.markdown("### 📦 Inventory Thresholds")
    low_stock_threshold = st.slider("Low Stock (months)", 0.0, 2.0, 0.8, 0.1)
    high_stock_threshold = st.slider("High Stock (months)", 1.0, 6.0, 2.0, 0.1)
    
    # Financial Thresholds
    st.markdown("---")
    st.markdown("### 💰 Financial Thresholds")
    high_margin_threshold = st.slider("High Margin Threshold (%)", 0, 100, 40)
    low_margin_threshold = st.slider("Low Margin Threshold (%)", 0, 100, 20)
    
    # --- 🎨 THEME SELECTOR (ENHANCED EDITION) ---
    st.markdown("---")
    st.markdown("### 🎨 UI Theme Settings")
    
    theme_choice = st.selectbox(
        "Pilih Tema Dashboard:",
        [
            "⚪ Tema Semula (Enhanced Light)", 
            "⬛ Material Dark (Solid & Colored)"
        ],
        index=0
    )

    theme_css = ""
    
    if theme_choice == "⚪ Tema Semula (Enhanced Light)":
        # Perbaikan: Background utama dibuat abu-abu yang lebih jelas agar chart putih sangat menonjol
        theme_css = """
        <style>
            /* Background utama abu-abu solid (lebih gelap sedikit) */
            [data-testid="stAppViewContainer"] { background-color: #E5E7EB !important; color: #333333 !important; }
            
            /* Sidebar tetap putih dengan border abu-abu agar kontras */
            [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #D1D5DB !important; }
            
            /* Membungkus Chart Plotly dengan kotak putih & shadow yang lebih tegas */
            .stPlotlyChart {
                background-color: #FFFFFF !important;
                border-radius: 12px !important;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.08) !important;
                padding: 10px !important;
            }
            
            /* Membungkus Dataframe dengan kotak putih */
            .stDataFrame {
                background-color: #FFFFFF !important;
                border-radius: 12px !important;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.08) !important;
                padding: 10px !important;
            }
        </style>
        """
        
    elif theme_choice == "⬛ Material Dark (Solid & Colored)":
        # Perbaikan: Tabel diubah dark mode. Card bewarna dibiarkan tetap bewarna (gradient tidak ditimpa).
        theme_css = """
        <style>
            /* Base Material Dark BG */
            [data-testid="stAppViewContainer"] { background-color: #1A2035 !important; color: #FFFFFF !important; }
            [data-testid="stSidebar"] { background-color: #1F283E !important; border-right: none !important; }
            h1, h2, h3, h4, p, label, .stMarkdown, .stText { color: #FFFFFF !important; }
            hr { border-color: rgba(255,255,255,0.1) !important; }
            
            /* === FIX 1: TABEL (DATAFRAME) MENJADI DARK === */
            [data-testid="stDataFrame"] > div, 
            [data-testid="stDataFrame"] table, 
            [data-testid="stDataFrame"] th, 
            [data-testid="stDataFrame"] td {
                background-color: #1F283E !important;
                color: #FFFFFF !important;
                border-color: rgba(255,255,255,0.05) !important;
            }
            /* Warna Header Tabel sedikit lebih gelap */
            [data-testid="stDataFrame"] th { background-color: #171d30 !important; }
            
            /* === FIX 2: BIARKAN GRADIENT CARD TETAP BERWARNA === */
            /* Pastikan font di dalam card berwarna tetap putih agar terbaca */
            .grad-label, .grad-value, .grad-sub, .tm-title, .tm-main-val, .tm-unit, .tm-sub-row, .fin-title, .fin-val, .fin-sub { 
                color: #FFFFFF !important; 
                text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important; 
            }
            
            /* Custom HTML Cards yang memang tidak punya warna khusus, kita buat Dark Blue solid */
            .p-card, .sku-header, .metric-highlight {
                background-color: #1F283E !important;
                border: none !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 20px 0 rgba(0,0,0,.14), 0 7px 10px -5px rgba(0,0,0,.4) !important;
                color: #FFFFFF !important;
            }
            .p-val, .p-label { color: #FFFFFF !important; text-shadow: none !important;}
            
            /* Native Streamlit Metrics */
            [data-testid="stMetric"] {
                background-color: #1F283E !important;
                border-radius: 8px !important;
                border: none !important;
                box-shadow: 0 4px 20px 0 rgba(0,0,0,.14) !important;
                padding: 15px !important;
            }
            [data-testid="stMetricValue"] { color: #FFFFFF !important; }
            
            /* Tabs Styling */
            .stTabs [data-baseweb="tab"] { background: #1F283E !important; color: #A9AFBB !important; border: none !important; }
            .stTabs [aria-selected="true"] { background: #9C27B0 !important; color: white !important; box-shadow: 0 4px 20px 0 rgba(0,0,0,.14), 0 7px 10px -5px rgba(156,39,176,.4) !important; }
            
            /* Fix Plotly White Backgrounds to Transparent */
            .js-plotly-plot .plotly .bg, .js-plotly-plot .plotly .paper-bg { fill: transparent !important; }
            .js-plotly-plot .plotly text { fill: #A9AFBB !important; }
            .js-plotly-plot .plotly .gridlayer path { stroke: rgba(255,255,255,0.05) !important; }
        </style>
        """

    if theme_css:
        st.markdown(theme_css, unsafe_allow_html=True)

# Data quality check
if 'show_stats' in st.session_state and st.session_state.show_stats:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Data Quality Check")
    
    for df_name, df in [("Product", df_product), ("Sales", df_sales), 
                       ("Forecast", df_forecast), ("PO", df_po), 
                       ("Stock", df_stock), ("Financial", df_financial)]:
        if not df.empty:
            checks = validate_data_quality(df, df_name)
            with st.sidebar.expander(f"{df_name} Data"):
                for check_name, check_result in checks.items():
                    st.write(f"{check_name}: {check_result}")

# --- MAIN DASHBOARD ---
# =================================================================================
# 📈 PREMIUM FORECAST ACCURACY DASHBOARD SECTION (UPDATED)
# =================================================================================
st.subheader("📈 Forecast Accuracy Performance Trends")

if monthly_performance:
    # 1. Prepare Data
    summary_data = []
    for month, data in sorted(monthly_performance.items()):
        summary_data.append({
            'Month': month,
            'Month_Display': month.strftime('%b %Y'),
            'Accuracy': data['accuracy'],
            'Total_SKUs': data['total_records'],
            'Under': data['status_counts'].get('Under', 0),
            'Over': data['status_counts'].get('Over', 0),
            'Accurate': data['status_counts'].get('Accurate', 0),
            'MAPE': data['mape']
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Month')

    if not summary_df.empty:
        # --- A. METRIC CARDS (Gradient Style) ---
        # Calculate Aggregates
        avg_acc = summary_df['Accuracy'].mean()
        last_acc = summary_df['Accuracy'].iloc[-1]
        prev_acc = summary_df['Accuracy'].iloc[-2] if len(summary_df) > 1 else last_acc
        delta_acc = last_acc - prev_acc
        
        best_month = summary_df.loc[summary_df['Accuracy'].idxmax()]
        stability = max(0, 100 - summary_df['Accuracy'].std())

        # CSS khusus untuk Gradient Cards
        st.markdown("""
        <style>
            .grad-card {
                border-radius: 15px;
                padding: 1.5rem;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
                margin-bottom: 1rem;
                position: relative;
                overflow: hidden;
            }
            .grad-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }
            
            /* Glassmorphism overlay effect */
            .grad-card::before {
                content: "";
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
                opacity: 0.5;
                pointer-events: none;
            }

            .grad-label { 
                font-size: 0.85rem; 
                font-weight: 600; 
                text-transform: uppercase; 
                letter-spacing: 1px;
                opacity: 0.9;
                margin-bottom: 0.5rem;
                position: relative; 
                z-index: 1;
            }
            .grad-value { 
                font-size: 2.2rem; 
                font-weight: 800; 
                margin-bottom: 0.2rem;
                position: relative; 
                z-index: 1;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .grad-sub { 
                font-size: 0.85rem; 
                font-weight: 500; 
                opacity: 0.9;
                display: flex;
                align-items: center;
                gap: 5px;
                position: relative; 
                z-index: 1;
            }
            .pill {
                background: rgba(255,255,255,0.25);
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.75rem;
                backdrop-filter: blur(4px);
            }
        </style>
        """, unsafe_allow_html=True)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        # Helper Function untuk membuat Gradient Card
        def gradient_card(title, value, sub_text, pill_text, gradient_css):
            return f"""
            <div class="grad-card" style="{gradient_css}">
                <div class="grad-label">{title}</div>
                <div class="grad-value">{value}</div>
                <div class="grad-sub">
                    <span class="pill">{pill_text}</span> {sub_text}
                </div>
            </div>
            """

        with kpi1:
            # Current Accuracy - Blue/Indigo Gradient
            arrow = "▲" if delta_acc >= 0 else "▼"
            grad_css = "background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);"
            st.markdown(gradient_card(
                "Current Accuracy", 
                f"{last_acc:.1f}%", 
                "vs last month", 
                f"{arrow} {abs(delta_acc):.1f}%", 
                grad_css
            ), unsafe_allow_html=True)

        with kpi2:
            # Average YTD - Teal/Cyan Gradient
            avg_diff = avg_acc - 80
            pill_icon = "✅" if avg_diff >= 0 else "⚠️"
            grad_css = "background: linear-gradient(135deg, #0891B2 0%, #22D3EE 100%);"
            st.markdown(gradient_card(
                "Average (YTD)", 
                f"{avg_acc:.1f}%", 
                "vs Target (80%)", 
                f"{pill_icon} {abs(avg_diff):.1f}%", 
                grad_css
            ), unsafe_allow_html=True)

        with kpi3:
            # Best Performance - Emerald/Green Gradient
            grad_css = "background: linear-gradient(135deg, #059669 0%, #10B981 100%);"
            st.markdown(gradient_card(
                "Best Performance", 
                f"{best_month['Accuracy']:.1f}%", 
                "Highest Record", 
                f"📅 {best_month['Month_Display']}", 
                grad_css
            ), unsafe_allow_html=True)

        with kpi4:
            # Stability - Orange/Amber Gradient
            grad_css = "background: linear-gradient(135deg, #EA580C 0%, #F59E0B 100%);"
            st.markdown(gradient_card(
                "Stability Score", 
                f"{stability:.0f}", 
                "Consistency Metric", 
                "📈 0-100", 
                grad_css
            ), unsafe_allow_html=True)

        st.write("") # Spacer

        # --- B. ADVANCED COMBO CHART (Updated Logic) ---
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 1. Background Target Zones
        fig.add_hrect(
            y0=80, y1=110,
            fillcolor="rgba(16, 185, 129, 0.08)", layer="below", line_width=0,
            secondary_y=False
        )
        fig.add_hrect(
            y0=70, y1=80,
            fillcolor="rgba(245, 158, 11, 0.08)", layer="below", line_width=0,
            secondary_y=False
        )

        # 2. Context Layer: Bar Chart for Total SKUs
        fig.add_trace(
            go.Bar(
                x=summary_df['Month_Display'],
                y=summary_df['Total_SKUs'],
                name="Total SKUs",
                marker_color='rgba(156, 163, 175, 0.15)',
                hoverinfo='y',
                showlegend=True,
            ),
            secondary_y=True,
        )

        # 3. Main Layer: Accuracy Line with 3-Color Logic
        # Define colors list based on value
        marker_colors = []
        for val in summary_df['Accuracy']:
            if val < 70:
                marker_colors.append('#EF4444') # Merah
            elif val < 80:
                marker_colors.append('#F59E0B') # Kuning
            else:
                marker_colors.append('#10B981') # Hijau

        fig.add_trace(
            go.Scatter(
                x=summary_df['Month_Display'],
                y=summary_df['Accuracy'],
                name="Accuracy %",
                mode='lines+markers',
                line=dict(color='#6366F1', width=3, shape='spline', smoothing=1.3), # Garis tetap warna Indigo/Ungu biar elegan
                marker=dict(
                    size=14, # Sedikit diperbesar
                    color=marker_colors, # Pakai list warna yg sudah dilogika di atas
                    line=dict(width=2, color='white')
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Accuracy: <b>%{y:.1f}%</b><br>" +
                    "MAPE: %{customdata[0]:.1f}%<br>" +
                    "<extra></extra>"
                ),
                customdata=summary_df[['MAPE']]
            ),
            secondary_y=False,
        )

        # 4. Layout Styling
        fig.update_layout(
            height=480,
            title=dict(
                text='<b>📊 Accuracy Trend vs SKU Volume</b>',
                font=dict(size=16, color='#374151'),
                x=0, y=0.98
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=40, l=40, r=40)
        )

        fig.update_yaxes(
            title="<b>Accuracy (%)</b>", 
            range=[40, 100], 
            gridcolor='rgba(0,0,0,0.05)',
            secondary_y=False,
            tickfont=dict(color='#4F46E5', weight='bold')
        )
        
        fig.update_yaxes(
            title="Total SKUs", 
            showgrid=False, 
            visible=False,
            secondary_y=True
        )
        
        fig.update_xaxes(showgrid=False, tickfont=dict(weight='bold'))

        st.plotly_chart(fig, use_container_width=True)

        # --- C. AUTO INSIGHT ---
        # Logic warna insight
        if last_acc >= 80:
            insight_color = "#10B981" # Hijau
            status_text = "EXCELLENT PERFORMANCE 🚀"
        elif last_acc >= 70:
            insight_color = "#F59E0B" # Kuning
            status_text = "MODERATE PERFORMANCE ⚠️"
        else:
            insight_color = "#EF4444" # Merah
            status_text = "CRITICAL ATTENTION NEEDED 🚨"
        
        st.markdown(f"""
        <div style="background-color: white; border-radius: 10px; padding: 1rem; border-left: 6px solid {insight_color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05); color: #4B5563;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 1.5rem;">💡</div>
                <div>
                    <div style="font-weight: 800; font-size: 0.8rem; color: {insight_color}; letter-spacing: 1px;">{status_text}</div>
                    <div style="font-size: 0.95rem;">
                        Current accuracy is <strong>{last_acc:.1f}%</strong>. 
                        {'Stable performance.' if abs(delta_acc) < 2 else ('Improved by ' + f'{delta_acc:.1f}%' if delta_acc > 0 else 'Dropped by ' + f'{abs(delta_acc):.1f}%')} from previous month.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ No monthly performance data available to display trend.")

# SECTION 1: LAST 3 MONTHS PERFORMANCE (DIPERBESAR)
st.subheader("🎯 Forecast Performance - 3 Bulan Terakhir")

if last_3_months_performance:
    # Display last 3 months performance
    months_display = []
    
    # Create container untuk 3 bulan
    month_cols = st.columns(3)
    
    for i, (month, data) in enumerate(sorted(last_3_months_performance.items())):
        month_name = month.strftime('%b %Y')
        accuracy = data['accuracy']
        
        with month_cols[i]:
            under_count = data['status_counts'].get('Under', 0)
            accurate_count = data['status_counts'].get('Accurate', 0)
            over_count = data['status_counts'].get('Over', 0)
            total_records = data['total_records']
            
            # Create HTML dengan single line f-string
            html_content = (
                f'<div style="background: white; border-radius: 15px; padding: 1.5rem; margin: 0.5rem 0; box-shadow: 0 6px 20px rgba(0,0,0,0.1); border-top: 5px solid #667eea;">'
                f'<div style="text-align: center; margin-bottom: 1rem;">'
                f'<h3 style="margin: 0; color: #333;">{month_name}</h3>'
                f'<div style="font-size: 2rem; font-weight: 900; color: #667eea;">{accuracy:.1f}%</div>'
                f'<div style="font-size: 0.9rem; color: #666;">Overall Accuracy</div>'
                f'</div>'
                f'<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 1rem;">'
                f'<div style="text-align: center; padding: 0.5rem; background: #FFEBEE; border-radius: 8px;">'
                f'<div style="font-size: 1.5rem; font-weight: 900; color: #F44336;">{under_count}</div>'
                f'<div style="font-size: 0.8rem; color: #F44336;">Under</div>'
                f'</div>'
                f'<div style="text-align: center; padding: 0.5rem; background: #E8F5E9; border-radius: 8px;">'
                f'<div style="font-size: 1.5rem; font-weight: 900; color: #4CAF50;">{accurate_count}</div>'
                f'<div style="font-size: 0.8rem; color: #4CAF50;">Accurate</div>'
                f'</div>'
                f'<div style="text-align: center; padding: 0.5rem; background: #FFF3E0; border-radius: 8px;">'
                f'<div style="font-size: 1.5rem; font-weight: 900; color: #FF9800;">{over_count}</div>'
                f'<div style="font-size: 0.8rem; color: #FF9800;">Over</div>'
                f'</div>'
                f'</div>'
                f'<div style="text-align: center; font-size: 0.9rem; color: #666;">Total SKUs: {total_records}</div>'
                f'</div>'
            )
            
            st.markdown(html_content, unsafe_allow_html=True)
        
        months_display.append(month_name)
        
    # ==============================================================================
    # 1. TOTAL METRICS - BULAN TERAKHIR (Soft Pastel Gradient Version)
    # ==============================================================================
    st.divider()
    st.subheader("📊 Total Metrics - Rofo Bulan Terakhir")
    
    # Calculate metrics for LAST MONTH ONLY
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        # Count SKUs & Quantities
        under_count = last_month_data[last_month_data['Accuracy_Status'] == 'Under']['SKU_ID'].nunique()
        accurate_count = last_month_data[last_month_data['Accuracy_Status'] == 'Accurate']['SKU_ID'].nunique()
        over_count = last_month_data[last_month_data['Accuracy_Status'] == 'Over']['SKU_ID'].nunique()
        total_count_last_month = last_month_data['SKU_ID'].nunique()
        
        under_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Under']['Forecast_Qty'].sum()
        accurate_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Accurate']['Forecast_Qty'].sum()
        over_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Over']['Forecast_Qty'].sum()
        total_forecast_qty = last_month_data['Forecast_Qty'].sum()
        
        # Percentages
        under_pct = (under_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        accurate_pct = (accurate_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        over_pct = (over_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        
        under_qty_pct = (under_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0
        accurate_qty_pct = (accurate_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0
        over_qty_pct = (over_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0

        last_month_accuracy = monthly_performance[last_month]['accuracy']

        # --- CSS STYLE (Fixed Indentation & Soft Colors) ---
        st.markdown("""
<style>
    /* Card Styles */
    .tm-card {
        border-radius: 16px;
        padding: 1.2rem;
        color: white;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .tm-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1); }
    
    .tm-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .tm-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.95; }
    .tm-icon { font-size: 1.2rem; opacity: 0.9; background: rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 8px; }
    
    .tm-main-val { font-size: 1.8rem; font-weight: 800; margin-bottom: 0px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
    .tm-unit { font-size: 0.9rem; font-weight: 500; opacity: 0.9; margin-left: 2px; }
    
    .tm-sub-row { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; font-size: 0.8rem; opacity: 0.95; }
    .tm-badge { background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }

    /* Process Flow Styles */
    .process-container {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 15px;
        margin-top: 10px;
    }
    .p-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-top: 5px solid #ccc;
        position: relative;
    }
    .p-label { font-size: 0.8rem; font-weight: 700; color: #888; letter-spacing: 1px; margin-bottom: 5px; text-transform: uppercase; }
    .p-val { font-size: 2rem; font-weight: 800; color: #333; margin-bottom: 10px; }
    .p-badge-container { display: flex; gap: 8px; flex-wrap: wrap; }
    .p-badge { 
        font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 20px; 
        display: flex; align-items: center; gap: 4px;
    }
    
    /* Process Flow Colors */
    .border-rofo { border-top-color: #5C6BC0; } /* Soft Indigo */
    .border-po { border-top-color: #FFA726; }   /* Soft Orange */
    .border-sales { border-top-color: #66BB6A; } /* Soft Green */
    
    .text-rofo { color: #3949AB; }
    .text-po { color: #F57C00; }
    .text-sales { color: #2E7D32; }

    .bg-rofo-light { background-color: #E8EAF6; color: #3949AB; }
    .bg-po-light { background-color: #FFF3E0; color: #EF6C00; }
    .bg-sales-light { background-color: #E8F5E9; color: #2E7D32; }
</style>
""", unsafe_allow_html=True)

        # Helper Function untuk render card (TANPA INDENTASI DI HTML)
        def render_soft_card(title, icon, count, qty, qty_pct, bg_gradient):
            # HTML disusun tanpa indentasi agar aman dari bug Markdown
            html = f"""
<div class="tm-card" style="background: {bg_gradient};">
<div class="tm-header">
<span class="tm-title">{title}</span>
<span class="tm-icon">{icon}</span>
</div>
<div>
<span class="tm-main-val">{count}</span><span class="tm-unit">SKUs</span>
</div>
<div class="tm-sub-row">
<span>Qty: {qty:,.0f}</span>
<span class="tm-badge">{qty_pct:.1f}%</span>
</div>
</div>"""
            return html

        # Render Total Metrics
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            # Under - Soft Red
            st.markdown(render_soft_card("Under Forecast", "📉", under_count, under_forecast_qty, under_qty_pct, 
                "linear-gradient(135deg, #ef5350 0%, #e53935 100%)"), unsafe_allow_html=True)
        with c2:
            # Accurate - Soft Green
            st.markdown(render_soft_card("Accurate Forecast", "🎯", accurate_count, accurate_forecast_qty, accurate_qty_pct, 
                "linear-gradient(135deg, #26a69a 0%, #00897b 100%)"), unsafe_allow_html=True)
        with c3:
            # Over - Soft Orange
            st.markdown(render_soft_card("Over Forecast", "📈", over_count, over_forecast_qty, over_qty_pct, 
                "linear-gradient(135deg, #ffa726 0%, #fb8c00 100%)"), unsafe_allow_html=True)
        with c4:
            # Overall - Soft Indigo
            st.markdown(f"""
<div class="tm-card" style="background: linear-gradient(135deg, #5c6bc0 0%, #3949ab 100%);">
<div class="tm-header">
<span class="tm-title">OVERALL SCORE</span>
<span class="tm-icon">🏆</span>
</div>
<div>
<span class="tm-main-val">{last_month_accuracy:.1f}%</span>
</div>
<div class="tm-sub-row">
<span>{last_month.strftime('%B %Y')}</span>
<span class="tm-badge">Total: {total_count_last_month}</span>
</div>
</div>""", unsafe_allow_html=True)

    # ==============================================================================
    # 2. COMPARISON CARDS - PROCESS FLOW STYLE (FIXED RENDERING)
    # ==============================================================================
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        # Calculate Totals
        rofo_tot = last_month_data['Forecast_Qty'].sum()
        po_tot = last_month_data['PO_Qty'].sum()
        
        # Get Sales
        sales_tot = 0
        if not df_sales.empty:
            sales_tot = df_sales[df_sales['Month'] == last_month]['Sales_Qty'].sum()

        # Calculate Ratios
        po_vs_rofo = (po_tot / rofo_tot * 100) if rofo_tot > 0 else 0
        sales_vs_rofo = (sales_tot / rofo_tot * 100) if rofo_tot > 0 else 0
        sales_vs_po = (sales_tot / po_tot * 100) if po_tot > 0 else 0

        st.write("") # Spacer
        st.subheader(f"🔄 Business Flow Performance (Rofo -> PO -> Sales Performance): {last_month.strftime('%B %Y')}")

        # HTML Structure - TANPA INDENTASI SAMA SEKALI
        html_process = f"""
<div class="process-container">
<div class="p-card border-rofo">
<div class="p-label">1. PLANNING (ROFO)</div>
<div class="p-val text-rofo">{rofo_tot:,.0f}</div>
<div class="p-badge-container">
<span class="p-badge bg-rofo-light">🎯 Baseline Target</span>
</div>
</div>
<div class="p-card border-po">
<div class="p-label">2. EXECUTION (PO)</div>
<div class="p-val text-po">{po_tot:,.0f}</div>
<div class="p-badge-container">
<span class="p-badge bg-po-light">{po_vs_rofo:.1f}% vs Rofo</span>
<span class="p-badge" style="background:#f5f5f5; color:#666;">Gap: {po_tot - rofo_tot:+,.0f}</span>
</div>
</div>
<div class="p-card border-sales">
<div class="p-label">3. REALIZATION (SALES)</div>
<div class="p-val text-sales">{sales_tot:,.0f}</div>
<div class="p-badge-container">
<span class="p-badge bg-sales-light">{sales_vs_rofo:.1f}% vs Rofo</span>
<span class="p-badge bg-sales-light">{sales_vs_po:.1f}% vs PO</span>
</div>
</div>
</div>
"""
        st.markdown(html_process, unsafe_allow_html=True)

st.divider()
# SECTION 2: MONTHLY EVALUATION (UNDER & OVER ONLY) DENGAN FILTER BULAN
st.subheader("📋 Evaluasi Rofo per Bulan (Under & Over Forecast)")

if monthly_performance:
    sorted_months = sorted(monthly_performance.keys())
    if sorted_months:
        
        # --- MULAI SCRIPT FILTER BULAN ---
        # 1. Buat list pilihan bulan dalam format string (misal: 'Jan 2025')
        month_options = [m.strftime('%b %Y') for m in sorted_months]
        
        # 2. Buat dropdown selectbox. default index diset ke yang paling akhir (bulan terbaru)
        selected_month_str = st.selectbox(
            "📅 Pilih Bulan Evaluasi:", 
            options=month_options, 
            index=len(month_options) - 1
        )
        
        # 3. Cari kembali key datetime aslinya berdasarkan pilihan user
        selected_month_idx = month_options.index(selected_month_str)
        selected_month_key = sorted_months[selected_month_idx]
        
        # 4. Timpa variabel lama agar script tab HTML di bawahnya tetap berjalan normal tanpa error
        last_month_data = monthly_performance[selected_month_key]
        last_month_name = selected_month_str
        # --- AKHIR SCRIPT FILTER BULAN ---
        
        # Create tabs for Under and Over SKUs (Script di bawah ini tetap sama)
        eval_tab1, eval_tab2 = st.tabs([f"📉 UNDER Forecast ({last_month_name})", f"📈 OVER Forecast ({last_month_name})"])
        
        with eval_tab1:
            under_skus_df = last_month_data['under_skus']
            if not under_skus_df.empty:
                # Add inventory data
                if 'inventory_df' in inventory_metrics:
                    inventory_data = inventory_metrics['inventory_df'][['SKU_ID', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']]
                    under_skus_df = pd.merge(under_skus_df, inventory_data, on='SKU_ID', how='left')
                
                # TAMBAH: Get last 3 months sales data
                sales_cols_last_3 = []
                if not df_sales.empty:
                    # Get last 3 months from sales data
                    sales_months = sorted(df_sales['Month'].unique())
                    if len(sales_months) >= 3:
                        last_3_sales_months = sales_months[-3:]
                        
                        # Create pivot for last 3 months sales
                        try:
                            sales_pivot = df_sales[df_sales['Month'].isin(last_3_sales_months)].pivot_table(
                                index='SKU_ID',
                                columns='Month',
                                values='Sales_Qty',
                                aggfunc='sum',
                                fill_value=0
                            ).reset_index()
                            
                            # Rename columns to month names
                            month_rename = {}
                            for col in sales_pivot.columns:
                                if isinstance(col, datetime):
                                    month_rename[col] = col.strftime('%b-%Y')
                            sales_pivot = sales_pivot.rename(columns=month_rename)
                            
                            # Merge with under_skus_df
                            under_skus_df = pd.merge(
                                under_skus_df,
                                sales_pivot,
                                on='SKU_ID',
                                how='left'
                            )
                            
                            # Get the sales column names
                            sales_cols_last_3 = [col for col in sales_pivot.columns if isinstance(col, str) and '-' in col]
                            sales_cols_last_3 = sorted(sales_cols_last_3[-3:])  # Get last 3 months
                            
                        except Exception as e:
                            st.warning(f"Tidak bisa menambahkan data sales 3 bulan terakhir: {str(e)}")
                
                # Prepare display columns - TAMBAH sales columns
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Accuracy_Status',
                              'Forecast_Qty', 'PO_Qty', 'PO_Rofo_Ratio', 
                              'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
                
                # Tambah sales columns jika ada
                display_cols.extend(sales_cols_last_3)
                
                # Filter available columns
                available_cols = [col for col in display_cols if col in under_skus_df.columns]
                
                # Pastikan Product_Name selalu ada
                if 'Product_Name' not in available_cols and 'Product_Name' in under_skus_df.columns:
                    available_cols.insert(1, 'Product_Name')
                
                # Format the dataframe
                display_df = under_skus_df[available_cols].copy()
                
                # Add formatted columns
                if 'PO_Rofo_Ratio' in display_df.columns:
                    display_df['PO_Rofo_Ratio'] = display_df['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%")
                
                if 'Cover_Months' in display_df.columns:
                    display_df['Cover_Months'] = display_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if x < 999 else "N/A")
                
                if 'Avg_Monthly_Sales_3M' in display_df.columns:
                    display_df['Avg_Monthly_Sales_3M'] = display_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
                
                # Format sales columns
                for col in sales_cols_last_3:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
                
                # Rename columns for display
                column_names = {
                    'SKU_ID': 'SKU ID',
                    'Product_Name': 'Product Name',
                    'Brand': 'Brand',
                    'SKU_Tier': 'Tier',
                    'Accuracy_Status': 'Status',
                    'Forecast_Qty': 'Forecast Qty',
                    'PO_Qty': 'PO Qty',
                    'PO_Rofo_Ratio': 'PO/Rofo %',
                    'Stock_Qty': 'Stock Available',
                    'Avg_Monthly_Sales_3M': 'Avg Sales (3M)',
                    'Cover_Months': 'Cover (Months)'
                }
                
                # Add sales columns to rename dict
                for col in sales_cols_last_3:
                    column_names[col] = col
                
                display_df = display_df.rename(columns=column_names)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500
                )
                
                # Summary dengan HIGHLIGHT
                total_forecast = under_skus_df['Forecast_Qty'].sum()
                total_po = under_skus_df['PO_Qty'].sum()
                avg_ratio = under_skus_df['PO_Rofo_Ratio'].mean()
                selisih_qty = total_po - total_forecast
                selisih_persen = (selisih_qty / total_forecast * 100) if total_forecast > 0 else 0
                po_rofo_pct = (total_po / total_forecast * 100) if total_forecast > 0 else 0
                
                # Buat HTML content
                html_content = f"""
                <div style="background: #FFEBEE; border-left: 5px solid #F44336; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h4 style="color: #C62828; margin-top: 0;">📉 UNDER FORECAST SUMMARY - {last_month_name}</h4>
                    
                    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 24px; color: #F44336; font-weight: bold; margin-bottom: 5px;">{avg_ratio:.1f}%</div>
                            <div style="font-size: 12px; color: #666;">Avg PO/Rofo</div>
                            <div style="font-size: 10px; color: #999;">Target: 80-120%</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #2E7D32; font-weight: bold; margin-bottom: 5px;">{total_forecast:,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Total Rofo</div>
                            <div style="font-size: 10px; color: #999;">Forecast Qty</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #1565C0; font-weight: bold; margin-bottom: 5px;">{total_po:,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Total PO</div>
                            <div style="font-size: 10px; color: #999;">Purchase Order</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 24px; color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: bold; margin-bottom: 5px;">{selisih_qty:+,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Selisih Qty</div>
                            <div style="font-size: 11px; color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: 600;">({selisih_persen:+.1f}%)</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #FF9800; font-weight: bold; margin-bottom: 5px;">{po_rofo_pct:.1f}%</div>
                            <div style="font-size: 12px; color: #666;">PO/Rofo %</div>
                            <div style="font-size: 10px; color: #999;">Overall Ratio</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(244, 67, 54, 0.3); font-size: 14px; color: #666;">
                        <strong>Total UNDER Forecast SKUs: {len(under_skus_df)}</strong> | 
                        <span style="color: #F44336;">Avg PO/Rofo: {avg_ratio:.1f}%</span> | 
                        <span style="color: #2E7D32;">Rofo: {total_forecast:,.0f}</span> | 
                        <span style="color: #1565C0;">PO: {total_po:,.0f}</span> | 
                        <span style="color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: bold;">Selisih: {selisih_qty:+,.0f} ({selisih_persen:+.1f}%)</span>
                    </div>
                </div>
                """
                
                # Tampilkan dengan st.html()
                st.html(html_content)
            else:
                st.success(f"✅ No SKUs with UNDER forecast in {last_month_name}")
        
        with eval_tab2:
            over_skus_df = last_month_data['over_skus']
            if not over_skus_df.empty:
                # Add inventory data
                if 'inventory_df' in inventory_metrics:
                    inventory_data = inventory_metrics['inventory_df'][['SKU_ID', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']]
                    over_skus_df = pd.merge(over_skus_df, inventory_data, on='SKU_ID', how='left')
                
                # TAMBAH: Get last 3 months sales data
                sales_cols_last_3 = []
                if not df_sales.empty:
                    # Get last 3 months from sales data
                    sales_months = sorted(df_sales['Month'].unique())
                    if len(sales_months) >= 3:
                        last_3_sales_months = sales_months[-3:]
                        
                        # Create pivot for last 3 months sales
                        try:
                            sales_pivot = df_sales[df_sales['Month'].isin(last_3_sales_months)].pivot_table(
                                index='SKU_ID',
                                columns='Month',
                                values='Sales_Qty',
                                aggfunc='sum',
                                fill_value=0
                            ).reset_index()
                            
                            # Rename columns to month names
                            month_rename = {}
                            for col in sales_pivot.columns:
                                if isinstance(col, datetime):
                                    month_rename[col] = col.strftime('%b-%Y')
                            sales_pivot = sales_pivot.rename(columns=month_rename)
                            
                            # Merge with over_skus_df
                            over_skus_df = pd.merge(
                                over_skus_df,
                                sales_pivot,
                                on='SKU_ID',
                                how='left'
                            )
                            
                            # Get the sales column names
                            sales_cols_last_3 = [col for col in sales_pivot.columns if isinstance(col, str) and '-' in col]
                            sales_cols_last_3 = sorted(sales_cols_last_3[-3:])  # Get last 3 months
                            
                        except Exception as e:
                            st.warning(f"Tidak bisa menambahkan data sales 3 bulan terakhir: {str(e)}")
                
                # Prepare display columns - TAMBAH sales columns
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Accuracy_Status',
                              'Forecast_Qty', 'PO_Qty', 'PO_Rofo_Ratio', 
                              'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
                
                # Tambah sales columns jika ada
                display_cols.extend(sales_cols_last_3)
                
                # Filter available columns
                available_cols = [col for col in display_cols if col in over_skus_df.columns]
                
                # Pastikan Product_Name selalu ada
                if 'Product_Name' not in available_cols and 'Product_Name' in over_skus_df.columns:
                    available_cols.insert(1, 'Product_Name')
                
                # Format the dataframe
                display_df = over_skus_df[available_cols].copy()
                
                # Add formatted columns
                if 'PO_Rofo_Ratio' in display_df.columns:
                    display_df['PO_Rofo_Ratio'] = display_df['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%")
                
                if 'Cover_Months' in display_df.columns:
                    display_df['Cover_Months'] = display_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if x < 999 else "N/A")
                
                if 'Avg_Monthly_Sales_3M' in display_df.columns:
                    display_df['Avg_Monthly_Sales_3M'] = display_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
                
                # Format sales columns
                for col in sales_cols_last_3:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
                
                # Rename columns for display
                column_names = {
                    'SKU_ID': 'SKU ID',
                    'Product_Name': 'Product Name',
                    'Brand': 'Brand',
                    'SKU_Tier': 'Tier',
                    'Accuracy_Status': 'Status',
                    'Forecast_Qty': 'Forecast Qty',
                    'PO_Qty': 'PO Qty',
                    'PO_Rofo_Ratio': 'PO/Rofo %',
                    'Stock_Qty': 'Stock Available',
                    'Avg_Monthly_Sales_3M': 'Avg Sales (3M)',
                    'Cover_Months': 'Cover (Months)'
                }
                
                # Add sales columns to rename dict
                for col in sales_cols_last_3:
                    column_names[col] = col
                
                display_df = display_df.rename(columns=column_names)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500
                )
                
                # Summary dengan HIGHLIGHT
                total_forecast = over_skus_df['Forecast_Qty'].sum()
                total_po = over_skus_df['PO_Qty'].sum()
                avg_ratio = over_skus_df['PO_Rofo_Ratio'].mean()
                selisih_qty = total_po - total_forecast
                selisih_persen = (selisih_qty / total_forecast * 100) if total_forecast > 0 else 0
                po_rofo_pct = (total_po / total_forecast * 100) if total_forecast > 0 else 0
                
                # Buat HTML content untuk OVER
                html_content_over = f"""
                <div style="background: #FFF3E0; border-left: 5px solid #FF9800; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h4 style="color: #EF6C00; margin-top: 0;">📈 OVER FORECAST SUMMARY - {last_month_name}</h4>
                    
                    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 24px; color: #FF9800; font-weight: bold; margin-bottom: 5px;">{avg_ratio:.1f}%</div>
                            <div style="font-size: 12px; color: #666;">Avg PO/Rofo</div>
                            <div style="font-size: 10px; color: #999;">Target: 80-120%</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #2E7D32; font-weight: bold; margin-bottom: 5px;">{total_forecast:,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Total Rofo</div>
                            <div style="font-size: 10px; color: #999;">Forecast Qty</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #1565C0; font-weight: bold; margin-bottom: 5px;">{total_po:,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Total PO</div>
                            <div style="font-size: 10px; color: #999;">Purchase Order</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 24px; color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: bold; margin-bottom: 5px;">{selisih_qty:+,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Selisih Qty</div>
                            <div style="font-size: 11px; color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: 600;">({selisih_persen:+.1f}%)</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #FF9800; font-weight: bold; margin-bottom: 5px;">{po_rofo_pct:.1f}%</div>
                            <div style="font-size: 12px; color: #666;">PO/Rofo %</div>
                            <div style="font-size: 10px; color: #999;">Overall Ratio</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255, 152, 0, 0.3); font-size: 14px; color: #666;">
                        <strong>Total OVER Forecast SKUs: {len(over_skus_df)}</strong> | 
                        <span style="color: #FF9800;">Avg PO/Rofo: {avg_ratio:.1f}%</span> | 
                        <span style="color: #2E7D32;">Rofo: {total_forecast:,.0f}</span> | 
                        <span style="color: #1565C0;">PO: {total_po:,.0f}</span> | 
                        <span style="color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: bold;">Selisih: {selisih_qty:+,.0f} ({selisih_persen:+.1f}%)</span>
                    </div>
                </div>
                """
                
                # Tampilkan dengan st.html()
                st.html(html_content_over)
            else:
                st.success(f"✅ No SKUs with OVER forecast in {last_month_name}")

st.divider()

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📈 Monthly Performance Details",
    "🏷️ Forecast Performance by Brand & Tier Analysis",
    "📦 Inventory Analysis",
    "🔍 SKU Evaluation",
    "📈 Sales & Forecast Analysis",
    "🛒 Ecommerce Forecast",  
    "💰 Profitability Analysis",
    "🤝 Reseller Forecast",
    "🚚 Fulfillment Cost Analysis",
    "📋 Data Explorer" 
])

# --- TAB 1: MONTHLY PERFORMANCE DETAILS (PREMIUM HEATMAP) ---
with tab1:
    st.subheader("📅 Monthly Performance Details")
    
    if monthly_performance:
        # 1. Prepare Data
        summary_data = []
        prev_accuracy = 0
        
        for i, (month, data) in enumerate(sorted(monthly_performance.items())):
            # Tentukan Status & Icon
            acc = data['accuracy']
            if acc >= 90:
                status_icon = "🌟 Excellent"
            elif acc >= 80:
                status_icon = "✅ Good"
            elif acc >= 70:
                status_icon = "⚠️ Fair"
            else:
                status_icon = "🛑 Poor"
            
            # Hitung MoM Change (Delta)
            delta = acc - prev_accuracy if i > 0 else 0
            delta_str = f"{delta:+.1f}%" if i > 0 else "-"
            prev_accuracy = acc

            summary_data.append({
                'Month_Raw': month,
                'Month': month.strftime('%b %Y'),
                'Status': status_icon,
                'Accuracy': acc,
                'MoM': delta_str, # Month over Month Change
                'Under': data['status_counts'].get('Under', 0),
                'Accurate': data['status_counts'].get('Accurate', 0),
                'Over': data['status_counts'].get('Over', 0),
                'Total SKUs': data['total_records'],
                'MAPE': data['mape']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 2. Styling dengan Pandas Styler (Soft Pastel Heatmap)
        def highlight_accuracy(val):
            color = '#d1fae5' if val >= 80 else '#fef3c7' if val >= 70 else '#fee2e2'
            return f'background-color: {color}; color: #374151; font-weight: bold;'

        # Create Styler Object (MENGGUNAKAN .map() BUKAN .applymap() AGAR TIDAK ERROR)
        styler = summary_df.style\
            .background_gradient(subset=['Under'], cmap='Reds', vmin=0, vmax=summary_df['Under'].max()*1.5)\
            .background_gradient(subset=['Accurate'], cmap='Greens', vmin=0, vmax=summary_df['Accurate'].max())\
            .background_gradient(subset=['Over'], cmap='Oranges', vmin=0, vmax=summary_df['Over'].max()*1.5)\
            .map(highlight_accuracy, subset=['Accuracy'])\
            .format({
                'Accuracy': '{:.1f}%',
                'MAPE': '{:.1f}%',
                'Under': '{:,}',
                'Accurate': '{:,}',
                'Over': '{:,}',
                'Total SKUs': '{:,}'
            })

        # 3. Render Dataframe
        st.dataframe(
            styler,
            column_order=['Month', 'Status', 'Accuracy', 'MoM', 'Under', 'Accurate', 'Over', 'Total SKUs', 'MAPE'],
            column_config={
                "Month": st.column_config.TextColumn("Period", help="Month of analysis"),
                "Status": st.column_config.TextColumn("Health Score"),
                "Accuracy": st.column_config.ProgressColumn(
                    "Accuracy %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "MoM": st.column_config.TextColumn("Trend (MoM)", help="Change from previous month"),
                "Under": st.column_config.NumberColumn("📉 Under", help="Count of Under-forecast SKUs"),
                "Accurate": st.column_config.NumberColumn("🎯 Accurate", help="Count of Accurate SKUs"),
                "Over": st.column_config.NumberColumn("📈 Over", help="Count of Over-forecast SKUs"),
                "Total SKUs": st.column_config.NumberColumn("Total Items"),
                "MAPE": st.column_config.NumberColumn("MAPE", help="Mean Absolute Percentage Error")
            },
            use_container_width=True,
            height=500,
            hide_index=True
        )
        
        # Legend Kecil
        st.caption("""
        🎨 **Color Legend:** - **Accuracy:** 🟩 Hijau (>80%), 🟨 Kuning (70-80%), 🟥 Merah (<70%).
        - **Under/Over:** Semakin pekat warnanya, semakin banyak SKU yang bermasalah di kategori tersebut.
        """)
        
        # ==============================================================================
        # 3. PREMIUM FORECAST BIAS ANALYSIS (Diverging Chart Version)
        # ==============================================================================
        if not forecast_bias.empty:
            st.divider()
            st.subheader("🎯 Forecast Bias & Health Analysis")
            
            # 1. Hitung Metrics Utama
            avg_bias_val = forecast_bias['Avg_Bias_Percentage'].mean()
            
            # Tentukan Tendency (Kecenderungan)
            if avg_bias_val > 5:
                tendency = "UNDER-FORECASTING (Demand > Plan)"
                tendency_icon = "📉" 
                tendency_color = "#3949ab" 
                risk_msg = "Risk: Potential Lost Sales (Stockout)"
            elif avg_bias_val < -5:
                tendency = "OVER-FORECASTING (Demand < Plan)"
                tendency_icon = "📈" 
                tendency_color = "#ef5350" 
                risk_msg = "Risk: Excess Stock & Obsolescence"
            else:
                tendency = "BALANCED (Good Accuracy)"
                tendency_icon = "⚖️"
                tendency_color = "#26a69a" 
                risk_msg = "Status: Healthy Forecast"

            # Hitung Jumlah Bulan Warning
            critical_months = len(forecast_bias[abs(forecast_bias['Avg_Bias_Percentage']) > 20])
            
            # --- 2. BIAS HEALTH CARDS (CSS) ---
            st.markdown(f"""
            <style>
                .bias-card {{
                    background-color: white;
                    border-radius: 12px;
                    padding: 1.2rem;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                    border-left: 5px solid {tendency_color};
                    height: 100%;
                }}
                .bias-label {{ font-size: 0.8rem; color: #888; font-weight: 600; text-transform: uppercase; }}
                .bias-val {{ font-size: 1.8rem; font-weight: 800; color: #333; margin: 5px 0; }}
                .bias-sub {{ font-size: 0.9rem; color: {tendency_color}; font-weight: 600; }}
                .bias-desc {{ font-size: 0.8rem; color: #666; margin-top: 5px; }}
            </style>
            """, unsafe_allow_html=True)
            
            bc1, bc2, bc3 = st.columns(3)
            
            with bc1:
                st.markdown(f"""
                <div class="bias-card">
                    <div class="bias-label">AVERAGE BIAS (YTD)</div>
                    <div class="bias-val">{avg_bias_val:+.1f}%</div>
                    <div class="bias-sub">{tendency_icon} {tendency}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with bc2:
                st.markdown(f"""
                <div class="bias-card" style="border-left-color: #ffa726;">
                    <div class="bias-label">IMPACT ANALYSIS</div>
                    <div class="bias-val" style="font-size: 1.2rem; margin-top: 15px;">{risk_msg}</div>
                    <div class="bias-desc">Based on average deviation direction</div>
                </div>
                """, unsafe_allow_html=True)
            
            with bc3:
                status_color = "#ef5350" if critical_months > 0 else "#26a69a"
                st.markdown(f"""
                <div class="bias-card" style="border-left-color: {status_color};">
                    <div class="bias-label">VOLATILITY CHECK</div>
                    <div class="bias-val">{critical_months} <span style="font-size:1rem;">Months</span></div>
                    <div class="bias-desc">Months with >20% Deviation (Critical)</div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- 3. DIVERGING BAR CHART (Visualisasi Bias PREMIUM) ---
            st.write("") # Spacer
            
            colors = []
            for val in forecast_bias['Avg_Bias_Percentage']:
                if abs(val) <= 10:
                    colors.append('#4db6ac') # Soft Teal (Aman)
                elif abs(val) <= 20:
                    colors.append('#ffb74d') # Soft Orange (Warning)
                else:
                    colors.append('#ef5350') # Soft Red (Critical)

            fig_bias = go.Figure()

            fig_bias.add_trace(go.Bar(
                x=forecast_bias['Month'].dt.strftime('%b-%Y'),
                y=forecast_bias['Avg_Bias_Percentage'],
                text=[f"{x:+.1f}%" for x in forecast_bias['Avg_Bias_Percentage']],
                textposition='auto',
                textfont=dict(color='white', weight='bold'), 
                marker_color=colors,
                name='Bias %',
                hovertemplate='<b>%{x}</b><br>Bias: %{y:+.1f}%<extra></extra>'
            ))

            fig_bias.add_hrect(y0=-10, y1=10, fillcolor="green", opacity=0.05, line_width=0, annotation_text="🟢 SAFE ZONE (±10%)", annotation_position="top left", annotation_font_color="green")
            fig_bias.add_hrect(y0=-20, y1=-10, fillcolor="yellow", opacity=0.05, line_width=0)
            fig_bias.add_hrect(y0=10, y1=20, fillcolor="yellow", opacity=0.05, line_width=0)
            
            max_bias = max(forecast_bias['Avg_Bias_Percentage'].max(), 25)
            min_bias = min(forecast_bias['Avg_Bias_Percentage'].min(), -25)
            
            fig_bias.add_annotation(
                x=0.02, y=0.95, xref="paper", yref="paper",
                text="⬆️ UNDER FORECAST<br><span style='font-size:10px'>(Risiko Stockout / Lost Sales)</span>",
                showarrow=False, font=dict(color="#ef5350", size=12, weight="bold"), align="left"
            )
            fig_bias.add_annotation(
                x=0.02, y=0.05, xref="paper", yref="paper",
                text="⬇️ OVER FORECAST<br><span style='font-size:10px'>(Risiko Dead Stock / Overstock)</span>",
                showarrow=False, font=dict(color="#ef5350", size=12, weight="bold"), align="left"
            )

            fig_bias.update_layout(
                title="<b>📉 Monthly Forecast Bias Trend & Health Diagnostics</b>",
                yaxis_title="Bias Percentage (%)",
                xaxis_title="Period",
                height=450,
                hovermode="x unified",
                yaxis=dict(
                    zeroline=True, zerolinewidth=3, zerolinecolor='rgba(0,0,0,0.5)', 
                    range=[min_bias*1.2, max_bias*1.2] 
                ), 
                plot_bgcolor='white',
                margin=dict(t=50, b=20, l=20, r=20)
            )

            st.plotly_chart(fig_bias, use_container_width=True)
            
            st.caption("""
            ℹ️ **Cara Membaca:**
            - **Bar ke Atas (+):** Realisasi (PO) > Forecast. Artinya **Under-Forecast** (Kurang plan, potensi lost sales).
            - **Bar ke Bawah (-):** Realisasi (PO) < Forecast. Artinya **Over-Forecast** (Plan ketinggian, potensi overstock).
            - **Zona Hijau:** Bias ±10% dianggap sehat.
            """)
    else:
        st.warning("⚠️ No monthly performance data available.")
        
# --- TAB 2: FORECAST PERFORMANCE BY BRAND & TIER ANALYSIS (FINAL FIXED) ---
with tab2:
    st.subheader("🏷️ Brand & Tier Strategic Analysis")
    st.caption("Portfolio Management: Brand Performance Positioning & Tier Health")

    # ==============================================================================
    # 1. DATA PREPARATION (Last Month vs Last 12 Months)
    # ==============================================================================
    
    # A. Tentukan Periode
    all_months = sorted(monthly_performance.keys()) if monthly_performance else []
    
    if not all_months:
        st.warning("⚠️ Belum ada data performa bulanan.")
        st.stop()

    last_month_date = all_months[-1]
    last_12_months_list = all_months[-12:] if len(all_months) >= 12 else all_months

    # B. Helper untuk Hitung Brand Performance berdasarkan List Bulan
    def get_brand_perf_by_period(months_target, label):
        # Filter Dataframes
        df_f_filtered = df_forecast[df_forecast['Month'].isin(months_target)].copy()
        df_p_filtered = df_po[df_po['Month'].isin(months_target)].copy()
        
        # Merge Basic Info
        df_f_filtered = add_product_info_to_data(df_f_filtered, df_product)
        df_p_filtered = add_product_info_to_data(df_p_filtered, df_product)
        
        # Group by Brand
        f_group = df_f_filtered.groupby('Brand')['Forecast_Qty'].sum().reset_index()
        p_group = df_p_filtered.groupby('Brand')['PO_Qty'].sum().reset_index()
        
        # Merge Forecast & PO
        merged = pd.merge(f_group, p_group, on='Brand', how='outer').fillna(0)
        
        # Hitung SKU Count (ambil dari forecast data active)
        sku_count = df_f_filtered.groupby('Brand')['SKU_ID'].nunique().reset_index(name='SKU_Count')
        merged = pd.merge(merged, sku_count, on='Brand', how='left').fillna(0)
        
        # ---------------------------------------------------------
        # PERBAIKAN LOGIKA AKURASI: HANYA HITUNG JIKA ROFO > 0
        # ---------------------------------------------------------
        sku_level = pd.merge(
            df_f_filtered.groupby(['Brand', 'SKU_ID'])['Forecast_Qty'].sum().reset_index(),
            df_p_filtered.groupby(['Brand', 'SKU_ID'])['PO_Qty'].sum().reset_index(),
            on=['Brand', 'SKU_ID'], how='outer'
        ).fillna(0)
        
        # FILTER: Buang SKU yang Rofo-nya 0 dari perhitungan akurasi
        valid_sku_level = sku_level[sku_level['Forecast_Qty'] > 0].copy()
        
        if not valid_sku_level.empty:
            # Hitung akurasi
            valid_sku_level['Accuracy'] = valid_sku_level.apply(
                lambda x: 100 - abs((x['PO_Qty']/x['Forecast_Qty']*100)-100), axis=1
            )
            # Opsional: Jika PO over sangat jauh (misal 300%), akurasi bisa minus. Kita batasi minimal 0%.
            valid_sku_level['Accuracy'] = valid_sku_level['Accuracy'].clip(lower=0)
            
            # Rata-rata akurasi per Brand (hanya dari SKU yang valid)
            brand_acc = valid_sku_level.groupby('Brand')['Accuracy'].mean().reset_index()
        else:
            brand_acc = pd.DataFrame(columns=['Brand', 'Accuracy'])
        
        # Gabungkan hasil akurasi ke dataframe utama
        final_df = pd.merge(merged, brand_acc, on='Brand', how='left')
        
        # Jika ada brand yang isinya Rofo 0 semua, akurasinya di-set 0 (atau bisa di-set None)
        final_df['Accuracy'] = final_df['Accuracy'].fillna(0)
        
        return final_df

    # C. UI Selector Period
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        view_period = st.radio(
            "📅 Pilih Periode Analisis:",
            ["Bulan Terakhir", "1 Tahun Terakhir (L12M)"],
            horizontal=False
        )

    # D. Generate Data sesuai Pilihan
    if view_period == "Bulan Terakhir":
        active_df = get_brand_perf_by_period([last_month_date], "Last Month")
        period_label = last_month_date.strftime('%B %Y')
    else:
        active_df = get_brand_perf_by_period(last_12_months_list, "Last 12M")
        period_label = f"Last 12 Months ({len(last_12_months_list)} periods)"

    st.markdown(f"#### 📊 Analisis Periode: {period_label}")

    if not active_df.empty:
        # ==============================================================================
        # 2. BRAND KPI CARDS (Soft Gradient - Fixed HTML)
        # ==============================================================================
        
        best_brand = active_df.loc[active_df['Accuracy'].idxmax()]
        high_vol_brand = active_df.loc[active_df['Forecast_Qty'].idxmax()]
        most_sku_brand = active_df.loc[active_df['SKU_Count'].idxmax()]
        
        # Calculate Weighted Accuracy Portfolio
        total_vol = active_df['Forecast_Qty'].sum()
        # Weighted avg accuracy
        weighted_acc = (active_df['Accuracy'] * active_df['Forecast_Qty']).sum() / total_vol if total_vol > 0 else 0

        # CSS TANPA INDENTASI
        st.markdown("""
<style>
.b-card {
border-radius: 12px;
padding: 1.2rem;
color: white;
box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
position: relative;
overflow: hidden;
transition: transform 0.3s ease;
}
.b-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1); }
.b-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; margin-bottom: 5px; }
.b-val { font-size: 1.4rem; font-weight: 800; margin-bottom: 5px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.b-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; display: flex; align-items: center; gap: 5px; }
.b-badge { background: rgba(255,255,255,0.25); padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px); }
</style>
""", unsafe_allow_html=True)

        def render_brand_card_fixed(label, brand_name, metric_val, metric_label, gradient):
            return f"""
<div class="b-card" style="background: {gradient};">
<div class="b-label">{label}</div>
<div class="b-val">{brand_name}</div>
<div class="b-sub">
<span class="b-badge">{metric_val}</span> {metric_label}
</div>
</div>
"""

        bc1, bc2, bc3, bc4 = st.columns(4)
        
        with bc1:
            st.markdown(render_brand_card_fixed(
                "🏆 Best Accuracy", best_brand['Brand'], f"{best_brand['Accuracy']:.1f}%", "Avg Accuracy",
                "linear-gradient(135deg, #10B981 0%, #059669 100%)"
            ), unsafe_allow_html=True)
            
        with bc2:
            st.markdown(render_brand_card_fixed(
                "📦 Highest Volume", high_vol_brand['Brand'], f"{high_vol_brand['Forecast_Qty']:,.0f}", "Units Fcst",
                "linear-gradient(135deg, #6366F1 0%, #4338CA 100%)"
            ), unsafe_allow_html=True)
            
        with bc3:
            st.markdown(render_brand_card_fixed(
                "🗂️ Most SKUs", most_sku_brand['Brand'], f"{most_sku_brand['SKU_Count']}", "Active Items",
                "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)"
            ), unsafe_allow_html=True)
            
        with bc4:
            st.markdown(render_brand_card_fixed(
                "⚖️ Portfolio Health", "All Brands", f"{weighted_acc:.1f}%", "Vol. Wgt. Acc",
                "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)"
            ), unsafe_allow_html=True)

        # ==============================================================================
        # 3. STRATEGIC MAGIC QUADRANT (SCATTER PLOT) - PREMIUM EDITION
        # ==============================================================================
        st.write("")
        st.subheader("🎯 Strategic Brand Positioning (S&OP Portfolio Matrix)")
        st.caption("Membagi brand ke dalam 4 kuadran strategis berdasarkan kontribusi Volume dan keandalan Akurasi.")
        
        # Hapus col_quad1, col_quad2, kita buat grafiknya full width (lebar penuh)
        scatter_data = active_df.copy()
        # Hindari error log-scale jika ada volume 0
        scatter_data = scatter_data[scatter_data['Forecast_Qty'] > 0]
        
        median_vol = scatter_data['Forecast_Qty'].median()
        target_acc = 80
        
        fig_quad = px.scatter(
            scatter_data,
            x='Forecast_Qty',
            y='Accuracy',
            size='SKU_Count',
            color='Accuracy',
            text='Brand',
            color_continuous_scale='RdYlGn',
            size_max=50,
            custom_data=['Brand', 'SKU_Count', 'Forecast_Qty', 'Accuracy']
        )
        
        # Quadrant Zones Background
        max_vol = scatter_data['Forecast_Qty'].max() * 1.5
        fig_quad.add_shape(type="rect",
            x0=median_vol, y0=0, x1=max_vol, y1=target_acc,
            fillcolor="rgba(239, 68, 68, 0.08)", line_width=0, layer="below" # Red tint
        )
        fig_quad.add_shape(type="rect",
            x0=median_vol, y0=target_acc, x1=max_vol, y1=110,
            fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, layer="below" # Green tint
        )
        
        # Garis Pembatas (Crosshair)
        fig_quad.add_hline(y=target_acc, line_dash="dash", line_color="gray", annotation_text="Target Accuracy (80%)")
        fig_quad.add_vline(x=median_vol, line_dash="dash", line_color="gray", annotation_text="Median Volume")

        # --- FITUR PRO: WATERMARK LABEL LANGSUNG DI DALAM GRAFIK ---
        fig_quad.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper", text="🌟 STARS<br><span style='font-size:12px'>High Vol, High Acc</span>", showarrow=False, font=dict(size=18, color="#10B981", weight="bold"), opacity=0.3, align="right")
        fig_quad.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text="🚨 RISK (PRIORITY FIX)<br><span style='font-size:12px'>High Vol, Low Acc</span>", showarrow=False, font=dict(size=18, color="#EF4444", weight="bold"), opacity=0.3, align="right")
        fig_quad.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text="❓ NICHE / QUESTION<br><span style='font-size:12px'>Low Vol, High Acc</span>", showarrow=False, font=dict(size=18, color="#F59E0B", weight="bold"), opacity=0.3, align="left")
        fig_quad.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper", text="💤 SLEEPERS<br><span style='font-size:12px'>Low Vol, Low Acc</span>", showarrow=False, font=dict(size=18, color="#6B7280", weight="bold"), opacity=0.3, align="left")

        fig_quad.update_traces(
            textposition='top center',
            textfont=dict(weight='bold', color='#1F2937'),
            hovertemplate="<b>%{customdata[0]}</b><br>Accuracy: %{y:.1f}%<br>Volume: %{x:,.0f}<br>SKUs: %{marker.size}<extra></extra>"
        )
        
        fig_quad.update_layout(
            height=550, # Dibuat lebih tinggi agar lapang
            xaxis_title="Forecast Volume (Log Scale) ➡️",
            yaxis_title="Accuracy (%) ➡️",
            xaxis_type="log", 
            yaxis_range=[max(0, scatter_data['Accuracy'].min()-10), min(105, scatter_data['Accuracy'].max()+10)], # Dinamis
            plot_bgcolor="white",
            coloraxis_showscale=False, # Sembunyikan colorbar legend karena sudah ada zona warna
            margin=dict(t=20, l=20, r=20, b=20)
        )
        st.plotly_chart(fig_quad, use_container_width=True)

        # ==============================================================================
        # 4. BRAND PERFORMANCE COMBO CHART (BAR + LINE) - DYNAMIC COLORS
        # ==============================================================================
        st.divider()
        st.subheader("📊 Brand Detail: Volume vs Accuracy")
        
        # Sort by Volume
        chart_df = active_df.sort_values('Forecast_Qty', ascending=False)

        fig_combo = go.Figure()

        # Bar: Volume
        fig_combo.add_trace(go.Bar(
            x=chart_df['Brand'],
            y=chart_df['Forecast_Qty'],
            name='Forecast Volume',
            marker_color='rgba(99, 102, 241, 0.2)', # Indigo Transparan elegan
            marker_line_color='rgba(99, 102, 241, 0.8)',
            marker_line_width=1.5,
            yaxis='y1'
        ))

        # Dynamic Marker Colors untuk Line Chart
        line_colors = ['#10B981' if acc >= 80 else '#F59E0B' if acc >= 70 else '#EF4444' for acc in chart_df['Accuracy']]

        # Line: Accuracy
        fig_combo.add_trace(go.Scatter(
            x=chart_df['Brand'],
            y=chart_df['Accuracy'],
            name='Accuracy %',
            mode='lines+markers+text',
            text=[f"{acc:.1f}%" for acc in chart_df['Accuracy']],
            textposition="top center",
            textfont=dict(color='#1F2937', size=10, weight='bold'),
            line=dict(color='#4B5563', width=2), # Garis abu-abu gelap
            marker=dict(size=12, color=line_colors, line=dict(width=2, color='white')), # Titik warna-warni
            yaxis='y2'
        ))

        fig_combo.update_layout(
            height=450,
            xaxis_title="Brand (Sorted by Volume)",
            yaxis=dict(
                title="Forecast Volume",
                showgrid=False
            ),
            yaxis2=dict(
                title="Accuracy (%)",
                overlaying='y',
                side='right',
                range=[0, 120], # Ruang ekstra untuk label teks
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)'
            ),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            plot_bgcolor='white',
            margin=dict(t=40, b=0)
        )
        
        # Tambah garis target akurasi
        fig_combo.add_hline(y=80, line_dash="dot", line_color="#EF4444", yref="y2", annotation_text="Target 80%")

        st.plotly_chart(fig_combo, use_container_width=True)

        # ==============================================================================
        # 5. TIER ANALYSIS REPLACEMENT (STACKED BAR) - NEW!
        # ==============================================================================
        st.divider()
        c_tier1, c_tier2 = st.columns(2)
        
        with c_tier1:
            st.subheader("🧬 Tier & Brand Composition")
            st.caption("Komposisi Jumlah SKU berdasarkan Tier")
            
            # Sunburst tetap dipertahankan karena bagus untuk hierarki
            if not df_product.empty and 'SKU_Tier' in df_product.columns:
                sunburst_data = df_product[df_product['Status'].str.upper() == 'ACTIVE'].groupby(['SKU_Tier', 'Brand']).size().reset_index(name='Count')
                fig_sun = px.sunburst(
                    sunburst_data,
                    path=['SKU_Tier', 'Brand'],
                    values='Count',
                    color='SKU_Tier',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_sun.update_layout(height=400, margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(fig_sun, use_container_width=True)

        with c_tier2:
            st.subheader("🏆 Tier Performance (Bar Chart)")
            st.caption(f"Rata-rata Akurasi per Tier ({period_label})")
            
            # Hitung data per Tier untuk periode terpilih
            # Kita perlu re-calculate karena active_df tadi per Brand
            
            # 1. Filter raw data lagi berdasarkan periode
            df_f_tier = df_forecast[df_forecast['Month'].isin(last_12_months_list if view_period != "Bulan Terakhir" else [last_month_date])].copy()
            df_p_tier = df_po[df_po['Month'].isin(last_12_months_list if view_period != "Bulan Terakhir" else [last_month_date])].copy()
            
            # 2. Add Tier Info
            df_f_tier = add_product_info_to_data(df_f_tier, df_product)
            df_p_tier = add_product_info_to_data(df_p_tier, df_product)
            
            # 3. Group by Tier
            tier_stats = pd.merge(
                df_f_tier.groupby(['SKU_Tier', 'SKU_ID'])['Forecast_Qty'].sum().reset_index(),
                df_p_tier.groupby(['SKU_Tier', 'SKU_ID'])['PO_Qty'].sum().reset_index(),
                on=['SKU_Tier', 'SKU_ID'], how='outer'
            ).fillna(0)
            
            # Hitung akurasi per SKU
            tier_stats['Accuracy'] = tier_stats.apply(
                lambda x: 100 - abs((x['PO_Qty']/x['Forecast_Qty']*100)-100) if x['Forecast_Qty'] > 0 else 0, axis=1
            )
            
            # Average per Tier
            tier_summary = tier_stats.groupby('SKU_Tier')['Accuracy'].mean().reset_index()
            tier_summary = tier_summary.sort_values('Accuracy', ascending=False)
            
            # Bar Chart Horizontal
            fig_bar_tier = go.Figure()
            fig_bar_tier.add_trace(go.Bar(
                y=tier_summary['SKU_Tier'],
                x=tier_summary['Accuracy'],
                orientation='h',
                marker_color='#10B981', # Emerald
                text=[f"{x:.1f}%" for x in tier_summary['Accuracy']],
                textposition='auto'
            ))
            
            # Add Target Line
            fig_bar_tier.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="Target 80%")
            
            fig_bar_tier.update_layout(
                height=400,
                title="Average Accuracy by Tier",
                xaxis_title="Accuracy %",
                yaxis_title="Tier",
                xaxis_range=[0, 110],
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_bar_tier, use_container_width=True)

    else:
        st.info("📊 Data tidak tersedia untuk analisis brand.")

# --- TAB 3: INVENTORY HEALTH, COVERAGE & AGING (FIXED VALUE) ---
with tab3:
    st.subheader("📦 Inventory Health & Optimization Dashboard")
    st.caption("Comprehensive Stock Analysis: Value, Coverage, Warehouse Capacity, and Aging Profile")

    # ==============================================================================
    # 0. SIDEBAR INPUT FOR THIS TAB
    # ==============================================================================
    with st.expander("⚙️ Warehouse Settings", expanded=False):
        WH_CAPACITY = st.number_input(
            "🏢 Total Warehouse Capacity (pcs)",
            min_value=1000, max_value=10000000, value=250000, step=10000,
            help="Kapasitas maksimal gudang dalam satuan pcs/unit"
        )

    # ==============================================================================
    # 1. DATA PREPARATION & CLEANING (ROBUST FIX)
    # ==============================================================================
    if not df_stock.empty:
        df_batch = df_stock.copy()
        
        # 1.1. Standardize Category Column
        col_cat = 'Stock_Category'
        if col_cat not in df_batch.columns:
            candidates = [c for c in df_batch.columns if 'cat' in c.lower() or 'kategori' in c.lower()]
            col_cat = candidates[0] if candidates else None
        
        if col_cat:
            df_batch = df_batch.rename(columns={col_cat: 'Stock_Category'})
            
        # Clean Category & Qty
        if 'Stock_Category' in df_batch.columns:
            df_batch['Stock_Category'] = df_batch['Stock_Category'].astype(str).str.strip()
        else:
            df_batch['Stock_Category'] = 'Uncategorized'
            
        df_batch['Stock_Qty'] = pd.to_numeric(df_batch['Stock_Qty'], errors='coerce').fillna(0)
        df_batch = df_batch[df_batch['Stock_Qty'] > 0] # Filter stok > 0

        # 1.2. MERGE PRODUCT INFO (THE FIX IS HERE)
        # Hapus dulu kolom info produk yang mungkin menempel di df_stock (biar tidak double _x _y)
        cols_to_drop = ['Product_Name', 'Brand', 'Status', 'Floor_Price', 'SKU_Tier', 'Net_Order_Price']
        df_batch = df_batch.drop(columns=[c for c in cols_to_drop if c in df_batch.columns], errors='ignore')

        # Merge fresh dari Product Master
        if not df_product.empty:
            # Pastikan kolom-kolom ini ada di df_product
            master_cols = ['SKU_ID'] + [c for c in cols_to_drop if c in df_product.columns]
            
            # Merge
            df_batch = pd.merge(df_batch, df_product[master_cols], on='SKU_ID', how='left')
            
            # Fill missing text
            for txt_col in ['Status', 'Product_Name', 'Brand']:
                if txt_col in df_batch.columns:
                    df_batch[txt_col] = df_batch[txt_col].fillna('Unknown')

            # 1.3. CALCULATE VALUE
            if 'Floor_Price' in df_batch.columns:
                df_batch['Floor_Price'] = pd.to_numeric(df_batch['Floor_Price'], errors='coerce').fillna(0)
                df_batch['Total_Value'] = df_batch['Stock_Qty'] * df_batch['Floor_Price']
            else:
                df_batch['Total_Value'] = 0
                st.warning("⚠️ Kolom 'Floor_Price' tidak ditemukan di Product Master. Nilai Aset = 0.")

        # 1.4. EXPIRY LOGIC
        def get_expiry_desc(row):
            expiry_cols = [c for c in row.index if 'expir' in c.lower() or 'ed' in c.lower()]
            if not expiry_cols: return 'Not Defined'
            val = row[expiry_cols[0]]
            if pd.isna(val) or str(val).strip() in ['', '-', 'nan']: return 'Not Defined'
            try:
                exp_date = pd.to_datetime(val, dayfirst=True, errors='coerce')
                if pd.isna(exp_date): return 'Not Defined'
                days = (exp_date - pd.Timestamp.now()).days
                if days < 0: return '❌ EXPIRED'
                elif days <= 30: return '🚨 Critical (<30 Days)'
                elif days <= 90: return '⚠️ NED (1-3 Months)'
                elif days <= 180: return '📅 NED (3-6 Months)'
                elif days <= 365: return '✅ Safe (6-12 Months)'
                else: return '🌟 Fresh (>1 Year)'
            except: return 'Not Defined'

        df_batch['Expiry_Category'] = df_batch.apply(get_expiry_desc, axis=1)

        # 1.5. COVERAGE LOGIC
        df_stock_agg = df_batch.groupby('SKU_ID')['Stock_Qty'].sum().reset_index()
        
        # Get Sales Data
        df_avg_sales = pd.DataFrame()
        if not df_sales.empty:
            months = sorted(df_sales['Month'].unique())
            last_3 = months[-3:] if len(months) >= 3 else months
            df_sales_3m = df_sales[df_sales['Month'].isin(last_3)]
            df_avg_sales = df_sales_3m.groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
            df_avg_sales.rename(columns={'Sales_Qty': 'Avg_Sales'}, inplace=True)
        
        df_cover = pd.merge(df_stock_agg, df_avg_sales, on='SKU_ID', how='left')
        df_cover['Avg_Sales'] = df_cover['Avg_Sales'].fillna(0)
        
        df_cover['Cover_Months'] = np.where(
            df_cover['Avg_Sales'] > 0, 
            df_cover['Stock_Qty'] / df_cover['Avg_Sales'], 
            999
        )
        
        total_global_stock = df_cover['Stock_Qty'].sum()
        total_global_avg_sales = df_cover['Avg_Sales'].sum()
        
        global_cover_months = (total_global_stock / total_global_avg_sales) if total_global_avg_sales > 0 else 0
        
        current_occupancy = df_batch['Stock_Qty'].sum()
        occupancy_pct = (current_occupancy / WH_CAPACITY * 100)

        # ==============================================================================
        # 2. EXECUTIVE KPI CARDS (PASTEL & SMART VALUE)
        # ==============================================================================
        total_val = df_batch['Total_Value'].sum()
        total_sku = df_batch['SKU_ID'].nunique()
        
        risk_mask = df_batch['Expiry_Category'].isin(['❌ EXPIRED', '🚨 Critical (<30 Days)'])
        risk_val = df_batch[risk_mask]['Total_Value'].sum()
        risk_pct = (risk_val / total_val * 100) if total_val > 0 else 0

        # Helper: Format Uang Pintar
        def format_currency_smart(value):
            if value >= 1_000_000_000: return f"Rp {value/1e9:,.1f} M"
            elif value >= 1_000_000: return f"Rp {value/1e6:,.1f} Jt"
            else: return f"Rp {value:,.0f}"

        # Gunakan Master Function dari Sidebar
        val_display = format_rupiah(total_val)
        risk_display = format_rupiah(risk_val)

        # CSS Styles
        st.markdown("""
        <style>
            .inv-card {
                border-radius: 12px; padding: 1.2rem; color: white;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05); transition: transform 0.3s;
                position: relative; overflow: hidden;
            }
            .inv-card:hover { transform: translateY(-3px); }
            .inv-label { font-size: 0.8rem; font-weight: 700; opacity: 0.9; text-transform: uppercase; margin-bottom: 5px; }
            .inv-val { font-size: 1.6rem; font-weight: 800; margin-bottom: 5px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
            .inv-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; }
        </style>
        """, unsafe_allow_html=True)

        def render_inv_card(title, val, sub, bg):
            return f"""
            <div class="inv-card" style="background: {bg};">
                <div class="inv-label">{title}</div>
                <div class="inv-val">{val}</div>
                <div class="inv-sub">{sub}</div>
            </div>
            """

        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            # Soft Indigo
            st.markdown(render_inv_card("Total Asset Value", val_display, f"{total_sku:,} Items", 
                "linear-gradient(135deg, #7986cb 0%, #5c6bc0 100%)"), unsafe_allow_html=True)
        with c2:
            # Soft Teal
            st.markdown(render_inv_card("Total Quantity", f"{current_occupancy:,.0f}", f"{occupancy_pct:.1f}% Capacity", 
                "linear-gradient(135deg, #4db6ac 0%, #26a69a 100%)"), unsafe_allow_html=True)
        with c3:
            # Soft Orange
            st.markdown(render_inv_card("Global Stock Cover", f"{global_cover_months:.1f} Mo", "Total Stock / Total Sales", 
                "linear-gradient(135deg, #ffb74d 0%, #ffa726 100%)"), unsafe_allow_html=True)
        with c4:
            # Soft Red or Green depending on risk
            risk_bg = "linear-gradient(135deg, #ef5350 0%, #e53935 100%)" if risk_pct > 5 else "linear-gradient(135deg, #66bb6a 0%, #43a047 100%)"
            st.markdown(render_inv_card("Expiry Risk Value", risk_display, f"{risk_pct:.1f}% of Total", 
                risk_bg), unsafe_allow_html=True)

        # --- BARIS KEDUA: ADVANCED INVENTORY METRICS ---
        st.write("") # Spacer kecil agar tidak terlalu menempel
        
        # Ekstrak list SKU yang HANYA berstatus 'SKU Regular' (TANPA FILTER ACTIVE)
        regular_skus_only = df_batch[
            df_batch['Stock_Category'].str.contains('Regular', case=False, na=False)
        ]['SKU_ID'].unique()

        # Filter df_cover khusus untuk perhitungan KPI
        df_cover_filtered = df_cover[df_cover['SKU_ID'].isin(regular_skus_only)]
        
        # Kalkulasi Metrik Baru
        inv_turnover = (12 / global_cover_months) if global_cover_months > 0 else 0
        
        # 🔥 UBAH ANGKA 0.3 DI SINI MENJADI 0.1
        stockout_skus = len(df_cover_filtered[df_cover_filtered['Cover_Months'] < 0.2]) 
        
        total_skus_cover = len(df_cover_filtered)
        stockout_rate = (stockout_skus / total_skus_cover * 100) if total_skus_cover > 0 else 0
        replenish_skus = len(df_cover_filtered[df_cover_filtered['Cover_Months'] < 0.8])

        c5, c6, c7 = st.columns(3)
        
        with c5:
            # Soft Purple untuk Turnover
            st.markdown(render_inv_card("Inventory Turnover", f"{inv_turnover:.1f}x", "Annualized Ratio", 
                "linear-gradient(135deg, #ab47bc 0%, #8e24aa 100%)"), unsafe_allow_html=True)
        with c6:
            # Merah Tua jika Stockout Rate > 5%, Hijau jika aman
            so_bg = "linear-gradient(135deg, #e53935 0%, #c62828 100%)" if stockout_rate > 5 else "linear-gradient(135deg, #43a047 0%, #2e7d32 100%)"
            st.markdown(render_inv_card("Stock Out Rate", f"{stockout_rate:.1f}%", f"{stockout_skus} SKUs (< 0.2 Mo)", 
                so_bg), unsafe_allow_html=True)
        with c7:
            # Amber/Orange Tua untuk Need Replenishment
            st.markdown(render_inv_card("Need Replenishment", f"{replenish_skus} SKUs", "SKUs (< 0.8 Mo Cover)", 
                "linear-gradient(135deg, #fb8c00 0%, #ef6c00 100%)"), unsafe_allow_html=True)

        # ==============================================================================
        # 3. STOCK COVER & OCCUPANCY DASHBOARD (GAUGE DENGAN 1 DESIMAL)
        # ==============================================================================
        st.write("")
        st.subheader("⚡ Inventory Health & Warehouse Utilization")
        
        col_speed1, col_speed2 = st.columns(2)
        
        with col_speed1:
            # Gauge: Global Coverage menggunakan ECharts
            if ECHARTS_AVAILABLE:
                cover_value = round(global_cover_months, 1)
                
                option_cover = {
                    "series": [{
                        "type": 'gauge',
                        "startAngle": 180, # Setengah lingkaran presisi
                        "endAngle": 0,
                        "center": ['50%', '70%'], # Agak ke bawah agar muat
                        "radius": '100%',
                        "min": 0,
                        "max": 6,
                        "splitNumber": 6,
                        "axisLine": {
                            "lineStyle": {
                                "width": 25, # Lebar rel
                                "color": [
                                    [0.133, '#EF4444'], # 0 - 0.8 Bulan (Red/Under)
                                    [0.333, '#10B981'], # 0.8 - 2.0 Bulan (Green/Ideal)
                                    [1, '#F59E0B']      # > 2.0 Bulan (Orange/Over)
                                ]
                            }
                        },
                        "progress": {"show": False}, # Matikan progress agar tidak meluber
                        "pointer": {
                            "length": '60%',
                            "width": 6,
                            "itemStyle": {"color": '#4B5563'} # Jarum abu-abu elegan
                        },
                        "axisTick": {"show": False},
                        "splitLine": {
                            "length": 25,
                            "lineStyle": {"color": 'white', "width": 3} # Pemisah putih
                        },
                        "axisLabel": {
                            "distance": 30,
                            "color": '#6B7280',
                            "fontSize": 12,
                            "fontWeight": 'bold',
                            "formatter": '{value}'
                        },
                        "title": {
                            "show": True,
                            "offsetCenter": [0, '35%'], # Posisi judul "Stock Coverage"
                            "fontSize": 14,
                            "fontWeight": 'bold',
                            "color": '#6B7280'
                        },
                        "detail": {
                            "show": True,
                            "valueAnimation": True,
                            "fontSize": 35,
                            "fontWeight": '900',
                            "offsetCenter": [0, '-15%'], # Posisi angka
                            "color": '#1F2937',
                            "formatter": '{value} Mo'
                        },
                        "data": [{"value": cover_value, "name": "Stock Coverage"}]
                    }]
                }
                
                try:
                    st_echarts(options=option_cover, height="280px", key="gauge_cover_tab3")
                except Exception as e:
                    st.warning(f"⚠️ Gagal render gauge: {str(e)}")
                    st.metric("Global Stock Cover", f"{global_cover_months:.1f} Mo")
            else:
                st.metric("Global Stock Cover", f"{global_cover_months:.1f} Mo")
            
            st.caption(f"📊 Current: **{global_cover_months:.1f} Bulan** | Target: **0.8 - 2.0 Bulan**")
        
        with col_speed2:
            # Gauge: WH Occupancy
            if ECHARTS_AVAILABLE:
                occ_value = round(occupancy_pct, 1)
                
                option_occ = {
                    "series": [{
                        "type": 'gauge',
                        "startAngle": 180,
                        "endAngle": 0,
                        "center": ['50%', '70%'],
                        "radius": '100%',
                        "min": 0,
                        "max": 100,
                        "splitNumber": 5,
                        "axisLine": {
                            "lineStyle": {
                                "width": 25,
                                "color": [
                                    [0.8, '#10B981'],  # 0-80% (Green/Aman)
                                    [0.9, '#F59E0B'],  # 80-90% (Orange/Warning)
                                    [1, '#EF4444']     # 90-100% (Red/Critical)
                                ]
                            }
                        },
                        "progress": {"show": False},
                        "pointer": {
                            "length": '60%',
                            "width": 6,
                            "itemStyle": {"color": '#4B5563'}
                        },
                        "axisTick": {"show": False},
                        "splitLine": {
                            "length": 25,
                            "lineStyle": {"color": 'white', "width": 3}
                        },
                        "axisLabel": {
                            "distance": 30,
                            "color": '#6B7280',
                            "fontSize": 12,
                            "fontWeight": 'bold',
                            "formatter": '{value}%'
                        },
                        "title": {
                            "show": True,
                            "offsetCenter": [0, '35%'],
                            "fontSize": 14,
                            "fontWeight": 'bold',
                            "color": '#6B7280'
                        },
                        "detail": {
                            "show": True,
                            "valueAnimation": True,
                            "fontSize": 35,
                            "fontWeight": '900',
                            "offsetCenter": [0, '-15%'],
                            "color": '#1F2937',
                            "formatter": '{value}%'
                        },
                        "data": [{"value": occ_value, "name": "Occupancy"}]
                    }]
                }
                
                try:
                    st_echarts(options=option_occ, height="280px", key="gauge_occ_tab3")
                except Exception as e:
                    st.warning(f"⚠️ Gagal render gauge: {str(e)}")
                    st.metric("Warehouse Occupancy", f"{occupancy_pct:.1f}%")
            else:
                st.metric("Warehouse Occupancy", f"{occupancy_pct:.1f}%")
            
            st.caption(f"📦 Capacity Used: **{current_occupancy:,.0f}** / **{WH_CAPACITY:,.0f}** pcs ({occ_value:.1f}%)")


        # ==============================================================================
        # 3.5 ACTIONABLE INVENTORY ALERTS (ACTIVE & REGULAR SKU ONLY)
        # ==============================================================================
        st.divider()
        st.subheader("🚨 Actionable Inventory Alerts (Active & Regular SKU Only)")
        st.caption("Daftar *SKU Regular* berstatus **Active** yang membutuhkan tindakan operasional segera berdasarkan sales 3 bulan terakhir.")
        
        # Ekstrak list SKU yang HANYA berstatus 'SKU Regular' DAN 'Active'
        active_regular_skus = df_batch[
            (df_batch['Stock_Category'].str.contains('Regular', case=False, na=False)) &
            (df_batch['Status'].str.upper() == 'ACTIVE')
        ]['SKU_ID'].unique()
        
        if 'high_stock' in inventory_metrics and 'low_stock' in inventory_metrics:
            df_low = inventory_metrics['low_stock'].copy()
            df_high = inventory_metrics['high_stock'].copy()
            
            # Terapkan Filter Lapis Ganda & SORTING (Urutkan dari yang terparah)
            df_low = df_low[df_low['SKU_ID'].isin(active_regular_skus)].sort_values('Cover_Months', ascending=True) # Stock paling kritis (0.0) di atas
            df_high = df_high[df_high['SKU_ID'].isin(active_regular_skus)].sort_values('Cover_Months', ascending=False) # Dead stock terlama (999) di atas
            
            cols_to_show = ['SKU_ID', 'Product_Name', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
            
            col_alert1, col_alert2 = st.columns(2)
            
            with col_alert1:
                st.markdown(f"**📉 Need Replenishment (< 0.8 Bulan): <span style='color:#EF4444;'>{len(df_low)} SKUs</span>**", unsafe_allow_html=True)
                if not df_low.empty:
                    disp_low = df_low[cols_to_show].rename(columns={'Avg_Monthly_Sales_3M':'Sales/Mo', 'Cover_Months':'Cover'})
                    disp_low['Stock_Qty'] = disp_low['Stock_Qty'].apply(lambda x: f"{x:,.0f}")
                    disp_low['Sales/Mo'] = disp_low['Sales/Mo'].apply(lambda x: f"{x:,.0f}")
                    disp_low['Cover'] = disp_low['Cover'].apply(lambda x: f"{x:.1f}")
                    st.dataframe(disp_low, use_container_width=True, height=250, hide_index=True)
                else:
                    st.success("✅ Tidak ada SKU kritis. Semua aman!")
                    
            with col_alert2:
                st.markdown(f"**📦 Overstock / Dead Stock Alert (> 2 Bulan): <span style='color:#F59E0B;'>{len(df_high)} SKUs</span>**", unsafe_allow_html=True)
                if not df_high.empty:
                    disp_high = df_high[cols_to_show].rename(columns={'Avg_Monthly_Sales_3M':'Sales/Mo', 'Cover_Months':'Cover'})
                    disp_high['Stock_Qty'] = disp_high['Stock_Qty'].apply(lambda x: f"{x:,.0f}")
                    disp_high['Sales/Mo'] = disp_high['Sales/Mo'].apply(lambda x: f"{x:,.0f}")
                    disp_high['Cover'] = disp_high['Cover'].apply(lambda x: "No Sales" if x > 900 else f"{x:.1f}")
                    st.dataframe(disp_high, use_container_width=True, height=250, hide_index=True)
                else:
                    st.success("✅ Tidak ada SKU Overstock. Gudang efisien!")

        # ==============================================================================
        # 4. ABC ANALYSIS & FINANCIAL AGING EXPOSURE (KHUSUS SKU REGULAR)
        # ==============================================================================
        st.divider()
        st.subheader("🧬 Strategic Inventory Classification & Risk Exposure (SKU Regular)")
        st.caption("Memetakan prioritas *SKU Regular* berdasarkan nilai aset (ABC Analysis) dan risiko kedaluwarsa secara finansial.")
        
        # Buat dataframe khusus SKU Regular untuk perhitungan ABC & Aging
        df_batch_regular = df_batch[df_batch['Stock_Category'].str.contains('Regular', case=False, na=False)].copy()
        
        col_abc, col_age = st.columns([1, 1.2])
        
        with col_abc:
            # --- KONSEP ABC ANALYSIS ---
            st.markdown("**📊 ABC Inventory Classification (By Value)**")
            
            if not df_batch_regular.empty:
                df_abc = df_batch_regular.groupby(['SKU_ID', 'Product_Name'])['Total_Value'].sum().reset_index()
                df_abc = df_abc.sort_values('Total_Value', ascending=False)
                df_abc['Cum_Value'] = df_abc['Total_Value'].cumsum()
                df_abc['Cum_Pct'] = df_abc['Cum_Value'] / df_abc['Total_Value'].sum() * 100
                
                def classify_abc(pct):
                    if pct <= 80: return 'A (Top 80% Value)'
                    elif pct <= 95: return 'B (Next 15% Value)'
                    else: return 'C (Bottom 5% Value)'
                    
                df_abc['ABC_Class'] = df_abc['Cum_Pct'].apply(classify_abc)
                abc_summary = df_abc.groupby('ABC_Class').agg(
                    SKU_Count=('SKU_ID', 'count'),
                    Total_Value=('Total_Value', 'sum')
                ).reset_index()
                
                fig_abc = px.pie(abc_summary, values='Total_Value', names='ABC_Class', hole=0.5,
                                 color='ABC_Class',
                                 color_discrete_map={
                                     'A (Top 80% Value)': '#10B981', # Emerald
                                     'B (Next 15% Value)': '#3B82F6', # Blue
                                     'C (Bottom 5% Value)': '#9CA3AF'  # Gray
                                 },
                                 custom_data=['SKU_Count'])
                
                fig_abc.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    hovertemplate="<b>%{label}</b><br>Value: Rp %{value:,.0f}<br>Total SKUs: %{customdata[0]}<extra></extra>"
                )
                fig_abc.update_layout(height=380, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                
                fig_abc.add_annotation(text=f"<b>{len(df_abc)}</b><br>Regular SKUs", x=0.5, y=0.5, font_size=18, showarrow=False)
                st.plotly_chart(fig_abc, use_container_width=True)
            else:
                st.info("Tidak ada data SKU Regular.")

        with col_age:
            # --- FINANCIAL AGING PROFILE ---
            st.markdown("**⏳ Financial Exposure by Expiry Status**")
            
            if not df_batch_regular.empty:
                age_dist = df_batch_regular.groupby('Expiry_Category').agg({'Total_Value': 'sum', 'Stock_Qty': 'sum'}).reset_index()
                order_list = ['❌ EXPIRED', '🚨 Critical (<30 Days)', '⚠️ NED (1-3 Months)', '📅 NED (3-6 Months)', '✅ Safe (6-12 Months)', '🌟 Fresh (>1 Year)', 'Not Defined']
                age_dist['Expiry_Category'] = pd.Categorical(age_dist['Expiry_Category'], categories=order_list, ordered=True)
                age_dist = age_dist.sort_values('Expiry_Category')
                
                color_map = {
                    '❌ EXPIRED': '#EF4444', '🚨 Critical (<30 Days)': '#F87171',
                    '⚠️ NED (1-3 Months)': '#F59E0B', '📅 NED (3-6 Months)': '#FBBF24',
                    '✅ Safe (6-12 Months)': '#34D399', '🌟 Fresh (>1 Year)': '#10B981', 'Not Defined': '#9CA3AF'
                }
                
                fig_age = px.bar(
                    age_dist, x='Total_Value', y='Expiry_Category', orientation='h',
                    color='Expiry_Category', color_discrete_map=color_map,
                    text=age_dist['Total_Value'].apply(lambda x: f"Rp {x/1e6:,.0f} Jt" if x > 0 else "")
                )
                
                fig_age.update_traces(textposition='outside', textfont=dict(weight='bold', color='#4B5563'))
                fig_age.update_layout(
                    height=380, showlegend=False, plot_bgcolor='white',
                    xaxis=dict(title="Trapped Capital (Rupiah)", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                    yaxis=dict(title="", autorange="reversed"),
                    margin=dict(t=10, l=10, r=40, b=10)
                )
                st.plotly_chart(fig_age, use_container_width=True)

        # ==============================================================================
        # 5. INVENTORY MATRIX (CATEGORY VS EXPIRY) - PREMIUM STYLING
        # ==============================================================================
        st.divider()
        st.subheader("🗓️ Inventory Matrix: Risk Detection")
        st.caption("Peta persebaran kuantitas stok berdasarkan Kategori dan Status Kedaluwarsa.")
        
        pivot = pd.pivot_table(df_batch, values='Stock_Qty', index='Stock_Category', columns='Expiry_Category', aggfunc='sum', fill_value=0)
        existing_cols = [c for c in order_list if c in pivot.columns]
        pivot = pivot[existing_cols]
        pivot['TOTAL (Qty)'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('TOTAL (Qty)', ascending=False)
        
        # Styling dengan warna background yang lebih halus (Soft Reds untuk kolom bahaya)
        risk_cols = [c for c in existing_cols if 'EXPIRED' in c or 'Critical' in c or 'NED' in c]
        safe_cols = [c for c in existing_cols if 'Safe' in c or 'Fresh' in c]
        
        styler = pivot.style.format("{:,.0f}")
        if risk_cols:
            styler = styler.background_gradient(cmap='Reds', subset=risk_cols, vmin=0)
        if safe_cols:
            styler = styler.background_gradient(cmap='Greens', subset=safe_cols, vmin=0)
            
        st.dataframe(styler, use_container_width=True, height=350)

        # ==============================================================================
        # 6. DRILL-DOWN ANALYSIS
        # ==============================================================================
        st.divider()
        with st.expander("🔍 Drill-Down & Search Data", expanded=True):
            f1, f2, f3 = st.columns([1, 1, 2])
            with f1: sel_status = st.multiselect("Status:", df_batch['Status'].unique(), default=['Active'])
            with f2: sel_expiry = st.multiselect("Expiry:", df_batch['Expiry_Category'].unique())
            with f3: search_sku = st.text_input("Search (SKU/Name):", placeholder="Type here...")
            
            df_drill = df_batch.copy()
            if sel_status: df_drill = df_drill[df_drill['Status'].isin(sel_status)]
            if sel_expiry: df_drill = df_drill[df_drill['Expiry_Category'].isin(sel_expiry)]
            if search_sku: 
                df_drill = df_drill[df_drill['SKU_ID'].str.contains(search_sku, case=False) | df_drill['Product_Name'].str.contains(search_sku, case=False)]
            
            # Display
            cols = ['SKU_ID', 'Product_Name', 'Status', 'Stock_Category', 'Expiry_Category', 'Stock_Qty', 'Floor_Price', 'Total_Value']
            if 'Expiry_Date' in df_batch.columns: cols.insert(5, 'Expiry_Date')
            
            final_cols = [c for c in cols if c in df_drill.columns]
            
            st.dataframe(
                df_drill[final_cols].sort_values('Total_Value', ascending=False), 
                column_config={
                    "Total_Value": st.column_config.NumberColumn("Value", format="Rp %d"),
                    "Floor_Price": st.column_config.NumberColumn("Price", format="Rp %d"),
                    "Stock_Qty": st.column_config.NumberColumn("Qty")
                },
                use_container_width=True
            )

    else:
        st.warning("⚠️ Data Stok Kosong.")

# --- TAB 4: SKU EVALUATION (SKU 360 INSIGHT DECK) ---
with tab4:
    st.subheader("🔍 SKU 360° Deep Dive Analysis")
    st.caption("Micro-level analysis for individual Product Performance, Health, and Financials")

    if not df_product_active.empty:
        # Siapkan list SEMUA SKU Active untuk dropdown
        available_skus = df_product_active.apply(lambda x: f"{x['SKU_ID']} - {x['Product_Name']}", axis=1).tolist()
        
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_sku_display = st.selectbox(
                "📋 Search & Select SKU (All Active Products):", 
                options=sorted(available_skus)
            )
        
        if selected_sku_display:
            selected_sku = selected_sku_display.split(" - ")[0]
            
            # --- 1. GET PRODUCT MASTER INFO ---
            sku_master = df_product_active[df_product_active['SKU_ID'] == selected_sku].iloc[0]
            product_name = sku_master.get('Product_Name', 'Unknown')
            brand = sku_master.get('Brand', 'Unknown')
            tier = sku_master.get('SKU_Tier', 'Standard')
            price = sku_master.get('Floor_Price', 0)
            
            # --- 2. GET INVENTORY INFO ---
            stock_qty = 0
            avg_sales_3m = 0
            cover_months = 0
            if 'inventory_df' in inventory_metrics:
                inv_row = inventory_metrics['inventory_df'][inventory_metrics['inventory_df']['SKU_ID'] == selected_sku]
                if not inv_row.empty:
                    stock_qty = inv_row.iloc[0]['Stock_Qty']
                    avg_sales_3m = inv_row.iloc[0].get('Avg_Monthly_Sales_3M', 0)
                    cover_months = inv_row.iloc[0].get('Cover_Months', 0)
            
            # --- 3. GET LAST MONTH FORECAST INFO (SAFE EVALUATION) ---
            last_month_name = "N/A"
            acc_val = "N/A"
            if monthly_performance:
                last_month = sorted(monthly_performance.keys())[-1]
                last_month_name = last_month.strftime('%b')
                lm_data = monthly_performance[last_month]['data']
                sku_lm = lm_data[lm_data['SKU_ID'] == selected_sku]
                
                if not sku_lm.empty:
                    fcst_q = sku_lm.iloc[0]['Forecast_Qty']
                    po_q = sku_lm.iloc[0]['PO_Qty']
                    if fcst_q > 0:
                        acc = (po_q / fcst_q * 100)
                        acc_val = f"{acc:.1f}%"
                    elif po_q > 0:
                        acc_val = "Over (No Plan)"
                    else:
                        acc_val = "0%"

            # --- 4. GET FINANCIAL INFO (YTD) ---
            ytd_revenue = 0
            ytd_margin_pct = 0
            if not df_financial.empty:
                sku_fin = df_financial[df_financial['SKU_ID'] == selected_sku]
                if not sku_fin.empty:
                    ytd_revenue = sku_fin['Revenue'].sum()
                    ytd_gross = sku_fin['Gross_Margin'].sum()
                    ytd_margin_pct = (ytd_gross / ytd_revenue * 100) if ytd_revenue > 0 else 0

            # ==============================================================================
            # 5. SKU PROFILE HEADER (HTML/CSS)
            # ==============================================================================
            st.markdown("""
            <style>
                .sku-header { background-color: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-left: 6px solid #6366F1; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; }
                .sku-title-box { flex: 2; min-width: 300px; }
                .sku-title { font-size: 1.4rem; font-weight: 800; color: #1F2937; margin-bottom: 0.5rem; }
                .sku-badges { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
                .badge { padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: flex; align-items: center; gap: 5px; }
                .badge-blue { background: #E0E7FF; color: #4338CA; }
                .badge-purple { background: #F3E8FF; color: #7E22CE; }
                .badge-gray { background: #F3F4F6; color: #4B5563; }
                .sku-fin-box { flex: 1; text-align: right; min-width: 200px; border-left: 1px solid #E5E7EB; padding-left: 20px; }
                .fin-label { font-size: 0.8rem; color: #6B7280; font-weight: 600; text-transform: uppercase; }
                .fin-val-big { font-size: 1.5rem; font-weight: 800; color: #10B981; }
                .fin-margin { font-size: 0.9rem; font-weight: 700; color: #F59E0B; }
            </style>
            """, unsafe_allow_html=True)
            
            # Format Uang Cerdas
            rev_display = format_rupiah(ytd_revenue)

            st.markdown(f"""
            <div class="sku-header">
                <div class="sku-title-box">
                    <div class="sku-title">{product_name} <span style="font-weight:400; font-size:1rem; color:#6B7280;">({selected_sku})</span></div>
                    <div class="sku-badges">
                        <span class="badge badge-blue">🏷️ {brand}</span>
                        <span class="badge badge-purple">💎 {tier}</span>
                        <span class="badge badge-gray">Unit Price: Rp {price:,.0f}</span>
                    </div>
                </div>
                <div class="sku-fin-box">
                    <div class="fin-label">YTD Revenue Generation</div>
                    <div class="fin-val-big">{rev_display}</div>
                    <div class="fin-margin">Avg Margin: {ytd_margin_pct:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ==============================================================================
            # 6. KEY METRICS GRID (Pastel Gradient Cards)
            # ==============================================================================
            # Prepare Colors based on Logic
            cover_color = "linear-gradient(135deg, #10B981 0%, #059669 100%)" # Green (Ideal)
            if cover_months < 0.8: cover_color = "linear-gradient(135deg, #EF4444 0%, #B91C1C 100%)" # Red
            elif cover_months > 2.0: cover_color = "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)" 

            def render_sku_card(label, val, sub, bg):
                return f"""
                <div style="background: {bg}; border-radius: 12px; padding: 1.2rem; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 0.8rem; font-weight: 600; opacity: 0.9; text-transform: uppercase;">{label}</div>
                    <div style="font-size: 1.6rem; font-weight: 800; margin: 5px 0;">{val}</div>
                    <div style="font-size: 0.85rem; opacity: 0.95; font-weight: 500;">{sub}</div>
                </div>
                """

            k1, k2, k3, k4 = st.columns(4)
            with k1: st.markdown(render_sku_card("Current Stock", f"{stock_qty:,.0f}", "Units Available", "linear-gradient(135deg, #6366F1 0%, #4338CA 100%)"), unsafe_allow_html=True)
            with k2: st.markdown(render_sku_card("Stock Cover", f"{cover_months:.1f} Mo", "Inventory Health", cover_color), unsafe_allow_html=True)
            with k3: st.markdown(render_sku_card("Avg Sales (3M)", f"{avg_sales_3m:,.0f}", "Monthly Velocity", "linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%)"), unsafe_allow_html=True)
            with k4: st.markdown(render_sku_card("PO vs Rofo", acc_val, f"Last Month: {last_month_name}", "linear-gradient(135deg, #EC4899 0%, #DB2777 100%)"), unsafe_allow_html=True)

            # ==============================================================================
            # 7. SUPPLY CHAIN PULSE CHART
            # ==============================================================================
            st.write("")
            st.subheader("📈 Supply Chain Pulse (Trend Analysis)")
            
            hist_data = []
            if not df_sales.empty:
                sales_months = sorted(df_sales['Month'].unique())
                target_months = sales_months[-12:] if len(sales_months) >= 12 else sales_months
                
                for m in target_months:
                    s_qty = df_sales[(df_sales['Month'] == m) & (df_sales['SKU_ID'] == selected_sku)]['Sales_Qty'].sum()
                    f_qty = df_forecast[(df_forecast['Month'] == m) & (df_forecast['SKU_ID'] == selected_sku)]['Forecast_Qty'].sum() if not df_forecast.empty else 0
                    p_qty = df_po[(df_po['Month'] == m) & (df_po['SKU_ID'] == selected_sku)]['PO_Qty'].sum() if not df_po.empty else 0
                    
                    hist_data.append({
                        'Month': m, 'Month_Txt': m.strftime('%b-%y'),
                        'Sales': s_qty, 'Forecast': f_qty, 'PO': p_qty
                    })
            
            if hist_data:
                df_hist = pd.DataFrame(hist_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_hist['Month_Txt'], y=df_hist['Sales'], name='Sales (Real)', mode='lines', fill='tozeroy', line=dict(color='#10B981', width=3), fillcolor='rgba(16, 185, 129, 0.1)'))
                fig.add_trace(go.Scatter(x=df_hist['Month_Txt'], y=df_hist['Forecast'], name='Forecast (Plan)', mode='lines+markers', line=dict(color='#6366F1', width=3, dash='dash'), marker=dict(size=6, color='#6366F1')))
                fig.add_trace(go.Bar(x=df_hist['Month_Txt'], y=df_hist['PO'], name='PO (Order)', marker_color='rgba(245, 158, 11, 0.4)', marker_line_color='#F59E0B', marker_line_width=1.5))
                fig.update_layout(height=450, hovermode="x unified", plot_bgcolor="white", legend=dict(orientation="h", y=1.1), yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', title="Units"), margin=dict(t=50, l=20, r=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

            # ==============================================================================
            # 8. SMART DIAGNOSTICS & QUICK STATS
            # ==============================================================================
            st.write("")
            c_diag1, c_diag2 = st.columns([1.5, 1])

            with c_diag1:
                st.subheader("🩺 Smart Diagnostics")
                diagnoses = []
                if cover_months < 0.8: diagnoses.append(("🔴", "High Stockout Risk", f"Stock cover is only {cover_months:.1f} months. Urgent replenishment needed."))
                elif cover_months > 2.0: diagnoses.append(("🟡", "Overstock Alert", f"Stock cover is {cover_months:.1f} months. Consider holding PO or running promo."))
                else: diagnoses.append(("🟢", "Healthy Inventory", "Stock levels are optimal (0.8 - 2.0 months)."))

                if len(df_hist) >= 3:
                    last_3_sales = df_hist['Sales'].tail(3).mean()
                    prev_3_sales = df_hist['Sales'].iloc[-6:-3].mean() if len(df_hist) >= 6 else last_3_sales
                    if prev_3_sales > 0:
                        growth = (last_3_sales - prev_3_sales) / prev_3_sales * 100
                        if growth > 20: diagnoses.append(("🚀", "Surging Demand", f"Sales trend is up +{growth:.1f}% vs prev period. Adjust forecast up."))
                        elif growth < -20: diagnoses.append(("📉", "Declining Sales", f"Sales trend is down {growth:.1f}%. Review forecast down."))

                for icon, title, desc in diagnoses:
                    bg_col = "#F0FDF4" if icon == "🟢" else "#FEF2F2" if icon == "🔴" else "#FFFBEB"
                    border_col = "#22C55E" if icon == "🟢" else "#EF4444" if icon == "🔴" else "#F59E0B"
                    st.markdown(f"""
                    <div style="background:{bg_col}; border-left:4px solid {border_col}; padding:12px; border-radius:8px; margin-bottom:10px;">
                        <div style="font-weight:700; color:#374151; display:flex; align-items:center; gap:8px;">
                            <span style="font-size:1.2rem;">{icon}</span> {title}
                        </div>
                        <div style="font-size:0.9rem; color:#4B5563; margin-left:32px;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with c_diag2:
                st.subheader("📋 Quick Stats")
                last_fcst = df_hist.iloc[-1]['Forecast'] if hist_data else 0
                last_po = df_hist.iloc[-1]['PO'] if hist_data else 0
                last_sales = df_hist.iloc[-1]['Sales'] if hist_data else 0
                stats_data = {
                    "Metric": ["Forecast (Last Mo)", "PO (Last Mo)", "Sales (Last Mo)", "Bias (Last Mo)"],
                    "Value": [f"{last_fcst:,.0f}", f"{last_po:,.0f}", f"{last_sales:,.0f}", f"{last_po - last_fcst:+,.0f}"]
                }
                st.table(pd.DataFrame(stats_data))

    else:
        st.info("👋 Please ensure Sales and Monthly Performance data are loaded to view SKU insights.")

# --- TAB 5: SALES & FORECAST ANALYSIS (EASY TO UNDERSTAND VERSION) ---
with tab5:
    st.subheader("📈 Realization & Gap Analysis")
    st.caption("Membandingkan Perencanaan (Rofo), Eksekusi (PO), dan Hasil Akhir (Sales)")

    if not df_sales.empty and not df_forecast.empty:
        # ==============================================================================
        # 1. DATA PREPARATION & FILTER TAHUN
        # ==============================================================================
        
        # Get all unique months from the datasets
        all_months = sorted(list(set(df_sales['Month'].unique()) | set(df_forecast['Month'].unique()) | set(df_po['Month'].unique())))
        
        # Ekstrak Tahun unik dari semua bulan
        available_years = sorted(list(set([m.year for m in all_months if pd.notnull(m)])))
        
        # UI Filter Multi-Select untuk Tahun
        selected_years = st.multiselect(
            "📅 Filter Tahun:",
            options=available_years,
            default=available_years, # Default: tampilkan semua tahun
            help="Pilih satu atau beberapa tahun untuk menganalisis performa pada periode tertentu."
        )
        
        # Filter list bulan berdasarkan tahun yang dipilih
        filtered_months = [m for m in all_months if m.year in selected_years]

        # Cegah error jika user menghapus semua pilihan tahun
        if not filtered_months:
            st.warning("⚠️ Silakan pilih minimal 1 tahun untuk menampilkan data.")
        else:
            monthly_data = []
            for month in filtered_months:
                # Sales, Forecast, PO
                s_qty = df_sales[df_sales['Month'] == month]['Sales_Qty'].sum()
                f_qty = df_forecast[df_forecast['Month'] == month]['Forecast_Qty'].sum()
                p_qty = df_po[df_po['Month'] == month]['PO_Qty'].sum()
                
                monthly_data.append({
                    'Month': month,
                    'Month_Txt': month.strftime('%b-%y'),
                    'Rofo': f_qty,
                    'Sales': s_qty,
                    'PO': p_qty,
                    # Hitung selisih untuk visualisasi
                    'Gap_Sales_Rofo': s_qty - f_qty
                })
                
            df_trend = pd.DataFrame(monthly_data)
            
            # Totals for KPI
            total_rofo = df_trend['Rofo'].sum()
            total_sales = df_trend['Sales'].sum()
            total_po = df_trend['PO'].sum()
            
            # ==============================================================================
            # 2. KPI CARDS (PASTEL GRADIENT)
            # ==============================================================================
            st.markdown("""
            <style>
                .kpi-box {
                    border-radius: 12px; padding: 1.2rem; color: white;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05); position: relative;
                    transition: transform 0.3s;
                }
                .kpi-box:hover { transform: translateY(-3px); }
                .kpi-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; opacity: 0.9; margin-bottom: 5px; }
                .kpi-num { font-size: 1.8rem; font-weight: 800; margin-bottom: 0px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
                .kpi-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; }
            </style>
            """, unsafe_allow_html=True)

            def render_kpi(title, val, sub, gradient):
                return f"""
                <div class="kpi-box" style="background: {gradient};">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-num">{val}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """

            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                # Rofo (Plan) - Indigo
                st.markdown(render_kpi("1. PLAN (ROFO)", f"{total_rofo:,.0f}", "Total Forecast", 
                    "linear-gradient(135deg, #7986cb 0%, #5c6bc0 100%)"), unsafe_allow_html=True)
            with c2:
                # PO (Execution) - Orange
                po_vs_rofo = (total_po / total_rofo * 100) if total_rofo > 0 else 0
                st.markdown(render_kpi("2. EXECUTION (PO)", f"{total_po:,.0f}", f"{po_vs_rofo:.1f}% of Plan", 
                    "linear-gradient(135deg, #ffb74d 0%, #ffa726 100%)"), unsafe_allow_html=True)
            with c3:
                # Sales (Result) - Green
                sales_vs_rofo = (total_sales / total_rofo * 100) if total_rofo > 0 else 0
                st.markdown(render_kpi("3. RESULT (SALES)", f"{total_sales:,.0f}", f"{sales_vs_rofo:.1f}% Achievement", 
                    "linear-gradient(135deg, #4db6ac 0%, #26a69a 100%)"), unsafe_allow_html=True)
            with c4:
                # Gap - Red/Grey
                gap = total_sales - total_rofo
                gap_col = "linear-gradient(135deg, #ef5350 0%, #e53935 100%)" if gap < 0 else "linear-gradient(135deg, #66bb6a 0%, #43a047 100%)"
                st.markdown(render_kpi("GAP (SALES vs PLAN)", f"{gap:+,.0f}", "Units Variance", gap_col), unsafe_allow_html=True)

            # ==============================================================================
            # 3. MAIN COMPARISON CHART (ANIMATED STORYTELLING DENGAN VIZZU)
            # ==============================================================================
            st.divider()
            st.subheader("🎬 Performance Triad Story: Plan vs Exec vs Result")
            st.caption("Klik tombol 'Play' (▶️) di bawah grafik untuk melihat animasi perubahan sudut pandang data.")

            if VIZZU_AVAILABLE and not df_trend.empty:
                # 1. Siapkan Data dalam format Long (Tidy) untuk Vizzu
                df_vizzu = df_trend[['Month_Txt', 'Rofo', 'PO', 'Sales']].copy()
                df_vizzu = df_vizzu.melt(id_vars=['Month_Txt'], value_vars=['Rofo', 'PO', 'Sales'], var_name='Metric', value_name='Quantity')
                
                # Buat objek Data Vizzu
                vizzu_data = Data()
                vizzu_data.add_data_frame(df_vizzu)
                
                # Inisialisasi Story
                story = Story(data=vizzu_data)
                
                # Atur warna custom (Rofo=Indigo, PO=Orange, Sales=Teal)
                custom_style = Style({
                    "plot": {
                        "marker": { "colorPalette": "#3949AB #FFB74D #4DB6AC" },
                        "xAxis": { "label": { "angle": "-45" } }
                    }
                })

                # SLIDE 1: OVERALL TOTAL (Bar Chart Biasa)
                slide1 = Slide(
                    Step(
                        Config({
                            "x": "Metric", "y": "Quantity", "color": "Metric",
                            "title": "1. Overall Performance Comparison (Total Horizon)",
                            "label": "Quantity"
                        }),
                        custom_style
                    )
                )
                story.add_slide(slide1)

                # SLIDE 2: BREAKDOWN PER BULAN (Grouped Bar Chart)
                slide2 = Slide(
                    Step(
                        Config({
                            "x": ["Month_Txt", "Metric"], "y": "Quantity", "color": "Metric",
                            "title": "2. Monthly Breakdown (Rofo vs PO vs Sales)",
                            "label": None # Matikan label agar tidak semrawut
                        })
                    )
                )
                story.add_slide(slide2)

                # SLIDE 3: PERUBAHAN KE TREND (Line Chart)
                slide3 = Slide(
                    Step(
                        Config({
                            "geometry": "line",
                            "x": "Month_Txt", "y": "Quantity", "color": "Metric",
                            "title": "3. Trend Analysis Over Time"
                        })
                    )
                )
                story.add_slide(slide3)

                # Render Story di Streamlit menggunakan iframe/HTML
                story.set_feature("tooltip", True)
                story.set_size("100%", "450px")
                
                import streamlit.components.v1 as components
                # Tampilkan animasi
                components.html(story.to_html(), height=500)
                
            else:
                # Fallback ke Plotly biasa jika Vizzu tidak terinstall
                st.warning("Memuat grafik standar. Install `ipyvizzu-story` untuk melihat animasi presentasi.")
                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(x=df_trend['Month_Txt'], y=df_trend['Rofo'], name='Plan (Rofo)', mode='lines+markers', line=dict(color='#3949AB', width=3, dash='dash')))
                fig_main.add_trace(go.Bar(x=df_trend['Month_Txt'], y=df_trend['PO'], name='Execution (PO)', marker_color='#FFB74D'))
                fig_main.add_trace(go.Bar(x=df_trend['Month_Txt'], y=df_trend['Sales'], name='Result (Sales)', marker_color='#4DB6AC'))
                fig_main.update_layout(height=450, barmode='group', hovermode="x unified", plot_bgcolor='white')
                st.plotly_chart(fig_main, use_container_width=True)

            # ==============================================================================
            # 4. TOP GAP ANALYSIS (VOLUME RISK IMPACT)
            # ==============================================================================
            st.divider()
            st.subheader("🚨 Top Gap Analysis (Risk Impact)")
            st.caption("Mendeteksi SKU dengan selisih unit terbesar: **Risiko Overstock** (Plan terlalu tinggi) dan **Risiko Stockout** (Sales melampaui Plan).")

            # Filter data & Aggregate per SKU
            df_f_filtered = df_forecast[df_forecast['Month'].dt.year.isin(selected_years)]
            df_s_filtered = df_sales[df_sales['Month'].dt.year.isin(selected_years)]

            df_f_sku = df_f_filtered.groupby(['SKU_ID', 'Product_Name'])['Forecast_Qty'].sum().reset_index()
            df_s_sku = df_s_filtered.groupby(['SKU_ID', 'Product_Name'])['Sales_Qty'].sum().reset_index()
            
            df_gap = pd.merge(df_f_sku, df_s_sku, on=['SKU_ID', 'Product_Name'], how='outer').fillna(0)
            
            # Hitung Gap Aktual (Sales - Rofo)
            df_gap['Gap_Qty'] = df_gap['Sales_Qty'] - df_gap['Forecast_Qty']
            
            # --- FIX ERROR: Pastikan Product_Name adalah string dan potong dengan aman ---
            df_gap['Product_Name'] = df_gap['Product_Name'].astype(str)
            df_gap['Display_Name'] = df_gap['Product_Name'].apply(lambda x: x[:25] + "..." if len(x) > 25 else x)
            
            # 1. Over-forecast (Gap Negatif -> Plan lebih besar dari Sales)
            top_over = df_gap[df_gap['Gap_Qty'] < 0].copy()
            top_over['Excess_Qty'] = abs(top_over['Gap_Qty']) # Jadikan absolut untuk chart
            top_over = top_over.sort_values('Excess_Qty', ascending=False).head(10)
            
            # 2. Under-forecast (Gap Positif -> Sales lebih besar dari Plan)
            top_under = df_gap[df_gap['Gap_Qty'] > 0].copy()
            top_under = top_under.sort_values('Gap_Qty', ascending=False).head(10)

            c_gap1, c_gap2 = st.columns(2)

            with c_gap1:
                st.markdown("##### 📉 Top Over-Forecasted (Excess Units)")
                st.caption("Barang terjual lebih sedikit dari rencana. Risiko penumpukan stok (*Overstock*).")
                
                if not top_over.empty:
                    fig_drop = go.Figure()
                    fig_drop.add_trace(go.Bar(
                        y=top_over['Display_Name'], 
                        x=top_over['Excess_Qty'], 
                        orientation='h', marker_color='#EF5350', # Red
                        text=[f"{x:,.0f} pcs" for x in top_over['Excess_Qty']],
                        textposition='auto', name='Excess Qty'
                    ))
                    fig_drop.update_layout(
                        height=400, xaxis_title="Excess Units vs Plan",
                        yaxis=dict(autorange="reversed"), plot_bgcolor='white', margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_drop, use_container_width=True)
                else:
                    st.success("✅ Tidak ada barang Over-Forecast di periode ini!")

            with c_gap2:
                st.markdown("##### 🚀 Top Under-Forecasted (Shortage Risk)")
                st.caption("Barang laku keras melebihi rencana. Risiko *Stockout* dan potensi *Lost Sales*.")
                
                if not top_under.empty:
                    fig_spike = go.Figure()
                    fig_spike.add_trace(go.Bar(
                        y=top_under['Display_Name'], 
                        x=top_under['Gap_Qty'], 
                        orientation='h', marker_color='#10B981', # Green
                        text=[f"+{x:,.0f} pcs" for x in top_under['Gap_Qty']],
                        textposition='auto', name='Shortage Qty'
                    ))
                    fig_spike.update_layout(
                        height=400, xaxis_title="Extra Units Sold vs Plan",
                        yaxis=dict(autorange="reversed", side="right"), plot_bgcolor='white', margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_spike, use_container_width=True)
                else:
                    st.success("✅ Tidak ada barang Under-Forecast di periode ini!")

    else:
        st.info("ℹ️ Membutuhkan data Sales dan Forecast untuk menampilkan analisis.")


# --- TAB 6: ECOMMERCE FORECAST INTELLIGENCE (REVAMPED WITH AI ENGINE) ---
with tab6:
    st.subheader("🔮 Ecommerce Forecast Intelligence")

    viewer_tab, ai_tab = st.tabs([
        "📊 Existing Forecast Viewer",
        "🤖 AI Demand Forecasting Engine"
    ])

    # =========================================================================
    # SUB-TAB 1: EXISTING FORECAST VIEWER (unchanged from original)
    # =========================================================================
    with viewer_tab:
        if not df_ecomm_forecast.empty:
            id_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Status',
                       'Floor_Price', 'Net_Order_Price']
            forecast_cols = [c for c in df_ecomm_forecast.columns if c not in id_cols]

            valid_fcst_cols = []
            for c in forecast_cols:
                try:
                    if df_ecomm_forecast[c].dtype == object:
                        sample = df_ecomm_forecast[c].iloc[0]
                        if isinstance(sample, str) and any(i.isdigit() for i in sample):
                            valid_fcst_cols.append(c)
                    elif np.issubdtype(df_ecomm_forecast[c].dtype, np.number):
                        valid_fcst_cols.append(c)
                except:
                    pass

            col_date_map = []
            if valid_fcst_cols:
                for c in valid_fcst_cols:
                    clean_c = str(c).strip()
                    for fmt in ['%b-%y', '%b %y', '%b-%Y', '%B %Y', '%Y-%m', '%b_%y']:
                        try:
                            dt = datetime.strptime(clean_c, fmt)
                            col_date_map.append({'col': c, 'date': dt})
                            break
                        except:
                            continue

            if not col_date_map:
                st.warning("⚠️ Tidak dapat membaca kolom bulan secara otomatis.")
            else:
                col_date_map.sort(key=lambda x: x['date'])
                sorted_fcst_cols = [x['col'] for x in col_date_map]

                df_fcst_active = df_ecomm_forecast.copy()
                for c in sorted_fcst_cols:
                    df_fcst_active[c] = pd.to_numeric(df_fcst_active[c], errors='coerce').fillna(0)

                if 'Floor_Price' not in df_fcst_active.columns:
                    df_fcst_active = add_product_info_to_data(df_fcst_active, df_product)

                total_fcst_qty = df_fcst_active[sorted_fcst_cols].sum().sum()

                total_fcst_val = 0
                has_price = False
                if 'Floor_Price' in df_fcst_active.columns:
                    df_fcst_active['Floor_Price'] = pd.to_numeric(df_fcst_active['Floor_Price'], errors='coerce').fillna(0)
                    row_sums = df_fcst_active[sorted_fcst_cols].sum(axis=1)
                    total_fcst_val = (row_sums * df_fcst_active['Floor_Price']).sum()
                    has_price = True

                st.markdown("""
                <style>
                    .fcst-card { border-radius:12px; padding:1.2rem; color:white; box-shadow:0 4px 10px rgba(0,0,0,0.05); transition:transform 0.3s; }
                    .fcst-card:hover { transform:translateY(-3px); }
                    .fcst-title { font-size:0.8rem; font-weight:700; text-transform:uppercase; opacity:0.9; margin-bottom:5px; }
                    .fcst-val { font-size:1.8rem; font-weight:800; text-shadow:0 1px 2px rgba(0,0,0,0.1); }
                    .fcst-sub { font-size:0.85rem; font-weight:500; opacity:0.95; }
                </style>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""<div class="fcst-card" style="background:linear-gradient(135deg,#6366F1 0%,#4338CA 100%);"><div class="fcst-title">Total Forecast Volume</div><div class="fcst-val">{total_fcst_qty:,.0f}</div><div class="fcst-sub">Units</div></div>""", unsafe_allow_html=True)
                with c2:
                    val_text = format_rupiah(total_fcst_val) if has_price else "N/A"
                    st.markdown(f"""<div class="fcst-card" style="background:linear-gradient(135deg,#10B981 0%,#059669 100%);"><div class="fcst-title">Total Forecast Value</div><div class="fcst-val">{val_text}</div><div class="fcst-sub">Gross Revenue</div></div>""", unsafe_allow_html=True)
                with c3:
                    peak_month = df_fcst_active[sorted_fcst_cols].sum().idxmax()
                    st.markdown(f"""<div class="fcst-card" style="background:linear-gradient(135deg,#F59E0B 0%,#D97706 100%);"><div class="fcst-title">Peak Season</div><div class="fcst-val">{str(peak_month).split('.')[0]}</div><div class="fcst-sub">Highest Volume Month</div></div>""", unsafe_allow_html=True)

                st.divider()
                st.subheader("📈 Monthly Forecast Trend")
                monthly_agg = df_fcst_active[sorted_fcst_cols].sum().reset_index()
                monthly_agg.columns = ['Month', 'Qty']

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Bar(x=monthly_agg['Month'], y=monthly_agg['Qty'], name='Monthly Forecast', marker_color='#818CF8'))
                fig_trend.add_trace(go.Scatter(x=monthly_agg['Month'], y=[monthly_agg['Qty'].mean()] * len(monthly_agg), name='Average', mode='lines', line=dict(color='#F59E0B', width=2, dash='dash')))
                fig_trend.update_layout(height=350, hovermode="x unified", plot_bgcolor='white', margin=dict(t=20, b=20))
                st.plotly_chart(fig_trend, use_container_width=True)

                st.divider()
                st.subheader("📅 Quarterly Brand Analysis")
                q_map = {'Q1': ['jan','feb','mar'], 'Q2': ['apr','may','jun'],
                         'Q3': ['jul','aug','sep'], 'Q4': ['oct','nov','dec']}
                quarter_cols_map = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
                for col in sorted_fcst_cols:
                    m_str = str(col).lower()[:3]
                    for q, months in q_map.items():
                        if m_str in months:
                            quarter_cols_map[q].append(col)
                active_quarters = [q for q, cols in quarter_cols_map.items() if len(cols) > 0]

                if active_quarters and 'Brand' in df_fcst_active.columns:
                    q_tab1, q_tab2 = st.tabs(["📦 By Quantity (Heatmap)", "💰 By Value (Heatmap)"])
                    with q_tab1:
                        q_brand_qty = []
                        for brand in df_fcst_active['Brand'].unique():
                            brand_data = df_fcst_active[df_fcst_active['Brand'] == brand]
                            row = {'Brand': brand}
                            total_row = 0
                            for q in active_quarters:
                                q_val = brand_data[quarter_cols_map[q]].sum().sum()
                                row[q] = q_val; total_row += q_val
                            row['Total'] = total_row
                            q_brand_qty.append(row)
                        df_all_qty = pd.DataFrame(q_brand_qty)
                        df_q_qty = df_all_qty.sort_values('Total', ascending=False).head(15)
                        grand_total_qty = {'Brand': 'TOTAL (ALL BRANDS)'}
                        for col in active_quarters + ['Total']:
                            grand_total_qty[col] = df_all_qty[col].sum()
                        df_q_qty = pd.concat([df_q_qty, pd.DataFrame([grand_total_qty])], ignore_index=True)
                        display_cols = active_quarters + ['Total']
                        fig_heat_qty = go.Figure(data=go.Heatmap(
                            z=df_q_qty[display_cols].values, x=display_cols, y=df_q_qty['Brand'],
                            colorscale='Blues', text=df_q_qty[display_cols].values, texttemplate="%{text:,.0f}"
                        ))
                        fig_heat_qty.update_layout(height=550, title="Top 15 Brands - Quarterly Volume", yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_heat_qty, use_container_width=True)

                    with q_tab2:
                        if has_price:
                            q_brand_val = []
                            df_fcst_active['Temp_Price'] = df_fcst_active['Floor_Price'].fillna(0)
                            for brand in df_fcst_active['Brand'].unique():
                                brand_data = df_fcst_active[df_fcst_active['Brand'] == brand]
                                row = {'Brand': brand}; total_row = 0
                                for q in active_quarters:
                                    q_val = sum((brand_data[c] * brand_data['Temp_Price']).sum() for c in quarter_cols_map[q])
                                    row[q] = q_val; total_row += q_val
                                row['Total'] = total_row
                                q_brand_val.append(row)
                            df_all_val = pd.DataFrame(q_brand_val)
                            df_q_val = df_all_val.sort_values('Total', ascending=False).head(15)
                            grand_total_val = {'Brand': 'TOTAL (ALL BRANDS)'}
                            for col in active_quarters + ['Total']:
                                grand_total_val[col] = df_all_val[col].sum()
                            df_q_val = pd.concat([df_q_val, pd.DataFrame([grand_total_val])], ignore_index=True)
                            display_cols = active_quarters + ['Total']
                            fig_heat_val = go.Figure(data=go.Heatmap(
                                z=df_q_val[display_cols].values, x=display_cols, y=df_q_val['Brand'],
                                colorscale='Greens', text=df_q_val[display_cols].values, texttemplate="Rp %{text:,.0f}"
                            ))
                            fig_heat_val.update_layout(height=550, title="Top 15 Brands - Quarterly Revenue Projection", yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig_heat_val, use_container_width=True)
                        else:
                            st.warning("⚠️ Data Harga (Floor_Price) tidak ditemukan.")

        else:
            st.info("ℹ️ Silakan upload data forecast di sheet 'Forecast_2026_Ecomm' terlebih dahulu.")

    # =========================================================================
    # SUB-TAB 2: AI DEMAND FORECASTING ENGINE
    # =========================================================================
    with ai_tab:
        st.subheader("🤖 AI Demand Forecasting Engine")
        st.caption("Machine Learning-powered demand planning | Training data: historical ecomm sales | Horizon: 12 months")

        # --- STATUS BADGES (Library availability) ---
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.success("✅ NumPy / Pandas (Moving Average + Linear)")
        with col_b2:
            if STATSMODELS_AVAILABLE:
                st.success("✅ Statsmodels (Holt-Winters)")
            else:
                st.error("❌ Statsmodels not installed")
        with col_b3:
            if PROPHET_AVAILABLE:
                st.success("✅ Prophet (Meta) — Available")
            else:
                st.warning("⚠️ Prophet not installed — pip install prophet")

        st.divider()

        if df_sales.empty:
            st.warning("⚠️ Tidak ada data Sales historis. Upload data ke sheet 'Sales' terlebih dahulu.")
        else:
            # =================================================================
            # SECTION 1: CONFIGURATION PANEL
            # =================================================================
            st.markdown("### ⚙️ Forecast Configuration")
            cfg1, cfg2 = st.columns([1, 2])

            with cfg1:
                granularity = st.radio(
                    "📊 Analysis Level:",
                    ["Per SKU", "Per Brand"],
                    horizontal=True,
                    key="ai_granularity"
                )

            with cfg2:
                if granularity == "Per SKU":
                    # Build SKU options with product name
                    if not df_product_active.empty:
                        sku_opts = df_product_active.apply(
                            lambda r: f"{r['SKU_ID']} — {r.get('Product_Name', '')}", axis=1
                        ).tolist()
                    else:
                        sku_opts = df_sales['SKU_ID'].unique().tolist()
                    selected_entity_display = st.selectbox(
                        "🔍 Select SKU:", sorted(sku_opts), key="ai_sku_sel"
                    )
                    entity_value = selected_entity_display.split(" — ")[0].strip()
                    entity_label = selected_entity_display
                else:
                    brand_opts = sorted(df_sales['Brand'].dropna().unique().tolist()) if 'Brand' in df_sales.columns else []
                    entity_value = st.selectbox("🏷️ Select Brand:", brand_opts, key="ai_brand_sel")
                    entity_label = entity_value

            # Method selection with description
            METHOD_INFO = {
                # Tambahkan sebelum penutup } dari METHOD_INFO:
                "Rolling Baseline × MtM Seasonal ⭐": {
                    "key": "rolling_mtm",
                    "desc": "Baseline 3 bulan terakhir × historical MtM growth ratio. "
                            "Metode S&OP praktisi — adaptif, interpretable, proven.",
                    "icon": "🎯"
                },
                "Moving Average (Simple)": {
                    "key": "ma_simple",
                    "desc": "Rata-rata N bulan terakhir + trend adjustment. Cocok untuk data stabil.",
                    "icon": "📊"
                },
                "Moving Average (Weighted)": {
                    "key": "ma_weighted",
                    "desc": "Bulan terbaru diberi bobot lebih besar. Cocok untuk data dengan momentum.",
                    "icon": "⚖️"
                },
                "Holt-Winters (Exponential Smoothing)": {
                    "key": "holt_winters",
                    "desc": "Model statistik yang memperhitungkan trend + seasonalitas. Akurat untuk data bulanan.",
                    "icon": "📈"
                },
                "Linear Trend + Seasonality": {
                    "key": "linear_seasonal",
                    "desc": "Regresi linear + faktor musiman per bulan. Transparan dan interpretable.",
                    "icon": "📐"
                },
                "Prophet (Meta)": {
                    "key": "prophet",
                    "desc": "Model ML Facebook/Meta. Terbaik untuk deteksi trend & holiday effect.",
                    "icon": "🔮"
                }
            }

            m_opts = list(METHOD_INFO.keys())
            if not STATSMODELS_AVAILABLE:
                m_opts = [m for m in m_opts if "Holt-Winters" not in m]
            if not PROPHET_AVAILABLE:
                m_opts = [m for m in m_opts if "Prophet" not in m]

            p1, p2 = st.columns([2, 1])
            with p1:
                selected_method_name = st.selectbox(
                    "📐 Forecast Method:", m_opts, key="ai_method_sel"
                )
            with p2:
                method_info = METHOD_INFO[selected_method_name]
                st.info(f"{method_info['icon']} {method_info['desc']}")

            method_key = method_info["key"]

            # Method parameters
            params = {}
            if method_key in ("ma_simple", "ma_weighted"):
                params['window'] = st.slider(
                    "📅 Lookback Window (months):", 2, 12, 3,
                    help="Jumlah bulan historis sebagai dasar kalkulasi",
                    key="ai_window"
                )
            # Tambah setelah blok params MA:
            elif method_key == "rolling_mtm":
                params['baseline_window'] = st.slider(
                    "📅 Baseline Window (months):", 2, 6, 3,
                    help="Jumlah bulan terakhir sebagai baseline. "
                         "Default 3 = rata-rata 3 bulan terakhir actual.",
                    key="ai_baseline_window"
                )
                st.info(
                    "ℹ️ Method ini mereplikasi workflow S&OP Existing: "
                    "Baseline rolling × MtM seasonal ratio dari semua history. "
                    "Anomaly detection aktif otomatis."
                )

            # Forecast horizon
            n_forecast = 12
            st.caption(f"📆 Forecast Horizon: **12 months forward** dari bulan terakhir data historis")

            # Generate button
            st.write("")

            fc = None
            ci_lo = None
            ci_hi = None
            mape = None
            backtest_df = None
            existing_plan = pd.Series(dtype=float)
            rolling_warnings = []
            rolling_avg_mtm = {}

            generate_btn = st.button(
                "🚀 Generate AI Forecast",
                type="primary",
                use_container_width=True,
                key="ai_gen_btn"
            )

            # =================================================================
            # SECTION 2: GENERATE & DISPLAY RESULTS
            # =================================================================
            if generate_btn:
                with st.spinner(f"🔄 Running {selected_method_name} for {entity_label}..."):

                    # Prepare time series
                    ts = ai_prepare_ts(df_sales, granularity.split(" ")[1], entity_value)

                    if len(ts) < 3:
                        st.error(f"❌ Data historis terlalu sedikit ({len(ts)} bulan). Minimal 3 bulan data diperlukan.")
                    else:
                        # Run forecast
                        fc, ci_lo, ci_hi = None, None, None

                        if method_key == "ma_simple":
                            fc, ci_lo, ci_hi = ai_forecast_ma(ts, n_forecast, params.get('window', 3), False)
                        elif method_key == "ma_weighted":
                            fc, ci_lo, ci_hi = ai_forecast_ma(ts, n_forecast, params.get('window', 3), True)
                        elif method_key == "holt_winters":
                            fc, ci_lo, ci_hi = ai_forecast_holt_winters(ts, n_forecast)
                        elif method_key == "linear_seasonal":
                            fc, ci_lo, ci_hi = ai_forecast_linear_seasonal(ts, n_forecast)
                        elif method_key == "prophet":
                            fc, ci_lo, ci_hi = ai_forecast_prophet(ts, n_forecast)
                        # Tambah di bagian ini, setelah elif method_key == "prophet":
                        elif method_key == "rolling_mtm":
                            result = ai_forecast_rolling_baseline_mtm(
                                ts, n_forecast,
                                baseline_window=params.get('baseline_window', 3)
                            )
                            fc, ci_lo, ci_hi = result[0], result[1], result[2]
                            rolling_warnings = result[3]
                            rolling_avg_mtm = result[4]

                        if fc is None:
                            st.error("❌ Forecasting gagal. Coba method lain atau cek ketersediaan data.")
                        else:
                        # Anomaly warnings — tampil sebelum chart
                            if method_key == "rolling_mtm" and rolling_warnings:
                                st.markdown("#### ⚠️ Baseline Anomaly Detected")
                                for w_item in rolling_warnings:
                                    icon = "🚨" if w_item['type'] == 'spike' else "📉"
                                    msg = (
                                        f"{icon} **{w_item['month']}**: "
                                        f"Sales = **{w_item['value']:,.0f}** "
                                        f"({'%.1fx' % w_item['ratio']} {'di atas' if w_item['type'] == 'spike' else 'di bawah'} "
                                        f"historical avg {w_item['avg']:,.0f}). "
                                        f"Baseline mungkin {'over-estimated' if w_item['type'] == 'spike' else 'under-estimated'}. "
                                        f"**Validasi manual disarankan.**"
                                    )
                                    if w_item['type'] == 'spike':
                                        st.warning(msg)
                                    else:
                                        st.info(msg)
                                # Backtest
                                mape, backtest_df = ai_backtest(ts, method_key, params, holdout=3)

                            # =====================================================
                            # RESULT METRICS ROW (IMPROVED WITH FINANCIAL IMPACT)
                            # =====================================================
                            st.divider()
                            st.markdown("### 📊 AI Forecast Results & Financial Impact")

                            rm1, rm2, rm3, rm4 = st.columns(4)
                            total_fcst_12m = fc.sum()
                            avg_monthly = fc.mean()
                            peak_m = fc.idxmax().strftime('%b-%y')
                            peak_v = fc.max()

                            with rm1:
                                st.metric("Total 12M Forecast", f"{total_fcst_12m:,.0f} units")
                            with rm2:
                                st.metric("Avg Monthly", f"{avg_monthly:,.0f} units")
                            with rm3:
                                st.metric("Peak Month", peak_m, delta=f"{peak_v:,.0f} units")
                            with rm4:
                                if mape is not None:
                                    # Ubah MAPE menjadi Confidence Level bahasa Bisnis
                                    if mape < 15:
                                        conf_level = "High Confidence 🟢"
                                        mape_color = "normal"
                                    elif mape <= 25:
                                        conf_level = "Moderate 🟡"
                                        mape_color = "off"
                                    else:
                                        conf_level = "Low Confidence 🔴"
                                        mape_color = "inverse"
                                        
                                    st.metric(
                                        "AI Confidence Level",
                                        conf_level,
                                        delta=f"Error (MAPE): {mape:.1f}%",
                                        delta_color=mape_color
                                    )
                                else:
                                    st.metric("AI Confidence Level", "N/A", delta="Need >6mo history")

                            # --- FINANCIAL IMPACT CALCULATION ---
                            existing_plan = ai_get_existing_plan(
                                df_ecomm_forecast,
                                granularity.split(" ")[1],
                                entity_value,
                                ecomm_forecast_month_cols
                            )

                            if not existing_plan.empty:
                                plan_aligned = existing_plan.reindex(fc.index, fill_value=0)
                                gap_qty = total_fcst_12m - plan_aligned.sum()
                                
                                # Dapatkan Harga (Floor Price) untuk menghitung dampak Finansial
                                est_price = 0
                                if granularity == "Per SKU":
                                    price_df = df_product_active[df_product_active['SKU_ID'].astype(str) == str(entity_value)]
                                    if not price_df.empty:
                                        est_price = price_df.iloc[0].get('Floor_Price', 0)
                                else: # Jika per Brand, ambil rata-rata harga brand tersebut
                                    price_df = df_product_active[df_product_active['Brand'] == entity_value]
                                    if not price_df.empty:
                                        est_price = price_df['Floor_Price'].mean()
                                
                                financial_gap = gap_qty * est_price
                                
                                # Tampilkan Insight Box Khusus Finansial
                                if gap_qty > 0:
                                    fin_bg = "#F0FDF4"
                                    fin_border = "#22C55E"
                                    fin_icon = "💰"
                                    fin_title = "Potential Lost Revenue Identified!"
                                    fin_desc = f"AI memprediksi demand **{gap_qty:,.0f} unit lebih tinggi** dari rencana manual. Jika akurasi AI ini benar, kita berisiko kehabisan barang (stockout) dan kehilangan potensi Gross Revenue sebesar **Rp {financial_gap/1e6:,.1f} Juta**."
                                elif gap_qty < 0:
                                    fin_bg = "#FEF2F2"
                                    fin_border = "#EF4444"
                                    fin_icon = "⚠️"
                                    fin_title = "Overstock Risk Identified!"
                                    fin_desc = f"AI memprediksi demand **{abs(gap_qty):,.0f} unit lebih rendah** dari rencana manual. Menahan laju PO sangat disarankan untuk mencegah modal mati (*Trapped Capital*) sebesar **Rp {abs(financial_gap)/1e6:,.1f} Juta**."
                                else:
                                    fin_bg = "#F8FAFC"
                                    fin_border = "#94A3B8"
                                    fin_icon = "⚖️"
                                    fin_title = "Plan Aligned with AI"
                                    fin_desc = "Forecast manual sudah sejalan dengan proyeksi AI. Lanjutkan eksekusi sesuai rencana."

                                st.markdown(f"""
                                <div style="background:{fin_bg}; border-left:5px solid {fin_border}; padding:15px; border-radius:8px; margin: 15px 0;">
                                    <div style="font-weight:800; font-size:1.1rem; color:#1E293B; margin-bottom:5px;">{fin_icon} {fin_title}</div>
                                    <div style="font-size:0.95rem; color:#475569;">{fin_desc}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                plan_aligned = pd.Series([0]*len(fc), index=fc.index)

                            # =====================================================
                            # MAIN FORECAST CHART
                            # =====================================================
                            fig_fc = go.Figure()

                            # Historical — solid area
                            fig_fc.add_trace(go.Scatter(
                                x=ts.index, y=ts.values,
                                name='Historical Sales',
                                mode='lines+markers',
                                line=dict(color='#6366F1', width=2.5),
                                marker=dict(size=5, color='#6366F1'),
                                fill='tozeroy',
                                fillcolor='rgba(99,102,241,0.08)'
                            ))

                            # Confidence interval — shaded band
                            if ci_lo is not None and ci_hi is not None:
                                fig_fc.add_trace(go.Scatter(
                                    x=list(fc.index) + list(fc.index[::-1]),
                                    y=list(ci_hi) + list(ci_lo[::-1]),
                                    fill='toself',
                                    fillcolor='rgba(245,158,11,0.12)',
                                    line=dict(color='rgba(0,0,0,0)'),
                                    name='Confidence Interval',
                                    showlegend=True
                                ))

                            # Forecast line — dashed
                            fig_fc.add_trace(go.Scatter(
                                x=fc.index, y=fc.values,
                                name=f'AI Forecast ({selected_method_name})',
                                mode='lines+markers',
                                line=dict(color='#F59E0B', width=3, dash='dash'),
                                marker=dict(size=7, color='#F59E0B',
                                            line=dict(width=2, color='white')),
                                text=[f"{v:,.0f}" for v in fc.values],
                                textposition='top center'
                            ))

                            # Vertical separator
                            vline_x = ts.index[-1].strftime('%Y-%m-%d')
                            
                            fig_fc.add_shape(
                                type="line",
                                x0=vline_x, x1=vline_x,
                                y0=0, y1=1,
                                xref="x", yref="paper",
                                line=dict(dash="dot", color="gray", width=1.5)
                            )
                            
                            fig_fc.add_annotation(
                                x=vline_x,
                                y=1,
                                xref="x", yref="paper",
                                text="Forecast Start",
                                showarrow=False,
                                xanchor="left",
                                yanchor="top",
                                font=dict(size=11, color="gray"),
                                bgcolor="rgba(255,255,255,0.7)"
                            )

                            fig_fc.update_layout(
                                height=480,
                                title=dict(
                                    text=f"<b>📈 Demand Forecast: {entity_label}</b><br>"
                                         f"<sup>Method: {selected_method_name} | Horizon: 12 Months</sup>",
                                    font=dict(size=15)
                                ),
                                xaxis_title="Month",
                                yaxis_title="Quantity (Units)",
                                hovermode="x unified",
                                plot_bgcolor='white',
                                legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center'),
                                margin=dict(t=80, b=20, l=20, r=20)
                            )
                            st.plotly_chart(fig_fc, use_container_width=True)

                            # =====================================================
                            # BACKTEST + COMPARISON SIDE BY SIDE
                            # =====================================================
                            bc1, bc2 = st.columns(2)

                            with bc1:
                                st.markdown("#### 🧪 Backtest Accuracy (Last 3 Months)")
                                if backtest_df is not None:
                                    # Color APE column
                                    def color_ape(val):
                                        if val < 10: return 'background-color:#d1fae5'
                                        elif val < 25: return 'background-color:#fef3c7'
                                        else: return 'background-color:#fee2e2'

                                    styled = backtest_df.style.applymap(
                                        color_ape, subset=['APE (%)']
                                    ).format({'APE (%)': '{:.1f}%'})

                                    st.dataframe(styled, use_container_width=True, hide_index=True)

                                    if mape is not None:
                                        if mape < 15:
                                            st.success(f"🎯 MAPE {mape:.1f}% — Excellent accuracy")
                                        elif mape < 25:
                                            st.warning(f"⚠️ MAPE {mape:.1f}% — Acceptable, monitor closely")
                                        else:
                                            st.error(f"🚨 MAPE {mape:.1f}% — High error, consider other methods")
                                else:
                                    st.info("ℹ️ Insufficient data for backtest (need >6 months).")

                            with bc2:
                                st.markdown("#### 📋 AI Forecast vs Existing Plan")
                                existing_plan = ai_get_existing_plan(
                                    df_ecomm_forecast,
                                    granularity.split(" ")[1],
                                    entity_value,
                                    ecomm_forecast_month_cols
                                )

                                if not existing_plan.empty:
                                    # Align plan with AI forecast dates
                                    plan_aligned = existing_plan.reindex(fc.index, fill_value=0)

                                    fig_comp = go.Figure()
                                    fig_comp.add_trace(go.Bar(
                                        x=fc.index.strftime('%b-%y'),
                                        y=plan_aligned.values,
                                        name='Existing Plan',
                                        marker_color='rgba(99,102,241,0.5)',
                                        marker_line_color='#6366F1',
                                        marker_line_width=1.5
                                    ))
                                    fig_comp.add_trace(go.Bar(
                                        x=fc.index.strftime('%b-%y'),
                                        y=fc.values,
                                        name='AI Suggestion',
                                        marker_color='rgba(245,158,11,0.7)',
                                        marker_line_color='#F59E0B',
                                        marker_line_width=1.5
                                    ))
                                    fig_comp.update_layout(
                                        height=300,
                                        barmode='group',
                                        plot_bgcolor='white',
                                        legend=dict(orientation="h", y=1.1),
                                        margin=dict(t=40, b=10, l=10, r=10)
                                    )
                                    st.plotly_chart(fig_comp, use_container_width=True)

                                    # Gap analysis
                                    gap_total = fc.sum() - plan_aligned.sum()
                                    gap_pct = (gap_total / plan_aligned.sum() * 100) if plan_aligned.sum() > 0 else 0
                                    if abs(gap_pct) > 10:
                                        icon = "📈" if gap_pct > 0 else "📉"
                                        st.info(f"{icon} AI suggests **{abs(gap_pct):.1f}% {'higher' if gap_pct > 0 else 'lower'}** vs existing plan ({gap_total:+,.0f} units)")
                                    else:
                                        st.success("✅ AI forecast closely aligned with existing plan (±10%)")
                                else:
                                    st.info("ℹ️ No existing plan found for this entity in Forecast_2026_Ecomm.")

                            # =====================================================
                            # OUTPUT TABLE + DOWNLOAD
                            # =====================================================
                            # MtM Heatmap — khusus rolling_mtm method
                            if method_key == "rolling_mtm" and rolling_avg_mtm:
                                st.divider()
                                st.markdown("### 🗓️ MtM Seasonal Ratio Heatmap")
                                st.caption(
                                    "Rata-rata rasio pertumbuhan month-to-month dari seluruh data historis. "
                                    "Nilai > 1.0 = bulan tersebut historically tumbuh vs bulan sebelumnya."
                                )
                            
                                mtm_df = ai_build_mtm_heatmap_data(rolling_avg_mtm)
                            
                                fig_mtm = go.Figure(data=go.Heatmap(
                                    z=mtm_df.values,
                                    x=mtm_df.columns.tolist(),
                                    y=mtm_df.index.tolist(),
                                    colorscale=[
                                        [0.0,  '#ef4444'],   # Merah  = decline (< 0.8)
                                        [0.35, '#fef3c7'],   # Kuning = flat (~1.0)
                                        [0.5,  '#ffffff'],
                                        [0.65, '#d1fae5'],   # Hijau muda = growth
                                        [1.0,  '#10b981']    # Hijau tua = strong growth
                                    ],
                                    zmid=1.0,
                                    text=np.where(
                                        np.isnan(mtm_df.values),
                                        '',
                                        np.round(mtm_df.values, 2).astype(str)
                                    ),
                                    texttemplate="%{text}x",
                                    hovertemplate=(
                                        "From: <b>%{y}</b> → To: <b>%{x}</b><br>"
                                        "Avg Ratio: <b>%{z:.2f}x</b><br>"
                                        "<extra></extra>"
                                    )
                                ))
                            
                                fig_mtm.update_layout(
                                    height=420,
                                    title="Historical MtM Growth Ratio Matrix (Median across all years)",
                                    xaxis_title="To Month",
                                    yaxis_title="From Month",
                                    margin=dict(t=50, b=20, l=20, r=20)
                                )
                            
                                st.plotly_chart(fig_mtm, use_container_width=True)
                            
                                # Highlight top seasonal transitions
                                st.markdown("**📊 Top Seasonal Transitions:**")
                                top_cols = st.columns(3)
                            
                                flat_ratios = [
                                    {'transition': f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][k[0]-1]} → {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][k[1]-1]}",
                                     'ratio': v}
                                    for k, v in rolling_avg_mtm.items()
                                ]
                                flat_ratios_sorted = sorted(flat_ratios, key=lambda x: x['ratio'], reverse=True)
                            
                                for i, item in enumerate(flat_ratios_sorted[:3]):
                                    with top_cols[i]:
                                        st.metric(
                                            f"#{i+1} Peak Growth",
                                            item['transition'],
                                            delta=f"+{(item['ratio']-1)*100:.1f}%"
                                        )
                            st.divider()
                            st.markdown("### 📥 Forecast Output")

                            # Build output dataframe
                            output_rows = []
                            for date, qty in fc.items():
                                existing_qty = 0
                                if not existing_plan.empty and date in existing_plan.index:
                                    existing_qty = existing_plan[date]

                                output_rows.append({
                                    'Entity': entity_value,
                                    'Entity_Type': granularity.split(" ")[1],
                                    'Month': date.strftime('%b-%y'),
                                    'AI_Forecast_Qty': round(qty),
                                    'CI_Lower': round(ci_lo[list(fc.index).index(date)]) if ci_lo is not None else None,
                                    'CI_Upper': round(ci_hi[list(fc.index).index(date)]) if ci_hi is not None else None,
                                    'Existing_Plan_Qty': round(existing_qty),
                                    'Gap_vs_Plan': round(qty - existing_qty),
                                    'Method': selected_method_name,
                                    'Backtest_MAPE_Pct': round(mape, 1) if mape is not None else None
                                })

                            output_df = pd.DataFrame(output_rows)

                            # Display styled table
                            def highlight_gap(val):
                                if isinstance(val, (int, float)):
                                    if val > 0: return 'color: #16a34a; font-weight: bold'
                                    elif val < 0: return 'color: #dc2626; font-weight: bold'
                                return ''

                            st.dataframe(
                                output_df.style.applymap(highlight_gap, subset=['Gap_vs_Plan']),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'AI_Forecast_Qty': st.column_config.NumberColumn('AI Forecast', format='%d'),
                                    'CI_Lower': st.column_config.NumberColumn('CI Lower', format='%d'),
                                    'CI_Upper': st.column_config.NumberColumn('CI Upper', format='%d'),
                                    'Existing_Plan_Qty': st.column_config.NumberColumn('Existing Plan', format='%d'),
                                    'Gap_vs_Plan': st.column_config.NumberColumn('Gap vs Plan', format='%+d'),
                                }
                            )

                            # Real CSV Download
                            csv_bytes = output_df.to_csv(index=False).encode('utf-8')
                            filename = f"AI_Forecast_{entity_value.replace(' ', '_')}_{method_key}_{datetime.now().strftime('%Y%m%d')}.csv"

                            st.download_button(
                                label="📥 Download Forecast as CSV",
                                data=csv_bytes,
                                file_name=filename,
                                mime='text/csv',
                                type='primary',
                                use_container_width=True
                            )

                            st.caption(
                                f"💡 **Tips penggunaan:** Bandingkan kolom *AI_Forecast_Qty* dengan *Existing_Plan_Qty*. "
                                f"Jika MAPE backtest < 20%, AI suggestion layak dijadikan referensi revisi plan. "
                                f"Selalu validasi dengan knowledge lokal (promo, listing baru, dll)."
                            )

# --- TAB 7: PROFITABILITY & MARGIN ANALYSIS (WITH TIER ANALYSIS) ---
with tab7:
    st.subheader("💰 Profitability & Margin Intelligence")
    st.caption("Financial Projection 2026: Revenue, Cost of Goods Sold (COGS), and Gross Margin Analysis")

    # ==============================================================================
    # 1. DATA MERGING & PREPARATION (ECOMM + RESELLER)
    # ==============================================================================
    combined_data = []
    
    if not df_ecomm_forecast.empty:
        fcst_cols = [c for c in df_ecomm_forecast.columns if any(char.isdigit() for char in str(c))]
        if fcst_cols:
            df_e = df_ecomm_forecast.melt(id_vars=['SKU_ID'], value_vars=fcst_cols, var_name='Month_Label', value_name='Qty')
            df_e['Channel'] = 'Ecommerce'
            combined_data.append(df_e)

    if not df_reseller_forecast.empty:
        fcst_cols_res = [c for c in df_reseller_forecast.columns if any(char.isdigit() for char in str(c))]
        if fcst_cols_res:
            df_r = df_reseller_forecast.melt(id_vars=['SKU_ID'], value_vars=fcst_cols_res, var_name='Month_Label', value_name='Qty')
            df_r['Channel'] = 'Reseller'
            combined_data.append(df_r)

    if combined_data:
        df_fin = pd.concat(combined_data, ignore_index=True)
        df_fin['Qty'] = pd.to_numeric(df_fin['Qty'], errors='coerce').fillna(0)
        df_fin = df_fin[df_fin['Qty'] > 0]

        if not df_product.empty:
            cols_price = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Floor_Price', 'Net_Order_Price']
            existing_cols = [c for c in cols_price if c in df_product.columns]
            df_fin = pd.merge(df_fin, df_product[existing_cols], on='SKU_ID', how='left')
            
            if 'Floor_Price' in df_fin.columns: df_fin['Floor_Price'] = pd.to_numeric(df_fin['Floor_Price'], errors='coerce').fillna(0)
            if 'Net_Order_Price' in df_fin.columns: df_fin['Net_Order_Price'] = pd.to_numeric(df_fin['Net_Order_Price'], errors='coerce').fillna(0)
            
            # CALCULATE FINANCIALS
            df_fin['Revenue'] = df_fin['Qty'] * df_fin['Floor_Price']
            df_fin['COGS'] = df_fin['Qty'] * df_fin['Net_Order_Price']
            df_fin['Gross_Margin'] = df_fin['Revenue'] - df_fin['COGS']
            
            # ==============================================================================
            # 2. FINANCIAL HEALTH CARDS (PASTEL STYLE)
            # ==============================================================================
            total_rev = df_fin['Revenue'].sum()
            total_cogs = df_fin['COGS'].sum()
            total_margin = df_fin['Gross_Margin'].sum()
            margin_pct = (total_margin / total_rev * 100) if total_rev > 0 else 0
            
            st.markdown("""
            <style>
                .fin-card { border-radius: 12px; padding: 1.2rem; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.05); transition: transform 0.3s; position: relative; overflow: hidden; }
                .fin-card:hover { transform: translateY(-3px); }
                .fin-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; opacity: 0.9; margin-bottom: 5px; }
                .fin-val { font-size: 1.5rem; font-weight: 800; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
                .fin-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; }
            </style>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            
            with c1: st.markdown(f"""<div class="fin-card" style="background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%);"><div class="fin-title">Total Revenue</div><div class="fin-val">{format_rupiah(total_rev)}</div><div class="fin-sub">Gross Sales</div></div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="fin-card" style="background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);"><div class="fin-title">Total COGS (HPP)</div><div class="fin-val">{format_rupiah(total_cogs)}</div><div class="fin-sub">Cost of Goods</div></div>""", unsafe_allow_html=True)
            with c3: st.markdown(f"""<div class="fin-card" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%);"><div class="fin-title">Gross Margin (Cuan)</div><div class="fin-val">{format_rupiah(total_margin)}</div><div class="fin-sub">Net Profit (Gross)</div></div>""", unsafe_allow_html=True)
            
            color_m = "#10B981" if margin_pct > 30 else "#EF4444"
            with c4: st.markdown(f"""<div class="fin-card" style="background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);"><div class="fin-title">Blended Margin %</div><div class="fin-val">{margin_pct:.1f}%</div><div class="fin-sub">Profitability Ratio</div></div>""", unsafe_allow_html=True)

            # ==============================================================================
            # 3. WATERFALL & CHANNEL MIX
            # ==============================================================================
            st.divider()
            c_water, c_mix = st.columns([2, 1])
            
            with c_water:
                st.subheader("🌊 Financial Waterfall")
                fig_water = go.Figure(go.Waterfall(
                    name = "20", orientation = "v",
                    measure = ["relative", "relative", "total"],
                    x = ["Total Revenue", "COGS (Cost)", "Gross Margin"],
                    textposition = "outside",
                    text = [format_rupiah(total_rev), format_rupiah(-total_cogs), format_rupiah(total_margin)],
                    y = [total_rev, -total_cogs, total_margin],
                    connector = {"line":{"color":"rgb(63, 63, 63)"}},
                    increasing = {"marker":{"color":"#6366F1"}},
                    decreasing = {"marker":{"color":"#F59E0B"}},
                    totals = {"marker":{"color":"#10B981"}} 
                ))
                fig_water.update_layout(height=400, title="Profitability Flow: Revenue to Margin", showlegend=False)
                st.plotly_chart(fig_water, use_container_width=True)
                
            with c_mix:
                st.subheader("🏢 Margin by Channel")
                channel_data = df_fin.groupby('Channel')['Gross_Margin'].sum().reset_index()
                fig_don = px.pie(channel_data, values='Gross_Margin', names='Channel', hole=0.4, 
                                 color_discrete_sequence=['#10B981', '#3B82F6'], title="Profit Contribution")
                fig_don.update_layout(height=400, showlegend=True, legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig_don, use_container_width=True)

            # ==============================================================================
            # 4. PROFITABILITY BY SKU TIER
            # ==============================================================================
            if 'SKU_Tier' in df_fin.columns:
                st.divider()
                st.subheader("💎 Profitability by SKU Tier")
                
                tier_fin = df_fin.groupby('SKU_Tier').agg({'Revenue': 'sum', 'Gross_Margin': 'sum', 'Qty': 'sum'}).reset_index()
                tier_fin['Margin_Pct'] = (tier_fin['Gross_Margin'] / tier_fin['Revenue'] * 100).fillna(0)
                tier_fin = tier_fin.sort_values('Gross_Margin', ascending=False)

                t1, t2 = st.columns(2)

                with t1:
                    fig_tier_val = go.Figure()
                    fig_tier_val.add_trace(go.Bar(
                        y=tier_fin['SKU_Tier'],
                        x=tier_fin['Gross_Margin'],
                        orientation='h',
                        text=[format_rupiah(x) for x in tier_fin['Gross_Margin']],
                        textposition='auto',
                        marker_color='#6366F1',
                        name='Gross Margin (Rp)'
                    ))
                    fig_tier_val.update_layout(
                        height=400, title="💰 Total Gross Margin Contribution by Tier",
                        xaxis_title="Gross Margin (Rp)", yaxis_title="Tier",
                        plot_bgcolor='white', yaxis=dict(autorange="reversed"),
                        xaxis=dict(showticklabels=False)
                    )
                    st.plotly_chart(fig_tier_val, use_container_width=True)

                with t2:
                    colors = ['#10B981' if val >= 40 else '#F59E0B' if val >= 20 else '#EF4444' for val in tier_fin['Margin_Pct']]
                    fig_tier_pct = go.Figure()
                    fig_tier_pct.add_trace(go.Bar(
                        y=tier_fin['SKU_Tier'], x=tier_fin['Margin_Pct'],
                        orientation='h', text=[f"{x:.1f}%" for x in tier_fin['Margin_Pct']],
                        textposition='auto', marker_color=colors, name='Margin %'
                    ))
                    avg_margin_tier = tier_fin['Margin_Pct'].mean()
                    fig_tier_pct.add_vline(x=avg_margin_tier, line_dash="dash", line_color="gray", annotation_text="Avg")
                    fig_tier_pct.update_layout(
                        height=400, title="📊 Efficiency: Margin % by Tier",
                        xaxis_title="Margin %", yaxis_title="Tier",
                        plot_bgcolor='white', yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_tier_pct, use_container_width=True)

            # ==============================================================================
            # 5. PROFITABILITY MATRIX (SCATTER PLOT)
            # ==============================================================================
            st.divider()
            st.subheader("🎯 Profitability Matrix (SKU Level)")
            
            sku_fin = df_fin.groupby(['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier']).agg({'Revenue': 'sum', 'Gross_Margin': 'sum'}).reset_index()
            sku_fin['Margin_Pct'] = (sku_fin['Gross_Margin'] / sku_fin['Revenue'] * 100).fillna(0)
            
            # Khusus Scatter, tooltip kita amankan dengan format_rupiah manual
            sku_fin['Display_Rev'] = sku_fin['Revenue'].apply(format_rupiah)
            sku_fin['Display_Mar'] = sku_fin['Gross_Margin'].apply(format_rupiah)
            
            fig_scat = px.scatter(
                sku_fin, x='Revenue', y='Margin_Pct', size='Gross_Margin', color='Brand',
                hover_name='Product_Name', 
                hover_data={'Revenue': False, 'Margin_Pct': True, 'Display_Rev': True, 'Display_Mar': True, 'SKU_Tier': True, 'Gross_Margin': False},
                size_max=50, title="Matrix: Revenue vs Margin %"
            )
            
            avg_rev = sku_fin['Revenue'].mean()
            avg_mar = sku_fin['Margin_Pct'].mean()
            fig_scat.add_hline(y=avg_mar, line_dash="dash", line_color="gray", annotation_text="Avg Margin")
            fig_scat.add_vline(x=avg_rev, line_dash="dash", line_color="gray", annotation_text="Avg Revenue")
            
            max_x = sku_fin['Revenue'].max() * 1.1
            max_y = sku_fin['Margin_Pct'].max() * 1.1
            fig_scat.add_shape(type="rect", x0=avg_rev, y0=avg_mar, x1=max_x, y1=max_y, fillcolor="rgba(16, 185, 129, 0.1)", layer="below", line_width=0)
            fig_scat.add_shape(type="rect", x0=avg_rev, y0=0, x1=max_x, y1=avg_mar, fillcolor="rgba(245, 158, 11, 0.1)", layer="below", line_width=0)
            
            fig_scat.update_layout(height=500, plot_bgcolor='white', xaxis_type="log", xaxis=dict(showticklabels=not (not st.session_state.unlocked))) 
            st.plotly_chart(fig_scat, use_container_width=True)

            # ==============================================================================
            # 6. PARETO CUAN (80/20 RULE)
            # ==============================================================================
            st.divider()
            st.subheader("📉 Pareto Cuan: Top Profit Contributors")
            
            sku_pareto = sku_fin.sort_values('Gross_Margin', ascending=False)
            sku_pareto['Cum_Margin'] = sku_pareto['Gross_Margin'].cumsum()
            sku_pareto['Cum_Pct'] = sku_pareto['Cum_Margin'] / total_margin * 100
            
            top_30 = sku_pareto.head(30)
            
            fig_par = go.Figure()
            fig_par.add_trace(go.Bar(x=top_30['Product_Name'].str[:20], y=top_30['Gross_Margin'], name='Gross Margin', marker_color='#10B981', hovertext=top_30['Display_Mar'], hoverinfo="x+text"))
            fig_par.add_trace(go.Scatter(x=top_30['Product_Name'].str[:20], y=top_30['Cum_Pct'], name='Cumulative %', yaxis='y2', mode='lines+markers', line=dict(color='#F59E0B')))
            
            fig_par.update_layout(
                height=450, title="Top 30 SKUs by Gross Margin",
                yaxis=dict(title="Gross Margin", showticklabels=not (not st.session_state.unlocked)),
                yaxis2=dict(title="Cumulative %", overlaying='y', side='right', range=[0, 110], showgrid=False),
                xaxis=dict(tickangle=-45), hovermode="x unified", plot_bgcolor='white'
            )
            fig_par.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="80% Threshold", yref="y2")
            st.plotly_chart(fig_par, use_container_width=True)

            # Detail Table
            with st.expander("📋 View Financial Detail Table"):
                disp_fin = df_fin.groupby(['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier']).agg({
                    'Qty': 'sum', 'Revenue': 'sum', 'Gross_Margin': 'sum'
                }).reset_index().sort_values('Gross_Margin', ascending=False)
                
                disp_fin['Margin %'] = (disp_fin['Gross_Margin'] / disp_fin['Revenue'] * 100).fillna(0)
                
                # Gunakan Format Master
                disp_fin['Revenue'] = disp_fin['Revenue'].apply(format_rupiah)
                disp_fin['Gross_Margin'] = disp_fin['Gross_Margin'].apply(format_rupiah)
                disp_fin['Margin %'] = disp_fin['Margin %'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(disp_fin, use_container_width=True)

        else:
            st.warning("⚠️ Data Harga ('Floor_Price', 'Net_Order_Price') tidak ditemukan di Product Master.")
    else:
        st.info("ℹ️ Tidak ada data forecast Ecommerce atau Reseller untuk dianalisis.")

# --- TAB 8: RESELLER PERFORMANCE DASHBOARD ---
with tab8:
    st.subheader("🤝 Reseller Performance Dashboard")
    st.markdown("**Comprehensive Reseller Analytics: Forecast Accuracy, Sales Performance & Inventory Planning**")
    
    # ================ 1. RESELLER PERFORMANCE TABS ================
    tab_res1, tab_res2, tab_res3, tab_res4 = st.tabs([
        "📈 Performance Overview",
        "🎯 Forecast Accuracy",
        "💰 Financial Analysis", 
        "📊 Data Explorer"
    ])
    
    # --- TAB 1: PERFORMANCE OVERVIEW ---
    with tab_res1:
        st.subheader("📊 Reseller Performance Overview")
        
        # Container untuk metrik utama
        metric_container = st.container()
        
        with metric_container:
            col1, col2, col3, col4 = st.columns(4)
            
            # Data untuk bulan Jan 26
            jan_26_data = {}
            
            # 1. Rofo Jan 26
            rofo_jan26 = 0
            if not df_past_rofo_reseller.empty:
                rofo_jan26 = df_past_rofo_reseller[
                    df_past_rofo_reseller['Month_Label'].str.contains('Jan 26', na=False)
                ]['Forecast_Qty'].sum()
            
            # 2. Sales Jan 26
            sales_jan26 = 0
            if not df_sales_reseller.empty:
                sales_jan26 = df_sales_reseller[
                    df_sales_reseller['Month_Label'].str.contains('Jan 26', na=False)
                ]['Sales_Qty'].sum()
            
            # 3. PO Jan 26
            po_jan26 = 0
            if not df_past_po_reseller.empty:
                po_jan26 = df_past_po_reseller[
                    df_past_po_reseller['Month_Label'].str.contains('Jan 26', na=False)
                ]['PO_Qty'].sum()
            
            # 4. Active SKUs - jumlah SKU unik di forecast 2026
            active_skus = len(df_reseller_forecast) if not df_reseller_forecast.empty else 0
            
            # 5. Accuracy Jan 26
            accuracy_jan26 = 0
            if rofo_jan26 > 0:
                accuracy_jan26 = 100 - abs((po_jan26 / rofo_jan26 * 100) - 100)
            
            with col1:
                st.metric("Rofo Jan 26", f"{rofo_jan26:,.0f}")
            
            with col2:
                st.metric("Sales Jan 26", f"{sales_jan26:,.0f}")
            
            with col3:
                st.metric("PO Jan 26", f"{po_jan26:,.0f}")
            
            with col4:
                st.metric("Active SKUs", f"{active_skus:,}")
            
            # Baris kedua untuk accuracy
            col5, col6 = st.columns(2)
            
            with col5:
                st.metric("Jan 26 Accuracy", f"{accuracy_jan26:.1f}%")
            
            with col6:
                # Calculate average sales per active SKU
                avg_sales_per_sku = sales_jan26 / active_skus if active_skus > 0 else 0
                st.metric("Avg Sales/SKU", f"{avg_sales_per_sku:.1f}")
        
        # ROW 2: Triple Comparison Chart - FIXED MONTH ORDER
        st.divider()
        st.subheader("📈 Triple Comparison: Forecast vs PO vs Sales")
        
        if not df_sales_reseller.empty and not df_past_rofo_reseller.empty and not df_past_po_reseller.empty:
            # Aggregate monthly data
            monthly_comparison = []
            
            # Gabungkan semua bulan unik
            all_months_set = set()
            
            # Add months from sales
            if 'Month_Label' in df_sales_reseller.columns:
                all_months_set.update(df_sales_reseller['Month_Label'].unique())
            
            # Add months from rofo
            if 'Month_Label' in df_past_rofo_reseller.columns:
                all_months_set.update(df_past_rofo_reseller['Month_Label'].unique())
            
            # Add months from po
            if 'Month_Label' in df_past_po_reseller.columns:
                all_months_set.update(df_past_po_reseller['Month_Label'].unique())
            
            # Parse bulan untuk sorting
            month_data = []
            for month_label in all_months_set:
                try:
                    # Convert month label to datetime for sorting
                    month_str = str(month_label).strip()
                    if ' ' in month_str:
                        month_part, year_part = month_str.split(' ')
                        month_date = datetime.strptime(f"{month_part[:3]}-{year_part}", "%b-%y")
                    elif '-' in month_str:
                        month_part, year_part = month_str.split('-')
                        month_date = datetime.strptime(f"{month_part[:3]}-{year_part}", "%b-%y")
                    else:
                        continue
                    
                    month_data.append({
                        'label': month_label,
                        'date': month_date,
                        'display': month_date.strftime('%b-%y')
                    })
                except:
                    continue
            
            # Sort by date
            month_data.sort(key=lambda x: x['date'])
            
            # Collect data for sorted months
            for month_info in month_data:
                month_label = month_info['label']
                month_display = month_info['display']
                
                # Sales
                sales_qty = df_sales_reseller[df_sales_reseller['Month_Label'] == month_label]['Sales_Qty'].sum()
                
                # Rofo
                rofo_qty = df_past_rofo_reseller[df_past_rofo_reseller['Month_Label'] == month_label]['Forecast_Qty'].sum()
                
                # PO
                po_qty = df_past_po_reseller[df_past_po_reseller['Month_Label'] == month_label]['PO_Qty'].sum()
                
                # Skip jika semua 0
                if sales_qty == 0 and rofo_qty == 0 and po_qty == 0:
                    continue
                
                monthly_comparison.append({
                    'Month': month_display,
                    'Month_Date': month_info['date'],
                    'Sales': sales_qty,
                    'Rofo': rofo_qty,
                    'PO': po_qty
                })
            
            if monthly_comparison:
                comp_df = pd.DataFrame(monthly_comparison)
                comp_df = comp_df.sort_values('Month_Date')
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=comp_df['Month'],
                    y=comp_df['Rofo'],
                    name='Rofo',
                    marker_color='#667eea',
                    opacity=0.7
                ))
                
                fig.add_trace(go.Bar(
                    x=comp_df['Month'],
                    y=comp_df['PO'],
                    name='PO',
                    marker_color='#FF9800',
                    opacity=0.7
                ))
                
                fig.add_trace(go.Bar(
                    x=comp_df['Month'],
                    y=comp_df['Sales'],
                    name='Sales',
                    marker_color='#4CAF50',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    height=400,
                    title='Reseller Performance: Rofo vs PO vs Sales',
                    xaxis_title='Month',
                    yaxis_title='Quantity',
                    barmode='group',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No comparison data available for the selected period")
        else:
            st.info("ℹ️ Need sales, rofo, and PO data for comparison analysis")
        
        # ROW 3: Brand Performance Matrix - SIMPLE BAR CHART
        st.divider()
        st.subheader("🏷️ Top Performing Brands (Reseller)")
        
        if not df_reseller_forecast.empty and 'Brand' in df_reseller_forecast.columns:
            # Aggregate brand performance
            brand_performance = []
            brands = df_reseller_forecast['Brand'].unique()
            
            for brand in brands:
                brand_data = df_reseller_forecast[df_reseller_forecast['Brand'] == brand]
                
                # Forecast 2026
                forecast_2026 = brand_data[reseller_forecast_cols].sum().sum() if reseller_forecast_cols else 0
                
                # Sales Jan 26 (jika ada)
                sales_jan26 = 0
                if not df_sales_reseller.empty and 'Brand' in df_sales_reseller.columns:
                    brand_sales = df_sales_reseller[
                        (df_sales_reseller['Brand'] == brand) & 
                        (df_sales_reseller['Month_Label'].str.contains('Jan 26'))
                    ]
                    sales_jan26 = brand_sales['Sales_Qty'].sum()
                
                brand_performance.append({
                    'Brand': brand,
                    'Forecast_2026': forecast_2026,
                    'Sales_Jan26': sales_jan26,
                    'SKU_Count': len(brand_data)
                })
            
            if brand_performance:
                brand_df = pd.DataFrame(brand_performance)
                brand_df = brand_df.sort_values('Forecast_2026', ascending=False).head(10)
                
                # Simple Bar Chart (tidak kombinasi line)
                fig_brand = go.Figure()
                
                # Bar: Forecast 2026
                fig_brand.add_trace(go.Bar(
                    x=brand_df['Brand'],
                    y=brand_df['Forecast_2026'],
                    name='Forecast 2026',
                    marker_color='#667eea',
                    text=[f"{x:,.0f}" for x in brand_df['Forecast_2026']],
                    textposition='auto'
                ))
                
                fig_brand.update_layout(
                    height=400,
                    title='Top 10 Brands by Forecast 2026',
                    xaxis_title='Brand',
                    yaxis_title='Forecast Quantity',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_brand, use_container_width=True)
    
    # --- TAB 2: FORECAST ACCURACY ---
    with tab_res2:
        st.subheader("🎯 Reseller Forecast Accuracy Analysis")
        
        if not df_past_rofo_reseller.empty and not df_past_po_reseller.empty:
            # Hitung accuracy per SKU untuk Jan 26
            accuracy_data = []
            
            # Cari SKU yang ada di Jan 26
            rofo_jan26 = df_past_rofo_reseller[df_past_rofo_reseller['Month_Label'].str.contains('Jan 26', na=False)]
            po_jan26 = df_past_po_reseller[df_past_po_reseller['Month_Label'].str.contains('Jan 26', na=False)]
            
            # Gabungkan SKU yang ada di kedua dataset
            common_skus = set(rofo_jan26['SKU_ID']).intersection(set(po_jan26['SKU_ID']))
            
            for sku in common_skus:
                rofo_qty = rofo_jan26[rofo_jan26['SKU_ID'] == sku]['Forecast_Qty'].sum()
                po_qty = po_jan26[po_jan26['SKU_ID'] == sku]['PO_Qty'].sum()
                
                if rofo_qty > 0:
                    accuracy = (min(rofo_qty, po_qty) / max(rofo_qty, po_qty) * 100)
                    status = 'Accurate' if accuracy >= 80 else 'Under' if po_qty < rofo_qty else 'Over'
                    
                    # Get brand, product info, dan sales
                    brand = ''
                    product = ''
                    sales_qty = 0
                    
                    # Cari dari rofo data
                    sku_rofo_data = rofo_jan26[rofo_jan26['SKU_ID'] == sku]
                    if not sku_rofo_data.empty:
                        brand = sku_rofo_data.iloc[0].get('Brand', '')
                        product = sku_rofo_data.iloc[0].get('Product_Name', '')
                    
                    # Cari sales untuk SKU ini di Jan 26
                    if not df_sales_reseller.empty:
                        sales_data = df_sales_reseller[
                            (df_sales_reseller['SKU_ID'] == sku) & 
                            (df_sales_reseller['Month_Label'].str.contains('Jan 26', na=False))
                        ]
                        sales_qty = sales_data['Sales_Qty'].sum() if not sales_data.empty else 0
                    
                    accuracy_data.append({
                        'SKU_ID': sku,
                        'Brand': brand,
                        'Product_Name': product,
                        'Rofo_Qty': rofo_qty,
                        'PO_Qty': po_qty,
                        'Sales_Qty': sales_qty,  # TAMBAHKAN INI
                        'Accuracy': accuracy,
                        'Status': status,
                        'Variance': po_qty - rofo_qty,
                        'Variance_Pct': ((po_qty - rofo_qty) / rofo_qty * 100) if rofo_qty > 0 else 0
                    })
            
            if accuracy_data:
                accuracy_df = pd.DataFrame(accuracy_data)
                
                # Summary Metrics
                col_acc1, col_acc2, col_acc3, col_acc4 = st.columns(4)
                
                with col_acc1:
                    avg_accuracy = accuracy_df['Accuracy'].mean()
                    st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
                
                with col_acc2:
                    accurate_count = len(accuracy_df[accuracy_df['Accuracy'] >= 80])
                    total_count = len(accuracy_df)
                    st.metric("Accurate SKUs", f"{accurate_count}/{total_count}")
                
                with col_acc3:
                    under_count = len(accuracy_df[accuracy_df['Status'] == 'Under'])
                    st.metric("Under Forecast", f"{under_count}")
                
                with col_acc4:
                    over_count = len(accuracy_df[accuracy_df['Status'] == 'Over'])
                    st.metric("Over Forecast", f"{over_count}")
                
                # Accuracy Distribution Chart
                st.divider()
                st.subheader("📊 Accuracy Distribution")
                
                fig_dist = go.Figure()
                
                # Histogram accuracy
                fig_dist.add_trace(go.Histogram(
                    x=accuracy_df['Accuracy'],
                    nbinsx=20,
                    name='Accuracy Distribution',
                    marker_color='#667eea',
                    opacity=0.7
                ))
                
                fig_dist.update_layout(
                    height=300,
                    title='Forecast Accuracy Distribution',
                    xaxis_title='Accuracy %',
                    yaxis_title='Number of SKUs',
                    bargap=0.1
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Detail Table dengan Sales_Qty
                st.divider()
                st.subheader("📋 SKU-Level Accuracy Details")
                
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'Rofo_Qty', 'PO_Qty', 
                              'Sales_Qty', 'Accuracy', 'Status', 'Variance', 'Variance_Pct']
                
                available_cols = [col for col in display_cols if col in accuracy_df.columns]
                
                detail_df = accuracy_df[available_cols].copy()
                detail_df['Accuracy'] = detail_df['Accuracy'].apply(lambda x: f"{x:.1f}%")
                detail_df['Variance_Pct'] = detail_df['Variance_Pct'].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(
                    detail_df.sort_values('Accuracy'),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("📊 No accuracy data available for Jan 26")
        else:
            st.info("ℹ️ Need past rofo and PO data for accuracy analysis")
    
    # --- TAB 3: FINANCIAL ANALYSIS ---
    with tab_res3:
        st.subheader("💰 Reseller Financial Analysis")
        
        # Cek apakah ada data harga
        has_price_data = 'Floor_Price' in df_reseller_forecast.columns
        
        if has_price_data and reseller_forecast_cols:
            # Calculate financial projections - PERBAIKAN: hitung per SKU dengan price masing-masing
            df_financial = df_reseller_forecast.copy()
            
            # Ensure price is numeric
            df_financial['Floor_Price'] = pd.to_numeric(df_financial['Floor_Price'], errors='coerce').fillna(0)
            
            # Calculate monthly revenue projections - FIXED: Hitung revenue per SKU lalu sum
            monthly_revenue = {}
            total_revenue_2026 = 0
            
            # Debug: tampilkan sample data
            st.caption(f"📊 Data sample: {len(df_financial)} SKUs, {len(reseller_forecast_cols)} bulan")
            
            for month_col in reseller_forecast_cols:
                # Hitung revenue untuk bulan ini: SUM(quantity * floor_price per SKU)
                month_revenue = 0
                for idx, row in df_financial.iterrows():
                    qty = pd.to_numeric(row[month_col], errors='coerce')
                    price = row['Floor_Price']
                    if pd.notna(qty) and pd.notna(price):
                        month_revenue += qty * price
                
                monthly_revenue[month_col] = month_revenue
                total_revenue_2026 += month_revenue
            
            # Financial Metrics
            col_fin1, col_fin2, col_fin3 = st.columns(3)
            
            with col_fin1:
                st.metric("Total Revenue 2026", format_rupiah(total_revenue_2026))
            
            with col_fin2:
                avg_monthly_rev = total_revenue_2026 / len(reseller_forecast_cols) if reseller_forecast_cols else 0
                st.metric("Avg Monthly Revenue", format_rupiah(avg_monthly_rev))
            
            with col_fin3:
                if monthly_revenue:
                    peak_month = max(monthly_revenue, key=monthly_revenue.get)
                    peak_rev = monthly_revenue.get(peak_month, 0)
                    st.metric("Peak Revenue Month", format_rupiah(peak_rev), delta=peak_month)
                else:
                    st.metric("Peak Revenue Month", "Rp 0")
            
            # Revenue Trend Chart - FIXED ORDER
            st.divider()
            st.subheader("📈 Monthly Revenue Projection (Feb 26 - Jan 27)")
            
            if monthly_revenue:
                # Sort months chronologically
                revenue_list = []
                for month_col, revenue in monthly_revenue.items():
                    try:
                        month_str = str(month_col).strip().upper()
                        
                        # Parse berbagai format bulan
                        if '_' in month_str:
                            month_part, year_part = month_str.split('_')
                            month_name = month_part[:3]
                            year_num = int(year_part) if len(year_part) == 2 else 2026
                            year_full = 2000 + year_num if year_num < 100 else year_num
                        elif ' ' in month_str:
                            month_part, year_part = month_str.split(' ')
                            month_name = month_part[:3]
                            year_num = int(year_part) if year_part.isdigit() else 2026
                            year_full = 2000 + year_num if year_num < 100 else year_num
                        elif '-' in month_str:
                            month_part, year_part = month_str.split('-')
                            month_name = month_part[:3]
                            year_num = int(year_part) if year_part.isdigit() else 2026
                            year_full = 2000 + year_num if year_num < 100 else year_num
                        else:
                            month_name = month_str[:3]
                            year_full = 2026
                        
                        # Map nama bulan ke angka
                        month_map = {
                            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                        }
                        
                        month_num = month_map.get(month_name, 1)
                        month_date = datetime(year_full, month_num, 1)
                        display_name = f"{month_name}-{str(year_full)[-2:]}"
                        
                        revenue_list.append({
                            'Month': month_col,
                            'Month_Date': month_date,
                            'Revenue': revenue,
                            'Display': display_name
                        })
                    except Exception as e:
                        st.write(f"⚠️ Error parsing {month_col}: {str(e)}")
                        continue
                
                if revenue_list:
                    revenue_df = pd.DataFrame(revenue_list)
                    revenue_df = revenue_df.sort_values('Month_Date')
                    
                    # Filter Feb 26 - Jan 27
                    start_date = datetime(2026, 2, 1)
                    end_date = datetime(2027, 2, 1)  # Termasuk Jan 27
                    
                    revenue_filtered = revenue_df[
                        (revenue_df['Month_Date'] >= start_date) & 
                        (revenue_df['Month_Date'] < end_date)
                    ].copy()
                    
                    # Debug info
                    st.caption(f"📅 Menampilkan {len(revenue_filtered)} bulan (Feb 26 - Jan 27)")
                    
                    if not revenue_filtered.empty:
                        # Urutkan display name sesuai urutan kronologis
                        revenue_filtered = revenue_filtered.sort_values('Month_Date')
                        
                        fig_rev = go.Figure()
                        
                        fig_rev.add_trace(go.Bar(
                            x=revenue_filtered['Display'],
                            y=revenue_filtered['Revenue'],
                            name='Projected Revenue',
                            marker_color='#4CAF50',
                            text=[format_rupiah(x) for x in revenue_filtered['Revenue']],
                            textposition='auto'
                        ))
                        
                        fig_rev.update_layout(
                            height=400,
                            title='Reseller Revenue Projection (Feb 26 - Jan 27)',
                            xaxis_title='Month',
                            yaxis_title='Revenue (Rp)',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_rev, use_container_width=True)
                        
                        # Tampilkan data tabel
                        with st.expander("📋 View Revenue Data"):
                            display_df = revenue_filtered[['Month', 'Display', 'Revenue']].copy()
                            display_df['Revenue'] = display_df['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
                            st.dataframe(display_df)
                    else:
                        st.warning("⚠️ Tidak ada data untuk periode Feb 26 - Jan 27")
                        # Tampilkan semua data yang ada
                        with st.expander("📋 Lihat Semua Data Revenue"):
                            all_df = revenue_df[['Month', 'Display', 'Month_Date', 'Revenue']].copy()
                            all_df['Revenue'] = all_df['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
                            all_df['Month_Date'] = all_df['Month_Date'].dt.strftime('%Y-%m')
                            st.dataframe(all_df)
                else:
                    st.warning("⚠️ Tidak ada data revenue yang bisa di-parse")
            else:
                st.warning("⚠️ Tidak ada data revenue")
                
            # PERBAIKAN: Revenue by Brand - hitung dengan benar per SKU
            st.divider()
            st.subheader("🏷️ Revenue Contribution by Brand")
            
            if 'Brand' in df_financial.columns:
                # Hitung revenue per brand
                brand_revenue_dict = {}
                
                for brand in df_financial['Brand'].unique():
                    brand_data = df_financial[df_financial['Brand'] == brand]
                    brand_rev = 0
                    
                    # Hitung revenue untuk semua bulan
                    for month_col in reseller_forecast_cols:
                        for idx, row in brand_data.iterrows():
                            qty = pd.to_numeric(row[month_col], errors='coerce')
                            price = row['Floor_Price']
                            if pd.notna(qty) and pd.notna(price):
                                brand_rev += qty * price
                    
                    brand_revenue_dict[brand] = {
                        'Revenue': brand_rev,
                        'SKU_Count': len(brand_data),
                        'Avg_Price': brand_data['Floor_Price'].mean() if not brand_data['Floor_Price'].isna().all() else 0
                    }
                
                if brand_revenue_dict:
                    # Convert to dataframe
                    brand_revenue_list = []
                    for brand, data in brand_revenue_dict.items():
                        brand_revenue_list.append({
                            'Brand': brand,
                            'Revenue': data['Revenue'],
                            'SKU_Count': data['SKU_Count'],
                            'Avg_Price': data['Avg_Price']
                        })
                    
                    brand_rev_df = pd.DataFrame(brand_revenue_list).sort_values('Revenue', ascending=False)
                    
                    fig_brand_rev = go.Figure()
                    
                    fig_brand_rev.add_trace(go.Bar(
                        x=brand_rev_df['Brand'],
                        y=brand_rev_df['Revenue'],
                        name='Revenue',
                        marker_color='#9C27B0',
                        text=[format_rupiah(x) for x in brand_rev_df['Revenue']],
                        textposition='auto'
                    ))
                    
                    fig_brand_rev.update_layout(
                        height=400,
                        title='Brand Revenue Contribution 2026',
                        xaxis_title='Brand',
                        yaxis_title='Revenue (Rp)'
                    )
                    
                    st.plotly_chart(fig_brand_rev, use_container_width=True)
                    
                    # Tampilkan tabel ringkasan
                    with st.expander("📋 Brand Revenue Summary"):
                        summary_df = brand_rev_df.copy()
                        summary_df['Revenue'] = summary_df['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
                        summary_df['Avg_Price'] = summary_df['Avg_Price'].apply(lambda x: f"Rp {x:,.0f}")
                        summary_df['Revenue_Share'] = (brand_rev_df['Revenue'] / total_revenue_2026 * 100).apply(lambda x: f"{x:.1f}%")
                        st.dataframe(summary_df[['Brand', 'SKU_Count', 'Revenue', 'Revenue_Share', 'Avg_Price']])
        
        else:
            if not has_price_data:
                st.info("ℹ️ Add 'Floor_Price' column to Reseller forecast data for financial analysis")
            else:
                st.info("ℹ️ No forecast columns available for financial analysis")
    
    # --- TAB 4: DATA EXPLORER ---
    with tab_res4:
        st.subheader("📊 Reseller Data Explorer")
        
        # Tabs for different datasets
        exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
            "Forecast 2026",
            "Sales History",
            "Past Rofo",
            "Past PO"
        ])
        
        with exp_tab1:
            st.markdown("**Forecast 2026 Data**")
            if not df_reseller_forecast.empty:
                # Filter controls
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    exp_brands = []
                    if 'Brand' in df_reseller_forecast.columns:
                        exp_brands = st.multiselect(
                            "Filter Brands",
                            options=df_reseller_forecast['Brand'].unique().tolist(),
                            default=[],
                            key="exp_brands_fcst"
                        )
                
                with exp_col2:
                    exp_months = st.multiselect(
                        "Months to Show",
                        options=reseller_forecast_cols,
                        default=reseller_forecast_cols[:6] if reseller_forecast_cols else [],
                        key="exp_months_fcst"
                    )
                
                # Apply filters
                df_exp = df_reseller_forecast.copy()
                if exp_brands and 'Brand' in df_exp.columns:
                    df_exp = df_exp[df_exp['Brand'].isin(exp_brands)]
                
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Floor_Price']
                if exp_months:
                    display_cols.extend(exp_months)
                
                # Filter available columns
                available_cols = [col for col in display_cols if col in df_exp.columns]
                
                st.dataframe(
                    df_exp[available_cols].head(100),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No forecast data available")
        
        with exp_tab2:
            st.markdown("**Sales History Data**")
            if not df_sales_reseller.empty:
                st.dataframe(
                    df_sales_reseller.sort_values('Month', ascending=False).head(100),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No sales data available")
        
        with exp_tab3:
            st.markdown("**Past Rofo Data**")
            if not df_past_rofo_reseller.empty:
                st.dataframe(
                    df_past_rofo_reseller,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No past rofo data available")
        
        with exp_tab4:
            st.markdown("**Past PO Data**")
            if not df_past_po_reseller.empty:
                st.dataframe(
                    df_past_po_reseller,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No past PO data available")
        
        # --- DUMMY DOWNLOAD BUTTONS (SECURITY LOCK) ---
        st.divider()
        st.subheader("📥 Download Data")
        
        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
        
        # Fungsi Helper untuk memanggil Alarm
        def show_security_alert():
            st.error("🔒 **SECURITY ALERT: RESTRICTED ACCESS**")
            st.warning("Data extraction & CSV download features are disabled in this environment to protect company confidentiality.")
            st.toast("Access Denied: Enterprise Security Policy.", icon="🚫")
        
        with col_dl1:
            if not df_reseller_forecast.empty:
                if st.button("📥 Download Forecast 2026", use_container_width=True, key="dl_fcst_fake"):
                    show_security_alert()
        
        with col_dl2:
            if not df_sales_reseller.empty:
                if st.button("📥 Download Sales", use_container_width=True, key="dl_sales_fake"):
                    show_security_alert()
        
        with col_dl3:
            if not df_past_rofo_reseller.empty:
                if st.button("📥 Download Past Rofo", use_container_width=True, key="dl_rofo_fake"):
                    show_security_alert()
        
        with col_dl4:
            if not df_past_po_reseller.empty:
                if st.button("📥 Download Past PO", use_container_width=True, key="dl_po_fake"):
                    show_security_alert()
        
# --- TAB 9: FULFILLMENT COST ANALYSIS (UNIT ECONOMICS) ---
with tab9:
    st.subheader("🚚 Fulfillment Cost Intelligence")
    st.caption("Operational Efficiency: Cost per Order (CPO), Contribution, and Unit Economics Analysis")

    # ==============================================================================
    # 1. DATA PREPARATION
    # ==============================================================================
    df_bs = all_data.get('fulfillment', pd.DataFrame())

    if not df_bs.empty:
        # Sort kronologis
        if 'Month_Date' not in df_bs.columns:
             # Fallback parsing jika kolom date belum ada
             df_bs['Month_Date'] = pd.to_datetime(df_bs['Month'], format='%b-%y', errors='coerce')
        
        df_bs = df_bs.sort_values('Month_Date')

        # CALCULATE METRICS (Termasuk CPO)
        # Pastikan numeric
        num_cols = ['Total Order(BS)', 'Total Cost', 'GMV (Fullfil By BS)', 'GMV Total (MP)']
        for c in num_cols:
            if c in df_bs.columns:
                df_bs[c] = pd.to_numeric(df_bs[c], errors='coerce').fillna(0)

        # 1. Cost Per Order (CPO)
        df_bs['CPO'] = df_bs.apply(
            lambda x: x['Total Cost'] / x['Total Order(BS)'] if x['Total Order(BS)'] > 0 else 0, axis=1
        )

        # 2. Basket Size (BSA) - Recalculate to be safe
        df_bs['BSA_Calc'] = df_bs.apply(
            lambda x: x['GMV (Fullfil By BS)'] / x['Total Order(BS)'] if x['Total Order(BS)'] > 0 else 0, axis=1
        )

        # 3. Contribution %
        df_bs['Contrib_Pct'] = df_bs.apply(
            lambda x: (x['GMV (Fullfil By BS)'] / x['GMV Total (MP)'] * 100) if x['GMV Total (MP)'] > 0 else 0, axis=1
        )

        # Get Last Month Data
        last_row = df_bs.iloc[-1]
        prev_row = df_bs.iloc[-2] if len(df_bs) > 1 else last_row

        # ==============================================================================
        # 2. EFFICIENCY KPI CARDS (PASTEL GRADIENT)
        # ==============================================================================
        
        # Helper Format
        def fmt_kpi(val, is_money=False):
            if is_money:
                if val >= 1e9: return f"Rp {val/1e9:,.1f} M"
                elif val >= 1e6: return f"Rp {val/1e6:,.1f} Jt"
                return f"Rp {val:,.0f}"
            else:
                return f"{val:,.0f}"

        # Calculate Deltas
        d_cpo = last_row['CPO'] - prev_row['CPO']
        d_gmv = last_row['GMV (Fullfil By BS)'] - prev_row['GMV (Fullfil By BS)']
        d_ord = last_row['Total Order(BS)'] - prev_row['Total Order(BS)']
        d_share = last_row['Contrib_Pct'] - prev_row['Contrib_Pct']

        # CSS (Reused for Consistency)
        st.markdown("""
        <style>
            .eff-card {
                border-radius: 12px; padding: 1.2rem; color: white;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05); transition: transform 0.3s;
                position: relative; overflow: hidden;
            }
            .eff-card:hover { transform: translateY(-3px); }
            .eff-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; opacity: 0.9; margin-bottom: 5px; }
            .eff-val { font-size: 1.6rem; font-weight: 800; margin-bottom: 5px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
            .eff-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; display: flex; align-items: center; gap: 5px; }
            .eff-badge { background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            bg_cpo = "linear-gradient(135deg, #10B981 0%, #059669 100%)" if d_cpo <= 0 else "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)"
            icon_cpo = "▼" if d_cpo <= 0 else "▲"
            st.markdown(f"""
            <div class="eff-card" style="background: {bg_cpo};">
                <div class="eff-title">Cost Per Order (CPO)</div>
                <div class="eff-val">{format_rupiah(last_row['CPO'])}</div>
                <div class="eff-sub">
                    <span class="eff-badge">{icon_cpo} {format_rupiah(abs(d_cpo))}</span> vs Last Mo
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            # Total Orders (Volume) - Blue
            icon_ord = "▲" if d_ord >= 0 else "▼"
            st.markdown(f"""
            <div class="eff-card" style="background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);">
                <div class="eff-title">Total Processed Orders</div>
                <div class="eff-val">{last_row['Total Order(BS)']:,.0f}</div>
                <div class="eff-sub">
                    <span class="eff-badge">{icon_ord} {abs(d_ord):,.0f}</span> Transactions
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            # Contribution - Indigo
            icon_share = "▲" if d_share >= 0 else "▼"
            st.markdown(f"""
            <div class="eff-card" style="background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%);">
                <div class="eff-title">BS Contribution %</div>
                <div class="eff-val">{last_row['Contrib_Pct']:.1f}%</div>
                <div class="eff-sub">
                    <span class="eff-badge">{icon_share} {abs(d_share):.1f}%</span> of Total GMV
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="eff-card" style="background: linear-gradient(135deg, #F97316 0%, #C2410C 100%);">
                <div class="eff-title">Total Fulfillment Cost</div>
                <div class="eff-val">{format_rupiah(last_row['Total Cost'])}</div>
                <div class="eff-sub">Ratio: {last_row['%Cost']:.1f}% of GMV</div>
            </div>
            """, unsafe_allow_html=True)

        # ==============================================================================
        # 3. UNIT ECONOMICS CHART (NEW!)
        # ==============================================================================
        st.divider()
        st.subheader("⚖️ Unit Economics: Basket Size vs Cost per Order")
        st.caption("Membandingkan rata-rata nilai belanja customer (BSA) dengan biaya per transaksi (CPO). **Gap yang melebar (BSA naik, CPO turun) = Efisiensi Meningkat.**")

        fig_unit = go.Figure()

        # Line: Basket Size (Higher is better)
        fig_unit.add_trace(go.Scatter(
            x=df_bs['Month'], y=df_bs['BSA_Calc'],
            name='Basket Size (BSA)',
            mode='lines+markers',
            line=dict(color='#6366F1', width=3), # Indigo
            marker=dict(size=8, color='#6366F1'),
            hovertemplate='BSA: Rp %{y:,.0f}'
        ))

        # Line: Cost Per Order (Lower is better)
        fig_unit.add_trace(go.Scatter(
            x=df_bs['Month'], y=df_bs['CPO'],
            name='Cost Per Order (CPO)',
            mode='lines+markers',
            line=dict(color='#EF4444', width=3, dash='dot'), # Red Dotted
            marker=dict(size=8, color='#EF4444', symbol='diamond'),
            hovertemplate='CPO: Rp %{y:,.0f}',
            yaxis='y2' # Dual Axis biar skalanya pas
        ))

        fig_unit.update_layout(
            height=450,
            xaxis_title="Month",
            yaxis=dict(title="Basket Size (Rp)", showgrid=False),
            yaxis2=dict(
                title="Cost per Order (Rp)", 
                overlaying='y', side='right', 
                showgrid=True, gridcolor='rgba(0,0,0,0.05)'
            ),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
            plot_bgcolor='white'
        )
        st.plotly_chart(fig_unit, use_container_width=True)

        # ==============================================================================
        # 4. EFFICIENCY TREND & SCALE ANALYSIS
        # ==============================================================================
        st.divider()
        col_scale, col_ratio = st.columns(2)

        with col_scale:
            st.subheader("📦 Volume vs Cost Trend")
            st.caption("Hubungan antara kenaikan volume order dengan total biaya.")
            
            fig_eff = go.Figure()
            
            # Bar: Total Orders
            fig_eff.add_trace(go.Bar(
                x=df_bs['Month'], y=df_bs['Total Order(BS)'],
                name='Total Orders',
                marker_color='#3B82F6', # Blue
                opacity=0.6
            ))
            
            # Line: Total Cost
            fig_eff.add_trace(go.Scatter(
                x=df_bs['Month'], y=df_bs['Total Cost'],
                name='Total Cost (Rp)',
                mode='lines+markers',
                line=dict(color='#F97316', width=3), # Orange
                yaxis='y2'
            ))
            
            fig_eff.update_layout(
                height=400,
                xaxis_title="Month",
                yaxis=dict(title="Orders (Qty)", showgrid=False),
                yaxis2=dict(title="Total Cost (Rp)", overlaying='y', side='right', showgrid=True),
                legend=dict(orientation="h", y=1.1),
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        with col_ratio:
            st.subheader("📉 % Cost Ratio Trend")
            st.caption("Persentase biaya terhadap GMV (Target: Semakin rendah semakin baik).")
            
            fig_ratio = go.Figure()
            
            fig_ratio.add_trace(go.Scatter(
                x=df_bs['Month'], y=df_bs['%Cost'],
                mode='lines+markers',
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)', # Soft Green fill
                line=dict(color='#10B981', width=3), # Emerald
                marker=dict(size=8),
                name='% Cost'
            ))
            
            # Add Average Line
            avg_cost_pct = df_bs['%Cost'].mean()
            fig_ratio.add_hline(y=avg_cost_pct, line_dash="dash", line_color="gray", annotation_text="Avg Ratio")
            
            fig_ratio.update_layout(
                height=400,
                xaxis_title="Month",
                yaxis=dict(title="% Cost Ratio", range=[0, max(df_bs['%Cost'])*1.2]),
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

        # ==============================================================================
        # 5. MARKET SHARE VISUALIZATION
        # ==============================================================================
        st.divider()
        st.subheader("🏢 Market Share: BS vs Non-BS")
        
        # Calculate Non-BS GMV
        df_bs['GMV Non-BS'] = df_bs['GMV Total (MP)'] - df_bs['GMV (Fullfil By BS)']
        
        fig_share = go.Figure()
        
        fig_share.add_trace(go.Bar(
            x=df_bs['Month'], y=df_bs['GMV (Fullfil By BS)'],
            name='Fulfilled by BS',
            marker_color='#6366F1' # Indigo
        ))
        
        fig_share.add_trace(go.Bar(
            x=df_bs['Month'], y=df_bs['GMV Non-BS'],
            name='Non-BS',
            marker_color='#E5E7EB' # Light Gray
        ))
        
        fig_share.update_layout(
            height=400,
            title="Monthly GMV Contribution (Stacked)",
            barmode='stack',
            plot_bgcolor='white',
            yaxis=dict(title="GMV (Rp)"),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_share, use_container_width=True)

        # Detail Data
        with st.expander("📋 View Detail Data"):
            disp_cols = ['Month', 'Total Order(BS)', 'GMV (Fullfil By BS)', 'Total Cost', 'CPO', '%Cost']
            df_disp = df_bs[disp_cols].copy()
            
            # Formatting
            df_disp['Total Cost'] = df_disp['Total Cost'].apply(lambda x: f"Rp {x:,.0f}")
            df_disp['GMV (Fullfil By BS)'] = df_disp['GMV (Fullfil By BS)'].apply(lambda x: f"Rp {x:,.0f}")
            df_disp['CPO'] = df_disp['CPO'].apply(lambda x: f"Rp {x:,.0f}")
            df_disp['%Cost'] = df_disp['%Cost'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(df_disp, use_container_width=True)

    else:
        st.warning("⚠️ Data 'BS_Fullfilment_Cost' belum tersedia atau format tidak sesuai.")

# --- TAB 10: DATA EXPLORER (MOVED TO END & SECURED) ---
with tab10:
    st.subheader("📋 Raw Data Explorer")
    st.caption("Akses view-only ke Master Data & Database historis.")
    
    dataset_options = {
        "Product Master": df_product,
        "Active Products": df_product_active,
        "Sales Data": df_sales,
        "Forecast Data": df_forecast,
        "PO Data": df_po,
        "Stock Data": df_stock,
        "Financial Data": df_financial,
        "Inventory Financial": df_inventory_financial
    }
    
    selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()))
    df_selected = dataset_options[selected_dataset]
    
    if not df_selected.empty:
        # Ensure Product_Name is shown alongside SKU_ID if available
        if 'SKU_ID' in df_selected.columns and 'Product_Name' in df_selected.columns:
            # Reorder columns to show SKU_ID and Product_Name first
            cols = list(df_selected.columns)
            if 'Product_Name' in cols:
                cols.remove('Product_Name')
                cols.insert(1, 'Product_Name')
            df_selected = df_selected[cols]
        
        # Data info
        st.write(f"**Rows:** {df_selected.shape[0]:,} | **Columns:** {df_selected.shape[1]}")
        
        # Column selector
        if st.checkbox("Select Columns", False):
            all_columns = df_selected.columns.tolist()
            selected_columns = st.multiselect("Choose columns:", all_columns, default=all_columns[:10])
            df_display = df_selected[selected_columns]
        else:
            df_display = df_selected
        
        # Data preview
        st.dataframe(
            df_display,
            use_container_width=True,
            height=500
        )
        
        # --- DUMMY DOWNLOAD BUTTON (SECURITY LOCK) ---
        st.divider()
        if st.button("📥 Download CSV", use_container_width=True, type="secondary"):
            st.error("🔒 **SECURITY ALERT: RESTRICTED ACCESS**")
            st.warning("Data extraction & CSV download features are disabled in this environment to protect company confidentiality and raw data integrity. Please request clearance from the S&OP Admin.")
            st.toast("Access Denied: Enterprise Security Policy.", icon="🚫")
            
    else:
        st.warning("No data available for selected dataset")

# --- FOOTER ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <p>🚀 <strong>Inventory Intelligence Dashboard v6.0</strong> | Professional Inventory Control & Financial Analytics</p>
    <p>✅ Product Auto-Lookup | ✅ Financial Analysis | ✅ Inventory Analysis</p>
    <p>💰 Profitability Dashboard | 📊 Seasonality Analysis | 🎯 Margin Segmentation</p>
    <p>📈 Data since January 2025 | 🔄 Real-time Google Sheets Integration</p>
</div>
""", unsafe_allow_html=True)
