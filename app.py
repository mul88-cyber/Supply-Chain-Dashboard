# Simpan kode berikut dalam file baru dengan nama app_fixed.py

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
import warnings
from tenacity import retry, stop_after_attempt, wait_exponential
import math
warnings.filterwarnings('ignore')

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
st.caption(f"🚀 Inventory Control & Forecast Analytics - D2C Demand Planner Mulyanto | Real-time Insights | Updated: {datetime.now().strftime('%d %B %Y %H:%M')}")

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
    
    sheet_id = "1jcs8L0CysdzxemPz1EYVVfVhsSR-ik46khIw5jhhBgw"
    data = {}

    # --- HELPER: Baca Sheet Manual ---
    def safe_read_stock_sheet(sheet_name):
        try:
            ws = _client.open_by_key(sheet_id).worksheet(sheet_name)
            raw_data = ws.get_all_values()
            if len(raw_data) < 2: return pd.DataFrame()
            headers = [str(h).strip() for h in raw_data[0]]
            df = pd.DataFrame(raw_data[1:], columns=headers)
            df = df.loc[:, df.columns != '']
            return df
        except: return pd.DataFrame()

    try:
        # 1. PRODUCT MASTER
        ws_prod = _client.open_by_key(sheet_id).worksheet("Product_Master")
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
        ws_sales = _client.open_by_key(sheet_id).worksheet("Sales")
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
        ws_rofo = _client.open_by_key(sheet_id).worksheet("Rofo")
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
        ws_po = _client.open_by_key(sheet_id).worksheet("PO")
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
            ws_ecomm = _client.open_by_key(sheet_id).worksheet("Forecast_2026_Ecomm")
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
            ws_reseller = _client.open_by_key(sheet_id).worksheet("Forecast_2026_Reseller")
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
            ws_bs = _client.open_by_key(sheet_id).worksheet("BS_Fullfilment_Cost")
            df_bs = pd.DataFrame(ws_bs.get_all_records())
            
            # Cleaning Headers & Data
            df_bs.columns = [c.strip() for c in df_bs.columns]
            
            def clean_currency(x):
                if isinstance(x, str):
                    return pd.to_numeric(x.replace(',', '').replace('%', ''), errors='coerce')
                return x

            numeric_cols = ['Total Order(BS)', 'GMV (Fullfil By BS)', 'GMV Total (MP)', 'Total Cost', 'BSA', '%Cost']
            
            for col in numeric_cols:
                if col in df_bs.columns:
                    df_bs[col] = df_bs[col].apply(clean_currency).fillna(0)
            
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
    sheet_id = "1jcs8L0CysdzxemPz1EYVVfVhsSR-ik46khIw5jhhBgw"
    reseller_data = {}
    
    try:
        # 1. FORECAST 2026 RESELLER
        ws_fcst = _client.open_by_key(sheet_id).worksheet("Forecast_2026_Reseller")
        df_fcst_raw = pd.DataFrame(ws_fcst.get_all_records())
        df_fcst_raw.columns = [col.strip() for col in df_fcst_raw.columns]
        
        all_month_cols = [c for c in df_fcst_raw.columns if any(m in c.upper() for m in 
                      ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        
        hist_cols = []
        fcst_cols = []
        
        for col in all_month_cols:
            col_str = str(col).upper()
            if '25' in col_str or '2025' in col_str:
                hist_cols.append(col)
            else:
                fcst_cols.append(col)
        
        for col in all_month_cols:
            df_fcst_raw[col] = pd.to_numeric(df_fcst_raw[col], errors='coerce').fillna(0)
        
        reseller_data['forecast'] = df_fcst_raw
        reseller_data['forecast_month_cols'] = fcst_cols
        reseller_data['historical_month_cols'] = hist_cols
        
        # 2. SALES RESELLER
        try:
            ws_sales = _client.open_by_key(sheet_id).worksheet("Sales_Reseller")
            df_sales_raw = pd.DataFrame(ws_sales.get_all_records())
            df_sales_raw.columns = [col.strip() for col in df_sales_raw.columns]
            
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
            ws_rofo = _client.open_by_key(sheet_id).worksheet("Past_Rofo_Reseller")
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
            ws_po = _client.open_by_key(sheet_id).worksheet("Past_PO_Reseller")
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
        required_price_cols = ['Floor_Price', 'Net_Order_Price']
        price_cols_exist = all(col in df_product.columns for col in required_price_cols)
        
        if not price_cols_exist:
            st.warning("⚠️ Price columns missing in Product Master")
            return pd.DataFrame()
        
        if 'Floor_Price' not in df_sales.columns or 'Net_Order_Price' not in df_sales.columns:
            df_sales = add_product_info_to_data(df_sales, df_product)
        
        df_sales['Floor_Price'] = df_sales['Floor_Price'].fillna(0)
        df_sales['Net_Order_Price'] = df_sales['Net_Order_Price'].fillna(0)
        
        df_sales['Revenue'] = df_sales['Sales_Qty'] * df_sales['Floor_Price']
        df_sales['Cost'] = df_sales['Sales_Qty'] * df_sales['Net_Order_Price']
        df_sales['Gross_Margin'] = df_sales['Revenue'] - df_sales['Cost']
        df_sales['Margin_Percentage'] = np.where(
            df_sales['Revenue'] > 0,
            (df_sales['Gross_Margin'] / df_sales['Revenue'] * 100),
            0
        )
        
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
        if 'Floor_Price' not in df_product.columns or 'Net_Order_Price' not in df_product.columns:
            return pd.DataFrame()
        
        if 'Floor_Price' not in df_stock.columns or 'Net_Order_Price' not in df_stock.columns:
            df_stock = add_product_info_to_data(df_stock, df_product)
        
        df_stock['Floor_Price'] = df_stock['Floor_Price'].fillna(0)
        df_stock['Net_Order_Price'] = df_stock['Net_Order_Price'].fillna(0)
        
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
        df_financial['Year'] = df_financial['Month'].dt.year
        df_financial['Month_Num'] = df_financial['Month'].dt.month
        df_financial['Month_Name'] = df_financial['Month'].dt.strftime('%b')
        
        seasonal_pattern = df_financial.groupby(['Month_Num', 'Month_Name']).agg({
            'Revenue': 'mean',
            'Gross_Margin': 'mean',
            'Sales_Qty': 'mean'
        }).reset_index()
        
        overall_avg_revenue = seasonal_pattern['Revenue'].mean()
        seasonal_pattern['Seasonal_Index_Revenue'] = seasonal_pattern['Revenue'] / overall_avg_revenue
        
        overall_avg_margin = seasonal_pattern['Gross_Margin'].mean()
        seasonal_pattern['Seasonal_Index_Margin'] = seasonal_pattern['Gross_Margin'] / overall_avg_margin
        
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
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        common_months = sorted(set(forecast_months) & set(po_months))
        
        if not common_months:
            return {}
        
        bias_results = []
        
        for month in common_months:
            df_f_month = df_forecast[df_forecast['Month'] == month]
            df_p_month = df_po[df_po['Month'] == month]
            
            df_merged = pd.merge(
                df_f_month[['SKU_ID', 'Forecast_Qty']],
                df_p_month[['SKU_ID', 'PO_Qty']],
                on='SKU_ID',
                how='inner'
            )
            
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
    """Calculate performance for each month separately - HANYA SKU dengan Forecast_Qty > 0"""
    
    monthly_performance = {}
    
    if df_forecast.empty or df_po.empty:
        return monthly_performance
    
    try:
        df_forecast = add_product_info_to_data(df_forecast, df_product)
        df_po = add_product_info_to_data(df_po, df_product)
        
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        all_months = sorted(set(list(forecast_months) + list(po_months)))
        
        for month in all_months:
            df_forecast_month = df_forecast[
                (df_forecast['Month'] == month) & 
                (df_forecast['Forecast_Qty'] > 0)
            ].copy()
            
            df_po_month = df_po[df_po['Month'] == month].copy()
            
            if df_forecast_month.empty or df_po_month.empty:
                continue
            
            df_merged = pd.merge(
                df_forecast_month,
                df_po_month,
                on=['SKU_ID'],
                how='inner',
                suffixes=('_forecast', '_po')
            )
            
            if not df_merged.empty:
                if 'Product_Name' not in df_merged.columns or 'Brand' not in df_merged.columns:
                    df_merged = add_product_info_to_data(df_merged, df_product)
                
                df_merged['PO_Rofo_Ratio'] = np.where(
                    df_merged['Forecast_Qty'] > 0,
                    (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100,
                    0
                )
                
                conditions = [
                    df_merged['PO_Rofo_Ratio'] < 80,
                    (df_merged['PO_Rofo_Ratio'] >= 80) & (df_merged['PO_Rofo_Ratio'] <= 120),
                    df_merged['PO_Rofo_Ratio'] > 120
                ]
                choices = ['Under', 'Accurate', 'Over']
                df_merged['Accuracy_Status'] = np.select(conditions, choices, default='Unknown')
                
                df_merged['Absolute_Percentage_Error'] = abs(df_merged['PO_Rofo_Ratio'] - 100)
                
                valid_skus = df_merged[df_merged['Forecast_Qty'] > 0]
                if not valid_skus.empty:
                    mape = valid_skus['Absolute_Percentage_Error'].mean()
                else:
                    mape = 0
                    
                monthly_accuracy = 100 - mape
                
                status_counts = df_merged['Accuracy_Status'].value_counts().to_dict()
                total_records = len(df_merged)
                status_percentages = {k: (v/total_records*100) for k, v in status_counts.items()}
                
                monthly_performance[month] = {
                    'accuracy': monthly_accuracy,
                    'mape': mape,
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
        df_stock_agg = df_stock.groupby('SKU_ID').agg({
            'Stock_Qty': 'sum'
        }).reset_index()
        
        df_stock_agg = add_product_info_to_data(df_stock_agg, df_product)
        
        df_sales = add_product_info_to_data(df_sales, df_product)
        
        if not df_sales.empty:
            sales_months = sorted(df_sales['Month'].unique())
            if len(sales_months) >= 3:
                last_3_sales_months = sales_months[-3:]
                df_sales_last_3 = df_sales[df_sales['Month'].isin(last_3_sales_months)].copy()
            else:
                df_sales_last_3 = df_sales.copy()
        
        if not df_sales.empty and not df_sales_last_3.empty:
            avg_monthly_sales = df_sales_last_3.groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
            avg_monthly_sales.columns = ['SKU_ID', 'Avg_Monthly_Sales_3M']
        else:
            avg_monthly_sales = pd.DataFrame(columns=['SKU_ID', 'Avg_Monthly_Sales_3M'])
        
        df_inventory = pd.merge(
            df_stock_agg,
            df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']],
            on='SKU_ID',
            how='left',
            suffixes=('', '_master')
        )
        
        df_inventory = df_inventory.loc[:,~df_inventory.columns.duplicated()]
        
        df_inventory = pd.merge(df_inventory, avg_monthly_sales, on='SKU_ID', how='left')
        df_inventory['Avg_Monthly_Sales_3M'] = df_inventory['Avg_Monthly_Sales_3M'].fillna(0)
        
        df_inventory['Cover_Months'] = np.where(
            df_inventory['Avg_Monthly_Sales_3M'] > 0,
            df_inventory['Stock_Qty'] / df_inventory['Avg_Monthly_Sales_3M'],
            999
        )
        
        conditions = [
            df_inventory['Cover_Months'] < 0.8,
            (df_inventory['Cover_Months'] >= 0.8) & (df_inventory['Cover_Months'] <= 1.5),
            df_inventory['Cover_Months'] > 1.5
        ]
        choices = ['Need Replenishment', 'Ideal/Healthy', 'High Stock']
        df_inventory['Inventory_Status'] = np.select(conditions, choices, default='Unknown')
        
        high_stock_df = df_inventory[df_inventory['Inventory_Status'] == 'High Stock'].copy().sort_values('Cover_Months', ascending=False)
        low_stock_df = df_inventory[df_inventory['Inventory_Status'] == 'Need Replenishment'].copy().sort_values('Cover_Months', ascending=True)
        
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
        df_sales = add_product_info_to_data(df_sales, df_product)
        df_forecast = add_product_info_to_data(df_forecast, df_product)
        df_po = add_product_info_to_data(df_po, df_product)
        
        if 'Status' in df_product.columns:
            active_skus = df_product[df_product['Status'].str.upper() == 'ACTIVE']['SKU_ID'].tolist()
            
            df_sales = df_sales[df_sales['SKU_ID'].isin(active_skus)]
            df_forecast = df_forecast[df_forecast['SKU_ID'].isin(active_skus)]
            if not df_po.empty:
                df_po = df_po[df_po['SKU_ID'].isin(active_skus)]
        
        sales_months = sorted(df_sales['Month'].unique())
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        
        common_months = sorted(set(sales_months) & set(forecast_months) & set(po_months))
        
        if not common_months:
            return results
        
        last_month = common_months[-1]
        
        df_sales_month = df_sales[df_sales['Month'] == last_month].copy()
        df_forecast_month = df_forecast[df_forecast['Month'] == last_month].copy()
        df_po_month = df_po[df_po['Month'] == last_month].copy()
        
        df_forecast_month = df_forecast_month[df_forecast_month['Forecast_Qty'] > 0]
        
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
        
        df_merged = add_product_info_to_data(df_merged, df_product)
        
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
        
        df_merged['Forecast_Deviation'] = abs(df_merged['Sales_vs_Forecast_Ratio'] - 100)
        df_merged['PO_Deviation'] = abs(df_merged['Sales_vs_PO_Ratio'] - 100)
        
        high_deviation_skus = df_merged[
            (df_merged['Forecast_Deviation'] > 30) | 
            (df_merged['PO_Deviation'] > 30)
        ].copy()
        
        high_deviation_skus = high_deviation_skus.sort_values('Forecast_Deviation', ascending=False)
        
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
        df_forecast = add_product_info_to_data(df_forecast, df_product)
        df_po = add_product_info_to_data(df_po, df_product)
        
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        common_months = sorted(set(forecast_months) & set(po_months))
        
        if not common_months:
            return pd.DataFrame()
        
        last_month = common_months[-1]
        
        df_forecast_month = df_forecast[df_forecast['Month'] == last_month].copy()
        df_po_month = df_po[df_po['Month'] == last_month].copy()
        
        df_merged = pd.merge(
            df_forecast_month,
            df_po_month,
            on=['SKU_ID'],
            how='inner'
        )
        
        if 'Brand' not in df_merged.columns:
            df_merged = add_product_info_to_data(df_merged, df_product)
        
        if 'Brand' not in df_merged.columns:
            return pd.DataFrame()
        
        df_merged['PO_Rofo_Ratio'] = np.where(
            df_merged['Forecast_Qty'] > 0,
            (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100,
            0
        )
        
        conditions = [
            df_merged['PO_Rofo_Ratio'] < 80,
            (df_merged['PO_Rofo_Ratio'] >= 80) & (df_merged['PO_Rofo_Ratio'] <= 120),
            df_merged['PO_Rofo_Ratio'] > 120
        ]
        choices = ['Under', 'Accurate', 'Over']
        df_merged['Accuracy_Status'] = np.select(conditions, choices, default='Unknown')
        
        brand_performance = df_merged.groupby('Brand').agg({
            'SKU_ID': 'count',
            'Forecast_Qty': 'sum',
            'PO_Qty': 'sum',
            'PO_Rofo_Ratio': lambda x: 100 - abs(x - 100).mean()
        }).reset_index()
        
        brand_performance.columns = ['Brand', 'SKU_Count', 'Total_Forecast', 'Total_PO', 'Accuracy']
        
        brand_performance['PO_vs_Forecast_Ratio'] = (brand_performance['Total_PO'] / brand_performance['Total_Forecast'] * 100)
        brand_performance['Qty_Difference'] = brand_performance['Total_PO'] - brand_performance['Total_Forecast']
        
        status_counts = df_merged.groupby(['Brand', 'Accuracy_Status']).size().unstack(fill_value=0).reset_index()
        
        brand_performance = pd.merge(brand_performance, status_counts, on='Brand', how='left')
        
        for status in ['Under', 'Accurate', 'Over']:
            if status not in brand_performance.columns:
                brand_performance[status] = 0
        
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
        
        sku_profitability['Avg_Margin_Per_SKU'] = sku_profitability['Gross_Margin'] / sku_profitability['Sales_Qty']
        sku_profitability['Margin_Percentage'] = np.where(
            sku_profitability['Revenue'] > 0,
            (sku_profitability['Gross_Margin'] / sku_profitability['Revenue'] * 100),
            0
        )
        
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
    
    checks['Total Rows'] = f"📊 {len(df):,} rows"
    checks['Total Columns'] = f"📋 {len(df.columns)} columns"
    
    missing_values = df.isnull().sum().sum()
    missing_pct = (missing_values / (len(df) * len(df.columns)) * 100)
    checks['Missing Values'] = f"⚠️ {missing_values:,} ({missing_pct:.1f}%)" if missing_values > 0 else f"✅ {missing_values:,}"
    
    duplicates = df.duplicated().sum()
    checks['Duplicate Rows'] = f"⚠️ {duplicates:,}" if duplicates > 0 else f"✅ {duplicates:,}"
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        zero_values = (df[numeric_cols] == 0).sum().sum()
        zero_pct = (zero_values / (len(df) * len(numeric_cols)) * 100)
        checks['Zero Values'] = f"📉 {zero_values:,} ({zero_pct:.1f}%)"
    
    if len(numeric_cols) > 0:
        negative_values = (df[numeric_cols] < 0).sum().sum()
        if negative_values > 0:
            checks['Negative Values'] = f"❌ {negative_values:,}"
    
    if 'Month' in df.columns:
        try:
            min_date = df['Month'].min()
            max_date = df['Month'].max()
            checks['Date Range'] = f"📅 {min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}"
        except:
            pass
    
    return checks

# --- ====================================================== ---
# ---                DASHBOARD INITIALIZATION               ---
# --- ====================================================== ---

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
    
    df_ecomm_forecast = all_data.get('ecomm_forecast', pd.DataFrame())
    ecomm_forecast_month_cols = all_data.get('ecomm_forecast_month_cols', [])
    
    df_reseller_forecast = all_data.get('reseller_forecast', pd.DataFrame())
    reseller_all_month_cols = all_data.get('reseller_all_month_cols', [])
    reseller_historical_cols = all_data.get('reseller_historical_cols', [])
    reseller_forecast_cols = all_data.get('reseller_forecast_cols', [])
    
    df_rofo_onwards = df_ecomm_forecast
    rofo_onwards_month_cols = ecomm_forecast_month_cols
    
    # --- LOAD DATA RESELLER LENGKAP (BARU) ---
    with st.spinner('🔄 Loading Reseller Data...'):
        reseller_complete_data = load_reseller_complete_data(client)
        
        if df_reseller_forecast.empty and 'forecast' in reseller_complete_data:
            df_reseller_forecast = reseller_complete_data.get('forecast', pd.DataFrame())
        
        if not reseller_forecast_cols and 'forecast_month_cols' in reseller_complete_data:
            reseller_forecast_cols = reseller_complete_data.get('forecast_month_cols', [])
        
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
    
    # Financial metrics in sidebar
    if not df_financial.empty:
        st.markdown("---")
        st.markdown("### 💰 Financial Overview")
        
        total_revenue = df_financial['Revenue'].sum()
        total_margin = df_financial['Gross_Margin'].sum()
        avg_margin_pct = (total_margin / total_revenue * 100) if total_revenue > 0 else 0
        
        st.metric("Total Revenue", f"Rp {total_revenue:,.0f}")
        st.metric("Total Margin", f"Rp {total_margin:,.0f}")
        st.metric("Avg Margin %", f"{avg_margin_pct:.1f}%")
    
    st.markdown("---")
    
    # Threshold Settings
    st.markdown("### ⚙️ Threshold Settings")
    under_threshold = st.slider("Under Forecast Threshold (%)", 0, 100, 80)
    over_threshold = st.slider("Over Forecast Threshold (%)", 100, 200, 120)
    
    st.markdown("---")
    
    # Inventory Thresholds
    st.markdown("### 📦 Inventory Thresholds")
    low_stock_threshold = st.slider("Low Stock (months)", 0.0, 2.0, 0.8, 0.1)
    high_stock_threshold = st.slider("High Stock (months)", 1.0, 6.0, 1.5, 0.1)
    
    # Financial Thresholds
    st.markdown("---")
    st.markdown("### 💰 Financial Thresholds")
    high_margin_threshold = st.slider("High Margin Threshold (%)", 0, 100, 40)
    low_margin_threshold = st.slider("Low Margin Threshold (%)", 0, 100, 20)
    
    # Dark mode toggle
    st.markdown("---")
    dark_mode = st.checkbox("🌙 Dark Mode", value=False)
    if dark_mode:
        st.markdown("""
        <style>
            .stApp { background-color: #0E1117; color: white; }
            .stDataFrame { background-color: #1E1E1E; }
        </style>
        """, unsafe_allow_html=True)

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

# PERUBAHAN 1: Chart Accuracy Trend di Paling Atas
st.subheader("📈 Accuracy Trend Over Time")

if monthly_performance:
    summary_data = []
    for month, data in sorted(monthly_performance.items()):
        summary_data.append({
            'Month': month,
            'Month_Display': month.strftime('%b-%Y'),
            'Accuracy (%)': data['accuracy'],
            'Under': data['status_counts'].get('Under', 0),
            'Accurate': data['status_counts'].get('Accurate', 0),
            'Over': data['status_counts'].get('Over', 0),
            'Total SKUs': data['total_records'],
            'MAPE': data['mape']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if not summary_df.empty:
        summary_df = summary_df.sort_values('Month')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=summary_df['Month_Display'],
            y=summary_df['Accuracy (%)'],
            mode='lines+markers+text',
            line=dict(color='#667eea', width=4),
            marker=dict(size=12, color='#764ba2'),
            text=summary_df['Accuracy (%)'].apply(lambda x: f"{x:.1f}%"),
            textposition="top center"
        ))
        
        fig.update_layout(
            height=500,
            title_text='<b>Forecast Accuracy Trend Over Time</b>',
            title_x=0.5,
            xaxis_title='<b>Month-Year</b>',
            yaxis_title='<b>Accuracy (%)</b>',
            yaxis_ticksuffix="%",
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# SECTION 1: LAST 3 MONTHS PERFORMANCE (DIPERBESAR)
st.subheader("🎯 Forecast Performance - 3 Bulan Terakhir")

if last_3_months_performance:
    months_display = []
    
    month_cols = st.columns(3)
    
    for i, (month, data) in enumerate(sorted(last_3_months_performance.items())):
        month_name = month.strftime('%b %Y')
        accuracy = data['accuracy']
        
        with month_cols[i]:
            under_count = data['status_counts'].get('Under', 0)
            accurate_count = data['status_counts'].get('Accurate', 0)
            over_count = data['status_counts'].get('Over', 0)
            total_records = data['total_records']
            
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
        
    # TOTAL METRICS - BULAN TERAKHIR (dengan Qty dan persentase)
    st.divider()
    st.subheader("📊 Total Metrics - Bulan Terakhir")
    
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        under_count = last_month_data[last_month_data['Accuracy_Status'] == 'Under']['SKU_ID'].nunique()
        accurate_count = last_month_data[last_month_data['Accuracy_Status'] == 'Accurate']['SKU_ID'].nunique()
        over_count = last_month_data[last_month_data['Accuracy_Status'] == 'Over']['SKU_ID'].nunique()
        total_count_last_month = last_month_data['SKU_ID'].nunique()
        
        under_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Under']['Forecast_Qty'].sum()
        accurate_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Accurate']['Forecast_Qty'].sum()
        over_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Over']['Forecast_Qty'].sum()
        total_forecast_qty = last_month_data['Forecast_Qty'].sum()
        
        under_pct = (under_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        accurate_pct = (accurate_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        over_pct = (over_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        
        under_forecast_pct = (under_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0
        accurate_forecast_pct = (accurate_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0
        over_forecast_pct = (over_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0
    
    col_total1, col_total2, col_total3, col_total4 = st.columns(4)
    
    with col_total1:
        html_under = (
            f'<div style="background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 4px solid #F44336;">'
            f'<div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">UNDER FORECAST</div>'
            f'<div style="font-size: 1.5rem; font-weight: 800; color: #F44336;">{under_count} SKUs</div>'
            f'<div style="font-size: 0.9rem; color: #888;">Qty: {under_forecast_qty:,.0f}</div>'
            f'<div style="font-size: 0.8rem; color: #999;">SKU: {under_pct:.1f}% | Qty: {under_forecast_pct:.1f}%</div>'
            f'</div>'
        )
        st.markdown(html_under, unsafe_allow_html=True)
    
    with col_total2:
        html_accurate = (
            f'<div style="background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 4px solid #4CAF50;">'
            f'<div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">ACCURATE FORECAST</div>'
            f'<div style="font-size: 1.5rem; font-weight: 800; color: #4CAF50;">{accurate_count} SKUs</div>'
            f'<div style="font-size: 0.9rem; color: #888;">Qty: {accurate_forecast_qty:,.0f}</div>'
            f'<div style="font-size: 0.8rem; color: #999;">SKU: {accurate_pct:.1f}% | Qty: {accurate_forecast_pct:.1f}%</div>'
            f'</div>'
        )
        st.markdown(html_accurate, unsafe_allow_html=True)
    
    with col_total3:
        html_over = (
            f'<div style="background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 4px solid #FF9800;">'
            f'<div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">OVER FORECAST</div>'
            f'<div style="font-size: 1.5rem; font-weight: 800; color: #FF9800;">{over_count} SKUs</div>'
            f'<div style="font-size: 0.9rem; color: #888;">Qty: {over_forecast_qty:,.0f}</div>'
            f'<div style="font-size: 0.8rem; color: #999;">SKU: {over_pct:.1f}% | Qty: {over_forecast_pct:.1f}%</div>'
            f'</div>'
        )
        st.markdown(html_over, unsafe_allow_html=True)
    
    with col_total4:
        last_month_accuracy = monthly_performance[last_month]['accuracy']
        html_overall = (
            f'<div style="background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 4px solid #667eea;">'
            f'<div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">OVERALL</div>'
            f'<div style="font-size: 1.8rem; font-weight: 800; color: #667eea;">{last_month_accuracy:.1f}%</div>'
            f'<div style="font-size: 0.9rem; color: #888;">{last_month.strftime("%b %Y")}</div>'
            f'<div style="font-size: 0.8rem; color: #999;">Total SKUs: {total_count_last_month}</div>'
            f'</div>'
        )
        st.markdown(html_overall, unsafe_allow_html=True)
    
    st.caption(f"""
    **Bulan {last_month.strftime('%b %Y')}:** Total Forecast: {total_forecast_qty:,.0f} | Total SKUs: {total_count_last_month} | Overall Accuracy: {last_month_accuracy:.1f}%
    """)
    
    # TOTAL ROFO DAN PO BULAN TERAKHIR
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        total_rofo_last_month = last_month_data['Forecast_Qty'].sum()
        total_po_last_month = last_month_data['PO_Qty'].sum()
        selisih_qty = total_po_last_month - total_rofo_last_month
        selisih_persen = (selisih_qty / total_rofo_last_month * 100) if total_rofo_last_month > 0 else 0
    
    st.divider()
    st.subheader("📈 Total Rofo vs PO vs Sales - Bulan Terakhir")
    
    total_sales_last_month = 0
    sales_vs_rofo_pct = 0
    sales_vs_po_pct = 0
    
    if not df_sales.empty and monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        df_sales_last_month = df_sales[df_sales['Month'] == last_month].copy()
        total_sales_last_month = df_sales_last_month['Sales_Qty'].sum()
        
        if total_rofo_last_month > 0:
            sales_vs_rofo_pct = (total_sales_last_month / total_rofo_last_month * 100)
        
        if total_po_last_month > 0:
            sales_vs_po_pct = (total_sales_last_month / total_po_last_month * 100)
    
    rofo_col1, rofo_col2, rofo_col3, rofo_col4, rofo_col5, rofo_col6 = st.columns(6)
    
    with rofo_col1:
        st.metric(
            "Total Rofo Qty",
            f"{total_rofo_last_month:,.0f}",
            help="Total quantity dari forecast/Rofo bulan terakhir"
        )
    
    with rofo_col2:
        st.metric(
            "Total PO Qty", 
            f"{total_po_last_month:,.0f}",
            help="Total quantity dari Purchase Order bulan terakhir"
        )
    
    with rofo_col3:
        st.metric(
            "Total Sales Qty",
            f"{total_sales_last_month:,.0f}",
            help="Total quantity dari Sales bulan terakhir"
        )
    
    with rofo_col4:
        delta_sales_rofo = f"{sales_vs_rofo_pct-100:+.1f}%" if sales_vs_rofo_pct > 0 else "0%"
        st.metric(
            "Sales/Rofo %",
            f"{sales_vs_rofo_pct:.1f}%",
            delta=delta_sales_rofo,
            delta_color="normal" if 80 <= sales_vs_rofo_pct <= 120 else "off",
            help="Persentase Sales vs Rofo (100% = Sales = Rofo)"
        )
    
    with rofo_col5:
        delta_sales_po = f"{sales_vs_po_pct-100:+.1f}%" if sales_vs_po_pct > 0 else "0%"
        st.metric(
            "Sales/PO %",
            f"{sales_vs_po_pct:.1f}%",
            delta=delta_sales_po,
            delta_color="normal" if 80 <= sales_vs_po_pct <= 120 else "off",
            help="Persentase Sales vs PO (100% = Sales = PO)"
        )
    
    with rofo_col6:
        delta_po_rofo = f"{selisih_persen:+.1f}%"
        st.metric(
            "PO/Rofo %",
            f"{(total_po_last_month/total_rofo_last_month*100 if total_rofo_last_month > 0 else 0):.1f}%",
            delta=delta_po_rofo,
            delta_color="normal" if abs(selisih_persen) < 20 else "off",
            help="Persentase PO vs Rofo (100% = PO = Rofo)"
        )
    
    st.caption(f"""
    **Bulan {last_month.strftime('%b %Y')}:** 
    • **Rofo:** {total_rofo_last_month:,.0f} | 
    • **PO:** {total_po_last_month:,.0f} | 
    • **Sales:** {total_sales_last_month:,.0f} | 
    • **Sales/Rofo:** {sales_vs_rofo_pct:.1f}% | 
    • **Sales/PO:** {sales_vs_po_pct:.1f}% | 
    • **PO/Rofo:** {(total_po_last_month/total_rofo_last_month*100 if total_rofo_last_month > 0 else 0):.1f}%
    """)
else:
    st.warning("⚠️ Insufficient data for monthly performance analysis")

st.divider()
# SECTION 2: LAST MONTH EVALUATION (UNDER & OVER ONLY)
st.subheader("📋 Evaluasi Rofo - Bulan Terakhir (Under & Over Forecast)")

if monthly_performance:
    sorted_months = sorted(monthly_performance.keys())
    if sorted_months:
        last_month = sorted_months[-1]
        last_month_data = monthly_performance[last_month]
        last_month_name = last_month.strftime('%b %Y')
        
        eval_tab1, eval_tab2 = st.tabs([f"📉 UNDER Forecast ({last_month_name})", f"📈 OVER Forecast ({last_month_name})"])
        
        with eval_tab1:
            under_skus_df = last_month_data['under_skus']
            if not under_skus_df.empty:
                if 'inventory_df' in inventory_metrics:
                    inventory_data = inventory_metrics['inventory_df'][['SKU_ID', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']]
                    under_skus_df = pd.merge(under_skus_df, inventory_data, on='SKU_ID', how='left')
                
                sales_cols_last_3 = []
                if not df_sales.empty:
                    sales_months = sorted(df_sales['Month'].unique())
                    if len(sales_months) >= 3:
                        last_3_sales_months = sales_months[-3:]
                        
                        try:
                            sales_pivot = df_sales[df_sales['Month'].isin(last_3_sales_months)].pivot_table(
                                index='SKU_ID',
                                columns='Month',
                                values='Sales_Qty',
                                aggfunc='sum',
                                fill_value=0
                            ).reset_index()
                            
                            month_rename = {}
                            for col in sales_pivot.columns:
                                if isinstance(col, datetime):
                                    month_rename[col] = col.strftime('%b-%Y')
                            sales_pivot = sales_pivot.rename(columns=month_rename)
                            
                            under_skus_df = pd.merge(
                                under_skus_df,
                                sales_pivot,
                                on='SKU_ID',
                                how='left'
                            )
                            
                            sales_cols_last_3 = [col for col in sales_pivot.columns if isinstance(col, str) and '-' in col]
                            sales_cols_last_3 = sorted(sales_cols_last_3[-3:])
                            
                        except Exception as e:
                            st.warning(f"Tidak bisa menambahkan data sales 3 bulan terakhir: {str(e)}")
                
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Accuracy_Status',
                              'Forecast_Qty', 'PO_Qty', 'PO_Rofo_Ratio', 
                              'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
                
                display_cols.extend(sales_cols_last_3)
                
                available_cols = [col for col in display_cols if col in under_skus_df.columns]
                
                if 'Product_Name' not in available_cols and 'Product_Name' in under_skus_df.columns:
                    available_cols.insert(1, 'Product_Name')
                
                display_df = under_skus_df[available_cols].copy()
                
                if 'PO_Rofo_Ratio' in display_df.columns:
                    display_df['PO_Rofo_Ratio'] = display_df['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%")
                
                if 'Cover_Months' in display_df.columns:
                    display_df['Cover_Months'] = display_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if x < 999 else "N/A")
                
                if 'Avg_Monthly_Sales_3M' in display_df.columns:
                    display_df['Avg_Monthly_Sales_3M'] = display_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
                
                for col in sales_cols_last_3:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
                
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
                
                for col in sales_cols_last_3:
                    column_names[col] = col
                
                display_df = display_df.rename(columns=column_names)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500
                )
                
                total_forecast = under_skus_df['Forecast_Qty'].sum()
                total_po = under_skus_df['PO_Qty'].sum()
                avg_ratio = under_skus_df['PO_Rofo_Ratio'].mean()
                selisih_qty = total_po - total_forecast
                selisih_persen = (selisih_qty / total_forecast * 100) if total_forecast > 0 else 0
                po_rofo_pct = (total_po / total_forecast * 100) if total_forecast > 0 else 0
                
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
                
                st.html(html_content)
            else:
                st.success(f"✅ No SKUs with UNDER forecast in {last_month_name}")
        
        with eval_tab2:
            over_skus_df = last_month_data['over_skus']
            if not over_skus_df.empty:
                if 'inventory_df' in inventory_metrics:
                    inventory_data = inventory_metrics['inventory_df'][['SKU_ID', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']]
                    over_skus_df = pd.merge(over_skus_df, inventory_data, on='SKU_ID', how='left')
                
                sales_cols_last_3 = []
                if not df_sales.empty:
                    sales_months = sorted(df_sales['Month'].unique())
                    if len(sales_months) >= 3:
                        last_3_sales_months = sales_months[-3:]
                        
                        try:
                            sales_pivot = df_sales[df_sales['Month'].isin(last_3_sales_months)].pivot_table(
                                index='SKU_ID',
                                columns='Month',
                                values='Sales_Qty',
                                aggfunc='sum',
                                fill_value=0
                            ).reset_index()
                            
                            month_rename = {}
                            for col in sales_pivot.columns:
                                if isinstance(col, datetime):
                                    month_rename[col] = col.strftime('%b-%Y')
                            sales_pivot = sales_pivot.rename(columns=month_rename)
                            
                            over_skus_df = pd.merge(
                                over_skus_df,
                                sales_pivot,
                                on='SKU_ID',
                                how='left'
                            )
                            
                            sales_cols_last_3 = [col for col in sales_pivot.columns if isinstance(col, str) and '-' in col]
                            sales_cols_last_3 = sorted(sales_cols_last_3[-3:])
                            
                        except Exception as e:
                            st.warning(f"Tidak bisa menambahkan data sales 3 bulan terakhir: {str(e)}")
                
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Accuracy_Status',
                              'Forecast_Qty', 'PO_Qty', 'PO_Rofo_Ratio', 
                              'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
                
                display_cols.extend(sales_cols_last_3)
                
                available_cols = [col for col in display_cols if col in over_skus_df.columns]
                
                if 'Product_Name' not in available_cols and 'Product_Name' in over_skus_df.columns:
                    available_cols.insert(1, 'Product_Name')
                
                display_df = over_skus_df[available_cols].copy()
                
                if 'PO_Rofo_Ratio' in display_df.columns:
                    display_df['PO_Rofo_Ratio'] = display_df['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%")
                
                if 'Cover_Months' in display_df.columns:
                    display_df['Cover_Months'] = display_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if x < 999 else "N/A")
                
                if 'Avg_Monthly_Sales_3M' in display_df.columns:
                    display_df['Avg_Monthly_Sales_3M'] = display_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
                
                for col in sales_cols_last_3:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
                
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
                
                for col in sales_cols_last_3:
                    column_names[col] = col
                
                display_df = display_df.rename(columns=column_names)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500
                )
                
                total_forecast = over_skus_df['Forecast_Qty'].sum()
                total_po = over_skus_df['PO_Qty'].sum()
                avg_ratio = over_skus_df['PO_Rofo_Ratio'].mean()
                selisih_qty = total_po - total_forecast
                selisih_persen = (selisih_qty / total_forecast * 100) if total_forecast > 0 else 0
                po_rofo_pct = (total_po / total_forecast * 100) if total_forecast > 0 else 0
                
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
    "📋 Data Explorer",
    "🛒 Ecommerce Forecast",  
    "💰 Profitability Analysis",
    "🤝 Reseller Forecast",
    "🚚 Fulfillment Cost Analysis"
])

# --- TAB 1: MONTHLY PERFORMANCE DETAILS ---
with tab1:
    st.subheader("📅 Monthly Performance Details")
    
    if monthly_performance:
        summary_data = []
        for month, data in sorted(monthly_performance.items()):
            summary_data.append({
                'Month': month.strftime('%b %Y'),
                'Accuracy (%)': data['accuracy'],
                'Under': data['status_counts'].get('Under', 0),
                'Accurate': data['status_counts'].get('Accurate', 0),
                'Over': data['status_counts'].get('Over', 0),
                'Total SKUs': data['total_records'],
                'MAPE': data['mape']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        st.dataframe(
            summary_df,
            column_config={
                "Accuracy (%)": st.column_config.ProgressColumn(
                    "Accuracy %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),
                "MAPE": st.column_config.NumberColumn("MAPE %", format="%.1f%%")
            },
            use_container_width=True,
            height=400
        )
        
        if not forecast_bias.empty:
            st.divider()
            st.subheader("📉 Forecast Bias Analysis")
            
            fig_bias = go.Figure()
            fig_bias.add_trace(go.Bar(
                x=forecast_bias['Month'].dt.strftime('%b-%Y'),
                y=forecast_bias['Avg_Bias_Percentage'],
                name='Forecast Bias %',
                marker_color=forecast_bias['Avg_Bias_Percentage'].apply(
                    lambda x: '#4CAF50' if x >= -10 and x <= 10 else '#FF9800' if x >= -20 and x <= 20 else '#F44336'
                )
            ))
            
            fig_bias.update_layout(
                height=300,
                title='Monthly Forecast Bias (Positive = Over-forecast, Negative = Under-forecast)',
                xaxis_title='Month',
                yaxis_title='Bias %'
            )
            
            st.plotly_chart(fig_bias, use_container_width=True)

# --- FOOTER ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <p>🚀 <strong>Inventory Intelligence Dashboard v6.0</strong> | Professional Inventory Control & Financial Analytics</p>
    <p>✅ Product Name Auto-Lookup | ✅ Financial Analysis with Price Data | ✅ Inventory Value Analysis</p>
    <p>💰 Profitability Dashboard | 📊 Seasonality Analysis | 🎯 Margin Segmentation</p>
    <p>📈 Data since January 2024 | 🔄 Real-time Google Sheets Integration</p>
</div>
""", unsafe_allow_html=True)
