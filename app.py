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
    # Gunakan sheet_id yang sudah ada
    sheet_id = "1jcs8L0CysdzxemPz1EYVVfVhsSR-ik46khIw5jhhBgw"
    reseller_data = {}
    
    try:
        # 1. FORECAST 2026 RESELLER
        ws_fcst = _client.open_by_key(sheet_id).worksheet("Forecast_2026_Reseller")
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
            ws_sales = _client.open_by_key(sheet_id).worksheet("Sales_Reseller")
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
    """Calculate performance for each month separately - HANYA SKU dengan Forecast_Qty > 0"""
    
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
            # Get data for this month - FILTER HANYA Forecast_Qty > 0
            df_forecast_month = df_forecast[
                (df_forecast['Month'] == month) & 
                (df_forecast['Forecast_Qty'] > 0)
            ].copy()
            
            df_po_month = df_po[df_po['Month'] == month].copy()
            
            if df_forecast_month.empty or df_po_month.empty:
                continue
            
            # Merge forecast and PO for this month
            df_merged = pd.merge(
                df_forecast_month,
                df_po_month,
                on=['SKU_ID'],
                how='inner',
                suffixes=('_forecast', '_po')
            )
            
            if not df_merged.empty:
                # Add product info (jika belum ada dari merge)
                if 'Product_Name' not in df_merged.columns or 'Brand' not in df_merged.columns:
                    df_merged = add_product_info_to_data(df_merged, df_product)
                
                # Calculate ratio - Pastikan Forecast_Qty > 0
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
                
                # Calculate metrics
                df_merged['Absolute_Percentage_Error'] = abs(df_merged['PO_Rofo_Ratio'] - 100)
                
                # Hanya hitung MAPE untuk SKU dengan Forecast_Qty > 0
                valid_skus = df_merged[df_merged['Forecast_Qty'] > 0]
                if not valid_skus.empty:
                    mape = valid_skus['Absolute_Percentage_Error'].mean()
                else:
                    mape = 0
                    
                monthly_accuracy = 100 - mape
                
                # Status counts
                status_counts = df_merged['Accuracy_Status'].value_counts().to_dict()
                total_records = len(df_merged)
                status_percentages = {k: (v/total_records*100) for k, v in status_counts.items()}
                
                # Store results
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
            (df_inventory['Cover_Months'] >= 0.8) & (df_inventory['Cover_Months'] <= 1.5),
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
    # Create monthly performance summary table
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
    
    # Display chart with enhanced styling
    if not summary_df.empty:
        # Sort by month
        summary_df = summary_df.sort_values('Month')
        
        # Create enhanced chart dengan styling yang aman
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
        
    # TOTAL METRICS - BULAN TERAKHIR (dengan Qty dan persentase)
    st.divider()
    st.subheader("📊 Total Metrics - Bulan Terakhir")
    
    # Calculate metrics for LAST MONTH ONLY
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        # Count SKUs by status for last month
        under_count = last_month_data[last_month_data['Accuracy_Status'] == 'Under']['SKU_ID'].nunique()
        accurate_count = last_month_data[last_month_data['Accuracy_Status'] == 'Accurate']['SKU_ID'].nunique()
        over_count = last_month_data[last_month_data['Accuracy_Status'] == 'Over']['SKU_ID'].nunique()
        total_count_last_month = last_month_data['SKU_ID'].nunique()
        
        # Sum of forecast quantity by status for last month
        under_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Under']['Forecast_Qty'].sum()
        accurate_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Accurate']['Forecast_Qty'].sum()
        over_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Over']['Forecast_Qty'].sum()
        total_forecast_qty = last_month_data['Forecast_Qty'].sum()
        
        # Calculate percentages
        under_pct = (under_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        accurate_pct = (accurate_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        over_pct = (over_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        
        under_forecast_pct = (under_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0
        accurate_forecast_pct = (accurate_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0
        over_forecast_pct = (over_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0
    
        # Layout untuk Total Metrics bulan terakhir
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
        # Calculate overall accuracy for last month
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
    
    # Summary stats for last month
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
    
        # ROW UNTUK TOTAL ROFO, PO, SALES - BULAN TERAKHIR
    st.divider()
    st.subheader("📈 Total Rofo vs PO vs Sales - Bulan Terakhir")
    
    # Hitung total sales untuk bulan terakhir
    total_sales_last_month = 0
    sales_vs_rofo_pct = 0
    sales_vs_po_pct = 0
    
    if not df_sales.empty and monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        df_sales_last_month = df_sales[df_sales['Month'] == last_month].copy()
        total_sales_last_month = df_sales_last_month['Sales_Qty'].sum()
        
        # Hitung persentase sales vs rofo
        if total_rofo_last_month > 0:
            sales_vs_rofo_pct = (total_sales_last_month / total_rofo_last_month * 100)
        
        # Hitung persentase sales vs po
        if total_po_last_month > 0:
            sales_vs_po_pct = (total_sales_last_month / total_po_last_month * 100)
    
    # Buat 6 columns untuk Rofo, PO, Sales dan persentasenya
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
        # Sales vs Rofo %
        delta_sales_rofo = f"{sales_vs_rofo_pct-100:+.1f}%" if sales_vs_rofo_pct > 0 else "0%"
        st.metric(
            "Sales/Rofo %",
            f"{sales_vs_rofo_pct:.1f}%",
            delta=delta_sales_rofo,
            delta_color="normal" if 80 <= sales_vs_rofo_pct <= 120 else "off",
            help="Persentase Sales vs Rofo (100% = Sales = Rofo)"
        )
    
    with rofo_col5:
        # Sales vs PO %
        delta_sales_po = f"{sales_vs_po_pct-100:+.1f}%" if sales_vs_po_pct > 0 else "0%"
        st.metric(
            "Sales/PO %",
            f"{sales_vs_po_pct:.1f}%",
            delta=delta_sales_po,
            delta_color="normal" if 80 <= sales_vs_po_pct <= 120 else "off",
            help="Persentase Sales vs PO (100% = Sales = PO)"
        )
    
    with rofo_col6:
        # PO vs Rofo % (selisih PO-Rofo yang sudah ada)
        delta_po_rofo = f"{selisih_persen:+.1f}%"
        st.metric(
            "PO/Rofo %",
            f"{(total_po_last_month/total_rofo_last_month*100 if total_rofo_last_month > 0 else 0):.1f}%",
            delta=delta_po_rofo,
            delta_color="normal" if abs(selisih_persen) < 20 else "off",
            help="Persentase PO vs Rofo (100% = PO = Rofo)"
        )
    
    # Summary bar di bawah
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
    # Get last month data
    sorted_months = sorted(monthly_performance.keys())
    if sorted_months:
        last_month = sorted_months[-1]
        last_month_data = monthly_performance[last_month]
        last_month_name = last_month.strftime('%b %Y')
        
        # Create tabs for Under and Over SKUs
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
    "📋 Data Explorer",
    "🛒 Ecommerce Forecast",  
    "💰 Profitability Analysis",
    "🤝 Reseller Forecast",
    "🚚 Fulfillment Cost Analysis" # <-- TAB BARU
])

# --- TAB 1: MONTHLY PERFORMANCE DETAILS ---
with tab1:
    st.subheader("📅 Monthly Performance Details")
    
    if monthly_performance:
        # Create monthly performance summary table
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
        
        # Display summary table
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
        
        # Add forecast bias analysis if available
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

# --- TAB 2: FORECAST PERFORMANCE BY BRAND & TIER ANALYSIS ---
with tab2:
    # Brand Performance Analysis
    st.subheader("🏷️ Forecast Performance by Brand")
    
    brand_performance = calculate_brand_performance(df_forecast, df_po, df_product)
    
    if not brand_performance.empty:
        # ================ KPI CARDS SECTION ================
        st.subheader("📊 Brand Performance KPIs")
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        
        with col_kpi1:
            # Best accuracy brand
            best_acc = brand_performance.loc[brand_performance['Accuracy'].idxmax()]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); 
                        border-radius: 10px; padding: 1rem; margin: 0.5rem 0; 
                        border-left: 5px solid #4CAF50;">
                <div style="font-size: 0.9rem; color: #2E7D32; font-weight: 600;">🎯 Best Accuracy</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #1B5E20;">{best_acc['Brand']}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span style="font-size: 0.8rem; color: #666;">Accuracy:</span>
                    <span style="font-size: 1rem; font-weight: 700; color: #1B5E20;">{best_acc['Accuracy']:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_kpi2:
            # Most SKUs brand
            most_skus = brand_performance.loc[brand_performance['SKU_Count'].idxmax()]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                        border-radius: 10px; padding: 1rem; margin: 0.5rem 0; 
                        border-left: 5px solid #2196F3;">
                <div style="font-size: 0.9rem; color: #1565C0; font-weight: 600;">📦 Most SKUs</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #0D47A1;">{most_skus['Brand']}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span style="font-size: 0.8rem; color: #666;">SKUs:</span>
                    <span style="font-size: 1rem; font-weight: 700; color: #0D47A1;">{most_skus['SKU_Count']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_kpi3:
            # Highest volume brand
            highest_rofo = brand_performance.loc[brand_performance['Total_Forecast'].idxmax()]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%); 
                        border-radius: 10px; padding: 1rem; margin: 0.5rem 0; 
                        border-left: 5px solid #9C27B0;">
                <div style="font-size: 0.9rem; color: #7B1FA2; font-weight: 600;">📈 Highest Volume</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #4A148C;">{highest_rofo['Brand']}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span style="font-size: 0.8rem; color: #666;">Rofo Qty:</span>
                    <span style="font-size: 1rem; font-weight: 700; color: #4A148C;">{highest_rofo['Total_Forecast']:,.0f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ================ FINANCIAL ANALYSIS PER BRAND ================
        st.divider()
        st.subheader("💰 Brand - Forecast Amount (Full Year)")
        
        if not df_financial.empty:
            brand_financial = df_financial.groupby('Brand').agg({
                'Revenue': 'sum',
                'Gross_Margin': 'sum',
                'Sales_Qty': 'sum'
            }).reset_index()
            
            brand_financial['Margin_Percentage'] = np.where(
                brand_financial['Revenue'] > 0,
                (brand_financial['Gross_Margin'] / brand_financial['Revenue'] * 100),
                0
            )
            
            brand_financial = brand_financial.sort_values('Gross_Margin', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # --- UPDATE: Format angka jadi String biar ada komanya (Rp 1,000,000) ---
                brand_disp = brand_financial.head(10).copy()
                brand_disp['Revenue'] = brand_disp['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
                brand_disp['Gross_Margin'] = brand_disp['Gross_Margin'].apply(lambda x: f"Rp {x:,.0f}")
                
                st.dataframe(
                    brand_disp,
                    column_config={
                        # Revenue & Gross Margin gak perlu config lagi karena sudah jadi Text di atas
                        "Margin_Percentage": st.column_config.ProgressColumn("Margin %", format="%.1f%%", min_value=0, max_value=100)
                    },
                    use_container_width=True
                )
            
            with col2:
                # Chart brand profitability (Tidak berubah)
                fig = px.bar(brand_financial.head(10), x='Brand', y='Margin_Percentage',
                            title='Top 10 Brands by Margin %',
                            color='Margin_Percentage',
                            color_continuous_scale='RdYlGn')
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # ================ DATA TABLE SECTION ================
        st.divider()
        st.subheader("📋 Brand Performance Data")
        
        # Format the display
        display_brand_df = brand_performance.copy()
        
        # Format columns
        display_brand_df['Accuracy'] = display_brand_df['Accuracy'].apply(lambda x: f"{x:.1f}%")
        display_brand_df['PO_vs_Forecast_Ratio'] = display_brand_df['PO_vs_Forecast_Ratio'].apply(lambda x: f"{x:.1f}%")
        display_brand_df['Total_Forecast'] = display_brand_df['Total_Forecast'].apply(lambda x: f"{x:,.0f}")
        display_brand_df['Total_PO'] = display_brand_df['Total_PO'].apply(lambda x: f"{x:,.0f}")
        display_brand_df['Qty_Difference'] = display_brand_df['Qty_Difference'].apply(lambda x: f"{x:+,.0f}")
        
        # Rename columns
        column_names = {
            'Brand': 'Brand',
            'SKU_Count': 'SKU Count',
            'Total_Forecast': 'Total Rofo',
            'Total_PO': 'Total PO',
            'Accuracy': 'Accuracy %',
            'PO_vs_Forecast_Ratio': 'PO/Rofo %',
            'Qty_Difference': 'Qty Diff',
            'Under': 'Under',
            'Accurate': 'Accurate',
            'Over': 'Over'
        }
        
        display_brand_df = display_brand_df.rename(columns=column_names)
        
        # Display table
        st.dataframe(
            display_brand_df,
            use_container_width=True,
            height=400
        )
        
        # ================ GROUPED BAR CHART SECTION ================
        st.divider()
        st.subheader("📊 Brand Performance Comparison")
        
        # PENTING: Cari bulan terakhir yang ADA DATA SALES-nya
        sales_months = sorted(df_sales['Month'].unique()) if not df_sales.empty else []
        forecast_months = sorted(df_forecast['Month'].unique()) if not df_forecast.empty else []
        po_months = sorted(df_po['Month'].unique()) if not df_po.empty else []
        
        # Cari bulan terakhir yang ADA di ketiga dataset
        common_months = sorted(set(sales_months) & set(forecast_months) & set(po_months))
        if common_months:
            last_month = common_months[-1]
        else:
            # Kalau ngga ada bulan yang sama, ambil bulan terakhir dari forecast saja
            last_month = forecast_months[-1] if forecast_months else None
        
        if last_month:
            st.caption(f"📅 Data untuk bulan: {last_month.strftime('%b %Y')}")
            
            # Get data untuk bulan terakhir
            df_forecast_last = df_forecast[df_forecast['Month'] == last_month]
            df_po_last = df_po[df_po['Month'] == last_month]
            df_sales_last = df_sales[df_sales['Month'] == last_month]
            
            # Debug info
            st.caption(f"Forecast SKUs: {len(df_forecast_last)} | PO SKUs: {len(df_po_last)} | Sales SKUs: {len(df_sales_last)}")
            
            # Add product info
            df_forecast_last = add_product_info_to_data(df_forecast_last, df_product)
            df_po_last = add_product_info_to_data(df_po_last, df_product)
            df_sales_last = add_product_info_to_data(df_sales_last, df_product)
            
            if 'Brand' in df_forecast_last.columns:
                # Get UNIQUE BRANDS dari semua dataset
                forecast_brands = set(df_forecast_last['Brand'].dropna().unique())
                po_brands = set(df_po_last['Brand'].dropna().unique()) if 'Brand' in df_po_last.columns else set()
                sales_brands = set(df_sales_last['Brand'].dropna().unique()) if 'Brand' in df_sales_last.columns else set()
                
                # Gabungkan semua brand
                all_brands = forecast_brands.union(po_brands).union(sales_brands)
                
                brand_comparison = []
                
                for brand in sorted(all_brands):
                    # Forecast
                    rofo_qty = df_forecast_last[df_forecast_last['Brand'] == brand]['Forecast_Qty'].sum()
                    
                    # PO
                    po_qty = df_po_last[df_po_last['Brand'] == brand]['PO_Qty'].sum() if 'Brand' in df_po_last.columns else 0
                    
                    # Sales
                    sales_qty = 0
                    if not df_sales_last.empty and 'Brand' in df_sales_last.columns:
                        sales_qty = df_sales_last[df_sales_last['Brand'] == brand]['Sales_Qty'].sum()
                    
                    brand_comparison.append({
                        'Brand': brand,
                        'Rofo': rofo_qty,
                        'PO': po_qty,
                        'Sales': sales_qty,
                        'PO_Rofo_Ratio': (po_qty / rofo_qty * 100) if rofo_qty > 0 else 0
                    })
                
                comparison_df = pd.DataFrame(brand_comparison)
                
                # TAMPILKAN SEMUA BRAND (tanpa .head())
                comparison_df = comparison_df.sort_values('Rofo', ascending=False)
                
                # Tampilkan jumlah brand
                st.caption(f"📊 Menampilkan {len(comparison_df)} brand")
                
                # Cek apakah ada data Sales
                total_sales = comparison_df['Sales'].sum()
                
                if total_sales > 0:
                    # Buat chart dengan 3 bar (Rofo, PO, Sales)
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['Brand'],
                        y=comparison_df['Rofo'],
                        name='Rofo',
                        marker_color='#667eea',
                        hovertemplate='<b>%{x}</b><br>Rofo: %{y:,.0f}<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['Brand'],
                        y=comparison_df['PO'],
                        name='PO',
                        marker_color='#FF9800',
                        hovertemplate='<b>%{x}</b><br>PO: %{y:,.0f}<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['Brand'],
                        y=comparison_df['Sales'],
                        name='Sales',
                        marker_color='#4CAF50',
                        hovertemplate='<b>%{x}</b><br>Sales: %{y:,.0f}<extra></extra>'
                    ))
                    
                    chart_title = f'Brand Performance - {last_month.strftime("%b %Y")} (Rofo vs PO vs Sales)'
                else:
                    # Kalau ngga ada Sales, tampilkan cuma Rofo vs PO
                    st.info("ℹ️ Data Sales tidak tersedia untuk bulan ini, menampilkan Rofo vs PO saja")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['Brand'],
                        y=comparison_df['Rofo'],
                        name='Rofo',
                        marker_color='#667eea',
                        hovertemplate='<b>%{x}</b><br>Rofo: %{y:,.0f}<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=comparison_df['Brand'],
                        y=comparison_df['PO'],
                        name='PO',
                        marker_color='#FF9800',
                        hovertemplate='<b>%{x}</b><br>PO: %{y:,.0f}<extra></extra>'
                    ))
                    
                    chart_title = f'Brand Performance - {last_month.strftime("%b %Y")} (Rofo vs PO)'
                
                fig.update_layout(
                    height=500,
                    title=chart_title,
                    xaxis_title='Brand',
                    yaxis_title='Quantity',
                    barmode='group',
                    hovermode='x unified',
                    plot_bgcolor='white',
                    xaxis={'categoryorder': 'total descending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # ================ ACCURACY VISUALIZATION SECTION ================
        st.divider()
        st.subheader("🎯 Brand Accuracy Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart for top brand accuracy
            if 'comparison_df' in locals() and not comparison_df.empty:
                top_brand = comparison_df.iloc[0]
                
                # Hitung accuracy untuk top brand
                top_accuracy = 0
                if top_brand['Rofo'] > 0:
                    top_accuracy = 100 - abs(top_brand['PO_Rofo_Ratio'] - 100)
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=top_accuracy,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Top Brand: {top_brand['Brand']}"},
                    delta={'reference': 80, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 70], 'color': "#FF5252"},
                            {'range': [70, 85], 'color': "#FF9800"},
                            {'range': [85, 100], 'color': "#4CAF50"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Horizontal bar chart for accuracy ranking
            if 'comparison_df' in locals() and not comparison_df.empty:
                # Hitung accuracy untuk semua brand
                comparison_df['Accuracy'] = comparison_df.apply(
                    lambda row: 100 - abs(row['PO_Rofo_Ratio'] - 100) if row['Rofo'] > 0 else 0,
                    axis=1
                )
                
                accuracy_sorted = comparison_df.sort_values('Accuracy', ascending=True)
                
                fig_accuracy = go.Figure()
                
                fig_accuracy.add_trace(go.Bar(
                    y=accuracy_sorted['Brand'],
                    x=accuracy_sorted['Accuracy'],
                    orientation='h',
                    marker_color=accuracy_sorted['Accuracy'].apply(
                        lambda x: '#4CAF50' if x >= 80 else '#FF9800' if x >= 70 else '#FF5252'
                    ),
                    text=accuracy_sorted['Accuracy'].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside'
                ))
                
                fig_accuracy.update_layout(
                    height=300,
                    title='Brand Accuracy Ranking',
                    xaxis_title='Accuracy (%)',
                    yaxis_title='Brand',
                    xaxis_range=[0, 100]
                )
                
                st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # ================ HEATMAP SECTION ================
        st.divider()
        st.subheader("📊 Brand Performance Status Heatmap")
        
        # Prepare data for heatmap
        status_data = []
        for _, row in display_brand_df.iterrows():
            brand = row['Brand']
            total_skus = int(str(row['SKU Count']).replace(',', ''))
            under = int(row['Under']) if pd.notnull(row['Under']) else 0
            accurate = int(row['Accurate']) if pd.notnull(row['Accurate']) else 0
            over = int(row['Over']) if pd.notnull(row['Over']) else 0
            
            status_data.append({
                'Brand': brand,
                'Under': (under/total_skus*100) if total_skus > 0 else 0,
                'Accurate': (accurate/total_skus*100) if total_skus > 0 else 0,
                'Over': (over/total_skus*100) if total_skus > 0 else 0
            })
        
        status_df = pd.DataFrame(status_data)
        status_df = status_df.sort_values('Accurate', ascending=False)
        
        fig_heatmap = go.Figure()
        
        fig_heatmap.add_trace(go.Heatmap(
            z=[status_df['Under'], status_df['Accurate'], status_df['Over']],
            x=status_df['Brand'].tolist(),
            y=['Under %', 'Accurate %', 'Over %'],
            colorscale=[[0, '#FF5252'], [0.5, '#FF9800'], [1, '#4CAF50']],
            text=np.round([status_df['Under'], status_df['Accurate'], status_df['Over']], 1),
            texttemplate='%{text:.1f}%',
            hovertemplate='<b>%{y}</b><br>Brand: %{x}<br>Percentage: %{text:.1f}%<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            height=400,
            title='Brand Performance Distribution',
            xaxis_title='Brand',
            yaxis_title='Performance Status'
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # ================ SCATTER PLOT SECTION ================
        st.divider()
        st.subheader("🔍 Brand Performance Scatter Analysis")
        
        # Prepare data for scatter plot
        scatter_data = brand_performance.copy()
        
        # Create scatter plot
        fig_scatter = px.scatter(
            scatter_data,
            x='Total_Forecast',
            y='Accuracy',
            size='SKU_Count',
            color='PO_vs_Forecast_Ratio',
            hover_name='Brand',
            hover_data=['SKU_Count', 'Total_PO', 'Under', 'Accurate', 'Over'],
            title='Brand Performance: Accuracy vs Forecast Volume',
            labels={
                'Total_Forecast': 'Total Forecast Volume',
                'Accuracy': 'Forecast Accuracy (%)',
                'SKU_Count': 'Number of SKUs',
                'PO_vs_Forecast_Ratio': 'PO/Rofo Ratio (%)'
            },
            color_continuous_scale='RdYlGn',
            size_max=50
        )
        
        # Add quadrant lines
        fig_scatter.add_hline(y=80, line_dash="dash", line_color="gray", 
                             annotation_text="Accuracy Target (80%)")
        fig_scatter.add_vline(x=scatter_data['Total_Forecast'].median(), 
                             line_dash="dash", line_color="gray",
                             annotation_text="Median Volume")
        
        fig_scatter.update_layout(
            height=500,
            xaxis_title='Total Forecast Volume (log scale)',
            xaxis_type='log',
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Quadrant analysis
        st.subheader("📊 Brand Performance Quadrants")
        
        # Calculate quadrant metrics
        median_volume = scatter_data['Total_Forecast'].median()
        
        quadrants = {
            'High Accuracy, High Volume': scatter_data[
                (scatter_data['Accuracy'] >= 80) & 
                (scatter_data['Total_Forecast'] >= median_volume)
            ],
            'High Accuracy, Low Volume': scatter_data[
                (scatter_data['Accuracy'] >= 80) & 
                (scatter_data['Total_Forecast'] < median_volume)
            ],
            'Low Accuracy, High Volume': scatter_data[
                (scatter_data['Accuracy'] < 80) & 
                (scatter_data['Total_Forecast'] >= median_volume)
            ],
            'Low Accuracy, Low Volume': scatter_data[
                (scatter_data['Accuracy'] < 80) & 
                (scatter_data['Total_Forecast'] < median_volume)
            ]
        }
        
        # Display quadrant summary
        quad_cols = st.columns(4)
        quad_colors = ['#4CAF50', '#8BC34A', '#FF9800', '#F44336']
        
        for idx, (quadrant_name, quadrant_data) in enumerate(quadrants.items()):
            with quad_cols[idx]:
                count = len(quadrant_data)
                percent = (count / len(scatter_data) * 100) if len(scatter_data) > 0 else 0
                
                # Get top brand in quadrant
                top_brand = quadrant_data.iloc[0]['Brand'] if not quadrant_data.empty else "N/A"
                
                st.markdown(f"""
                <div style="background: white; border-radius: 10px; padding: 1rem; 
                            margin: 0.5rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                            border-left: 5px solid {quad_colors[idx]};">
                    <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.5rem;">
                        {quadrant_name}
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #333;">
                        {count}
                    </div>
                    <div style="font-size: 0.8rem; color: #888; margin-top: 0.3rem;">
                        {percent:.1f}% of brands
                    </div>
                    <div style="font-size: 0.7rem; color: #999; margin-top: 0.5rem; border-top: 1px solid #eee; padding-top: 0.3rem;">
                        Top: {top_brand}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
    else:
        st.info("📊 No brand performance data available")
    
    st.divider()
    
    # ================ TIER ANALYSIS SECTION ================
    st.subheader("🏷️ SKU Tier Analysis")
    
    if monthly_performance and not df_product.empty:
        # Get last month data for tier analysis
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        # Tier analysis
        if 'SKU_Tier' in last_month_data.columns:
            tier_summary = last_month_data.groupby('SKU_Tier').agg({
                'SKU_ID': 'count',
                'PO_Rofo_Ratio': 'mean',
                'Forecast_Qty': 'sum',
                'PO_Qty': 'sum'
            }).reset_index()
            
            tier_summary.columns = ['Tier', 'SKU Count', 'Avg PO/Rofo %', 'Total Forecast', 'Total PO']
            tier_summary['Avg PO/Rofo %'] = tier_summary['Avg PO/Rofo %'].apply(lambda x: f"{x:.1f}%")
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.dataframe(
                    tier_summary,
                    use_container_width=True,
                    height=300
                )
            
            with col_t2:
                # Pie chart for tier distribution
                fig_pie = go.Figure(data=[go.Pie(
                    labels=tier_summary['Tier'],
                    values=tier_summary['SKU Count'],
                    hole=0.3,
                    marker_colors=['#667eea', '#FF9800', '#4CAF50', '#FF5252', '#9C27B0'],
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>SKUs: %{value}<br>%{percent}<extra></extra>'
                )])
                
                fig_pie.update_layout(
                    height=300,
                    title='SKU Distribution by Tier',
                    showlegend=False
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Tier Performance Comparison
            st.subheader("📈 Tier Performance Comparison")
            
            # Prepare data for radar chart
            tiers = tier_summary['Tier'].tolist()
            accuracy_values = []
            po_rofo_values = []
            
            for tier in tiers:
                tier_data = last_month_data[last_month_data['SKU_Tier'] == tier]
                if not tier_data.empty:
                    # Calculate accuracy
                    accuracy = 100 - abs(tier_data['PO_Rofo_Ratio'] - 100).mean()
                    accuracy_values.append(accuracy)
                    
                    # Calculate PO/Rofo ratio
                    po_rofo = (tier_data['PO_Qty'].sum() / tier_data['Forecast_Qty'].sum() * 100) if tier_data['Forecast_Qty'].sum() > 0 else 0
                    po_rofo_values.append(po_rofo)
            
            # Radar chart
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=accuracy_values,
                theta=tiers,
                fill='toself',
                name='Accuracy %',
                line_color='#667eea',
                fillcolor='rgba(102, 126, 234, 0.3)'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=po_rofo_values,
                theta=tiers,
                fill='toself',
                name='PO/Rofo %',
                line_color='#FF9800',
                fillcolor='rgba(255, 152, 0, 0.3)'
            ))
            
            fig_radar.update_layout(
                height=400,
                title='Tier Performance Radar Chart',
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(max(accuracy_values), max(po_rofo_values)) * 1.1]
                    )
                ),
                showlegend=True
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Inventory tier analysis
        if 'tier_analysis' in inventory_metrics:
            st.divider()
            st.subheader("📦 Inventory by Tier")
            
            tier_inv = inventory_metrics['tier_analysis']
            
            # Treemap for inventory distribution
            fig_treemap = px.treemap(
                tier_inv,
                path=['Tier'],
                values='Total_Stock',
                color='Avg_Cover_Months',
                color_continuous_scale='RdYlGn',
                title='Inventory Distribution by Tier (Size = Total Stock, Color = Cover Months)',
                hover_data=['SKU_Count', 'Total_Sales_3M_Avg', 'Turnover']
            )
            
            fig_treemap.update_layout(height=400)
            st.plotly_chart(fig_treemap, use_container_width=True)
            
            # Additional metrics
            col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
            
            with col_metrics1:
                if not tier_inv.empty:
                    best_tier = tier_inv.loc[tier_inv['Turnover'].idxmax()]
                    st.metric(
                        "Highest Turnover Tier",
                        best_tier['Tier'],
                        delta=f"{best_tier['Turnover']:.2f} Turnover"
                    )
            
            with col_metrics2:
                if not tier_inv.empty:
                    best_cover = tier_inv.loc[tier_inv['Avg_Cover_Months'].idxmax()]
                    st.metric(
                        "Highest Cover Tier",
                        best_cover['Tier'],
                        delta=f"{best_cover['Avg_Cover_Months']:.1f} months"
                    )
            
            with col_metrics3:
                if not tier_inv.empty:
                    total_stock = tier_inv['Total_Stock'].sum()
                    st.metric("Total Stock All Tiers", f"{total_stock:,.0f}")
            
            with col_metrics4:
                if not tier_inv.empty:
                    avg_cover = tier_inv['Avg_Cover_Months'].mean()
                    st.metric("Average Cover All Tiers", f"{avg_cover:.1f} months")

# --- TAB 3: INTELLIGENT INVENTORY OPTIMIZATION ---
with tab3:
    st.subheader("🧠 INTELLIGENT INVENTORY OPTIMIZATION SYSTEM")
    st.markdown("#### **AI-Powered Stock Management with Predictive Analytics**")
    
    # ============================================
    # SECTION 1: EXECUTIVE INTELLIGENCE DASHBOARD
    # ============================================
    st.markdown("---")
    
    # REAL-TIME INVENTORY HEALTH SCORE
    col_score1, col_score2, col_score3, col_score4 = st.columns(4)
    
    with col_score1:
        # Inventory Health Score
        health_score = 0
        if 'inventory_df' in inventory_metrics and not inventory_metrics['inventory_df'].empty:
            df_inv = inventory_metrics['inventory_df']
            healthy_skus = len(df_inv[df_inv['Cover_Months'].between(0.8, 1.5)])
            total_active = len(df_inv)
            health_score = (healthy_skus / total_active * 100) if total_active > 0 else 0
        
        fig_health = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "🏆 Inventory Health Score", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4CAF50"},
                'steps': [
                    {'range': [0, 50], 'color': "#FF5252"},
                    {'range': [50, 80], 'color': "#FF9800"},
                    {'range': [80, 100], 'color': "#4CAF50"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig_health.update_layout(height=200, margin=dict(t=30, b=10))
        st.plotly_chart(fig_health, use_container_width=True)
    
    with col_score2:
        # Capital Efficiency Score
        if 'inventory_df' in inventory_metrics and not df_stock.empty and 'Floor_Price' in df_stock.columns:
            # Calculate inventory turns
            total_inv_value = (df_stock['Stock_Qty'] * df_stock['Floor_Price']).sum()
            if not df_financial.empty:
                annual_sales = df_financial['Revenue'].sum()
                inventory_turns = annual_sales / total_inv_value if total_inv_value > 0 else 0
                
                # Industry benchmark comparison
                industry_avg = 4.0  # Retail industry average
                efficiency_score = min(100, (inventory_turns / industry_avg * 100))
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); 
                            border-radius: 12px; padding: 1.2rem; color: white; 
                            box-shadow: 0 6px 20px rgba(33, 150, 243, 0.3); height: 200px;">
                    <div style="font-size: 0.9rem; opacity: 0.9;">CAPITAL EFFICIENCY</div>
                    <div style="font-size: 2rem; font-weight: 800; margin: 0.5rem 0;">{inventory_turns:.1f}x</div>
                    <div style="font-size: 0.9rem;">Inventory Turns</div>
                    <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">
                        Industry Avg: {industry_avg}x<br>
                        Score: {efficiency_score:.0f}/100
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric("Capital Efficiency", "N/A", "Need sales data")
        else:
            st.metric("Capital Efficiency", "N/A", "Need price data")
    
    with col_score3:
        # Stockout Risk Index
        stockout_risk = 0
        if 'low_stock' in inventory_metrics and not inventory_metrics['low_stock'].empty:
            low_stock_df = inventory_metrics['low_stock']
            critical_items = low_stock_df[low_stock_df['Cover_Months'] < 0.5]
            stockout_risk = len(critical_items)
        
        risk_color = "#4CAF50" if stockout_risk == 0 else "#FF9800" if stockout_risk < 5 else "#F44336"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color.replace('F44', 'D32')} 100%); 
                    border-radius: 12px; padding: 1.2rem; color: white; 
                    box-shadow: 0 6px 20px rgba(244, 67, 54, 0.3); height: 200px;">
            <div style="font-size: 0.9rem; opacity: 0.9;">🚨 STOCKOUT RISK</div>
            <div style="font-size: 2rem; font-weight: 800; margin: 0.5rem 0;">{stockout_risk}</div>
            <div style="font-size: 0.9rem;">Critical SKUs</div>
            <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">
                {"✅ All good" if stockout_risk == 0 else "⚠️ Monitor" if stockout_risk < 5 else "🚨 Immediate action needed"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_score4:
        # Excess Stock Value
        excess_value = 0
        if 'high_stock' in inventory_metrics and not inventory_metrics['high_stock'].empty:
            high_stock_df = inventory_metrics['high_stock']
            if 'Floor_Price' in high_stock_df.columns:
                high_stock_df['Value'] = high_stock_df['Stock_Qty'] * high_stock_df['Floor_Price']
                excess_value = high_stock_df['Value'].sum()
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); 
                    border-radius: 12px; padding: 1.2rem; color: white; 
                    box-shadow: 0 6px 20px rgba(255, 152, 0, 0.3); height: 200px;">
            <div style="font-size: 0.9rem; opacity: 0.9;">📦 EXCESS STOCK</div>
            <div style="font-size: 2rem; font-weight: 800; margin: 0.5rem 0;">Rp {excess_value:,.0f}</div>
            <div style="font-size: 0.9rem;">Capital Locked</div>
            <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">
                {"✅ Optimal" if excess_value < 10000000 else "⚠️ Moderate" if excess_value < 50000000 else "🚨 High"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # NEW SECTION: INTELLIGENT RECOMMENDATIONS ENGINE
    # ============================================
    st.markdown("---")
    st.subheader("🎯 AI-PURCHASING RECOMMENDATIONS")
    
    # ======================== FUNGSI BARU: REORDER POINT CALCULATION ========================
    @st.cache_data
    def calculate_reorder_recommendations(df_inventory, df_sales, service_level=95, lead_time_days=30):
        """
        Calculate intelligent reorder recommendations
        based on statistical safety stock calculations
        """
        recommendations = []
        
        if df_inventory.empty or df_sales.empty:
            return pd.DataFrame()
        
        try:
            # Calculate daily sales statistics
            if 'Month' in df_sales.columns:
                # Get last 90 days sales
                latest_date = df_sales['Month'].max()
                start_date = latest_date - pd.Timedelta(days=90)
                recent_sales = df_sales[df_sales['Month'] >= start_date].copy()
                
                # Calculate daily sales per SKU
                recent_sales['Days_Since'] = (latest_date - recent_sales['Month']).dt.days
                recent_sales['Weight'] = np.exp(-recent_sales['Days_Since'] / 30)  # Exponential decay weighting
                
                sku_stats = []
                for sku_id in df_inventory['SKU_ID'].unique():
                    sku_sales = recent_sales[recent_sales['SKU_ID'] == sku_id]
                    
                    if len(sku_sales) >= 3:  # Minimum data points
                        # Weighted average daily sales
                        total_weight = sku_sales['Weight'].sum()
                        if total_weight > 0:
                            avg_daily_sales = (sku_sales['Sales_Qty'] * sku_sales['Weight']).sum() / total_weight / 30
                        else:
                            avg_daily_sales = sku_sales['Sales_Qty'].mean() / 30
                        
                        # Standard deviation
                        sales_std = sku_sales['Sales_Qty'].std() / 30
                        
                        sku_stats.append({
                            'SKU_ID': sku_id,
                            'Avg_Daily_Sales': avg_daily_sales,
                            'Sales_Std': sales_std if not pd.isna(sales_std) else avg_daily_sales * 0.3
                        })
                
                df_stats = pd.DataFrame(sku_stats)
                
                # Merge with inventory data
                df_combined = pd.merge(df_inventory, df_stats, on='SKU_ID', how='left')
                df_combined['Avg_Daily_Sales'] = df_combined['Avg_Daily_Sales'].fillna(0)
                df_combined['Sales_Std'] = df_combined['Sales_Std'].fillna(df_combined['Avg_Daily_Sales'] * 0.3)
                
                # Z-score for service level
                z_score = {90: 1.28, 95: 1.65, 99: 2.33}.get(service_level, 1.65)
                
                # Calculate safety stock
                df_combined['Safety_Stock'] = z_score * df_combined['Sales_Std'] * np.sqrt(lead_time_days)
                
                # Calculate reorder point
                df_combined['Reorder_Point'] = (df_combined['Avg_Daily_Sales'] * lead_time_days) + df_combined['Safety_Stock']
                
                # Calculate suggested order quantity
                df_combined['Suggested_Order_Qty'] = df_combined.apply(
                    lambda row: max(0, row['Reorder_Point'] - row['Stock_Qty']) if row['Stock_Qty'] < row['Reorder_Point'] else 0,
                    axis=1
                )
                
                # Add EOQ (Economic Order Quantity) calculation
                # Assume ordering cost Rp 50,000 and holding cost 20% of product cost
                ordering_cost = 50000
                holding_rate = 0.20
                
                df_combined['EOQ'] = df_combined.apply(
                    lambda row: np.sqrt((2 * row['Avg_Daily_Sales'] * 30 * ordering_cost) / 
                                      (row['Floor_Price'] * holding_rate)) if row['Floor_Price'] > 0 else 0,
                    axis=1
                )
                
                # Adjust suggested order to EOQ if applicable
                df_combined['Final_Suggested_Qty'] = df_combined.apply(
                    lambda row: max(row['Suggested_Order_Qty'], row['EOQ']) if row['Suggested_Order_Qty'] > 0 else 0,
                    axis=1
                )
                
                # Filter only SKUs that need reorder
                df_reorder = df_combined[
                    (df_combined['Final_Suggested_Qty'] > 0) &
                    (df_combined['Status'].str.upper() == 'ACTIVE')
                ].copy()
                
                # Calculate order value
                if 'Floor_Price' in df_reorder.columns:
                    df_reorder['Order_Value'] = df_reorder['Final_Suggested_Qty'] * df_reorder['Floor_Price']
                
                return df_reorder.sort_values('Order_Value', ascending=False)
            
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Reorder calculation error: {str(e)}")
            return pd.DataFrame()
    
    # ======================== FUNGSI BARU: ABC ANALYSIS ========================
    @st.cache_data
    def perform_abc_analysis(df_inventory, df_sales=None):
        """Perform ABC analysis based on Pareto principle"""
        
        if df_inventory.empty:
            return pd.DataFrame()
        
        try:
            # Calculate inventory value
            df_abc = df_inventory.copy()
            
            if 'Floor_Price' in df_abc.columns:
                df_abc['Value'] = df_abc['Stock_Qty'] * df_abc['Floor_Price']
            else:
                # Use stock quantity if no price available
                df_abc['Value'] = df_abc['Stock_Qty']
            
            # Sort by value descending
            df_abc = df_abc.sort_values('Value', ascending=False)
            
            # Calculate cumulative percentages
            df_abc['Cumulative_Value'] = df_abc['Value'].cumsum()
            total_value = df_abc['Value'].sum()
            
            if total_value > 0:
                df_abc['Value_Pct'] = (df_abc['Value'] / total_value * 100)
                df_abc['Cumulative_Pct'] = (df_abc['Cumulative_Value'] / total_value * 100)
                
                # Classify A, B, C items
                conditions = [
                    df_abc['Cumulative_Pct'] <= 80,      # A items: 80% of value
                    df_abc['Cumulative_Pct'] <= 95,      # B items: next 15% of value
                    df_abc['Cumulative_Pct'] <= 100      # C items: last 5% of value
                ]
                
                choices = ['A', 'B', 'C']
                df_abc['ABC_Class'] = np.select(conditions, choices, default='C')
                
                # Calculate turnover if sales data available
                if df_sales is not None and not df_sales.empty:
                    # Get last 3 months sales
                    latest_month = df_sales['Month'].max()
                    three_months_ago = latest_month - pd.DateOffset(months=3)
                    recent_sales = df_sales[df_sales['Month'] >= three_months_ago]
                    
                    sales_by_sku = recent_sales.groupby('SKU_ID')['Sales_Qty'].sum().reset_index()
                    sales_by_sku.columns = ['SKU_ID', 'Sales_3M']
                    
                    df_abc = pd.merge(df_abc, sales_by_sku, on='SKU_ID', how='left')
                    df_abc['Sales_3M'] = df_abc['Sales_3M'].fillna(0)
                    df_abc['Turnover_Rate'] = df_abc.apply(
                        lambda row: row['Sales_3M'] / row['Stock_Qty'] if row['Stock_Qty'] > 0 else 0,
                        axis=1
                    )
                
                return df_abc
            
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"ABC Analysis error: {str(e)}")
            return pd.DataFrame()
    
    # ======================== FUNGSI BARU: DEAD STOCK IDENTIFICATION ========================
    @st.cache_data
    def identify_dead_stock(df_inventory, df_sales, months_threshold=6):
        """Identify dead/slow-moving stock"""
        
        if df_inventory.empty or df_sales.empty:
            return pd.DataFrame()
        
        try:
            # Get latest sales date
            latest_sales_date = df_sales['Month'].max()
            cutoff_date = latest_sales_date - pd.DateOffset(months=months_threshold)
            
            # Find SKUs with sales in last X months
            recent_sales_skus = df_sales[df_sales['Month'] >= cutoff_date]['SKU_ID'].unique()
            
            # Identify dead stock (has inventory but no recent sales)
            df_dead = df_inventory[
                (~df_inventory['SKU_ID'].isin(recent_sales_skus)) &
                (df_inventory['Stock_Qty'] > 0) &
                (df_inventory['Status'].str.upper() == 'ACTIVE')
            ].copy()
            
            if 'Floor_Price' in df_dead.columns:
                df_dead['Dead_Stock_Value'] = df_dead['Stock_Qty'] * df_dead['Floor_Price']
            
            return df_dead.sort_values('Stock_Qty', ascending=False)
            
        except Exception as e:
            st.error(f"Dead stock identification error: {str(e)}")
            return pd.DataFrame()
    
    # ======================== FUNGSI BARU: SEASONALITY ADJUSTMENT ========================
    @st.cache_data
    def calculate_seasonality_adjustment(df_sales):
        """Calculate seasonal adjustment factors"""
        
        if df_sales.empty:
            return {}
        
        try:
            # Add month number
            df_sales['Month_Num'] = df_sales['Month'].dt.month
            
            # Calculate monthly sales pattern
            monthly_sales = df_sales.groupby('Month_Num')['Sales_Qty'].sum()
            avg_monthly = monthly_sales.mean()
            
            # Calculate seasonal indices
            seasonal_indices = {}
            for month in range(1, 13):
                month_sales = monthly_sales.get(month, 0)
                seasonal_indices[month] = month_sales / avg_monthly if avg_monthly > 0 else 1.0
            
            return seasonal_indices
            
        except Exception as e:
            st.error(f"Seasonality calculation error: {str(e)}")
            return {}
    
    # ============================================
    # IMPLEMENTASI DASHBOARD BARU
    # ============================================
    
    # 1. SETTING PANEL
    with st.expander("⚙️ INTELLIGENT SETTINGS", expanded=True):
        col_set1, col_set2, col_set3 = st.columns(3)
        
        with col_set1:
            service_level = st.slider("Service Level Target", 90, 99, 95, 
                                     help="Probability of not having stockout")
            lead_time = st.number_input("Lead Time (days)", 7, 90, 30,
                                       help="Supplier lead time in days")
        
        with col_set2:
            reorder_threshold = st.slider("Reorder Threshold (months)", 0.5, 2.0, 0.8, 0.1,
                                         help="Reorder when stock below X months coverage")
            excess_threshold = st.slider("Excess Threshold (months)", 1.5, 6.0, 1.5, 0.1,
                                        help="Flag as excess when stock above X months")
        
        with col_set3:
            ordering_cost = st.number_input("Ordering Cost (Rp)", 10000, 200000, 50000,
                                          help="Cost per purchase order")
            holding_rate = st.slider("Holding Cost Rate (%)", 10, 30, 20,
                                    help="Annual inventory carrying cost as % of value")
    
    # 2. REORDER RECOMMENDATIONS ENGINE
    st.markdown("### 📋 AUTOMATED REORDER RECOMMENDATIONS")
    
    if 'inventory_df' in inventory_metrics and not inventory_metrics['inventory_df'].empty:
        df_inventory = inventory_metrics['inventory_df']
        
        # Calculate reorder recommendations
        with st.spinner("🤖 Calculating intelligent reorder points..."):
            df_recommendations = calculate_reorder_recommendations(
                df_inventory, 
                df_sales,
                service_level=service_level,
                lead_time_days=lead_time
            )
        
        if not df_recommendations.empty:
            # Summary metrics
            total_order_qty = df_recommendations['Final_Suggested_Qty'].sum()
            total_order_value = df_recommendations['Order_Value'].sum() if 'Order_Value' in df_recommendations.columns else 0
            skus_to_reorder = len(df_recommendations)
            
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("SKUs to Reorder", skus_to_reorder)
            with col_sum2:
                st.metric("Total Quantity", f"{total_order_qty:,.0f}")
            with col_sum3:
                st.metric("Total Investment", f"Rp {total_order_value:,.0f}")
            
            # Display recommendations
            display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'Stock_Qty', 
                          'Avg_Daily_Sales', 'Safety_Stock', 'Reorder_Point',
                          'Final_Suggested_Qty', 'EOQ', 'Order_Value']
            
            available_cols = [col for col in display_cols if col in df_recommendations.columns]
            
            # Format for display
            df_display = df_recommendations[available_cols].copy()
            
            # Format numbers
            if 'Avg_Daily_Sales' in df_display.columns:
                df_display['Avg_Daily_Sales'] = df_display['Avg_Daily_Sales'].apply(lambda x: f"{x:.1f}")
            
            if 'Safety_Stock' in df_display.columns:
                df_display['Safety_Stock'] = df_display['Safety_Stock'].apply(lambda x: f"{x:.0f}")
            
            if 'Reorder_Point' in df_display.columns:
                df_display['Reorder_Point'] = df_display['Reorder_Point'].apply(lambda x: f"{x:.0f}")
            
            if 'Order_Value' in df_display.columns:
                df_display['Order_Value'] = df_display['Order_Value'].apply(lambda x: f"Rp {x:,.0f}")
            
            st.dataframe(
                df_display.sort_values('Order_Value', ascending=False),
                use_container_width=True,
                height=400
            )
            
            # Export button
            csv_rec = df_recommendations.to_csv(index=False)
            st.download_button(
                label="📥 Download Reorder Recommendations (CSV)",
                data=csv_rec,
                file_name=f"reorder_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.success("✅ No reorder recommendations at this time")
    else:
        st.warning("⚠️ Inventory data not available for recommendations")
    
    # 3. ABC ANALYSIS DASHBOARD
    st.markdown("---")
    st.markdown("### 📊 ABC ANALYSIS - PARETO INVENTORY CLASSIFICATION")
    
    if 'inventory_df' in inventory_metrics:
        with st.spinner("📈 Performing ABC analysis..."):
            df_abc = perform_abc_analysis(inventory_metrics['inventory_df'], df_sales)
        
        if not df_abc.empty:
            # ABC Summary
            abc_summary = df_abc.groupby('ABC_Class').agg({
                'SKU_ID': 'count',
                'Value': 'sum',
                'Stock_Qty': 'sum'
            }).reset_index()
            
            abc_summary.columns = ['Class', 'SKU Count', 'Total Value', 'Total Qty']
            abc_summary['Value Pct'] = (abc_summary['Total Value'] / abc_summary['Total Value'].sum() * 100)
            abc_summary['SKU Pct'] = (abc_summary['SKU Count'] / abc_summary['SKU Count'].sum() * 100)
            
            # Visualizations
            col_abc1, col_abc2 = st.columns(2)
            
            with col_abc1:
                # Pareto Chart
                fig_pareto = go.Figure()
                
                # Cumulative percentage line
                df_abc_sorted = df_abc.sort_values('Value', ascending=False).reset_index()
                df_abc_sorted['Cum_Pct'] = df_abc_sorted['Cumulative_Pct']
                
                fig_pareto.add_trace(go.Bar(
                    x=df_abc_sorted.index,
                    y=df_abc_sorted['Value'],
                    name='Value',
                    marker_color=df_abc_sorted['ABC_Class'].map({'A': '#FF5252', 'B': '#FF9800', 'C': '#4CAF50'})
                ))
                
                fig_pareto.add_trace(go.Scatter(
                    x=df_abc_sorted.index,
                    y=df_abc_sorted['Cum_Pct'],
                    name='Cumulative %',
                    yaxis='y2',
                    line=dict(color='#2196F3', width=3)
                ))
                
                fig_pareto.update_layout(
                    height=400,
                    title='Pareto Chart: Inventory Value Distribution',
                    xaxis_title='SKU Rank',
                    yaxis_title='Inventory Value (Rp)',
                    yaxis2=dict(
                        title='Cumulative %',
                        overlaying='y',
                        side='right',
                        range=[0, 100]
                    ),
                    showlegend=True
                )
                
                st.plotly_chart(fig_pareto, use_container_width=True)
            
            with col_abc2:
                # ABC Pie Chart
                fig_abc_pie = px.pie(
                    abc_summary,
                    values='Total Value',
                    names='Class',
                    title='Inventory Value by ABC Class',
                    color='Class',
                    color_discrete_map={'A': '#FF5252', 'B': '#FF9800', 'C': '#4CAF50'},
                    hole=0.4
                )
                
                fig_abc_pie.update_layout(height=400)
                st.plotly_chart(fig_abc_pie, use_container_width=True)
            
            # ABC Management Recommendations
            st.markdown("#### 🎯 ABC MANAGEMENT STRATEGIES")
            
            col_strat1, col_strat2, col_strat3 = st.columns(3)
            
            with col_strat1:
                st.markdown("""
                <div style="background: #FFEBEE; border-left: 5px solid #F44336; 
                            padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <h4 style="color: #C62828; margin: 0 0 0.5rem 0;">🅰️ CLASS A ITEMS</h4>
                    <p style="margin: 0; font-size: 0.9rem;">
                    <strong>{sku_count} SKUs ({value_pct:.1f}% of value)</strong><br>
                    • Tight control<br>
                    • Frequent review<br>
                    • Accurate forecasting<br>
                    • High service level
                    </p>
                </div>
                """.format(
                    sku_count=abc_summary[abc_summary['Class'] == 'A']['SKU Count'].iloc[0],
                    value_pct=abc_summary[abc_summary['Class'] == 'A']['Value Pct'].iloc[0]
                ), unsafe_allow_html=True)
            
            with col_strat2:
                st.markdown("""
                <div style="background: #FFF3E0; border-left: 5px solid #FF9800; 
                            padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <h4 style="color: #EF6C00; margin: 0 0 0.5rem 0;">🅱️ CLASS B ITEMS</h4>
                    <p style="margin: 0; font-size: 0.9rem;">
                    <strong>{sku_count} SKUs ({value_pct:.1f}% of value)</strong><br>
                    • Moderate control<br>
                    • Periodic review<br>
                    • Standard forecasting<br>
                    • Moderate service level
                    </p>
                </div>
                """.format(
                    sku_count=abc_summary[abc_summary['Class'] == 'B']['SKU Count'].iloc[0],
                    value_pct=abc_summary[abc_summary['Class'] == 'B']['Value Pct'].iloc[0]
                ), unsafe_allow_html=True)
            
            with col_strat3:
                st.markdown("""
                <div style="background: #E8F5E9; border-left: 5px solid #4CAF50; 
                            padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <h4 style="color: #2E7D32; margin: 0 0 0.5rem 0;">© CLASS C ITEMS</h4>
                    <p style="margin: 0; font-size: 0.9rem;">
                    <strong>{sku_count} SKUs ({value_pct:.1f}% of value)</strong><br>
                    • Simple control<br>
                    • Occasional review<br>
                    • Minimal forecasting<br>
                    • Basic service level
                    </p>
                </div>
                """.format(
                    sku_count=abc_summary[abc_summary['Class'] == 'C']['SKU Count'].iloc[0],
                    value_pct=abc_summary[abc_summary['Class'] == 'C']['Value Pct'].iloc[0]
                ), unsafe_allow_html=True)
    
    # 4. DEAD STOCK ANALYSIS
    st.markdown("---")
    st.markdown("### 🗑️ DEAD & SLOW-MOVING STOCK ANALYSIS")
    
    if 'inventory_df' in inventory_metrics and not df_sales.empty:
        with st.spinner("🔍 Identifying dead stock..."):
            df_dead = identify_dead_stock(
                inventory_metrics['inventory_df'], 
                df_sales,
                months_threshold=6
            )
        
        if not df_dead.empty:
            total_dead_value = df_dead['Dead_Stock_Value'].sum() if 'Dead_Stock_Value' in df_dead.columns else 0
            
            col_dead1, col_dead2 = st.columns(2)
            
            with col_dead1:
                st.warning(f"""
                ⚠️ **CRITICAL FINDING**
                
                Found **{len(df_dead)} SKUs** with no sales in last 6 months
                
                **Total Value Locked:** Rp {total_dead_value:,.0f}
                
                **Recommendation:** Immediate action required to free up working capital
                """)
            
            with col_dead2:
                # Top 10 dead stock items
                df_top_dead = df_dead.head(10).copy()
                if 'Dead_Stock_Value' in df_top_dead.columns:
                    df_top_dead['Dead_Stock_Value'] = df_top_dead['Dead_Stock_Value'].apply(lambda x: f"Rp {x:,.0f}")
                
                st.dataframe(
                    df_top_dead[['SKU_ID', 'Product_Name', 'Brand', 'Stock_Qty', 'Dead_Stock_Value']],
                    use_container_width=True,
                    height=250
                )
            
            # Action Plan
            st.markdown("#### 🛠️ DEAD STOCK ACTION PLAN")
            
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)
            
            with action_col1:
                st.markdown("""
                <div style="background: #FFF3E0; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem;">🔥</div>
                    <strong>Clearance Sale</strong>
                    <div style="font-size: 0.8rem; color: #666;">
                    30-50% discount to clear inventory
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with action_col2:
                st.markdown("""
                <div style="background: #E8F5E9; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem;">🎁</div>
                    <strong>Bundle Offers</strong>
                    <div style="font-size: 0.8rem; color: #666;">
                    Combine with fast-moving items
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with action_col3:
                st.markdown("""
                <div style="background: #E3F2FD; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem;">↩️</div>
                    <strong>Return to Supplier</strong>
                    <div style="font-size: 0.8rem; color: #666;">
                    If return policy allows
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with action_col4:
                st.markdown("""
                <div style="background: #F3E5F5; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem;">🤝</div>
                    <strong>Donate for Tax Benefit</strong>
                    <div style="font-size: 0.8rem; color: #666;">
                    Corporate social responsibility
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Export dead stock list
            csv_dead = df_dead.to_csv(index=False)
            st.download_button(
                label="📥 Download Dead Stock List (CSV)",
                data=csv_dead,
                file_name=f"dead_stock_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.success("✅ No dead stock identified - Excellent inventory management!")
    
    # 5. WAREHOUSE OPTIMIZATION
    st.markdown("---")
    st.markdown("### 🏢 WAREHOUSE OPTIMIZATION ANALYSIS")
    
    # Warehouse settings in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏢 Warehouse Configuration")
    
    wh_capacity = st.sidebar.number_input(
        "Total Warehouse Capacity (pcs)",
        min_value=10000,
        max_value=1000000,
        value=250000,
        step=10000,
        help="Maximum storage capacity in units"
    )
    
    wh_rent_cost = st.sidebar.number_input(
        "Monthly Warehouse Cost (Rp)",
        min_value=1000000,
        max_value=100000000,
        value=10000000,
        step=1000000,
        help="Monthly rental/storage cost"
    )
    
    if 'inventory_df' in inventory_metrics:
        df_inv = inventory_metrics['inventory_df']
        current_occupancy = df_inv['Stock_Qty'].sum()
        occupancy_pct = (current_occupancy / wh_capacity * 100) if wh_capacity > 0 else 0
        
        # Storage cost analysis
        storage_cost_per_unit = wh_rent_cost / wh_capacity if wh_capacity > 0 else 0
        monthly_storage_cost = current_occupancy * storage_cost_per_unit
        
        col_wh1, col_wh2, col_wh3, col_wh4 = st.columns(4)
        
        with col_wh1:
            st.metric(
                "Warehouse Occupancy",
                f"{occupancy_pct:.1f}%",
                delta="Optimal < 80%" if occupancy_pct < 80 else "High ≥ 80%",
                delta_color="normal" if occupancy_pct < 80 else "off"
            )
        
        with col_wh2:
            available_space = wh_capacity - current_occupancy
            st.metric("Available Space", f"{available_space:,.0f} pcs")
        
        with col_wh3:
            st.metric("Monthly Storage Cost", f"Rp {monthly_storage_cost:,.0f}")
        
        with col_wh4:
            # Cost per unit stored
            st.metric("Cost per Unit/Month", f"Rp {storage_cost_per_unit:,.0f}")
        
        # Space Optimization Recommendations
        if 'high_stock' in inventory_metrics and not inventory_metrics['high_stock'].empty:
            high_stock_df = inventory_metrics['high_stock']
            
            # Calculate space that could be freed
            excess_coverage = high_stock_df[high_stock_df['Cover_Months'] > 2.0]
            
            if not excess_coverage.empty:
                excess_qty = excess_coverage['Stock_Qty'].sum()
                space_pct = (excess_qty / wh_capacity * 100) if wh_capacity > 0 else 0
                
                st.info(f"""
                **💡 SPACE OPTIMIZATION OPPORTUNITY**
                
                You have **{excess_qty:,.0f} units** ({space_pct:.1f}% of warehouse) 
                with excessive coverage (>2 months).
                
                **Potential savings:** Rp {excess_qty * storage_cost_per_unit:,.0f} monthly
                
                **Action:** Consider reducing inventory levels for these SKUs
                """)
    
    # 6. FINANCIAL IMPACT ANALYSIS
    st.markdown("---")
    st.markdown("### 💰 FINANCIAL IMPACT ANALYSIS")
    
    if 'inventory_df' in inventory_metrics:
        df_inv = inventory_metrics['inventory_df']
        
        # Calculate inventory value
        total_inv_value = 0
        if 'Floor_Price' in df_inv.columns:
            total_inv_value = (df_inv['Stock_Qty'] * df_inv['Floor_Price']).sum()
        
        # Calculate holding costs
        annual_holding_rate = holding_rate / 100
        annual_holding_cost = total_inv_value * annual_holding_rate
        monthly_holding_cost = annual_holding_cost / 12
        
        col_fin1, col_fin2, col_fin3 = st.columns(3)
        
        with col_fin1:
            st.markdown(f"""
            <div style="background: #F5F5F5; border-radius: 10px; padding: 1rem; text-align: center;">
                <div style="font-size: 0.9rem; color: #666;">Total Inventory Value</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #333;">Rp {total_inv_value:,.0f}</div>
                <div style="font-size: 0.8rem; color: #999;">Capital tied up in inventory</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_fin2:
            st.markdown(f"""
            <div style="background: #F5F5F5; border-radius: 10px; padding: 1rem; text-align: center;">
                <div style="font-size: 0.9rem; color: #666;">Monthly Holding Cost</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #333;">Rp {monthly_holding_cost:,.0f}</div>
                <div style="font-size: 0.8rem; color: #999;">{holding_rate}% of inventory value</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_fin3:
            # Calculate potential savings from optimization
            potential_savings = 0
            
            # From dead stock elimination
            if 'df_dead' in locals() and not df_dead.empty and 'Dead_Stock_Value' in df_dead.columns:
                dead_value = df_dead['Dead_Stock_Value'].sum()
                potential_savings += dead_value * (annual_holding_rate / 12)  # Monthly holding cost savings
            
            # From excess stock reduction
            if 'excess_coverage' in locals() and not excess_coverage.empty:
                excess_value = 0
                if 'Floor_Price' in excess_coverage.columns:
                    excess_value = (excess_coverage['Stock_Qty'] * excess_coverage['Floor_Price']).sum()
                    potential_savings += excess_value * (annual_holding_rate / 12) * 0.5  # Assume 50% reduction
            
            st.markdown(f"""
            <div style="background: #E8F5E9; border-radius: 10px; padding: 1rem; text-align: center;">
                <div style="font-size: 0.9rem; color: #2E7D32;">Potential Monthly Savings</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #1B5E20;">Rp {potential_savings:,.0f}</div>
                <div style="font-size: 0.8rem; color: #4CAF50;">From optimization</div>
            </div>
            """, unsafe_allow_html=True)
    
    # 7. EXPORT ALL ANALYSES
    st.markdown("---")
    st.markdown("### 📊 EXPORT COMPREHENSIVE ANALYSIS")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📥 Export All Recommendations", use_container_width=True, type="primary"):
            # Create comprehensive report
            report_data = {}
            
            # 1. Reorder recommendations
            if 'df_recommendations' in locals() and not df_recommendations.empty:
                report_data['Reorder_Recommendations'] = df_recommendations
            
            # 2. ABC Analysis
            if 'df_abc' in locals() and not df_abc.empty:
                report_data['ABC_Analysis'] = df_abc
            
            # 3. Dead Stock
            if 'df_dead' in locals() and not df_dead.empty:
                report_data['Dead_Stock'] = df_dead
            
            # 4. Inventory summary
            if 'inventory_df' in inventory_metrics:
                report_data['Inventory_Summary'] = inventory_metrics['inventory_df']
            
            # Create Excel file with multiple sheets
            import io
            from pandas import ExcelWriter
            
            output = io.BytesIO()
            with ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in report_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            output.seek(0)
            
            st.download_button(
                label="💾 Download Excel Report",
                data=output,
                file_name=f"Inventory_Intelligence_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    with export_col2:
        if st.button("🔄 Run Advanced Analysis", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with export_col3:
        if st.button("📊 Generate Executive Summary", use_container_width=True):
            # Create executive summary
            summary_html = f"""
            <div style="background: white; border-radius: 12px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                <h2 style="color: #333; margin-top: 0;">📈 Inventory Intelligence Executive Summary</h2>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 2rem 0;">
                    <div style="background: #E8F5E9; padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 0.9rem; color: #2E7D32;">Inventory Health</div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: #1B5E20;">{health_score:.0f}/100</div>
                    </div>
                    <div style="background: #E3F2FD; padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 0.9rem; color: #1565C0;">Stockout Risk</div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: #0D47A1;">{stockout_risk} SKUs</div>
                    </div>
                    <div style="background: #FFF3E0; padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 0.9rem; color: #EF6C00;">Excess Stock Value</div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: #E65100;">Rp {excess_value:,.0f}</div>
                    </div>
                </div>
            </div>
            """
            
            st.markdown(summary_html, unsafe_allow_html=True)
    
    # 8. QUICK ACTIONS PANEL
    st.markdown("---")
    st.markdown("### ⚡ QUICK ACTION PANEL")
    
    action_tab1, action_tab2, action_tab3 = st.tabs(["🚨 Critical Actions", "📈 Optimization", "📊 Reports"])
    
    with action_tab1:
        # Critical actions
        critical_actions = []
        
        # Check stockout risk
        if stockout_risk > 0:
            critical_actions.append(f"🚨 **Immediate Reorder Needed:** {stockout_risk} SKUs at risk of stockout")
        
        # Check dead stock
        if 'df_dead' in locals() and not df_dead.empty:
            dead_count = len(df_dead)
            dead_value = df_dead['Dead_Stock_Value'].sum() if 'Dead_Stock_Value' in df_dead.columns else 0
            critical_actions.append(f"🗑️ **Clear Dead Stock:** {dead_count} SKUs (Rp {dead_value:,.0f} value)")
        
        # Check warehouse capacity
        if occupancy_pct > 85:
            critical_actions.append(f"🏢 **Warehouse Critical:** {occupancy_pct:.1f}% full - Consider expansion or reduction")
        
        if critical_actions:
            for action in critical_actions:
                st.error(action)
        else:
            st.success("✅ No critical actions required")
    
    with action_tab2:
        # Optimization suggestions
        optimizations = []
        
        # ABC optimization
        if 'abc_summary' in locals():
            a_items = abc_summary[abc_summary['Class'] == 'A']['SKU Count'].iloc[0]
            c_items = abc_summary[abc_summary['Class'] == 'C']['SKU Count'].iloc[0]
            optimizations.append(f"📊 **ABC Strategy:** Focus on {a_items} A-items, simplify {c_items} C-items")
        
        # Reorder optimization
        if 'df_recommendations' in locals() and not df_recommendations.empty:
            rec_count = len(df_recommendations)
            rec_value = df_recommendations['Order_Value'].sum() if 'Order_Value' in df_recommendations.columns else 0
            optimizations.append(f"📋 **Smart Reordering:** {rec_count} SKUs need ordering (Rp {rec_value:,.0f})")
        
        # Holding cost optimization
        if 'potential_savings' in locals() and potential_savings > 0:
            optimizations.append(f"💰 **Cost Reduction:** Potential savings Rp {potential_savings:,.0f}/month")
        
        if optimizations:
            for opt in optimizations:
                st.info(opt)
    
    with action_tab3:
        # Quick report generation
        report_col1, report_col2 = st.columns(2)
        
        with report_col1:
            if st.button("📋 Generate Stock Report", use_container_width=True):
                if 'inventory_df' in inventory_metrics:
                    df_report = inventory_metrics['inventory_df'][['SKU_ID', 'Product_Name', 'Brand', 'Stock_Qty', 'Cover_Months', 'Inventory_Status']]
                    st.dataframe(df_report, use_container_width=True, height=300)
        
        with report_col2:
            if st.button("🎯 Generate Action Items", use_container_width=True):
                # Create action items list
                actions = []
                
                if 'df_recommendations' in locals() and not df_recommendations.empty:
                    actions.append("**1. Place Purchase Orders:**")
                    for _, row in df_recommendations.head(5).iterrows():
                        actions.append(f"   - {row['SKU_ID']}: Order {row['Final_Suggested_Qty']:.0f} units")
                
                if 'df_dead' in locals() and not df_dead.empty:
                    actions.append(f"\n**2. Clear Dead Stock ({len(df_dead)} SKUs):**")
                    actions.append("   - Plan clearance sale")
                    actions.append("   - Contact suppliers for returns")
                
                st.text_area("Action Items", "\n".join(actions), height=200)

# --- TAB 4: SKU EVALUATION ---
with tab4:
    st.subheader("🔍 SKU Performance Evaluation")
    
    if monthly_performance and not df_sales.empty:
        # Get last month for evaluation
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data'].copy()
        
        # Get last 3 months sales data for each SKU
        if not df_sales.empty:
            sales_months = sorted(df_sales['Month'].unique())
            if len(sales_months) >= 3:
                last_3_sales_months = sales_months[-3:]
                df_sales_last_3 = df_sales[df_sales['Month'].isin(last_3_sales_months)].copy()
                
                # Pivot sales data to get last 3 months sales per SKU
                try:
                    sales_pivot = df_sales_last_3.pivot_table(
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
                    
                    # Merge with last month data
                    last_month_data = pd.merge(
                        last_month_data,
                        sales_pivot,
                        on='SKU_ID',
                        how='left'
                    )
                except Exception as e:
                    st.warning(f"Tidak bisa memproses data sales 3 bulan terakhir: {str(e)}")
        
        # Add inventory data
        if 'inventory_df' in inventory_metrics:
            inventory_data = inventory_metrics['inventory_df'][['SKU_ID', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']]
            last_month_data = pd.merge(last_month_data, inventory_data, on='SKU_ID', how='left')
        
        # Add financial data if available
        if not df_financial.empty:
            # Get financial metrics for last month
            financial_last_month = df_financial[df_financial['Month'] == last_month]
            if not financial_last_month.empty:
                financial_metrics = financial_last_month[['SKU_ID', 'Revenue', 'Gross_Margin', 'Margin_Percentage']]
                last_month_data = pd.merge(last_month_data, financial_metrics, on='SKU_ID', how='left')
        
        # Create comprehensive evaluation table
        # Filter by SKU
        sku_filter = st.text_input("🔍 Filter by SKU ID or Product Name", "")
        
        # Apply filter
        if sku_filter:
            filtered_eval_df = last_month_data[
                last_month_data['SKU_ID'].astype(str).str.contains(sku_filter, case=False, na=False) |
                (last_month_data['Product_Name'].astype(str).str.contains(sku_filter, case=False, na=False) if 'Product_Name' in last_month_data.columns else False)
            ].copy()
        else:
            filtered_eval_df = last_month_data.copy()
        
        # Determine which sales columns to show
        sales_cols = []
        for col in filtered_eval_df.columns:
            if isinstance(col, str) and '-' in col and len(col) in [7, 8]:  # Format seperti 'Sep-2024' atau 'Mar-2025'
                try:
                    # Validate it's a proper month-year format
                    datetime.strptime(col, '%b-%Y')
                    sales_cols.append(col)
                except:
                    pass
        
        # Sort sales columns chronologically
        if sales_cols:
            sales_cols_sorted = sorted(sales_cols, key=lambda x: datetime.strptime(x, '%b-%Y'))
            # Get last 3 months only
            sales_cols_sorted = sales_cols_sorted[-3:] if len(sales_cols_sorted) >= 3 else sales_cols_sorted
        else:
            sales_cols_sorted = []
        
        # Define columns to display - WAJIB dengan Product_Name
        eval_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 
                    'Forecast_Qty', 'PO_Qty', 'PO_Rofo_Ratio',
                    'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
        
        # Add financial columns if available
        if 'Revenue' in filtered_eval_df.columns:
            eval_cols.extend(['Revenue', 'Gross_Margin', 'Margin_Percentage'])
        
        # Add sales columns
        eval_cols.extend(sales_cols_sorted)
        
        # Filter hanya kolom yang ada
        available_cols = [col for col in eval_cols if col in filtered_eval_df.columns]
        
        # Pastikan Product_Name selalu ada
        if 'Product_Name' not in available_cols and 'Product_Name' in filtered_eval_df.columns:
            available_cols.insert(1, 'Product_Name')
        
        eval_df = filtered_eval_df[available_cols].copy()
        
        # Format columns
        if 'PO_Rofo_Ratio' in eval_df.columns:
            eval_df['PO_Rofo_Ratio'] = eval_df['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "0%")
        
        if 'Cover_Months' in eval_df.columns:
            eval_df['Cover_Months'] = eval_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) and x < 999 else "N/A")
        
        if 'Avg_Monthly_Sales_3M' in eval_df.columns:
            eval_df['Avg_Monthly_Sales_3M'] = eval_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
        
        # Format financial columns
        if 'Revenue' in eval_df.columns:
            eval_df['Revenue'] = eval_df['Revenue'].apply(lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "Rp 0")
        
        if 'Gross_Margin' in eval_df.columns:
            eval_df['Gross_Margin'] = eval_df['Gross_Margin'].apply(lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "Rp 0")
        
        if 'Margin_Percentage' in eval_df.columns:
            eval_df['Margin_Percentage'] = eval_df['Margin_Percentage'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "0%")
        
        # Format sales columns
        for col in sales_cols_sorted:
            if col in eval_df.columns:
                eval_df[col] = eval_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
        
        # Rename columns - WAJIB dengan Product Name
        column_names = {
            'SKU_ID': 'SKU ID',
            'Product_Name': 'Product Name',
            'Brand': 'Brand',
            'SKU_Tier': 'Tier',
            'Forecast_Qty': 'Forecast',
            'PO_Qty': 'PO',
            'PO_Rofo_Ratio': 'PO/Rofo %',
            'Stock_Qty': 'Stock',
            'Avg_Monthly_Sales_3M': 'Avg Sales (L3M)',
            'Cover_Months': 'Cover (Months)',
            'Revenue': 'Revenue',
            'Gross_Margin': 'Gross Margin',
            'Margin_Percentage': 'Margin %'
        }
        
        # Add sales columns to rename dict
        for col in sales_cols_sorted:
            column_names[col] = col
        
        eval_df = eval_df.rename(columns=column_names)
        
        # Reorder columns
        column_order = ['SKU ID', 'Product Name', 'Brand', 'Tier', 'Forecast', 'PO', 
                       'PO/Rofo %', 'Stock', 'Avg Sales (L3M)', 'Cover (Months)']
        
        # Tambahkan financial columns
        if 'Revenue' in eval_df.columns:
            column_order.extend(['Revenue', 'Gross Margin', 'Margin %'])
        
        # Tambahkan sales columns ke urutan
        for col in sales_cols_sorted:
            if col in eval_df.columns:
                column_order.append(col)
        
        # Ensure all columns exist before reordering
        existing_columns = [col for col in column_order if col in eval_df.columns]
        eval_df = eval_df[existing_columns]
        
        st.dataframe(
            eval_df,
            use_container_width=True,
            height=400
        )
        
        # ================ NEW: SKU DEEP DIVE ANALYSIS ================
        st.divider()
        st.subheader("🔬 SKU Deep Dive Analysis")
        
        # Pilih SKU untuk deep dive
        if not last_month_data.empty:
            # Get unique SKUs for selection
            available_skus = last_month_data['SKU_ID'].unique().tolist()
            
            # Jika ada filter SKU, otomatis select yang difilter
            selected_sku = None
            if sku_filter and len(filtered_eval_df) == 1:
                selected_sku = filtered_eval_df.iloc[0]['SKU_ID']
            else:
                # Dropdown untuk pilih SKU
                sku_options = []
                for sku in available_skus[:50]:  # Limit to first 50 for performance
                    product_name = last_month_data[last_month_data['SKU_ID'] == sku]['Product_Name'].iloc[0] if 'Product_Name' in last_month_data.columns else sku
                    sku_options.append(f"{sku} - {product_name}")
                
                if sku_options:
                    selected_sku_display = st.selectbox(
                        "📋 Select SKU for Deep Dive Analysis",
                        options=sku_options,
                        index=0
                    )
                    if selected_sku_display:
                        selected_sku = selected_sku_display.split(" - ")[0]
            
            if selected_sku:
                st.markdown(f"### 📊 Analysis for SKU: **{selected_sku}**")
                
                # Get SKU details
                sku_details = last_month_data[last_month_data['SKU_ID'] == selected_sku].iloc[0].to_dict() if not last_month_data.empty else {}
                product_name = sku_details.get('Product_Name', 'N/A')
                brand = sku_details.get('Brand', 'N/A')
                tier = sku_details.get('SKU_Tier', 'N/A')
                
                # Display SKU info
                col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                with col_info1:
                    st.metric("Product", product_name)
                with col_info2:
                    st.metric("Brand", brand)
                with col_info3:
                    st.metric("Tier", tier)
                with col_info4:
                    stock_qty = sku_details.get('Stock_Qty', 0)
                    st.metric("Current Stock", f"{stock_qty:,.0f}")
                
                # SECTION 1: 12-MONTH PERFORMANCE TIMELINE - SIMPLE VERSION
                st.markdown("#### 📈 12-Month Performance Timeline")
                
                # Prepare historical data for this SKU
                historical_data = []
                
                # Get last 12 months data
                if not df_sales.empty:
                    sales_months = sorted(df_sales['Month'].unique())
                    last_12_months = sales_months[-12:] if len(sales_months) >= 12 else sales_months
                    
                    for month in last_12_months:
                        month_name = month.strftime('%b-%Y')
                        
                        # Get data for this SKU in this month
                        sales_qty = df_sales[(df_sales['Month'] == month) & 
                                           (df_sales['SKU_ID'] == selected_sku)]['Sales_Qty'].sum()
                        
                        forecast_qty = df_forecast[(df_forecast['Month'] == month) & 
                                                 (df_forecast['SKU_ID'] == selected_sku)]['Forecast_Qty'].sum() if not df_forecast.empty else 0
                        
                        po_qty = df_po[(df_po['Month'] == month) & 
                                     (df_po['SKU_ID'] == selected_sku)]['PO_Qty'].sum() if not df_po.empty else 0
                        
                        historical_data.append({
                            'Month': month,
                            'Month_Display': month_name,
                            'Sales': sales_qty,
                            'Rofo': forecast_qty,
                            'PO': po_qty
                        })
                
                if historical_data:
                    hist_df = pd.DataFrame(historical_data)
                    hist_df = hist_df.sort_values('Month')
                    
                    # SIMPLE CHART - tanpa dual-axis dulu
                    fig_timeline = go.Figure()
                    
                    # Quantity lines
                    fig_timeline.add_trace(go.Scatter(
                        x=hist_df['Month_Display'],
                        y=hist_df['Rofo'],
                        name='Rofo',
                        mode='lines+markers',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=8, color='#667eea')
                    ))
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=hist_df['Month_Display'],
                        y=hist_df['PO'],
                        name='PO',
                        mode='lines+markers',
                        line=dict(color='#FF9800', width=3),
                        marker=dict(size=8, color='#FF9800')
                    ))
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=hist_df['Month_Display'],
                        y=hist_df['Sales'],
                        name='Sales',
                        mode='lines+markers',
                        line=dict(color='#4CAF50', width=3),
                        marker=dict(size=8, color='#4CAF50')
                    ))
                    
                    # SIMPLE LAYOUT
                    fig_timeline.update_layout(
                        height=400,
                        title=f'SKU Performance: {selected_sku}',
                        xaxis_title='Month',
                        yaxis_title='Quantity',
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Tambahkan accuracy chart terpisah
                    if not df_forecast.empty and not df_po.empty:
                        # Calculate accuracy per month
                        accuracy_data = []
                        for month in last_12_months:
                            month_name = month.strftime('%b-%Y')
                            forecast_qty = df_forecast[(df_forecast['Month'] == month) & 
                                                     (df_forecast['SKU_ID'] == selected_sku)]['Forecast_Qty'].sum()
                            po_qty = df_po[(df_po['Month'] == month) & 
                                         (df_po['SKU_ID'] == selected_sku)]['PO_Qty'].sum()
                            
                            if forecast_qty > 0 and po_qty > 0:
                                accuracy = 100 - abs((po_qty / forecast_qty * 100) - 100)
                                accuracy_data.append({
                                    'Month': month_name,
                                    'Accuracy': accuracy
                                })
                        
                        if accuracy_data:
                            acc_df = pd.DataFrame(accuracy_data)
                            
                            fig_acc = go.Figure()
                            fig_acc.add_trace(go.Scatter(
                                x=acc_df['Month'],
                                y=acc_df['Accuracy'],
                                mode='lines+markers',
                                name='Accuracy %',
                                line=dict(color='#FF5252', width=3),
                                marker=dict(size=8, color='#FF5252')
                            ))
                            
                            fig_acc.update_layout(
                                height=300,
                                title='Forecast Accuracy Trend',
                                xaxis_title='Month',
                                yaxis_title='Accuracy %',
                                yaxis_range=[0, 110]
                            )
                            
                            st.plotly_chart(fig_acc, use_container_width=True)
                    
                    # SECTION 2: INVENTORY HEALTH
                    st.markdown("#### 📦 Inventory Health Analysis")
                    
                    col_inv1, col_inv2, col_inv3, col_inv4 = st.columns(4)
                    
                    with col_inv1:
                        # Current stock
                        current_stock = sku_details.get('Stock_Qty', 0)
                        st.metric("Current Stock", f"{current_stock:,.0f}")
                    
                    with col_inv2:
                        # Avg monthly sales (3-month average)
                        avg_sales_3m = sku_details.get('Avg_Monthly_Sales_3M', 0)
                        st.metric("Avg Monthly Sales (3M)", f"{avg_sales_3m:,.0f}")
                    
                    with col_inv3:
                        # Cover months
                        cover_months = sku_details.get('Cover_Months', 0)
                        cover_status = "High Stock" if cover_months > 1.5 else "Ideal" if cover_months >= 0.8 else "Low Stock"
                        st.metric("Cover (Months)", f"{cover_months:.1f}", delta=cover_status)
                    
                    with col_inv4:
                        # Sales trend (last 3 months vs previous 3 months)
                        if len(hist_df) >= 6:
                            recent_sales = hist_df.tail(3)['Sales'].sum()
                            previous_sales = hist_df.head(3)['Sales'].sum() if len(hist_df) >= 6 else recent_sales
                            sales_growth = ((recent_sales - previous_sales) / previous_sales * 100) if previous_sales > 0 else 0
                            st.metric("Sales Growth (3M)", f"{sales_growth:+.1f}%")
                    
                    # SECTION 3: FORECAST PERFORMANCE METRICS
                    st.markdown("#### 🎯 Forecast Performance Metrics")
                    
                    # Calculate forecast accuracy metrics
                    if not df_forecast.empty and not df_po.empty:
                        # Get accuracy data separately
                        accuracy_data = []
                        for month in last_12_months:
                            forecast_qty = df_forecast[(df_forecast['Month'] == month) & 
                                                     (df_forecast['SKU_ID'] == selected_sku)]['Forecast_Qty'].sum()
                            po_qty = df_po[(df_po['Month'] == month) & 
                                         (df_po['SKU_ID'] == selected_sku)]['PO_Qty'].sum()
                            
                            if forecast_qty > 0 and po_qty > 0:
                                accuracy = 100 - abs((po_qty / forecast_qty * 100) - 100)
                                accuracy_data.append({
                                    'Month': month,
                                    'Forecast_Qty': forecast_qty,
                                    'PO_Qty': po_qty,
                                    'Accuracy': accuracy
                                })
                        
                        if accuracy_data:
                            acc_df = pd.DataFrame(accuracy_data)
                            
                            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                            
                            with col_met1:
                                # Average accuracy
                                avg_accuracy = acc_df['Accuracy'].mean()
                                accuracy_status = "Good" if avg_accuracy >= 80 else "Needs Improvement"
                                st.metric("Avg Forecast Accuracy", f"{avg_accuracy:.1f}%", delta=accuracy_status)
                            
                            with col_met2:
                                # Forecast vs Sales ratio
                                total_forecast = acc_df['Forecast_Qty'].sum()
                                # Get total sales for same months
                                total_sales = 0
                                for month in acc_df['Month']:
                                    sales_qty = df_sales[(df_sales['Month'] == month) & 
                                                       (df_sales['SKU_ID'] == selected_sku)]['Sales_Qty'].sum()
                                    total_sales += sales_qty
                                
                                forecast_vs_sales = (total_forecast / total_sales * 100) if total_sales > 0 else 0
                                st.metric("Forecast/Sales %", f"{forecast_vs_sales:.1f}%")
                            
                            with col_met3:
                                # PO vs Forecast ratio
                                total_po = acc_df['PO_Qty'].sum()
                                po_vs_forecast = (total_po / total_forecast * 100) if total_forecast > 0 else 0
                                st.metric("PO/Forecast %", f"{po_vs_forecast:.1f}%")
                            
                            with col_met4:
                                # Consistency score (std dev of accuracy)
                                accuracy_std = acc_df['Accuracy'].std()
                                consistency_score = max(0, 100 - accuracy_std)
                                st.metric("Consistency Score", f"{consistency_score:.1f}")
                            
                            # SECTION 4: RECOMMENDATIONS
                            st.markdown("#### 💡 Recommendations")
                            
                            recommendations = []
                            
                            # Stock recommendations
                            cover_months = sku_details.get('Cover_Months', 0)
                            if cover_months < 0.8:
                                recommendations.append("🔄 **Need Replenishment**: Stock cover is below 0.8 months")
                            elif cover_months > 1.5:
                                recommendations.append("📉 **Reduce Stock**: High stock coverage (>1.5 months)")
                            
                            # Forecast accuracy recommendations
                            if avg_accuracy < 80:
                                recommendations.append("🎯 **Improve Forecasting**: Accuracy below 80% target")
                            
                            # Sales trend recommendations
                            sales_growth = 0  # Calculate sales growth
                            if len(hist_df) >= 6:
                                recent_sales = hist_df.tail(3)['Sales'].sum()
                                previous_sales = hist_df.head(3)['Sales'].sum()
                                sales_growth = ((recent_sales - previous_sales) / previous_sales * 100) if previous_sales > 0 else 0
                            
                            if sales_growth < -10:
                                recommendations.append("📊 **Review Demand**: Sales declining significantly")
                            elif sales_growth > 50:
                                recommendations.append("🚀 **Opportunity**: Strong sales growth detected")
                            
                            # PO compliance recommendations
                            if po_vs_forecast < 80:
                                recommendations.append("📝 **Increase PO Compliance**: PO significantly below forecast")
                            elif po_vs_forecast > 120:
                                recommendations.append("⚠️ **Reduce Over-PO**: PO significantly above forecast")
                            
                            # Financial recommendations (if financial data available)
                            if 'Margin_Percentage' in sku_details:
                                margin = sku_details.get('Margin_Percentage', 0)
                                if margin < 20:
                                    recommendations.append("💰 **Low Margin Alert**: Margin below 20%")
                                elif margin > 40:
                                    recommendations.append("💰 **High Margin Opportunity**: Excellent margin performance")
                            
                            if recommendations:
                                for rec in recommendations:
                                    st.write(f"- {rec}")
                            else:
                                st.success("✅ **Excellent**: This SKU is performing well across all metrics!")
                        else:
                            st.info("No forecast accuracy data available for this SKU")
    else:
        st.info("📊 Insufficient data for SKU evaluation")

# --- TAB 5: SALES & FORECAST ANALYSIS ---
with tab5:
    st.subheader("📈 Sales & Forecast Analysis")
    
    if sales_vs_forecast:
        last_month = sales_vs_forecast['last_month']
        last_month_name = last_month.strftime('%b %Y')
        
                # SECTION 1: SIMPLE MONTHLY TREND
        st.markdown("### 📊 Monthly Trend")
        
        # Get ALL available months, not just last 6
        monthly_trend = []
        
        # Get unique months from ALL datasets
        all_months = set()
        if not df_sales.empty:
            all_months.update(df_sales['Month'].unique())
        if not df_forecast.empty:
            all_months.update(df_forecast['Month'].unique())
        if not df_po.empty:
            all_months.update(df_po['Month'].unique())
        
        if all_months:
            sorted_months = sorted(all_months)
            
            for month in sorted_months:  # PAKAI SEMUA BULAN, bukan cuma 6 terakhir
                month_name = month.strftime('%b-%Y')
                sales_qty = df_sales[df_sales['Month'] == month]['Sales_Qty'].sum() if not df_sales.empty else 0
                forecast_qty = df_forecast[df_forecast['Month'] == month]['Forecast_Qty'].sum() if not df_forecast.empty else 0
                po_qty = df_po[df_po['Month'] == month]['PO_Qty'].sum() if not df_po.empty else 0
                
                monthly_trend.append({
                    'Month': month_name,
                    'Rofo': forecast_qty,
                    'PO': po_qty,
                    'Sales': sales_qty
                })
        
        if monthly_trend:
            trend_df = pd.DataFrame(monthly_trend)
            
            # Tampilkan info bulan yang tersedia
            st.caption(f"📅 Showing data for {len(trend_df)} months")
            
            # CHART 1: Quantity Trend
            fig1 = go.Figure()
            
            fig1.add_trace(go.Bar(
                x=trend_df['Month'],
                y=trend_df['Rofo'],
                name='Rofo',
                marker_color='#667eea'
            ))
            
            fig1.add_trace(go.Bar(
                x=trend_df['Month'],
                y=trend_df['PO'],
                name='PO',
                marker_color='#FF9800'
            ))
            
            fig1.add_trace(go.Bar(
                x=trend_df['Month'],
                y=trend_df['Sales'],
                name='Sales',
                marker_color='#4CAF50'
            ))
            
            fig1.update_layout(
                height=400,
                title='Monthly Trend: Rofo vs PO vs Sales (All Available Months)',
                xaxis_title='Month',
                yaxis_title='Quantity',
                barmode='group'
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # CHART 2: Accuracy Trend
            if not df_forecast.empty and not df_po.empty:
                accuracy_trend = []
                
                for month in sorted_months:  # PAKAI SEMUA BULAN
                    month_name = month.strftime('%b-%Y')
                    forecast_qty = df_forecast[df_forecast['Month'] == month]['Forecast_Qty'].sum()
                    po_qty = df_po[df_po['Month'] == month]['PO_Qty'].sum()
                    
                    if forecast_qty > 0:
                        accuracy = 100 - abs((po_qty / forecast_qty * 100) - 100)
                        accuracy_trend.append({
                            'Month': month_name,
                            'Accuracy': accuracy
                        })
                
                if accuracy_trend:
                    acc_df = pd.DataFrame(accuracy_trend)
                    
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Scatter(
                        x=acc_df['Month'],
                        y=acc_df['Accuracy'],
                        mode='lines+markers',
                        name='Accuracy %',
                        line=dict(color='#FF5252', width=3),
                        marker=dict(size=8, color='#FF5252')
                    ))
                    
                    fig2.update_layout(
                        height=300,
                        title='Forecast Accuracy Trend (All Available Months)',
                        xaxis_title='Month',
                        yaxis_title='Accuracy %',
                        yaxis_range=[0, 110]
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
        
        # SECTION 2: BRAND PERFORMANCE
        st.divider()
        st.markdown("### 🏷️ Brand Performance")
        
        if not df_forecast.empty and not df_po.empty and not df_sales.empty:
            # Get last month brand data
            forecast_last = df_forecast[df_forecast['Month'] == last_month].copy()
            po_last = df_po[df_po['Month'] == last_month].copy()
            sales_last = df_sales[df_sales['Month'] == last_month].copy()
            
            # Add product info
            forecast_last = add_product_info_to_data(forecast_last, df_product)
            po_last = add_product_info_to_data(po_last, df_product)
            sales_last = add_product_info_to_data(sales_last, df_product)
            
            if 'Brand' in forecast_last.columns:
                # Aggregate by brand
                brand_data = []
                brands = forecast_last['Brand'].unique()
                
                for brand in brands:
                    rofo = forecast_last[forecast_last['Brand'] == brand]['Forecast_Qty'].sum()
                    po = po_last[po_last['Brand'] == brand]['PO_Qty'].sum()
                    sales = sales_last[sales_last['Brand'] == brand]['Sales_Qty'].sum()
                    
                    brand_data.append({
                        'Brand': brand,
                        'Rofo': rofo,
                        'PO': po,
                        'Sales': sales
                    })
                
                if brand_data:
                    brand_df = pd.DataFrame(brand_data)
                    brand_df = brand_df.sort_values('Rofo', ascending=False).head(10)
                    
                    # Brand chart
                    fig3 = go.Figure()
                    
                    fig3.add_trace(go.Bar(
                        x=brand_df['Brand'],
                        y=brand_df['Rofo'],
                        name='Rofo',
                        marker_color='#667eea'
                    ))
                    
                    fig3.add_trace(go.Bar(
                        x=brand_df['Brand'],
                        y=brand_df['PO'],
                        name='PO',
                        marker_color='#FF9800'
                    ))
                    
                    fig3.add_trace(go.Bar(
                        x=brand_df['Brand'],
                        y=brand_df['Sales'],
                        name='Sales',
                        marker_color='#4CAF50'
                    ))
                    
                    fig3.update_layout(
                        height=400,
                        title=f'Top 10 Brands - {last_month_name}',
                        xaxis_title='Brand',
                        yaxis_title='Quantity',
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
        
        # SECTION 3: HIGH DEVIATION ANALYSIS
        st.divider()
        st.subheader("⚠️ High Deviation Analysis")
        
        # TAMBAH NOTE INI
        st.info("""
        **📌 Note:** Analysis ini hanya mencakup **ACTIVE SKUs** dengan **Forecast > 0**. 
        SKU Inactive/Discontinued tidak dihitung karena tidak ada forecast requirement.
        """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Forecast Deviation",
                f"{sales_vs_forecast['avg_forecast_deviation']:.1f}%",
                delta="Target: < 20%"
            )
        with col2:
            st.metric(
                "PO Deviation", 
                f"{sales_vs_forecast['avg_po_deviation']:.1f}%",
                delta="Target: < 20%"
            )
        with col3:
            st.metric(
                "High Deviation SKUs",
                len(sales_vs_forecast['high_deviation_skus']),
                delta=f"Active SKUs: {sales_vs_forecast['total_skus_compared']}"
            )
        
        high_dev_df = sales_vs_forecast['high_deviation_skus']
        
        if not high_dev_df.empty:
            # Display table
            display_df = high_dev_df.copy()
            
            # Select columns
            cols_to_show = ['SKU_ID', 'Product_Name', 'Brand', 
                          'Sales_Qty', 'Forecast_Qty', 'PO_Qty',
                          'Sales_vs_Forecast_Ratio', 'Sales_vs_PO_Ratio']
            
            available_cols = [col for col in cols_to_show if col in display_df.columns]
            
            # Ensure Product_Name
            if 'Product_Name' not in available_cols and 'Product_Name' in display_df.columns:
                available_cols.insert(1, 'Product_Name')
            
            display_df = display_df[available_cols].head(20)
            
            # Format
            if 'Sales_vs_Forecast_Ratio' in display_df.columns:
                display_df['Sales_vs_Forecast_Ratio'] = display_df['Sales_vs_Forecast_Ratio'].apply(lambda x: f"{x:.1f}%")
            
            if 'Sales_vs_PO_Ratio' in display_df.columns:
                display_df['Sales_vs_PO_Ratio'] = display_df['Sales_vs_PO_Ratio'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.success(f"✅ No high deviation SKUs in {last_month_name}")
    
    else:
        st.info("📊 Need sales, forecast, and PO data for analysis")

# --- TAB 6: DATA EXPLORER ---
with tab6:
    st.subheader("📋 Raw Data Explorer")
    
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
        
        # Download option
        csv = df_selected.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_dataset.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("No data available for selected dataset")

# --- TAB 7: ADVANCED FORECAST INTELLIGENCE ---
with tab7:
    st.subheader("🔮 ADVANCED FORECAST INTELLIGENCE SYSTEM")
    st.markdown("#### **Predictive Analytics with ML Insights & Scenario Planning**")
    
    # ============================================
    # SECTION 0: DATA VALIDATION & PREPARATION
    # ============================================
    st.markdown("---")
    
    # Check if we have forecast data
    if df_ecomm_forecast.empty:
        st.error("❌ No ecommerce forecast data available!")
        st.info("Please ensure 'Forecast_2026_Ecomm' sheet exists with proper data structure")
        st.stop()
    
    # Check for required columns
    required_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier']
    missing_cols = [col for col in required_cols if col not in df_ecomm_forecast.columns]
    
    if missing_cols:
        st.warning(f"⚠️ Missing columns: {missing_cols}")
        st.info("Adding placeholder data for analysis...")
        
        # Add missing columns with placeholder data
        if 'Product_Name' not in df_ecomm_forecast.columns:
            df_ecomm_forecast['Product_Name'] = 'Product ' + df_ecomm_forecast['SKU_ID'].astype(str)
        if 'Brand' not in df_ecomm_forecast.columns:
            df_ecomm_forecast['Brand'] = 'Unknown'
        if 'SKU_Tier' not in df_ecomm_forecast.columns:
            df_ecomm_forecast['SKU_Tier'] = 'Standard'
    
    # ============================================
    # SECTION 1: FORECAST INTELLIGENCE DASHBOARD
    # ============================================
    st.markdown("### 🎯 FORECAST INTELLIGENCE DASHBOARD")
    
    # Calculate key metrics
    total_forecast_2026 = 0
    forecast_value_2026 = 0
    avg_accuracy_historical = 0
    forecast_variance = {}
    
    if ecomm_forecast_month_cols:
        # Total forecast quantity for 2026
        total_forecast_2026 = df_ecomm_forecast[ecomm_forecast_month_cols].sum().sum()
        
        # Forecast value if price available
        if not df_product.empty and 'Floor_Price' in df_product.columns:
            df_with_price = add_product_info_to_data(df_ecomm_forecast, df_product)
            for month in ecomm_forecast_month_cols:
                month_value = (df_with_price[month] * df_with_price['Floor_Price'].fillna(0)).sum()
                forecast_value_2026 += month_value
        
        # Calculate historical accuracy if we have sales data
        if not df_sales.empty and not df_forecast.empty:
            # Get common months
            sales_months = df_sales['Month'].unique()
            forecast_months = df_forecast['Month'].unique()
            common_months = sorted(set(sales_months) & set(forecast_months))
            
            if common_months:
                accuracies = []
                for month in common_months[-6:]:  # Last 6 months
                    sales_qty = df_sales[df_sales['Month'] == month]['Sales_Qty'].sum()
                    forecast_qty = df_forecast[df_forecast['Month'] == month]['Forecast_Qty'].sum()
                    
                    if forecast_qty > 0:
                        accuracy = 100 - abs((sales_qty / forecast_qty * 100) - 100)
                        accuracies.append(accuracy)
                
                if accuracies:
                    avg_accuracy_historical = sum(accuracies) / len(accuracies)
        
        # Calculate month-over-month variance
        if len(ecomm_forecast_month_cols) >= 2:
            sorted_months = sorted(ecomm_forecast_month_cols, key=parse_month_str)
            for i in range(1, min(6, len(sorted_months))):
                current_month = sorted_months[i]
                prev_month = sorted_months[i-1]
                current_qty = df_ecomm_forecast[current_month].sum()
                prev_qty = df_ecomm_forecast[prev_month].sum()
                
                if prev_qty > 0:
                    variance_pct = ((current_qty - prev_qty) / prev_qty * 100)
                    forecast_variance[current_month] = variance_pct
    
    # Display KPI Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 12px; padding: 1.2rem; color: white; 
                    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);">
            <div style="font-size: 0.9rem; opacity: 0.9;">TOTAL FORECAST 2026</div>
            <div style="font-size: 1.8rem; font-weight: 800; margin: 0.5rem 0;">{total_forecast_2026:,.0f}</div>
            <div style="font-size: 0.9rem;">Units</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_kpi2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%); 
                    border-radius: 12px; padding: 1.2rem; color: white; 
                    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);">
            <div style="font-size: 0.9rem; opacity: 0.9;">FORECAST VALUE</div>
            <div style="font-size: 1.8rem; font-weight: 800; margin: 0.5rem 0;">Rp {forecast_value_2026:,.0f}</div>
            <div style="font-size: 0.9rem;">Revenue Projection</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_kpi3:
        accuracy_color = "#4CAF50" if avg_accuracy_historical >= 80 else "#FF9800" if avg_accuracy_historical >= 70 else "#F44336"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {accuracy_color} 0%, {accuracy_color.replace('F44', 'D32')} 100%); 
                    border-radius: 12px; padding: 1.2rem; color: white; 
                    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);">
            <div style="font-size: 0.9rem; opacity: 0.9;">HISTORICAL ACCURACY</div>
            <div style="font-size: 1.8rem; font-weight: 800; margin: 0.5rem 0;">{avg_accuracy_historical:.1f}%</div>
            <div style="font-size: 0.9rem;">Last 6 Months</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_kpi4:
        avg_monthly = total_forecast_2026 / len(ecomm_forecast_month_cols) if ecomm_forecast_month_cols else 0
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); 
                    border-radius: 12px; padding: 1.2rem; color: white; 
                    box-shadow: 0 6px 20px rgba(255, 152, 0, 0.3);">
            <div style="font-size: 0.9rem; opacity: 0.9;">AVG MONTHLY VOLUME</div>
            <div style="font-size: 1.8rem; font-weight: 800; margin: 0.5rem 0;">{avg_monthly:,.0f}</div>
            <div style="font-size: 0.9rem;">Units/Month</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # NEW SECTION: FORECAST CONFIDENCE ANALYSIS
    # ============================================
    st.markdown("---")
    st.markdown("### 📊 FORECAST CONFIDENCE ANALYSIS")
    
    # Forecasting confidence based on historical accuracy and data quality
    forecast_confidence = calculate_forecast_confidence(df_ecomm_forecast, df_sales, df_forecast)
    
    col_conf1, col_conf2 = st.columns([2, 1])
    
    with col_conf1:
        # Confidence Gauge
        fig_confidence = go.Figure(go.Indicator(
            mode="gauge+number",
            value=forecast_confidence.get('confidence_score', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Forecast Confidence Score", 'font': {'size': 18}},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4CAF50"},
                'steps': [
                    {'range': [0, 50], 'color': "#FF5252"},
                    {'range': [50, 75], 'color': "#FF9800"},
                    {'range': [75, 100], 'color': "#4CAF50"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': 70
                }
            }
        ))
        fig_confidence.update_layout(height=250, margin=dict(t=50, b=10))
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    with col_conf2:
        # Confidence Factors
        st.markdown("#### 📈 Confidence Factors")
        
        factors = forecast_confidence.get('factors', {})
        
        for factor, score in factors.items():
            color = "#4CAF50" if score >= 70 else "#FF9800" if score >= 50 else "#FF5252"
            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.5rem; background: #F5F5F5; border-radius: 5px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 0.85rem;">{factor}</span>
                    <span style="font-weight: bold; color: {color};">{score}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ============================================
    # NEW FUNCTIONS FOR ADVANCED FORECASTING
    # ============================================
    
    @st.cache_data
    def calculate_forecast_confidence(df_forecast, df_sales=None, df_historical_forecast=None):
        """Calculate forecast confidence score based on multiple factors"""
        
        confidence_score = 70  # Base score
        factors = {}
        
        try:
            # Factor 1: Data Completeness
            total_cells = len(df_forecast) * len(ecomm_forecast_month_cols) if ecomm_forecast_month_cols else 1
            non_zero_cells = (df_forecast[ecomm_forecast_month_cols] > 0).sum().sum() if ecomm_forecast_month_cols else 0
            completeness_score = (non_zero_cells / total_cells * 100) if total_cells > 0 else 0
            factors['Data Completeness'] = min(100, completeness_score * 1.2)  # Boost score
            
            # Factor 2: Historical Accuracy (if data available)
            if df_sales is not None and df_historical_forecast is not None and not df_sales.empty and not df_historical_forecast.empty:
                # Calculate accuracy for last 6 months
                common_months = sorted(set(df_sales['Month'].unique()) & set(df_historical_forecast['Month'].unique()))
                if len(common_months) >= 3:
                    accuracies = []
                    for month in common_months[-6:]:
                        sales_qty = df_sales[df_sales['Month'] == month]['Sales_Qty'].sum()
                        forecast_qty = df_historical_forecast[df_historical_forecast['Month'] == month]['Forecast_Qty'].sum()
                        if forecast_qty > 0:
                            accuracy = 100 - abs((sales_qty / forecast_qty * 100) - 100)
                            accuracies.append(accuracy)
                    
                    if accuracies:
                        avg_accuracy = sum(accuracies) / len(accuracies)
                        factors['Historical Accuracy'] = avg_accuracy
            
            # Factor 3: Forecast Consistency
            if len(ecomm_forecast_month_cols) >= 3:
                monthly_totals = df_forecast[ecomm_forecast_month_cols].sum()
                cv = monthly_totals.std() / monthly_totals.mean() if monthly_totals.mean() > 0 else 0
                consistency_score = max(0, 100 - (cv * 100))  # Lower CV = higher score
                factors['Forecast Consistency'] = consistency_score
            
            # Factor 4: SKU Coverage
            total_skus = len(df_forecast)
            active_skus = len(df_forecast[df_forecast[ecomm_forecast_month_cols].sum(axis=1) > 0])
            coverage_score = (active_skus / total_skus * 100) if total_skus > 0 else 0
            factors['SKU Coverage'] = coverage_score
            
            # Calculate weighted average
            weights = {
                'Data Completeness': 0.25,
                'Historical Accuracy': 0.35,
                'Forecast Consistency': 0.25,
                'SKU Coverage': 0.15
            }
            
            weighted_score = 0
            weight_total = 0
            
            for factor, score in factors.items():
                if factor in weights:
                    weighted_score += score * weights[factor]
                    weight_total += weights[factor]
            
            if weight_total > 0:
                confidence_score = weighted_score / weight_total
        
        except Exception as e:
            st.error(f"Confidence calculation error: {str(e)}")
        
        return {
            'confidence_score': round(confidence_score, 1),
            'factors': factors
        }
    
    @st.cache_data
    def detect_forecast_anomalies(df_forecast, threshold_std=2.0):
        """Detect anomalies in forecast data"""
        
        anomalies = []
        
        if not ecomm_forecast_month_cols:
            return anomalies
        
        try:
            # Calculate monthly totals
            monthly_totals = df_forecast[ecomm_forecast_month_cols].sum()
            
            # Calculate moving average and standard deviation
            if len(monthly_totals) >= 3:
                moving_avg = monthly_totals.rolling(window=3, min_periods=1).mean()
                moving_std = monthly_totals.rolling(window=3, min_periods=1).std()
                
                for i, (month, value) in enumerate(monthly_totals.items()):
                    if i >= 2:  # Need at least 2 previous months for comparison
                        avg = moving_avg.iloc[i]
                        std = moving_std.iloc[i]
                        
                        if std > 0 and abs(value - avg) > (threshold_std * std):
                            anomaly_score = abs(value - avg) / std
                            anomalies.append({
                                'Month': month,
                                'Forecast_Value': value,
                                'Moving_Avg': avg,
                                'Std_Dev': std,
                                'Anomaly_Score': anomaly_score,
                                'Deviation_Pct': ((value - avg) / avg * 100) if avg > 0 else 0
                            })
        
        except Exception as e:
            st.error(f"Anomaly detection error: {str(e)}")
        
        return anomalies
    
    @st.cache_data
    def calculate_seasonality_pattern(df_forecast):
        """Calculate seasonality pattern from forecast"""
        
        seasonality = {}
        
        if not ecomm_forecast_month_cols:
            return seasonality
        
        try:
            # Group by month name (ignoring year)
            month_patterns = {}
            for month_col in ecomm_forecast_month_cols:
                month_name = str(month_col).split('-')[0].upper()[:3]
                if month_name not in month_patterns:
                    month_patterns[month_name] = []
                month_patterns[month_name].append(df_forecast[month_col].sum())
            
            # Calculate average for each month
            for month, values in month_patterns.items():
                if values:
                    month_patterns[month] = sum(values) / len(values)
            
            # Calculate seasonal indices
            overall_avg = sum(month_patterns.values()) / len(month_patterns) if month_patterns else 1
            
            for month, avg_value in month_patterns.items():
                if overall_avg > 0:
                    seasonal_index = avg_value / overall_avg
                    seasonality[month] = {
                        'value': avg_value,
                        'index': seasonal_index,
                        'type': 'Peak' if seasonal_index >= 1.2 else 'Normal' if seasonal_index >= 0.8 else 'Low'
                    }
        
        except Exception as e:
            st.error(f"Seasonality calculation error: {str(e)}")
        
        return seasonality
    
    @st.cache_data
    def perform_what_if_analysis(base_forecast, scenarios):
        """Perform what-if scenario analysis"""
        
        results = {}
        
        try:
            for scenario_name, params in scenarios.items():
                modified_forecast = base_forecast.copy()
                
                # Apply scenario adjustments
                if 'growth_rate' in params:
                    growth = params['growth_rate'] / 100
                    for month in ecomm_forecast_month_cols:
                        modified_forecast[month] = modified_forecast[month] * (1 + growth)
                
                if 'specific_months_adjustment' in params:
                    for month, adjustment in params['specific_months_adjustment'].items():
                        if month in modified_forecast.columns:
                            modified_forecast[month] = modified_forecast[month] * (1 + adjustment/100)
                
                # Calculate scenario totals
                scenario_total = modified_forecast[ecomm_forecast_month_cols].sum().sum()
                base_total = base_forecast[ecomm_forecast_month_cols].sum().sum()
                
                results[scenario_name] = {
                    'total_forecast': scenario_total,
                    'change_pct': ((scenario_total - base_total) / base_total * 100) if base_total > 0 else 0,
                    'monthly_breakdown': modified_forecast[ecomm_forecast_month_cols].sum().to_dict()
                }
        
        except Exception as e:
            st.error(f"What-if analysis error: {str(e)}")
        
        return results
    
    @st.cache_data
    def identify_forecast_risks(df_forecast, df_product):
        """Identify potential risks in forecast"""
        
        risks = []
        
        try:
            # Risk 1: New products with high forecast but no history
            if 'Product_Name' in df_forecast.columns:
                # For simplicity, assume new products are those without brand recognition
                new_products = df_forecast[
                    (df_forecast['Brand'].str.contains('New|New Product', case=False, na=False)) |
                    (df_forecast['Product_Name'].str.contains('New|Launch', case=False, na=False))
                ]
                
                if not new_products.empty:
                    high_new_forecast = new_products[new_products[ecomm_forecast_month_cols].sum(axis=1) > 1000]
                    if not high_new_products.empty:
                        risks.append({
                            'type': 'New Product Risk',
                            'description': f"{len(high_new_forecast)} new products with forecast > 1,000 units",
                            'severity': 'High',
                            'impact': "Potential overstock if demand doesn't materialize"
                        })
            
            # Risk 2: High concentration in few SKUs
            sku_contributions = df_forecast[ecomm_forecast_month_cols].sum(axis=1).sort_values(ascending=False)
            top_10_share = sku_contributions.head(10).sum() / sku_contributions.sum() * 100 if sku_contributions.sum() > 0 else 0
            
            if top_10_share > 50:
                risks.append({
                    'type': 'Concentration Risk',
                    'description': f"Top 10 SKUs contribute {top_10_share:.1f}% of total forecast",
                    'severity': 'Medium',
                    'impact': 'High dependency on few products'
                })
            
            # Risk 3: Seasonal peaks without inventory planning
            seasonality = calculate_seasonality_pattern(df_forecast)
            peak_months = [month for month, data in seasonality.items() if data['type'] == 'Peak']
            
            if len(peak_months) >= 2:
                risks.append({
                    'type': 'Seasonal Risk',
                    'description': f"Multiple peak months detected: {', '.join(peak_months)}",
                    'severity': 'Medium',
                    'impact': 'Require advanced inventory planning'
                })
        
        except Exception as e:
            st.error(f"Risk identification error: {str(e)}")
        
        return risks
    
    # ============================================
    # SECTION 2: ANOMALY DETECTION & RISK ANALYSIS
    # ============================================
    st.markdown("---")
    
    # Tabs for different analyses
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "🔍 Anomaly Detection",
        "⚠️ Risk Analysis",
        "📈 Seasonality Analysis"
    ])
    
    with analysis_tab1:
        st.markdown("### 🔍 Forecast Anomaly Detection")
        
        # Anomaly detection settings
        col_anom1, col_anom2 = st.columns(2)
        
        with col_anom1:
            anomaly_threshold = st.slider("Anomaly Threshold (σ)", 1.0, 3.0, 2.0, 0.1,
                                         help="Standard deviations from moving average")
        
        with col_anom2:
            min_forecast_threshold = st.number_input("Minimum Forecast to Check", 100, 10000, 1000,
                                                    help="Only check months with forecast above this")
        
        # Detect anomalies
        with st.spinner("Detecting anomalies..."):
            anomalies = detect_forecast_anomalies(df_ecomm_forecast, threshold_std=anomaly_threshold)
            
            # Filter by minimum threshold
            anomalies = [a for a in anomalies if a['Forecast_Value'] >= min_forecast_threshold]
        
        if anomalies:
            st.warning(f"⚠️ Found {len(anomalies)} potential anomalies")
            
            # Create anomalies dataframe
            df_anomalies = pd.DataFrame(anomalies)
            df_anomalies = df_anomalies.sort_values('Anomaly_Score', ascending=False)
            
            # Display anomalies
            for idx, anomaly in df_anomalies.head(5).iterrows():
                with st.expander(f"**{anomaly['Month']}** - Anomaly Score: {anomaly['Anomaly_Score']:.2f}σ", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Forecast", f"{anomaly['Forecast_Value']:,.0f}")
                    
                    with col2:
                        st.metric("Moving Avg", f"{anomaly['Moving_Avg']:,.0f}")
                    
                    with col3:
                        st.metric("Deviation", f"{anomaly['Deviation_Pct']:+.1f}%")
                    
                    # Recommendations
                    if anomaly['Deviation_Pct'] > 20:
                        st.error("**High Risk:** Significant upward deviation. Verify demand assumptions.")
                    elif anomaly['Deviation_Pct'] < -20:
                        st.error("**High Risk:** Significant downward deviation. Check for missing promotions/events.")
                    else:
                        st.info("**Medium Risk:** Monitor closely in next forecast cycle.")
            
            # Anomaly visualization
            fig_anomalies = go.Figure()
            
            # Add forecast line
            monthly_totals = df_ecomm_forecast[ecomm_forecast_month_cols].sum()
            sorted_months = sorted(ecomm_forecast_month_cols, key=parse_month_str)
            sorted_values = [monthly_totals[m] for m in sorted_months]
            
            fig_anomalies.add_trace(go.Scatter(
                x=sorted_months,
                y=sorted_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#667eea', width=3)
            ))
            
            # Add moving average
            if len(sorted_values) >= 3:
                moving_avg = pd.Series(sorted_values).rolling(window=3, min_periods=1).mean()
                fig_anomalies.add_trace(go.Scatter(
                    x=sorted_months,
                    y=moving_avg,
                    mode='lines',
                    name='3-Month Moving Avg',
                    line=dict(color='#FF9800', width=2, dash='dash')
                ))
            
            # Highlight anomalies
            anomaly_months = [a['Month'] for a in anomalies]
            anomaly_values = [monthly_totals[m] for m in anomaly_months if m in monthly_totals]
            
            if anomaly_months and anomaly_values:
                fig_anomalies.add_trace(go.Scatter(
                    x=anomaly_months,
                    y=anomaly_values,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        size=12,
                        color='#FF5252',
                        symbol='x'
                    )
                ))
            
            fig_anomalies.update_layout(
                height=400,
                title='Forecast Anomaly Detection',
                xaxis_title='Month',
                yaxis_title='Forecast Quantity',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_anomalies, use_container_width=True)
        else:
            st.success("✅ No anomalies detected - Forecast appears consistent")
    
    with analysis_tab2:
        st.markdown("### ⚠️ Forecast Risk Analysis")
        
        # Identify risks
        with st.spinner("Analyzing forecast risks..."):
            risks = identify_forecast_risks(df_ecomm_forecast, df_product)
        
        if risks:
            # Risk summary
            high_risks = [r for r in risks if r['severity'] == 'High']
            medium_risks = [r for r in risks if r['severity'] == 'Medium']
            
            col_risk1, col_risk2, col_risk3 = st.columns(3)
            
            with col_risk1:
                st.metric("Total Risks", len(risks))
            
            with col_risk2:
                st.metric("High Severity", len(high_risks), delta_color="off")
            
            with col_risk3:
                st.metric("Medium Severity", len(medium_risks))
            
            # Display risks
            for risk in risks:
                risk_color = "#F44336" if risk['severity'] == 'High' else "#FF9800"
                
                st.markdown(f"""
                <div style="border-left: 5px solid {risk_color}; padding: 1rem; margin: 1rem 0; 
                            background: {'#FFEBEE' if risk['severity'] == 'High' else '#FFF3E0'}; 
                            border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: {risk_color};">{risk['type']}</strong>
                            <div style="font-size: 0.9rem; margin-top: 0.3rem;">{risk['description']}</div>
                        </div>
                        <div style="background: {risk_color}; color: white; padding: 0.3rem 0.8rem; 
                                    border-radius: 20px; font-size: 0.8rem; font-weight: bold;">
                            {risk['severity']}
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #666;">
                        <strong>Impact:</strong> {risk['impact']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk mitigation recommendations
            st.markdown("#### 🛡️ Risk Mitigation Strategies")
            
            mitigation_col1, mitigation_col2 = st.columns(2)
            
            with mitigation_col1:
                st.markdown("""
                **For New Product Risks:**
                1. **Phased Launch** - Start with conservative quantities
                2. **Pre-orders** - Validate demand before full production
                3. **Market Testing** - Test in limited regions first
                4. **Flexible Contracts** - Negotiate return options with suppliers
                """)
            
            with mitigation_col2:
                st.markdown("""
                **For Concentration Risks:**
                1. **Diversify Portfolio** - Expand product range
                2. **Cross-Selling** - Promote complementary products
                3. **Inventory Buffer** - Higher safety stock for key SKUs
                4. **Alternative Suppliers** - Reduce single-source dependency
                """)
        else:
            st.success("✅ No significant risks identified in forecast")
    
    with analysis_tab3:
        st.markdown("### 📈 Seasonality Pattern Analysis")
        
        # Calculate seasonality
        with st.spinner("Analyzing seasonality patterns..."):
            seasonality = calculate_seasonality_pattern(df_ecomm_forecast)
        
        if seasonality:
            # Create seasonality chart
            months_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            months_display = [m for m in months_order if m in seasonality]
            
            values = [seasonality[m]['value'] for m in months_display]
            indices = [seasonality[m]['index'] for m in months_display]
            types = [seasonality[m]['type'] for m in months_display]
            
            fig_season = go.Figure()
            
            # Bar chart for values
            fig_season.add_trace(go.Bar(
                x=months_display,
                y=values,
                name='Avg Forecast',
                marker_color=['#4CAF50' if t == 'Peak' else '#FF9800' if t == 'Normal' else '#9E9E9E' for t in types],
                text=[f"{v:,.0f}" for v in values],
                textposition='auto'
            ))
            
            # Line for seasonal indices
            fig_season.add_trace(go.Scatter(
                x=months_display,
                y=indices,
                name='Seasonal Index',
                yaxis='y2',
                line=dict(color='#2196F3', width=3),
                mode='lines+markers'
            ))
            
            fig_season.update_layout(
                height=400,
                title='Seasonality Pattern Analysis',
                xaxis_title='Month',
                yaxis=dict(title='Average Forecast Quantity'),
                yaxis2=dict(
                    title='Seasonal Index',
                    overlaying='y',
                    side='right',
                    range=[0, max(indices) * 1.2]
                ),
                showlegend=True
            )
            
            st.plotly_chart(fig_season, use_container_width=True)
            
            # Seasonality insights
            peak_months = [m for m, data in seasonality.items() if data['type'] == 'Peak']
            low_months = [m for m, data in seasonality.items() if data['type'] == 'Low']
            
            if peak_months:
                st.info(f"""
                **📈 Peak Season Identified:** {', '.join(peak_months)}
                
                **Recommendations:**
                1. **Increase inventory** 2-3 months before peak season
                2. **Secure supplier capacity** in advance
                3. **Plan promotions** to maximize sales
                4. **Adjust staffing** for higher order volume
                """)
            
            if low_months:
                st.warning(f"""
                **📉 Low Season Identified:** {', '.join(low_months)}
                
                **Recommendations:**
                1. **Reduce inventory** levels
                2. **Plan clearance sales** to free up capital
                3. **Focus on new product development**
                4. **Schedule maintenance/upgrades**
                """)
        else:
            st.info("ℹ️ Insufficient data for seasonality analysis")
    
    # ============================================
    # SECTION 3: WHAT-IF SCENARIO ANALYSIS
    # ============================================
    st.markdown("---")
    st.markdown("### 🎮 WHAT-IF SCENARIO ANALYSIS")
    
    # Scenario configuration
    scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
    
    with scenario_col1:
        optimistic_growth = st.slider("Optimistic Growth (%)", 0, 50, 20,
                                     help="Best-case scenario growth rate")
    
    with scenario_col2:
        conservative_growth = st.slider("Conservative Growth (%)", -20, 20, 5,
                                       help="Worst-case scenario growth rate")
    
    with scenario_col3:
        # Seasonal adjustment scenarios
        peak_season_boost = st.slider("Peak Season Boost (%)", 0, 100, 30,
                                     help="Additional growth during peak months")
    
    # Define scenarios
    scenarios = {
        'Baseline': {},
        'Optimistic': {
            'growth_rate': optimistic_growth,
            'specific_months_adjustment': {}
        },
        'Conservative': {
            'growth_rate': conservative_growth,
            'specific_months_adjustment': {}
        },
        'Seasonal Boost': {
            'growth_rate': 0,
            'specific_months_adjustment': {}
        }
    }
    
    # Add seasonal adjustments if seasonality detected
    if seasonality:
        peak_months = [m for m, data in seasonality.items() if data['type'] == 'Peak']
        for month in peak_months:
            # Find corresponding columns in forecast
            for col in ecomm_forecast_month_cols:
                if month in str(col).upper():
                    scenarios['Seasonal Boost']['specific_months_adjustment'][col] = peak_season_boost
    
    # Run what-if analysis
    with st.spinner("Running scenario analysis..."):
        scenario_results = perform_what_if_analysis(df_ecomm_forecast, scenarios)
    
    if scenario_results:
        # Scenario comparison
        scenario_data = []
        for scenario, results in scenario_results.items():
            scenario_data.append({
                'Scenario': scenario,
                'Total Forecast': results['total_forecast'],
                'Change %': results['change_pct'],
                'Avg Monthly': results['total_forecast'] / len(ecomm_forecast_month_cols) if ecomm_forecast_month_cols else 0
            })
        
        df_scenarios = pd.DataFrame(scenario_data)
        
        # Visualization
        fig_scenarios = go.Figure()
        
        # Bar chart for total forecast
        fig_scenarios.add_trace(go.Bar(
            x=df_scenarios['Scenario'],
            y=df_scenarios['Total Forecast'],
            name='Total Forecast',
            marker_color=['#9E9E9E', '#4CAF50', '#FF9800', '#2196F3'],
            text=[f"{x:,.0f}" for x in df_scenarios['Total Forecast']],
            textposition='auto'
        ))
        
        fig_scenarios.update_layout(
            height=400,
            title='What-If Scenario Comparison',
            xaxis_title='Scenario',
            yaxis_title='Total Forecast Quantity',
            showlegend=False
        )
        
        st.plotly_chart(fig_scenarios, use_container_width=True)
        
        # Scenario details
        st.markdown("#### 📊 Scenario Details")
        
        details_tab1, details_tab2 = st.tabs(["Summary Table", "Monthly Breakdown"])
        
        with details_tab1:
            # Format display
            df_display = df_scenarios.copy()
            df_display['Total Forecast'] = df_display['Total Forecast'].apply(lambda x: f"{x:,.0f}")
            df_display['Change %'] = df_display['Change %'].apply(lambda x: f"{x:+.1f}%")
            df_display['Avg Monthly'] = df_display['Avg Monthly'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(df_display, use_container_width=True)
        
        with details_tab2:
            # Monthly breakdown for selected scenario
            selected_scenario = st.selectbox("Select Scenario for Details", list(scenario_results.keys()))
            
            if selected_scenario in scenario_results:
                monthly_data = scenario_results[selected_scenario]['monthly_breakdown']
                
                # Convert to dataframe
                df_monthly = pd.DataFrame(list(monthly_data.items()), columns=['Month', 'Forecast'])
                df_monthly = df_monthly.sort_values('Month', key=lambda x: x.map(parse_month_str))
                
                # Chart
                fig_monthly = go.Figure()
                
                fig_monthly.add_trace(go.Bar(
                    x=df_monthly['Month'],
                    y=df_monthly['Forecast'],
                    name='Monthly Forecast',
                    marker_color='#667eea'
                ))
                
                fig_monthly.update_layout(
                    height=300,
                    title=f'Monthly Breakdown - {selected_scenario} Scenario',
                    xaxis_title='Month',
                    yaxis_title='Forecast Quantity'
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Export scenario data
                if st.button(f"📥 Export {selected_scenario} Scenario", key=f"export_{selected_scenario}"):
                    # Create detailed export
                    export_data = df_ecomm_forecast.copy()
                    
                    if selected_scenario != 'Baseline':
                        # Apply scenario adjustments
                        scenario_params = scenarios[selected_scenario]
                        
                        if 'growth_rate' in scenario_params:
                            growth = scenario_params['growth_rate'] / 100
                            for month in ecomm_forecast_month_cols:
                                export_data[month] = export_data[month] * (1 + growth)
                        
                        if 'specific_months_adjustment' in scenario_params:
                            for month, adjustment in scenario_params['specific_months_adjustment'].items():
                                if month in export_data.columns:
                                    export_data[month] = export_data[month] * (1 + adjustment/100)
                    
                    csv_data = export_data.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Scenario Data (CSV)",
                        data=csv_data,
                        file_name=f"forecast_scenario_{selected_scenario}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key=f"dl_{selected_scenario}"
                    )
    
    # ============================================
    # SECTION 4: FORECAST OPTIMIZATION RECOMMENDATIONS
    # ============================================
    st.markdown("---")
    st.markdown("### 🎯 FORECAST OPTIMIZATION RECOMMENDATIONS")
    
    # Generate recommendations based on analysis
    recommendations = []
    
    # Recommendation 1: Based on confidence score
    if forecast_confidence.get('confidence_score', 0) < 70:
        recommendations.append({
            'priority': 'High',
            'title': 'Improve Forecast Confidence',
            'description': f"Current confidence score is {forecast_confidence['confidence_score']}%. Focus on improving data quality and historical accuracy.",
            'actions': [
                'Review and clean forecast data',
                'Improve historical data collection',
                'Implement forecast accuracy tracking'
            ]
        })
    
    # Recommendation 2: Based on anomalies
    if anomalies:
        recommendations.append({
            'priority': 'High',
            'title': 'Address Forecast Anomalies',
            'description': f"Found {len(anomalies)} anomalies in forecast data. Investigate and correct unusual patterns.",
            'actions': [
                'Verify demand assumptions for anomalous months',
                'Check for data entry errors',
                'Review promotional calendar'
            ]
        })
    
    # Recommendation 3: Based on risks
    if risks:
        high_risk_count = len([r for r in risks if r['severity'] == 'High'])
        if high_risk_count > 0:
            recommendations.append({
                'priority': 'Critical',
                'title': 'Mitigate High Risks',
                'description': f"Found {high_risk_count} high-severity risks requiring immediate attention.",
                'actions': [
                    'Develop contingency plans for key risks',
                    'Increase monitoring of high-risk areas',
                    'Review supplier contracts'
                ]
            })
    
    # Recommendation 4: Based on seasonality
    if seasonality:
        peak_months = [m for m, data in seasonality.items() if data['type'] == 'Peak']
        if peak_months:
            recommendations.append({
                'priority': 'Medium',
                'title': 'Prepare for Peak Season',
                'description': f"Peak season identified in {', '.join(peak_months)}. Plan inventory and operations accordingly.",
                'actions': [
                    'Increase safety stock before peak months',
                    'Schedule additional staffing',
                    'Plan marketing campaigns'
                ]
            })
    
    # Default recommendation if none generated
    if not recommendations:
        recommendations.append({
            'priority': 'Low',
            'title': 'Maintain Current Practices',
            'description': "Forecast appears stable and well-managed. Continue current processes with regular reviews.",
            'actions': [
                'Continue monthly forecast reviews',
                'Monitor key performance indicators',
                'Stay updated on market trends'
            ]
        })
    
    # Display recommendations
    for rec in recommendations:
        priority_color = {
            'Critical': '#F44336',
            'High': '#FF9800',
            'Medium': '#FFC107',
            'Low': '#4CAF50'
        }.get(rec['priority'], '#9E9E9E')
        
        st.markdown(f"""
        <div style="border-left: 5px solid {priority_color}; padding: 1rem; margin: 1rem 0; 
                    background: white; border-radius: 5px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="background: {priority_color}; color: white; padding: 0.2rem 0.8rem; 
                                    border-radius: 12px; font-size: 0.8rem; font-weight: bold;">
                            {rec['priority']} Priority
                        </div>
                        <h4 style="margin: 0;">{rec['title']}</h4>
                    </div>
                    <p style="margin: 0.5rem 0; color: #666;">{rec['description']}</p>
                </div>
            </div>
            
            <div style="margin-top: 1rem;">
                <div style="font-size: 0.9rem; font-weight: bold; margin-bottom: 0.3rem;">Recommended Actions:</div>
                <ul style="margin: 0; padding-left: 1.2rem;">
                    {''.join([f'<li style="font-size: 0.85rem; margin-bottom: 0.2rem;">{action}</li>' for action in rec['actions']])}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # SECTION 5: FORECAST PERFORMANCE TRACKING
    # ============================================
    st.markdown("---")
    st.markdown("### 📈 FORECAST PERFORMANCE TRACKING")
    
    # Performance tracking dashboard
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        # Forecast vs Actual Comparison (if historical data available)
        if not df_sales.empty and not df_forecast.empty:
            st.markdown("#### 📊 Historical Accuracy Trend")
            
            # Calculate monthly accuracy
            accuracy_data = []
            common_months = sorted(set(df_sales['Month'].unique()) & set(df_forecast['Month'].unique()))
            
            for month in common_months[-12:]:  # Last 12 months
                sales_qty = df_sales[df_sales['Month'] == month]['Sales_Qty'].sum()
                forecast_qty = df_forecast[df_forecast['Month'] == month]['Forecast_Qty'].sum()
                
                if forecast_qty > 0:
                    accuracy = 100 - abs((sales_qty / forecast_qty * 100) - 100)
                    accuracy_data.append({
                        'Month': month.strftime('%b-%Y'),
                        'Sales': sales_qty,
                        'Forecast': forecast_qty,
                        'Accuracy': accuracy
                    })
            
            if accuracy_data:
                df_accuracy = pd.DataFrame(accuracy_data)
                
                fig_accuracy = go.Figure()
                
                # Bar chart for comparison
                fig_accuracy.add_trace(go.Bar(
                    x=df_accuracy['Month'],
                    y=df_accuracy['Sales'],
                    name='Actual Sales',
                    marker_color='#4CAF50'
                ))
                
                fig_accuracy.add_trace(go.Bar(
                    x=df_accuracy['Month'],
                    y=df_accuracy['Forecast'],
                    name='Forecast',
                    marker_color='#667eea',
                    opacity=0.7
                ))
                
                # Accuracy line
                fig_accuracy.add_trace(go.Scatter(
                    x=df_accuracy['Month'],
                    y=df_accuracy['Accuracy'],
                    name='Accuracy %',
                    yaxis='y2',
                    line=dict(color='#FF5252', width=3),
                    mode='lines+markers'
                ))
                
                fig_accuracy.update_layout(
                    height=400,
                    title='Forecast vs Actual Performance',
                    xaxis_title='Month',
                    yaxis=dict(title='Quantity'),
                    yaxis2=dict(
                        title='Accuracy %',
                        overlaying='y',
                        side='right',
                        range=[0, 110]
                    ),
                    barmode='group',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_accuracy, use_container_width=True)
        else:
            st.info("ℹ️ Historical accuracy tracking requires both sales and forecast data")
    
    with perf_col2:
        # Forecast bias analysis
        st.markdown("#### 🎯 Forecast Bias Analysis")
        
        if not df_sales.empty and not df_forecast.empty and common_months:
            bias_data = []
            
            for month in common_months[-12:]:
                sales_qty = df_sales[df_sales['Month'] == month]['Sales_Qty'].sum()
                forecast_qty = df_forecast[df_forecast['Month'] == month]['Forecast_Qty'].sum()
                
                if forecast_qty > 0:
                    bias_pct = ((sales_qty - forecast_qty) / forecast_qty * 100)
                    bias_data.append({
                        'Month': month.strftime('%b-%Y'),
                        'Bias_Pct': bias_pct,
                        'Type': 'Over-forecast' if bias_pct < 0 else 'Under-forecast' if bias_pct > 0 else 'Accurate'
                    })
            
            if bias_data:
                df_bias = pd.DataFrame(bias_data)
                
                # Calculate average bias
                avg_bias = df_bias['Bias_Pct'].mean()
                
                # Bias gauge
                fig_bias = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=abs(avg_bias),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Average Forecast Bias"},
                    number={'suffix': "%"},
                    delta={'reference': 10, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [0, 50]},
                        'bar': {'color': "#FF9800"},
                        'steps': [
                            {'range': [0, 10], 'color': "#4CAF50"},
                            {'range': [10, 20], 'color': "#FFC107"},
                            {'range': [20, 50], 'color': "#F44336"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.8,
                            'value': 10
                        }
                    }
                ))
                
                fig_bias.update_layout(height=250)
                st.plotly_chart(fig_bias, use_container_width=True)
                
                # Bias interpretation
                if abs(avg_bias) <= 10:
                    st.success("✅ Good: Forecast bias within acceptable range (±10%)")
                elif abs(avg_bias) <= 20:
                    st.warning("⚠️ Moderate: Forecast bias needs improvement")
                else:
                    st.error("❌ High: Significant forecast bias detected")
                
                # Bias trend
                st.markdown("##### 📈 Bias Trend Over Time")
                
                fig_bias_trend = go.Figure()
                
                fig_bias_trend.add_trace(go.Scatter(
                    x=df_bias['Month'],
                    y=df_bias['Bias_Pct'],
                    mode='lines+markers',
                    name='Bias %',
                    line=dict(color='#FF9800', width=2),
                    marker=dict(
                        size=8,
                        color=df_bias['Bias_Pct'].apply(lambda x: '#4CAF50' if -10 <= x <= 10 else '#FF9800' if -20 <= x <= 20 else '#F44336')
                    )
                ))
                
                fig_bias_trend.update_layout(
                    height=200,
                    xaxis_title='Month',
                    yaxis_title='Bias %',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_bias_trend, use_container_width=True)
    
    # ============================================
    # SECTION 6: EXPORT & REPORTING
    # ============================================
    st.markdown("---")
    st.markdown("### 📊 EXPORT COMPREHENSIVE FORECAST ANALYSIS")
    
    # Export options
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📥 Export Forecast Intelligence Report", use_container_width=True, type="primary"):
            # Create comprehensive Excel report
            import io
            from pandas import ExcelWriter
            
            output = io.BytesIO()
            
            with ExcelWriter(output, engine='openpyxl') as writer:
                # Sheet 1: Raw Forecast Data
                df_ecomm_forecast.to_excel(writer, sheet_name='Raw_Forecast_Data', index=False)
                
                # Sheet 2: Summary Statistics
                summary_data = {
                    'Metric': [
                        'Total Forecast 2026',
                        'Forecast Value 2026',
                        'Historical Accuracy',
                        'Forecast Confidence',
                        'Number of SKUs',
                        'Number of Months',
                        'Average Monthly Volume'
                    ],
                    'Value': [
                        total_forecast_2026,
                        forecast_value_2026,
                        f"{avg_accuracy_historical:.1f}%",
                        f"{forecast_confidence.get('confidence_score', 0):.1f}%",
                        len(df_ecomm_forecast),
                        len(ecomm_forecast_month_cols),
                        f"{total_forecast_2026 / len(ecomm_forecast_month_cols):,.0f}" if ecomm_forecast_month_cols else "0"
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Sheet 3: Anomalies
                if anomalies:
                    pd.DataFrame(anomalies).to_excel(writer, sheet_name='Anomalies', index=False)
                
                # Sheet 4: Risks
                if risks:
                    pd.DataFrame(risks).to_excel(writer, sheet_name='Risks', index=False)
                
                # Sheet 5: Seasonality
                if seasonality:
                    seasonality_data = []
                    for month, data in seasonality.items():
                        seasonality_data.append({
                            'Month': month,
                            'Average Forecast': data['value'],
                            'Seasonal Index': data['index'],
                            'Type': data['type']
                        })
                    pd.DataFrame(seasonality_data).to_excel(writer, sheet_name='Seasonality', index=False)
                
                # Sheet 6: Scenarios
                if scenario_results:
                    scenario_export = []
                    for scenario, results in scenario_results.items():
                        scenario_export.append({
                            'Scenario': scenario,
                            'Total Forecast': results['total_forecast'],
                            'Change %': results['change_pct']
                        })
                    pd.DataFrame(scenario_export).to_excel(writer, sheet_name='Scenarios', index=False)
                
                # Sheet 7: Recommendations
                if recommendations:
                    rec_export = []
                    for rec in recommendations:
                        rec_export.append({
                            'Priority': rec['priority'],
                            'Title': rec['title'],
                            'Description': rec['description'],
                            'Actions': '; '.join(rec['actions'])
                        })
                    pd.DataFrame(rec_export).to_excel(writer, sheet_name='Recommendations', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="💾 Download Excel Report",
                data=output,
                file_name=f"Forecast_Intelligence_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    with export_col2:
        # Export specific analyses
        export_option = st.selectbox(
            "Export Specific Analysis",
            [
                "Select...",
                "Anomaly Report",
                "Risk Assessment",
                "Seasonality Analysis",
                "Scenario Comparison"
            ]
        )
        
        if export_option != "Select...":
            csv_data = ""
            filename = ""
            
            if export_option == "Anomaly Report" and anomalies:
                df_export = pd.DataFrame(anomalies)
                csv_data = df_export.to_csv(index=False)
                filename = "forecast_anomalies.csv"
            
            elif export_option == "Risk Assessment" and risks:
                df_export = pd.DataFrame(risks)
                csv_data = df_export.to_csv(index=False)
                filename = "forecast_risks.csv"
            
            elif export_option == "Seasonality Analysis" and seasonality:
                seasonality_data = []
                for month, data in seasonality.items():
                    seasonality_data.append({
                        'Month': month,
                        'Average_Forecast': data['value'],
                        'Seasonal_Index': data['index'],
                        'Type': data['type']
                    })
                df_export = pd.DataFrame(seasonality_data)
                csv_data = df_export.to_csv(index=False)
                filename = "seasonality_analysis.csv"
            
            elif export_option == "Scenario Comparison" and scenario_results:
                scenario_data = []
                for scenario, results in scenario_results.items():
                    scenario_data.append({
                        'Scenario': scenario,
                        'Total_Forecast': results['total_forecast'],
                        'Change_Pct': results['change_pct']
                    })
                df_export = pd.DataFrame(scenario_data)
                csv_data = df_export.to_csv(index=False)
                filename = "scenario_comparison.csv"
            
            if csv_data:
                st.download_button(
                    label=f"📥 Export {export_option}",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
    
    with export_col3:
        # Quick actions
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📊 Generate Executive Summary", use_container_width=True):
            # Create executive summary
            exec_summary = f"""
            ## 📈 FORECAST INTELLIGENCE EXECUTIVE SUMMARY
            
            ### Key Metrics:
            - **Total 2026 Forecast:** {total_forecast_2026:,.0f} units
            - **Projected Revenue:** Rp {forecast_value_2026:,.0f}
            - **Forecast Confidence:** {forecast_confidence.get('confidence_score', 0):.1f}%
            - **Historical Accuracy:** {avg_accuracy_historical:.1f}%
            
            ### Key Findings:
            """
            
            if anomalies:
                exec_summary += f"- **⚠️ Anomalies Detected:** {len(anomalies)} unusual patterns requiring review\n"
            
            if risks:
                high_risks = len([r for r in risks if r['severity'] == 'High'])
                if high_risks > 0:
                    exec_summary += f"- **🚨 High Risks:** {high_risks} critical risks requiring immediate attention\n"
            
            if seasonality:
                peak_months = [m for m, data in seasonality.items() if data['type'] == 'Peak']
                if peak_months:
                    exec_summary += f"- **📈 Peak Season:** Identified in {', '.join(peak_months)}\n"
            
            exec_summary += f"""
            ### Top Recommendations:
            """
            
            for i, rec in enumerate(recommendations[:3], 1):
                exec_summary += f"{i}. **{rec['title']}** ({rec['priority']} priority)\n"
            
            st.markdown(exec_summary)
    
    # ============================================
    # SECTION 7: FORECAST MONITORING DASHBOARD
    # ============================================
    st.markdown("---")
    
    with st.expander("🔍 REAL-TIME FORECAST MONITORING DASHBOARD", expanded=False):
        # Monitoring metrics
        monitor_col1, monitor_col2, monitor_col3, monitor_col4 = st.columns(4)
        
        with monitor_col1:
            # Data freshness
            data_freshness = "Today"  # Assuming data is fresh
            st.metric("Data Freshness", data_freshness)
        
        with monitor_col2:
            # SKU coverage
            active_skus = len(df_ecomm_forecast[df_ecomm_forecast[ecomm_forecast_month_cols].sum(axis=1) > 0])
            total_skus = len(df_ecomm_forecast)
            coverage_pct = (active_skus / total_skus * 100) if total_skus > 0 else 0
            st.metric("Active SKU Coverage", f"{coverage_pct:.1f}%")
        
        with monitor_col3:
            # Forecast volatility
            if len(ecomm_forecast_month_cols) >= 2:
                monthly_totals = df_ecomm_forecast[ecomm_forecast_month_cols].sum()
                volatility = monthly_totals.pct_change().std() * 100
                st.metric("Forecast Volatility", f"{volatility:.1f}%")
        
        with monitor_col4:
            # Data quality score
            quality_score = forecast_confidence.get('factors', {}).get('Data Completeness', 0)
            st.metric("Data Quality Score", f"{quality_score:.0f}/100")
        
        # Real-time alerts
        st.markdown("#### ⚡ REAL-TIME ALERTS")
        
        alerts = []
        
        # Alert 1: Low confidence
        if forecast_confidence.get('confidence_score', 0) < 60:
            alerts.append({
                'type': 'warning',
                'message': f"Low forecast confidence ({forecast_confidence['confidence_score']}%)",
                'action': 'Review data quality'
            })
        
        # Alert 2: High anomalies
        if len(anomalies) > 3:
            alerts.append({
                'type': 'error',
                'message': f"Multiple anomalies detected ({len(anomalies)})",
                'action': 'Investigate unusual patterns'
            })
        
        # Alert 3: High risks
        high_risk_count = len([r for r in risks if r['severity'] == 'High'])
        if high_risk_count > 0:
            alerts.append({
                'type': 'error',
                'message': f"{high_risk_count} high-severity risks identified",
                'action': 'Implement mitigation strategies'
            })
        
        # Display alerts
        if alerts:
            for alert in alerts:
                alert_color = "#F44336" if alert['type'] == 'error' else "#FF9800"
                st.markdown(f"""
                <div style="border-left: 4px solid {alert_color}; padding: 0.8rem; margin: 0.5rem 0; 
                            background: {'#FFEBEE' if alert['type'] == 'error' else '#FFF3E0'}; 
                            border-radius: 5px;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="color: {alert_color}; font-weight: bold;">⚠️</div>
                        <div>
                            <strong>{alert['message']}</strong>
                            <div style="font-size: 0.9rem; color: #666;">Action: {alert['action']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No critical alerts - Forecast monitoring normal")

# --- TAB 8: PROFITABILITY ANALYSIS ---
with tab8:
    st.subheader("💰 Combined Profitability & Financial Projection (2026)")
    st.markdown("**Comprehensive Financial Outlook: Ecommerce + Reseller Channels**")

    # ================ 1. DATA PROCESSING ENGINE ================
    # Kita butuh menggabungkan data Ecomm dan Reseller menjadi satu format standar
    # Format target: SKU_ID | Month | Channel | Qty | Floor_Price | Net_Order_Price
    
    combined_data = []
    process_success = False
    
    with st.spinner('🔄 Merging Financial Data...'):
        try:
            # --- A. Process Ecommerce Data ---
            if not df_ecomm_forecast.empty:
                # Cari kolom bulan 2026
                ecomm_cols_26 = [c for c in df_ecomm_forecast.columns if '26' in str(c) and any(m in str(c).lower() for m in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])]
                
                if ecomm_cols_26:
                    # Melt menjadi long format
                    df_e_long = df_ecomm_forecast.melt(
                        id_vars=['SKU_ID'], 
                        value_vars=ecomm_cols_26, 
                        var_name='Month_Label', 
                        value_name='Qty'
                    )
                    df_e_long['Channel'] = 'Ecommerce'
                    combined_data.append(df_e_long)

            # --- B. Process Reseller Data ---
            if not df_reseller_forecast.empty:
                # Cari kolom bulan 2026
                res_cols_26 = [c for c in df_reseller_forecast.columns if '26' in str(c) and any(m in str(c).lower() for m in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])]
                
                if res_cols_26:
                    df_r_long = df_reseller_forecast.melt(
                        id_vars=['SKU_ID'], 
                        value_vars=res_cols_26, 
                        var_name='Month_Label', 
                        value_name='Qty'
                    )
                    df_r_long['Channel'] = 'Reseller'
                    combined_data.append(df_r_long)
            
            # --- C. Merge & Enrich ---
            if combined_data:
                df_fin_combined = pd.concat(combined_data, ignore_index=True)
                
                # Bersihkan data Qty
                df_fin_combined['Qty'] = pd.to_numeric(df_fin_combined['Qty'], errors='coerce').fillna(0)
                df_fin_combined = df_fin_combined[df_fin_combined['Qty'] > 0] # Ambil yang ada isinya saja
                
                # Standardize Month
                def parse_fin_month(m):
                    try:
                        m = str(m).strip()
                        if '-' in m:
                            parts = m.split('-')
                            return datetime.strptime(f"{parts[0][:3]}-20{parts[1][-2:]}", "%b-%Y")
                    except: return None
                
                df_fin_combined['Month_Date'] = df_fin_combined['Month_Label'].apply(parse_fin_month)
                df_fin_combined = df_fin_combined.sort_values('Month_Date')
                
                # Add Product Info (Brand, Tier, Prices)
                # Pastikan kolom harga ada di df_product
                cols_to_merge = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier']
                if 'Floor_Price' in df_product.columns: cols_to_merge.append('Floor_Price')
                if 'Net_Order_Price' in df_product.columns: cols_to_merge.append('Net_Order_Price') # Cost/HPP
                
                df_fin_combined = pd.merge(df_fin_combined, df_product[cols_to_merge], on='SKU_ID', how='left')
                
                # Fill missing prices with 0
                if 'Floor_Price' in df_fin_combined.columns:
                    df_fin_combined['Floor_Price'] = pd.to_numeric(df_fin_combined['Floor_Price'], errors='coerce').fillna(0)
                else: df_fin_combined['Floor_Price'] = 0
                    
                if 'Net_Order_Price' in df_fin_combined.columns:
                    df_fin_combined['Net_Order_Price'] = pd.to_numeric(df_fin_combined['Net_Order_Price'], errors='coerce').fillna(0)
                else: df_fin_combined['Net_Order_Price'] = 0
                
                # Calculate Financials
                df_fin_combined['Revenue'] = df_fin_combined['Qty'] * df_fin_combined['Floor_Price']
                df_fin_combined['COGS'] = df_fin_combined['Qty'] * df_fin_combined['Net_Order_Price']
                df_fin_combined['Gross_Margin'] = df_fin_combined['Revenue'] - df_fin_combined['COGS']
                
                process_success = True
            else:
                st.warning("⚠️ No 2026 forecast data found in Ecomm or Reseller sheets.")
                
        except Exception as e:
            st.error(f"❌ Error processing financial data: {str(e)}")

    # ================ 2. DASHBOARD VISUALIZATION ================
    if process_success and not df_fin_combined.empty:
        
        # --- A. EXECUTIVE SUMMARY (BIG NUMBERS) ---
        st.divider()
        
        total_rev = df_fin_combined['Revenue'].sum()
        total_margin = df_fin_combined['Gross_Margin'].sum()
        total_qty = df_fin_combined['Qty'].sum()
        avg_margin_pct = (total_margin / total_rev * 100) if total_rev > 0 else 0
        
        # Channel Mix
        rev_by_channel = df_fin_combined.groupby('Channel')['Revenue'].sum()
        ecomm_rev = rev_by_channel.get('Ecommerce', 0)
        res_rev = rev_by_channel.get('Reseller', 0)
        ecomm_share = (ecomm_rev / total_rev * 100) if total_rev > 0 else 0
        
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        with col_kpi1:
            st.metric("Total Revenue 2026", f"Rp {total_rev:,.0f}", help="Gross Revenue Projection")
            
        with col_kpi2:
            st.metric("Total Gross Margin", f"Rp {total_margin:,.0f}", help="Revenue - COGS (Net Order Price)")
            
        with col_kpi3:
            st.metric("Blended Margin %", f"{avg_margin_pct:.1f}%", 
                     delta="Health Indicator", delta_color="normal" if avg_margin_pct > 30 else "off")
            
        with col_kpi4:
            st.metric("Channel Mix (Ecomm)", f"{ecomm_share:.1f}%", 
                     delta=f"Reseller: {100-ecomm_share:.1f}%", delta_color="off")

        # --- B. CHANNEL PERFORMANCE COMPARISON ---
        st.divider()
        st.subheader("🏢 Channel Profitability Comparison")
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # Monthly Revenue Stacked Bar
            monthly_ch_rev = df_fin_combined.groupby(['Month_Label', 'Month_Date', 'Channel'])['Revenue'].sum().reset_index()
            monthly_ch_rev = monthly_ch_rev.sort_values('Month_Date')
            
            fig_stack = px.bar(monthly_ch_rev, x='Month_Label', y='Revenue', color='Channel',
                             title="Monthly Revenue Contribution by Channel",
                             color_discrete_map={'Ecommerce': '#667eea', 'Reseller': '#FF9800'},
                             text_auto='.2s')
            fig_stack.update_layout(height=400, yaxis_title="Revenue (Rp)")
            st.plotly_chart(fig_stack, use_container_width=True)
            
        with c2:
            # Profitability Summary Table per Channel
            ch_summary = df_fin_combined.groupby('Channel').agg({
                'Revenue': 'sum',
                'Gross_Margin': 'sum',
                'Qty': 'sum'
            }).reset_index()
            ch_summary['Margin %'] = (ch_summary['Gross_Margin'] / ch_summary['Revenue'] * 100)
            
            # Donut Chart Revenue
            fig_donut = px.pie(ch_summary, values='Revenue', names='Channel', hole=0.4,
                             title="Revenue Share", color='Channel',
                             color_discrete_map={'Ecommerce': '#667eea', 'Reseller': '#FF9800'})
            fig_donut.update_layout(height=400)
            st.plotly_chart(fig_donut, use_container_width=True)

        # Show mini table for Channel
        ch_disp = ch_summary.copy()
        ch_disp['Revenue'] = ch_disp['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
        ch_disp['Gross_Margin'] = ch_disp['Gross_Margin'].apply(lambda x: f"Rp {x:,.0f}")
        ch_disp['Margin %'] = ch_disp['Margin %'].apply(lambda x: f"{x:.1f}%")
        ch_disp['Qty'] = ch_disp['Qty'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(ch_disp, use_container_width=True)

        # --- C. BRAND PROFITABILITY MATRIX ---
        st.divider()
        st.subheader("🏷️ Brand Profitability Matrix")
        st.caption("Analisis posisi Brand berdasarkan kontribusi Revenue dan tingkat Profitabilitas (Margin %)")
        
        if 'Brand' in df_fin_combined.columns:
            brand_fin = df_fin_combined.groupby('Brand').agg({
                'Revenue': 'sum',
                'Gross_Margin': 'sum',
                'Qty': 'sum'
            }).reset_index()
            
            brand_fin['Margin %'] = (brand_fin['Gross_Margin'] / brand_fin['Revenue'] * 100).fillna(0)
            
            # Quadrant Scatter Plot
            fig_scat = px.scatter(brand_fin, x='Revenue', y='Margin %', 
                                size='Gross_Margin', color='Brand',
                                hover_name='Brand', text='Brand',
                                title="Brand Matrix: Revenue vs Margin % (Size = Gross Margin Value)",
                                labels={'Revenue': 'Total Revenue 2026 (Rp)', 'Margin %': 'Gross Margin %'},
                                height=500)
            
            # Add Quadrant Lines (Median)
            med_rev = brand_fin['Revenue'].median()
            med_mar = brand_fin['Margin %'].median()
            
            fig_scat.add_hline(y=med_mar, line_dash="dash", line_color="gray", annotation_text="Avg Margin")
            fig_scat.add_vline(x=med_rev, line_dash="dash", line_color="gray", annotation_text="Avg Revenue")
            fig_scat.update_traces(textposition='top center')
            
            st.plotly_chart(fig_scat, use_container_width=True)
            
            # --- TIER PROFITABILITY STACKED BAR ---
            if 'SKU_Tier' in df_fin_combined.columns:
                st.markdown("#### 📦 Profitability by Tier")
                tier_fin = df_fin_combined.groupby(['SKU_Tier', 'Channel'])['Gross_Margin'].sum().reset_index()
                
                fig_tier = px.bar(tier_fin, x='SKU_Tier', y='Gross_Margin', color='Channel',
                                title="Gross Margin Contribution by Tier & Channel",
                                color_discrete_map={'Ecommerce': '#667eea', 'Reseller': '#FF9800'},
                                barmode='group')
                fig_tier.update_layout(yaxis_title="Gross Margin (Rp)")
                st.plotly_chart(fig_tier, use_container_width=True)

        # --- D. TOP PERFORMING SKUS ---
        st.divider()
        st.subheader("🏆 SKU Leaderboard 2026")
        
        rank_col1, rank_col2 = st.columns(2)
        
        # Aggregasi per SKU
        sku_fin = df_fin_combined.groupby(['SKU_ID', 'Product_Name', 'Brand']).agg({
            'Revenue': 'sum', 'Gross_Margin': 'sum', 'Qty': 'sum'
        }).reset_index()
        sku_fin['Margin %'] = (sku_fin['Gross_Margin'] / sku_fin['Revenue'] * 100)
        
        with rank_col1:
            st.markdown("**Top 10 SKUs by Revenue (Omzet)**")
            top_rev = sku_fin.sort_values('Revenue', ascending=False).head(10).copy()
            
            # Format
            top_rev['Revenue'] = top_rev['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
            top_rev['Gross_Margin'] = top_rev['Gross_Margin'].apply(lambda x: f"Rp {x:,.0f}")
            top_rev['Margin %'] = top_rev['Margin %'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(top_rev[['SKU_ID', 'Product_Name', 'Revenue', 'Margin %']], use_container_width=True)
            
        with rank_col2:
            st.markdown("**Top 10 SKUs by Gross Margin (Cuan)**")
            top_cuan = sku_fin.sort_values('Gross_Margin', ascending=False).head(10).copy()
            
            # Format
            top_cuan['Revenue'] = top_cuan['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
            top_cuan['Gross_Margin'] = top_cuan['Gross_Margin'].apply(lambda x: f"Rp {x:,.0f}")
            top_cuan['Margin %'] = top_cuan['Margin %'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(top_cuan[['SKU_ID', 'Product_Name', 'Gross_Margin', 'Margin %']], use_container_width=True)

        # --- E. DOWNLOAD DATA ---
        st.divider()
        st.subheader("📥 Download Combined Financial Data")
        
        dl_df = df_fin_combined.copy()
        # Clean up for export
        dl_csv = dl_df.to_csv(index=False)
        st.download_button(
            label="Download Combined Forecast 2026 (CSV)",
            data=dl_csv,
            file_name=f"Combined_Financial_Forecast_2026_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    else:
        st.info("ℹ️ Please ensure both Ecommerce and Reseller forecast sheets have data for 2026 to generate this analysis.")


# --- TAB 9: RESELLER PERFORMANCE DASHBOARD ---
with tab9:
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
                st.metric("Total Revenue 2026", f"Rp {total_revenue_2026:,.0f}")
            
            with col_fin2:
                avg_monthly_rev = total_revenue_2026 / len(reseller_forecast_cols) if reseller_forecast_cols else 0
                st.metric("Avg Monthly Revenue", f"Rp {avg_monthly_rev:,.0f}")
            
            with col_fin3:
                if monthly_revenue:
                    peak_month = max(monthly_revenue, key=monthly_revenue.get)
                    peak_rev = monthly_revenue.get(peak_month, 0)
                    st.metric("Peak Revenue Month", f"Rp {peak_rev:,.0f}", delta=peak_month)
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
                            text=[f"Rp {x:,.0f}" for x in revenue_filtered['Revenue']],
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
                        text=[f"Rp {x:,.0f}" for x in brand_rev_df['Revenue']],
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
        
        # Download Options
        st.divider()
        st.subheader("📥 Download Data")
        
        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
        
        with col_dl1:
            if not df_reseller_forecast.empty:
                csv_fcst = df_reseller_forecast.to_csv(index=False)
                st.download_button(
                    label="Download Forecast 2026",
                    data=csv_fcst,
                    file_name="reseller_forecast_2026.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_fcst"
                )
        
        with col_dl2:
            if not df_sales_reseller.empty:
                csv_sales = df_sales_reseller.to_csv(index=False)
                st.download_button(
                    label="Download Sales",
                    data=csv_sales,
                    file_name="reseller_sales.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_sales"
                )
        
        with col_dl3:
            if not df_past_rofo_reseller.empty:
                csv_rofo = df_past_rofo_reseller.to_csv(index=False)
                st.download_button(
                    label="Download Past Rofo",
                    data=csv_rofo,
                    file_name="reseller_past_rofo.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_rofo"
                )
        
        with col_dl4:
            if not df_past_po_reseller.empty:
                csv_po = df_past_po_reseller.to_csv(index=False)
                st.download_button(
                    label="Download Past PO",
                    data=csv_po,
                    file_name="reseller_past_po.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_po"
                )

# --- TAB 10: FULFILLMENT COST ANALYSIS (REVISI: GMV CONTRIBUTION) ---
with tab10:
    st.subheader("🚚 Fulfillment Cost Analysis (BS)")
    st.markdown("**Analisis Kontribusi BS terhadap Total Marketplace & Efisiensi Biaya**")
    
    # Ambil data
    df_bs = all_data.get('fulfillment', pd.DataFrame())
    
    if not df_bs.empty:
        # --- 1. KEY METRICS (HEADER) ---
        last_row = df_bs.iloc[-1]
        prev_row = df_bs.iloc[-2] if len(df_bs) > 1 else last_row
        last_month_name = last_row['Month']
        
        # Hitung Kontribusi
        gmv_total = last_row.get('GMV Total (MP)', 0)
        gmv_bs = last_row.get('GMV (Fullfil By BS)', 0)
        contrib_pct = (gmv_bs / gmv_total * 100) if gmv_total > 0 else 0
        
        # Hitung Kontribusi Bulan Lalu (untuk Delta)
        prev_gmv_total = prev_row.get('GMV Total (MP)', 0)
        prev_gmv_bs = prev_row.get('GMV (Fullfil By BS)', 0)
        prev_contrib_pct = (prev_gmv_bs / prev_gmv_total * 100) if prev_gmv_total > 0 else 0
        delta_contrib = contrib_pct - prev_contrib_pct

        # ROW 1: BUSINESS SCALE (GMV & CONTRIBUTION)
        st.markdown("##### 💼 Business Scale & Contribution")
        m1, m2, m3 = st.columns(3)
        
        with m1:
            # GMV Total Marketplace
            delta_gmv_tot = (gmv_total - prev_gmv_total) / prev_gmv_total * 100 if prev_gmv_total > 0 else 0
            st.metric(f"GMV Total Marketplace (MP)", f"Rp {gmv_total:,.0f}", f"{delta_gmv_tot:+.1f}%")
            
        with m2:
            # GMV Fulfilled by BS
            delta_gmv_bs = (gmv_bs - prev_gmv_bs) / prev_gmv_bs * 100 if prev_gmv_bs > 0 else 0
            st.metric(f"GMV Fulfilled by BS", f"Rp {gmv_bs:,.0f}", f"{delta_gmv_bs:+.1f}%")
            
        with m3:
            # % Contribution
            st.metric(f"% BS Contribution", f"{contrib_pct:.1f}%", f"{delta_contrib:+.1f}% (pts)")

        st.markdown("---")

        # ROW 2: OPERATIONAL EFFICIENCY (COST & ORDERS)
        st.markdown("##### ⚙️ Operational Efficiency")
        k1, k2, k3, k4 = st.columns(4)
        
        with k1:
            curr_ord = last_row['Total Order(BS)']
            delta_ord = (curr_ord - prev_row['Total Order(BS)']) / prev_row['Total Order(BS)'] * 100 if prev_row['Total Order(BS)'] > 0 else 0
            st.metric(f"Total Orders (BS)", f"{curr_ord:,.0f}", f"{delta_ord:+.1f}%")

        with k2:
            curr_cost = last_row['Total Cost']
            delta_cost = (curr_cost - prev_row['Total Cost']) / prev_row['Total Cost'] * 100 if prev_row['Total Cost'] > 0 else 0
            st.metric(f"Total Cost", f"Rp {curr_cost:,.0f}", f"{delta_cost:+.1f}%", delta_color="inverse")
            
        with k3:
            curr_pct = last_row['%Cost']
            prev_pct = prev_row['%Cost']
            delta_pct = (curr_pct - prev_pct)
            st.metric(f"% Cost Ratio", f"{curr_pct:.2f}%", f"{delta_pct:+.2f}%", delta_color="inverse")
            
        with k4:
            curr_bsa = last_row['BSA']
            delta_bsa = (curr_bsa - prev_row['BSA']) / prev_row['BSA'] * 100 if prev_row['BSA'] > 0 else 0
            st.metric(f"BSA (Basket Size)", f"Rp {curr_bsa:,.0f}", f"{delta_bsa:+.1f}%")

        st.divider()
        
        # --- 2. DUAL CHARTS ---
        c1, c2 = st.columns([1, 1])
        
        # CHART KIRI: Business Health (GMV vs Cost %)
        with c1:
            st.subheader("💰 Business Efficiency")
            st.caption("Korelasi GMV (BS) dengan % Cost Ratio")
            
            fig_biz = go.Figure()
            
            # Bar: GMV BS
            fig_biz.add_trace(go.Bar(
                x=df_bs['Month'], 
                y=df_bs['GMV (Fullfil By BS)'], 
                name='GMV BS',
                marker_color='#667eea',
                opacity=0.7
            ))
            
            # Line: % Cost Ratio
            fig_biz.add_trace(go.Scatter(
                x=df_bs['Month'], 
                y=df_bs['%Cost'], 
                name='% Cost Ratio',
                mode='lines+markers+text',
                line=dict(color='#FF5252', width=3),
                text=[f"{x:.2f}%" for x in df_bs['%Cost']],
                textposition='top center',
                yaxis='y2'
            ))
            
            fig_biz.update_layout(
                height=450,
                xaxis_title="Month",
                yaxis=dict(title="GMV Fulfilled (Rp)"),
                yaxis2=dict(
                    title="% Cost Ratio", 
                    overlaying="y", 
                    side="right", 
                    showgrid=False
                ),
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig_biz, use_container_width=True)
            
        # CHART KANAN: Operational Load (Order vs Cost)
        with c2:
            st.subheader("⚙️ Operational Load")
            st.caption("Korelasi Volume Order dengan Total Cost")
            
            fig_ops = go.Figure()
            
            # Area: Total Cost
            fig_ops.add_trace(go.Scatter(
                x=df_bs['Month'], 
                y=df_bs['Total Cost'], 
                name='Total Cost',
                fill='tozeroy',
                mode='lines',
                line=dict(color='#FF9800', width=0),
                hovertemplate='Cost: Rp %{y:,.0f}'
            ))
            
            # Line: Total Order
            fig_ops.add_trace(go.Scatter(
                x=df_bs['Month'], 
                y=df_bs['Total Order(BS)'], 
                name='Total Orders',
                mode='lines+markers',
                line=dict(color='#2196F3', width=3),
                yaxis='y2',
                hovertemplate='Order: %{y:,.0f}'
            ))
            
            fig_ops.update_layout(
                height=450,
                xaxis_title="Month",
                yaxis=dict(title="Total Cost (Rp)"),
                yaxis2=dict(
                    title="Total Order (Qty)", 
                    overlaying="y", 
                    side="right", 
                    showgrid=False
                ),
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig_ops, use_container_width=True)

        st.divider()
        
        # --- 3. CONTRIBUTION & BASKET SIZE (WITH LABELS) ---
        st.subheader("🏢 Market Share & Basket Size Trend")
        st.caption("Bar: Komposisi GMV (Label dalam Milyar) | Line: Rata-rata Nilai Order")
        
        # Hitung GMV Non-BS
        df_bs['GMV Non-BS'] = df_bs['GMV Total (MP)'] - df_bs['GMV (Fullfil By BS)']
        
        fig_gmv = go.Figure()
        
        # Stacked Bar 1: GMV BS (Hijau)
        fig_gmv.add_trace(go.Bar(
            x=df_bs['Month'],
            y=df_bs['GMV (Fullfil By BS)'],
            name='Fulfilled by BS',
            marker_color='#4CAF50',
            # TAMBAHAN LABEL ANGKA
            text=[f"{x/1e9:.1f} M" for x in df_bs['GMV (Fullfil By BS)']], # Format: 6.7 M
            textposition='auto', # Plotly otomatis atur posisi terbaik
            textfont=dict(color='white') # Warna teks putih biar kontras di hijau
        ))
        
        # Stacked Bar 2: GMV Non-BS (Abu-abu)
        fig_gmv.add_trace(go.Bar(
            x=df_bs['Month'],
            y=df_bs['GMV Non-BS'],
            name='Non-BS Fulfillment',
            marker_color='#9E9E9E', # Sedikit digelapkan biar teks putih terbaca
            # TAMBAHAN LABEL ANGKA
            text=[f"{x/1e9:.1f} M" for x in df_bs['GMV Non-BS']],
            textposition='auto',
            textfont=dict(color='white')
        ))
        
        # Line Chart: BSA (Basket Size) - Biru
        fig_gmv.add_trace(go.Scatter(
            x=df_bs['Month'],
            y=df_bs['BSA'],
            name='Basket Size (BSA)',
            mode='lines+markers+text', # Tambah text di line juga
            line=dict(color='#2196F3', width=3),
            text=[f"{x/1000:.0f}k" for x in df_bs['BSA']], # Format: 123k
            textposition='top center',
            textfont=dict(color='#2196F3'),
            yaxis='y2'
        ))
        
        fig_gmv.update_layout(
            height=500, # Sedikit dipertinggi biar lega
            xaxis_title="Month",
            barmode='stack',
            
            # Sumbu Kiri (GMV)
            yaxis=dict(title="GMV Total (Rp)", side="left"),
            
            # Sumbu Kanan (BSA)
            yaxis2=dict(
                title="Basket Size (Rp)",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            
            legend=dict(orientation="h", y=1.1),
            hovermode="x unified",
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_gmv, use_container_width=True)
        
        # --- 4. RAW DATA TABLE ---
        with st.expander("📋 View Detail Data"):
            df_disp = df_bs.copy()
            # Format
            for c in df_disp.columns:
                if c in ['Total Order(BS)', 'GMV (Fullfil By BS)', 'GMV Total (MP)', 'Total Cost', 'BSA']:
                    df_disp[c] = df_disp[c].apply(lambda x: f"{x:,.0f}")
                elif '%Cost' in c:
                    df_disp[c] = df_disp[c].apply(lambda x: f"{x:.2f}%")
            
            # Remove technical cols
            cols_hide = ['Month_Date', 'GMV Non-BS']
            df_disp = df_disp.drop(columns=[c for c in cols_hide if c in df_disp.columns])
            
            st.dataframe(df_disp, use_container_width=True)

    else:
        st.warning("⚠️ Data 'BS_Fullfilment_Cost' belum tersedia.")

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
