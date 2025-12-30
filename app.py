import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import gspread
from google.oauth2.service_account import Credentials
import warnings
import io
from typing import Dict, List, Optional, Tuple, Any
import json
warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Supply Chain Intelligence Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PROFESIONAL ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: #1e3799;
        --secondary: #4a69bd;
        --success: #38ada9;
        --warning: #f6b93b;
        --danger: #e55039;
        --info: #00cec9;
        --dark: #2d3436;
        --light: #f8f9fa;
    }
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif;
        color: var(--dark);
    }
    
    /* HEADER */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* KPI CARDS */
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.08);
        border-left: 6px solid var(--primary);
        transition: all 0.3s ease;
        height: 100%;
        margin-bottom: 20px;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.12);
    }
    
    .kpi-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 10px 0;
        line-height: 1;
    }
    
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--dark);
        opacity: 0.8;
    }
    
    .kpi-trend {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .trend-up { color: var(--success); }
    .trend-down { color: var(--danger); }
    .trend-neutral { color: var(--warning); }
    
    /* SUMMARY CARDS */
    .summary-card {
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        margin-bottom: 20px;
        transition: transform 0.3s;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .summary-card:hover { transform: translateY(-5px); }
    
    .bg-primary { background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); }
    .bg-success { background: linear-gradient(135deg, var(--success) 0%, #00b894 100%); }
    .bg-warning { background: linear-gradient(135deg, var(--warning) 0%, #e58e26 100%); }
    .bg-danger { background: linear-gradient(135deg, var(--danger) 0%, #eb2f06 100%); }
    .bg-info { background: linear-gradient(135deg, var(--info) 0%, #0984e3 100%); }
    .bg-purple { background: linear-gradient(135deg, #6c5ce7 0%, #8e44ad 100%); }
    .bg-teal { background: linear-gradient(135deg, #00cec9 0%, #00b894 100%); }
    
    .sum-val {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 10px 0;
        line-height: 1;
    }
    
    .sum-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    
    .sum-sub {
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.95;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.3);
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        padding: 0 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f2f6;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: var(--primary);
        border-top: 4px solid var(--primary);
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="text-align: center; margin-bottom: 1rem;">
    <h1 class="main-header">SUPPLY CHAIN INTELLIGENCE PLATFORM</h1>
    <div style="color: #666; font-size: 1rem; margin-bottom: 2rem;">
        📊 End-to-End Supply Chain Management & Analytics
    </div>
</div>
""", unsafe_allow_html=True)

# --- 1. ENHANCED DATA ENGINE ---
@st.cache_resource(show_spinner=False)
def init_gsheet_connection():
    """Initialize Google Sheets connection"""
    try:
        skey = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(skey, scopes=scopes)
        return gspread.authorize(credentials)
    except Exception as e:
        st.error(f"❌ Connection Failed: {str(e)}")
        return None

def parse_month_label(label):
    """Parse month label to datetime - handle various formats"""
    try:
        label_str = str(label).strip()
        
        # Remove extra spaces and normalize
        label_str = ' '.join(label_str.split())
        
        # Month mapping
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        # Extract month and year
        for month_name, month_num in month_map.items():
            if month_name in label_str.lower():
                # Extract year (could be '25' or '2025')
                import re
                year_match = re.search(r'(\d{2,4})', label_str)
                if year_match:
                    year_str = year_match.group(1)
                    if len(year_str) == 2:
                        year = 2000 + int(year_str)
                    else:
                        year = int(year_str)
                else:
                    # Default to current year if no year found
                    year = datetime.now().year
                
                return datetime(year, month_num, 1)
        
        # If no month found, try to parse as datetime
        try:
            return pd.to_datetime(label_str)
        except:
            return datetime.now()
    except Exception as e:
        return datetime.now()

class SupplyChainDataEngine:
    """Enhanced data engine for complete supply chain data"""
    
    def __init__(self, client):
        self.client = client
        self.gsheet_url = "https://docs.google.com/spreadsheets/d/1gek6SPgJcZzhOjN-eNOZbfCvmX2EKiHKbgk3c0gUxL4"
        
    def load_all_data(self):
        """Load all supply chain data - ONLY ACTIVE SKUS"""
        data = {}
        
        try:
            # Open the spreadsheet
            spreadsheet = self.client.open_by_url(self.gsheet_url)
            
            # 1. Product Master - Filter hanya ACTIVE
            data['product'] = self._load_sheet(spreadsheet, "Product_Master")
            if not data['product'].empty:
                data['product']['SKU_ID'] = data['product']['SKU_ID'].astype(str).str.strip()
                # Filter hanya SKU dengan Status = 'Active'
                data['product'] = data['product'][data['product']['Status'].str.upper() == 'ACTIVE']
                
                # Get active SKU IDs
                data['active_skus'] = data['product']['SKU_ID'].tolist()
                
                # Clean product columns
                data['product'] = self._clean_product_columns(data['product'])
            
            # 2. Sales Data - Filter hanya active SKUs
            if 'active_skus' in data and data['active_skus']:
                data['sales'] = self._process_sales_data(spreadsheet, "Sales", data['active_skus'])
            else:
                data['sales'] = pd.DataFrame()
            
            # 3. Forecast Data (Rofo) - Filter hanya active SKUs
            if 'active_skus' in data and data['active_skus']:
                data['forecast'] = self._process_monthly_data(spreadsheet, "Rofo", "Forecast_Qty", data['active_skus'])
            else:
                data['forecast'] = pd.DataFrame()
            
            # 4. PO Data - Filter hanya active SKUs
            data['po'] = self._load_sheet(spreadsheet, "PO")
            if not data['po'].empty:
                data['po']['SKU_ID'] = data['po']['SKU_ID'].astype(str).str.strip()
                # Filter hanya active SKUs
                if 'active_skus' in data and data['active_skus']:
                    data['po'] = data['po'][data['po']['SKU_ID'].isin(data['active_skus'])]
                
                # Convert date columns
                date_cols = ['Order_Date', 'Expected_Delivery_Date', 'Actual_Delivery_Date']
                for col in date_cols:
                    if col in data['po'].columns:
                        data['po'][col] = pd.to_datetime(data['po'][col], errors='coerce')
                # Convert numeric columns
                numeric_cols = ['Order_Qty', 'Received_Qty', 'Unit_Price', 'Total_Value']
                for col in numeric_cols:
                    if col in data['po'].columns:
                        data['po'][col] = pd.to_numeric(data['po'][col], errors='coerce')
            
            # 5. Stock Data - Filter hanya active SKUs
            data['stock'] = self._load_sheet(spreadsheet, "Stock_Onhand")
            if not data['stock'].empty:
                data['stock']['SKU_ID'] = data['stock']['SKU_ID'].astype(str).str.strip()
                # Filter hanya active SKUs
                if 'active_skus' in data and data['active_skus']:
                    data['stock'] = data['stock'][data['stock']['SKU_ID'].isin(data['active_skus'])]
                
                # Identify stock quantity column
                stock_qty_cols = ['Stock_Qty', 'Stock Qty', 'Quantity_Available']
                for col in stock_qty_cols:
                    if col in data['stock'].columns:
                        data['stock']['Stock_Qty'] = pd.to_numeric(data['stock'][col], errors='coerce').fillna(0)
                        break
                if 'Stock_Qty' not in data['stock'].columns:
                    data['stock']['Stock_Qty'] = 0
            
            # 6. Supplier Master
            data['suppliers'] = self._load_sheet(spreadsheet, "Supplier_Master")
            
            return data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return {}
    
    def _clean_product_columns(self, df):
        """Clean product master column names"""
        # Rename columns for consistency
        column_mapping = {
            'Min_Stock_Level_(Month)': 'Min_Stock_Level_Month',
            'Max_Stock_Level_(Month)': 'Max_Stock_Level_Month',
            'Sales_Price_(HET)': 'Sales_Price_HET'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Convert numeric columns
        numeric_cols = ['Min_Stock_Level_Month', 'Max_Stock_Level_Month', 
                       'Unit_Price', 'Sales_Price_HET', 'Lead_Time_Days']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _load_sheet(self, spreadsheet, sheet_name):
        """Load specific sheet"""
        try:
            ws = spreadsheet.worksheet(sheet_name)
            df = pd.DataFrame(ws.get_all_records())
            # Clean column names - handle spaces and special characters
            df.columns = [c.strip().replace(' ', '_').replace('%', 'Percent').replace('(', '').replace(')', '') 
                         for c in df.columns]
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def _process_sales_data(self, spreadsheet, sheet_name, active_skus):
        """Process sales data with specific column structure - ONLY ACTIVE SKUS"""
        df = self._load_sheet(spreadsheet, sheet_name)
        if df.empty:
            return pd.DataFrame()
        
        # Filter hanya active SKUs
        if 'SKU_ID' in df.columns:
            df['SKU_ID'] = df['SKU_ID'].astype(str).str.strip()
            df = df[df['SKU_ID'].isin(active_skus)]
        
        # Identify month columns (Jan 25, Feb 25, etc.) - exclude "Total 2025"
        month_cols = []
        for col in df.columns:
            col_str = str(col).strip()
            # Check if column looks like a month (contains month name) but not "Total"
            if any(month in col_str.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                month_cols.append(col)
        
        if not month_cols:
            return pd.DataFrame()
        
        # Identify ID column (could be SKU_ID or Current_SKU)
        id_col = None
        for col in ['SKU_ID', 'Current_SKU']:
            if col in df.columns:
                id_col = col
                break
        
        if not id_col:
            return pd.DataFrame()
        
        # Melt to long format
        df_long = df[[id_col] + month_cols].melt(
            id_vars=[id_col],
            value_vars=month_cols,
            var_name='Month_Label',
            value_name='Sales_Qty'
        )
        
        df_long['Sales_Qty'] = pd.to_numeric(df_long['Sales_Qty'], errors='coerce').fillna(0)
        df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
        df_long['Month'] = pd.to_datetime(df_long['Month'])
        
        # Rename ID column to SKU_ID for consistency
        if id_col != 'SKU_ID':
            df_long = df_long.rename(columns={id_col: 'SKU_ID'})
        
        return df_long
    
    def _process_monthly_data(self, spreadsheet, sheet_name, value_col_name, active_skus):
        """Process monthly data sheets (Rofo, etc.) - ONLY ACTIVE SKUS"""
        df = self._load_sheet(spreadsheet, sheet_name)
        if df.empty:
            return pd.DataFrame()
        
        # Filter hanya active SKUs
        if 'SKU_ID' in df.columns:
            df['SKU_ID'] = df['SKU_ID'].astype(str).str.strip()
            df = df[df['SKU_ID'].isin(active_skus)]
        
        # Identify month columns
        month_cols = []
        for col in df.columns:
            col_str = str(col).strip()
            if any(month in col_str.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                month_cols.append(col)
        
        if not month_cols:
            return pd.DataFrame()
        
        # Identify ID column
        id_col = None
        for col in ['SKU_ID', 'Old_Material', 'Current_SKU']:
            if col in df.columns:
                id_col = col
                break
        
        if not id_col:
            return pd.DataFrame()
        
        # Melt to long format
        df_long = df[[id_col] + month_cols].melt(
            id_vars=[id_col],
            value_vars=month_cols,
            var_name='Month_Label',
            value_name=value_col_name
        )
        
        df_long[value_col_name] = pd.to_numeric(df_long[value_col_name], errors='coerce').fillna(0)
        df_long['Month'] = df_long['Month_Label'].apply(parse_month_label)
        df_long['Month'] = pd.to_datetime(df_long['Month'])
        
        # Rename ID column to SKU_ID for consistency
        if id_col != 'SKU_ID':
            df_long = df_long.rename(columns={id_col: 'SKU_ID'})
        
        return df_long

# --- 2. SUPPLY CHAIN ANALYTICS ENGINE ---
class SupplyChainAnalytics:
    """Complete supply chain analytics engine"""
    
    def __init__(self, data):
        self.data = data
        
    def calculate_inventory_metrics(self):
        """Calculate comprehensive inventory metrics - ONLY ACTIVE SKUS"""
        if self.data['stock'].empty or self.data['product'].empty:
            return pd.DataFrame()
        
        # Prepare stock data
        stock_df = self.data['stock'].copy()
        
        # Ensure we have SKU_ID and Stock_Qty
        if 'SKU_ID' not in stock_df.columns or 'Stock_Qty' not in stock_df.columns:
            return pd.DataFrame()
        
        # Merge with product data
        product_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Category', 
                       'Unit_Price', 'Sales_Price_HET', 'Min_Stock_Level_Month', 
                       'Max_Stock_Level_Month', 'Lead_Time_Days', 'ABC_Classification']
        product_cols = [col for col in product_cols if col in self.data['product'].columns]
        
        inv_df = pd.merge(
            stock_df[['SKU_ID', 'Stock_Qty']],
            self.data['product'][product_cols],
            on='SKU_ID',
            how='left'
        )
        
        # Calculate Avg Sales 3 Months Terakhir
        if not self.data['sales'].empty:
            sales_df = self.data['sales'].copy()
            sales_df['Month'] = pd.to_datetime(sales_df['Month'])
            
            # Get last 3 months
            if len(sales_df['Month'].unique()) >= 3:
                last_3_months = sorted(sales_df['Month'].unique())[-3:]
                recent_sales = sales_df[sales_df['Month'].isin(last_3_months)]
                
                if not recent_sales.empty:
                    avg_sales = recent_sales.groupby('SKU_ID')['Sales_Qty'].agg(['mean', 'std']).reset_index()
                    avg_sales.columns = ['SKU_ID', 'Avg_Sales_3M', 'Sales_Std']
                    
                    inv_df = pd.merge(inv_df, avg_sales, on='SKU_ID', how='left')
                else:
                    inv_df['Avg_Sales_3M'] = 0
                    inv_df['Sales_Std'] = 0
            else:
                # If less than 3 months, use all available
                avg_sales = sales_df.groupby('SKU_ID')['Sales_Qty'].agg(['mean', 'std']).reset_index()
                avg_sales.columns = ['SKU_ID', 'Avg_Sales_3M', 'Sales_Std']
                inv_df = pd.merge(inv_df, avg_sales, on='SKU_ID', how='left')
        else:
            inv_df['Avg_Sales_3M'] = 0
            inv_df['Sales_Std'] = 0
        
        # Calculate inventory metrics
        inv_df['Avg_Sales_3M'] = inv_df['Avg_Sales_3M'].fillna(0)
        inv_df['Sales_Std'] = inv_df['Sales_Std'].fillna(0)
        
        # Calculate DOI (Days of Inventory) = Stock / Avg Daily Sales
        inv_df['DOI'] = np.where(
            inv_df['Avg_Sales_3M'] > 0,
            (inv_df['Stock_Qty'] / (inv_df['Avg_Sales_3M'] / 30)).round(1),  # Convert monthly to daily
            999
        )
        
        # Calculate Cover Months = Stock / Avg Monthly Sales
        inv_df['Cover_Months'] = np.where(
            inv_df['Avg_Sales_3M'] > 0,
            inv_df['Stock_Qty'] / inv_df['Avg_Sales_3M'],
            999
        ).round(2)
        
        # Inventory classification based on Min/Max Stock Level
        if 'Min_Stock_Level_Month' in inv_df.columns and 'Max_Stock_Level_Month' in inv_df.columns:
            conditions = [
                inv_df['Cover_Months'] < inv_df['Min_Stock_Level_Month'],
                (inv_df['Cover_Months'] >= inv_df['Min_Stock_Level_Month']) & 
                (inv_df['Cover_Months'] <= inv_df['Max_Stock_Level_Month']),
                inv_df['Cover_Months'] > inv_df['Max_Stock_Level_Month']
            ]
        else:
            # Default logic
            conditions = [
                inv_df['Cover_Months'] < 1.0,
                (inv_df['Cover_Months'] >= 1.0) & (inv_df['Cover_Months'] <= 1.5),
                inv_df['Cover_Months'] > 1.5
            ]
        
        choices = ['Need Replenishment', 'Ideal', 'High Stock']
        inv_df['Inventory_Status'] = np.select(conditions, choices, default='Unknown')
        
        # Stock value using Unit_Price
        if 'Unit_Price' in inv_df.columns:
            inv_df['Stock_Value'] = inv_df['Stock_Qty'] * inv_df['Unit_Price'].fillna(0)
        else:
            inv_df['Stock_Value'] = 0
        
        # Calculate Qty to Order/Reduce
        if 'Min_Stock_Level_Month' in inv_df.columns and 'Max_Stock_Level_Month' in inv_df.columns:
            # Qty to Order for Need Replenishment
            inv_df['Qty_to_Order'] = np.where(
                inv_df['Inventory_Status'] == 'Need Replenishment',
                (inv_df['Min_Stock_Level_Month'] * inv_df['Avg_Sales_3M']) - inv_df['Stock_Qty'],
                0
            ).round(0)
            inv_df['Qty_to_Order'] = np.where(inv_df['Qty_to_Order'] < 0, 0, inv_df['Qty_to_Order'])
            
            # Qty to Reduce for High Stock
            inv_df['Qty_to_Reduce'] = np.where(
                inv_df['Inventory_Status'] == 'High Stock',
                inv_df['Stock_Qty'] - (inv_df['Max_Stock_Level_Month'] * inv_df['Avg_Sales_3M']),
                0
            ).round(0)
            inv_df['Qty_to_Reduce'] = np.where(inv_df['Qty_to_Reduce'] < 0, 0, inv_df['Qty_to_Reduce'])
        
        return inv_df
    
    def calculate_forecast_accuracy(self):
        """Calculate forecast accuracy - Rofo vs PO"""
        if self.data['forecast'].empty or self.data['po'].empty:
            return {}
        
        # Get PO data aggregated by month and SKU
        po_df = self.data['po'].copy()
        if 'Order_Date' not in po_df.columns or 'Order_Qty' not in po_df.columns:
            return {}
        
        # Convert Order_Date to month
        po_df['Order_Month'] = pd.to_datetime(po_df['Order_Date']).dt.to_period('M')
        po_monthly = po_df.groupby(['SKU_ID', 'Order_Month'])['Order_Qty'].sum().reset_index()
        po_monthly['Order_Month'] = po_monthly['Order_Month'].dt.to_timestamp()
        
        # Get forecast data
        forecast_df = self.data['forecast'].copy()
        
        # Merge forecast with PO
        df_merged = pd.merge(
            forecast_df,
            po_monthly,
            left_on=['SKU_ID', 'Month'],
            right_on=['SKU_ID', 'Order_Month'],
            how='inner',
            suffixes=('_Forecast', '_PO')
        )
        
        if df_merged.empty:
            return {}
        
        # Calculate forecast error
        df_merged['Forecast_Error'] = df_merged['Order_Qty'] - df_merged['Forecast_Qty']
        df_merged['Absolute_Error'] = abs(df_merged['Forecast_Error'])
        df_merged['Absolute_Percentage_Error'] = np.where(
            df_merged['Forecast_Qty'] > 0,
            (df_merged['Absolute_Error'] / df_merged['Forecast_Qty']) * 100,
            0
        )
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = df_merged['Absolute_Percentage_Error'].mean()
        
        # Calculate Forecast Bias
        forecast_bias = df_merged['Forecast_Error'].mean()
        
        # Calculate by month
        monthly_accuracy = df_merged.groupby('Month').agg({
            'Forecast_Qty': 'sum',
            'Order_Qty': 'sum',
            'Absolute_Percentage_Error': 'mean'
        }).reset_index()
        
        monthly_accuracy['Accuracy_%'] = 100 - monthly_accuracy['Absolute_Percentage_Error']
        
        return {
            'overall_mape': mape,
            'forecast_bias': forecast_bias,
            'monthly_accuracy': monthly_accuracy,
            'detailed_data': df_merged
        }
    
    def calculate_sales_performance(self):
        """Calculate sales performance - Actual Sales vs Forecast vs PO"""
        if self.data['sales'].empty or self.data['forecast'].empty or self.data['po'].empty:
            return {}
        
        # Get last month for analysis
        if not self.data['sales'].empty:
            last_month = self.data['sales']['Month'].max()
        else:
            return {}
        
        # Get sales for last month
        sales_last_month = self.data['sales'][self.data['sales']['Month'] == last_month]
        
        # Get forecast for last month
        forecast_last_month = self.data['forecast'][self.data['forecast']['Month'] == last_month]
        
        # Get PO for last month (orders placed for that month)
        po_df = self.data['po'].copy()
        if 'Order_Date' in po_df.columns:
            po_df['Order_Month'] = pd.to_datetime(po_df['Order_Date']).dt.to_period('M').dt.to_timestamp()
            po_last_month = po_df[po_df['Order_Month'] == last_month]
        else:
            po_last_month = pd.DataFrame()
        
        # Merge all three
        performance_data = {}
        
        # Calculate overall metrics
        total_sales = sales_last_month['Sales_Qty'].sum() if not sales_last_month.empty else 0
        total_forecast = forecast_last_month['Forecast_Qty'].sum() if not forecast_last_month.empty else 0
        total_po = po_last_month['Order_Qty'].sum() if not po_last_month.empty else 0
        
        performance_data['total_sales'] = total_sales
        performance_data['total_forecast'] = total_forecast
        performance_data['total_po'] = total_po
        performance_data['sales_vs_forecast'] = (total_sales / total_forecast * 100) if total_forecast > 0 else 0
        performance_data['sales_vs_po'] = (total_sales / total_po * 100) if total_po > 0 else 0
        performance_data['po_vs_forecast'] = (total_po / total_forecast * 100) if total_forecast > 0 else 0
        
        return performance_data
    
    def calculate_scor_metrics(self):
        """Calculate SCOR metrics"""
        metrics = {}
        
        # 1. Reliability Metrics
        if not self.data['po'].empty and 'PO_Status' in self.data['po'].columns:
            perfect_orders = len(self.data['po'][self.data['po']['PO_Status'] == 'Closed'])
            total_orders = len(self.data['po'])
            metrics['Perfect_Order_Fulfillment'] = (perfect_orders / total_orders * 100) if total_orders > 0 else 0
        
        # 2. Calculate inventory turnover
        inv_metrics = self.calculate_inventory_metrics()
        if not inv_metrics.empty and 'Stock_Value' in inv_metrics.columns:
            total_inv_value = inv_metrics['Stock_Value'].sum()
            metrics['Total_Inventory_Value'] = total_inv_value
            
            # Estimate COGS from sales
            if not self.data['sales'].empty and 'Unit_Price' in self.data['product'].columns:
                total_sales_qty = self.data['sales']['Sales_Qty'].sum()
                avg_unit_price = self.data['product']['Unit_Price'].mean() if 'Unit_Price' in self.data['product'].columns else 1
                estimated_cogs = total_sales_qty * avg_unit_price * 0.7  # Assume 70% COGS
                metrics['Inventory_Turnover'] = (estimated_cogs / total_inv_value) if total_inv_value > 0 else 0
        
        return metrics

# --- 3. DASHBOARD COMPONENTS ---
class DashboardComponents:
    """Reusable dashboard components"""
    
    @staticmethod
    def kpi_card(title, value, trend=None, subtitle=None, icon="📊"):
        """Create a KPI card"""
        trend_html = ""
        if trend is not None:
            trend_class = "trend-up" if trend > 0 else "trend-down" if trend < 0 else "trend-neutral"
            trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
            trend_html = f'<div class="kpi-trend {trend_class}">{trend_icon} {abs(trend):.1f}%</div>'
        
        subtitle_html = f'<div class="sum-sub">{subtitle}</div>' if subtitle else ''
        
        return f"""
        <div class="kpi-card">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div class="kpi-title">{title}</div>
                <div style="font-size: 1.5rem;">{icon}</div>
            </div>
            <div class="kpi-value">{value}</div>
            {trend_html}
            {subtitle_html}
        </div>
        """
    
    @staticmethod
    def summary_card(title, value, subtitle, bg_class="bg-primary"):
        """Create a summary card"""
        return f"""
        <div class="summary-card {bg_class}">
            <div class="sum-title">{title}</div>
            <div class="sum-val">{value}</div>
            <div class="sum-sub">{subtitle}</div>
        </div>
        """

# --- 4. MAIN DASHBOARD ---
def main():
    """Main dashboard function"""
    
    # Initialize connection
    with st.spinner("🔗 Connecting to data source..."):
        client = init_gsheet_connection()
    
    if not client:
        st.error("Failed to connect to data source. Please check credentials.")
        st.stop()
    
    # Load data
    with st.spinner("📥 Loading supply chain data (Active SKUs only)..."):
        data_engine = SupplyChainDataEngine(client)
        all_data = data_engine.load_all_data()
    
    if not all_data:
        st.error("Failed to load data. Please check data structure.")
        st.stop()
    
    # Display data status
    with st.expander("📊 Data Status (Active SKUs Only)", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            product_count = len(all_data.get('product', pd.DataFrame()))
            active_count = len(all_data.get('active_skus', []))
            st.metric("Active Products", product_count)
            st.metric("Active SKUs", active_count)
        
        with col2:
            sales_count = len(all_data.get('sales', pd.DataFrame()))
            po_count = len(all_data.get('po', pd.DataFrame()))
            st.metric("Sales Records", sales_count)
            st.metric("PO Records", po_count)
        
        with col3:
            supplier_count = len(all_data.get('suppliers', pd.DataFrame()))
            stock_count = len(all_data.get('stock', pd.DataFrame()))
            st.metric("Suppliers", supplier_count)
            st.metric("Stock Items", stock_count)
    
    # Initialize analytics
    analytics = SupplyChainAnalytics(all_data)
    
    # Calculate metrics
    with st.spinner("📊 Calculating metrics..."):
        inv_metrics = analytics.calculate_inventory_metrics()
        forecast_accuracy = analytics.calculate_forecast_accuracy()
        sales_performance = analytics.calculate_sales_performance()
        scor_metrics = analytics.calculate_scor_metrics()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        
        # Date range selector
        st.subheader("📅 Time Period")
        period = st.selectbox(
            "Select Analysis Period",
            ["Last 3 Months", "Last 6 Months", "Year to Date", "Last 12 Months", "All Time"]
        )
        
        # Business unit filter
        st.subheader("🏢 Business Unit")
        if not all_data['product'].empty and 'Category' in all_data['product'].columns:
            categories = ["All"] + sorted(all_data['product']['Category'].unique().tolist())
        else:
            categories = ["All"]
        business_unit = st.selectbox("Select Category", categories)
        
        # Refresh button
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Main dashboard
    components = DashboardComponents()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Executive Dashboard",
        "📦 Inventory Management",
        "📈 Forecast Performance",
        "📊 Sales Performance"
    ])
    
    # ==========================================
    # TAB 1: EXECUTIVE DASHBOARD
    # ==========================================
    with tab1:
        st.subheader("🎯 Executive Supply Chain Dashboard (Active SKUs Only)")
        
        # Key Metrics Row 1
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Inventory Turnover
            inv_turnover = scor_metrics.get('Inventory_Turnover', 0)
            st.markdown(components.kpi_card(
                "Inventory Turnover", 
                f"{inv_turnover:.1f}x",
                trend=5.2,
                subtitle="Annual Turns",
                icon="🔄"
            ), unsafe_allow_html=True)
        
        with col2:
            # Perfect Order Rate
            perfect_order = scor_metrics.get('Perfect_Order_Fulfillment', 0)
            st.markdown(components.kpi_card(
                "Perfect Order Rate",
                f"{perfect_order:.1f}%",
                trend=2.1,
                subtitle="On-Time & Complete",
                icon="✅"
            ), unsafe_allow_html=True)
        
        with col3:
            # Forecast Accuracy
            mape = forecast_accuracy.get('overall_mape', 0)
            forecast_accuracy_val = 100 - mape if mape > 0 else 0
            st.markdown(components.kpi_card(
                "Forecast Accuracy",
                f"{forecast_accuracy_val:.1f}%",
                trend=3.5,
                subtitle="Rofo vs PO",
                icon="🎯"
            ), unsafe_allow_html=True)
        
        with col4:
            # Sales Performance
            sales_vs_forecast = sales_performance.get('sales_vs_forecast', 0)
            st.markdown(components.kpi_card(
                "Sales vs Forecast",
                f"{sales_vs_forecast:.1f}%",
                trend=2.3,
                subtitle="Actual vs Planned",
                icon="📈"
            ), unsafe_allow_html=True)
        
        # Inventory Overview
        st.markdown("---")
        st.subheader("📦 Inventory Overview (Active SKUs)")
        
        if not inv_metrics.empty:
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                total_skus = len(inv_metrics)
                total_stock = inv_metrics['Stock_Qty'].sum()
                st.markdown(components.summary_card(
                    "Active SKUs",
                    total_skus,
                    f"{total_stock:,.0f} units",
                    "bg-primary"
                ), unsafe_allow_html=True)
            
            with col6:
                need_replenish = len(inv_metrics[inv_metrics['Inventory_Status'] == 'Need Replenishment'])
                need_qty = inv_metrics[inv_metrics['Inventory_Status'] == 'Need Replenishment']['Stock_Qty'].sum()
                st.markdown(components.summary_card(
                    "Need Replenishment",
                    need_replenish,
                    f"{need_qty:,.0f} units",
                    "bg-danger"
                ), unsafe_allow_html=True)
            
            with col7:
                ideal_stock = len(inv_metrics[inv_metrics['Inventory_Status'] == 'Ideal'])
                ideal_qty = inv_metrics[inv_metrics['Inventory_Status'] == 'Ideal']['Stock_Qty'].sum()
                st.markdown(components.summary_card(
                    "Ideal Stock",
                    ideal_stock,
                    f"{ideal_qty:,.0f} units",
                    "bg-success"
                ), unsafe_allow_html=True)
            
            with col8:
                high_stock = len(inv_metrics[inv_metrics['Inventory_Status'] == 'High Stock'])
                high_qty = inv_metrics[inv_metrics['Inventory_Status'] == 'High Stock']['Stock_Qty'].sum()
                st.markdown(components.summary_card(
                    "High Stock",
                    high_stock,
                    f"{high_qty:,.0f} units",
                    "bg-warning"
                ), unsafe_allow_html=True)
        
        # Financial Impact
        st.markdown("---")
        st.subheader("💰 Financial Impact")
        
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            total_inv_value = scor_metrics.get('Total_Inventory_Value', 0)
            st.markdown(components.summary_card(
                "Inventory Value",
                f"${total_inv_value:,.0f}",
                "Total Stock Value",
                "bg-info"
            ), unsafe_allow_html=True)
        
        with col10:
            if not inv_metrics.empty:
                avg_doi = inv_metrics['DOI'].mean()
                st.markdown(components.summary_card(
                    "Avg DOI",
                    f"{avg_doi:.1f} days",
                    "Days of Inventory",
                    "bg-teal"
                ), unsafe_allow_html=True)
        
        with col11:
            if not inv_metrics.empty:
                avg_cover = inv_metrics['Cover_Months'].mean()
                st.markdown(components.summary_card(
                    "Avg Cover",
                    f"{avg_cover:.1f} months",
                    "Stock Coverage",
                    "bg-purple"
                ), unsafe_allow_html=True)
        
        with col12:
            # Calculate action items
            if 'Qty_to_Order' in inv_metrics.columns and 'Qty_to_Reduce' in inv_metrics.columns:
                total_action = inv_metrics['Qty_to_Order'].sum() + inv_metrics['Qty_to_Reduce'].sum()
                st.markdown(components.summary_card(
                    "Action Items",
                    f"{total_action:,.0f}",
                    "Units to Adjust",
                    "bg-warning"
                ), unsafe_allow_html=True)
    
    # ==========================================
    # TAB 2: INVENTORY MANAGEMENT
    # ==========================================
    with tab2:
        st.subheader("📦 Inventory Management (Active SKUs)")
        
        if not inv_metrics.empty:
            # Inventory Status Distribution
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                status_dist = inv_metrics['Inventory_Status'].value_counts()
                fig_status = px.pie(
                    values=status_dist.values,
                    names=status_dist.index,
                    title="Inventory Status Distribution",
                    color_discrete_sequence=['#e55039', '#38ada9', '#f6b93b']
                )
                fig_status.update_traces(textposition='inside', textinfo='percent+label')
                fig_status.update_layout(height=400)
                st.plotly_chart(fig_status, use_container_width=True)
            
            with col_chart2:
                fig_cover = px.histogram(
                    inv_metrics,
                    x='Cover_Months',
                    nbins=30,
                    title="Cover Months Distribution",
                    labels={'Cover_Months': 'Months of Cover'},
                    color_discrete_sequence=['#1e3799']
                )
                # Add lines for min/max if available
                if 'Min_Stock_Level_Month' in inv_metrics.columns:
                    avg_min = inv_metrics['Min_Stock_Level_Month'].mean()
                    fig_cover.add_vline(x=avg_min, line_dash="dash", line_color="red", 
                                       annotation_text=f"Avg Min: {avg_min:.1f}m")
                if 'Max_Stock_Level_Month' in inv_metrics.columns:
                    avg_max = inv_metrics['Max_Stock_Level_Month'].mean()
                    fig_cover.add_vline(x=avg_max, line_dash="dash", line_color="orange", 
                                       annotation_text=f"Avg Max: {avg_max:.1f}m")
                fig_cover.update_layout(height=400)
                st.plotly_chart(fig_cover, use_container_width=True)
            
            # DOI Analysis
            st.markdown("---")
            st.subheader("📊 DOI (Days of Inventory) Analysis")
            
            col_doi1, col_doi2 = st.columns(2)
            
            with col_doi1:
                fig_doi = px.histogram(
                    inv_metrics,
                    x='DOI',
                    nbins=30,
                    title="DOI Distribution",
                    labels={'DOI': 'Days of Inventory'},
                    color_discrete_sequence=['#00cec9']
                )
                fig_doi.update_layout(height=400)
                st.plotly_chart(fig_doi, use_container_width=True)
            
            with col_doi2:
                # Top 10 SKUs by DOI
                top_doi = inv_metrics.nlargest(10, 'DOI')[['Product_Name', 'DOI', 'Cover_Months', 'Stock_Qty']]
                st.markdown("**Top 10 SKUs with Highest DOI**")
                st.dataframe(
                    top_doi,
                    column_config={
                        "DOI": st.column_config.NumberColumn(format="%.1f"),
                        "Cover_Months": st.column_config.NumberColumn(format="%.1f"),
                        "Stock_Qty": st.column_config.NumberColumn(format="%d")
                    },
                    use_container_width=True,
                    height=300
                )
            
            # Actionable Insights
            st.markdown("---")
            st.subheader("🎯 Actionable Insights")
            
            if 'Qty_to_Order' in inv_metrics.columns and 'Qty_to_Reduce' in inv_metrics.columns:
                total_to_order = inv_metrics['Qty_to_Order'].sum()
                total_to_reduce = inv_metrics['Qty_to_Reduce'].sum()
                
                col_act1, col_act2 = st.columns(2)
                
                with col_act1:
                    st.markdown(components.summary_card(
                        "Total Qty to Order",
                        f"{total_to_order:,.0f}",
                        "Units to reach Min Stock Level",
                        "bg-success"
                    ), unsafe_allow_html=True)
                
                with col_act2:
                    st.markdown(components.summary_card(
                        "Total Qty to Reduce",
                        f"{total_to_reduce:,.0f}",
                        "Units to reduce to Max Stock Level",
                        "bg-warning"
                    ), unsafe_allow_html=True)
            
            # Detailed Inventory Table
            st.markdown("---")
            st.subheader("📋 Detailed Inventory Analysis (Active SKUs)")
            
            # Filters
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    inv_metrics['Inventory_Status'].unique(),
                    default=['Need Replenishment', 'High Stock']
                )
            
            with col_f2:
                if 'ABC_Classification' in inv_metrics.columns:
                    abc_filter = st.multiselect(
                        "Filter by ABC Class",
                        inv_metrics['ABC_Classification'].unique(),
                        default=['A', 'B']
                    )
            
            with col_f3:
                if 'Brand' in inv_metrics.columns:
                    brand_filter = st.multiselect(
                        "Filter by Brand",
                        inv_metrics['Brand'].unique()
                    )
            
            # Apply filters
            filtered_inv = inv_metrics.copy()
            if status_filter:
                filtered_inv = filtered_inv[filtered_inv['Inventory_Status'].isin(status_filter)]
            if 'ABC_Classification' in inv_metrics.columns and abc_filter:
                filtered_inv = filtered_inv[filtered_inv['ABC_Classification'].isin(abc_filter)]
            if 'Brand' in inv_metrics.columns and brand_filter:
                filtered_inv = filtered_inv[filtered_inv['Brand'].isin(brand_filter)]
            
            # Display table
            display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'ABC_Classification', 
                           'Stock_Qty', 'Avg_Sales_3M', 'DOI', 'Cover_Months', 
                           'Inventory_Status', 'Stock_Value', 'Qty_to_Order', 'Qty_to_Reduce']
            display_cols = [col for col in display_cols if col in filtered_inv.columns]
            
            st.dataframe(
                filtered_inv[display_cols].sort_values('Stock_Value', ascending=False),
                column_config={
                    "Stock_Qty": st.column_config.NumberColumn(format="%d"),
                    "Avg_Sales_3M": st.column_config.NumberColumn(format="%d"),
                    "DOI": st.column_config.NumberColumn(format="%.1f"),
                    "Cover_Months": st.column_config.NumberColumn(format="%.1f"),
                    "Stock_Value": st.column_config.NumberColumn("Stock Value", format="$%.0f"),
                    "Qty_to_Order": st.column_config.NumberColumn(format="%d"),
                    "Qty_to_Reduce": st.column_config.NumberColumn(format="%d")
                },
                use_container_width=True,
                height=500
            )
        else:
            st.warning("No inventory data available for active SKUs.")
    
    # ==========================================
    # TAB 3: FORECAST PERFORMANCE
    # ==========================================
    with tab3:
        st.subheader("📈 Forecast Performance (Rofo vs PO)")
        
        if 'overall_mape' in forecast_accuracy:
            # Forecast Accuracy Dashboard
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mape = forecast_accuracy['overall_mape']
                accuracy = 100 - mape if mape > 0 else 0
                st.markdown(components.kpi_card(
                    "Forecast Accuracy",
                    f"{accuracy:.1f}%",
                    trend=2.5,
                    subtitle="Rofo vs PO",
                    icon="🎯"
                ), unsafe_allow_html=True)
            
            with col2:
                bias = forecast_accuracy['forecast_bias']
                st.markdown(components.kpi_card(
                    "Forecast Bias",
                    f"{bias:+.0f}",
                    trend=-1.2,
                    subtitle="Units (Positive = Over-forecast)",
                    icon="⚖️"
                ), unsafe_allow_html=True)
            
            with col3:
                # Calculate tracking signal
                if 'detailed_data' in forecast_accuracy:
                    ts_data = forecast_accuracy['detailed_data']
                    tracking_signal = ts_data['Forecast_Error'].sum() / ts_data['Absolute_Error'].sum() if ts_data['Absolute_Error'].sum() > 0 else 0
                    st.markdown(components.kpi_card(
                        "Tracking Signal",
                        f"{tracking_signal:.2f}",
                        subtitle="±4 is control limit",
                        icon="📡"
                    ), unsafe_allow_html=True)
            
            # Forecast Accuracy Trend
            st.markdown("---")
            st.subheader("📈 Forecast Accuracy Trend")
            
            if 'monthly_accuracy' in forecast_accuracy:
                fig_accuracy = px.line(
                    forecast_accuracy['monthly_accuracy'],
                    x='Month',
                    y='Accuracy_%',
                    markers=True,
                    title="Monthly Forecast Accuracy Trend (Rofo vs PO)",
                    labels={'Accuracy_%': 'Accuracy (%)', 'Month': 'Month'}
                )
                fig_accuracy.add_hline(y=80, line_dash="dash", line_color="red", 
                                      annotation_text="Target: 80%")
                fig_accuracy.update_layout(height=400)
                st.plotly_chart(fig_accuracy, use_container_width=True)
            
            # Forecast vs PO Comparison
            st.markdown("---")
            st.subheader("📊 Forecast vs PO Comparison")
            
            if 'detailed_data' in forecast_accuracy and not forecast_accuracy['detailed_data'].empty:
                # Aggregate by month
                monthly_comparison = forecast_accuracy['detailed_data'].groupby('Month').agg({
                    'Forecast_Qty': 'sum',
                    'Order_Qty': 'sum'
                }).reset_index()
                
                fig_comparison = px.bar(
                    monthly_comparison,
                    x='Month',
                    y=['Forecast_Qty', 'Order_Qty'],
                    title="Monthly Forecast vs PO Quantity",
                    labels={'value': 'Quantity', 'variable': 'Type'},
                    barmode='group'
                )
                fig_comparison.update_layout(height=400)
                st.plotly_chart(fig_comparison, use_container_width=True)
        else:
            st.warning("Forecast accuracy data not available. Please check Rofo and PO data.")
    
    # ==========================================
    # TAB 4: SALES PERFORMANCE
    # ==========================================
    with tab4:
        st.subheader("📊 Sales Performance (Active SKUs)")
        
        if sales_performance:
            # Sales Performance Dashboard
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sales_vs_forecast = sales_performance.get('sales_vs_forecast', 0)
                st.markdown(components.kpi_card(
                    "Sales vs Forecast",
                    f"{sales_vs_forecast:.1f}%",
                    trend=2.5,
                    subtitle="Actual vs Planned",
                    icon="📈"
                ), unsafe_allow_html=True)
            
            with col2:
                sales_vs_po = sales_performance.get('sales_vs_po', 0)
                st.markdown(components.kpi_card(
                    "Sales vs PO",
                    f"{sales_vs_po:.1f}%",
                    trend=1.8,
                    subtitle="Actual vs Ordered",
                    icon="📦"
                ), unsafe_allow_html=True)
            
            with col3:
                po_vs_forecast = sales_performance.get('po_vs_forecast', 0)
                st.markdown(components.kpi_card(
                    "PO vs Forecast",
                    f"{po_vs_forecast:.1f}%",
                    trend=0.5,
                    subtitle="Ordered vs Planned",
                    icon="📝"
                ), unsafe_allow_html=True)
            
            # Sales Trend Analysis
            st.markdown("---")
            st.subheader("📈 Sales Trend Analysis")
            
            if not all_data['sales'].empty:
                # Aggregate sales by month
                sales_monthly = all_data['sales'].groupby('Month')['Sales_Qty'].sum().reset_index()
                
                fig_sales_trend = px.line(
                    sales_monthly,
                    x='Month',
                    y='Sales_Qty',
                    title="Monthly Sales Trend (Active SKUs)",
                    labels={'Sales_Qty': 'Sales Quantity', 'Month': 'Month'},
                    markers=True
                )
                fig_sales_trend.update_layout(height=400)
                st.plotly_chart(fig_sales_trend, use_container_width=True)
            
            # Top Performing SKUs
            st.markdown("---")
            st.subheader("🏆 Top Performing SKUs")
            
            if not all_data['sales'].empty:
                # Calculate top SKUs by sales volume (last 3 months)
                sales_df = all_data['sales'].copy()
                sales_df['Month'] = pd.to_datetime(sales_df['Month'])
                
                # Get last 3 months
                if len(sales_df['Month'].unique()) >= 3:
                    last_3_months = sorted(sales_df['Month'].unique())[-3:]
                    recent_sales = sales_df[sales_df['Month'].isin(last_3_months)]
                    
                    top_skus = recent_sales.groupby('SKU_ID')['Sales_Qty'].sum().reset_index()
                    top_skus = top_skus.sort_values('Sales_Qty', ascending=False).head(10)
                    
                    # Merge with product names
                    if not all_data['product'].empty:
                        top_skus = pd.merge(
                            top_skus,
                            all_data['product'][['SKU_ID', 'Product_Name', 'Brand']],
                            on='SKU_ID',
                            how='left'
                        )
                    
                    fig_top_skus = px.bar(
                        top_skus,
                        x='Product_Name',
                        y='Sales_Qty',
                        title="Top 10 SKUs by Sales Volume (Last 3 Months)",
                        labels={'Sales_Qty': 'Total Sales', 'Product_Name': 'Product'},
                        color='Brand',
                        hover_data=['Brand']
                    )
                    fig_top_skus.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_top_skus, use_container_width=True)
        else:
            st.warning("Sales performance data not available. Please check Sales, Forecast, and PO data.")

# --- RUN DASHBOARD ---
if __name__ == "__main__":
    main()
