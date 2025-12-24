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
from scipy import stats
import calendar
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
        st.warning(f"Warning parsing date '{label}': {str(e)}")
        return datetime.now()

class SupplyChainDataEngine:
    """Enhanced data engine for complete supply chain data"""
    
    def __init__(self, client):
        self.client = client
        self.gsheet_url = "https://docs.google.com/spreadsheets/d/1gek6SPgJcZzhOjN-eNOZbfCvmX2EKiHKbgk3c0gUxL4"
        
    def load_all_data(self):
        """Load all supply chain data"""
        data = {}
        
        try:
            # Open the spreadsheet
            spreadsheet = self.client.open_by_url(self.gsheet_url)
            
            # 1. Product Master
            data['product'] = self._load_sheet(spreadsheet, "Product_Master")
            if not data['product'].empty:
                data['product']['SKU_ID'] = data['product']['SKU_ID'].astype(str).str.strip()
                # Calculate ABC classification if not exists or empty
                if 'ABC_Classification' not in data['product'].columns or data['product']['ABC_Classification'].isna().all():
                    data['product'] = self._calculate_abc_classification(data['product'])
            
            # 2. Sales Data
            data['sales'] = self._process_sales_data(spreadsheet, "Sales")
            
            # 3. Forecast Data (Rofo)
            data['forecast'] = self._process_monthly_data(spreadsheet, "Rofo", "Forecast_Qty")
            
            # 4. PO Data
            data['po'] = self._load_sheet(spreadsheet, "PO")
            if not data['po'].empty:
                # Convert date columns
                date_cols = ['Order_Date', 'Expected_Delivery_Date', 'Actual_Delivery_Date']
                for col in date_cols:
                    if col in data['po'].columns:
                        data['po'][col] = pd.to_datetime(data['po'][col], errors='coerce')
            
            # 5. Stock Data
            data['stock'] = self._load_sheet(spreadsheet, "Stock_Onhand")
            if not data['stock'].empty:
                data['stock']['SKU_ID'] = data['stock']['SKU_ID'].astype(str).str.strip()
                # Identify stock quantity column
                stock_qty_cols = ['Stock Qty', 'Quantity_Available', 'STOCK SAP']
                for col in stock_qty_cols:
                    if col in data['stock'].columns:
                        data['stock']['Stock_Qty'] = pd.to_numeric(data['stock'][col], errors='coerce').fillna(0)
                        break
            
            # 6. Supplier Master
            data['suppliers'] = self._load_sheet(spreadsheet, "Supplier_Master")
            
            # 7. Customer Master
            data['customers'] = self._load_sheet(spreadsheet, "Customer_Master")
            
            # 8. Transportation Data
            data['transportation'] = self._load_sheet(spreadsheet, "Transportation_Logs")
            if not data['transportation'].empty:
                date_cols = ['Shipment_Date', 'Delivery_Date']
                for col in date_cols:
                    if col in data['transportation'].columns:
                        data['transportation'][col] = pd.to_datetime(data['transportation'][col], errors='coerce')
            
            # 9. Warehouse Operations
            data['warehouse'] = self._load_sheet(spreadsheet, "Warehouse_Operations")
            if not data['warehouse'].empty and 'Activity_Date' in data['warehouse'].columns:
                data['warehouse']['Activity_Date'] = pd.to_datetime(data['warehouse']['Activity_Date'], errors='coerce')
            
            # 10. Demand Planning
            data['demand_planning'] = self._load_sheet(spreadsheet, "Demand_Planning")
            if not data['demand_planning'].empty and 'Month' in data['demand_planning'].columns:
                data['demand_planning']['Month'] = pd.to_datetime(data['demand_planning']['Month'], errors='coerce')
            
            # 11. KPI Targets
            data['kpi_targets'] = self._load_sheet(spreadsheet, "KPIs_Targets")
            
            # Active SKUs
            if not data['product'].empty and 'Status' in data['product'].columns:
                data['active_skus'] = data['product'][data['product']['Status'] == 'Active']['SKU_ID'].tolist()
            else:
                data['active_skus'] = []
            
            return data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _load_sheet(self, spreadsheet, sheet_name):
        """Load specific sheet"""
        try:
            ws = spreadsheet.worksheet(sheet_name)
            df = pd.DataFrame(ws.get_all_records())
            # Clean column names - handle spaces and special characters
            df.columns = [c.strip().replace(' ', '_').replace('%', 'Percent') for c in df.columns]
            return df
        except Exception as e:
            st.warning(f"⚠️ Sheet '{sheet_name}' not found or empty: {str(e)}")
            return pd.DataFrame()
    
    def _process_sales_data(self, spreadsheet, sheet_name):
        """Process sales data with specific column structure"""
        df = self._load_sheet(spreadsheet, sheet_name)
        if df.empty:
            return pd.DataFrame()
        
        # Identify month columns (Jan 25, Feb 25, etc.)
        month_cols = []
        for col in df.columns:
            col_str = str(col).strip()
            # Check if column looks like a month (contains month name)
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
    
    def _process_monthly_data(self, spreadsheet, sheet_name, value_col_name):
        """Process monthly data sheets (Rofo, etc.)"""
        df = self._load_sheet(spreadsheet, sheet_name)
        if df.empty:
            return pd.DataFrame()
        
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
    
    def _calculate_abc_classification(self, df_product):
        """Calculate ABC classification based on revenue potential"""
        if df_product.empty:
            df_product['ABC_Classification'] = 'C'
            return df_product
        
        # If we have Unit_Price and historical sales data, we could calculate properly
        # For now, assign based on SKU_Tier or random
        if 'SKU_Tier' in df_product.columns:
            conditions = [
                df_product['SKU_Tier'].isin(['A', 'Premium', 'High']),
                df_product['SKU_Tier'].isin(['B', 'Medium']),
                df_product['SKU_Tier'].isin(['C', 'Low', 'Basic'])
            ]
            choices = ['A', 'B', 'C']
        else:
            # Random assignment if no tier
            np.random.seed(42)
            choices = ['A', 'B', 'C']
            probs = [0.2, 0.3, 0.5]
            df_product['ABC_Classification'] = np.random.choice(choices, size=len(df_product), p=probs)
            return df_product
        
        df_product['ABC_Classification'] = np.select(conditions, choices, default='C')
        return df_product

# --- 2. SUPPLY CHAIN ANALYTICS ENGINE ---
class SupplyChainAnalytics:
    """Complete supply chain analytics engine"""
    
    def __init__(self, data):
        self.data = data
        
    def calculate_inventory_metrics(self):
        """Calculate comprehensive inventory metrics"""
        if self.data['stock'].empty or self.data['product'].empty:
            return pd.DataFrame()
        
        # Prepare stock data
        stock_df = self.data['stock'].copy()
        
        # Ensure we have SKU_ID and Stock_Qty
        if 'SKU_ID' not in stock_df.columns or 'Stock_Qty' not in stock_df.columns:
            st.warning("Stock data missing required columns")
            return pd.DataFrame()
        
        # Merge with product data
        product_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Category', 
                       'Unit_Cost', 'Min_Stock_Level', 'Max_Stock_Level', 'Lead_Time_Days']
        product_cols = [col for col in product_cols if col in self.data['product'].columns]
        
        inv_df = pd.merge(
            stock_df[['SKU_ID', 'Stock_Qty']],
            self.data['product'][product_cols],
            on='SKU_ID',
            how='left'
        )
        
        # Calculate sales metrics
        if not self.data['sales'].empty:
            sales_df = self.data['sales'].copy()
            sales_df['Month'] = pd.to_datetime(sales_df['Month'])
            
            # Get last 3 months
            if len(sales_df['Month'].unique()) >= 3:
                last_3_months = sorted(sales_df['Month'].unique())[-3:]
                recent_sales = sales_df[sales_df['Month'].isin(last_3_months)]
                
                avg_sales = recent_sales.groupby('SKU_ID')['Sales_Qty'].agg(['mean', 'std']).reset_index()
                avg_sales.columns = ['SKU_ID', 'Avg_Sales_3M', 'Sales_Std']
                avg_sales['Sales_CV'] = np.where(
                    avg_sales['Avg_Sales_3M'] > 0, 
                    avg_sales['Sales_Std'] / avg_sales['Avg_Sales_3M'], 
                    0
                )
                
                inv_df = pd.merge(inv_df, avg_sales, on='SKU_ID', how='left')
            else:
                # If less than 3 months, use all available
                avg_sales = sales_df.groupby('SKU_ID')['Sales_Qty'].agg(['mean', 'std']).reset_index()
                avg_sales.columns = ['SKU_ID', 'Avg_Sales_3M', 'Sales_Std']
                avg_sales['Sales_CV'] = np.where(
                    avg_sales['Avg_Sales_3M'] > 0, 
                    avg_sales['Sales_Std'] / avg_sales['Avg_Sales_3M'], 
                    0
                )
                inv_df = pd.merge(inv_df, avg_sales, on='SKU_ID', how='left')
        else:
            inv_df['Avg_Sales_3M'] = 0
            inv_df['Sales_CV'] = 0
            inv_df['Sales_Std'] = 0
        
        # Calculate inventory metrics
        inv_df['Avg_Sales_3M'] = inv_df['Avg_Sales_3M'].fillna(0)
        inv_df['Sales_Std'] = inv_df['Sales_Std'].fillna(0)
        
        inv_df['Cover_Months'] = np.where(
            inv_df['Avg_Sales_3M'] > 0,
            inv_df['Stock_Qty'] / inv_df['Avg_Sales_3M'],
            999
        ).round(2)
        
        # Inventory classification
        conditions = [
            inv_df['Cover_Months'] < 1.0,
            (inv_df['Cover_Months'] >= 1.0) & (inv_df['Cover_Months'] <= 1.5),
            inv_df['Cover_Months'] > 1.5
        ]
        choices = ['Need Replenishment', 'Ideal', 'High Stock']
        inv_df['Inventory_Status'] = np.select(conditions, choices, default='Unknown')
        
        # Stock value
        if 'Unit_Cost' in inv_df.columns:
            inv_df['Stock_Value'] = inv_df['Stock_Qty'] * inv_df['Unit_Cost'].fillna(0)
        else:
            inv_df['Stock_Value'] = 0
        
        # Calculate EOQ (Economic Order Quantity)
        if 'Unit_Cost' in inv_df.columns:
            # D = annual demand (Avg_Sales_3M * 4)
            # S = ordering cost (assume 100 for demo)
            # H = holding cost (assume 25% of unit cost)
            D = inv_df['Avg_Sales_3M'] * 4
            S = 100  # Fixed ordering cost
            H = inv_df['Unit_Cost'].fillna(0) * 0.25  # Holding cost per unit
            
            inv_df['EOQ'] = np.where(
                (D > 0) & (H > 0),
                np.sqrt((2 * D * S) / H),
                0
            ).round(0)
            
            # Calculate reorder point
            if 'Lead_Time_Days' in inv_df.columns:
                # Daily demand
                daily_demand = inv_df['Avg_Sales_3M'] / 30
                # Safety stock (95% service level)
                z_score = 1.65
                lead_time_days = inv_df['Lead_Time_Days'].fillna(30)
                safety_stock = z_score * inv_df['Sales_Std'] * np.sqrt(lead_time_days / 30)
                
                inv_df['Reorder_Point'] = (daily_demand * lead_time_days + safety_stock).round(0)
                inv_df['Safety_Stock'] = safety_stock.round(0)
        
        return inv_df
    
    def calculate_monthly_performance(self):
        """Calculate monthly forecast performance"""
        if self.data['forecast'].empty or self.data['po'].empty:
            return {}
        
        # We need to restructure PO data to compare with forecast
        # For now, return empty dict and we'll handle it differently
        return {}
    
    def calculate_supplier_performance(self):
        """Calculate supplier performance metrics"""
        if self.data['suppliers'].empty:
            return pd.DataFrame()
        
        supplier_metrics = self.data['suppliers'].copy()
        
        # Clean numeric columns
        numeric_cols = ['Performance_Score', 'Lead_Time_Avg', 'On_Time_Delivery_Percent', 'Quality_Score']
        for col in numeric_cols:
            if col in supplier_metrics.columns:
                supplier_metrics[col] = pd.to_numeric(supplier_metrics[col], errors='coerce')
        
        # Calculate OTD Score
        if 'On_Time_Delivery_Percent' in supplier_metrics.columns:
            supplier_metrics['OTD_Score'] = pd.cut(
                supplier_metrics['On_Time_Delivery_Percent'],
                bins=[0, 85, 95, 100],
                labels=['Poor', 'Good', 'Excellent']
            )
        
        # Calculate Quality Score Category
        if 'Quality_Score' in supplier_metrics.columns:
            supplier_metrics['Quality_Score_Category'] = pd.cut(
                supplier_metrics['Quality_Score'],
                bins=[0, 80, 90, 100],
                labels=['Needs Improvement', 'Acceptable', 'Excellent']
            )
        
        return supplier_metrics
    
    def calculate_logistics_metrics(self):
        """Calculate logistics and transportation metrics"""
        if self.data['transportation'].empty:
            return pd.DataFrame()
        
        trans_df = self.data['transportation'].copy()
        
        # Clean date columns
        date_cols = ['Shipment_Date', 'Delivery_Date']
        for col in date_cols:
            if col in trans_df.columns:
                trans_df[col] = pd.to_datetime(trans_df[col], errors='coerce')
        
        # Calculate on-time delivery
        if 'Delivery_Date' in trans_df.columns and 'Shipment_Date' in trans_df.columns:
            # Calculate transit time
            trans_df['Transit_Time_Actual'] = (trans_df['Delivery_Date'] - trans_df['Shipment_Date']).dt.days
            
            # Compare with expected if available
            if 'Transit_Time_Days' in trans_df.columns:
                trans_df['Transit_Time_Days'] = pd.to_numeric(trans_df['Transit_Time_Days'], errors='coerce')
                trans_df['Days_Late'] = trans_df['Transit_Time_Actual'] - trans_df['Transit_Time_Days']
                trans_df['On_Time'] = trans_df['Days_Late'] <= 0
        
        # Carrier performance
        if 'Carrier' in trans_df.columns:
            carrier_perf = trans_df.groupby('Carrier').agg({
                'Shipment_ID': 'count',
                'Days_Late': 'mean',
                'Freight_Cost': 'mean',
                'Transit_Time_Actual': 'mean'
            }).reset_index()
            
            carrier_perf.columns = ['Carrier', 'Total_Shipments', 'Avg_Days_Late', 
                                   'Avg_Freight_Cost', 'Avg_Transit_Time']
            
            # Calculate on-time rate
            if 'On_Time' in trans_df.columns:
                on_time_rate = trans_df.groupby('Carrier')['On_Time'].mean() * 100
                carrier_perf['On_Time_Rate'] = carrier_perf['Carrier'].map(on_time_rate)
        
        return carrier_perf if 'Carrier' in trans_df.columns else pd.DataFrame()
    
    def calculate_warehouse_metrics(self):
        """Calculate warehouse performance metrics"""
        if self.data['warehouse'].empty:
            return pd.DataFrame()
        
        warehouse_df = self.data['warehouse'].copy()
        
        # Clean date
        if 'Activity_Date' in warehouse_df.columns:
            warehouse_df['Activity_Date'] = pd.to_datetime(warehouse_df['Activity_Date'], errors='coerce')
            warehouse_df['Month'] = warehouse_df['Activity_Date'].dt.to_period('M')
        
        # Clean numeric columns
        numeric_cols = ['Quantity', 'Processing_Time_Min', 'Damage_Qty']
        for col in numeric_cols:
            if col in warehouse_df.columns:
                warehouse_df[col] = pd.to_numeric(warehouse_df[col], errors='coerce')
        
        # Calculate productivity
        if 'Processing_Time_Min' in warehouse_df.columns and 'Quantity' in warehouse_df.columns:
            # Units per hour
            warehouse_df['Units_per_Hour'] = np.where(
                warehouse_df['Processing_Time_Min'] > 0,
                (warehouse_df['Quantity'] / warehouse_df['Processing_Time_Min']) * 60,
                0
            )
        
        return warehouse_df
    
    def calculate_demand_planning_accuracy(self):
        """Calculate forecast accuracy metrics"""
        if self.data['sales'].empty or self.data['forecast'].empty:
            return {}
        
        # Merge sales and forecast
        df_merged = pd.merge(
            self.data['sales'],
            self.data['forecast'],
            on=['SKU_ID', 'Month'],
            how='inner',
            suffixes=('_Sales', '_Forecast')
        )
        
        if df_merged.empty:
            return {}
        
        # Calculate forecast error
        df_merged['Forecast_Error'] = df_merged['Sales_Qty'] - df_merged['Forecast_Qty']
        df_merged['Absolute_Error'] = abs(df_merged['Forecast_Error'])
        df_merged['Absolute_Percentage_Error'] = np.where(
            df_merged['Sales_Qty'] > 0,
            (df_merged['Absolute_Error'] / df_merged['Sales_Qty']) * 100,
            0
        )
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = df_merged['Absolute_Percentage_Error'].mean()
        
        # Calculate Forecast Bias
        forecast_bias = df_merged['Forecast_Error'].mean()
        
        # Calculate by month
        monthly_accuracy = df_merged.groupby('Month').agg({
            'Sales_Qty': 'sum',
            'Forecast_Qty': 'sum',
            'Absolute_Percentage_Error': 'mean'
        }).reset_index()
        
        monthly_accuracy['Accuracy_%'] = 100 - monthly_accuracy['Absolute_Percentage_Error']
        
        return {
            'overall_mape': mape,
            'forecast_bias': forecast_bias,
            'monthly_accuracy': monthly_accuracy,
            'detailed_data': df_merged
        }
    
    def calculate_scor_metrics(self):
        """Calculate SCOR (Supply Chain Operations Reference) metrics"""
        metrics = {}
        
        # 1. Reliability Metrics
        if not self.data['po'].empty and 'PO_Status' in self.data['po'].columns:
            perfect_orders = len(self.data['po'][self.data['po']['PO_Status'] == 'Closed'])
            total_orders = len(self.data['po'])
            metrics['Perfect_Order_Fulfillment'] = (perfect_orders / total_orders * 100) if total_orders > 0 else 0
        
        # 2. Responsiveness Metrics
        if 'Lead_Time_Days' in self.data['product'].columns:
            avg_lead_time = self.data['product']['Lead_Time_Days'].mean()
            metrics['Average_Lead_Time'] = avg_lead_time
        
        # 3. Agility Metrics
        if not self.data['sales'].empty:
            sales_std = self.data['sales']['Sales_Qty'].std()
            sales_mean = self.data['sales']['Sales_Qty'].mean()
            metrics['Demand_Variability_CV'] = (sales_std / sales_mean) if sales_mean > 0 else 0
        
        # 4. Costs Metrics - Calculate from inventory metrics
        inv_metrics = self.calculate_inventory_metrics()
        if not inv_metrics.empty and 'Stock_Value' in inv_metrics.columns:
            total_inv_value = inv_metrics['Stock_Value'].sum()
            metrics['Total_Inventory_Value'] = total_inv_value
        
        # 5. Asset Management Metrics
        if 'Total_Inventory_Value' in metrics and not self.data['sales'].empty:
            # Simplified inventory turnover calculation
            # This would need cost of goods sold data for accurate calculation
            metrics['Inventory_Turnover'] = 12  # Placeholder
        
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
    with st.spinner("📥 Loading supply chain data..."):
        data_engine = SupplyChainDataEngine(client)
        all_data = data_engine.load_all_data()
    
    if not all_data:
        st.error("Failed to load data. Please check data structure.")
        st.stop()
    
    # Display data status
    with st.expander("📊 Data Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Products", len(all_data.get('product', pd.DataFrame())))
            st.metric("Active SKUs", len(all_data.get('active_skus', [])))
        
        with col2:
            st.metric("Sales Records", len(all_data.get('sales', pd.DataFrame())))
            st.metric("PO Records", len(all_data.get('po', pd.DataFrame())))
        
        with col3:
            st.metric("Suppliers", len(all_data.get('suppliers', pd.DataFrame())))
            st.metric("Customers", len(all_data.get('customers', pd.DataFrame())))
    
    # Initialize analytics
    analytics = SupplyChainAnalytics(all_data)
    
    # Calculate metrics
    with st.spinner("📊 Calculating metrics..."):
        inv_metrics = analytics.calculate_inventory_metrics()
        supplier_metrics = analytics.calculate_supplier_performance()
        logistics_metrics = analytics.calculate_logistics_metrics()
        warehouse_metrics = analytics.calculate_warehouse_metrics()
        demand_accuracy = analytics.calculate_demand_planning_accuracy()
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
        bu_options = ["All", "FMCG", "Electronics", "Pharmaceutical", "Retail"]
        business_unit = st.selectbox("Select Business Unit", bu_options)
        
        # View mode
        st.subheader("👁️ View Mode")
        view_mode = st.radio(
            "Select View",
            ["Executive Summary", "Detailed Analysis", "Operational View"]
        )
        
        # Refresh button
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Main dashboard
    components = DashboardComponents()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Executive Dashboard",
        "📦 Inventory Management",
        "🚚 Logistics & Procurement",
        "📈 Demand Planning",
        "🔧 Operations"
    ])
    
    # ==========================================
    # TAB 1: EXECUTIVE DASHBOARD
    # ==========================================
    with tab1:
        st.subheader("🎯 Executive Supply Chain Dashboard")
        
        # Key Metrics Row 1
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Inventory Turnover
            inv_turnover = scor_metrics.get('Inventory_Turnover', 0)
            st.markdown(components.kpi_card(
                "Inventory Turnover", 
                f"{inv_turnover:.1f}x",
                trend=5.2,
                subtitle="Last 12 Months",
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
            mape = demand_accuracy.get('overall_mape', 0)
            forecast_accuracy = 100 - mape if mape > 0 else 0
            st.markdown(components.kpi_card(
                "Forecast Accuracy",
                f"{forecast_accuracy:.1f}%",
                trend=3.5,
                subtitle="Mean Absolute % Error",
                icon="🎯"
            ), unsafe_allow_html=True)
        
        with col4:
            # Cash-to-Cash Cycle Time
            c2c_cycle = 45  # Placeholder
            st.markdown(components.kpi_card(
                "Cash-to-Cash Cycle",
                f"{c2c_cycle} days",
                trend=-2.3,
                subtitle="Days of Working Capital",
                icon="💰"
            ), unsafe_allow_html=True)
        
        # Inventory Overview
        st.markdown("---")
        st.subheader("📦 Inventory Overview")
        
        if not inv_metrics.empty:
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                total_skus = len(inv_metrics)
                st.markdown(components.summary_card(
                    "Total SKUs",
                    total_skus,
                    "Active Products",
                    "bg-primary"
                ), unsafe_allow_html=True)
            
            with col6:
                need_replenish = len(inv_metrics[inv_metrics['Inventory_Status'] == 'Need Replenishment'])
                st.markdown(components.summary_card(
                    "Need Replenishment",
                    need_replenish,
                    "< 1 month cover",
                    "bg-danger"
                ), unsafe_allow_html=True)
            
            with col7:
                ideal_stock = len(inv_metrics[inv_metrics['Inventory_Status'] == 'Ideal'])
                st.markdown(components.summary_card(
                    "Ideal Stock",
                    ideal_stock,
                    "1-1.5 months cover",
                    "bg-success"
                ), unsafe_allow_html=True)
            
            with col8:
                high_stock = len(inv_metrics[inv_metrics['Inventory_Status'] == 'High Stock'])
                st.markdown(components.summary_card(
                    "High Stock",
                    high_stock,
                    "> 1.5 months cover",
                    "bg-warning"
                ), unsafe_allow_html=True)
        
        # Financial Impact
        st.markdown("---")
        st.subheader("💰 Financial Impact")
        
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            total_inv_value = scor_metrics.get('Total_Inventory_Value', 0)
            st.markdown(components.summary_card(
                "Total Inventory Value",
                f"${total_inv_value:,.0f}",
                "Current Stock Value",
                "bg-info"
            ), unsafe_allow_html=True)
        
        with col10:
            if not inv_metrics.empty:
                high_stock_value = inv_metrics[inv_metrics['Inventory_Status'] == 'High Stock']['Stock_Value'].sum()
                st.markdown(components.summary_card(
                    "Excess Inventory",
                    f"${high_stock_value:,.0f}",
                    "Potential Savings",
                    "bg-warning"
                ), unsafe_allow_html=True)
        
        with col11:
            if not all_data['transportation'].empty and 'Freight_Cost' in all_data['transportation'].columns:
                total_freight = all_data['transportation']['Freight_Cost'].sum()
                st.markdown(components.summary_card(
                    "Freight Cost",
                    f"${total_freight:,.0f}",
                    "Total Cost",
                    "bg-purple"
                ), unsafe_allow_html=True)
        
        with col12:
            stockout_cost = 125000  # Placeholder
            st.markdown(components.summary_card(
                "Stockout Cost",
                f"${stockout_cost:,.0f}",
                "Estimated Annual Impact",
                "bg-danger"
            ), unsafe_allow_html=True)
    
    # ==========================================
    # TAB 2: INVENTORY MANAGEMENT
    # ==========================================
    with tab2:
        st.subheader("📦 Advanced Inventory Analytics")
        
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
                fig_cover.add_vline(x=1.0, line_dash="dash", line_color="red", 
                                   annotation_text="Min (1 month)")
                fig_cover.add_vline(x=1.5, line_dash="dash", line_color="orange", 
                                   annotation_text="Max (1.5 months)")
                fig_cover.update_layout(height=400)
                st.plotly_chart(fig_cover, use_container_width=True)
            
            # ABC Analysis
            st.markdown("---")
            st.subheader("📊 ABC Analysis")
            
            if 'ABC_Classification' in inv_metrics.columns:
                abc_data = inv_metrics.groupby('ABC_Classification').agg({
                    'SKU_ID': 'count',
                    'Stock_Value': 'sum'
                }).reset_index()
                
                abc_col1, abc_col2 = st.columns(2)
                
                with abc_col1:
                    fig_abc_count = px.bar(
                        abc_data,
                        x='ABC_Classification',
                        y='SKU_ID',
                        title="SKU Count by ABC Classification",
                        labels={'SKU_ID': 'Number of SKUs', 'ABC_Classification': 'Classification'},
                        color='ABC_Classification',
                        color_discrete_map={'A': '#e55039', 'B': '#f6b93b', 'C': '#38ada9'}
                    )
                    fig_abc_count.update_layout(height=350)
                    st.plotly_chart(fig_abc_count, use_container_width=True)
                
                with abc_col2:
                    fig_abc_value = px.pie(
                        abc_data,
                        values='Stock_Value',
                        names='ABC_Classification',
                        title="Inventory Value by ABC Classification",
                        color='ABC_Classification',
                        color_discrete_map={'A': '#e55039', 'B': '#f6b93b', 'C': '#38ada9'}
                    )
                    fig_abc_value.update_traces(textposition='inside', textinfo='percent+label')
                    fig_abc_value.update_layout(height=350)
                    st.plotly_chart(fig_abc_value, use_container_width=True)
            
            # Detailed Inventory Table
            st.markdown("---")
            st.subheader("📋 Detailed Inventory Analysis")
            
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
                           'Stock_Qty', 'Avg_Sales_3M', 'Cover_Months', 
                           'Inventory_Status', 'Stock_Value']
            display_cols = [col for col in display_cols if col in filtered_inv.columns]
            
            st.dataframe(
                filtered_inv[display_cols].sort_values('Stock_Value', ascending=False),
                column_config={
                    "Stock_Qty": st.column_config.NumberColumn(format="%d"),
                    "Avg_Sales_3M": st.column_config.NumberColumn(format="%d"),
                    "Cover_Months": st.column_config.NumberColumn(format="%.1f"),
                    "Stock_Value": st.column_config.NumberColumn("Stock Value", format="$%.0f")
                },
                use_container_width=True,
                height=500
            )
        else:
            st.warning("No inventory data available. Please check Stock_Onhand sheet.")
    
    # ==========================================
    # TAB 3: LOGISTICS & PROCUREMENT
    # ==========================================
    with tab3:
        st.subheader("🚚 Logistics & Procurement Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Supplier Performance
            st.subheader("🏭 Supplier Performance")
            
            if not supplier_metrics.empty:
                # Top suppliers
                if 'Performance_Score' in supplier_metrics.columns:
                    top_suppliers = supplier_metrics.nlargest(10, 'Performance_Score')
                    
                    fig_suppliers = px.bar(
                        top_suppliers,
                        x='Supplier_Name',
                        y='Performance_Score',
                        title="Top 10 Suppliers by Performance Score",
                        labels={'Performance_Score': 'Score', 'Supplier_Name': 'Supplier'},
                        color='Performance_Score',
                        color_continuous_scale='viridis'
                    )
                    fig_suppliers.update_layout(height=400)
                    st.plotly_chart(fig_suppliers, use_container_width=True)
        
        with col2:
            # Transportation Analytics
            st.subheader("🚛 Transportation Analytics")
            
            if not logistics_metrics.empty:
                # Carrier Performance
                fig_carrier = px.bar(
                    logistics_metrics,
                    x='Carrier',
                    y='On_Time_Rate',
                    title="Carrier On-Time Delivery Rate",
                    labels={'On_Time_Rate': 'On-Time Rate (%)', 'Carrier': 'Carrier'},
                    color='On_Time_Rate',
                    color_continuous_scale='plasma'
                )
                fig_carrier.update_layout(height=400)
                st.plotly_chart(fig_carrier, use_container_width=True)
        
        # Procurement Analytics
        st.markdown("---")
        st.subheader("📋 Procurement Analytics")
        
        if not all_data['po'].empty:
            col_po1, col_po2 = st.columns(2)
            
            with col_po1:
                # PO Status Distribution
                if 'PO_Status' in all_data['po'].columns:
                    po_status = all_data['po']['PO_Status'].value_counts().reset_index()
                    po_status.columns = ['Status', 'Count']
                    
                    fig_po = px.pie(
                        po_status,
                        values='Count',
                        names='Status',
                        title="Purchase Order Status Distribution",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_po.update_traces(textposition='inside', textinfo='percent+label')
                    fig_po.update_layout(height=300)
                    st.plotly_chart(fig_po, use_container_width=True)
            
            with col_po2:
                # PO Value by Month
                if 'Order_Date' in all_data['po'].columns and 'Total_Value' in all_data['po'].columns:
                    all_data['po']['Order_Month'] = pd.to_datetime(all_data['po']['Order_Date']).dt.to_period('M')
                    po_monthly = all_data['po'].groupby('Order_Month')['Total_Value'].sum().reset_index()
                    po_monthly['Order_Month'] = po_monthly['Order_Month'].astype(str)
                    
                    fig_po_value = px.line(
                        po_monthly,
                        x='Order_Month',
                        y='Total_Value',
                        title="Monthly PO Value Trend",
                        labels={'Total_Value': 'Total Value ($)', 'Order_Month': 'Month'},
                        markers=True
                    )
                    fig_po_value.update_layout(height=300)
                    st.plotly_chart(fig_po_value, use_container_width=True)
    
    # ==========================================
    # TAB 4: DEMAND PLANNING
    # ==========================================
    with tab4:
        st.subheader("📈 Demand Planning Analytics")
        
        if 'overall_mape' in demand_accuracy:
            # Forecast Accuracy Dashboard
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mape = demand_accuracy['overall_mape']
                accuracy = 100 - mape
                st.markdown(components.kpi_card(
                    "Forecast Accuracy",
                    f"{accuracy:.1f}%",
                    trend=2.5,
                    subtitle="Overall MAPE",
                    icon="🎯"
                ), unsafe_allow_html=True)
            
            with col2:
                bias = demand_accuracy['forecast_bias']
                st.markdown(components.kpi_card(
                    "Forecast Bias",
                    f"{bias:+.0f}",
                    trend=-1.2,
                    subtitle="Units (Positive = Over-forecast)",
                    icon="⚖️"
                ), unsafe_allow_html=True)
            
            with col3:
                # Calculate tracking signal
                if 'detailed_data' in demand_accuracy:
                    ts_data = demand_accuracy['detailed_data']
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
            
            if 'monthly_accuracy' in demand_accuracy:
                fig_accuracy = px.line(
                    demand_accuracy['monthly_accuracy'],
                    x='Month',
                    y='Accuracy_%',
                    markers=True,
                    title="Monthly Forecast Accuracy Trend",
                    labels={'Accuracy_%': 'Accuracy (%)', 'Month': 'Month'}
                )
                fig_accuracy.add_hline(y=80, line_dash="dash", line_color="red", 
                                      annotation_text="Target: 80%")
                fig_accuracy.update_layout(height=400)
                st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # Sales Trend Analysis
        st.markdown("---")
        st.subheader("📊 Sales Trend Analysis")
        
        if not all_data['sales'].empty:
            # Aggregate sales by month
            sales_monthly = all_data['sales'].groupby('Month')['Sales_Qty'].sum().reset_index()
            
            fig_sales_trend = px.line(
                sales_monthly,
                x='Month',
                y='Sales_Qty',
                title="Monthly Sales Trend",
                labels={'Sales_Qty': 'Sales Quantity', 'Month': 'Month'},
                markers=True
            )
            fig_sales_trend.update_layout(height=400)
            st.plotly_chart(fig_sales_trend, use_container_width=True)
    
    # ==========================================
    # TAB 5: OPERATIONS
    # ==========================================
    with tab5:
        st.subheader("🔧 Operations Dashboard")
        
        # Warehouse Operations
        st.markdown("### 🏭 Warehouse Operations")
        
        if not warehouse_metrics.empty:
            col_w1, col_w2 = st.columns(2)
            
            with col_w1:
                # Activity Distribution
                if 'Activity_Type' in warehouse_metrics.columns:
                    activity_dist = warehouse_metrics['Activity_Type'].value_counts().reset_index()
                    activity_dist.columns = ['Activity', 'Count']
                    
                    fig_activity = px.bar(
                        activity_dist,
                        x='Activity',
                        y='Count',
                        title="Warehouse Activity Distribution",
                        labels={'Count': 'Number of Activities', 'Activity': 'Activity Type'},
                        color='Count',
                        color_continuous_scale='viridis'
                    )
                    fig_activity.update_layout(height=400)
                    st.plotly_chart(fig_activity, use_container_width=True)
            
            with col_w2:
                # Monthly Activity Trend
                if 'Month' in warehouse_metrics.columns:
                    monthly_activity = warehouse_metrics.groupby('Month').size().reset_index()
                    monthly_activity.columns = ['Month', 'Activity_Count']
                    monthly_activity['Month'] = monthly_activity['Month'].astype(str)
                    
                    fig_monthly_act = px.line(
                        monthly_activity,
                        x='Month',
                        y='Activity_Count',
                        title="Monthly Warehouse Activity Trend",
                        labels={'Activity_Count': 'Number of Activities', 'Month': 'Month'},
                        markers=True
                    )
                    fig_monthly_act.update_layout(height=400)
                    st.plotly_chart(fig_monthly_act, use_container_width=True)
        
        # Data Quality Check
        st.markdown("---")
        st.subheader("📋 Data Quality Report")
        
        quality_data = []
        
        # Check each dataset
        datasets = [
            ('Product Master', all_data.get('product', pd.DataFrame())),
            ('Sales', all_data.get('sales', pd.DataFrame())),
            ('Forecast', all_data.get('forecast', pd.DataFrame())),
            ('Stock', all_data.get('stock', pd.DataFrame())),
            ('PO', all_data.get('po', pd.DataFrame()))
        ]
        
        for name, df in datasets:
            if not df.empty:
                total_rows = len(df)
                missing_sku = df['SKU_ID'].isna().sum() if 'SKU_ID' in df.columns else 0
                quality_score = ((total_rows - missing_sku) / total_rows * 100) if total_rows > 0 else 0
                
                quality_data.append({
                    'Dataset': name,
                    'Total Rows': total_rows,
                    'Missing SKU IDs': missing_sku,
                    'Quality Score %': quality_score
                })
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            st.dataframe(quality_df, use_container_width=True)

# --- RUN DASHBOARD ---
if __name__ == "__main__":
    main()
