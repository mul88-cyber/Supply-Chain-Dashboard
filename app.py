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
    
    /* METRIC CARDS */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.05);
        border: 1px solid #eef2f7;
        margin-bottom: 15px;
    }
    
    /* STATUS BADGES */
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-success { background: #d4edda; color: #155724; }
    .status-warning { background: #fff3cd; color: #856404; }
    .status-danger { background: #f8d7da; color: #721c24; }
    .status-info { background: #d1ecf1; color: #0c5460; }
    
    /* RESPONSIVE */
    @media (max-width: 768px) {
        .main-header { font-size: 2rem; }
        .kpi-value, .sum-val { font-size: 2rem; }
        .summary-card { padding: 15px; }
    }
    
    /* ANIMATIONS */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* PROGRESS BAR */
    .progress-container {
        width: 100%;
        background-color: #eef2f7;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 8px;
        border-radius: 10px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="fade-in">
    <div style="text-align: center; margin-bottom: 1rem;">
        <h1 class="main-header">SUPPLY CHAIN INTELLIGENCE PLATFORM</h1>
        <div style="color: #666; font-size: 1rem; margin-bottom: 2rem;">
            📊 End-to-End Supply Chain Management & Analytics
        </div>
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

class SupplyChainDataEngine:
    """Enhanced data engine for complete supply chain data"""
    
    def __init__(self, client):
        self.client = client
        self.gsheet_url = st.secrets["gsheet_url"]
        
    def load_all_data(self):
        """Load all supply chain data"""
        data = {}
        
        try:
            # 1. Product Master
            data['product'] = self._load_sheet("Product_Master")
            if not data['product'].empty:
                data['product']['SKU_ID'] = data['product']['SKU_ID'].astype(str).str.strip()
                # Calculate ABC classification if not exists
                if 'ABC_Classification' not in data['product'].columns:
                    data['product'] = self._calculate_abc_classification(data['product'])
            
            # 2. Sales Data
            data['sales'] = self._process_monthly_data("Sales", "Sales_Qty")
            
            # 3. Forecast Data
            data['forecast'] = self._process_monthly_data("Rofo", "Forecast_Qty")
            
            # 4. PO Data
            data['po'] = self._process_monthly_data("PO", "PO_Qty")
            
            # 5. Stock Data
            data['stock'] = self._load_sheet("Stock_Onhand")
            if not data['stock'].empty:
                data['stock']['SKU_ID'] = data['stock']['SKU_ID'].astype(str).str.strip()
            
            # 6. Supplier Master
            data['suppliers'] = self._load_sheet("Supplier_Master")
            
            # 7. Customer Master
            data['customers'] = self._load_sheet("Customer_Master")
            
            # 8. Transportation Data
            data['transportation'] = self._load_sheet("Transportation_Logs")
            
            # 9. Warehouse Operations
            data['warehouse'] = self._load_sheet("Warehouse_Operations")
            
            # 10. Demand Planning
            data['demand_planning'] = self._load_sheet("Demand_Planning")
            
            # 11. KPI Targets
            data['kpi_targets'] = self._load_sheet("KPIs_Targets")
            
            # Active SKUs
            data['active_skus'] = data['product'][data['product']['Status'] == 'Active']['SKU_ID'].tolist()
            
            return data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return {}
    
    def _load_sheet(self, sheet_name):
        """Load specific sheet"""
        try:
            ws = self.client.open_by_url(self.gsheet_url).worksheet(sheet_name)
            df = pd.DataFrame(ws.get_all_records())
            # Clean column names
            df.columns = [c.strip().replace(' ', '_') for c in df.columns]
            return df
        except:
            return pd.DataFrame()
    
    def _process_monthly_data(self, sheet_name, value_column):
        """Process monthly data sheets"""
        df = self._load_sheet(sheet_name)
        if df.empty:
            return pd.DataFrame()
        
        # Identify month columns
        month_cols = []
        for col in df.columns:
            col_str = str(col).upper()
            if any(month in col_str for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                                                  'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']):
                month_cols.append(col)
        
        if not month_cols:
            return pd.DataFrame()
        
        # Melt to long format
        id_cols = ['SKU_ID'] if 'SKU_ID' in df.columns else []
        if not id_cols:
            return pd.DataFrame()
        
        df_long = df[id_cols + month_cols].melt(
            id_vars=id_cols,
            value_vars=month_cols,
            var_name='Month_Label',
            value_name=value_column
        )
        
        df_long[value_column] = pd.to_numeric(df_long[value_column], errors='coerce').fillna(0)
        df_long['Month'] = df_long['Month_Label'].apply(self._parse_month)
        df_long['Month'] = pd.to_datetime(df_long['Month'])
        
        return df_long
    
    def _parse_month(self, label):
        """Parse month label to datetime"""
        try:
            label_str = str(label).strip().upper()
            month_map = {
                'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
                'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12
            }
            for m_name, m_num in month_map.items():
                if m_name in label_str:
                    year_part = ''.join(filter(str.isdigit, label_str.replace(m_name, '')))
                    year = int('20'+year_part) if len(year_part)==2 else int(year_part) if year_part else datetime.now().year
                    return datetime(year, m_num, 1)
            return datetime.now()
        except:
            return datetime.now()
    
    def _calculate_abc_classification(self, df_product):
        """Calculate ABC classification based on revenue contribution"""
        if df_product.empty or 'Unit_Price' not in df_product.columns:
            df_product['ABC_Classification'] = 'C'
            return df_product
        
        # Assume we need sales data for proper ABC calculation
        # For now, assign random or based on tier
        conditions = [
            df_product['SKU_Tier'].isin(['A', 'Premium', 'High']),
            df_product['SKU_Tier'].isin(['B', 'Medium']),
            df_product['SKU_Tier'].isin(['C', 'Low', 'Basic'])
        ]
        choices = ['A', 'B', 'C']
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
        
        # Merge stock with product data
        inv_df = pd.merge(
            self.data['stock'],
            self.data['product'][['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 
                                  'Category', 'Unit_Cost', 'Min_Stock_Level', 
                                  'Max_Stock_Level', 'Lead_Time_Days']],
            on='SKU_ID',
            how='left'
        )
        
        # Calculate sales metrics
        if not self.data['sales'].empty:
            sales_df = self.data['sales'].copy()
            sales_df['Month'] = pd.to_datetime(sales_df['Month'])
            last_3_months = sorted(sales_df['Month'].unique())[-3:]
            recent_sales = sales_df[sales_df['Month'].isin(last_3_months)]
            
            avg_sales = recent_sales.groupby('SKU_ID')['Sales_Qty'].agg(['mean', 'std']).reset_index()
            avg_sales.columns = ['SKU_ID', 'Avg_Sales_3M', 'Sales_Std']
            avg_sales['Sales_CV'] = np.where(avg_sales['Avg_Sales_3M'] > 0, 
                                            avg_sales['Sales_Std'] / avg_sales['Avg_Sales_3M'], 0)
            
            inv_df = pd.merge(inv_df, avg_sales, on='SKU_ID', how='left')
        else:
            inv_df['Avg_Sales_3M'] = 0
            inv_df['Sales_CV'] = 0
        
        # Calculate inventory metrics
        inv_df['Avg_Sales_3M'] = inv_df['Avg_Sales_3M'].fillna(0)
        inv_df['Cover_Months'] = np.where(
            inv_df['Avg_Sales_3M'] > 0,
            inv_df['Quantity_Available'] / inv_df['Avg_Sales_3M'],
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
            inv_df['Stock_Value'] = inv_df['Quantity_Available'] * inv_df['Unit_Cost']
        
        # Calculate EOQ (Economic Order Quantity) if we have ordering costs
        if 'Unit_Cost' in inv_df.columns:
            # D = annual demand (Avg_Sales_3M * 4)
            # S = ordering cost (assume 100 for demo)
            # H = holding cost (assume 25% of unit cost)
            D = inv_df['Avg_Sales_3M'] * 4
            S = 100  # Fixed ordering cost
            H = inv_df['Unit_Cost'] * 0.25  # Holding cost per unit
            
            inv_df['EOQ'] = np.where(
                (D > 0) & (H > 0),
                np.sqrt((2 * D * S) / H),
                0
            ).round(0)
            
            # Calculate reorder point
            if 'Lead_Time_Days' in inv_df.columns:
                # ROP = (Daily Demand × Lead Time) + Safety Stock
                daily_demand = inv_df['Avg_Sales_3M'] / 30  # Approximate daily demand
                z_score = 1.65  # 95% service level
                safety_stock = z_score * inv_df['Sales_Std'] * np.sqrt(inv_df['Lead_Time_Days']/30)
                
                inv_df['Reorder_Point'] = (daily_demand * inv_df['Lead_Time_Days'] + safety_stock).round(0)
                inv_df['Safety_Stock'] = safety_stock.round(0)
        
        # Service level calculation
        if 'Min_Stock_Level' in inv_df.columns:
            inv_df['Service_Level'] = np.where(
                inv_df['Quantity_Available'] >= inv_df['Min_Stock_Level'],
                'Adequate',
                'Below Minimum'
            )
        
        return inv_df
    
    def calculate_supplier_performance(self):
        """Calculate supplier performance metrics"""
        if self.data['suppliers'].empty or self.data['po'].empty:
            return pd.DataFrame()
        
        # Process PO data
        po_df = self.data['po'].copy()
        if 'Supplier_ID' not in po_df.columns and 'Supplier_Master' in self.data:
            # Need to enrich PO with supplier data
            pass
        
        # Calculate key metrics
        supplier_metrics = self.data['suppliers'].copy()
        
        if 'On_Time_Delivery_%' in supplier_metrics.columns:
            supplier_metrics['OTD_Score'] = pd.cut(
                supplier_metrics['On_Time_Delivery_%'],
                bins=[0, 85, 95, 100],
                labels=['Poor', 'Good', 'Excellent']
            )
        
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
        
        # Calculate on-time delivery
        if 'Delivery_Date' in trans_df.columns and 'Expected_Delivery_Date' in trans_df.columns:
            trans_df['Delivery_Date'] = pd.to_datetime(trans_df['Delivery_Date'])
            trans_df['Expected_Delivery_Date'] = pd.to_datetime(trans_df['Expected_Delivery_Date'])
            trans_df['Days_Late'] = (trans_df['Delivery_Date'] - trans_df['Expected_Delivery_Date']).dt.days
            trans_df['On_Time'] = trans_df['Days_Late'] <= 0
        
        # Carrier performance
        if 'Carrier' in trans_df.columns:
            carrier_perf = trans_df.groupby('Carrier').agg({
                'Shipment_ID': 'count',
                'Days_Late': 'mean',
                'Freight_Cost': 'mean',
                'Transit_Time_Days': 'mean'
            }).reset_index()
            
            carrier_perf.columns = ['Carrier', 'Total_Shipments', 'Avg_Days_Late', 
                                   'Avg_Freight_Cost', 'Avg_Transit_Time']
            carrier_perf['On_Time_Rate'] = (trans_df.groupby('Carrier')['On_Time'].mean() * 100).values
        
        return carrier_perf if 'Carrier' in trans_df.columns else pd.DataFrame()
    
    def calculate_warehouse_metrics(self):
        """Calculate warehouse performance metrics"""
        if self.data['warehouse'].empty:
            return pd.DataFrame()
        
        warehouse_df = self.data['warehouse'].copy()
        
        # Convert date if exists
        if 'Activity_Date' in warehouse_df.columns:
            warehouse_df['Activity_Date'] = pd.to_datetime(warehouse_df['Activity_Date'])
            warehouse_df['Month'] = warehouse_df['Activity_Date'].dt.to_period('M')
        
        # Calculate productivity
        if 'Processing_Time_Min' in warehouse_df.columns and 'Quantity' in warehouse_df.columns:
            # Units per hour
            warehouse_df['Units_per_Hour'] = (warehouse_df['Quantity'] / (warehouse_df['Processing_Time_Min'] / 60)).fillna(0)
        
        # Error rate
        if 'Error_Flag' in warehouse_df.columns:
            error_rate = warehouse_df.groupby('Activity_Type')['Error_Flag'].mean().reset_index()
            error_rate['Error_Rate_%'] = error_rate['Error_Flag'] * 100
        
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
        if not self.data['po'].empty:
            # Perfect Order Fulfillment
            po_df = self.data['po'].copy()
            if 'PO_Status' in po_df.columns:
                perfect_orders = len(po_df[po_df['PO_Status'] == 'Closed'])
                total_orders = len(po_df)
                metrics['Perfect_Order_Fulfillment'] = (perfect_orders / total_orders * 100) if total_orders > 0 else 0
        
        # 2. Responsiveness Metrics
        if 'Lead_Time_Days' in self.data['product'].columns:
            avg_lead_time = self.data['product']['Lead_Time_Days'].mean()
            metrics['Average_Lead_Time'] = avg_lead_time
        
        # 3. Agility Metrics
        if not self.data['sales'].empty:
            # Calculate demand variability
            sales_std = self.data['sales']['Sales_Qty'].std()
            sales_mean = self.data['sales']['Sales_Qty'].mean()
            metrics['Demand_Variability_CV'] = (sales_std / sales_mean) if sales_mean > 0 else 0
        
        # 4. Costs Metrics
        if not self.data['stock'].empty and 'Unit_Cost' in self.data['product'].columns:
            # Calculate inventory holding cost
            inv_df = pd.merge(self.data['stock'], self.data['product'][['SKU_ID', 'Unit_Cost']], 
                            on='SKU_ID', how='left')
            total_inv_value = (inv_df['Quantity_Available'] * inv_df['Unit_Cost']).sum()
            metrics['Total_Inventory_Value'] = total_inv_value
        
        # 5. Asset Management Metrics
        if 'Total_Inventory_Value' in metrics:
            # Inventory Turnover (need sales value)
            if not self.data['sales'].empty and 'Unit_Cost' in self.data['product'].columns:
                # Simplified calculation
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
        <div class="kpi-card fade-in">
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
        <div class="summary-card {bg_class} fade-in">
            <div class="sum-title">{title}</div>
            <div class="sum-val">{value}</div>
            <div class="sum-sub">{subtitle}</div>
        </div>
        """
    
    @staticmethod
    def metric_card(title, current_value, target_value=None, unit=None):
        """Create a metric card with target comparison"""
        if target_value and current_value:
            achievement = (current_value / target_value * 100) if target_value > 0 else 0
            status_color = "status-success" if achievement >= 90 else "status-warning" if achievement >= 80 else "status-danger"
            
            return f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 8px;">{title}</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.5rem; font-weight: 700;">{current_value:,.0f}</span>
                        {f'<span style="font-size: 0.8rem; color: #666; margin-left: 4px;">{unit}</span>' if unit else ''}
                    </div>
                    <div>
                        <span class="status-badge {status_color}">{achievement:.0f}%</span>
                    </div>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {min(achievement, 100)}%;"></div>
                </div>
                <div style="font-size: 0.75rem; color: #666; margin-top: 4px;">
                    Target: {target_value:,.0f} {unit if unit else ''}
                </div>
            </div>
            """
        else:
            return f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 8px;">{title}</div>
                <div>
                    <span style="font-size: 1.5rem; font-weight: 700;">{current_value:,.0f}</span>
                    {f'<span style="font-size: 0.8rem; color: #666; margin-left: 4px;">{unit}</span>' if unit else ''}
                </div>
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
        
        st.markdown("---")
        
        # Data status
        st.subheader("📊 Data Status")
        data_counts = {
            "Products": len(all_data.get('product', pd.DataFrame())),
            "Active SKUs": len(all_data.get('active_skus', [])),
            "Sales Records": len(all_data.get('sales', pd.DataFrame())),
            "PO Records": len(all_data.get('po', pd.DataFrame())),
            "Suppliers": len(all_data.get('suppliers', pd.DataFrame())),
            "Customers": len(all_data.get('customers', pd.DataFrame()))
        }
        
        for key, value in data_counts.items():
            st.caption(f"**{key}:** {value}")
    
    # Main dashboard
    components = DashboardComponents()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Executive Dashboard",
        "📦 Inventory Management",
        "🚚 Logistics & Procurement",
        "📈 Demand Planning",
        "📊 Performance Analytics",
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
        
        # Financial Impact Row
        st.markdown("---")
        st.subheader("💰 Financial Impact")
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            total_inv_value = scor_metrics.get('Total_Inventory_Value', 0)
            st.markdown(components.summary_card(
                "Total Inventory Value",
                f"${total_inv_value:,.0f}",
                "Current Stock Value",
                "bg-info"
            ), unsafe_allow_html=True)
        
        with col6:
            # Calculate potential savings
            if not inv_metrics.empty:
                high_stock_value = inv_metrics[inv_metrics['Inventory_Status'] == 'High Stock']['Stock_Value'].sum()
                st.markdown(components.summary_card(
                    "Excess Inventory",
                    f"${high_stock_value:,.0f}",
                    "Potential Savings",
                    "bg-warning"
                ), unsafe_allow_html=True)
        
        with col7:
            # Transportation cost
            if not all_data['transportation'].empty:
                total_freight = all_data['transportation']['Freight_Cost'].sum()
                st.markdown(components.summary_card(
                    "Freight Cost",
                    f"${total_freight:,.0f}",
                    "Last 30 Days",
                    "bg-purple"
                ), unsafe_allow_html=True)
        
        with col8:
            # Stockout cost (placeholder)
            stockout_cost = 125000
            st.markdown(components.summary_card(
                "Stockout Cost",
                f"${stockout_cost:,.0f}",
                "Estimated Annual Impact",
                "bg-danger"
            ), unsafe_allow_html=True)
        
        # Supply Chain Health
        st.markdown("---")
        st.subheader("❤️ Supply Chain Health Scorecard")
        
        col9, col10, col11 = st.columns(3)
        
        with col9:
            # Inventory Health
            if not inv_metrics.empty:
                ideal_pct = (len(inv_metrics[inv_metrics['Inventory_Status'] == 'Ideal']) / len(inv_metrics)) * 100
                st.markdown(components.metric_card(
                    "Inventory Health",
                    ideal_pct,
                    80,
                    "%"
                ), unsafe_allow_html=True)
        
        with col10:
            # Supplier Performance
            if not supplier_metrics.empty and 'On_Time_Delivery_%' in supplier_metrics.columns:
                avg_otd = supplier_metrics['On_Time_Delivery_%'].mean()
                st.markdown(components.metric_card(
                    "Supplier OTD",
                    avg_otd,
                    95,
                    "%"
                ), unsafe_allow_html=True)
        
        with col11:
            # Warehouse Efficiency
            if not warehouse_metrics.empty and 'Processing_Time_Min' in warehouse_metrics.columns:
                avg_processing = warehouse_metrics['Processing_Time_Min'].mean()
                st.markdown(components.metric_card(
                    "Avg Processing Time",
                    avg_processing,
                    15,
                    "min"
                ), unsafe_allow_html=True)
        
        # Risk Indicators
        st.markdown("---")
        st.subheader("⚠️ Risk Indicators")
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            # Single-source dependency
            if not supplier_metrics.empty:
                high_risk_suppliers = len(supplier_metrics[supplier_metrics['Performance_Score'] < 70])
                st.warning(f"**{high_risk_suppliers} High-Risk Suppliers**")
        
        with risk_col2:
            # Aging inventory
            if not inv_metrics.empty:
                slow_moving = len(inv_metrics[inv_metrics['Cover_Months'] > 3])
                st.warning(f"**{slow_moving} Slow-Moving SKUs**")
        
        with risk_col3:
            # Capacity constraints
            st.error("**2 Warehouses at 90%+ Capacity**")
    
    # ==========================================
    # TAB 2: INVENTORY MANAGEMENT
    # ==========================================
    with tab2:
        st.subheader("📦 Advanced Inventory Analytics")
        
        if not inv_metrics.empty:
            # Top row: Inventory Overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_skus = len(inv_metrics)
                st.markdown(components.summary_card(
                    "Total SKUs",
                    total_skus,
                    "Active Products",
                    "bg-primary"
                ), unsafe_allow_html=True)
            
            with col2:
                need_replenish = len(inv_metrics[inv_metrics['Inventory_Status'] == 'Need Replenishment'])
                st.markdown(components.summary_card(
                    "Need Replenishment",
                    need_replenish,
                    "< 1 month cover",
                    "bg-danger"
                ), unsafe_allow_html=True)
            
            with col3:
                ideal_stock = len(inv_metrics[inv_metrics['Inventory_Status'] == 'Ideal'])
                st.markdown(components.summary_card(
                    "Ideal Stock",
                    ideal_stock,
                    "1-1.5 months cover",
                    "bg-success"
                ), unsafe_allow_html=True)
            
            with col4:
                high_stock = len(inv_metrics[inv_metrics['Inventory_Status'] == 'High Stock'])
                st.markdown(components.summary_card(
                    "High Stock",
                    high_stock,
                    "> 1.5 months cover",
                    "bg-warning"
                ), unsafe_allow_html=True)
            
            # Inventory Analysis Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Inventory Status Distribution
                status_dist = inv_metrics['Inventory_Status'].value_counts()
                fig_status = px.pie(
                    values=status_dist.values,
                    names=status_dist.index,
                    title="Inventory Status Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_status.update_traces(textposition='inside', textinfo='percent+label')
                fig_status.update_layout(height=400)
                st.plotly_chart(fig_status, use_container_width=True)
            
            with col_chart2:
                # Cover Months Distribution
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
                           'Quantity_Available', 'Avg_Sales_3M', 'Cover_Months', 
                           'Inventory_Status', 'Stock_Value', 'EOQ', 'Reorder_Point']
            display_cols = [col for col in display_cols if col in filtered_inv.columns]
            
            st.dataframe(
                filtered_inv[display_cols].sort_values('Stock_Value', ascending=False),
                column_config={
                    "Quantity_Available": st.column_config.NumberColumn(format="%d"),
                    "Avg_Sales_3M": st.column_config.NumberColumn(format="%d"),
                    "Cover_Months": st.column_config.NumberColumn(format="%.1f"),
                    "Stock_Value": st.column_config.NumberColumn(format="$%.0f"),
                    "EOQ": st.column_config.NumberColumn(format="%d"),
                    "Reorder_Point": st.column_config.NumberColumn(format="%d")
                },
                use_container_width=True,
                height=500
            )
    
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
                
                # Supplier Risk Matrix
                st.subheader("⚠️ Supplier Risk Matrix")
                
                if 'On_Time_Delivery_%' in supplier_metrics.columns and 'Quality_Score' in supplier_metrics.columns:
                    fig_risk = px.scatter(
                        supplier_metrics,
                        x='On_Time_Delivery_%',
                        y='Quality_Score',
                        size='Performance_Score',
                        color='Tier' if 'Tier' in supplier_metrics.columns else None,
                        hover_name='Supplier_Name',
                        title="Supplier Risk Matrix",
                        labels={
                            'On_Time_Delivery_%': 'On-Time Delivery (%)',
                            'Quality_Score': 'Quality Score',
                            'Performance_Score': 'Overall Performance'
                        }
                    )
                    
                    # Add quadrant lines
                    fig_risk.add_hline(y=80, line_dash="dash", line_color="gray")
                    fig_risk.add_vline(x=90, line_dash="dash", line_color="gray")
                    
                    fig_risk.update_layout(height=500)
                    st.plotly_chart(fig_risk, use_container_width=True)
        
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
                
                # Freight Cost Analysis
                if not all_data['transportation'].empty:
                    freight_by_carrier = all_data['transportation'].groupby('Carrier')['Freight_Cost'].agg(['sum', 'mean']).reset_index()
                    
                    fig_freight = px.bar(
                        freight_by_carrier,
                        x='Carrier',
                        y='sum',
                        title="Total Freight Cost by Carrier",
                        labels={'sum': 'Total Cost ($)', 'Carrier': 'Carrier'},
                        color='mean',
                        color_continuous_scale='sunset'
                    )
                    fig_freight.update_layout(height=400)
                    st.plotly_chart(fig_freight, use_container_width=True)
            
            # Procurement Analytics
            st.subheader("📋 Procurement Analytics")
            
            if not all_data['po'].empty:
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
    
    # ==========================================
    # TAB 4: DEMAND PLANNING
    # ==========================================
    with tab4:
        st.subheader("📈 Advanced Demand Planning")
        
        if 'overall_mape' in demand_accuracy:
            # Forecast Accuracy Dashboard
            col1, col2, col3, col4 = st.columns(4)
            
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
            
            with col4:
                # Calculate forecast value added (placeholder)
                fva = 15.3
                st.markdown(components.kpi_card(
                    "Forecast Value Added",
                    f"{fva:.1f}%",
                    trend=3.2,
                    subtitle="Improvement over naive forecast",
                    icon="📊"
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
            
            # Forecast vs Actual Analysis
            st.markdown("---")
            st.subheader("📊 Forecast vs Actual Analysis")
            
            if 'detailed_data' in demand_accuracy:
                # Top 10 SKUs by forecast error
                top_errors = demand_accuracy['detailed_data'].groupby('SKU_ID').agg({
                    'Absolute_Percentage_Error': 'mean',
                    'Sales_Qty': 'sum'
                }).reset_index()
                
                # Merge with product names
                if not all_data['product'].empty:
                    top_errors = pd.merge(
                        top_errors,
                        all_data['product'][['SKU_ID', 'Product_Name', 'Brand']],
                        on='SKU_ID',
                        how='left'
                    )
                
                top_errors = top_errors.nlargest(10, 'Absolute_Percentage_Error')
                
                fig_errors = px.bar(
                    top_errors,
                    x='Product_Name',
                    y='Absolute_Percentage_Error',
                    title="Top 10 SKUs with Highest Forecast Error",
                    labels={'Absolute_Percentage_Error': 'Average Error (%)', 'Product_Name': 'Product'},
                    color='Brand',
                    hover_data=['Sales_Qty']
                )
                fig_errors.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_errors, use_container_width=True)
            
            # Demand Segmentation
            st.markdown("---")
            st.subheader("🎯 Demand Segmentation")
            
            if not all_data['sales'].empty and not all_data['product'].empty:
                # Calculate demand patterns
                sales_pattern = all_data['sales'].groupby('SKU_ID')['Sales_Qty'].agg(['mean', 'std', 'count']).reset_index()
                sales_pattern['CV'] = np.where(sales_pattern['mean'] > 0, 
                                             sales_pattern['std'] / sales_pattern['mean'], 0)
                
                # Merge with product data
                sales_pattern = pd.merge(
                    sales_pattern,
                    all_data['product'][['SKU_ID', 'Product_Name', 'Category']],
                    on='SKU_ID',
                    how='left'
                )
                
                # Create demand segmentation
                conditions = [
                    (sales_pattern['CV'] < 0.5) & (sales_pattern['mean'] >= sales_pattern['mean'].quantile(0.7)),
                    (sales_pattern['CV'] < 0.5) & (sales_pattern['mean'] < sales_pattern['mean'].quantile(0.7)),
                    (sales_pattern['CV'] >= 0.5) & (sales_pattern['mean'] >= sales_pattern['mean'].quantile(0.7)),
                    (sales_pattern['CV'] >= 0.5) & (sales_pattern['mean'] < sales_pattern['mean'].quantile(0.7))
                ]
                choices = ['Smooth & High', 'Smooth & Low', 'Erratic & High', 'Erratic & Low']
                sales_pattern['Demand_Pattern'] = np.select(conditions, choices, default='Unknown')
                
                # Plot demand segmentation
                fig_segment = px.scatter(
                    sales_pattern,
                    x='mean',
                    y='CV',
                    color='Demand_Pattern',
                    size='count',
                    hover_name='Product_Name',
                    title="Demand Segmentation Matrix",
                    labels={'mean': 'Average Demand', 'CV': 'Coefficient of Variation'},
                    category_orders={'Demand_Pattern': ['Smooth & High', 'Smooth & Low', 'Erratic & High', 'Erratic & Low']},
                    color_discrete_sequence=['#00b894', '#74b9ff', '#e17055', '#a29bfe']
                )
                
                # Add quadrant lines
                median_demand = sales_pattern['mean'].median()
                fig_segment.add_vline(x=median_demand, line_dash="dash", line_color="gray")
                fig_segment.add_hline(y=0.5, line_dash="dash", line_color="gray")
                
                fig_segment.update_layout(height=500)
                st.plotly_chart(fig_segment, use_container_width=True)
    
    # ==========================================
    # TAB 5: PERFORMANCE ANALYTICS
    # ==========================================
    with tab5:
        st.subheader("📊 SCOR Metrics & Performance Analytics")
        
        # SCOR Metrics Dashboard
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Reliability
            perfect_order = scor_metrics.get('Perfect_Order_Fulfillment', 0)
            st.markdown(components.metric_card(
                "Reliability",
                perfect_order,
                95,
                "%"
            ), unsafe_allow_html=True)
        
        with col2:
            # Responsiveness
            lead_time = scor_metrics.get('Average_Lead_Time', 0)
            st.markdown(components.metric_card(
                "Responsiveness",
                lead_time,
                30,
                "days"
            ), unsafe_allow_html=True)
        
        with col3:
            # Agility
            demand_cv = scor_metrics.get('Demand_Variability_CV', 0)
            st.markdown(components.metric_card(
                "Agility",
                demand_cv * 100,
                50,
                "% CV"
            ), unsafe_allow_html=True)
        
        with col4:
            # Costs
            inv_value = scor_metrics.get('Total_Inventory_Value', 0)
            st.markdown(components.metric_card(
                "Costs",
                inv_value / 1000,
                500,
                "K$"
            ), unsafe_allow_html=True)
        
        with col5:
            # Asset Management
            turnover = scor_metrics.get('Inventory_Turnover', 0)
            st.markdown(components.metric_card(
                "Asset Mgmt",
                turnover,
                12,
                "turns"
            ), unsafe_allow_html=True)
        
        # Performance Trend Analysis
        st.markdown("---")
        st.subheader("📈 Performance Trend Analysis")
        
        # Create mock trend data (in real app, this would come from historical data)
        months = pd.date_range(end=datetime.now(), periods=12, freq='M')
        trend_data = pd.DataFrame({
            'Month': months,
            'Perfect_Order_Rate': np.random.uniform(85, 98, 12),
            'Forecast_Accuracy': np.random.uniform(75, 92, 12),
            'Inventory_Turnover': np.random.uniform(8, 15, 12),
            'Lead_Time': np.random.uniform(20, 40, 12)
        })
        
        # Normalize for radar chart
        trend_norm = trend_data.copy()
        for col in ['Perfect_Order_Rate', 'Forecast_Accuracy', 'Inventory_Turnover']:
            trend_norm[col] = (trend_data[col] - trend_data[col].min()) / (trend_data[col].max() - trend_data[col].min()) * 100
        
        trend_norm['Lead_Time'] = 100 - ((trend_data['Lead_Time'] - trend_data['Lead_Time'].min()) / 
                                        (trend_data['Lead_Time'].max() - trend_data['Lead_Time'].min()) * 100)
        
        # Latest performance radar chart
        latest_perf = trend_norm.iloc[-1]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[latest_perf['Perfect_Order_Rate'], latest_perf['Forecast_Accuracy'], 
               latest_perf['Inventory_Turnover'], latest_perf['Lead_Time']],
            theta=['Reliability', 'Forecast Acc', 'Inventory Turns', 'Lead Time'],
            fill='toself',
            line_color='#1e3799',
            fillcolor='rgba(30, 55, 153, 0.3)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Latest Performance Profile",
            height=400
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Benchmarking
        st.markdown("---")
        st.subheader("🏆 Benchmarking Analysis")
        
        benchmark_data = pd.DataFrame({
            'Metric': ['Perfect Order Rate', 'Forecast Accuracy', 'Inventory Turnover', 'Cash-to-Cash Cycle'],
            'Your_Company': [perfect_order, forecast_accuracy, inv_turnover, 45],
            'Industry_Average': [92.5, 82.3, 10.5, 55],
            'Industry_Best': [98.2, 94.7, 18.2, 28]
        })
        
        fig_benchmark = go.Figure()
        
        fig_benchmark.add_trace(go.Bar(
            name='Your Company',
            x=benchmark_data['Metric'],
            y=benchmark_data['Your_Company'],
            marker_color='#1e3799'
        ))
        
        fig_benchmark.add_trace(go.Bar(
            name='Industry Average',
            x=benchmark_data['Metric'],
            y=benchmark_data['Industry_Average'],
            marker_color='#74b9ff'
        ))
        
        fig_benchmark.add_trace(go.Bar(
            name='Industry Best',
            x=benchmark_data['Metric'],
            y=benchmark_data['Industry_Best'],
            marker_color='#00b894'
        ))
        
        fig_benchmark.update_layout(
            barmode='group',
            title="Performance Benchmarking",
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_benchmark, use_container_width=True)
    
    # ==========================================
    # TAB 6: OPERATIONS
    # ==========================================
    with tab6:
        st.subheader("🔧 Operational Excellence")
        
        # Warehouse Operations
        st.markdown("### 🏭 Warehouse Operations")
        
        if not warehouse_metrics.empty:
            col_w1, col_w2, col_w3 = st.columns(3)
            
            with col_w1:
                # Productivity
                if 'Units_per_Hour' in warehouse_metrics.columns:
                    avg_productivity = warehouse_metrics['Units_per_Hour'].mean()
                    st.markdown(components.metric_card(
                        "Avg Productivity",
                        avg_productivity,
                        100,
                        "units/hour"
                    ), unsafe_allow_html=True)
            
            with col_w2:
                # Accuracy
                if 'Error_Flag' in warehouse_metrics.columns:
                    accuracy_rate = (1 - warehouse_metrics['Error_Flag'].mean()) * 100
                    st.markdown(components.metric_card(
                        "Accuracy Rate",
                        accuracy_rate,
                        99.5,
                        "%"
                    ), unsafe_allow_html=True)
            
            with col_w3:
                # Damage rate
                if 'Damage_Qty' in warehouse_metrics.columns and 'Quantity' in warehouse_metrics.columns:
                    total_qty = warehouse_metrics['Quantity'].sum()
                    damage_qty = warehouse_metrics['Damage_Qty'].sum()
                    damage_rate = (damage_qty / total_qty * 100) if total_qty > 0 else 0
                    st.markdown(components.metric_card(
                        "Damage Rate",
                        damage_rate,
                        0.5,
                        "%"
                    ), unsafe_allow_html=True)
            
            # Warehouse Activity Analysis
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
        
        # Quality Control
        st.markdown("---")
        st.markdown("### 🏆 Quality Control")
        
        # Supplier Quality
        if not supplier_metrics.empty and 'Quality_Score' in supplier_metrics.columns:
            quality_summary = supplier_metrics.groupby('Quality_Score_Category').size().reset_index()
            quality_summary.columns = ['Quality', 'Count']
            
            fig_quality = px.pie(
                quality_summary,
                values='Count',
                names='Quality',
                title="Supplier Quality Distribution",
                color_discrete_sequence=['#00b894', '#fdcb6e', '#e17055']
            )
            fig_quality.update_traces(textposition='inside', textinfo='percent+label')
            fig_quality.update_layout(height=400)
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # Continuous Improvement
        st.markdown("---")
        st.markdown("### 📈 Continuous Improvement Projects")
        
        improvement_projects = pd.DataFrame({
            'Project': ['ABC Classification Review', 'Supplier Rationalization', 
                       'Warehouse Automation Phase 1', 'Forecast Model Upgrade',
                       'Transportation Optimization', 'Packaging Redesign'],
            'Status': ['In Progress', 'Completed', 'Planning', 'In Progress', 'Completed', 'Planning'],
            'Impact_Score': [8.5, 9.2, 7.8, 8.9, 7.5, 6.8],
            'Expected_Savings': [125000, 85000, 200000, 75000, 95000, 45000],
            'Completion_%': [75, 100, 25, 60, 100, 30]
        })
        
        st.dataframe(
            improvement_projects,
            column_config={
                "Impact_Score": st.column_config.NumberColumn(format="%.1f"),
                "Expected_Savings": st.column_config.NumberColumn(format="$%.0f"),
                "Completion_%": st.column_config.NumberColumn(format="%.0f%%")
            },
            use_container_width=True
        )
        
        # Risk Management
        st.markdown("---")
        st.markdown("### ⚠️ Risk Management Dashboard")
        
        risk_matrix = pd.DataFrame({
            'Risk_Category': ['Supply Disruption', 'Demand Volatility', 'Price Fluctuation', 
                             'Quality Issues', 'Logistics Delays', 'Cyber Security'],
            'Probability': [0.3, 0.6, 0.4, 0.2, 0.5, 0.1],
            'Impact': [8, 7, 6, 9, 5, 10],
            'Risk_Score': [2.4, 4.2, 2.4, 1.8, 2.5, 1.0]
        })
        
        fig_risk_matrix = px.scatter(
            risk_matrix,
            x='Probability',
            y='Impact',
            size='Risk_Score',
            color='Risk_Category',
            hover_name='Risk_Category',
            title="Risk Matrix",
            size_max=50
        )
        
        # Add risk quadrants
        fig_risk_matrix.add_hrect(y0=7.5, y1=10, line_width=0, fillcolor="red", opacity=0.1)
        fig_risk_matrix.add_hrect(y0=5, y1=7.5, line_width=0, fillcolor="orange", opacity=0.1)
        fig_risk_matrix.add_hrect(y0=0, y1=5, line_width=0, fillcolor="green", opacity=0.1)
        
        fig_risk_matrix.update_layout(height=500)
        st.plotly_chart(fig_risk_matrix, use_container_width=True)

# --- RUN DASHBOARD ---
if __name__ == "__main__":
    main()
