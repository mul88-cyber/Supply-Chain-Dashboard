# app.py - Supply Chain Dashboard (FIXED VERSION)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Supply Chain Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize Google Sheets connection
@st.cache_resource
def init_gsheets():
    """Initialize Google Sheets connection"""
    try:
        # Load credentials from Streamlit secrets
        creds_dict = st.secrets["gcp_service_account"]
        
        # Define the required scopes
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # Create credentials
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        
        # Authorize the client
        client = gspread.authorize(creds)
        
        # Open the spreadsheet (replace with your spreadsheet ID)
        spreadsheet_id = "YOUR_SPREADSHEET_ID"  # Ganti dengan ID spreadsheet-mu
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        return spreadsheet
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {str(e)}")
        return None

# Safe data loading function
def safe_load_sheet(spreadsheet, sheet_name, required_columns=None):
    """Load sheet data with error handling for empty sheets"""
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_records()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # If sheet is completely empty or has only headers
        if len(df) == 0:
            st.warning(f"⚠️ Sheet '{sheet_name}' is empty or has no data rows.")
            
            # Create empty DataFrame with expected columns if provided
            if required_columns:
                df = pd.DataFrame(columns=required_columns)
            return df
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns safely
        numeric_cols = ['Unit_Cost', 'Price', 'Quantity', 'Stock', 'Demand', 
                       'Cost', 'Revenue', 'Amount', 'Units']
        
        for col in df.columns:
            if any(numeric in col for numeric in numeric_cols):
                # Replace empty strings and non-numeric values
                df[col] = df[col].replace(['', ' ', 'N/A', 'NaN', 'null', None], 0)
                # Convert to numeric, forcing errors to NaN then to 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    except Exception as e:
        st.warning(f"⚠️ Could not load sheet '{sheet_name}': {str(e)}")
        # Return empty DataFrame with expected columns
        if required_columns:
            return pd.DataFrame(columns=required_columns)
        return pd.DataFrame()

# Main data loading function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_all_data():
    """Load all sheets from Google Spreadsheet"""
    
    spreadsheet = init_gsheets()
    if spreadsheet is None:
        return {}
    
    # Define expected columns for each sheet (for empty sheet handling)
    sheet_structures = {
        'Products': ['Product_ID', 'Product_Name', 'Category', 'Unit_Cost', 'Price', 'Supplier'],
        'Sales_Records': ['Date', 'Product_ID', 'Quantity', 'Price', 'Customer', 'Region'],
        'Inventory': ['Product_ID', 'Stock_Quantity', 'Min_Stock', 'Max_Stock', 'Location'],
        'Purchase_Orders': ['PO_Number', 'Supplier', 'Product_ID', 'Quantity', 'Unit_Cost', 'Status'],
        'Customers': ['Customer_ID', 'Customer_Name', 'Region', 'Credit_Limit', 'Payment_Terms'],
        'Suppliers': ['Supplier_ID', 'Supplier_Name', 'Category', 'Lead_Time', 'Rating'],
        'Shipments': ['Shipment_ID', 'Order_ID', 'Carrier', 'Status', 'Estimated_Delivery', 'Actual_Delivery'],
        'Returns': ['Return_ID', 'Order_ID', 'Product_ID', 'Quantity', 'Reason', 'Status'],
        'Forecast': ['Period', 'Product_ID', 'Forecast_Demand', 'Actual_Demand', 'Accuracy'],
        'KPI_Summary': ['KPI', 'Target', 'Actual', 'Variance', 'Status'],
        'Raw_Data': []  # No predefined structure
    }
    
    all_data = {}
    
    # Load each sheet with safe loading
    for sheet_name, expected_columns in sheet_structures.items():
        try:
            df = safe_load_sheet(spreadsheet, sheet_name, expected_columns)
            all_data[sheet_name] = df
        except:
            # Create empty DataFrame if sheet doesn't exist
            all_data[sheet_name] = pd.DataFrame(columns=expected_columns)
    
    return all_data

# FIXED: Inventory metrics calculation with empty data handling
def calculate_inventory_metrics(inventory_df):
    """Calculate inventory metrics with safe handling"""
    
    # Check if inventory data exists
    if inventory_df.empty or len(inventory_df) == 0:
        return {
            'total_inventory_value': 0,
            'avg_holding_cost': 0,
            'total_eoq': 0,
            'safety_stock': 0,
            'stockout_risk': 'Low',
            'turnover_ratio': 0
        }
    
    # Ensure required columns exist
    for col in ['Unit_Cost', 'Stock_Quantity', 'Demand_Rate']:
        if col not in inventory_df.columns:
            inventory_df[col] = 0
    
    # Convert numeric columns safely
    numeric_cols = ['Unit_Cost', 'Stock_Quantity', 'Demand_Rate']
    for col in numeric_cols:
        inventory_df[col] = inventory_df[col].replace(['', ' ', None], 0)
        inventory_df[col] = pd.to_numeric(inventory_df[col], errors='coerce').fillna(0)
    
    # FIXED LINE: Now safe to calculate
    H = inventory_df['Unit_Cost'] * 0.25  # Holding cost (25% of unit cost)
    
    # Calculate EOQ (Economic Order Quantity)
    # EOQ = sqrt((2 * D * S) / H)
    # Where D = demand, S = order cost (assume $50), H = holding cost
    D = inventory_df['Demand_Rate'].fillna(0)
    S = 50  # Fixed order cost
    eoq = np.sqrt((2 * D * S) / H.replace(0, 0.001))  # Avoid division by zero
    eoq = eoq.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate inventory value
    inventory_value = (inventory_df['Unit_Cost'] * inventory_df['Stock_Quantity']).sum()
    
    return {
        'total_inventory_value': round(inventory_value, 2),
        'avg_holding_cost': round(H.mean(), 2) if len(H) > 0 else 0,
        'total_eoq': round(eoq.sum(), 2),
        'safety_stock': round(D.std() * 1.65 if len(D) > 1 else 0, 2),  # 95% service level
        'stockout_risk': 'High' if (inventory_df['Stock_Quantity'] < 10).any() else 'Low',
        'turnover_ratio': round(D.sum() / inventory_value if inventory_value > 0 else 0, 2)
    }

# Main dashboard function
def main():
    st.title("📊 Supply Chain Analytics Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        st.markdown("---")
        
        # Data refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.info("**Data Status:**")
        
        # Load data
        with st.spinner("Loading data from Google Sheets..."):
            data = load_all_data()
        
        # Show data status
        for sheet_name, df in data.items():
            row_count = len(df)
            if row_count > 0:
                st.success(f"✓ {sheet_name}: {row_count} rows")
            else:
                st.warning(f"○ {sheet_name}: No data")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "📦 Inventory", 
        "💰 Sales", 
        "🚚 Logistics", 
        "📋 Details"
    ])
    
    with tab1:
        st.header("Executive Overview")
        
        # Create KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Total Products
            products_count = len(data['Products']) if not data['Products'].empty else 0
            st.metric("Total Products", products_count)
        
        with col2:
            # Total Sales
            if not data['Sales_Records'].empty and 'Quantity' in data['Sales_Records'].columns:
                total_sales = data['Sales_Records']['Quantity'].sum()
            else:
                total_sales = 0
            st.metric("Total Units Sold", int(total_sales))
        
        with col3:
            # Inventory Value
            inv_metrics = calculate_inventory_metrics(data['Inventory'])
            st.metric("Inventory Value", f"${inv_metrics['total_inventory_value']:,.0f}")
        
        with col4:
            # Active SKUs
            if not data['Inventory'].empty and 'Stock_Quantity' in data['Inventory'].columns:
                active_skus = (data['Inventory']['Stock_Quantity'] > 0).sum()
            else:
                active_skus = 0
            st.metric("Active SKUs", active_skus)
    
    with tab2:
        st.header("Inventory Analytics")
        
        # Show inventory metrics
        metrics = calculate_inventory_metrics(data['Inventory'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Holding Cost per Unit", f"${metrics['avg_holding_cost']}")
        with col2:
            st.metric("Economic Order Quantity", f"{metrics['total_eoq']:.0f} units")
        with col3:
            st.metric("Stockout Risk", metrics['stockout_risk'])
        
        # Display inventory data if exists
        if not data['Inventory'].empty:
            st.subheader("Inventory Data")
            st.dataframe(data['Inventory'], use_container_width=True)
        else:
            st.info("📝 Inventory data is empty. Please add data to the 'Inventory' sheet in Google Sheets.")
    
    with tab3:
        st.header("Sales Analytics")
        
        if not data['Sales_Records'].empty:
            # Sales trend chart
            if 'Date' in data['Sales_Records'].columns:
                data['Sales_Records']['Date'] = pd.to_datetime(data['Sales_Records']['Date'], errors='coerce')
                daily_sales = data['Sales_Records'].groupby(
                    data['Sales_Records']['Date'].dt.date
                )['Quantity'].sum().reset_index()
                
                fig = px.line(daily_sales, x='Date', y='Quantity', 
                            title="Daily Sales Trend")
                st.plotly_chart(fig, use_container_width=True)
            
            # Top products
            if 'Product_ID' in data['Sales_Records'].columns:
                top_products = data['Sales_Records'].groupby('Product_ID')['Quantity'].sum().nlargest(10)
                fig2 = px.bar(x=top_products.index, y=top_products.values,
                            title="Top 10 Products by Sales")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("📝 Sales data is empty. Please add data to the 'Sales_Records' sheet.")
    
    with tab4:
        st.header("Logistics & Supply")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Purchase Orders")
            if not data['Purchase_Orders'].empty:
                po_status = data['Purchase_Orders']['Status'].value_counts()
                fig = px.pie(values=po_status.values, names=po_status.index,
                           title="PO Status Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No purchase order data available.")
        
        with col2:
            st.subheader("Shipment Status")
            if not data['Shipments'].empty and 'Status' in data['Shipments'].columns:
                shipment_status = data['Shipments']['Status'].value_counts()
                fig = px.bar(x=shipment_status.index, y=shipment_status.values,
                           title="Shipment Status")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No shipment data available.")
    
    with tab5:
        st.header("Data Details")
        
        # Select sheet to view
        sheet_to_view = st.selectbox(
            "Select Sheet to View",
            list(data.keys()),
            index=0
        )
        
        if not data[sheet_to_view].empty:
            st.dataframe(data[sheet_to_view], use_container_width=True)
            
            # Download button
            csv = data[sheet_to_view].to_csv(index=False)
            st.download_button(
                label=f"Download {sheet_to_view} as CSV",
                data=csv,
                file_name=f"{sheet_to_view.lower()}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"The '{sheet_to_view}' sheet is currently empty.")

# Run the app
if __name__ == "__main__":
    main()
