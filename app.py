import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import gspread
from google.oauth2.service_account import Credentials
import warnings
import os

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Supply Chain Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (DARK/LIGHT MODE COMPATIBLE) ---
st.markdown("""
<style>
    .kpi-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #2980b9;
        text-align: center;
    }
    .kpi-title { font-size: 14px; color: #7f8c8d; font-weight: bold; text-transform: uppercase; }
    .kpi-value { font-size: 32px; font-weight: 800; color: #2c3e50; margin: 10px 0; }
    .status-badge { padding: 5px 10px; border-radius: 15px; font-weight: bold; color: white; }
    .stockout { background-color: #e74c3c; }
    .warning { background-color: #f39c12; }
    .healthy { background-color: #27ae60; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADER ENGINE ---
@st.cache_resource
def get_gsheet_client():
    try:
        # Coba load dari secrets Streamlit
        if "gcp_service_account" in st.secrets:
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            return gspread.authorize(creds)
        return None
    except Exception:
        return None

def load_data(sheet_name, csv_file, client=None):
    """Smart Loader: Coba GSheet dulu -> Kalau gagal, coba CSV lokal -> Kalau gagal, return Empty DF"""
    df = pd.DataFrame()
    
    # 1. Coba GSheet
    if client:
        try:
            url = "https://docs.google.com/spreadsheets/d/1gek6SPgJcZzhOjN-eNOZbfCvmX2EKiHKbgk3c0gUxL4"
            sh = client.open_by_url(url)
            ws = sh.worksheet(sheet_name)
            df = pd.DataFrame(ws.get_all_records())
        except:
            pass
            
    # 2. Coba CSV Lokal (Fallback)
    if df.empty:
        possible_paths = [csv_file, f"data/{csv_file}", f"Supply Chain_Data.xlsx - {csv_file}"]
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    break
                except:
                    continue
    
    # 3. Clean Columns
    if not df.empty:
        df.columns = [str(c).strip().replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
        
    return df

# --- MAIN APP ---
def main():
    st.markdown("## 🏭 Supply Chain Command Center")
    st.markdown("Dashboard operasional untuk monitoring Inventory, Demand Planning, dan PO Fulfillment.")
    
    # Initialize Data
    client = get_gsheet_client()
    
    with st.spinner("Menghubungkan data supply chain..."):
        df_prod = load_data("Product_Master", "Product_Master.csv", client)
        df_stock = load_data("Stock_Onhand", "Stock_Onhand.csv", client)
        df_sales = load_data("Sales", "Sales.csv", client)
        df_rofo = load_data("Rofo", "Rofo.csv", client)
        df_po = load_data("PO", "PO.csv", client)

    # Validasi Data Kritis
    if df_prod.empty or df_stock.empty:
        st.error("🚨 Data Kosong! Pastikan file CSV di-upload ke repo atau Secrets GSheet sudah benar.")
        st.stop()

    # --- DATA PROCESSING & MODELING ---
    # 1. Master Table
    master = df_prod.copy()
    if 'SKU_ID' not in master.columns:
        # Fallback jika nama kolom beda
        col_map = {c: 'SKU_ID' for c in master.columns if 'sku' in c.lower() and 'id' in c.lower()}
        master.rename(columns=col_map, inplace=True)
    
    master['SKU_ID'] = master['SKU_ID'].astype(str)
    
    # 2. Inventory Processing
    df_stock['SKU_ID'] = df_stock['SKU_ID'].astype(str)
    qty_col = next((c for c in df_stock.columns if 'qty' in c.lower() or 'stock' in c.lower()), 'Stock_Qty')
    stock_agg = df_stock.groupby('SKU_ID')[qty_col].sum().reset_index().rename(columns={qty_col: 'Onhand_Qty'})
    master = pd.merge(master, stock_agg, on='SKU_ID', how='left')
    master['Onhand_Qty'] = master['Onhand_Qty'].fillna(0)

    # 3. Pipeline (PO) Processing
    master['Intransit_Qty'] = 0
    if not df_po.empty:
        df_po['SKU_ID'] = df_po['SKU_ID'].astype(str)
        # Filter status PO aktif
        if 'Status' in df_po.columns:
            active_po = df_po[df_po['Status'].str.lower().isin(['open', 'confirmed', 'in transit'])]
            po_agg = active_po.groupby('SKU_ID')['Qty'].sum().reset_index()
            master = pd.merge(master, po_agg.rename(columns={'Qty': 'Intransit_Qty'}), on='SKU_ID', how='left')
            master['Intransit_Qty'] = master['Intransit_Qty_y'].fillna(0) # Handle merge duplicate cols

    # 4. Forecast Processing
    # Deteksi kolom bulan (Jan, Feb, atau 2025-01-01)
    month_cols = [c for c in df_rofo.columns if any(m in str(c).lower() for m in ['jan','feb','mar','apr','may','jun','2024','2025'])]
    
    if month_cols:
        df_rofo['SKU_ID'] = df_rofo['SKU_ID'].astype(str)
        # Bersihkan data non-numeric
        for c in month_cols:
            df_rofo[c] = pd.to_numeric(df_rofo[c], errors='coerce').fillna(0)
        
        df_rofo['Avg_Forecast'] = df_rofo[month_cols].mean(axis=1)
        master = pd.merge(master, df_rofo[['SKU_ID', 'Avg_Forecast']], on='SKU_ID', how='left')
    else:
        master['Avg_Forecast'] = 0 # Default jika tidak ada data forecast
        
    master['Avg_Forecast'] = master['Avg_Forecast'].fillna(0)

    # --- SIMULATION & FILTER (SIDEBAR) ---
    st.sidebar.header("🎛️ Control Panel")
    
    # Brand Filter
    brands = master['Brand'].unique().tolist() if 'Brand' in master.columns else []
    sel_brands = st.sidebar.multiselect("Filter Brand", brands, default=brands)
    
    if sel_brands:
        master = master[master['Brand'].isin(sel_brands)]

    st.sidebar.markdown("### 🔮 What-If Simulation")
    growth_scenario = st.sidebar.slider("Simulasi Kenaikan Demand (%)", -50, 100, 0, 5)
    
    # Apply Simulation
    master['Simulated_Demand'] = master['Avg_Forecast'] * (1 + growth_scenario/100)
    master['Daily_Demand'] = master['Simulated_Demand'] / 30
    
    # Calculate DOI (Days of Inventory)
    master['DOI'] = np.where(master['Daily_Demand'] > 0, 
                             (master['Onhand_Qty'] + master['Intransit_Qty']) / master['Daily_Demand'], 
                             999) # 999 = Dead Stock (No Demand)

    # Calculate Status
    def get_health(row):
        if row['Onhand_Qty'] == 0: return 'Stockout'
        if row['DOI'] < 30: return 'Critical Low'
        if row['DOI'] > 120: return 'Overstock'
        if row['Daily_Demand'] == 0: return 'Dead Stock'
        return 'Healthy'
    
    master['Status'] = master.apply(get_health, axis=1)
    
    # Valuation
    cost_col = next((c for c in master.columns if 'cost' in c.lower() or 'cogs' in c.lower() or 'price' in c.lower()), None)
    if cost_col:
        master['Valuation'] = master['Onhand_Qty'] * master[cost_col]
    else:
        master['Valuation'] = 0

    # --- DASHBOARD TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "📦 Inventory Health & ABC", "🚀 Planning & Replenishment"])

    with tab1:
        # KPI Row
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Inventory Asset</div><div class="kpi-value">Rp {master['Valuation'].sum()/1e9:,.2f} M</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total SKU Active</div><div class="kpi-value">{len(master)}</div></div>""", unsafe_allow_html=True)
        
        stockouts = len(master[master['Status']=='Stockout'])
        c3.markdown(f"""<div class="kpi-card" style="border-left: 5px solid #e74c3c;"><div class="kpi-title">Stockout Alert</div><div class="kpi-value" style="color: #e74c3c;">{stockouts}</div></div>""", unsafe_allow_html=True)
        
        overstocks = len(master[master['Status']=='Overstock'])
        c4.markdown(f"""<div class="kpi-card" style="border-left: 5px solid #f39c12;"><div class="kpi-title">Overstock SKU</div><div class="kpi-value" style="color: #f39c12;">{overstocks}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        
        # Charts
        chart_col1, chart_col2 = st.columns([2, 1])
        with chart_col1:
            st.subheader("Inventory Health Distribution")
            fig_health = px.bar(master['Status'].value_counts().reset_index(), 
                                x='Status', y='count', color='Status', 
                                color_discrete_map={'Healthy':'#27ae60', 'Critical Low':'#e67e22', 'Stockout':'#c0392b', 'Overstock':'#2980b9', 'Dead Stock':'#7f8c8d'})
            st.plotly_chart(fig_health, use_container_width=True)
            
        with chart_col2:
            st.subheader("Top 5 Brand by Value")
            if 'Brand' in master.columns:
                brand_val = master.groupby('Brand')['Valuation'].sum().sort_values(ascending=False).head(5)
                fig_brand = px.pie(values=brand_val, names=brand_val.index, hole=0.4)
                st.plotly_chart(fig_brand, use_container_width=True)

    with tab2:
        st.subheader("🔍 ABC Analysis (Pareto)")
        st.info("ABC Analysis membantu fokus pada SKU yang memberikan kontribusi value terbesar.")
        
        # Calculate ABC
        master = master.sort_values('Valuation', ascending=False)
        master['Cum_Val'] = master['Valuation'].cumsum()
        master['Total_Val'] = master['Valuation'].sum()
        master['Cum_Pct'] = 100 * master['Cum_Val'] / master['Total_Val']
        
        def get_abc(pct):
            if pct <= 80: return 'A'
            elif pct <= 95: return 'B'
            return 'C'
        
        master['ABC_Class'] = master['Cum_Pct'].apply(get_abc)
        
        col_abc1, col_abc2 = st.columns([3, 1])
        with col_abc1:
            fig_pareto = px.bar(master, x='Product_Name', y='Valuation', color='ABC_Class',
                                title='Pareto Chart (Inventory Value)',
                                color_discrete_map={'A':'#2ecc71', 'B':'#f1c40f', 'C':'#e74c3c'})
            fig_pareto.update_xaxes(showticklabels=False) # Hide too many labels
            st.plotly_chart(fig_pareto, use_container_width=True)
            
        with col_abc2:
            st.write(master['ABC_Class'].value_counts().rename("SKU Count"))
            st.markdown("**Insight:** Fokuskan kontrol stok pada item **Kelas A** (hijau).")

    with tab3:
        st.subheader("📋 Replenishment Advice")
        st.markdown("Rekomendasi PO berdasarkan: **Forecast Demand + Safety Stock (60 Hari) - (Stok Gudang + Barang OTW)**")
        
        target_days = 60
        master['Target_Stock'] = master['Daily_Demand'] * target_days
        master['Net_Requirement'] = master['Target_Stock'] - (master['Onhand_Qty'] + master['Intransit_Qty'])
        master['Suggest_PO'] = master['Net_Requirement'].apply(lambda x: x if x > 0 else 0)
        
        # Filter only items needing PO
        po_plan = master[master['Suggest_PO'] > 0].sort_values('Suggest_PO', ascending=False)
        
        display_cols = ['SKU_ID', 'Product_Name', 'Status', 'Onhand_Qty', 'Intransit_Qty', 'Avg_Forecast', 'Suggest_PO']
        # Pastikan kolom ada sebelum display
        display_cols = [c for c in display_cols if c in po_plan.columns]
        
        st.dataframe(po_plan[display_cols].style.background_gradient(subset=['Suggest_PO'], cmap='Greens'), use_container_width=True)
        
        # Download Button
        csv = po_plan.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download PO Plan (CSV)",
            data=csv,
            file_name=f"Replenishment_Plan_{date.today()}.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
