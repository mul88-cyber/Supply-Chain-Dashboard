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

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Supply Chain Command Center",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM (TAMPILAN PREMIUM) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-container:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    
    .big-number { font-size: 2.2rem; font-weight: 800; color: #1f77b4; margin: 0; }
    .metric-label { font-size: 0.9rem; color: #666; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .trend-positive { color: #2ca02c; font-size: 0.8rem; font-weight: bold; }
    .trend-negative { color: #d62728; font-size: 0.8rem; font-weight: bold; }
    
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def parse_month_label(label):
    """Membaca format bulan yang berantakan"""
    try:
        label_str = str(label).strip().lower()
        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        
        for m_name, m_num in month_map.items():
            if m_name in label_str:
                # Cari tahun (2 digit atau 4 digit)
                import re
                year_match = re.search(r'(\d{2,4})', label_str)
                year = int(year_match.group(1)) if year_match else datetime.now().year
                if year < 100: year += 2000
                return datetime(year, m_num, 1)
        return pd.to_datetime(label_str)
    except:
        return datetime.now()

# --- DATA ENGINE (GSHEET + CSV FALLBACK) ---
@st.cache_resource
def get_data_engine():
    """Mencoba konek GSheet, jika gagal pakai CSV lokal"""
    try:
        skey = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(skey, scopes=scopes)
        client = gspread.authorize(creds)
        return client, "gsheet"
    except Exception:
        return None, "csv"

def load_dataset(source_type, client, sheet_name, csv_filename):
    """Loader cerdas: Ambil dari GSheet atau CSV"""
    df = pd.DataFrame()
    
    # 1. Coba GSheet
    if source_type == "gsheet":
        try:
            url = "https://docs.google.com/spreadsheets/d/1gek6SPgJcZzhOjN-eNOZbfCvmX2EKiHKbgk3c0gUxL4" # GANTI URL DISINI JIKA PERLU
            sh = client.open_by_url(url)
            ws = sh.worksheet(sheet_name)
            df = pd.DataFrame(ws.get_all_records())
        except Exception as e:
            # Silent fail, lanjut ke CSV
            pass

    # 2. Coba CSV Lokal (Fallback)
    if df.empty:
        # Coba beberapa variasi nama file (karena upload manual kadang beda nama)
        candidates = [
            csv_filename,
            f"Supply Chain_Data.xlsx - {csv_filename}",
            f"data/{csv_filename}" 
        ]
        for f in candidates:
            if os.path.exists(f):
                try:
                    df = pd.read_csv(f)
                    break
                except: continue
    
    # 3. Bersihkan Kolom
    if not df.empty:
        df.columns = [str(c).strip().replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
    
    return df

# --- MAIN APP LOGIC ---
def main():
    st.markdown("<h1 style='text-align: center;'>🏭 Supply Chain Command Center</h1>", unsafe_allow_html=True)
    
    # 1. Load Data
    client, source_mode = get_data_engine()
    
    with st.spinner(f"🚀 Mengambil data (Mode: {source_mode.upper()})..."):
        df_prod = load_dataset(source_mode, client, "Product_Master", "Product_Master.csv")
        df_stock = load_dataset(source_mode, client, "Stock_Onhand", "Stock_Onhand.csv")
        df_rofo = load_dataset(source_mode, client, "Rofo", "Rofo.csv")
        df_sales = load_dataset(source_mode, client, "Sales", "Sales.csv")
        df_po = load_dataset(source_mode, client, "PO", "PO.csv")
    
    # Validasi Data Minimal
    if df_prod.empty or df_stock.empty:
        st.error("❌ Data kritis (Product/Stock) tidak ditemukan! Pastikan file CSV sudah di-upload ke repo atau Secrets GSheet benar.")
        st.stop()

    # --- SIDEBAR: SIMULATION CONTROL ---
    st.sidebar.header("🎛️ Simulation & Filter")
    
    # Filter Brand
    all_brands = df_prod['Brand'].unique().tolist() if 'Brand' in df_prod.columns else []
    sel_brands = st.sidebar.multiselect("Filter Brand", all_brands, default=all_brands)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 What-If Analysis")
    demand_shock = st.sidebar.slider("Simulasi Perubahan Demand (%)", -50, 50, 0, 5, help="Geser untuk melihat dampak kenaikan/penurunan demand terhadap ketahanan stok.")
    shock_factor = 1 + (demand_shock / 100)

    # --- DATA PROCESSING ---
    # 1. Master Data Join
    master = df_prod.copy()
    if 'SKU_ID' not in master.columns: st.error("Kolom 'SKU_ID' hilang di Product Master!"); st.stop()
    
    # 2. Join Stock
    # Pastikan tipe data sama (string) untuk join
    master['SKU_ID'] = master['SKU_ID'].astype(str)
    df_stock['SKU_ID'] = df_stock['SKU_ID'].astype(str)
    
    # Cari kolom qty yg benar
    qty_col = next((c for c in ['Stock_Qty', 'Stock_Qty', 'Quantity'] if c in df_stock.columns), None)
    if qty_col:
        stock_agg = df_stock.groupby('SKU_ID')[qty_col].sum().reset_index().rename(columns={qty_col: 'Onhand_Qty'})
        master = pd.merge(master, stock_agg, on='SKU_ID', how='left')
    else:
        master['Onhand_Qty'] = 0
    master['Onhand_Qty'] = master['Onhand_Qty'].fillna(0)

    # 3. Join Forecast (Rofo)
    # Ambil rata-rata forecast per SKU
    rofo_cols = [c for c in df_rofo.columns if any(m in c.lower() for m in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])]
    if rofo_cols:
        # Bersihkan data non-numeric
        for c in rofo_cols:
            df_rofo[c] = pd.to_numeric(df_rofo[c], errors='coerce').fillna(0)
        
        df_rofo['Avg_Forecast'] = df_rofo[rofo_cols].mean(axis=1) * shock_factor # APLIKASI SIMULASI DISINI
        df_rofo['SKU_ID'] = df_rofo['SKU_ID'].astype(str)
        master = pd.merge(master, df_rofo[['SKU_ID', 'Avg_Forecast']], on='SKU_ID', how='left')
    else:
        master['Avg_Forecast'] = 0
    master['Avg_Forecast'] = master['Avg_Forecast'].fillna(0)

    # 4. Join PO (In-Transit)
    if not df_po.empty and 'Status' in df_po.columns:
        df_po['SKU_ID'] = df_po['SKU_ID'].astype(str)
        # Asumsi status PO yg belum datang
        open_po = df_po[df_po['Status'].astype(str).str.lower().isin(['open', 'in transit', 'confirmed'])]
        po_agg = open_po.groupby('SKU_ID')['Qty'].sum().reset_index().rename(columns={'Qty': 'Intransit_Qty'})
        master = pd.merge(master, po_agg, on='SKU_ID', how='left')
    else:
        master['Intransit_Qty'] = 0
    master['Intransit_Qty'] = master['Intransit_Qty'].fillna(0)

    # --- FILTERING ---
    if 'Brand' in master.columns:
        master = master[master['Brand'].isin(sel_brands)]

    # --- KALKULASI METRIK SUPPLY CHAIN ---
    # DOI (Days of Inventory)
    master['Daily_Demand'] = master['Avg_Forecast'] / 30
    master['DOI'] = np.where(master['Daily_Demand'] > 0, 
                             (master['Onhand_Qty'] + master['Intransit_Qty']) / master['Daily_Demand'], 
                             999) # 999 = Dead Stock / No Forecast
    
    # Status Stok
    def get_status(row):
        if row['Onhand_Qty'] == 0: return '🔴 Stockout'
        if row['DOI'] < 30: return '🟠 Critical (<30 Days)'
        if row['DOI'] > 120: return '🔵 Overstock (>120 Days)'
        if row['Daily_Demand'] == 0: return '⚪ Dead Stock'
        return '🟢 Healthy'
    
    master['Status'] = master.apply(get_status, axis=1)
    
    # Nilai Inventory (Valuation)
    price_col = next((c for c in master.columns if 'price' in c.lower() or 'cogs' in c.lower()), None)
    if price_col:
        master['Inv_Value'] = master['Onhand_Qty'] * master[price_col]
    else:
        master['Inv_Value'] = 0

    # --- DASHBOARD TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Summary", "📦 Inventory Health", "🔮 Forecast & Simulation", "📋 Replenishment Plan"])

    with tab1:
        # KPI Cards Custom
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            val = master['Inv_Value'].sum()
            st.markdown(f"""<div class="metric-container">
                <p class="metric-label">Total Inventory Value</p>
                <p class="big-number">Rp {val/1e9:,.2f} M</p>
            </div>""", unsafe_allow_html=True)
            
        with kpi2:
            count = len(master)
            st.markdown(f"""<div class="metric-container">
                <p class="metric-label">Total SKU Active</p>
                <p class="big-number">{count}</p>
            </div>""", unsafe_allow_html=True)
            
        with kpi3:
            stockout = len(master[master['Status'].str.contains('Stockout')])
            st.markdown(f"""<div class="metric-container">
                <p class="metric-label">Stockout SKU</p>
                <p class="big-number" style="color: #d62728;">{stockout}</p>
                <p class="trend-negative">Lost Sales Risk!</p>
            </div>""", unsafe_allow_html=True)
            
        with kpi4:
            overstock = len(master[master['Status'].str.contains('Overstock')])
            st.markdown(f"""<div class="metric-container">
                <p class="metric-label">Overstock SKU</p>
                <p class="big-number" style="color: #ff7f0e;">{overstock}</p>
                <p class="trend-negative">Cashflow Locked</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("### 📈 Inventory Composition")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            fig_pie = px.pie(master, names='Status', title='Stock Health Distribution', 
                             color='Status', 
                             color_discrete_map={
                                 '🟢 Healthy':'#2ca02c', 
                                 '🟠 Critical (<30 Days)':'#ff7f0e', 
                                 '🔴 Stockout':'#d62728', 
                                 '🔵 Overstock (>120 Days)':'#1f77b4',
                                 '⚪ Dead Stock': '#7f7f7f'
                             }, hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            if price_col:
                top_val = master.sort_values('Inv_Value', ascending=False).head(10)
                fig_bar = px.bar(top_val, x='Inv_Value', y='Product_Name', orientation='h',
                                 title='Top 10 Value Holders (Pareto)', text_auto='.2s',
                                 color='Inv_Value', color_continuous_scale='Blues')
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("🔍 Stock Level Analysis (DOI)")
        st.info("Days of Inventory (DOI) = Stok Saat Ini / Rata-rata Penualan Harian (Forecast).")
        
        fig_hist = px.histogram(master[master['DOI'] < 365], x='DOI', nbins=50, 
                                title='Distribusi Ketahanan Stok (Days)',
                                color='Status', marginal='box')
        fig_hist.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Min Safe (30)")
        fig_hist.add_vline(x=120, line_dash="dash", line_color="blue", annotation_text="Max Limit (120)")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.dataframe(master[['SKU_ID', 'Product_Name', 'Onhand_Qty', 'Avg_Forecast', 'DOI', 'Status']].sort_values('DOI'), use_container_width=True)

    with tab3:
        st.subheader("🔮 Projected Inventory (Forecasted)")
        st.markdown(f"**Skenario Simulasi:** Demand berubah sebesar **{demand_shock}%**")
        
        sel_sku = st.selectbox("Pilih SKU untuk Proyeksi:", master['SKU_ID'].unique())
        
        # Ambil data forecast SKU terpilih
        sku_data = master[master['SKU_ID'] == sel_sku].iloc[0]
        sku_rofo = df_rofo[df_rofo['SKU_ID'] == sel_sku]
        
        if not sku_rofo.empty:
            # Buat grafik proyeksi run-down stok
            months = [c for c in rofo_cols] # Kolom bulan dari Rofo
            monthly_forecast = sku_rofo[months].values.flatten() * shock_factor
            
            curr_stock = sku_data['Onhand_Qty']
            proj_stock = []
            
            for fc in monthly_forecast:
                curr_stock = curr_stock - fc
                # Tambah logika PO masuk disini jika ada data tanggal PO
                proj_stock.append(curr_stock)
                
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(x=months, y=proj_stock, mode='lines+markers', name='Projected Stock'))
            fig_proj.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Stockout Point')
            
            fig_proj.update_layout(title=f"Proyeksi Stok 12 Bulan ke Depan: {sku_data['Product_Name']}",
                                   xaxis_title="Bulan", yaxis_title="Estimasi Sisa Stok")
            st.plotly_chart(fig_proj, use_container_width=True)
            
            if any(s < 0 for s in proj_stock):
                st.warning(f"⚠️ Peringatan: SKU ini diprediksi akan Stockout dalam periode forecast dengan simulasi demand {demand_shock}%!")
        else:
            st.warning("Tidak ada data forecast untuk SKU ini.")

    with tab4:
        st.subheader("📋 Replenishment Advice (Plan Order)")
        
        # Simple Replenishment Logic
        target_doi = 60 # Target stok 2 bulan
        master['Target_Stock'] = master['Daily_Demand'] * target_doi
        master['Net_Requirement'] = master['Target_Stock'] - (master['Onhand_Qty'] + master['Intransit_Qty'])
        master['Suggest_PO'] = master['Net_Requirement'].apply(lambda x: x if x > 0 else 0)
        
        plan_df = master[master['Suggest_PO'] > 0][['SKU_ID', 'Product_Name', 'Onhand_Qty', 'Intransit_Qty', 'Daily_Demand', 'Suggest_PO']].sort_values('Suggest_PO', ascending=False)
        
        st.success(f"Ditemukan {len(plan_df)} SKU yang perlu di-restock untuk mencapai coverage {target_doi} hari.")
        
        st.dataframe(plan_df.style.background_gradient(subset=['Suggest_PO'], cmap='Greens'), use_container_width=True)
        
        # Download Button
        csv = plan_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download PO Plan (CSV)", csv, "replenishment_plan.csv", "text/csv")

if __name__ == "__main__":
    main()
