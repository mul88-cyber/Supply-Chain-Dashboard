import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import gspread
from google.oauth2.service_account import Credentials
from dateutil.relativedelta import relativedelta
import warnings
from tenacity import retry, stop_after_attempt, wait_exponential
import math
warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Inventory Intelligence Pro v7",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# THEME DEFINITIONS
# ==============================================================================
THEMES = {
    "🌐 Corporate Blue (Light)": {
        "id": "corporate_blue",
        "bg":          "#F0F4FF",
        "sidebar_bg":  "#FFFFFF",
        "card_bg":     "#FFFFFF",
        "text":        "#0F172A",
        "text_muted":  "#475569",
        "border":      "#CBD5E1",
        "accent1":     "#2563EB",   # vivid blue
        "accent2":     "#7C3AED",   # vivid purple
        "tab_active":  "linear-gradient(135deg,#2563EB,#7C3AED)",
        "tab_inactive":"#E2E8F0",
        "tab_text_inactive": "#334155",
        "header_grad": "linear-gradient(90deg,#2563EB 0%,#7C3AED 100%)",
        "chart_bg":    "#FFFFFF",
        "chart_paper": "#FFFFFF",
        "chart_font":  "#0F172A",
        "chart_grid":  "rgba(15,23,42,0.06)",
        "card_shadow": "0 4px 20px rgba(37,99,235,0.12)",
        "metric_border":"#2563EB",
        "plotly_template": "plotly_white",
        "colors":["#2563EB","#7C3AED","#0EA5E9","#06B6D4","#10B981","#F59E0B","#EF4444"],
    },
    "🌙 Dark Corporate": {
        "id": "dark_corporate",
        "bg":          "#0A0F1E",
        "sidebar_bg":  "#0D1425",
        "card_bg":     "#111827",
        "text":        "#F1F5F9",
        "text_muted":  "#94A3B8",
        "border":      "#1E293B",
        "accent1":     "#3B82F6",
        "accent2":     "#8B5CF6",
        "tab_active":  "linear-gradient(135deg,#3B82F6,#8B5CF6)",
        "tab_inactive":"#1E293B",
        "tab_text_inactive": "#94A3B8",
        "header_grad": "linear-gradient(90deg,#3B82F6 0%,#8B5CF6 50%,#06B6D4 100%)",
        "chart_bg":    "#111827",
        "chart_paper": "#111827",
        "chart_font":  "#E2E8F0",
        "chart_grid":  "rgba(148,163,184,0.08)",
        "card_shadow": "0 4px 24px rgba(0,0,0,0.5)",
        "metric_border":"#3B82F6",
        "plotly_template": "plotly_dark",
        "colors":["#3B82F6","#8B5CF6","#06B6D4","#10B981","#F59E0B","#F43F5E","#A78BFA"],
    },
    "🔥 Midnight Red": {
        "id": "midnight_red",
        "bg":          "#0D0A0A",
        "sidebar_bg":  "#120D0D",
        "card_bg":     "#1A1010",
        "text":        "#FAF5F5",
        "text_muted":  "#9CA3AF",
        "border":      "#2D1515",
        "accent1":     "#EF4444",
        "accent2":     "#F97316",
        "tab_active":  "linear-gradient(135deg,#EF4444,#F97316)",
        "tab_inactive":"#1F1111",
        "tab_text_inactive": "#9CA3AF",
        "header_grad": "linear-gradient(90deg,#EF4444 0%,#F97316 50%,#FBBF24 100%)",
        "chart_bg":    "#1A1010",
        "chart_paper": "#1A1010",
        "chart_font":  "#FAF5F5",
        "chart_grid":  "rgba(250,245,245,0.06)",
        "card_shadow": "0 4px 24px rgba(239,68,68,0.2)",
        "metric_border":"#EF4444",
        "plotly_template": "plotly_dark",
        "colors":["#EF4444","#F97316","#FBBF24","#10B981","#3B82F6","#8B5CF6","#EC4899"],
    },
    "🌿 Executive Green": {
        "id": "exec_green",
        "bg":          "#F0FDF4",
        "sidebar_bg":  "#FFFFFF",
        "card_bg":     "#FFFFFF",
        "text":        "#052E16",
        "text_muted":  "#166534",
        "border":      "#BBF7D0",
        "accent1":     "#059669",
        "accent2":     "#0284C7",
        "tab_active":  "linear-gradient(135deg,#059669,#0284C7)",
        "tab_inactive":"#D1FAE5",
        "tab_text_inactive": "#065F46",
        "header_grad": "linear-gradient(90deg,#059669 0%,#0284C7 100%)",
        "chart_bg":    "#FFFFFF",
        "chart_paper": "#FFFFFF",
        "chart_font":  "#052E16",
        "chart_grid":  "rgba(5,46,22,0.06)",
        "card_shadow": "0 4px 20px rgba(5,150,105,0.15)",
        "metric_border":"#059669",
        "plotly_template": "plotly_white",
        "colors":["#059669","#0284C7","#7C3AED","#F59E0B","#EF4444","#0EA5E9","#10B981"],
    },
    "⚡ Neon Dark": {
        "id": "neon_dark",
        "bg":          "#05050F",
        "sidebar_bg":  "#08081A",
        "card_bg":     "#0D0D24",
        "text":        "#E0E7FF",
        "text_muted":  "#818CF8",
        "border":      "#1E1B4B",
        "accent1":     "#6366F1",
        "accent2":     "#22D3EE",
        "tab_active":  "linear-gradient(135deg,#6366F1,#22D3EE)",
        "tab_inactive":"#0F0F2D",
        "tab_text_inactive": "#818CF8",
        "header_grad": "linear-gradient(90deg,#6366F1 0%,#22D3EE 50%,#A78BFA 100%)",
        "chart_bg":    "#0D0D24",
        "chart_paper": "#0D0D24",
        "chart_font":  "#E0E7FF",
        "chart_grid":  "rgba(99,102,241,0.1)",
        "card_shadow": "0 0 20px rgba(99,102,241,0.3), 0 4px 15px rgba(0,0,0,0.5)",
        "metric_border":"#6366F1",
        "plotly_template": "plotly_dark",
        "colors":["#6366F1","#22D3EE","#A78BFA","#34D399","#FBBF24","#F43F5E","#38BDF8"],
    },
}

# ==============================================================================
# GLOBAL BASE CSS  (mobile-first, theme-agnostic structure)
# ==============================================================================
BASE_CSS = """
<style>
/* ── PRINT ───────────────────────────────────────────────────────── */
@media print {
    *{overflow:visible!important;position:static!important;display:block!important;
      float:none!important;height:auto!important;max-height:none!important;
      width:auto!important;max-width:none!important;
      -webkit-print-color-adjust:exact!important;print-color-adjust:exact!important;
      break-inside:avoid!important;}
    [data-testid="stSidebar"],[data-testid="stHeader"],.stButton,.stDeployButton,
    footer,.stDownloadButton,.stActionButton,button,.stAlert{
        display:none!important;height:0!important;width:0!important;
        opacity:0!important;visibility:hidden!important;}
    [data-testid="stAppViewContainer"],[data-testid="stMain"]{
        position:static!important;width:100vw!important;height:auto!important;
        margin:0!important;padding:0!important;overflow:visible!important;display:block!important;}
}

/* ── MOBILE RESPONSIVE ───────────────────────────────────────────── */
/* Stack columns on small screens */
@media (max-width: 768px) {
    /* Reduce padding on mobile */
    [data-testid="block-container"] {
        padding: 0.5rem !important;
    }
    /* Make columns stack vertically */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    /* Smaller header on mobile */
    .main-header {
        font-size: 1.6rem !important;
        padding: 0.5rem !important;
    }
    /* Cards full width on mobile */
    .grad-card {
        margin-bottom: 0.6rem !important;
    }
    .grad-value {
        font-size: 1.5rem !important;
    }
    /* Tabs scrollable on mobile */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        flex-wrap: nowrap !important;
        scrollbar-width: none !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }
    .stTabs [data-baseweb="tab"] {
        min-width: max-content !important;
        font-size: 0.78rem !important;
        padding: 8px 12px !important;
        height: 40px !important;
    }
    /* Sidebar collapsed by default hint */
    [data-testid="stSidebar"] {
        min-width: 0 !important;
    }
    /* Metric cards responsive */
    [data-testid="stMetric"] {
        padding: 0.6rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    /* Charts full width */
    .stPlotlyChart {
        width: 100% !important;
    }
    /* Dataframe horizontal scroll */
    .stDataFrame {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }
    /* Hide sidebar toggle label on very small screens */
    .st-emotion-cache-1cypcdb { font-size: 0.7rem !important; }
}

@media (max-width: 480px) {
    .main-header { font-size: 1.2rem !important; }
    .grad-value  { font-size: 1.3rem !important; }
    .grad-label  { font-size: 0.7rem !important; }
    [data-testid="stHorizontalBlock"] > div { min-width: 48% !important; flex: 1 1 48% !important; }
}

/* ── CARD BASE ───────────────────────────────────────────────────── */
.grad-card{
    border-radius:14px;padding:1.2rem 1.4rem;color:white;
    transition:transform .25s ease, box-shadow .25s ease;
    position:relative;overflow:hidden;margin-bottom:.8rem;
}
.grad-card:hover{transform:translateY(-4px);}
.grad-card::before{
    content:"";position:absolute;top:-40%;right:-20%;
    width:180px;height:180px;border-radius:50%;
    background:rgba(255,255,255,0.07);pointer-events:none;
}
.grad-label{font-size:.72rem;font-weight:700;text-transform:uppercase;
            letter-spacing:1.2px;opacity:.85;margin-bottom:.3rem;}
.grad-value{font-size:1.9rem;font-weight:800;margin-bottom:.15rem;
            text-shadow:0 2px 6px rgba(0,0,0,.2);line-height:1.1;}
.grad-sub{font-size:.8rem;font-weight:500;opacity:.88;
          display:flex;align-items:center;gap:5px;flex-wrap:wrap;}
.pill{background:rgba(255,255,255,.22);padding:2px 9px;border-radius:20px;
      font-size:.72rem;font-weight:600;backdrop-filter:blur(6px);
      border:1px solid rgba(255,255,255,.15);}

/* ── ALERT BANNERS ───────────────────────────────────────────────── */
.alert-critical{
    background:linear-gradient(135deg,#DC2626,#B91C1C);
    color:white;border-radius:10px;padding:1rem;margin:.4rem 0;
    border-left:5px solid #7F1D1D;font-weight:700;
    box-shadow:0 4px 12px rgba(220,38,38,0.3);}
.alert-warning{
    background:linear-gradient(135deg,#D97706,#B45309);
    color:white;border-radius:10px;padding:1rem;margin:.4rem 0;
    border-left:5px solid #78350F;font-weight:700;
    box-shadow:0 4px 12px rgba(217,119,6,0.3);}
.alert-ok{
    background:linear-gradient(135deg,#059669,#047857);
    color:white;border-radius:10px;padding:1rem;margin:.4rem 0;
    border-left:5px solid #064E3B;font-weight:700;}

/* ── DATAFRAME ───────────────────────────────────────────────────── */
.stDataFrame{border-radius:10px;overflow:hidden;}

/* ── METRIC CARD NATIVE STREAMLIT ────────────────────────────────── */
[data-testid="stMetric"]{
    border-radius:12px;padding:1rem 1.2rem;
    transition:transform .2s ease;
}
[data-testid="stMetric"]:hover{ transform:translateY(-2px); }

/* ── SMOOTH SCROLLBAR ────────────────────────────────────────────── */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{border-radius:3px;}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ==============================================================================
# THEME SELECTION — must happen BEFORE sidebar renders widgets
# ==============================================================================
# Use session state so theme persists across reruns
if "theme_name" not in st.session_state:
    st.session_state.theme_name = "🌐 Corporate Blue (Light)"

# Quick theme selector in a top-of-page row (mobile friendly)
_top_col1, _top_col2 = st.columns([3, 1])
with _top_col2:
    _selected_theme = st.selectbox(
        "🎨 Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme_name),
        key="theme_selector_top",
        label_visibility="collapsed",
    )
    st.session_state.theme_name = _selected_theme

T = THEMES[st.session_state.theme_name]   # active theme dict

# Inject dynamic theme CSS
def _theme_css(t):
    is_dark = t["id"] in ("dark_corporate","midnight_red","neon_dark")
    scrollbar_color = t["accent1"]
    neon_glow = f"box-shadow:{t['card_shadow']};" if t["id"]=="neon_dark" else ""
    return f"""
<style>
/* ── APP BACKGROUND ──────────────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] > div {{
    background-color:{t["bg"]} !important;
    color:{t["text"]} !important;
}}
/* ── SIDEBAR ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color:{t["sidebar_bg"]} !important;
    border-right:1px solid {t["border"]} !important;
}}
[data-testid="stSidebar"] * {{
    color:{t["text"]} !important;
}}
/* ── ALL TEXT ────────────────────────────────────────────────────── */
h1,h2,h3,h4,h5,h6,p,span,div,label {{
    color:{t["text"]};
}}
.stMarkdown, .stCaption, [data-testid="stCaptionContainer"] {{
    color:{t["text_muted"]} !important;
}}
/* ── HEADER GRADIENT TEXT ────────────────────────────────────────── */
.main-header {{
    background:{t["header_grad"]};
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    font-size:2.6rem;font-weight:900;
    text-align:center;padding:.8rem .5rem .4rem;
    margin-bottom:.2rem;
}}
/* ── TABS ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background:{t["bg"]} !important;
    gap:6px;padding:6px 0;
}}
.stTabs [data-baseweb="tab"] {{
    background:{t["tab_inactive"]} !important;
    color:{t["tab_text_inactive"]} !important;
    border-radius:10px 10px 0 0 !important;
    font-weight:700;font-size:.88rem;
    border:1px solid {t["border"]} !important;
    transition:all .2s ease;
}}
.stTabs [aria-selected="true"] {{
    background:{t["tab_active"]} !important;
    color:#FFFFFF !important;
    border-color:transparent !important;
    box-shadow:0 4px 14px rgba(0,0,0,.25) !important;
}}
/* ── NATIVE STREAMLIT METRICS ────────────────────────────────────── */
[data-testid="stMetric"] {{
    background:{t["card_bg"]} !important;
    border:1px solid {t["border"]} !important;
    border-top:3px solid {t["metric_border"]} !important;
    box-shadow:{t["card_shadow"]} !important;
    {neon_glow}
}}
[data-testid="stMetricValue"] {{
    color:{t["accent1"]} !important;
    font-weight:800 !important;
}}
[data-testid="stMetricLabel"] {{
    color:{t["text_muted"]} !important;
    font-weight:600 !important;
}}
[data-testid="stMetricDelta"] {{
    font-weight:700 !important;
}}
/* ── DATAFRAME ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border:1px solid {t["border"]} !important;
    box-shadow:{t["card_shadow"]} !important;
    background:{t["card_bg"]} !important;
}}
/* ── SLIDERS & INPUTS ────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div {{
    background:{t["accent1"]} !important;
}}
/* ── BUTTONS ─────────────────────────────────────────────────────── */
[data-testid="baseButton-primary"] {{
    background:{t["tab_active"]} !important;
    border:none !important;
    font-weight:700 !important;
}}
/* ── EXPANDER ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border:1px solid {t["border"]} !important;
    background:{t["card_bg"]} !important;
    border-radius:10px !important;
}}
/* ── BLOCK CONTAINER ─────────────────────────────────────────────── */
[data-testid="block-container"] {{
    background:{t["bg"]} !important;
}}
/* ── SCROLLBAR ACCENT ────────────────────────────────────────────── */
::-webkit-scrollbar-thumb {{ background:{scrollbar_color}; }}
/* ── SELECT BOX & MULTISELECT ────────────────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {{
    background:{t["card_bg"]} !important;
    border-color:{t["border"]} !important;
    color:{t["text"]} !important;
}}
</style>
"""

st.markdown(_theme_css(T), unsafe_allow_html=True)

# Plotly theme helper — inject into every chart call
def plotly_layout(fig, title="", height=420):
    """Apply active theme to any plotly figure."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color=T["chart_font"])) if title else {},
        height=height,
        plot_bgcolor=T["chart_bg"],
        paper_bgcolor=T["chart_paper"],
        font=dict(color=T["chart_font"], size=12),
        xaxis=dict(showgrid=False, tickfont=dict(color=T["chart_font"])),
        yaxis=dict(gridcolor=T["chart_grid"], tickfont=dict(color=T["chart_font"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["chart_font"])),
        margin=dict(t=50, b=30, l=30, r=30),
        hoverlabel=dict(bgcolor=T["card_bg"], font_color=T["text"], font_size=13),
    )
    return fig

# ==============================================================================
# HEADER  (rendered after theme injection)
# ==============================================================================
st.markdown(f'<h1 class="main-header">💰 FORECAST & INVENTORY CONTROL PRO v7</h1>', unsafe_allow_html=True)
st.caption(f"🚀 D2C Demand Planner · {T['id'].replace('_',' ').title()} · Updated: {datetime.now().strftime('%d %B %Y %H:%M')}")

# ==============================================================================
# HELPERS — PERFORMANCE (vectorised, no iterrows)
# ==============================================================================

def validate_month_format(month_str):
    if pd.isna(month_str):
        return datetime.now()
    month_str = str(month_str).strip().upper()
    month_map = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
                 'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    for fmt in ['%b-%Y','%b-%y','%B %Y','%m/%Y','%Y-%m']:
        try:
            return datetime.strptime(month_str, fmt)
        except:
            pass
    for mn, num in month_map.items():
        if mn in month_str:
            yr = month_str.replace(mn,'').replace('-','').replace(' ','').strip()
            year = (2000+int(yr)) if (yr and yr.isdigit() and len(yr)==2) else (int(yr) if (yr and yr.isdigit()) else datetime.now().year)
            return datetime(year, num, 1)
    return datetime.now()


def add_product_info(df, df_product):
    """Vectorised merge — replaces any loop-based product lookup."""
    if df.empty or df_product.empty or 'SKU_ID' not in df.columns:
        return df
    price_cols = [c for c in ['Floor_Price','Net_Order_Price'] if c in df_product.columns]
    keep = ['SKU_ID','Product_Name','Brand','SKU_Tier','Status'] + price_cols
    keep = [c for c in keep if c in df_product.columns]
    info = df_product[keep].drop_duplicates('SKU_ID')
    drop = [c for c in keep if c != 'SKU_ID' and c in df.columns]
    return pd.merge(df.drop(columns=drop, errors='ignore'), info, on='SKU_ID', how='left')


def fmt_money(x):
    if x >= 1e9:  return f"Rp {x/1e9:,.1f} M"
    if x >= 1e6:  return f"Rp {x/1e6:,.1f} Jt"
    return f"Rp {x:,.0f}"


def fmt_card(title, icon, val, sub, gradient):
    return f"""
    <div class="grad-card" style="background:{gradient};">
      <div class="grad-label">{title}</div>
      <div class="grad-value">{icon} {val}</div>
      <div class="grad-sub"><span class="pill">{sub}</span></div>
    </div>"""

# ==============================================================================
# GOOGLE SHEETS CONNECTION
# ==============================================================================

@st.cache_resource(show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def init_gsheet():
    try:
        skey = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(skey, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Koneksi Gagal: {e}")
        return None


def safe_read(client, url, sheet_name):
    try:
        ws = client.open_by_url(url).worksheet(sheet_name)
        raw = ws.get_all_values()
        if len(raw) < 2:
            return pd.DataFrame()
        headers = [str(h).strip() for h in raw[0]]
        df = pd.DataFrame(raw[1:], columns=headers)
        return df.loc[:, df.columns != '']
    except:
        return pd.DataFrame()


def melt_wide(df, id_cols, value_name):
    """Melt wide month-column format to long."""
    month_cols = [c for c in df.columns
                  if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
    if not month_cols:
        return pd.DataFrame()
    id_cols = [c for c in id_cols if c in df.columns]
    out = df.melt(id_vars=id_cols, value_vars=month_cols, var_name='Month_Label', value_name=value_name)
    out[value_name] = pd.to_numeric(out[value_name], errors='coerce').fillna(0)
    out['Month'] = out['Month_Label'].apply(validate_month_format)
    return out


# ==============================================================================
# DATA LOADING  (single cached function — all sheets)
# ==============================================================================

@st.cache_data(ttl=300, max_entries=3, show_spinner=False)
def load_all_data(_client):
    url = st.secrets["gsheet_url"]
    D = {}

    # ── Product Master ────────────────────────────────────────────
    try:
        ws = _client.open_by_url(url).worksheet("Product_Master")
        df_prod = pd.DataFrame(ws.get_all_records())
        df_prod.columns = [c.strip().replace(' ','_') for c in df_prod.columns]
        for c in ['Floor_Price','Net_Order_Price']:
            if c in df_prod.columns:
                df_prod[c] = pd.to_numeric(df_prod[c], errors='coerce').fillna(0)
        if 'Status' not in df_prod.columns:
            df_prod['Status'] = 'Active'
        D['product']        = df_prod
        D['product_active'] = df_prod[df_prod['Status'].str.upper()=='ACTIVE'].copy()
        active_skus         = D['product_active']['SKU_ID'].tolist()
    except Exception as e:
        st.error(f"Product Master error: {e}")
        return D

    # ── Sales ─────────────────────────────────────────────────────
    try:
        ws = _client.open_by_url(url).worksheet("Sales")
        df_s = pd.DataFrame(ws.get_all_records())
        df_s.columns = [c.strip() for c in df_s.columns]
        df_s = melt_wide(df_s, ['SKU_ID','SKU_Name','Product_Name','Brand','SKU_Tier'], 'Sales_Qty')
        df_s = df_s[df_s['SKU_ID'].isin(active_skus)]
        D['sales'] = add_product_info(df_s, df_prod).sort_values('Month')
    except Exception as e:
        st.warning(f"Sales: {e}")
        D['sales'] = pd.DataFrame()

    # ── Rofo / Forecast ───────────────────────────────────────────
    try:
        ws = _client.open_by_url(url).worksheet("Rofo")
        df_r = pd.DataFrame(ws.get_all_records())
        df_r.columns = [c.strip() for c in df_r.columns]
        df_r = melt_wide(df_r, ['SKU_ID','Product_Name','Brand'], 'Forecast_Qty')
        df_r = df_r[df_r['SKU_ID'].isin(active_skus)]
        D['forecast'] = add_product_info(df_r, df_prod)
    except Exception as e:
        st.warning(f"Rofo: {e}")
        D['forecast'] = pd.DataFrame()

    # ── PO ────────────────────────────────────────────────────────
    try:
        ws = _client.open_by_url(url).worksheet("PO")
        df_p = pd.DataFrame(ws.get_all_records())
        df_p.columns = [c.strip() for c in df_p.columns]
        df_p = melt_wide(df_p, ['SKU_ID'], 'PO_Qty')
        df_p = df_p[df_p['SKU_ID'].isin(active_skus)]
        D['po'] = add_product_info(df_p, df_prod)
    except Exception as e:
        st.warning(f"PO: {e}")
        D['po'] = pd.DataFrame()

    # ── Stock On-Hand ─────────────────────────────────────────────
    try:
        df_st = safe_read(_client, url, "Stock_Onhand")
        if not df_st.empty and 'SKU_ID' in df_st.columns and 'Qty_Available' in df_st.columns:
            df_st = df_st.rename(columns={'Qty_Available':'Stock_Qty','Product_Code':'Anchanto_Code',
                                          'Stock_Category':'Stock_Category','Expiry_Date':'Expiry_Date'})
            df_st['Stock_Qty'] = pd.to_numeric(df_st['Stock_Qty'], errors='coerce').fillna(0)
            df_st['SKU_ID']    = df_st['SKU_ID'].astype(str).str.strip()
            D['stock'] = df_st
        else:
            D['stock'] = pd.DataFrame(columns=['SKU_ID','Stock_Qty'])
    except Exception as e:
        st.warning(f"Stock: {e}")
        D['stock'] = pd.DataFrame(columns=['SKU_ID','Stock_Qty'])

    # ── Ecomm Forecast 2026 ───────────────────────────────────────
    try:
        ws = _client.open_by_url(url).worksheet("Forecast_2026_Ecomm")
        df_e = pd.DataFrame(ws.get_all_records())
        df_e.columns = [c.strip().replace(' ','_') for c in df_e.columns]
        month_cols = [c for c in df_e.columns
                      if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        for c in month_cols:
            df_e[c] = pd.to_numeric(df_e[c], errors='coerce').fillna(0)
        D['ecomm_forecast']            = df_e
        D['ecomm_forecast_month_cols'] = month_cols
    except:
        D['ecomm_forecast']            = pd.DataFrame()
        D['ecomm_forecast_month_cols'] = []

    # ── Reseller Forecast 2026 ────────────────────────────────────
    try:
        ws = _client.open_by_url(url).worksheet("Forecast_2026_Reseller")
        df_res = pd.DataFrame(ws.get_all_records())
        df_res.columns = [c.strip().replace(' ','_') for c in df_res.columns]
        all_mc = [c for c in df_res.columns
                  if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        for c in all_mc:
            df_res[c] = pd.to_numeric(df_res[c], errors='coerce').fillna(0)
        cutoff = datetime(2026, 1, 1)
        hist_cols  = []
        fcst_cols  = []
        for c in all_mc:
            try:
                cs = str(c).upper().replace('_',' ').replace('-',' ')
                parts = cs.split()
                mo = datetime.strptime(parts[0][:3], '%b').month
                yr_raw = ''.join(filter(str.isdigit, parts[1])) if len(parts)>1 else ''
                yr = (2000+int(yr_raw)) if len(yr_raw)==2 else (int(yr_raw) if yr_raw else datetime.now().year)
                (fcst_cols if datetime(yr,mo,1) >= cutoff else hist_cols).append(c)
            except:
                pass
        D['reseller_forecast']      = df_res
        D['reseller_all_months']    = all_mc
        D['reseller_hist_cols']     = hist_cols
        D['reseller_fcst_cols']     = fcst_cols
    except:
        D['reseller_forecast']   = pd.DataFrame()
        D['reseller_all_months'] = []
        D['reseller_hist_cols']  = []
        D['reseller_fcst_cols']  = []

    # ── Reseller sub-sheets ───────────────────────────────────────
    for sheet, key, val_col in [
        ("Sales_Reseller",      "sales_reseller",    "Sales_Qty"),
        ("Past_Rofo_Reseller",  "past_rofo_reseller","Forecast_Qty"),
        ("Past_PO_Reseller",    "past_po_reseller",  "PO_Qty"),
    ]:
        try:
            ws  = _client.open_by_url(url).worksheet(sheet)
            df_ = pd.DataFrame(ws.get_all_records())
            df_.columns = [c.strip() for c in df_.columns]
            D[key] = melt_wide(df_, ['SKU_ID','Brand','Product_Name','SKU_Tier','Floor_Price'], val_col)
        except:
            D[key] = pd.DataFrame()

    # ── BS Fulfillment Cost ───────────────────────────────────────
    try:
        ws   = _client.open_by_url(url).worksheet("BS_Fullfilment_Cost")
        df_b = pd.DataFrame(ws.get_all_records())
        df_b.columns = [c.strip() for c in df_b.columns]
        def clean_num(x):
            if isinstance(x, str):
                return pd.to_numeric(x.replace(',','').replace('%',''), errors='coerce')
            return x
        for c in ['Total Order(BS)','GMV (Fullfil By BS)','GMV Total (MP)','Total Cost','BSA','%Cost']:
            if c in df_b.columns:
                df_b[c] = df_b[c].apply(clean_num).fillna(0)
        df_b['Month_Date'] = pd.to_datetime(df_b['Month'], format='%b-%y', errors='coerce')
        D['fulfillment'] = df_b.sort_values('Month_Date')
    except Exception as e:
        st.warning(f"Fulfillment: {e}")
        D['fulfillment'] = pd.DataFrame()

    return D


# ==============================================================================
# ANALYTICS FUNCTIONS  (all vectorised)
# ==============================================================================

def calc_monthly_performance(df_forecast, df_po, df_product):
    """Vectorised monthly accuracy — no per-row loops."""
    results = {}
    if df_forecast.empty or df_po.empty:
        return results

    df_f = add_product_info(df_forecast, df_product)
    df_p = add_product_info(df_po, df_product)
    df_f = df_f[df_f['Forecast_Qty'] > 0]

    months = sorted(set(df_f['Month'].unique()) & set(df_p['Month'].unique()))
    for month in months:
        f_m = df_f[df_f['Month'] == month][['SKU_ID','Forecast_Qty','Product_Name','Brand','SKU_Tier']]
        p_m = df_p[df_p['Month'] == month][['SKU_ID','PO_Qty']]
        merged = pd.merge(f_m, p_m, on='SKU_ID', how='inner')
        if merged.empty:
            continue
        merged['PO_Rofo_Ratio']  = merged['PO_Qty'] / merged['Forecast_Qty'] * 100
        # Under  : PO/Rofo < 80%   (strictly less than)
        # Accurate: 80% <= PO/Rofo <= 120%  (inclusive both ends)
        # Over   : PO/Rofo > 120%  (strictly greater than)
        merged['Accuracy_Status'] = np.select(
            [
                merged['PO_Rofo_Ratio'] < 80,
                merged['PO_Rofo_Ratio'] > 120,
            ],
            ['Under', 'Over'],
            default='Accurate'
        )
        merged['APE'] = (merged['PO_Rofo_Ratio'] - 100).abs()

        sc   = merged['Accuracy_Status'].value_counts().to_dict()
        tot  = len(merged)
        acc  = sc.get('Accurate',0) / tot * 100 if tot else 0
        mape = merged['APE'].mean()

        results[month] = dict(
            accuracy=acc, mape=mape, total_records=tot,
            status_counts=sc, data=merged,
            under_skus  = merged[merged['Accuracy_Status']=='Under'],
            accurate_skus=merged[merged['Accuracy_Status']=='Accurate'],
            over_skus   = merged[merged['Accuracy_Status']=='Over'],
        )
    return results


def calc_inventory_metrics(df_stock, df_sales, df_product):
    """Vectorised inventory coverage — no iterrows."""
    if df_stock.empty:
        return {}
    # Aggregate stock to SKU level
    agg = df_stock.groupby('SKU_ID', as_index=False)['Stock_Qty'].sum()
    agg = add_product_info(agg, df_product)

    # 3-month average sales
    if not df_sales.empty:
        months = sorted(df_sales['Month'].unique())
        last3  = months[-3:] if len(months)>=3 else months
        avg_s  = (df_sales[df_sales['Month'].isin(last3)]
                  .groupby('SKU_ID', as_index=False)['Sales_Qty'].mean()
                  .rename(columns={'Sales_Qty':'Avg_Monthly_Sales_3M'}))
    else:
        avg_s = pd.DataFrame(columns=['SKU_ID','Avg_Monthly_Sales_3M'])

    inv = pd.merge(agg, avg_s, on='SKU_ID', how='left')
    inv['Avg_Monthly_Sales_3M'] = inv['Avg_Monthly_Sales_3M'].fillna(0)
    inv['Cover_Months'] = np.where(
        inv['Avg_Monthly_Sales_3M'] > 0,
        inv['Stock_Qty'] / inv['Avg_Monthly_Sales_3M'],
        999
    )
    inv['Inventory_Status'] = pd.cut(
        inv['Cover_Months'],
        bins=[-np.inf, 0.8, 1.5, np.inf],
        labels=['Need Replenishment','Ideal/Healthy','High Stock']
    ).astype(str)

    return dict(
        inventory_df = inv,
        high_stock   = inv[inv['Inventory_Status']=='High Stock'].sort_values('Cover_Months', ascending=False),
        low_stock    = inv[inv['Inventory_Status']=='Need Replenishment'].sort_values('Cover_Months'),
        total_stock  = inv['Stock_Qty'].sum(),
        total_skus   = len(inv),
        avg_cover    = inv[inv['Cover_Months']<999]['Cover_Months'].mean(),
        health_score = len(inv[inv['Inventory_Status']=='Ideal/Healthy']) / len(inv) * 100 if len(inv) else 0,
    )


def calc_financial(df_sales, df_product):
    """Calculate revenue, COGS, margin — vectorised."""
    if df_sales.empty or df_product.empty:
        return pd.DataFrame()
    df = add_product_info(df_sales, df_product)
    for c in ['Floor_Price','Net_Order_Price']:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)
    df['Revenue']          = df['Sales_Qty'] * df['Floor_Price']
    df['Cost']             = df['Sales_Qty'] * df['Net_Order_Price']
    df['Gross_Margin']     = df['Revenue'] - df['Cost']
    df['Margin_Percentage']= np.where(df['Revenue']>0, df['Gross_Margin']/df['Revenue']*100, 0)
    return df


def calc_yoy(df_sales):
    """Year-over-Year comparison table."""
    if df_sales.empty:
        return pd.DataFrame()
    df = df_sales.copy()
    df['Year']  = df['Month'].dt.year
    df['Mo_Num']= df['Month'].dt.month
    pivot = df.groupby(['Mo_Num','Year'])['Sales_Qty'].sum().unstack(fill_value=0)
    pivot.index = [datetime(2000,m,1).strftime('%b') for m in pivot.index]
    years = sorted(pivot.columns)
    if len(years) >= 2:
        pivot['YoY Growth %'] = ((pivot[years[-1]] - pivot[years[-2]]) / pivot[years[-2]].replace(0,np.nan) * 100).round(1)
    return pivot.reset_index().rename(columns={'Mo_Num':'Month'})


def calc_channel_accuracy(df_forecast, df_ecomm, df_rofo_reseller, df_po_reseller, df_po, df_product):
    """
    Forecast accuracy split by channel.
    - Ecomm   : df_forecast (long, has Month + Forecast_Qty) vs df_po (long, has Month + PO_Qty)
    - Reseller: df_rofo_reseller (long, past rofo) vs df_po_reseller (long, past PO)
    All inputs must be long-format DataFrames with columns [SKU_ID, Month, value_col].
    """
    result = {}

    def _safe_cols(df, required):
        """Return True only if df is non-empty and has all required columns."""
        return (not df.empty) and all(c in df.columns for c in required)

    # ── Ecomm channel ──────────────────────────────────────────────────────
    if _safe_cols(df_forecast, ['SKU_ID','Month','Forecast_Qty']) and \
       _safe_cols(df_po,       ['SKU_ID','Month','PO_Qty']):
        merged = pd.merge(
            df_forecast[['SKU_ID','Month','Forecast_Qty']],
            df_po      [['SKU_ID','Month','PO_Qty']],
            on=['SKU_ID','Month'], how='inner'
        )
        merged = merged[merged['Forecast_Qty'] > 0]
        if not merged.empty:
            merged['Acc'] = 100 - (merged['PO_Qty'] / merged['Forecast_Qty'] * 100 - 100).abs()
            r = merged.groupby('Month')['Acc'].mean().reset_index(name='Accuracy')
            r['Channel'] = 'Ecommerce'
            result['Ecommerce'] = r

    # ── Reseller channel ───────────────────────────────────────────────────
    # Uses past_rofo_reseller vs past_po_reseller (both long format from melt_wide)
    if _safe_cols(df_rofo_reseller, ['SKU_ID','Month','Forecast_Qty']) and \
       _safe_cols(df_po_reseller,   ['SKU_ID','Month','PO_Qty']):
        mrgd = pd.merge(
            df_rofo_reseller[['SKU_ID','Month','Forecast_Qty']],
            df_po_reseller  [['SKU_ID','Month','PO_Qty']],
            on=['SKU_ID','Month'], how='inner'
        )
        mrgd = mrgd[mrgd['Forecast_Qty'] > 0]
        if not mrgd.empty:
            mrgd['Acc'] = 100 - (mrgd['PO_Qty'] / mrgd['Forecast_Qty'] * 100 - 100).abs()
            r = mrgd.groupby('Month')['Acc'].mean().reset_index(name='Accuracy')
            r['Channel'] = 'Reseller'
            result['Reseller'] = r

    return result


def get_critical_alerts(inventory_metrics, df_product):
    """Generate smart alerts list for sidebar and exec summary."""
    alerts = []
    if 'inventory_df' not in inventory_metrics:
        return alerts
    inv = inventory_metrics['inventory_df']

    # Low stock alerts
    low  = inv[inv['Inventory_Status']=='Need Replenishment']
    high = inv[inv['Inventory_Status']=='High Stock']

    for _, row in low.iterrows():
        alerts.append(dict(
            level='critical',
            icon='🔴',
            msg=f"STOCKOUT RISK: **{row.get('Product_Name', row['SKU_ID'])}** — Cover {row['Cover_Months']:.1f} mo",
            sku=row['SKU_ID']
        ))
    for _, row in high[high['Cover_Months'] > 3].iterrows():
        alerts.append(dict(
            level='warning',
            icon='🟡',
            msg=f"OVERSTOCK: **{row.get('Product_Name', row['SKU_ID'])}** — Cover {row['Cover_Months']:.1f} mo",
            sku=row['SKU_ID']
        ))
    return alerts


# ==============================================================================
# INITIALISE
# ==============================================================================
client = init_gsheet()
if client is None:
    st.error("❌ Tidak dapat terhubung ke Google Sheets")
    st.stop()

with st.spinner('🔄 Loading data…'):
    D = load_all_data(client)

df_product        = D.get('product',        pd.DataFrame())
df_product_active = D.get('product_active', pd.DataFrame())
df_sales          = D.get('sales',          pd.DataFrame())
df_forecast       = D.get('forecast',       pd.DataFrame())
df_po             = D.get('po',             pd.DataFrame())
df_stock          = D.get('stock',          pd.DataFrame())
df_ecomm          = D.get('ecomm_forecast', pd.DataFrame())
ecomm_month_cols  = D.get('ecomm_forecast_month_cols', [])
df_res_fcst       = D.get('reseller_forecast', pd.DataFrame())
res_fcst_cols     = D.get('reseller_fcst_cols', [])
df_sales_res      = D.get('sales_reseller',    pd.DataFrame())
df_rofo_res       = D.get('past_rofo_reseller',pd.DataFrame())
df_po_res         = D.get('past_po_reseller',  pd.DataFrame())
df_fulfillment    = D.get('fulfillment',        pd.DataFrame())

# Pre-compute heavy metrics once
monthly_perf    = calc_monthly_performance(df_forecast, df_po, df_product)
last_3_perf     = {k: monthly_perf[k] for k in sorted(monthly_perf)[-3:]} if monthly_perf else {}
inv_metrics     = calc_inventory_metrics(df_stock, df_sales, df_product)
df_financial    = calc_financial(df_sales, df_product)
df_yoy          = calc_yoy(df_sales)
channel_acc     = calc_channel_accuracy(df_forecast, df_ecomm, df_rofo_res, df_po_res, df_po, df_product)
alerts          = get_critical_alerts(inv_metrics, df_product)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        import streamlit.components.v1 as components
        if st.button("🖨️ PDF", use_container_width=True):
            components.html("<script>window.print();</script>", height=0)

    st.markdown("---")

    # ── LIVE ALERT PANEL ─────────────────────────────────────────
    if alerts:
        critical_count = sum(1 for a in alerts if a['level']=='critical')
        warning_count  = sum(1 for a in alerts if a['level']=='warning')
        st.markdown(f"### 🔔 Alerts ({len(alerts)})")
        if critical_count:
            st.error(f"🔴 {critical_count} Critical Stockout Risk")
        if warning_count:
            st.warning(f"🟡 {warning_count} Overstock Warning")
        with st.expander("View All Alerts"):
            for a in alerts[:20]:
                st.markdown(f"{a['icon']} {a['msg']}")

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if not df_product_active.empty:
        st.metric("Active SKUs", len(df_product_active))
    if not df_stock.empty:
        st.metric("Total Stock", f"{df_stock['Stock_Qty'].sum():,.0f}")
    if monthly_perf:
        lm = sorted(monthly_perf)[-1]
        st.metric("Latest Accuracy", f"{monthly_perf[lm]['accuracy']:.1f}%")
    if not df_financial.empty:
        st.markdown("---")
        st.markdown("### 💰 Financial")
        rev = df_financial['Revenue'].sum()
        mgn = df_financial['Gross_Margin'].sum()
        st.metric("Revenue", fmt_money(rev))
        st.metric("Margin",  fmt_money(mgn))
        st.metric("Margin %", f"{(mgn/rev*100 if rev>0 else 0):.1f}%")

    st.markdown("---")
    st.markdown("### ⚙️ Thresholds")
    under_thr   = st.slider("Under Forecast (%)",  0,   100, 80)
    over_thr    = st.slider("Over Forecast (%)",   100, 200, 120)
    low_cov     = st.slider("Low Stock (mo)",      0.0, 2.0, 0.8, 0.1)
    high_cov    = st.slider("High Stock (mo)",     1.0, 6.0, 1.5, 0.1)

    st.markdown("---")
    st.caption(f"🎨 Active theme: **{st.session_state.get('theme_name','—')}**  \nChange via selector top-right ↗")


# ==============================================================================
#  ███████  EXECUTIVE SUMMARY  (print-ready)
# ==============================================================================
with st.expander("📋 Executive Summary — Print-Ready (click to expand)", expanded=False):
    st.markdown("## 📋 Executive Summary")
    st.caption(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}")

    if monthly_perf:
        lm   = sorted(monthly_perf)[-1]
        acc  = monthly_perf[lm]['accuracy']
        tot  = monthly_perf[lm]['total_records']
        und  = monthly_perf[lm]['status_counts'].get('Under',0)
        ovr  = monthly_perf[lm]['status_counts'].get('Over',0)
        rev  = df_financial['Revenue'].sum()      if not df_financial.empty else 0
        mgn  = df_financial['Gross_Margin'].sum() if not df_financial.empty else 0
        stk  = df_stock['Stock_Qty'].sum()        if not df_stock.empty else 0

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Forecast Accuracy",  f"{acc:.1f}%",    f"{lm.strftime('%b %Y')}")
        ec2.metric("Total Revenue (YTD)", fmt_money(rev))
        ec3.metric("Gross Margin (YTD)",  fmt_money(mgn),  f"{(mgn/rev*100 if rev>0 else 0):.1f}%")
        ec4.metric("Total Stock Units",   f"{stk:,.0f}")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🎯 Forecast Health**")
            st.markdown(f"- Last Month Accuracy: **{acc:.1f}%**")
            st.markdown(f"- Under-Forecast SKUs: **{und}**")
            st.markdown(f"- Over-Forecast SKUs:  **{ovr}**")
            st.markdown(f"- Total SKUs Evaluated: **{tot}**")
        with c2:
            st.markdown("**📦 Inventory Health**")
            if 'inventory_df' in inv_metrics:
                inv = inv_metrics['inventory_df']
                low_n  = len(inv[inv['Inventory_Status']=='Need Replenishment'])
                hi_n   = len(inv[inv['Inventory_Status']=='High Stock'])
                ok_n   = len(inv[inv['Inventory_Status']=='Ideal/Healthy'])
                st.markdown(f"- Need Replenishment: **{low_n}** SKUs 🔴")
                st.markdown(f"- Ideal/Healthy:      **{ok_n}** SKUs 🟢")
                st.markdown(f"- High Stock:         **{hi_n}** SKUs 🟡")
                st.markdown(f"- Avg Coverage:       **{inv_metrics.get('avg_cover',0):.1f} months**")

        # Critical alerts in exec summary
        if alerts:
            st.markdown("---")
            st.markdown(f"**🔔 Active Alerts ({len(alerts)})**")
            for a in alerts[:10]:
                st.markdown(f"{a['icon']} {a['msg']}")


# ==============================================================================
#  ███████  ACCURACY TREND  (top-level, outside tabs)
# ==============================================================================
st.subheader("📈 Forecast Accuracy Performance Trends")

if monthly_perf:
    summary_rows = []
    prev_acc = 0
    for i,(month,data) in enumerate(sorted(monthly_perf.items())):
        acc = data['accuracy']
        summary_rows.append(dict(
            Month=month, Month_Display=month.strftime('%b %Y'),
            Accuracy=acc, MAPE=data['mape'],
            Total_SKUs=data['total_records'],
            Under=data['status_counts'].get('Under',0),
            Over=data['status_counts'].get('Over',0),
            Accurate=data['status_counts'].get('Accurate',0),
            Delta=acc-prev_acc if i>0 else 0,
        ))
        prev_acc = acc
    sum_df = pd.DataFrame(summary_rows)

    avg_acc  = sum_df['Accuracy'].mean()
    last_acc = sum_df['Accuracy'].iloc[-1]
    delta    = sum_df['Delta'].iloc[-1]
    best     = sum_df.loc[sum_df['Accuracy'].idxmax()]
    stability= max(0, 100 - sum_df['Accuracy'].std())

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        arrow = "▲" if delta>=0 else "▼"
        st.markdown(fmt_card("Current Accuracy","",f"{last_acc:.1f}%",
            f"{arrow} {abs(delta):.1f}% vs last month",
            "linear-gradient(135deg,#4F46E5,#7C3AED)"), unsafe_allow_html=True)
    with c2:
        diff = avg_acc - 80
        st.markdown(fmt_card("Average YTD","",f"{avg_acc:.1f}%",
            f"{'✅' if diff>=0 else '⚠️'} {abs(diff):.1f}% vs target 80%",
            "linear-gradient(135deg,#0891B2,#22D3EE)"), unsafe_allow_html=True)
    with c3:
        st.markdown(fmt_card("Best Month","🌟",f"{best['Accuracy']:.1f}%",
            best['Month_Display'],
            "linear-gradient(135deg,#059669,#10B981)"), unsafe_allow_html=True)
    with c4:
        st.markdown(fmt_card("Stability","",f"{stability:.0f}",
            "Consistency 0-100",
            "linear-gradient(135deg,#EA580C,#F59E0B)"), unsafe_allow_html=True)

    # Combo chart
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=sum_df['Month_Display'], y=sum_df['Total_SKUs'],
        name="Total SKUs", marker_color='rgba(156,163,175,.15)', showlegend=True), secondary_y=True)
    mc = ['#EF4444' if v<70 else '#F59E0B' if v<80 else '#10B981' for v in sum_df['Accuracy']]
    fig.add_trace(go.Scatter(x=sum_df['Month_Display'], y=sum_df['Accuracy'],
        name="Accuracy %", mode='lines+markers',
        line=dict(color=T['accent1'],width=3,shape='spline',smoothing=1.3),
        marker=dict(size=14,color=mc,line=dict(width=2,color='white')),
        hovertemplate="<b>%{x}</b><br>Accuracy: <b>%{y:.1f}%</b><extra></extra>"),
        secondary_y=False)
    fig.add_hrect(y0=80,y1=110,fillcolor="rgba(16,185,129,.08)",layer="below",line_width=0)
    fig.update_yaxes(title="Accuracy (%)",range=[40,100],secondary_y=False,
        gridcolor=T['chart_grid'],tickfont=dict(color=T['accent1'],weight='bold'))
    fig.update_yaxes(visible=False, secondary_y=True)
    fig.update_xaxes(showgrid=False, tickfont=dict(weight='bold'))
    fig.update_layout(height=460,plot_bgcolor=T['chart_bg'],paper_bgcolor=T['chart_paper'],
        font=dict(color=T['chart_font']),
        hovermode='x unified',
        legend=dict(orientation="h",y=1.02,x=1,xanchor="right",font=dict(color=T['chart_font'])),
        margin=dict(t=60,b=40,l=40,r=40))
    st.plotly_chart(fig, use_container_width=True)

    # Status colour
    if last_acc >= 80:   clr,lbl = "#10B981","EXCELLENT 🚀"
    elif last_acc >= 70: clr,lbl = "#F59E0B","MODERATE ⚠️"
    else:                clr,lbl = "#EF4444","CRITICAL 🚨"
    st.markdown(f"""
    <div style="background:white;border-radius:10px;padding:1rem;border-left:6px solid {clr};
         box-shadow:0 2px 5px rgba(0,0,0,.05);color:#4B5563;">
      <strong style="color:{clr};">{lbl}</strong> — Current accuracy <strong>{last_acc:.1f}%</strong>.
      {'Stable.' if abs(delta)<2 else ('Improved +'+f'{delta:.1f}%' if delta>0 else 'Dropped '+f'{abs(delta):.1f}%')} vs prev month.
    </div>""", unsafe_allow_html=True)


# ==============================================================================
#  ███████  LAST-3-MONTHS PANEL
# ==============================================================================
st.subheader("🎯 Forecast Performance — 3 Bulan Terakhir")
if last_3_perf:
    cols = st.columns(3)
    for i,(month,data) in enumerate(sorted(last_3_perf.items())):
        with cols[i]:
            acc = data['accuracy']
            sc  = data['status_counts']
            st.markdown(f"""
            <div style="background:white;border-radius:15px;padding:1.5rem;
                 box-shadow:0 6px 20px rgba(0,0,0,.1);border-top:5px solid #667eea;">
              <div style="text-align:center;">
                <h3 style="margin:0;color:#333;">{month.strftime('%b %Y')}</h3>
                <div style="font-size:2rem;font-weight:900;color:#667eea;">{acc:.1f}%</div>
                <div style="font-size:.9rem;color:#666;">Overall Accuracy</div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:1rem;">
                <div style="text-align:center;padding:.5rem;background:#FFEBEE;border-radius:8px;">
                  <div style="font-size:1.5rem;font-weight:900;color:#F44336;">{sc.get('Under',0)}</div>
                  <div style="font-size:.8rem;color:#F44336;">Under</div>
                </div>
                <div style="text-align:center;padding:.5rem;background:#E8F5E9;border-radius:8px;">
                  <div style="font-size:1.5rem;font-weight:900;color:#4CAF50;">{sc.get('Accurate',0)}</div>
                  <div style="font-size:.8rem;color:#4CAF50;">Accurate</div>
                </div>
                <div style="text-align:center;padding:.5rem;background:#FFF3E0;border-radius:8px;">
                  <div style="font-size:1.5rem;font-weight:900;color:#FF9800;">{sc.get('Over',0)}</div>
                  <div style="font-size:.8rem;color:#FF9800;">Over</div>
                </div>
              </div>
              <div style="text-align:center;font-size:.9rem;color:#666;margin-top:.5rem;">
                Total SKUs: {data['total_records']}
              </div>
            </div>""", unsafe_allow_html=True)

    # Business-flow panel
    if monthly_perf:
        lm   = sorted(monthly_perf)[-1]
        lmd  = monthly_perf[lm]['data']
        rofo = lmd['Forecast_Qty'].sum()
        po_  = lmd['PO_Qty'].sum()
        sal  = df_sales[df_sales['Month']==lm]['Sales_Qty'].sum() if not df_sales.empty else 0

        st.markdown(f"#### 🔄 Business Flow: {lm.strftime('%B %Y')}")
        bc1,bc2,bc3 = st.columns(3)
        bc1.metric("1. PLAN (Rofo)",    f"{rofo:,.0f}")
        bc2.metric("2. EXECUTION (PO)", f"{po_:,.0f}",
                   f"{po_/rofo*100:.1f}% vs Rofo" if rofo else "")
        bc3.metric("3. RESULT (Sales)", f"{sal:,.0f}",
                   f"{sal/rofo*100:.1f}% achievement" if rofo else "")

st.divider()

# Under / Over evaluation with month selector
st.subheader("📋 Evaluasi Rofo per Bulan")
if monthly_perf:
    sorted_months = sorted(monthly_perf)
    month_opts    = [m.strftime('%b %Y') for m in sorted_months]
    sel_str       = st.selectbox("📅 Pilih Bulan:", month_opts, index=len(month_opts)-1)
    sel_key       = sorted_months[month_opts.index(sel_str)]
    sel_data      = monthly_perf[sel_key]

    et1,et2 = st.tabs([f"📉 UNDER ({sel_str})", f"📈 OVER ({sel_str})"])

    def _eval_table(skus_df, label, bg, bc):
        if skus_df.empty:
            st.success(f"✅ No {label} SKUs in {sel_str}")
            return
        # Merge inventory
        if 'inventory_df' in inv_metrics:
            inv_cols = inv_metrics['inventory_df'][['SKU_ID','Stock_Qty','Avg_Monthly_Sales_3M','Cover_Months']]
            skus_df  = pd.merge(skus_df, inv_cols, on='SKU_ID', how='left')
        disp = skus_df[['SKU_ID','Product_Name','Brand','SKU_Tier','Forecast_Qty','PO_Qty','PO_Rofo_Ratio',
                         'Stock_Qty','Avg_Monthly_Sales_3M','Cover_Months']].copy()
        disp['PO_Rofo_Ratio']      = disp['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%")
        disp['Cover_Months']       = disp['Cover_Months'].apply(lambda x: f"{x:.1f}" if x<999 else "N/A")
        disp['Avg_Monthly_Sales_3M']= disp['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
        st.dataframe(disp, use_container_width=True, height=450)
        # Summary bar
        tf  = skus_df['Forecast_Qty'].sum()
        tp  = skus_df['PO_Qty'].sum()
        avg = skus_df['PO_Rofo_Ratio'].mean()
        diff= tp - tf
        pct = diff/tf*100 if tf else 0
        st.markdown(f"""
        <div style="background:{bg};border-left:5px solid {bc};padding:1rem;
             border-radius:10px;color:#333;margin-top:.5rem;">
          <strong>{label} Summary</strong> — {len(skus_df)} SKUs |
          Avg PO/Rofo: <b>{avg:.1f}%</b> |
          Rofo: <b>{tf:,.0f}</b> | PO: <b>{tp:,.0f}</b> |
          Gap: <b style="color:{'#F44336' if diff<0 else '#2E7D32'};">{diff:+,.0f} ({pct:+.1f}%)</b>
        </div>""", unsafe_allow_html=True)

    with et1: _eval_table(sel_data['under_skus'],   "UNDER", "#FFEBEE", "#F44336")
    with et2: _eval_table(sel_data['over_skus'],    "OVER",  "#FFF3E0", "#FF9800")

st.divider()


# ==============================================================================
#  ███████  MAIN TABS
# ==============================================================================
(tab1, tab2, tab3, tab4, tab5,
 tab6, tab7, tab8, tab9, tab10,
 tab11) = st.tabs([
    "📅 Monthly Details",
    "🏷️ Brand & Tier",
    "📦 Inventory",
    "🔍 SKU Deep Dive",
    "📈 Sales Analysis",
    "📋 Data Explorer",
    "🛒 Ecomm Forecast",
    "💰 Profitability",
    "🤝 Reseller",
    "🚚 Fulfillment Cost",
    "📊 YoY & Channel",   # NEW
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — MONTHLY PERFORMANCE DETAILS (heatmap table + bias diverging chart)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📅 Monthly Performance Details")
    if monthly_perf:
        rows = []
        prev = 0
        for i,(m,d) in enumerate(sorted(monthly_perf.items())):
            a = d['accuracy']
            rows.append(dict(
                Month=m.strftime('%b %Y'),
                Status=("🌟 Excellent" if a>=90 else "✅ Good" if a>=80 else "⚠️ Fair" if a>=70 else "🛑 Poor"),
                Accuracy=a, MoM=f"{a-prev:+.1f}%" if i>0 else "-",
                Under=d['status_counts'].get('Under',0),
                Accurate=d['status_counts'].get('Accurate',0),
                Over=d['status_counts'].get('Over',0),
                Total=d['total_records'], MAPE=d['mape'],
            ))
            prev = a
        tdf = pd.DataFrame(rows)

        def _hl_acc(v):
            clr = '#d1fae5' if v>=80 else '#fef3c7' if v>=70 else '#fee2e2'
            return f'background:{clr};color:#374151;font-weight:bold'

        styled = (tdf.style
            .background_gradient(subset=['Under'],    cmap='Reds',   vmin=0, vmax=tdf['Under'].max()*1.5)
            .background_gradient(subset=['Accurate'], cmap='Greens', vmin=0)
            .background_gradient(subset=['Over'],     cmap='Oranges',vmin=0, vmax=tdf['Over'].max()*1.5)
            .applymap(_hl_acc, subset=['Accuracy'])
            .format({'Accuracy':'{:.1f}%','MAPE':'{:.1f}%',
                     'Under':'{:,}','Accurate':'{:,}','Over':'{:,}','Total':'{:,}'}))

        st.dataframe(styled, column_order=['Month','Status','Accuracy','MoM','Under','Accurate','Over','Total','MAPE'],
            column_config={
                "Accuracy": st.column_config.ProgressColumn("Accuracy %",format="%.1f%%",min_value=0,max_value=100),
                "MoM":      st.column_config.TextColumn("Trend (MoM)"),
            }, use_container_width=True, height=500, hide_index=True)
        st.caption("🟩>80%  🟨70-80%  🟥<70%")

        # Bias diverging chart
        st.divider()
        st.subheader("🎯 Forecast Bias Trend")
        bias_rows = []
        for m, d in sorted(monthly_perf.items()):
            md = d['data']
            if md.empty: continue
            md = md[md['Forecast_Qty']>0].copy()
            md['Bias_Pct'] = (md['PO_Qty'] - md['Forecast_Qty']) / md['Forecast_Qty'] * 100
            bias_rows.append(dict(Month=m.strftime('%b %Y'), Avg_Bias=md['Bias_Pct'].mean()))
        if bias_rows:
            bdf = pd.DataFrame(bias_rows)
            colors = ['#4db6ac' if abs(v)<=10 else '#ffb74d' if abs(v)<=20 else '#ef5350'
                      for v in bdf['Avg_Bias']]
            fig_b = go.Figure(go.Bar(
                x=bdf['Month'], y=bdf['Avg_Bias'],
                text=[f"{v:+.1f}%" for v in bdf['Avg_Bias']],
                textposition='auto', marker_color=colors))
            fig_b.add_hrect(y0=-10,y1=10,fillcolor="green",opacity=.05,line_width=0)
            fig_b.add_hline(y=0,line_color='black',line_width=2)
            fig_b.update_layout(
        font=dict(color=T['chart_font']),height=380, plot_bgcolor=T['chart_bg'],
                title="Monthly Bias (+ = Under-forecast, - = Over-forecast)",
                yaxis_title="Bias %", xaxis_title="Month")
            st.plotly_chart(fig_b, use_container_width=True)
            st.caption("🟢 ±10% safe  🟡 ±20% warning  🔴 >20% critical")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BRAND & TIER  (with period selector)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🏷️ Brand & Tier Strategic Analysis")
    all_months = sorted(monthly_perf) if monthly_perf else []
    if not all_months:
        st.warning("No data.")
    else:
        lm_date  = all_months[-1]
        l12      = all_months[-12:]

        period   = st.radio("Period:", ["Last Month","L12M"], horizontal=True)
        p_months = [lm_date] if period=="Last Month" else l12
        p_label  = lm_date.strftime('%B %Y') if period=="Last Month" else f"Last {len(l12)} months"

        df_fp = df_forecast[df_forecast['Month'].isin(p_months)]
        df_pp = df_po      [df_po['Month']      .isin(p_months)]
        df_fp = add_product_info(df_fp, df_product)
        df_pp = add_product_info(df_pp, df_product)

        brand_f  = df_fp.groupby('Brand')['Forecast_Qty'].sum().reset_index()
        brand_p  = df_pp.groupby('Brand')['PO_Qty'].sum().reset_index()
        brand_sk = df_fp.groupby('Brand')['SKU_ID'].nunique().reset_index(name='SKU_Count')

        sku_acc  = pd.merge(
            df_fp.groupby(['Brand','SKU_ID'])['Forecast_Qty'].sum().reset_index(),
            df_pp.groupby(['Brand','SKU_ID'])['PO_Qty'].sum().reset_index(),
            on=['Brand','SKU_ID'], how='outer').fillna(0)
        sku_acc['Acc'] = sku_acc.apply(
            lambda r: 100-abs(r['PO_Qty']/r['Forecast_Qty']*100-100) if r['Forecast_Qty']>0 else 0, axis=1)
        brand_acc= sku_acc.groupby('Brand')['Acc'].mean().reset_index(name='Accuracy')

        brand_df = (pd.merge(brand_f, brand_p, on='Brand', how='outer')
                    .merge(brand_sk, on='Brand').merge(brand_acc, on='Brand').fillna(0))

        st.markdown(f"**{p_label}**")
        # KPI cards
        best_b  = brand_df.loc[brand_df['Accuracy'].idxmax()]
        hv_b    = brand_df.loc[brand_df['Forecast_Qty'].idxmax()]
        wtd_acc = (brand_df['Accuracy']*brand_df['Forecast_Qty']).sum()/brand_df['Forecast_Qty'].sum() if brand_df['Forecast_Qty'].sum() else 0

        bc1,bc2,bc3,bc4 = st.columns(4)
        with bc1: st.markdown(fmt_card("Best Accuracy","🏆",best_b['Brand'],f"{best_b['Accuracy']:.1f}%","linear-gradient(135deg,#10B981,#059669)"), unsafe_allow_html=True)
        with bc2: st.markdown(fmt_card("Highest Volume","📦",hv_b['Brand'],f"{hv_b['Forecast_Qty']:,.0f} units","linear-gradient(135deg,#6366F1,#4338CA)"), unsafe_allow_html=True)
        with bc3: st.markdown(fmt_card("Most SKUs","🗂️",brand_df.loc[brand_df['SKU_Count'].idxmax(),'Brand'],f"{brand_df['SKU_Count'].max()} items","linear-gradient(135deg,#F59E0B,#D97706)"), unsafe_allow_html=True)
        with bc4: st.markdown(fmt_card("Portfolio Health","⚖️","All Brands",f"{wtd_acc:.1f}% wgt acc","linear-gradient(135deg,#3B82F6,#2563EB)"), unsafe_allow_html=True)

        # Scatter quad
        st.divider()
        fig_q = px.scatter(brand_df, x='Forecast_Qty', y='Accuracy', size='SKU_Count',
            color='Accuracy', text='Brand', color_continuous_scale='RdYlGn', size_max=55,
            hover_data=['SKU_Count','Forecast_Qty'])
        fig_q.add_hline(y=80, line_dash="dash", line_color="gray")
        fig_q.add_vline(x=brand_df['Forecast_Qty'].median(), line_dash="dash", line_color="gray")
        fig_q.update_traces(textposition='top center')
        fig_q.update_layout(
        font=dict(color=T['chart_font']),height=480, xaxis_type="log", yaxis_range=[40,105],
            plot_bgcolor=T['chart_bg'], title="Brand Positioning Matrix")
        st.plotly_chart(fig_q, use_container_width=True)

        # Combo bar + line
        st.divider()
        srt = brand_df.sort_values('Forecast_Qty', ascending=False)
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(x=srt['Brand'], y=srt['Forecast_Qty'], name='Volume',
            marker_color='rgba(99,102,241,.6)'))
        fig_c.add_trace(go.Scatter(x=srt['Brand'], y=srt['Accuracy'], name='Accuracy %',
            yaxis='y2', mode='lines+markers',
            line=dict(color='#F59E0B',width=3), marker=dict(size=8,color='#F59E0B')))
        fig_c.update_layout(height=400,
            yaxis=dict(title="Volume",showgrid=False),
            yaxis2=dict(title="Accuracy %",overlaying='y',side='right',range=[0,110],showgrid=True),
            hovermode="x unified", plot_bgcolor=T['chart_bg'],
            legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig_c, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — INVENTORY  (with real-time alert panel)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📦 Inventory Health Dashboard")

    # Live alert banner
    if alerts:
        crit = [a for a in alerts if a['level']=='critical']
        warn = [a for a in alerts if a['level']=='warning']
        if crit:
            with st.expander(f"🔴 {len(crit)} CRITICAL Stockout Alerts — click to view", expanded=True):
                for a in crit:
                    st.markdown(f'<div class="alert-critical">{a["icon"]} {a["msg"]}</div>', unsafe_allow_html=True)
        if warn:
            with st.expander(f"🟡 {len(warn)} Overstock Warnings"):
                for a in warn:
                    st.markdown(f'<div class="alert-warning">{a["icon"]} {a["msg"]}</div>', unsafe_allow_html=True)

    if not df_stock.empty and 'inventory_df' in inv_metrics:
        inv = inv_metrics['inventory_df']

        # KPI
        WH_CAP = st.number_input("🏢 Warehouse Capacity (pcs)", 1000, 10_000_000, 250_000, 10_000)
        occ    = df_stock['Stock_Qty'].sum()
        occ_pct= occ / WH_CAP * 100

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Stock",    f"{occ:,.0f}",     f"{occ_pct:.1f}% capacity")
        c2.metric("Avg Coverage",   f"{inv_metrics['avg_cover']:.1f} mo")
        c3.metric("Health Score",   f"{inv_metrics['health_score']:.1f}%")
        c4.metric("Need Reorder",   f"{len(inv_metrics['low_stock'])} SKUs")

        # Gauges
        g1,g2 = st.columns(2)
        with g1:
            fig_g = go.Figure(go.Indicator(mode="gauge+number",value=inv_metrics['avg_cover'],
                domain={'x':[0,1],'y':[0,1]},title={'text':"Avg Coverage (Mo)"},
                gauge={'axis':{'range':[0,6]},'bar':{'color':'#7986cb'},
                       'steps':[{'range':[0,.8],'color':'#ef5350'},
                                 {'range':[.8,2],'color':'#4db6ac'},
                                 {'range':[2,6],'color':'#ffb74d'}],
                       'threshold':{'line':{'color':'black','width':4},'thickness':.75,'value':2}}))
            fig_g.update_layout(height=250,margin=dict(t=40,b=10))
            st.plotly_chart(fig_g, use_container_width=True)
        with g2:
            clr = "#4db6ac" if occ_pct<80 else "#ef5350"
            fig_g2 = go.Figure(go.Indicator(mode="gauge+number+delta",value=occ_pct,
                domain={'x':[0,1],'y':[0,1]},title={'text':"WH Occupancy (%)"},
                delta={'reference':80},
                gauge={'axis':{'range':[0,100]},'bar':{'color':clr},
                       'steps':[{'range':[0,60],'color':'#e0f2f1'},
                                 {'range':[60,85],'color':'#fff3e0'},
                                 {'range':[85,100],'color':'#ffebee'}],
                       'threshold':{'line':{'color':'red','width':4},'thickness':.75,'value':85}}))
            fig_g2.update_layout(height=250,margin=dict(t=40,b=10))
            st.plotly_chart(fig_g2, use_container_width=True)

        # Status distribution
        st.divider()
        status_counts = inv['Inventory_Status'].value_counts()
        fig_pie = px.pie(values=status_counts.values, names=status_counts.index,
            color=status_counts.index,
            color_discrete_map={'Need Replenishment':'#ef5350','Ideal/Healthy':'#4db6ac','High Stock':'#ffb74d'},
            hole=.4, title="Inventory Status Distribution")
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Drill-down table
        st.divider()
        with st.expander("🔍 Drill-Down", expanded=True):
            f1,f2 = st.columns(2)
            with f1: sel_status = st.multiselect("Status:", inv['Inventory_Status'].unique(), default=list(inv['Inventory_Status'].unique()))
            with f2: srch = st.text_input("Search SKU/Name:")
            d_ = inv[inv['Inventory_Status'].isin(sel_status)] if sel_status else inv
            if srch:
                d_ = d_[d_['SKU_ID'].str.contains(srch,case=False,na=False) |
                         d_.get('Product_Name',pd.Series(dtype=str)).str.contains(srch,case=False,na=False)]
            st.dataframe(d_.sort_values('Cover_Months'), use_container_width=True, height=400)
    else:
        st.warning("⚠️ No stock data.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — SKU DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔍 SKU 360° Deep Dive")
    if monthly_perf and not df_sales.empty:
        lm   = sorted(monthly_perf)[-1]
        lmd  = monthly_perf[lm]['data']
        sku_opts = [f"{r['SKU_ID']} — {r.get('Product_Name','N/A')}"
                    for _,r in lmd.sort_values('Forecast_Qty',ascending=False).head(200).iterrows()]
        sel  = st.selectbox("Select SKU:", sku_opts)
        sku  = sel.split(" — ")[0]
        row  = lmd[lmd['SKU_ID']==sku].iloc[0]

        inv_row = inv_metrics.get('inventory_df', pd.DataFrame())
        inv_row = inv_row[inv_row['SKU_ID']==sku] if not inv_row.empty else pd.DataFrame()
        stock_  = inv_row.iloc[0]['Stock_Qty']            if not inv_row.empty else 0
        avg_s   = inv_row.iloc[0]['Avg_Monthly_Sales_3M'] if not inv_row.empty else 0
        cover   = inv_row.iloc[0]['Cover_Months']          if not inv_row.empty else 0

        # Header card
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:1.5rem;
             box-shadow:0 4px 15px rgba(0,0,0,.05);border-left:6px solid #6366F1;margin-bottom:1rem;">
          <strong style="font-size:1.3rem;">{row.get('Product_Name','–')}</strong>
          <span style="color:#6B7280;"> ({sku})</span><br>
          <span style="background:#E0E7FF;color:#4338CA;padding:3px 10px;border-radius:15px;font-size:.8rem;">
            🏷️ {row.get('Brand','–')}</span>
          <span style="background:#F3E8FF;color:#7E22CE;padding:3px 10px;border-radius:15px;
                font-size:.8rem;margin-left:6px;">💎 {row.get('SKU_Tier','–')} Tier</span>
        </div>""", unsafe_allow_html=True)

        # Metrics
        k1,k2,k3,k4 = st.columns(4)
        cov_clr = "linear-gradient(135deg,#10B981,#059669)" if .8<=cover<=2 else \
                  ("linear-gradient(135deg,#EF4444,#B91C1C)" if cover<.8 else "linear-gradient(135deg,#F59E0B,#D97706)")
        with k1: st.markdown(fmt_card("Stock","",f"{stock_:,.0f}","units","linear-gradient(135deg,#6366F1,#4338CA)"), unsafe_allow_html=True)
        with k2: st.markdown(fmt_card("Cover","",f"{cover:.1f} mo","health",cov_clr), unsafe_allow_html=True)
        with k3: st.markdown(fmt_card("Avg Sales 3M","",f"{avg_s:,.0f}","monthly vel","linear-gradient(135deg,#0EA5E9,#0284C7)"), unsafe_allow_html=True)
        with k4:
            acc_v = f"{row['PO_Qty']/row['Forecast_Qty']*100:.1f}%" if row['Forecast_Qty']>0 else "N/A"
            st.markdown(fmt_card("PO/Rofo","",acc_v,f"Last {lm.strftime('%b')}","linear-gradient(135deg,#EC4899,#DB2777)"), unsafe_allow_html=True)

        # Trend chart
        hist = []
        for m in sorted(df_sales['Month'].unique())[-12:]:
            s = df_sales[(df_sales['Month']==m)&(df_sales['SKU_ID']==sku)]['Sales_Qty'].sum()
            f = df_forecast[(df_forecast['Month']==m)&(df_forecast['SKU_ID']==sku)]['Forecast_Qty'].sum() if not df_forecast.empty else 0
            p = df_po[(df_po['Month']==m)&(df_po['SKU_ID']==sku)]['PO_Qty'].sum() if not df_po.empty else 0
            hist.append({'Month':m.strftime('%b-%y'),'Sales':s,'Forecast':f,'PO':p})

        if hist:
            hdf = pd.DataFrame(hist)
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=hdf['Month'],y=hdf['Sales'],name='Sales',
                mode='lines',fill='tozeroy',line=dict(color='#10B981',width=3),fillcolor='rgba(16,185,129,.1)'))
            fig_t.add_trace(go.Scatter(x=hdf['Month'],y=hdf['Forecast'],name='Forecast',
                mode='lines+markers',line=dict(color='#6366F1',width=3,dash='dash'),marker=dict(size=6)))
            fig_t.add_trace(go.Bar(x=hdf['Month'],y=hdf['PO'],name='PO',
                marker_color='rgba(245,158,11,.4)',marker_line_color='#F59E0B',marker_line_width=1.5))
            fig_t.update_layout(
        font=dict(color=T['chart_font']),height=420,hovermode="x unified",plot_bgcolor=T['chart_bg'],
                legend=dict(orientation="h",y=1.1),
                yaxis=dict(title="Units",gridcolor=T['chart_grid']))
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Load sales & monthly performance data first.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SALES ANALYSIS  (with year filter)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📈 Realization & Gap Analysis")
    if not df_sales.empty and not df_forecast.empty:
        all_yrs = sorted({m.year for m in df_sales['Month'].unique()})
        sel_yrs = st.multiselect("Filter Year:", all_yrs, default=all_yrs)
        fmonths = [m for m in sorted(set(df_sales['Month'].unique())|set(df_forecast['Month'].unique())|set(df_po['Month'].unique()))
                   if m.year in sel_yrs]
        if not fmonths:
            st.warning("Select at least one year.")
        else:
            rows_t = []
            for m in fmonths:
                rows_t.append(dict(Month=m,Month_Txt=m.strftime('%b-%y'),
                    Rofo=df_forecast[df_forecast['Month']==m]['Forecast_Qty'].sum(),
                    PO  =df_po[df_po['Month']==m]['PO_Qty'].sum() if not df_po.empty else 0,
                    Sales=df_sales[df_sales['Month']==m]['Sales_Qty'].sum()))
            trd = pd.DataFrame(rows_t)

            tr,tp,ts = trd['Rofo'].sum(), trd['PO'].sum(), trd['Sales'].sum()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Plan (Rofo)", f"{tr:,.0f}")
            c2.metric("Total PO",          f"{tp:,.0f}", f"{tp/tr*100:.1f}% of plan" if tr else "")
            c3.metric("Total Sales",       f"{ts:,.0f}", f"{ts/tr*100:.1f}% ach."     if tr else "")
            c4.metric("Gap Sales vs Plan", f"{ts-tr:+,.0f}")

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=trd['Month_Txt'],y=trd['Rofo'],name='Rofo',
                mode='lines+markers',line=dict(color='#3949AB',width=3,dash='dash')))
            fig_t.add_trace(go.Bar(x=trd['Month_Txt'],y=trd['PO'],name='PO',
                marker_color=T['colors'][5],opacity=.7))
            fig_t.add_trace(go.Bar(x=trd['Month_Txt'],y=trd['Sales'],name='Sales',
                marker_color=T['colors'][2],opacity=.7))
            fig_t.update_layout(
        font=dict(color=T['chart_font']),height=420,barmode='group',hovermode='x unified',
                plot_bgcolor=T['chart_bg'],legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig_t, use_container_width=True)

            # Gap pareto
            st.divider()
            df_fg = add_product_info(df_forecast[df_forecast['Month'].dt.year.isin(sel_yrs)], df_product)
            df_sg = add_product_info(df_sales[df_sales['Month'].dt.year.isin(sel_yrs)],       df_product)
            fg = df_fg.groupby(['SKU_ID','Product_Name'])['Forecast_Qty'].sum().reset_index()
            sg = df_sg.groupby(['SKU_ID','Product_Name'])['Sales_Qty'].sum().reset_index()
            gap_df = pd.merge(fg,sg,on=['SKU_ID','Product_Name'],how='outer').fillna(0)
            gap_df['Gap'] = gap_df['Sales_Qty'] - gap_df['Forecast_Qty']

            g1,g2 = st.columns(2)
            with g1:
                st.markdown("##### 🚀 Top Demand Spikes")
                top_sp = gap_df[gap_df['Gap']>0].sort_values('Gap',ascending=False).head(10)
                fig_sp = go.Figure(go.Bar(y=top_sp['Product_Name'].str[:22],x=top_sp['Gap'],
                    orientation='h',marker_color=T['colors'][4],
                    text=[f"+{x:,.0f}" for x in top_sp['Gap']],textposition='auto'))
                fig_sp.update_layout(
        font=dict(color=T['chart_font']),height=380,plot_bgcolor=T['chart_bg'],
                    yaxis=dict(autorange="reversed"),xaxis_title="Extra Units vs Plan")
                st.plotly_chart(fig_sp, use_container_width=True)
            with g2:
                st.markdown("##### 🐌 Top Slow Movers")
                top_dr = gap_df[gap_df['Gap']<0].sort_values('Gap').head(10)
                fig_dr = go.Figure(go.Bar(y=top_dr['Product_Name'].str[:22],x=top_dr['Gap'],
                    orientation='h',marker_color=T['colors'][6],
                    text=[f"{x:,.0f}" for x in top_dr['Gap']],textposition='auto'))
                fig_dr.update_layout(
        font=dict(color=T['chart_font']),height=380,plot_bgcolor=T['chart_bg'],
                    yaxis=dict(autorange="reversed",side='right'),xaxis_title="Missed Units vs Plan")
                st.plotly_chart(fig_dr, use_container_width=True)
    else:
        st.info("Need Sales & Forecast data.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("📋 Raw Data Explorer")
    datasets = {"Product Master":df_product,"Active Products":df_product_active,
                "Sales":df_sales,"Forecast":df_forecast,"PO":df_po,
                "Stock":df_stock,"Financial":df_financial}
    sel_ds = st.selectbox("Dataset:", list(datasets))
    df_sel = datasets[sel_ds]
    if not df_sel.empty:
        st.write(f"**Rows:** {len(df_sel):,}  **Cols:** {df_sel.shape[1]}")
        st.dataframe(df_sel, use_container_width=True, height=480)
        st.download_button("📥 Download CSV", df_sel.to_csv(index=False),
            f"{sel_ds.replace(' ','_')}_{datetime.now():%Y%m%d}.csv", "text/csv", use_container_width=True)
    else:
        st.warning("No data.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — ECOMM FORECAST  (with quarterly heatmap + explorer)
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.subheader("🛒 Ecommerce Forecast Intelligence")
    if not df_ecomm.empty and ecomm_month_cols:
        df_e = df_ecomm.copy()
        id_cols = ['SKU_ID','Product_Name','Brand','SKU_Tier','Status','Floor_Price','Net_Order_Price']
        for c in ecomm_month_cols:
            df_e[c] = pd.to_numeric(df_e[c], errors='coerce').fillna(0)

        total_qty = df_e[ecomm_month_cols].sum().sum()
        has_price = 'Floor_Price' in df_e.columns
        if has_price:
            df_e['Floor_Price'] = pd.to_numeric(df_e['Floor_Price'], errors='coerce').fillna(0)
            total_val = (df_e[ecomm_month_cols].multiply(df_e['Floor_Price'], axis=0)).sum().sum()
        else:
            total_val = 0

        peak_m = df_e[ecomm_month_cols].sum().idxmax()

        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(fmt_card("Total Volume","",f"{total_qty:,.0f}","units","linear-gradient(135deg,#6366F1,#4338CA)"), unsafe_allow_html=True)
        with c2: st.markdown(fmt_card("Total Value","",fmt_money(total_val),"gross rev","linear-gradient(135deg,#10B981,#059669)"), unsafe_allow_html=True)
        with c3: st.markdown(fmt_card("Peak Month","📅",str(peak_m).split('.')[0],"highest vol","linear-gradient(135deg,#F59E0B,#D97706)"), unsafe_allow_html=True)

        # Monthly trend
        monthly_agg = df_e[ecomm_month_cols].sum()
        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(x=monthly_agg.index, y=monthly_agg.values, marker_color=T['accent2']))
        fig_t.add_hline(y=monthly_agg.mean(), line_dash='dash', line_color='#F59E0B', annotation_text='Avg')
        fig_t.update_layout(
        font=dict(color=T['chart_font']),height=320,plot_bgcolor=T['chart_bg'],hovermode='x unified',
            margin=dict(t=20,b=20))
        st.plotly_chart(fig_t, use_container_width=True)

        # Quarterly heatmap
        st.divider()
        st.subheader("📅 Quarterly Brand Heatmap")
        q_map = {'Q1':['jan','feb','mar'],'Q2':['apr','may','jun'],
                 'Q3':['jul','aug','sep'],'Q4':['oct','nov','dec']}
        q_cols = {q: [c for c in ecomm_month_cols if str(c).lower()[:3] in ms] for q,ms in q_map.items()}
        aq = [q for q,cs in q_cols.items() if cs]

        if aq and 'Brand' in df_e.columns:
            rows_q = []
            for brand in df_e['Brand'].unique():
                bd = df_e[df_e['Brand']==brand]
                r  = {'Brand':brand}
                for q in aq:
                    r[q] = bd[q_cols[q]].sum().sum()
                r['Total'] = sum(r[q] for q in aq)
                rows_q.append(r)
            qdf = pd.DataFrame(rows_q).sort_values('Total',ascending=False)
            grand = {'Brand':'TOTAL'}
            for q in aq+['Total']: grand[q] = qdf[q].sum()
            qdf = pd.concat([qdf, pd.DataFrame([grand])], ignore_index=True)
            disp_q = aq+['Total']
            fig_h = go.Figure(go.Heatmap(z=qdf[disp_q].values, x=disp_q, y=qdf['Brand'],
                colorscale='Blues', text=qdf[disp_q].values, texttemplate="%{text:,.0f}"))
            fig_h.update_layout(height=min(600,50+30*len(qdf)), yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig_h, use_container_width=True)

        # Explorer
        st.divider()
        st.subheader("📋 Forecast Data Explorer")
        f1,f2 = st.columns([1,2])
        with f1:
            sel_brands = st.multiselect("Brands:", df_e['Brand'].unique().tolist() if 'Brand' in df_e.columns else [])
            n_months   = st.slider("Months to show:", 3, len(ecomm_month_cols), 6)
        with f2:
            srch_e = st.text_input("Search SKU/Name (Ecomm):")
        de = df_e.copy()
        if sel_brands: de = de[de['Brand'].isin(sel_brands)]
        if srch_e:
            de = de[de['SKU_ID'].astype(str).str.contains(srch_e,case=False)|
                    de.get('Product_Name',pd.Series(dtype=str)).str.contains(srch_e,case=False)]
        base_cols = [c for c in ['SKU_ID','Product_Name','Brand','SKU_Tier'] if c in de.columns]
        st.dataframe(de[base_cols+ecomm_month_cols[:n_months]], use_container_width=True, height=460)
        st.download_button("📥 Download CSV", de.to_csv(index=False), "ecomm_forecast.csv","text/csv")
    else:
        st.info("Upload data to 'Forecast_2026_Ecomm' sheet.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — PROFITABILITY  (vectorised — no iterrows)
# ─────────────────────────────────────────────────────────────────────────────
with tab8:
    st.subheader("💰 Profitability & Margin Intelligence")

    def _build_fin(df_in, channel_name):
        """Vectorised financial calc from wide forecast df."""
        if df_in.empty:
            return pd.DataFrame()
        mc = [c for c in df_in.columns if any(char.isdigit() for char in str(c))]
        if not mc:
            return pd.DataFrame()
        needed = ['SKU_ID'] + [c for c in ['Product_Name','Brand','SKU_Tier','Floor_Price','Net_Order_Price'] if c in df_in.columns]
        df_long = df_in[needed + mc].melt(id_vars=needed, var_name='Month', value_name='Qty')
        df_long['Qty'] = pd.to_numeric(df_long['Qty'], errors='coerce').fillna(0)
        df_long = df_long[df_long['Qty']>0]
        if 'Floor_Price'    in df_long.columns: df_long['Floor_Price']    = pd.to_numeric(df_long['Floor_Price'],    errors='coerce').fillna(0)
        if 'Net_Order_Price' in df_long.columns: df_long['Net_Order_Price']= pd.to_numeric(df_long['Net_Order_Price'],errors='coerce').fillna(0)
        df_long['Revenue']      = df_long['Qty'] * df_long.get('Floor_Price',    pd.Series(0, index=df_long.index))
        df_long['COGS']         = df_long['Qty'] * df_long.get('Net_Order_Price',pd.Series(0, index=df_long.index))
        df_long['Gross_Margin'] = df_long['Revenue'] - df_long['COGS']
        df_long['Channel']      = channel_name
        return df_long

    df_fin_e = _build_fin(df_ecomm,    'Ecommerce')
    df_fin_r = _build_fin(df_res_fcst, 'Reseller')
    combined = pd.concat([df_fin_e, df_fin_r], ignore_index=True) if not (df_fin_e.empty and df_fin_r.empty) else pd.DataFrame()

    if not combined.empty:
        tr  = combined['Revenue'].sum()
        tc  = combined['COGS'].sum()
        tm  = combined['Gross_Margin'].sum()
        mp  = tm/tr*100 if tr else 0

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(fmt_card("Revenue","",    fmt_money(tr), "gross sales", "linear-gradient(135deg,#6366F1,#4338CA)"), unsafe_allow_html=True)
        with c2: st.markdown(fmt_card("COGS","",       fmt_money(tc), "cost of goods","linear-gradient(135deg,#F59E0B,#D97706)"), unsafe_allow_html=True)
        with c3: st.markdown(fmt_card("Gross Margin","",fmt_money(tm),"profit",       "linear-gradient(135deg,#10B981,#059669)"), unsafe_allow_html=True)
        with c4: st.markdown(fmt_card("Margin %","",   f"{mp:.1f}%", "blended",      "linear-gradient(135deg,#3B82F6,#2563EB)"), unsafe_allow_html=True)

        # Waterfall
        st.divider()
        fig_w = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative","relative","total"],
            x=["Revenue","COGS","Gross Margin"],
            text=[fmt_money(tr), fmt_money(-tc), fmt_money(tm)],
            y=[tr,-tc,tm],
            connector={"line":{"color":"#374151"}},
            increasing={"marker":{"color":"#6366F1"}},
            decreasing={"marker":{"color":"#F59E0B"}},
            totals={"marker":{"color":"#10B981"}}))
        fig_w.update_layout(
        font=dict(color=T['chart_font']),height=380, showlegend=False, plot_bgcolor=T['chart_bg'])
        st.plotly_chart(fig_w, use_container_width=True)

        # SKU matrix
        sku_g = combined.groupby(['SKU_ID','Product_Name','Brand','SKU_Tier']).agg(
            Revenue=('Revenue','sum'), Gross_Margin=('Gross_Margin','sum')).reset_index()
        sku_g['Margin_Pct'] = (sku_g['Gross_Margin']/sku_g['Revenue']*100).fillna(0)

        st.divider()
        fig_sc = px.scatter(sku_g, x='Revenue', y='Margin_Pct', size='Gross_Margin',
            color='Brand', hover_name='Product_Name', size_max=50,
            title="SKU Profitability Matrix (Revenue vs Margin %)")
        fig_sc.add_hline(y=sku_g['Margin_Pct'].mean(), line_dash='dash', line_color='gray')
        fig_sc.add_vline(x=sku_g['Revenue'].mean(),    line_dash='dash', line_color='gray')
        fig_sc.update_layout(
        font=dict(color=T['chart_font']),height=480, plot_bgcolor=T['chart_bg'], xaxis_type='log')
        st.plotly_chart(fig_sc, use_container_width=True)

        # Pareto
        st.divider()
        p30 = sku_g.sort_values('Gross_Margin',ascending=False).head(30).copy()
        p30['Cum_Pct'] = p30['Gross_Margin'].cumsum() / tm * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=p30['Product_Name'].str[:18], y=p30['Gross_Margin'],
            marker_color=T['colors'][4], name='Margin'))
        fig_p.add_trace(go.Scatter(x=p30['Product_Name'].str[:18], y=p30['Cum_Pct'],
            yaxis='y2', name='Cum %', line=dict(color='#F59E0B',width=2)))
        fig_p.add_hline(y=80,line_dash='dash',line_color='gray',annotation_text='80%',yref='y2')
        fig_p.update_layout(height=420,
            yaxis2=dict(overlaying='y',side='right',range=[0,110],showgrid=False),
            xaxis_tickangle=-45, hovermode='x unified', plot_bgcolor=T['chart_bg'])
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("Need Ecomm or Reseller forecast with price data.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 — RESELLER  (dynamic last month — no Jan-26 hardcode)
# ─────────────────────────────────────────────────────────────────────────────
with tab9:
    st.subheader("🤝 Reseller Performance Dashboard")

    tr1,tr2,tr3,tr4 = st.tabs(["📈 Overview","🎯 Accuracy","💰 Financial","📊 Explorer"])

    with tr1:
        # Dynamic: derive last available month from data
        def get_last_month_label(df, col='Month_Label'):
            if df.empty or col not in df.columns: return None
            try:
                def parse(s):
                    s = str(s).strip()
                    for sep in [' ','-','_']:
                        pts = s.replace(sep,' ').split()
                        if len(pts)==2:
                            try: return datetime.strptime(f"{pts[0][:3]}-{pts[1]}", "%b-%y")
                            except: pass
                    return None
                labels = df[col].unique()
                parsed = [(l, parse(l)) for l in labels]
                valid  = [(l,d) for l,d in parsed if d]
                if not valid: return None
                return max(valid, key=lambda x: x[1])[0]
            except: return None

        last_label = (get_last_month_label(df_rofo_res) or
                      get_last_month_label(df_sales_res) or
                      get_last_month_label(df_po_res))

        if last_label:
            st.caption(f"📅 Latest period detected: **{last_label}**")
            rofo_lm  = df_rofo_res [df_rofo_res ['Month_Label']==last_label]['Forecast_Qty'].sum() if not df_rofo_res.empty  else 0
            sales_lm = df_sales_res[df_sales_res['Month_Label']==last_label]['Sales_Qty'].sum()    if not df_sales_res.empty else 0
            po_lm    = df_po_res   [df_po_res   ['Month_Label']==last_label]['PO_Qty'].sum()       if not df_po_res.empty    else 0
            acc_lm   = 100-abs(po_lm/rofo_lm*100-100) if rofo_lm>0 else 0
        else:
            rofo_lm=sales_lm=po_lm=acc_lm=0
            st.warning("⚠️ Cannot detect latest period label.")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rofo (Latest)",  f"{rofo_lm:,.0f}")
        c2.metric("Sales (Latest)", f"{sales_lm:,.0f}")
        c3.metric("PO (Latest)",    f"{po_lm:,.0f}")
        c4.metric("Accuracy",       f"{acc_lm:.1f}%")

        # Triple comparison chart (chronological order)
        if not df_sales_res.empty and not df_rofo_res.empty and not df_po_res.empty:
            def _agg_by_month(df, val_col):
                if 'Month' not in df.columns: return pd.Series(dtype=float)
                return df.groupby('Month')[val_col].sum()

            s_agg = _agg_by_month(df_sales_res,'Sales_Qty')
            r_agg = _agg_by_month(df_rofo_res, 'Forecast_Qty')
            p_agg = _agg_by_month(df_po_res,   'PO_Qty')
            all_m = sorted(set(s_agg.index)|set(r_agg.index)|set(p_agg.index))
            comp  = pd.DataFrame({'Month':all_m,
                'Sales':   [s_agg.get(m,0) for m in all_m],
                'Rofo':    [r_agg.get(m,0) for m in all_m],
                'PO':      [p_agg.get(m,0) for m in all_m]})
            comp['Month_Txt'] = comp['Month'].apply(lambda x: x.strftime('%b-%y') if hasattr(x,'strftime') else str(x))

            fig_rc = go.Figure()
            for col,clr,name in [('Rofo','#667eea','Rofo'),('PO','#FF9800','PO'),('Sales','#4CAF50','Sales')]:
                fig_rc.add_trace(go.Bar(x=comp['Month_Txt'],y=comp[col],name=name,
                    marker_color=clr, opacity=.75))
            fig_rc.update_layout(
        font=dict(color=T['chart_font']),height=380,barmode='group',hovermode='x unified',
                plot_bgcolor=T['chart_bg'], legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig_rc, use_container_width=True)

    with tr2:
        # Vectorised accuracy per SKU
        if not df_rofo_res.empty and not df_po_res.empty and last_label:
            rf = df_rofo_res[df_rofo_res['Month_Label']==last_label].groupby('SKU_ID')['Forecast_Qty'].sum()
            pf = df_po_res  [df_po_res  ['Month_Label']==last_label].groupby('SKU_ID')['PO_Qty'].sum()
            sf = df_sales_res[df_sales_res['Month_Label']==last_label].groupby('SKU_ID')['Sales_Qty'].sum() if not df_sales_res.empty else pd.Series(dtype=float)

            acc_df = pd.concat([rf,pf,sf], axis=1).fillna(0)
            acc_df.columns = ['Forecast_Qty','PO_Qty','Sales_Qty']
            acc_df = acc_df[acc_df['Forecast_Qty']>0].copy()
            acc_df['Accuracy'] = 100 - (acc_df['PO_Qty']/acc_df['Forecast_Qty']*100-100).abs()
            acc_df['Status']   = pd.cut(acc_df['Accuracy'],bins=[-np.inf,80,np.inf],labels=['Need Review','Accurate'])

            c1,c2,c3 = st.columns(3)
            c1.metric("Avg Accuracy",   f"{acc_df['Accuracy'].mean():.1f}%")
            c2.metric("Accurate SKUs",  f"{(acc_df['Status']=='Accurate').sum()}/{len(acc_df)}")
            c3.metric("Need Review",    f"{(acc_df['Status']=='Need Review').sum()}")

            fig_hist = px.histogram(acc_df, x='Accuracy', nbins=20, title='Accuracy Distribution',
                color_discrete_sequence=[T['accent1']])
            fig_hist.update_layout(
        font=dict(color=T['chart_font']),height=300, plot_bgcolor=T['chart_bg'])
            st.plotly_chart(fig_hist, use_container_width=True)
            st.dataframe(acc_df.sort_values('Accuracy').reset_index(), use_container_width=True, height=380)
        else:
            st.info("Need Past_Rofo_Reseller and Past_PO_Reseller data.")

    with tr3:
        # Financial from reseller forecast — vectorised
        if not df_res_fcst.empty and res_fcst_cols and 'Floor_Price' in df_res_fcst.columns:
            df_rf = df_res_fcst.copy()
            df_rf['Floor_Price'] = pd.to_numeric(df_rf['Floor_Price'], errors='coerce').fillna(0)
            # Vectorised: multiply each month column by Floor_Price then sum
            rev_series = pd.Series({c: (pd.to_numeric(df_rf[c],errors='coerce').fillna(0) * df_rf['Floor_Price']).sum()
                                    for c in res_fcst_cols})
            total_rev = rev_series.sum()

            c1,c2 = st.columns(2)
            c1.metric("Total Revenue 2026", fmt_money(total_rev))
            c2.metric("Avg Monthly Rev",    fmt_money(total_rev/len(res_fcst_cols) if res_fcst_cols else 0))

            # Parse months for chronological chart
            rev_rows = []
            for col, rev in rev_series.items():
                try:
                    cs   = str(col).upper().replace('_',' ').replace('-',' ')
                    pts  = cs.split()
                    mo   = datetime.strptime(pts[0][:3],'%b').month
                    yr_r = ''.join(filter(str.isdigit, pts[1])) if len(pts)>1 else ''
                    yr   = (2000+int(yr_r)) if len(yr_r)==2 else (int(yr_r) if yr_r else 2026)
                    rev_rows.append({'Month_Date':datetime(yr,mo,1),'Label':f"{pts[0][:3]}-{str(yr)[-2:]}","Rev":rev})
                except: pass
            if rev_rows:
                rdf = pd.DataFrame(rev_rows).sort_values('Month_Date')
                fig_rev = go.Figure(go.Bar(x=rdf['Label'],y=rdf['Rev'],
                    marker_color=T['colors'][4],text=[fmt_money(v) for v in rdf['Rev']],textposition='auto'))
                fig_rev.update_layout(
        font=dict(color=T['chart_font']),height=380,plot_bgcolor=T['chart_bg'],
                    title="Reseller Monthly Revenue Projection",xaxis_title="Month",yaxis_title="Revenue (Rp)")
                st.plotly_chart(fig_rev, use_container_width=True)
        else:
            st.info("Add Floor_Price to Reseller forecast for financial analysis.")

    with tr4:
        st.markdown("**Forecast 2026**")
        if not df_res_fcst.empty:
            base = [c for c in ['SKU_ID','Product_Name','Brand','SKU_Tier','Floor_Price'] if c in df_res_fcst.columns]
            show_m = st.slider("Months:", 3, max(3,len(res_fcst_cols)), min(6,len(res_fcst_cols)))
            st.dataframe(df_res_fcst[base+res_fcst_cols[:show_m]], use_container_width=True, height=420)
            st.download_button("📥 Download", df_res_fcst.to_csv(index=False), "reseller_fcst.csv","text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 10 — FULFILLMENT COST  (with download)
# ─────────────────────────────────────────────────────────────────────────────
with tab10:
    st.subheader("🚚 Fulfillment Cost Intelligence")
    if not df_fulfillment.empty:
        df_b = df_fulfillment.copy()
        for c in ['Total Order(BS)','Total Cost','GMV (Fullfil By BS)','GMV Total (MP)']:
            if c in df_b.columns: df_b[c] = pd.to_numeric(df_b[c], errors='coerce').fillna(0)

        df_b['CPO']        = np.where(df_b['Total Order(BS)']>0, df_b['Total Cost']/df_b['Total Order(BS)'], 0)
        df_b['BSA_Calc']   = np.where(df_b['Total Order(BS)']>0, df_b['GMV (Fullfil By BS)']/df_b['Total Order(BS)'], 0)
        df_b['Contrib_Pct']= np.where(df_b['GMV Total (MP)']>0,  df_b['GMV (Fullfil By BS)']/df_b['GMV Total (MP)']*100, 0)
        if '%Cost' not in df_b.columns:
            df_b['%Cost'] = np.where(df_b['GMV (Fullfil By BS)']>0, df_b['Total Cost']/df_b['GMV (Fullfil By BS)']*100, 0)

        lr  = df_b.iloc[-1]
        pr  = df_b.iloc[-2] if len(df_b)>1 else lr
        d_cpo = lr['CPO'] - pr['CPO']

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("CPO (Latest)",       f"Rp {lr['CPO']:,.0f}",  f"{'▼' if d_cpo<=0 else '▲'} Rp {abs(d_cpo):,.0f}")
        c2.metric("Total Orders",       f"{lr['Total Order(BS)']:,.0f}")
        c3.metric("BS Contribution %",  f"{lr['Contrib_Pct']:.1f}%")
        c4.metric("% Cost Ratio",       f"{lr['%Cost']:.2f}%")

        # Unit economics combo
        fig_ue = go.Figure()
        fig_ue.add_trace(go.Scatter(x=df_b['Month'],y=df_b['BSA_Calc'],name='Basket Size',
            mode='lines+markers',line=dict(color='#6366F1',width=3)))
        fig_ue.add_trace(go.Scatter(x=df_b['Month'],y=df_b['CPO'],name='CPO',
            mode='lines+markers',line=dict(color='#EF4444',width=3,dash='dot'),
            marker=dict(symbol='diamond'),yaxis='y2'))
        fig_ue.update_layout(height=400,
            yaxis=dict(title="Basket Size (Rp)",showgrid=False),
            yaxis2=dict(title="CPO (Rp)",overlaying='y',side='right',showgrid=True),
            hovermode='x unified',legend=dict(orientation="h",y=1.1),plot_bgcolor=T['chart_bg'],
            title="Unit Economics: Basket Size vs Cost Per Order")
        st.plotly_chart(fig_ue, use_container_width=True)

        # Volume vs Cost + Cost Ratio
        st.divider()
        cv1,cv2 = st.columns(2)
        with cv1:
            fig_vc = go.Figure()
            fig_vc.add_trace(go.Bar(x=df_b['Month'],y=df_b['Total Order(BS)'],
                name='Orders',marker_color=T['accent1'],opacity=.6))
            fig_vc.add_trace(go.Scatter(x=df_b['Month'],y=df_b['Total Cost'],
                name='Total Cost',yaxis='y2',line=dict(color='#F97316',width=3)))
            fig_vc.update_layout(height=360,
                yaxis=dict(title="Orders",showgrid=False),
                yaxis2=dict(title="Cost (Rp)",overlaying='y',side='right'),
                plot_bgcolor=T['chart_bg'], title="Volume vs Cost")
            st.plotly_chart(fig_vc, use_container_width=True)
        with cv2:
            fig_cr = go.Figure(go.Scatter(x=df_b['Month'],y=df_b['%Cost'],
                mode='lines+markers',fill='tozeroy',
                fillcolor='rgba(16,185,129,.1)',line=dict(color='#10B981',width=3)))
            avg_c = df_b['%Cost'].mean()
            fig_cr.add_hline(y=avg_c, line_dash='dash', line_color='gray',
                annotation_text=f"Avg {avg_c:.1f}%")
            fig_cr.update_layout(
        font=dict(color=T['chart_font']),height=360,plot_bgcolor=T['chart_bg'],
                title="% Cost Ratio Trend",yaxis_title="% Cost")
            st.plotly_chart(fig_cr, use_container_width=True)

        # Market share stacked
        st.divider()
        df_b['GMV Non-BS'] = df_b['GMV Total (MP)'] - df_b['GMV (Fullfil By BS)']
        fig_ms = go.Figure()
        fig_ms.add_trace(go.Bar(x=df_b['Month'],y=df_b['GMV (Fullfil By BS)'],
            name='Fulfilled by BS',marker_color=T['accent1']))
        fig_ms.add_trace(go.Bar(x=df_b['Month'],y=df_b['GMV Non-BS'],
            name='Non-BS',marker_color=T['border']))
        fig_ms.update_layout(
        font=dict(color=T['chart_font']),barmode='stack',height=360,
            plot_bgcolor=T['chart_bg'],title="GMV Market Share")
        st.plotly_chart(fig_ms, use_container_width=True)

        # Detail + DOWNLOAD (was missing before)
        st.divider()
        disp_cols = [c for c in ['Month','Total Order(BS)','GMV (Fullfil By BS)',
                                  'Total Cost','CPO','%Cost','Contrib_Pct'] if c in df_b.columns]
        st.dataframe(df_b[disp_cols], use_container_width=True)
        st.download_button("📥 Download Fulfillment Data",
            df_b[disp_cols].to_csv(index=False),
            f"fulfillment_{datetime.now():%Y%m%d}.csv", "text/csv",
            use_container_width=True)
    else:
        st.warning("⚠️ 'BS_Fullfilment_Cost' sheet not found or empty.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 11 — YoY & CHANNEL ACCURACY  ★ NEW TAB ★
# ─────────────────────────────────────────────────────────────────────────────
with tab11:
    st.subheader("📊 Year-over-Year & Channel Accuracy")

    # ── YoY Sales Table ──────────────────────────────────────────
    st.markdown("### 📅 Year-over-Year Sales Comparison")
    if not df_yoy.empty:
        years = [c for c in df_yoy.columns if isinstance(c,int)]

        # Heatmap of monthly sales per year
        fig_yoy = go.Figure()
        colors_yoy = T['colors']
        for i,yr in enumerate(years):
            fig_yoy.add_trace(go.Bar(
                x=df_yoy['Month'] if 'Month' in df_yoy.columns else df_yoy.index,
                y=df_yoy[yr], name=str(yr),
                marker_color=colors_yoy[i % len(colors_yoy)]))
        if 'YoY Growth %' in df_yoy.columns:
            fig_yoy.add_trace(go.Scatter(
                x=df_yoy['Month'] if 'Month' in df_yoy.columns else df_yoy.index,
                y=df_yoy['YoY Growth %'],
                name="YoY Growth %", yaxis='y2',
                mode='lines+markers',
                line=dict(color='#374151',width=2,dash='dot')))
        fig_yoy.update_layout(
        font=dict(color=T['chart_font']),height=440, barmode='group', hovermode='x unified',
            plot_bgcolor=T['chart_bg'],
            yaxis2=dict(title="YoY Growth %",overlaying='y',side='right',showgrid=False),
            legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig_yoy, use_container_width=True)

        # Table with colour gradient
        styled_yoy = df_yoy.style
        for yr in years:
            styled_yoy = styled_yoy.background_gradient(subset=[yr], cmap='Blues')
        if 'YoY Growth %' in df_yoy.columns:
            styled_yoy = styled_yoy.background_gradient(subset=['YoY Growth %'], cmap='RdYlGn')
            styled_yoy = styled_yoy.format({'YoY Growth %': '{:+.1f}%'})
        for yr in years:
            styled_yoy = styled_yoy.format({yr: '{:,.0f}'})
        st.dataframe(styled_yoy, use_container_width=True, height=400)
        st.download_button("📥 Download YoY Table", df_yoy.to_csv(index=False),
            "yoy_comparison.csv", "text/csv")
    else:
        st.info("Need multi-year sales data for YoY analysis.")

    # ── Channel Accuracy ─────────────────────────────────────────
    st.divider()
    st.markdown("### 📡 Forecast Accuracy by Channel")
    if channel_acc:
        frames = []
        for ch, df_ch in channel_acc.items():
            if isinstance(df_ch, pd.DataFrame) and not df_ch.empty:
                frames.append(df_ch)
        if frames:
            ch_combined = pd.concat(frames, ignore_index=True)
            ch_combined['Month_Txt'] = ch_combined['Month'].apply(
                lambda x: x.strftime('%b %Y') if hasattr(x,'strftime') else str(x))

            fig_ch = px.line(ch_combined, x='Month_Txt', y='Accuracy', color='Channel',
                markers=True, title="Monthly Accuracy: Ecomm vs Reseller",
                color_discrete_map={'Ecommerce':'#667eea','Reseller':'#10B981'})
            fig_ch.add_hline(y=80, line_dash='dash', line_color='gray',
                annotation_text='Target 80%')
            fig_ch.update_layout(
        font=dict(color=T['chart_font']),height=380, plot_bgcolor=T['chart_bg'],
                yaxis=dict(range=[40,105]), hovermode='x unified',
                legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig_ch, use_container_width=True)

            # Summary table
            summary_ch = ch_combined.groupby('Channel').agg(
                Avg_Accuracy=('Accuracy','mean'),
                Best_Month_Acc=('Accuracy','max'),
                Worst_Month_Acc=('Accuracy','min')
            ).reset_index()
            st.dataframe(summary_ch.style.background_gradient(subset=['Avg_Accuracy'], cmap='RdYlGn'),
                use_container_width=True, hide_index=True)
        else:
            st.info("No channel accuracy data computed.")
    else:
        st.info("Need Rofo + PO data for channel accuracy analysis.")


# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.markdown("""
<div style="text-align:center;color:#666;font-size:.9rem;padding:1rem;">
  <strong>Inventory Intelligence Pro v7.0</strong> |
  ⚡ Vectorised Performance | 🔔 Smart Alerts | 📊 YoY & Channel Analysis |
  📋 Executive Summary | 🚚 Fulfillment Download | 🔄 Dynamic Month Detection
</div>""", unsafe_allow_html=True)
