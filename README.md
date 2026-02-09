# 🚀 Inventory Intelligence Dashboard v6.0

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![Status](https://img.shields.io/badge/Status-Production-success)

**Professional Dashboard for Demand Planning, Inventory Control & Financial Analytics.**
Designed to visualize real-time Supply Chain data from Google Sheets with advanced analytics.

---

## ✨ Key Features

### 📊 Demand Planning
* **Accuracy Heatmaps:** Track forecast accuracy month-over-month.
* **Bias Detection:** Identify systematic under/over-forecasting trends.
* **Performance Triad:** Analyze Gap between Plan (Rofo), Execution (PO), and Result (Sales).

### 📦 Inventory Optimization
* **Stock Health:** Monitor Months of Coverage and Warehouse Utilization.
* **Aging Analysis:** Detect Expired and Critical stock (<30 days).
* **SKU Evaluation:** 360-degree view of individual product performance.

### 💰 Financial Intelligence
* **Profitability:** Waterfall Chart (Revenue ➡️ COGS ➡️ Margin).
* **Unit Economics:** Analyze Basket Size (BSA) vs Cost per Order (CPO).
* **Cost Projection:** 2026 Budgeting simulation based on market share.

---

## 🛠️ Tech Stack

* **Core:** Python
* **UI Framework:** Streamlit
* **Charts:** Plotly Express & Graph Objects
* **Database:** Google Sheets API (Real-time connection)

---

## ⚙️ Quick Setup

### 1. Clone Repository
```bash
git clone [https://github.com/username/inventory-intelligence-pro.git](https://github.com/username/inventory-intelligence-pro.git)
cd inventory-intelligence-pro


2. Install Requirements

pip install -r requirements.txt

3. Configure Secrets
Create a file named .streamlit/secrets.toml and add your Google Cloud credentials

4. Run Application

streamlit run app.py

📂 Data Requirements
The dashboard requires a Google Sheet with the following tabs:

Sheet Name,Description
Product_Master,"SKU details, Prices, Brands, Tiers"
Sales,Historical sales data
Rofo,Rolling Forecast data
PO,Purchase Order history
Stock_Onhand,Current stock & Expiry dates
Forecast_2026_Ecomm,Next year projections (Ecommerce)
Forecast_2026_Reseller,Next year projections (Reseller)
BS_Fullfilment_Cost,Operational cost data for Unit Economics

Author: Mulyanto 

Last Update: Feb-2026

