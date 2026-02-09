🚀 Inventory Intelligence Dashboard v6.0
Professional Dashboard for Demand Planning, Inventory Control & Financial Analytics.

Built with Streamlit & Plotly to visualize real-time Supply Chain data.

✨ Key Features
📈 Forecast Accuracy: Heatmaps, Bias Detection, and Performance Triad (Plan vs Exec vs Result).

📦 Inventory Health: Stock Coverage Analysis, Aging Profile, and Warehouse Utilization.

💰 Profitability: Financial Waterfall, Unit Economics, and Pareto Analysis (80/20 Rule).

🔮 Future Planning: 2026 Forecast Projections, Scenario Planner (What-If), and Anomaly Detection.

🚚 Operational: Fulfillment Cost Efficiency (CPO) & Reseller Performance Tracking.

🛠️ Tech Stack
Python 3.10+

Streamlit (UI Framework)

Plotly (Interactive Charts)

Google Sheets API (Real-time Data Source)

⚙️ Quick Start
1. Installation

git clone https://github.com/username/inventory-intelligence-pro.git
cd inventory-intelligence-pro
pip install -r requirements.txt
2. Configure Secrets
Create a file .streamlit/secrets.toml and add your Google Cloud credentials:
📂 Data Requirements (Google Sheets)
Ensure your connected Google Sheet has the following tabs:

Product_Master

Sales & Sales_Reseller

Rofo & Past_Rofo_Reseller

PO & Past_PO_Reseller

Stock_Onhand

Forecast_2026_Ecomm & Forecast_2026_Reseller

BS_Fullfilment_Cost

Author: Mulyanto
