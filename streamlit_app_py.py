
# streamlit_app_py.py

# streamlit_app.py
import openai
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime, timedelta
import altair as alt
from prophet import Prophet
from fpdf import FPDF
from PIL import Image
import plotly.express as px
# Set page config
st.set_page_config(page_title="📊 AI-Powered Sales Forecast Dashboard", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
        .main {background-color: #f9f9f9;}
        .block-container {padding-top: 2rem;}
        .stButton>button {color: white; background: linear-gradient(to right, #00b4db, #0083b0); border: none;}
        .stDownloadButton>button {color: white; background-color: #28a745; border: none;}
        .stMetric {background-color: #ffffff !important; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Powered Sales Forecast & Risk Dashboard")

# ---- File Upload Section ----
st.sidebar.header("📁 Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# ---- Dummy Data Fallback ----
@st.cache_data
def generate_dummy_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq='D')
    products = ['Product A', 'Product B', 'Product C']
    regions = ['North', 'South', 'East', 'West']
    data = []
    for date in dates:
        for product in products:
            for region in regions:
                sales = np.random.poisson(lam=20)
                revenue = sales * np.random.uniform(10, 50)
                risk_flag = 1 if np.random.rand() < 0.05 else 0
                data.append([date, product, region, sales, revenue, risk_flag])
    df = pd.DataFrame(data, columns=['Date', 'Product', 'Region', 'Units Sold', 'Revenue', 'Risk Flag'])
    return df

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        sales_df = pd.read_csv(uploaded_file)
    else:
        sales_df = pd.read_excel(uploaded_file)
else:
    sales_df = generate_dummy_data()

# ---- Sidebar Filters ----
st.sidebar.header("🔍 Filter Data")
selected_products = st.sidebar.multiselect("Select Product(s)", sales_df['Product'].unique(), default=sales_df['Product'].unique())
selected_regions = st.sidebar.multiselect("Select Region(s)", sales_df['Region'].unique(), default=sales_df['Region'].unique())
selected_date = st.sidebar.date_input("Select Date Range", [sales_df['Date'].min(), sales_df['Date'].max()])

filtered_df = sales_df[
    (sales_df['Product'].isin(selected_products)) &
    (sales_df['Region'].isin(selected_regions)) &
    (sales_df['Date'] >= pd.to_datetime(selected_date[0])) &
    (sales_df['Date'] <= pd.to_datetime(selected_date[1]))
]

# ---- KPIs ----
total_revenue = filtered_df['Revenue'].sum()
total_units = filtered_df['Units Sold'].sum()
avg_daily_sales = filtered_df.groupby('Date')['Units Sold'].sum().mean()
risk_alerts = filtered_df['Risk Flag'].sum()

# ---- Sales Over Time Summary ----
sales_over_time = filtered_df.groupby('Date')[['Units Sold', 'Revenue']].sum().reset_index()

st.subheader("📊 Sales Overview Tiles")
tiles = alt.Chart(sales_over_time).mark_area(
    line={'color': 'teal'},
    color=alt.Gradient(
        gradient='linear',
        stops=[alt.GradientStop(color='lightblue', offset=0), alt.GradientStop(color='white', offset=1)],
        x1=1, x2=1, y1=1, y2=0)
).encode(
    x='Date:T',
    y='Revenue:Q'
).properties(height=100, width=300)
st.altair_chart(tiles, use_container_width=False)


# ---- AI Help Assistant ----
openai.api_key = st.secrets["OPENAI_API_KEY"]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": user_query}
    ]
)

st.write(response['choices'][0]['message']['content'])

# ---- Visualization Section ----
st.subheader("📈 Revenue Trend")
sales_over_time = filtered_df.groupby('Date')[['Units Sold', 'Revenue']].sum().reset_index()
fig1 = plt.figure(figsize=(10, 4))
plt.plot(sales_over_time['Date'], sales_over_time['Revenue'], color='teal')
plt.title("Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.grid(True)
st.pyplot(fig1)

st.subheader("🔥 Risk Heatmap by Region")
risk_map = filtered_df.groupby(['Region', 'Date'])['Risk Flag'].sum().unstack().fillna(0)
fig2, ax = plt.subplots(figsize=(10, 3))
sns.heatmap(risk_map, cmap="Reds", cbar=True, ax=ax)
st.pyplot(fig2)

# ---- Filtered Table ----
st.subheader("📋 Filtered Sales Table")
st.dataframe(filtered_df, use_container_width=True)
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Filtered Data", data=csv, file_name='filtered_sales.csv', mime='text/csv')

# ---- Forecasting ----
st.subheader("🔮 Sales Forecast (30 Days)")
forecast_data = filtered_df.groupby('Date')['Revenue'].sum().reset_index()
forecast_data = forecast_data.rename(columns={'Date': 'ds', 'Revenue': 'y'})

if len(forecast_data) > 30:
    model = Prophet()
    model.fit(forecast_data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    st.write("Forecast Graph")
    st.pyplot(model.plot(forecast))
    st.write("Forecast Breakdown")
    st.pyplot(model.plot_components(forecast))
else:
    st.warning("Not enough data for forecasting. Upload more or adjust filters.")

# ---- Risk Warnings ----
st.subheader("⚠️ Risk Warnings")
latest_data = filtered_df[filtered_df['Date'] == filtered_df['Date'].max()]
regions_with_risk = latest_data[latest_data['Risk Flag'] == 1]['Region'].unique()
if len(regions_with_risk) > 0:
    st.error(f"⚠️ Risk Alert: Regions affected - {', '.join(regions_with_risk)}")
else:
    st.success("✅ No recent risk alerts.")

# ---- AI Generated Insights ----
st.subheader("📌 AI Insights")
cutoff = filtered_df['Date'].max() - pd.Timedelta(days=30)
last_month = filtered_df[filtered_df['Date'] > cutoff]
prev_month = filtered_df[(filtered_df['Date'] <= cutoff) & (filtered_df['Date'] > cutoff - pd.Timedelta(days=30))]
insight_text = ""
deltas = []
for region in filtered_df['Region'].unique():
    last_revenue = last_month[last_month['Region'] == region]['Revenue'].sum()
    prev_revenue = prev_month[prev_month['Region'] == region]['Revenue'].sum()
    if prev_revenue > 0:
        change = (last_revenue - prev_revenue) / prev_revenue * 100
        deltas.append({"region": region, "change": change})
        if change < -20:
            insight_text += f"⚠️ Sales in the {region} dropped {abs(change):.1f}% vs last month\n"
        elif change > 20:
            insight_text += f"✅ Sales in the {region} increased {change:.1f}% vs last month\n"

if insight_text:
    st.info(insight_text)
else:
    st.success("Sales performance is stable.")

# ---- Regional Revenue Comparison ----
st.subheader("📉 Regional Change Comparison")
delta_df = pd.DataFrame(deltas)
if not delta_df.empty:
    bar = alt.Chart(delta_df).mark_bar().encode(
        x=alt.X('region:N', title='Region'),
        y=alt.Y('change:Q', title='% Change in Revenue'),
        color=alt.condition(alt.datum.change > 0, alt.value("green"), alt.value("red"))
    ).properties(title="30-Day Revenue Change by Region")
    st.altair_chart(bar, use_container_width=True)

# ---- Bonus Unique Feature: Fun Emoji Summary ----
st.subheader("🎯 Summary in Emojis")
emoji_sum = ""
if total_revenue > 100000:
    emoji_sum += "🤑 High Revenue\n"
if avg_daily_sales > 40:
    emoji_sum += "🚀 Strong Daily Sales\n"
if risk_alerts == 0:
    emoji_sum += "🛡️ Risk-Free Period\n"
else:
    emoji_sum += "⚠️ Stay Alert: Risk Detected\n"
st.markdown(f"**Summary**: \n{emoji_sum}")
st.progress(min(total_revenue / 150000, 1.0), text="Revenue Target")
st.progress(min(total_units / 10000, 1.0), text="Units Sold Target")

metrics_df = pd.DataFrame({
    'Metric': ['Revenue', 'Units Sold', 'Avg Daily Sales', 'Risk Events'],
    'Value': [total_revenue, total_units, avg_daily_sales, risk_alerts]
})

fig = px.line_polar(metrics_df, r='Value', theta='Metric', line_close=True,
                    title="📊 Performance Radar", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)
