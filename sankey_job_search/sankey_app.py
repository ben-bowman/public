import streamlit as st
import gspread
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict
from google.oauth2.service_account import Credentials

# Google Sheets settings
SHEET_NAME = "sankey_chart_data"
WORKSHEET_NAME = "Sheet1"  # Change if necessary

# Function to load Google Sheet
@st.cache_data(ttl=600)  # Refresh every 10 minutes
def load_google_sheet():
    # Load credentials from Streamlit secrets
    service_account_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=[
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ])
    
    # Authorize with gspread
    client = gspread.authorize(creds)
    sheet = client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)
    data = sheet.get_all_records()
    
    return pd.DataFrame(data)

# Load the data
df = load_google_sheet()

# Create columns for displaying metrics
m_col1, m_col2, m_col3 = st.columns(3)

# Custom styled metric boxes
def metric_box(column, label, value, bg_color, emoji):
    column.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: white;">
            <div style="font-size: 36px;">{emoji}</div>
            {label}<br><span style="font-size:24px;">{value}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display job search metrics with styled boxes
metric_box(m_col1, "Total Applications", "339", "#074650", "📑")
metric_box(m_col2, "Avg Response Time", "6.5 days", "#009292", "⏳")
metric_box(m_col3, "Days to Offer", "40 days", "#FE6DB6", "📅")

# Check if data is loaded correctly
if df.empty:
    st.error("No data found in the Google Sheet.")
else:
    # Convert columns into nodes and links for Sankey chart
    sources = df["Source"].tolist()
    targets = df["Target"].tolist()
    values = df["Value"].tolist()

    all_labels = list(OrderedDict.fromkeys(sources + targets))
    label_dict = {label: i for i, label in enumerate(all_labels)}
    
    source_indices = [label_dict[src] for src in sources]
    target_indices = [label_dict[tgt] for tgt in targets]
    
    # Custom color scheme
    node_colors = ["#074650", "#009292", "#FE6DB6", "#B66DFF", "#480091", "#B66DFF", "#B5DAFE", "#6DB6FF"]
    link_colors = [
        "rgba(7, 70, 80, 0.5)",   # Converted from "#074650"
        "rgba(0, 146, 146, 0.5)", # Converted from "#009292"
        "rgba(254, 109, 182, 0.5)", # Converted from "#FE6DB6"
        "rgba(254, 181, 218, 0.5)", # Converted from "#FEB5DA"
        "rgba(72, 0, 145, 0.5)",   # Converted from "#480091"
        "rgba(182, 109, 255, 0.5)", # Converted from "#B66DFF"
        "rgba(181, 218, 254, 0.5)", # Converted from "#B5DAFE"
        "rgba(7, 70, 80, 0.5)",   # Converted from "#074650"
        "rgba(0, 146, 146, 0.5)", # Converted from "#009292"
        "rgba(254, 109, 182, 0.5)", # Converted from "#FE6DB6"
        "rgba(254, 181, 218, 0.5)", # Converted from "#FEB5DA"
        "rgba(72, 0, 145, 0.5)",   # Converted from "#480091"
        "rgba(182, 109, 255, 0.5)", # Converted from "#B66DFF"
        "rgba(0, 0, 0, 0)" # Converted from "#6DB6FF" <- FULLY TRANSPARENT SO NO SHOW
    ]

    # Create Sankey Diagram with custom positioning
    sankey_fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="rgba(0,0,0,0)", width=0),  # Remove box around nodes
            label=all_labels,
            color=node_colors,
            x=[0.0, 0.0, 0.0, 0.33, 0.66, .33,   1,   1],  # Custom horizontal positions
            y=[0.3,   1, 0.9, 0.6,  0.5,    0.2, .9, 0.4],  # Adjust for better visualization
            hoverlabel=dict(bgcolor="white", font_size=16, font_family="Arial")  # Improve text contrast
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
    ))
    
    sankey_fig.update_layout(font=dict(size=18, color="black", family="Arial"))

    # Streamlit UI
    st.title("2025 Job Search")
        # Descriptive text
    st.write(
        "This dashboard represents my job search from start to finish. This streamlit dashboard includes key metrics from my search, "               "dynamically loads data from a google workbook using a service account and api, uses a Sankey diagram to show flow from "
        "application to offer, and pie charts to show more application details. I was extremely lucky to find an amazing opportunity "               "quickly. See more of my work at [www.benbowman.io](https://www.benbowman.io)."
    )
    st.plotly_chart(sankey_fig, use_container_width=True)

# Hardcoded data
pie_data = [
    {
        "labels": ["C", "VP", "Head", "Director", "Manager", "Other"], 
        "sizes": [0.5, 6, 5, 45, 19.5, 24], 
        "colors": ['#074650', '#009292', '#FE6DB6', '#FEB5DA', '#480091', '#B66DFF'],
        "title": "Job Levels"
    },
    {
        "labels": ["Yes", "No"], 
        "sizes": [85, 15], 
        "colors":  ['#FEB5DA', '#480091'],
        "title": "Custom Cover Letter?"
    },
    {
        "labels": ["Yes", "No"],
        "sizes": [42, 298], 
        "colors":  ['#FEB5DA', '#480091'],
        "title": "Quick Apply"
    }
]

# Create three columns
col1, col2, col3 = st.columns(3)

# Function to plot a pie chart
def plot_pie_chart(ax, labels, sizes, colors, title):
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)

# Plot and display each pie chart in its column
for col, data in zip([col1, col2, col3], pie_data):
    fig, ax = plt.subplots()
    plot_pie_chart(ax, data["labels"], data["sizes"], data["colors"], data["title"])
    col.pyplot(fig)
