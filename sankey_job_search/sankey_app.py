import streamlit as st
import gspread
import pandas as pd
import plotly.graph_objects as go
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
    node_colors = ["#074650", "#009292", "#FE6DB6", "#FEB5DA", "#480091", "#B66DFF", "#B5DAFE", "#6DB6FF"]
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
    st.title("Job Search Sankey Diagram")
        # Descriptive text
    st.write(
        "This Streamlit app dynamically loads data from a Google Sheet and visualizes it as a Sankey diagram. "
        "Sankey diagrams are useful for showing flow between different entities. "
        "This chart represents my current job search, and the results of each application and interview. It documents sources and success rates."
        "Refresh the app to get the latest data from the sheet. See more of my work at [www.benbowman.io](https://www.benbowman.io)."
    )
    st.plotly_chart(sankey_fig, use_container_width=True)

