import streamlit as st
import gspread
import pandas as pd
import plotly.graph_objects as go
from oauth2client.service_account import ServiceAccountCredentials
from io import StringIO

# Google Sheets credentials JSON file (Ensure this file is in the same directory or use environment variables)
GOOGLE_SHEET_CREDENTIALS = "service_account.json"
SHEET_NAME = "sankey_chart_data"
WORKSHEET_NAME = "Sheet1"  # Change if necessary

# Function to load Google Sheet
@st.cache_data(ttl=600)  # Refresh every 10 minutes
def load_google_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_SHEET_CREDENTIALS, scope)
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

    all_labels = list(set(sources + targets))
    label_dict = {label: i for i, label in enumerate(all_labels)}
    
    source_indices = [label_dict[src] for src in sources]
    target_indices = [label_dict[tgt] for tgt in targets]

    # New color scheme and improved text contrast
    node_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"] * (len(all_labels) // 6 + 1)
    link_colors = ["rgba(99, 110, 250, 0.6)", "rgba(239, 85, 59, 0.6)", "rgba(0, 204, 150, 0.6)", "rgba(171, 99, 250, 0.6)"] * (len(values) // 4 + 1)

    # Create Sankey Diagram with improved styling
    sankey_fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="rgba(0,0,0,0)", width=0),  # Remove box around nodes
            label=all_labels,
            color=node_colors,
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

    # Add profile pic
    st.image("ben-square.jpg", caption="Ben Bowman", width=150)
