import streamlit as st
import pandas as pd
import scipy.stats as stats
import plotly.express as px

# Title
st.title("A/B Testing Dashboard")

# Function to load default dataset
@st.cache_data
def load_default_data():
    return pd.read_csv("data/your_data.csv")  # Adjust the path if needed

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Load either uploaded data or default data
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", data.head())
else:
    data = load_default_data()
    st.write("### Default Dataset", data.head())

# Select columns for Control and Treatment
if not data.empty:
    st.sidebar.header("Select Test Columns")
    control_col = st.sidebar.selectbox("Select Control Group Column", data.columns)
    treatment_col = st.sidebar.selectbox("Select Treatment Group Column", data.columns)

    if control_col and treatment_col:
        control_data = data[control_col]
        treatment_data = data[treatment_col]

        # Perform A/B test
        t_stat, p_value = stats.ttest_ind(control_data, treatment_data, equal_var=False)
        st.write("### A/B Test Results")
        st.write(f"T-statistic: {t_stat:.4f}")
        st.write(f"P-value: {p_value:.4f}")

        # Visualize distributions
        st.write("### Distribution of Groups")
        fig = px.histogram(data, x=control_col, nbins=20, title="Control Group Distribution", opacity=0.6)
        fig.add_trace(px.histogram(data, x=treatment_col, nbins=20, title="Treatment Group Distribution", opacity=0.6).data[0])
        st.plotly_chart(fig)

        # Decision
        st.write("### Conclusion")
        if p_value < 0.05:
            st.success("Reject the null hypothesis. Significant difference detected!")
        else:
            st.warning("Fail to reject the null hypothesis. No significant difference.")
