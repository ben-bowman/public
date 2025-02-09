"""
Pricing Recommendation Dashboard

This script loads historical pricing and sales data from a SQLite database,
builds per-product pricing recommendation models that optimize revenue (via
a fitted demand curve), analyzes price elasticity, and then displays the
results and model performance in an interactive Streamlit dashboard.

Assumptions:
- The SQLite database is located in the 'data' folder (one level above this script's directory)
  and is named "dbt_output.db".
- The table "price_variation" exists and includes the following columns:
    - OrderID (int)
    - OrderDate (string or date) — will be parsed as a datetime
    - order_year (int)
    - ProductID (int)
    - ProductName (string)
    - net_price (string; may include currency symbols and commas and will be cleaned)
    - total_quantity_sold (int)
    - total_orders (int)
    - CategoryName (string)
- A product must have at least a certain number of records (default relaxed to 10) for a recommendation to be generated.
"""

from pathlib import Path
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------------
# Global Constants and Settings
# -------------------------------
DEFAULT_MIN_RECORDS_THRESHOLD = 10  # Relaxed threshold (default is now 10 instead of 15)
COST_FACTOR = 0.7  # Estimated cost is 70% of the current net price

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
def load_data(db_path):
    query = """
        SELECT 
            od.OrderID,
            o.OrderDate,
            strftime('%Y', o.OrderDate) AS order_year,
            od.ProductID,
            p.ProductName,
            p.CategoryID,
            c.CategoryName,
            ROUND(od.UnitPrice * (1 - od.Discount), 2) AS net_price,
            od.Quantity AS total_quantity_sold
        FROM "Order Details" od
        JOIN products p ON od.ProductID = p.ProductID
        JOIN orders o ON od.OrderID = o.OrderID
        JOIN categories c ON p.CategoryID = c.CategoryID
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def preprocess_data(df):
    """
    Preprocess the data by:
      - Converting 'OrderDate' to datetime.
      - Cleaning and converting 'net_price' to numeric.
    """
    df = df.copy()  # Avoid modifying a view
    
    # Convert 'OrderDate' to datetime if present
    if 'OrderDate' in df.columns:
        df.loc[:, 'OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        df = df.dropna(subset=['OrderDate'])
    
    # Clean and convert 'net_price' to numeric.
    # Only try to clean if the column is of a string/object type.
    if 'net_price' in df.columns:
        if pd.api.types.is_string_dtype(df['net_price']):
            df.loc[:, 'net_price'] = df['net_price'].str.replace(r'[\$,]', '', regex=True)
        df.loc[:, 'net_price'] = pd.to_numeric(df['net_price'], errors='coerce')
        df = df.dropna(subset=['net_price'])
    
    return df

def filter_by_category(df):
    """
    Filter the DataFrame by CategoryName using a sidebar multiselect filter.
    """
    df = df.copy()
    categories = sorted(df["CategoryName"].unique())
    selected_categories = st.sidebar.multiselect("Select Product Categories", categories, default=categories)
    df = df.loc[df["CategoryName"].isin(selected_categories)]
    return df

# -------------------------------
# Model Training and Evaluation
# -------------------------------
def train_model_for_product(df_product):
    """
    Train a linear regression model to predict total_quantity_sold from net_price for a single product.
    The first 70% of the time-ordered data is used for training and the remaining 30% for testing.
    """
    df_product = df_product.sort_values(by='OrderDate').copy()
    n = len(df_product)
    train_size = int(0.7 * n)
    
    train_data = df_product.iloc[:train_size].copy()
    test_data = df_product.iloc[train_size:].copy()
    
    X_train = train_data[['net_price']].values
    y_train = train_data['total_quantity_sold'].values
    X_test = test_data[['net_price']].values
    y_test = test_data['total_quantity_sold'].values
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    a = model.intercept_
    b = model.coef_[0]
    return model, a, b, r2, rmse

def compute_optimal_price(a, b):
    """
    Compute the optimal price that maximizes revenue based on the demand model Q = a + b * p.
    """
    if b == 0:
        return None
    return -a / (2 * b)

def compute_price_elasticity(a, b, price):
    """
    Compute the price elasticity of demand at a given price.
    """
    Q = a + b * price
    if Q == 0:
        return None
    return b * (price / Q)

# -------------------------------
# Generate Pricing Recommendations
# -------------------------------
def get_products_with_sufficient_data(df, threshold):
    """
    Identify products that have at least 'threshold' records.
    """
    product_counts = df['ProductID'].value_counts()
    sufficient_products = product_counts[product_counts >= threshold].index.tolist()
    insufficient_products = product_counts[product_counts < threshold].index.tolist()
    return sufficient_products, insufficient_products

def get_recommendations(df, sufficient_products):
    """
    For each product with sufficient data, train the model, compute the revenue-maximizing price,
    adjust the price based on cost considerations, and compute performance and elasticity metrics.
    """
    recommendations = []
    for product_id in sufficient_products:
        df_product = df[df['ProductID'] == product_id].sort_values(by='OrderDate').copy()
        current_price = df_product['net_price'].iloc[-1]
        
        try:
            model, a, b, r2, rmse = train_model_for_product(df_product)
        except Exception as e:
            st.warning(f"Model training failed for ProductID {product_id}: {e}")
            continue
        
        optimal_price = compute_optimal_price(a, b)
        min_price = df_product['net_price'].min()
        max_price = df_product['net_price'].max()
        cost_estimate = COST_FACTOR * current_price
        
        if optimal_price is None or np.isnan(optimal_price):
            recommended_price = current_price
        else:
            recommended_price = np.clip(optimal_price, min_price, max_price)
            recommended_price = max(recommended_price, cost_estimate)
        
        product_category = df_product["CategoryName"].iloc[0]
        
        recommendations.append({
            'ProductID': product_id,
            'ProductName': df_product['ProductName'].iloc[0],
            'CategoryName': product_category,
            'CurrentPrice': round(current_price, 2),
            'RecommendedPrice': round(recommended_price, 2),
            'ObservedMinPrice': round(min_price, 2),
            'ObservedMaxPrice': round(max_price, 2),
            'EstimatedCost': round(cost_estimate, 2),
            'R2': round(r2, 2),
            'RMSE': round(rmse, 2),
            'Elasticity (at Current Price)': round(
                elasticity, 2) if (elasticity := compute_price_elasticity(a, b, current_price)) is not None else None
        })
    return pd.DataFrame(recommendations)

# -------------------------------
# Streamlit Dashboard
# -------------------------------
def main():
    st.set_page_config(page_title="Pricing Recommendation Dashboard", layout="wide")
    st.title("Pricing Recommendation and Price Elasticity Dashboard")
    
    # Dynamic File Path Setup
    try:
        current_dir = Path(__file__).parent
    except NameError:
        current_dir = Path(os.getcwd())
    db_path = current_dir.parent / 'data' / 'dbt_output.db'
    
    # Data Loading
    st.sidebar.header("Data Loading")
    with st.spinner("Loading data from SQLite database..."):
        df = load_data(db_path)
    if df.empty:
        st.error("No data loaded. Please check the database connection and table name.")
        return
    st.sidebar.success("Data loaded successfully!")
    
    # Data Preprocessing and Filtering
    df = preprocess_data(df)
    df = filter_by_category(df)
    
    # Date Range Filter
    if 'OrderDate' in df.columns:
        st.sidebar.header("Date Filter")
        min_date = df['OrderDate'].min().date()
        max_date = df['OrderDate'].max().date()
        date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
        if isinstance(date_range, list) and len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['OrderDate'] >= pd.to_datetime(start_date)) & (df['OrderDate'] <= pd.to_datetime(end_date))]
    
    # Allow user to adjust the minimum records threshold
    min_records_threshold = st.sidebar.slider("Minimum Records per Product", min_value=1, max_value=30, value=DEFAULT_MIN_RECORDS_THRESHOLD)
    
    sufficient_products, insufficient_products = get_products_with_sufficient_data(df, min_records_threshold)
    recommendations_df = get_recommendations(df, sufficient_products)
    
    # Display Recommendations
    st.header("Pricing Recommendations")
    st.markdown(
        """
        The table below shows, for each product with sufficient historical data:
        - The current net price (most recent record)
        - The recommended price (optimizing revenue while ensuring price is above cost)
        - The observed net price range (min and max net prices)
        - An estimated cost (assumed to be 70% of the current net price)
        - Model performance metrics (R² and RMSE from a time-based test set)
        - Price elasticity at the current net price
        - Product category (from CategoryName)
        """
    )
    st.dataframe(recommendations_df)
    
    st.subheader("Products with Insufficient Data")
    if insufficient_products:
        df_insuff = df[df['ProductID'].isin(insufficient_products)].groupby(['ProductID', 'ProductName', 'CategoryName']).size().reset_index(name='RecordCount')
        st.dataframe(df_insuff)
    else:
        st.write("All products have sufficient data for recommendation.")
    
    # Price Elasticity Analysis
    st.header("Price Elasticity Analysis")
    st.markdown("Select a product from the dropdown below to view its price elasticity and demand curve.")
    
    if not recommendations_df.empty:
        # Use product names for the dropdown
        product_names = recommendations_df['ProductName'].tolist()
        selected_product_name = st.selectbox("Select a Product", product_names)
        # Retrieve the corresponding ProductID for filtering the full dataset
        selected_product_id = recommendations_df.loc[recommendations_df['ProductName'] == selected_product_name, 'ProductID'].iloc[0]
        
        df_product = df[df['ProductID'] == selected_product_id].sort_values(by='OrderDate').copy()
        product_name = df_product['ProductName'].iloc[0]
        
        model, a, b, r2, rmse = train_model_for_product(df_product)
        current_price = df_product['net_price'].iloc[-1]
        min_price = df_product['net_price'].min()
        max_price = df_product['net_price'].max()
        # If only one unique price exists, create a small range around it
        if min_price == max_price:
            min_price = current_price * 0.95
            max_price = current_price * 1.05
        
        price_range = np.linspace(min_price, max_price, 100)
        elasticity_values = [compute_price_elasticity(a, b, p) for p in price_range]
        
        # Plot Price Elasticity
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(price_range, elasticity_values, label="Price Elasticity", color="blue")
        ax.axvline(x=current_price, color='red', linestyle='--', label="Current Net Price")
        ax.set_xlabel("Net Price")
        ax.set_ylabel("Elasticity")
        ax.set_title(f"Price Elasticity for {product_name}")
        ax.legend()
        st.pyplot(fig)
        
        # Calculate elasticity at the current price and display it with interpretation
        current_elasticity = compute_price_elasticity(a, b, current_price)
        st.write(f"**Elasticity at the current price:** {current_elasticity:.2f}")
        st.markdown("""
        **Interpretation of Price Elasticity:**
        - **Elastic Demand:** If the absolute value of elasticity is greater than 1, a 1% change in price will lead to more than a 1% change in quantity demanded. This indicates that demand is very sensitive to price changes.
        - **Inelastic Demand:** If the absolute value of elasticity is less than 1, a 1% change in price will result in less than a 1% change in quantity demanded. This means that demand is relatively insensitive to price changes.
        - **Unit Elastic Demand:** If the elasticity is approximately equal to 1, a 1% change in price leads to a 1% change in quantity demanded.
        """)
        
        st.subheader("Historical Data and Demand Curve")
        st.markdown("The plot below shows the historical relationship between net_price and total_quantity_sold (demand) for the selected product. The fitted demand curve (from the regression model) is overlaid in orange.")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df_product, x='net_price', y='total_quantity_sold', ax=ax2, label="Historical Data")
        X_range = price_range.reshape(-1, 1)
        y_pred_line = model.predict(X_range)
        ax2.plot(price_range, y_pred_line, color='orange', label="Fitted Demand Curve")
        ax2.set_xlabel("Net Price")
        ax2.set_ylabel("Total Quantity Sold")
        ax2.set_title(f"Demand Curve for {product_name}")
        ax2.legend()
        st.pyplot(fig2)
        
        st.subheader("Model Performance (Test Set)")
        st.write(f"R-squared: {r2:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
    else:
        st.info("No products with sufficient data available for elasticity analysis.")

if __name__ == '__main__':
    main()
