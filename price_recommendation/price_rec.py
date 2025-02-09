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
    - net_price (float)
    - total_quantity_sold (int)
    - total_orders (int)
    - CategoryName (string)
- A product must have at least 15 records to have a recommendation generated.
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
MIN_RECORDS_THRESHOLD = 15  # Minimum records per product required for recommendation
COST_FACTOR = 0.7           # Estimated cost is 70% of the current net price

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
def load_data(db_path):
    """
    Connect to the SQLite database and load the price_variation table into a DataFrame.
    
    Parameters:
        db_path (str or Path): Full path to the SQLite database file.
    
    Returns:
        pd.DataFrame: DataFrame containing the data from price_variation.
    """
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM price_variation"
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def preprocess_data(df):
    """
    Preprocess the data by converting date strings to datetime objects and performing any
    other necessary cleaning.
    
    Parameters:
        df (pd.DataFrame): Raw DataFrame.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Convert 'OrderDate' to datetime if present
    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        df = df.dropna(subset=['OrderDate'])
    return df

def filter_by_category(df):
    """
    Filter the DataFrame by CategoryName using a sidebar multiselect filter.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Assume that the "CategoryName" column exists
    categories = sorted(df["CategoryName"].unique())
    selected_categories = st.sidebar.multiselect("Select Product Categories", categories, default=categories)
    df = df[df["CategoryName"].isin(selected_categories)]
    return df

# -------------------------------
# Model Training and Evaluation
# -------------------------------
def train_model_for_product(df_product):
    """
    Train a linear regression model to predict total_quantity_sold from net_price for a single product.
    Since the data is time ordered, the first 70% of the data is used for training and the
    last 30% for testing. Model performance (R2 and RMSE on the test set) is returned.
    
    Parameters:
        df_product (pd.DataFrame): DataFrame containing data for one product.
    
    Returns:
        model (LinearRegression): The fitted regression model.
        a (float): Intercept of the linear model.
        b (float): Coefficient (slope) of the net_price feature.
        r2 (float): R-squared metric on the test set.
        rmse (float): Root Mean Squared Error on the test set.
    """
    # Sort the data by OrderDate to respect the time ordering
    df_product = df_product.sort_values(by='OrderDate')
    n = len(df_product)
    train_size = int(0.7 * n)
    
    # Split into training and testing sets (time-based split)
    train_data = df_product.iloc[:train_size]
    test_data = df_product.iloc[train_size:]
    
    X_train = train_data[['net_price']].values
    y_train = train_data['total_quantity_sold'].values
    X_test = test_data[['net_price']].values
    y_test = test_data['total_quantity_sold'].values
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set and compute performance metrics
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    a = model.intercept_
    b = model.coef_[0]
    return model, a, b, r2, rmse

def compute_optimal_price(a, b):
    """
    Compute the optimal price that maximizes revenue given the linear demand model.
    For demand modeled as: Q = a + b * p, the revenue R = p * (a + b * p).
    The optimum is found by solving dR/dp = a + 2b * p = 0, so p = -a/(2b).
    
    Parameters:
        a (float): Intercept from the regression model.
        b (float): Slope (coefficient) from the regression model.
    
    Returns:
        float or None: The computed optimal price. Returns None if b is zero.
    """
    if b == 0:
        return None  # Avoid division by zero; no valid optimum if slope is zero.
    optimal_price = -a / (2 * b)
    return optimal_price

def compute_price_elasticity(a, b, price):
    """
    Compute the price elasticity of demand at a given price.
    Elasticity = (dQ/dp) * (p/Q), where dQ/dp = b and Q = a + b * p.
    
    Parameters:
        a (float): Intercept from the regression model.
        b (float): Slope (coefficient) from the regression model.
        price (float): The price at which to compute elasticity.
    
    Returns:
        float or None: The price elasticity. Returns None if division by zero occurs.
    """
    Q = a + b * price
    if Q == 0:
        return None
    elasticity = b * (price / Q)
    return elasticity

# -------------------------------
# Generate Pricing Recommendations
# -------------------------------
def get_products_with_sufficient_data(df, threshold=MIN_RECORDS_THRESHOLD):
    """
    Identify which products have enough data records to build a reliable model.
    
    Parameters:
        df (pd.DataFrame): The full dataset.
        threshold (int): Minimum number of records required per product.
    
    Returns:
        tuple: (sufficient_products, insufficient_products) lists of ProductIDs.
    """
    product_counts = df['ProductID'].value_counts()
    sufficient_products = product_counts[product_counts >= threshold].index.tolist()
    insufficient_products = product_counts[product_counts < threshold].index.tolist()
    return sufficient_products, insufficient_products

def get_recommendations(df, sufficient_products):
    """
    For each product with sufficient data, train the pricing model, compute the revenue-
    maximizing (optimal) price, adjust based on an estimated cost, and compute performance
    metrics and elasticity. If the computed optimum is not within the observed price range,
    it is clipped to the min/max values.
    
    Parameters:
        df (pd.DataFrame): The full dataset.
        sufficient_products (list): List of ProductIDs that have sufficient data.
    
    Returns:
        pd.DataFrame: A DataFrame containing recommendations and model metrics per product.
    """
    recommendations = []
    for product_id in sufficient_products:
        # Select and sort the product data
        df_product = df[df['ProductID'] == product_id].sort_values(by='OrderDate')
        # Use the most recent net_price as the "current" price
        current_price = df_product['net_price'].iloc[-1]
        
        try:
            model, a, b, r2, rmse = train_model_for_product(df_product)
        except Exception as e:
            # If model training fails, skip this product
            st.warning(f"Model training failed for ProductID {product_id}: {e}")
            continue
        
        # Compute the revenue-maximizing price (optimum)
        optimal_price = compute_optimal_price(a, b)
        
        # Obtain observed net_price range
        min_price = df_product['net_price'].min()
        max_price = df_product['net_price'].max()
        
        # Adjust the optimal price:
        # - If the optimal price is None or falls outside the observed range, clip it.
        # - Ensure that the recommended price is not below the estimated cost.
        cost_estimate = COST_FACTOR * current_price
        if optimal_price is None or np.isnan(optimal_price):
            recommended_price = current_price
        else:
            recommended_price = np.clip(optimal_price, min_price, max_price)
            recommended_price = max(recommended_price, cost_estimate)
        
        # Retrieve the product's category from the "CategoryName" column
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
    # Set the page configuration
    st.set_page_config(page_title="Pricing Recommendation Dashboard", layout="wide")
    st.title("Pricing Recommendation and Price Elasticity Dashboard")
    
    # -------------------------------
    # Dynamic File Path Setup
    # -------------------------------
    try:
        current_dir = Path(__file__).parent
    except NameError:
        current_dir = Path(os.getcwd())
    
    db_path = current_dir.parent / 'data' / 'dbt_output.db'
    
    # -------------------------------
    # Data Loading
    # -------------------------------
    st.sidebar.header("Data Loading")
    with st.spinner("Loading data from SQLite database..."):
        df = load_data(db_path)
    if df.empty:
        st.error("No data loaded. Please check the database connection and table name.")
        return
    st.sidebar.success("Data loaded successfully!")
    
    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    df = preprocess_data(df)
    df = filter_by_category(df)  # Applies filtering using "CategoryName"
    
    # Date Range Filter (if the OrderDate column exists)
    if 'OrderDate' in df.columns:
        st.sidebar.header("Date Filter")
        min_date = df['OrderDate'].min().date()
        max_date = df['OrderDate'].max().date()
        date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
        if isinstance(date_range, list) and len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['OrderDate'] >= pd.to_datetime(start_date)) & (df['OrderDate'] <= pd.to_datetime(end_date))]
    
    # -------------------------------
    # Identify Products with Sufficient Data
    # -------------------------------
    sufficient_products, insufficient_products = get_products_with_sufficient_data(df)
    
    # Generate pricing recommendations for products with sufficient data
    recommendations_df = get_recommendations(df, sufficient_products)
    
    # -------------------------------
    # Display Recommendations
    # -------------------------------
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
        # Summarize insufficient data products (showing ProductID, ProductName, CategoryName, and record count)
        df_insuff = df[df['ProductID'].isin(insufficient_products)].groupby(['ProductID', 'ProductName', 'CategoryName']).size().reset_index(name='RecordCount')
        st.dataframe(df_insuff)
    else:
        st.write("All products have sufficient data for recommendation.")
    
    # -------------------------------
    # Price Elasticity Analysis for a Selected Product
    # -------------------------------
    st.header("Price Elasticity Analysis")
    st.markdown("Select a product from the dropdown below to view its price elasticity and demand curve.")
    
    if not recommendations_df.empty:
        product_options = recommendations_df['ProductID'].tolist()
        selected_product = st.selectbox("Select a Product (by ProductID)", product_options)
        
        # Retrieve the full historical data for the selected product
        df_product = df[df['ProductID'] == selected_product].sort_values(by='OrderDate')
        product_name = df_product['ProductName'].iloc[0]
        
        # Train the model for the selected product
        model, a, b, r2, rmse = train_model_for_product(df_product)
        
        # Generate a range of net_price values between the observed minimum and maximum
        min_price = df_product['net_price'].min()
        max_price = df_product['net_price'].max()
        price_range = np.linspace(min_price, max_price, 100)
        
        # Compute elasticity for each net_price in the range
        elasticity_values = [compute_price_elasticity(a, b, p) for p in price_range]
        
        # Plot the elasticity curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(price_range, elasticity_values, label="Price Elasticity", color="blue")
        # Mark the current net_price on the elasticity plot
        current_price = df_product['net_price'].iloc[-1]
        ax.axvline(x=current_price, color='red', linestyle='--', label="Current Net Price")
        ax.set_xlabel("Net Price")
        ax.set_ylabel("Elasticity")
        ax.set_title(f"Price Elasticity for {product_name}")
        ax.legend()
        st.pyplot(fig)
        
        # -------------------------------
        # Historical Data and Demand Curve
        # -------------------------------
        st.header("Historical Data and Demand Curve")
        st.markdown("The plot below shows the historical relationship between net_price and total_quantity_sold (demand) for the selected product. "
                    "The fitted demand curve (from the regression model) is overlaid in orange.")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df_product, x='net_price', y='total_quantity_sold', ax=ax2, label="Historical Data")
        
        # Generate predictions over the net_price range for the demand curve
        X_range = price_range.reshape(-1, 1)
        y_pred_line = model.predict(X_range)
        ax2.plot(price_range, y_pred_line, color='orange', label="Fitted Demand Curve")
        ax2.set_xlabel("Net Price")
        ax2.set_ylabel("Total Quantity Sold")
        ax2.set_title(f"Demand Curve for {product_name}")
        ax2.legend()
        st.pyplot(fig2)
        
        # Display model performance for the selected product
        st.subheader("Model Performance (Test Set)")
        st.write(f"R-squared: {r2:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
    else:
        st.info("No products with sufficient data available for elasticity analysis.")

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == '__main__':
    main()
