"""
Pricing Recommendation Dashboard

This script loads historical pricing and sales data from a SQLite northwind database,
builds per-product pricing recommendation models that optimize profit (via
a fitted demand curve that incorporates cost data), analyzes price elasticity,
and then displays the results and model performance in an interactive Streamlit dashboard.

Assumptions:
- The SQLite database is located in the 'data' folder (one level above this script's directory)
  and is named "northwind.db".
- The SQL query returns individual order details from the "Order Details" table,
  along with OrderDate, ProductID, ProductName, net_price (after applying discount),
  total_quantity_sold, and CategoryName.
- Crude Cost Assumption: Actual cost data is not available, so cost is assumed to be a percentage
  of the net price. This percentage is set by the user via a sidebar slider (default 15%).
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
DEFAULT_COST_PERCENTAGE = 15        # Default cost assumption: 15% of net price

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

    if 'OrderDate' in df.columns:
        df.loc[:, 'OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        df = df.dropna(subset=['OrderDate'])

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
    Compute the revenue-maximizing price based on the demand model Q = a + b * p.
    (Revenue: R(p) = p * (a + b*p) with optimum p* = -a/(2b))
    """
    if b == 0:
        return None
    return -a / (2 * b)

def compute_profit_maximizing_price(a, b, cost):
    """
    Compute the profit-maximizing price given the demand model Q = a + b * p and cost C.
    Profit = (p - C) * (a + b * p)
    Optimum p* = (b * C - a) / (2b)
    """
    if b == 0:
        return None
    return (b * cost - a) / (2 * b)

def compute_price_elasticity(a, b, price):
    """
    Compute the price elasticity of demand at a given price.
    """
    Q = a + b * price
    if Q == 0:
        return None
    return b * (price / Q)

# -------------------------------
# Generate Pricing Recommendations (Profit-Maximizing)
# -------------------------------
def get_products_with_sufficient_data(df, threshold=1):
    """
    Identify products that have at least 'threshold' records.
    Since individual products all have >7000 records, we use threshold=1 to include all.
    """
    product_counts = df['ProductID'].value_counts()
    sufficient_products = product_counts[product_counts >= threshold].index.tolist()
    insufficient_products = product_counts[product_counts < threshold].index.tolist()
    return sufficient_products, insufficient_products

def get_recommendations_with_cost(df, sufficient_products, cost_factor):
    """
    For each product with sufficient data, train the model, compute the profit-maximizing price,
    adjust the price based on observed constraints, and compute performance and elasticity metrics.

    Cost is computed as: cost = cost_factor * current_net_price, where cost_factor is set via the sidebar slider.
    """
    recommendations = []
    for product_id in sufficient_products:
        df_product = df[df['ProductID'] == product_id].sort_values(by='OrderDate').copy()
        current_price = df_product['net_price'].iloc[-1]
        
        # Compute cost using the cost_factor from the slider
        cost_value = cost_factor * current_price

        try:
            model, a, b, r2, rmse = train_model_for_product(df_product)
        except Exception as e:
            st.warning(f"Model training failed for ProductID {product_id}: {e}")
            continue

        # Compute profit-maximizing price: p* = (b * cost - a) / (2b)
        if b != 0:
            profit_optimal_price = (b * cost_value - a) / (2 * b)
        else:
            profit_optimal_price = current_price

        min_price = df_product['net_price'].min()
        max_price = df_product['net_price'].max()
        recommended_price = np.clip(profit_optimal_price, min_price, max_price)
        recommended_price = max(recommended_price, cost_value)

        product_category = df_product["CategoryName"].iloc[0]
        profit_margin_current = (current_price - cost_value) / current_price

        elasticity_value = compute_price_elasticity(a, b, current_price)

        recommendations.append({
            'ProductID': product_id,
            'ProductName': df_product['ProductName'].iloc[0],
            'CategoryName': product_category,
            'CurrentPrice': round(current_price, 2),
            'RecommendedPrice': round(recommended_price, 2),
            'ObservedMinPrice': round(min_price, 2),
            'ObservedMaxPrice': round(max_price, 2),
            'Cost': round(cost_value, 2),
            'ProfitMargin (Current)': round(profit_margin_current, 2),
            'R2': round(r2, 2),
            'RMSE': round(rmse, 2),
            'Elasticity (at Current Price)': round(elasticity_value, 2) if elasticity_value is not None else None
        })
    return pd.DataFrame(recommendations)

# -------------------------------
# Streamlit Dashboard
# -------------------------------
def main():
    st.set_page_config(page_title="Pricing Recommendation Dashboard", layout="wide")
    st.title("Pricing Recommendation Dashboard")

    # Dynamic File Path Setup
    try:
        current_dir = Path(__file__).parent
    except NameError:
        current_dir = Path(os.getcwd())
    db_path = current_dir.parent / 'data' / 'northwind.db'

    # Sidebar: Data Loading Header
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

    # Sidebar: Date Filter
    if 'OrderDate' in df.columns:
        st.sidebar.header("Date Filter")
        min_date = df['OrderDate'].min().date()
        max_date = df['OrderDate'].max().date()
        date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
        if isinstance(date_range, list) and len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['OrderDate'] >= pd.to_datetime(start_date)) & (df['OrderDate'] <= pd.to_datetime(end_date))]

    # Sidebar: Settings (placed at the bottom of the sidebar)
    st.sidebar.markdown("## Settings")
    cost_percentage = st.sidebar.slider("Cost Per Unit (%)", min_value=0, max_value=100, value=DEFAULT_COST_PERCENTAGE)
    cost_factor = cost_percentage / 100.0

    # Include all products (threshold=1)
    sufficient_products, _ = get_products_with_sufficient_data(df, threshold=1)
    recommendations_df = get_recommendations_with_cost(df, sufficient_products, cost_factor)

    # Price Elasticity Analysis Section
    if not recommendations_df.empty:
        product_names = sorted(recommendations_df['ProductName'].tolist())
        selected_product_name = st.selectbox("Select a product from the dropdown to view a price recommendation, elasticity, and demand curve.", product_names)
        selected_product_id = recommendations_df.loc[
            recommendations_df['ProductName'] == selected_product_name, 'ProductID'
        ].iloc[0]

        recommended_price = recommendations_df.loc[
            recommendations_df['ProductID'] == selected_product_id, 'RecommendedPrice'
        ].iloc[0]
        st.write(f"**Recommended Price for {selected_product_name}: ${recommended_price:.2f}**")
        st.header("Price Elasticity Analysis")

        df_product = df[df['ProductID'] == selected_product_id].sort_values(by='OrderDate').copy()
        product_name = df_product['ProductName'].iloc[0]

        model, a, b, r2, rmse = train_model_for_product(df_product)
        current_price = df_product['net_price'].iloc[-1]
        min_price = df_product['net_price'].min()
        max_price = df_product['net_price'].max()
        if min_price == max_price:
            min_price = current_price * 0.95
            max_price = current_price * 1.05

        price_range = np.linspace(min_price, max_price, 100)
        elasticity_values = [compute_price_elasticity(a, b, p) for p in price_range]

        # Plot Price Elasticity with vertical lines for current and recommended prices
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(price_range, elasticity_values, label="Price Elasticity", color="blue")
        ax.axvline(x=current_price, color='red', linestyle='--', label="Current Net Price")
        ax.axvline(x=recommended_price, color='green', linestyle='--', label="Recommended Price")
        ax.set_xlabel("Net Price")
        ax.set_ylabel("Elasticity")
        ax.set_title(f"Price Elasticity for {product_name}")
        ax.legend()
        st.pyplot(fig)

        current_elasticity = compute_price_elasticity(a, b, current_price)
        st.write(f"**Elasticity at the current price:** {current_elasticity:.2f}")
        st.markdown("""
        **Interpretation of Price Elasticity:**
        - **Elastic Demand:** If the absolute value of elasticity is greater than 1, a 1% change in price will lead to more than a 1% change in quantity demanded (demand is very sensitive).
        - **Inelastic Demand:** If the absolute value of elasticity is less than 1, a 1% change in price will result in less than a 1% change in quantity demanded (demand is relatively insensitive).
        - **Unit Elastic Demand:** If the elasticity is approximately equal to 1, a 1% change in price leads to a 1% change in quantity demanded.
        """)

        st.subheader("Historical Data, Demand Curve, and Profit")
        st.markdown(
            "The plot below shows the historical relationship between net_price and total_quantity_sold (demand) for the selected product. "
            "The fitted demand curve (from the regression model) is overlaid in orange. "
            "Additionally, a dashed purple line (on a secondary right y-axis) shows profit in dollars, "
            "calculated as (net_price - cost) * predicted_quantity, where cost is computed using the cost slider."
        )
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df_product, x='net_price', y='total_quantity_sold', ax=ax2, label="Historical Data")
        X_range = price_range.reshape(-1, 1)
        y_pred_line = model.predict(X_range)
        ax2.plot(price_range, y_pred_line, color='orange', label="Fitted Demand Curve")
        ax2.set_xlabel("Net Price")
        ax2.set_ylabel("Total Quantity Sold")
        ax2.set_title(f"Demand Curve for {product_name}")

        # Compute profit: Profit = (price - cost) * predicted_quantity
        cost_value = cost_factor * current_price
        profit_values = (price_range - cost_value) * (a + b * price_range)
        ax2_profit = ax2.twinx()
        ax2_profit.plot(price_range, profit_values, color='purple', linestyle='--', label="Profit ($)")
        ax2_profit.set_ylabel("Profit ($)")

        # Add a vertical line at the price corresponding to maximum profit
        max_profit_index = np.argmax(profit_values)
        max_profit_price = price_range[max_profit_index]
        ax2.axvline(x=max_profit_price, color='magenta', linestyle=':', label="Max Profit Price")

        # Combine legends from both axes
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_profit.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        st.pyplot(fig2)
        st.markdown("""
        **Interpretation of Demand and Profit:**
        - **Demand Curve:** Demand is plotted as linear equation Q = a + b * p.
        - **Cost:** Comes from the cost slider.
        - **Profit:** Compute the profit-maximizing price given demand model and cost (C) above.
          Profit = (p - C) * (a + b * p)
          Optimum p* = (b * C - a) / (2b)
        """)
        st.subheader("Model Performance (Test Set)")
        st.markdown("""
        **R² (Coefficient of Determination):** R² a measure of how much the model explains what we see in the data. An R² close to 1 means that most of Q is explained by price (good). An R² near 0 means little is explained by price (bad).
    
        **RMSE (Root Mean Squared Error):** the average difference between the predicted and observed values. A lower RMSE suggests a better predictive accuracy (good). A higher RMSE suggests predictions are off (bad).
""")

        st.write(f"R-squared: {r2:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
    else:
        st.info("No products with sufficient data available for elasticity analysis.")

    # Move the Summary Price Recommendations to the bottom
    st.header("Summary Price Recommendations")
    st.markdown(
        "The table below summarizes the pricing recommendations based on the profit-maximizing model "
        "and the cost assumption (Cost Per Unit % slider). This summary helps you review the recommended "
        "prices, the current profit margins, and model performance metrics for each product."
    )
    st.dataframe(recommendations_df)

if __name__ == '__main__':
    main()
