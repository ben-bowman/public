# Pricing Recommendation Dashboard

The **Pricing Recommendation Dashboard** is an interactive web application designed to optimize product pricing by analyzing historical sales data from the Northwind database. The dashboard uses a demand model to estimate price elasticity, revenue, and profit, then recommends an optimal price to maximize profit.

## Overview

In many retail scenarios, pricing decisions are challenging due to limited variation in observed prices. In our dataset, most sales occur at the regular price with only rare discount events, which provide the only variation in pricing. This project leverages those variations to build a linear regression model that estimates demand as a function of price and then calculates a profit-maximizing price.

Key aspects include:
- **Demand Modeling:** A linear regression model is used to model demand (`total_quantity_sold`) as a function of net price.
- **Profit Maximization:** Instead of solely maximizing revenue, the model computes a profit-maximizing price based on the equation:  
  \[
  \text{Profit}(p) = (p - C) \times (a + b \times p)
  \]
  where \( C \) (the cost) is assumed to be a percentage of the net price if not provided.
- **Cost Assumption:** A crude cost assumption is used (default 15% of net price), which can be adjusted via a sidebar slider.
- **Visualization:** The dashboard features interactive charts including:
  - A **Price Elasticity Chart** with vertical lines indicating the current net price (red) and the recommended profit-maximizing price (green).
  - A **Demand & Profit Chart** that plots historical demand, the fitted demand curve, and a secondary profit curve with a vertical line indicating the price with maximum profit.
- **Data Limitations:** 
  - The price variation is primarily driven by rare discount events, so the model is largely influenced by these points.
  - The Northwind dataset may not capture all complexities of pricing strategies, and the linear regression approach may not fully capture non-linear dynamics in demand.

## Tools and Technologies

- **Python:** Programming language used for data processing, modeling, and application logic.
- **Streamlit:** Framework for building the interactive web dashboard.
- **SQLite:** Database engine used to host the Northwind data.
- **Pandas & NumPy:** Libraries used for data manipulation and numerical operations.
- **Scikit-learn:** Used for building and evaluating the linear regression model.
- **Matplotlib & Seaborn:** Used for creating visualizations and charts.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/northwind-price-rec.git
   cd northwind-price-rec
