
# Annual Sales Forecasting

## Project Overview

This project demonstrates a basic approach to **forecasting annual sales** using the ARIMA model. The data is sourced from the **Northwind database**, a sample dataset commonly used for demonstrating database concepts. The project covers data extraction, preprocessing, model building, and visualization.

## Purpose

The primary goal of this project is to showcase **time series forecasting techniques** in Python, particularly using the **ARIMA** model. 

> **Note**: The dataset used here is limited and does not provide enough data for highly accurate forecasts. This project is intended as a **simple exercise** to demonstrate the workflow of sales forecasting.

## Methodology

1. **Data Extraction**:
   - Connected to the `dbt_output.db` SQLite database using a dynamic path.
   - Extracted sales data from the `price_variation` table, focusing on order dates and total sales.

2. **Data Preparation**:
   - Aggregated daily sales data into **annual totals**. Annual because the data was too sparse for less granular.
   - Removed the **first and last years** from the dataset to avoid using partial years.

3. **Modeling**:
   - Used the **ARIMA (AutoRegressive Integrated Moving Average)** model with parameters `(1, 0, 1)` to forecast future sales.
   - No differencing was applied, as the data was treated as stationary for simplicity.

4. **Forecasting & Visualization**:
   - Forecasted sales for the **next 5 years**.
   - Generated a plot of actual vs. forecasted sales.

## Outputs

1. **CSV File**:
   - `sales_forecast_output.csv`: Contains actual annual sales and forecasted sales for the next 5 years.

2. **Sales Forecast Plot**:
   - `sales_forecast_plot.png`: A visualization of actual and forecasted sales.

   The plot includes:
   - **Actual Sales** (plotted with solid lines and markers).
   - **Forecasted Sales** (plotted with dashed lines and markers).
   - **Confidence Intervals** shown in shaded gray areas.

## Limitations

- The Northwind dataset provides **limited historical data**, which impacts the **accuracy** and **reliability** of the forecast.
- This project serves as a **demonstration of the forecasting process**, not as a production-ready model for real-world applications.
- For more accurate forecasting, a **larger dataset** with **more historical records** and possibly **seasonal adjustments** would be required.

## How to Run

1. Run the script using:

    ```bash
    python annual_sales_forecast_with_axis_labels.py
    ```

3. The outputs (`sales_forecast_output.csv` and `sales_forecast_plot.png`) will be generated in the same directory as the script.
