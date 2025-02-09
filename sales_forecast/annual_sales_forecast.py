
import sqlite3
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Step 1: Connect to the Database
try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path(os.getcwd())

db_path = current_dir.parent / 'data' / 'dbt_output.db'
conn = sqlite3.connect(db_path)

# Step 2: Extract and Prepare Data
query = """
SELECT 
    DATE(OrderDate) AS OrderDate, 
    SUM(net_price * total_quantity_sold) AS TotalSales 
FROM 
    price_variation 
GROUP BY 
    DATE(OrderDate)
ORDER BY 
    OrderDate;
"""

sales_data = pd.read_sql_query(query, conn)
sales_data['OrderDate'] = pd.to_datetime(sales_data['OrderDate'])
sales_data = sales_data.set_index('OrderDate')

# Step 3: Aggregate Data Yearly
yearly_sales = sales_data.resample('Y').sum()

# Step 4: Drop First and Last Year (to remove incomplete years)
if len(yearly_sales) > 2:
    yearly_sales = yearly_sales.iloc[1:-1]
else:
    print("Not enough data to drop first and last years. Proceeding with available data.")

# Step 5: Fit the ARIMA Model (No Differencing)
model = ARIMA(yearly_sales['TotalSales'], order=(1, 0, 1))
fitted_model = model.fit()

# Step 6: Forecast Future Sales
forecast_steps = 5  # Forecasting for the next 5 years
forecast = fitted_model.get_forecast(steps=forecast_steps)
forecast_df = forecast.summary_frame()

# Step 7: Assign Proper Dates to Forecast
last_date = yearly_sales.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=forecast_steps, freq='Y')
forecast_df.index = forecast_dates

# Step 8: Prepare Combined Data for Export (Actual Sales + Forecast)
combined_data = yearly_sales.copy()
combined_data['Forecast'] = None  # Initialize forecast column for actual data
forecast_data = forecast_df[['mean']].rename(columns={'mean': 'Forecast'})
final_data = pd.concat([combined_data, forecast_data])

# Step 9: Export to CSV
csv_path = current_dir / 'sales_forecast_output.csv'
final_data.to_csv(csv_path)
print(f"Forecast and actual sales data saved to {csv_path}")

# Step 10: Plotting the Sales Forecast
plt.figure(figsize=(12, 6))
plt.plot(yearly_sales.index, yearly_sales['TotalSales'], label='Actual Sales', marker='o')
plt.plot(forecast_df.index, forecast_df['mean'], label='Forecasted Sales', linestyle='--', marker='x')
plt.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='gray', alpha=0.2)

plt.title('Yearly Sales Forecast')
plt.xlabel('Year')
plt.ylabel('Total Sales ($M)')
plt.ylim(0, 50000000)  # Y-axis from $0 to $50M

# Format y-axis to show values in millions
import matplotlib.ticker as mtick
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1_000_000:.0f}M'))

plt.legend()
plt.savefig('sales_forecast_plot.png')

