# the linear model by Pradyuman Singh - 2021B5A71204P
# Group 09, Topic 29

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np
import matplotlib.pyplot as plt


data_dir = "./data"

city = "braga" # or faro or lisbon or setubal

prod_file = os.path.join(data_dir, "prod_"+city+".csv")
rad_file  = os.path.join(data_dir, "rad_"+city+".csv")

prod = pd.read_csv(prod_file)
rad  = pd.read_csv(rad_file)

prod['Date'] = pd.to_datetime(prod['Date'])
rad['time']  = pd.to_datetime(rad['time'])

# merging data
df = pd.merge_asof(
    prod.sort_values('Date'),
    rad.sort_values('time'),
    left_on='Date',
    right_on='time',

    direction='nearest'
)

# data for the night dropped
df = df[df['shortwave_radiation(W/m2)'] > 0]

# train and test split
train_df = df[df['Date'].dt.year.isin([2019, 2020, 2021])]
test_df  = df[df['Date'].dt.year == 2022]

X_train = train_df[['shortwave_radiation(W/m2)', 'temperature(C)', 'cloud_cover(pc)']]
y_train = train_df['Produced Energy (kWh)']

X_test  = test_df[['shortwave_radiation(W/m2)', 'temperature(C)', 'cloud_cover(pc)']]
y_test  = test_df['Produced Energy (kWh)']

model = LinearRegression()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)
print(y_pred)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== Linear Model Results (Train: 2019–2021, Test: 2022) ===")
print("MAE :", mae)
print("RMSE:", rmse)
print("R^2  :", r2)
print("\nCoefficients:")
for name, coef in zip(X_train.columns, model.coef_):
    print(f"{name}: {coef}")
print("\nIntercept:", model.intercept_)

# plotting ------------------------------------------------------------------------------


# preparing teh results dataframe
# ensure test_df is sorted by Date (merge_asof should have kept order but be safe)
test_sorted = test_df.sort_values('Date').reset_index(drop=True)
results = test_sorted[['Date', 'Produced Energy (kWh)']].copy()
results['predicted'] = y_pred
results['residual'] = results['Produced Energy (kWh)'] - results['predicted']

# time series plot (actual vs predicted)
plt.figure(figsize=(14,4))
plt.plot(results['Date'], results['Produced Energy (kWh)'], label='Actual', linewidth=1)
plt.plot(results['Date'], results['predicted'], label='Predicted', linewidth=1, alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Produced Energy (kWh)')
plt.title('Actual vs Predicted — Hourly (Test: 2022)')
plt.legend()
plt.tight_layout()
plt.show()

# scatter plot (predicted vs actual)
plt.figure(figsize=(6,6))
plt.scatter(results['Produced Energy (kWh)'], results['predicted'], alpha=0.4, s=10)
maxv = max(results[['Produced Energy (kWh)','predicted']].max())
plt.plot([0, maxv], [0, maxv], color='red', linestyle='--', label='y = x')
plt.xlabel('Actual Produced Energy (kWh)')
plt.ylabel('Predicted Produced Energy (kWh)')
plt.title('Predicted vs Actual (Hourly)')
plt.legend()
plt.tight_layout()
plt.show()

# residuals histogram (just checks the difference in the predicted and actual power output and makes a histogram)
plt.figure(figsize=(8,4))
plt.hist(results['residual'], bins=50, alpha=0.8)
plt.axvline(results['residual'].mean(), color='red', linestyle='--', label=f"mean={results['residual'].mean():.3f}")
plt.xlabel('Residual (Actual - Predicted) [kWh]')
plt.title('Residuals Distribution (Hourly)')
plt.legend()
plt.tight_layout()
plt.show()

# daily aggregate comparison(for the bigger trends, not that useful)
daily = results.set_index('Date').resample('D').sum()[['Produced Energy (kWh)', 'predicted']]
daily.rename(columns={'Produced Energy (kWh)': 'actual_daily', 'predicted': 'pred_daily'}, inplace=True)

plt.figure(figsize=(14,4))
plt.plot(daily.index, daily['actual_daily'], label='Actual daily sum', linewidth=1)
plt.plot(daily.index, daily['pred_daily'], label='Predicted daily sum', linewidth=1, alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Daily Produced Energy (kWh)')
plt.title('Daily Sum: Actual vs Predicted (Test: 2022)')
plt.legend()
plt.tight_layout()
plt.show()
# plotting ends-----------------------------------------------------------------------------------------


# print aggregated error metrics for daily aggregates (totals)
from sklearn.metrics import mean_absolute_error
daily_mae = mean_absolute_error(daily['actual_daily'], daily['pred_daily'])
print(f"Daily MAE (kWh/day): {daily_mae:.3f}")

# manual prediction from input

print("\nEnter values to get a prediction:")

rad_val = float(input("Shortwave radiation (W/m2): "))
temp_val = float(input("Temperature (C): "))
cloud_val = float(input("Cloud cover (%): "))

user_input = np.array([[rad_val, temp_val, cloud_val]])

predicted_energy = model.predict(user_input)[0]

print(f"\nPredicted Produced Energy (kWh): {predicted_energy}")