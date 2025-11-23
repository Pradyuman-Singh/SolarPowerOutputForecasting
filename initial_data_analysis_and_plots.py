import pandas as pd
import matplotlib.pyplot as plt
import os

city = "setubal"

prod_file = os.path.join("./data", "prod_"+city+".csv")
rad_file = os.path.join("./data", "rad_"+city+".csv")


prod = pd.read_csv(prod_file)
rad = pd.read_csv(rad_file)

prod['Date'] = pd.to_datetime(prod['Date'], errors='coerce')
rad['time'] = pd.to_datetime(rad['time'], errors='coerce')

print("\n=== Production Data ===")
print(prod.info())
print(prod.isna().sum())
print(prod.describe())

print("\n=== Radiation Data ===")
print(rad.info())
print(rad.isna().sum())
print(rad.describe())

# production vs time
plt.figure(figsize=(10,4))
plt.plot(prod['Date'], prod['Produced Energy (kWh)'])
plt.title("Produced Energy Over Time")
plt.xlabel("Date")
plt.ylabel("kWh")
plt.show()

# radiation vs time
plt.figure(figsize=(10,4))
plt.plot(rad['time'], rad['shortwave_radiation(W/m2)'], color='orange')
plt.title("Solar Radiation Over Time")
plt.xlabel("Date")
plt.ylabel("W/m2")
plt.show()

# merging the datasets
merged = pd.merge_asof(
    prod.sort_values('Date'),
    rad.sort_values('time'),
    left_on='Date',
    right_on='time',
    direction='nearest'
)

# correlation
corr = merged[['Produced Energy (kWh)', 'shortwave_radiation(W/m2)', 'temperature(C)', 'cloud_cover(pc)']].corr()
print("\n=== Correlation Matrix ===")
print(corr)

# ccatter plot fo energy vs radiation
plt.figure(figsize=(6,6))
plt.scatter(merged['shortwave_radiation(W/m2)'], merged['Produced Energy (kWh)'], alpha=0.5)
plt.title("Produced Energy vs Solar Radiation")
plt.xlabel("Radiation (W/m2)")
plt.ylabel("Energy (kWh)")
plt.show()
