# Solar Energy Production Forecasting

Forecasting hourly solar photovoltaic (PV) output using weather and radiation data from four locations in Portugal: **Braga, Faro, Lisbon, and Setúbal**.

Models implemented:

- **Baseline:** Linear Regression  
- **Intermediate:** Random Forest  
- **Advanced:** LSTM

---

## Project Structure

project/
│
├── data/
│ ├── prod_braga.csv
│ ├── rad_braga.csv
│ ├── prod_faro.csv
│ ├── rad_faro.csv
│ ├── prod_lisbon.csv
│ ├── rad_lisbon.csv
│ ├── prod_setubal.csv
│ ├── prod_setubal.csv
│
├── linear_model.py
├── random_forest_model.py
├── lstm_model.py
│
├── plots/
├── requirements.txt
└── README.md

---
Source:  
**Solar Power Production Dataset (Mendeley Data)**  
https://data.mendeley.com/datasets/dbh93b6vp8/3

### Install requirements
pip install -r requirements.txt

### Production Data (hourly)
| Column | Description |
|--------|-------------|
| Date | Timestamp |
| Produced Energy (kWh) | Hourly PV production |
| Specific Energy (kWh/kWp) | PV normalized output |

### Weather Data (hourly)
| Feature | Description |
|---------|-------------|
| shortwave_radiation | W/m² |
| temperature_2m | °C |
| cloud_cover | % |
| humidity, dew point, wind | removed during preprocessing |

Time span: **2019–2022**  
Granularity: **Hourly**

---

## Preprocessing Steps

- Align timestamps using `merge_asof`  
- Remove nighttime rows where radiation = 0  
- Remove low-importance variables:  
  - humidity  
  - dew point  
  - apparent temperature  
  - wind speed / direction  
- Keep only:  
  - `shortwave_radiation`  
  - `temperature`  
  - `cloud_cover`  
- Train/Test split:  
  - **Train:** 2019–2021  
  - **Test:** 2022  

## Models Implemented

### 1. Linear Regression (Baseline)
### 2. Random Forest Regressor (Intermediate)
### 3. LSTM (Advanced) 

- Input features: radiation, temperature, cloud cover  
- Metrics: MAE, RMSE, R²  

## Running the Code

- Running the linear model:
python3 linear_model.py

- Running the random forest model:
python3 random_forest_model.py

- Running the LSTM model:
python3 lstm_model.py
