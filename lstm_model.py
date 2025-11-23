# the LSTM model by Aman Phogat - 2021A7PS0582P
# Group 09, Topic 29

import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

DATA_DIR = Path("./data")
PLANTS = ["braga", "faro", "lisbon", "setubal"]
LOOKBACK = 24
EPOCHS = 12
BATCH_SIZE = 64
RANDOM_SEED = 42

OUT_DIR = Path("output")
PLOTS_DIR = OUT_DIR / "plots"
REPORT_DIR = OUT_DIR / "report"
COMBINED_DIR = OUT_DIR / "combined_model"
for d in [PLOTS_DIR, REPORT_DIR, COMBINED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def normalize_colname(c):
    return str(c).strip().lower().replace(" ", "_")

def detect_shortwave_col(df):
    names = [normalize_colname(c) for c in df.columns]
    candidates = [
        "shortwave_radiation(w/m2)",
        "shortwave_radiation_w/m2",
        "shortwave_radiation",
        "shortwave_radiation(w/m2)"
    ]
    for cand in candidates:
        if cand in names:
            return df.columns[names.index(cand)]
    for i,c in enumerate(names):
        if "shortwave" in c:
            return df.columns[i]
    raise KeyError("shortwave radiation column not found. Columns: " + ", ".join(df.columns.astype(str)))

def build_lstm_model(n_features, lookback=LOOKBACK):
    model = Sequential([
        LSTM(64, input_shape=(lookback, n_features), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def create_sequences(df, feature_cols, target_col="produced_kwh", lookback=LOOKBACK):
    X, y, idx = [], [], []
    arr = df[feature_cols + [target_col]].values
    for i in range(lookback, len(arr)):
        X.append(arr[i-lookback:i, :-1])
        y.append(arr[i, -1])
        idx.append(df.iloc[i]["datetime"])
    return np.array(X), np.array(y), np.array(idx)


results = {}
models_dict = {}
scalerX_dict = {}
scalery_dict = {}
feature_cols_dict = {}
lag_fill_dict = {}

for plant in PLANTS:
    print(f"\n=== {plant.upper()} ===")
    prod_path = DATA_DIR / f"prod_{plant}.csv"
    rad_path  = DATA_DIR / f"rad_{plant}.csv"
    if not prod_path.exists() or not rad_path.exists():
        print("Missing files for", plant, " — skipping.")
        continue

    prod = pd.read_csv(prod_path)
    rad  = pd.read_csv(rad_path)
    prod.columns = [normalize_colname(c) for c in prod.columns]
    rad.columns  = [normalize_colname(c) for c in rad.columns]


    if "datetime" in prod.columns:
        prod["datetime"] = pd.to_datetime(prod["datetime"], errors="coerce")
    else:
        prod["datetime"] = pd.to_datetime(prod.get("date", prod.columns[0]), errors="coerce")
    if "time" in rad.columns:
        rad["datetime"] = pd.to_datetime(rad["time"], errors="coerce")
    elif "datetime" in rad.columns:
        rad["datetime"] = pd.to_datetime(rad["datetime"], errors="coerce")
    else:
        raise KeyError(f"rad file missing time/datetime column for {plant}")


    target_col = None
    for c in prod.columns:
        if "produced" in c and "kwh" in c:
            target_col = c
            break
    if target_col is None:
        numeric_cols = [c for c in prod.columns if c != "datetime" and pd.api.types.is_numeric_dtype(prod[c])]
        if len(numeric_cols) == 0:
            raise KeyError("No numeric target found in production file for " + plant)
        target_col = numeric_cols[0]
    prod = prod[["datetime", target_col]].rename(columns={target_col: "produced_kwh"})


    sw_col = detect_shortwave_col(rad)
    rad = rad.rename(columns={sw_col: "shortwave_radiation"})

    weather_cols = [c for c in rad.columns if c not in ["time", "datetime", "shortwave_radiation"]]
    weather_cols = ["shortwave_radiation"] + weather_cols


    prod = prod.dropna(subset=["datetime"]).sort_values("datetime")
    rad  = rad.dropna(subset=["datetime"]).sort_values("datetime")
    merged = pd.merge_asof(prod, rad[["datetime"] + weather_cols].sort_values("datetime"),
                           on="datetime", direction="nearest", tolerance=pd.Timedelta("30min"))
    merged = merged.dropna(subset=["produced_kwh"]).reset_index(drop=True)
    merged[weather_cols] = merged[weather_cols].fillna(method="ffill").fillna(method="bfill")


    merged["datetime"] = pd.to_datetime(merged["datetime"])
    merged = merged.set_index("datetime").resample("H").mean().reset_index()

    merged_with_zeros = merged.copy()


    merged = merged[merged["shortwave_radiation"] != 0].reset_index(drop=True)

    merged["year"] = merged["datetime"].dt.year
    train_df = merged[merged["year"].isin([2019, 2020, 2021])].copy()
    test_df  = merged[merged["year"] == 2022].copy()
    if train_df.shape[0] == 0 or test_df.shape[0] == 0:
        print(f"Insufficient data after filtering for {plant} (train={len(train_df)}, test={len(test_df)}). Skipping.")
        continue


    train_df["lag1"] = train_df["produced_kwh"].shift(1).fillna(method="bfill")
    test_df["lag1"]  = test_df["produced_kwh"].shift(1).fillna(method="bfill")
    feature_cols = weather_cols + ["lag1"]


    for c in feature_cols + ["produced_kwh"]:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(method="ffill").fillna(method="bfill")
        test_df[c]  = pd.to_numeric(test_df[c], errors="coerce").fillna(method="ffill").fillna(method="bfill")


    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(train_df[feature_cols])
    scaler_y.fit(train_df[["produced_kwh"]])
    train_s = train_df.copy(); test_s = test_df.copy()
    train_s[feature_cols] = scaler_X.transform(train_df[feature_cols])
    test_s[feature_cols]  = scaler_X.transform(test_df[feature_cols])
    train_s["produced_kwh"] = scaler_y.transform(train_df[["produced_kwh"]])
    test_s["produced_kwh"]  = scaler_y.transform(test_df[["produced_kwh"]])


    train_s = train_s.dropna(subset=feature_cols + ["produced_kwh"]).reset_index(drop=True)
    test_s  = test_s.dropna(subset=feature_cols + ["produced_kwh"]).reset_index(drop=True)


    X_train, y_train, idx_train = create_sequences(train_s, feature_cols, target_col="produced_kwh", lookback=LOOKBACK)
    X_test,  y_test,  idx_test  = create_sequences(test_s,  feature_cols, target_col="produced_kwh", lookback=LOOKBACK)

    print("Samples:", "train", X_train.shape, "test", X_test.shape, "features", len(feature_cols))


    model = build_lstm_model(len(feature_cols), lookback=LOOKBACK)
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[es], verbose=1)


    y_pred_s = model.predict(X_test).reshape(-1)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).reshape(-1)
    y_true = scaler_y.inverse_transform(y_test.reshape(-1,1)).reshape(-1)


    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{plant.upper()} RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


    mask_nonzero = y_true != 0
    idx_plot = idx_test[mask_nonzero]
    y_true_plot = y_true[mask_nonzero]
    y_pred_plot = y_pred[mask_nonzero]

    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(idx_plot, y_true_plot, label="Actual", linewidth=1)
    ax.plot(idx_plot, y_pred_plot, label="Predicted", linewidth=1)
    ax.set_title(f"{plant.upper()} - 2022: Actual vs Predicted (non-zero produced_kwh)")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Produced Energy (kWh)")
    ax.legend()
    plt.tight_layout()
    png_path = PLOTS_DIR / f"{plant}_actual_vs_pred.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print("Saved plot:", png_path.resolve())


    fig, ax = plt.subplots(figsize=(14,4))
    ax.scatter(idx_test, y_true, s=6, label="Actual", alpha=0.6)
    ax.scatter(idx_test, y_pred, s=6, label="Predicted", alpha=0.6)
    ax.set_title(f"{plant.upper()} - 2022 Hourly Scatter Actual vs Predicted (incl zeros)")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Produced Energy (kWh)")
    ax.legend()
    plt.tight_layout()
    png_path2 = PLOTS_DIR / f"{plant}_hourly_scatter.png"
    fig.savefig(png_path2, dpi=200)
    plt.close(fig)
    print("Saved plot:", png_path2.resolve())


    m_full = merged_with_zeros.copy()

    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(m_full["datetime"], m_full["produced_kwh"], linewidth=0.8)
    ax.set_title(f"{plant.upper()} - Produced Energy over time (includes zeros)")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Produced Energy (kWh)")
    plt.tight_layout()
    png_path3 = PLOTS_DIR / f"{plant}_produced_overtime.png"
    fig.savefig(png_path3, dpi=200)
    plt.close(fig)
    print("Saved plot:", png_path3.resolve())


    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(m_full["datetime"], m_full["shortwave_radiation"], linewidth=0.8)
    ax.set_title(f"{plant.upper()} - Shortwave Radiation over time")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Shortwave Radiation (W/m2)")
    plt.tight_layout()
    png_path4 = PLOTS_DIR / f"{plant}_radiation_overtime.png"
    fig.savefig(png_path4, dpi=200)
    plt.close(fig)
    print("Saved plot:", png_path4.resolve())


    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(m_full["shortwave_radiation"], m_full["produced_kwh"], s=6, alpha=0.6)
    ax.set_title(f"{plant.upper()} - Produced Energy vs Shortwave Radiation (includes zeros)")
    ax.set_xlabel("Shortwave Radiation (W/m2)")
    ax.set_ylabel("Produced Energy (kWh)")
    plt.tight_layout()
    png_path5 = PLOTS_DIR / f"{plant}_prod_vs_radiation.png"
    fig.savefig(png_path5, dpi=200)
    plt.close(fig)
    print("Saved plot:", png_path5.resolve())


    results[plant] = {"rmse": rmse, "mae": mae, "r2": r2,
                      "png_actual_vs_pred": str(png_path.resolve()),
                      "png_hourly_scatter": str(png_path2.resolve()),
                      "png_produced_overtime": str(png_path3.resolve()),
                      "png_radiation_overtime": str(png_path4.resolve()),
                      "png_prod_vs_radiation": str(png_path5.resolve())}


    models_dict[plant] = model
    scalerX_dict[plant] = scaler_X
    scalery_dict[plant] = scaler_y
    feature_cols_dict[plant] = feature_cols
    lag_fill_dict[plant] = float(train_df["produced_kwh"].mean())


combined_list = []
for plant in PLANTS:
    prod_path = DATA_DIR / f"prod_{plant}.csv"
    rad_path  = DATA_DIR / f"rad_{plant}.csv"
    if not prod_path.exists() or not rad_path.exists():
        continue
    prod = pd.read_csv(prod_path); rad = pd.read_csv(rad_path)
    prod.columns = [normalize_colname(c) for c in prod.columns]
    rad.columns  = [normalize_colname(c) for c in rad.columns]
    if "datetime" in prod.columns:
        prod["datetime"] = pd.to_datetime(prod["datetime"], errors="coerce")
    else:
        prod["datetime"] = pd.to_datetime(prod.get("date", prod.columns[0]), errors="coerce")
    if "time" in rad.columns:
        rad["datetime"] = pd.to_datetime(rad["time"], errors="coerce")
    elif "datetime" in rad.columns:
        rad["datetime"] = pd.to_datetime(rad["datetime"], errors="coerce")

    target_col = None
    for c in prod.columns:
        if "produced" in c and "kwh" in c:
            target_col = c; break
    if target_col is None:
        numeric_cols = [c for c in prod.columns if c != "datetime" and pd.api.types.is_numeric_dtype(prod[c])]
        if len(numeric_cols)==0: continue
        target_col = numeric_cols[0]
    prod = prod[["datetime", target_col]].rename(columns={target_col: "produced_kwh"})

    try:
        swc = detect_shortwave_col(rad)
    except Exception:
        continue
    rad = rad.rename(columns={swc: "shortwave_radiation"})
    rad = rad[rad["shortwave_radiation"] != 0]
    weather_cols = [c for c in rad.columns if c not in ["time","datetime","shortwave_radiation"]]
    weather_cols = ["shortwave_radiation"] + weather_cols
    merged = pd.merge_asof(prod.sort_values("datetime"),
                           rad[["datetime"] + weather_cols].sort_values("datetime"),
                           on="datetime", direction="nearest", tolerance=pd.Timedelta("30min"))
    merged = merged.dropna(subset=["produced_kwh"]).reset_index(drop=True)
    merged[weather_cols] = merged[weather_cols].fillna(method="ffill").fillna(method="bfill")
    merged["datetime"] = pd.to_datetime(merged["datetime"])
    merged = merged.set_index("datetime").resample("H").mean().reset_index()
    merged = merged[merged["shortwave_radiation"] != 0].reset_index(drop=True)
    merged["plant"] = plant
    combined_list.append(merged)

if len(combined_list) > 0:
    combined = pd.concat(combined_list, ignore_index=True)
    combined["year"] = combined["datetime"].dt.year
    train_c = combined[combined["year"].isin([2019,2020,2021])].copy()
    test_c  = combined[combined["year"] == 2022].copy()
    train_c["lag1"] = train_c["produced_kwh"].shift(1).fillna(method="bfill")
    test_c["lag1"]  = test_c["produced_kwh"].shift(1).fillna(method="bfill")

    plant_dummies = pd.get_dummies(train_c["plant"], prefix="plant")
    train_c = pd.concat([train_c.reset_index(drop=True), plant_dummies.reset_index(drop=True)], axis=1)
    plant_dummies_t = pd.get_dummies(test_c["plant"], prefix="plant")
    test_c = pd.concat([test_c.reset_index(drop=True), plant_dummies_t.reset_index(drop=True)], axis=1)
    all_dummy_cols = sorted(set([c for c in train_c.columns if c.startswith("plant_")] + [c for c in test_c.columns if c.startswith("plant_")]))
    for c in all_dummy_cols:
        if c not in train_c.columns: train_c[c] = 0
        if c not in test_c.columns: test_c[c] = 0

    exclude = set(["datetime","produced_kwh","year","plant","lag1"])
    numeric_cols = [c for c in train_c.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_c[c]) and not c.startswith("plant_")]
    feature_cols_comb = numeric_cols + sorted(all_dummy_cols) + ["lag1"]

    scaler_Xc = MinMaxScaler(); scaler_yc = MinMaxScaler()
    scaler_Xc.fit(train_c[feature_cols_comb]); scaler_yc.fit(train_c[["produced_kwh"]])
    train_sc = train_c.copy(); test_sc = test_c.copy()
    train_sc[feature_cols_comb] = scaler_Xc.transform(train_c[feature_cols_comb])
    test_sc[feature_cols_comb]  = scaler_Xc.transform(test_c[feature_cols_comb])
    train_sc["produced_kwh"] = scaler_yc.transform(train_c[["produced_kwh"]])
    test_sc["produced_kwh"]  = scaler_yc.transform(test_c[["produced_kwh"]])
    train_sc = train_sc.dropna(subset=feature_cols_comb + ["produced_kwh"]).reset_index(drop=True)
    test_sc  = test_sc.dropna(subset=feature_cols_comb + ["produced_kwh"]).reset_index(drop=True)

    def create_seq_comb(df, feat_cols, lookback=LOOKBACK):
        X,y,idx = [],[],[]
        data = df[feat_cols + ["produced_kwh"]].values
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, :-1])
            y.append(data[i, -1])
            idx.append(df.iloc[i]["datetime"])
        return np.array(X), np.array(y), np.array(idx)

    Xtr_c, ytr_c, idx_tr_c = create_seq_comb(train_sc, feature_cols_comb, lookback=LOOKBACK)
    Xte_c, yte_c, idx_te_c = create_seq_comb(test_sc,  feature_cols_comb, lookback=LOOKBACK)

    print("Combined samples:", Xtr_c.shape, Xte_c.shape)
    model_c = build_lstm_model(len(feature_cols_comb), lookback=LOOKBACK)
    model_c.fit(Xtr_c, ytr_c, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[EarlyStopping(monitor="val_loss", patience=3)], verbose=1)
    yp_s = model_c.predict(Xte_c).reshape(-1)
    yp = scaler_yc.inverse_transform(yp_s.reshape(-1,1)).reshape(-1)
    yt = scaler_yc.inverse_transform(yte_c.reshape(-1,1)).reshape(-1)
    rmse_c = math.sqrt(mean_squared_error(yt, yp))
    mae_c  = mean_absolute_error(yt, yp)
    r2_c   = r2_score(yt, yp)
    print("COMBINED RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(rmse_c, mae_c, r2_c))

    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(idx_te_c, yt, label="Actual", linewidth=1)
    ax.plot(idx_te_c, yp, label="Predicted", linewidth=1)
    ax.set_title("COMBINED - 2022 Actual vs Predicted")
    ax.set_xlabel("Datetime"); ax.set_ylabel("Produced Energy (kWh)")
    ax.legend()
    png_comb = COMBINED_DIR / "combined.png"
    fig.savefig(png_comb, dpi=200)
    plt.close(fig)
    results["combined"] = {"rmse": rmse_c, "mae": mae_c, "r2": r2_c, "png": str(png_comb.resolve())}
    print("Saved combined plot:", png_comb.resolve())


report_pdf = REPORT_DIR / "Solar_Report.pdf"
with PdfPages(report_pdf) as pdf:

    fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
    ax.text(0.5, 0.8, "Solar Production - 2019-2022\nLSTM Predictions Report", fontsize=18, ha="center")
    ax.text(0.5, 0.72, "Filter: shortwave_radiation != 0 (daylight only)\nTraining: 2019-2021 → Prediction: 2022", fontsize=10, ha="center")
    pdf.savefig(fig); plt.close(fig)

    for p in [k for k in results.keys() if k!="combined"]:
        r = results[p]
        fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
        ax.text(0.5, 0.9, f"{p.upper()} - Metrics", fontsize=16, ha="center")
        ax.text(0.1, 0.75, f"RMSE: {r['rmse']:.4f}", fontsize=12)
        ax.text(0.1, 0.70, f"MAE : {r['mae']:.4f}", fontsize=12)
        ax.text(0.1, 0.65, f"R2  : {r['r2']:.4f}", fontsize=12)
        pdf.savefig(fig); plt.close(fig)


        try:
            img = plt.imread(r['png_actual_vs_pred'])
            fig = plt.figure(figsize=(8.27, 4.5)); plt.imshow(img); plt.axis("off")
            pdf.savefig(fig); plt.close(fig)
        except Exception as e:
            print("Could not include png_actual_vs_pred for", p, e)


        try:
            img = plt.imread(r['png_hourly_scatter'])
            fig = plt.figure(figsize=(8.27, 4.5)); plt.imshow(img); plt.axis("off")
            pdf.savefig(fig); plt.close(fig)
        except Exception as e:
            print("Could not include png_hourly_scatter for", p, e)


        try:
            img = plt.imread(r['png_produced_overtime'])
            fig = plt.figure(figsize=(8.27, 4.5)); plt.imshow(img); plt.axis("off")
            pdf.savefig(fig); plt.close(fig)
        except Exception as e:
            print("Could not include png_produced_overtime for", p, e)


        try:
            img = plt.imread(r['png_radiation_overtime'])
            fig = plt.figure(figsize=(8.27, 4.5)); plt.imshow(img); plt.axis("off")
            pdf.savefig(fig); plt.close(fig)
        except Exception as e:
            print("Could not include png_radiation_overtime for", p, e)


        try:
            img = plt.imread(r['png_prod_vs_radiation'])
            fig = plt.figure(figsize=(8.27, 4.5)); plt.imshow(img); plt.axis("off")
            pdf.savefig(fig); plt.close(fig)
        except Exception as e:
            print("Could not include png_prod_vs_radiation for", p, e)

    if "combined" in results:
        rc = results["combined"]
        fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
        ax.text(0.5, 0.9, "COMBINED - Metrics", fontsize=16, ha="center")
        ax.text(0.1, 0.75, f"RMSE: {rc['rmse']:.4f}", fontsize=12)
        ax.text(0.1, 0.70, f"MAE : {rc['mae']:.4f}", fontsize=12)
        ax.text(0.1, 0.65, f"R2  : {rc['r2']:.4f}", fontsize=12)
        pdf.savefig(fig); plt.close(fig)
        try:
            img = plt.imread(rc['png'])
            fig = plt.figure(figsize=(8.27, 4.5)); plt.imshow(img); plt.axis("off")
            pdf.savefig(fig); plt.close(fig)
        except Exception as e:
            print("Could not include combined png", e)

print("PDF report saved to:", report_pdf.resolve())
print("PNG plots saved to:", PLOTS_DIR.resolve())
print("Combined PNG:", COMBINED_DIR.resolve())


def predict_from_user_input():
    print("\n=== Manual Prediction: Enter Weather Values ===")

    def _ask_float(prompt):
        while True:
            try:
                return float(input(prompt))
            except:
                print("Enter a valid number.")

    sw = _ask_float("Enter shortwave_radiation (W/m2): ")
    temp = _ask_float("Enter temperature (C): ")
    cloud = _ask_float("Enter cloud_cover (pc): ")

    user_row = {
        "shortwave_radiation": sw,
        "temperature(c)": temp,
        "cloud_cover(pc)": cloud
    }

    print("\nPredicted Produced Energy:")
    for plant in sorted(models_dict.keys()):
        model = models_dict[plant]
        scalerX = scalerX_dict[plant]
        scalery = scalery_dict[plant]
        feature_cols = feature_cols_dict[plant]
        lag_fill = lag_fill_dict[plant]

        seq = []
        for i in range(LOOKBACK):
            row = []
            for feat in feature_cols:
                if feat == "lag1":
                    row.append(lag_fill)
                else:
                    row.append(user_row.get(feat, 0.0))
            seq.append(row)

        seq = np.array(seq)
        seq_scaled = scalerX.transform(seq)
        seq_scaled = seq_scaled.reshape(1, LOOKBACK, len(feature_cols))

        pred_scaled = model.predict(seq_scaled).reshape(-1)
        pred = scalery.inverse_transform(pred_scaled.reshape(-1,1))[0][0]

        print(f" - {plant.capitalize()}: {pred:.2f} kWh")



