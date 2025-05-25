
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.title("🔧 Remaining Useful Life Prediction (CNN-LSTM)")

st.write("Upload a CSV file with the following features:")
st.code("op_setting_3, T2_fan_inlet_temp, T24_LPC_outlet_temp, T50_LPT_outlet_temp, "
        "P2_fan_inlet_pressure, P15_bypass_duct_pressure, P30_HPC_outlet_pressure, "
        "Nf_fan_speed, Nc_core_speed, Ps30_HPC_static_pressure, NRf_corrected_fan_speed, "
        "NRc_corrected_core_speed, BPR_bypass_ratio, farB_burner_fuel_air_ratio, "
        "htBleed_bleed_enthalpy, W32_LPT_coolant_bleed")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_features = [
        'op_setting_3', 'T2_fan_inlet_temp', 'T24_LPC_outlet_temp', 'T50_LPT_outlet_temp',
        'P2_fan_inlet_pressure', 'P15_bypass_duct_pressure', 'P30_HPC_outlet_pressure',
        'Nf_fan_speed', 'Nc_core_speed', 'Ps30_HPC_static_pressure',
        'NRf_corrected_fan_speed', 'NRc_corrected_core_speed', 'BPR_bypass_ratio',
        'farB_burner_fuel_air_ratio', 'htBleed_bleed_enthalpy', 'W32_LPT_coolant_bleed'
    ]

    if not all(col in df.columns for col in required_features):
        st.error("❌ Missing one or more required features in your CSV file.")
    else:
        df = df[required_features]

        # Apply MinMax scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        # Reshape to [1, timesteps, features] for model input
        timesteps = scaled_data.shape[0]
        input_data = scaled_data.reshape(1, timesteps, len(required_features))

        # Load the trained model
        model = load_model('best_cnn_lstm_model.h5')
        prediction = model.predict(input_data)

        st.success("📈 Predicted Remaining Useful Life (RUL): {:.2f}".format(prediction[0][0]))
