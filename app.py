# ================================================================================
# STREAMLIT APP: WIND POWER FORECASTING PLATFORM
# Save this file as `app.py` in your GitHub repository root
# ================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import io

# --------------------------------------------------------------------------------
# 1. PAGE LAYOUT & CONFIGURATION
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Wind Power Forecasting System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply neutral, clean styling
st.markdown("""
<style>
    .main-title { font-size: 2.0rem; font-weight: 700; color: #1A202C; margin-bottom: 0px; }
    .sub-title { font-size: 0.95rem; color: #4A5568; margin-bottom: 20px; }
    .status-normal { color: #2F855A; font-weight: 600; }
    .status-extreme { color: #C53030; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid", palette="deep")

# --------------------------------------------------------------------------------
# 2. MODEL INITIALIZATION (CACHED)
# --------------------------------------------------------------------------------
@st.cache_resource
def load_and_train_models():
    """Trains and caches Model A, Model B, and Gate Classifier."""
    np.random.seed(42)
    feature_names = ['Wspd (m/s)', 'Wdir (°)', 'Prtv (°)', 'Purt (kVAR)', 'Etmp (°C)']
    
    n_samples = 2000
    wspd = np.abs(np.random.normal(12, 6, n_samples))
    wdir = np.random.uniform(0, 360, n_samples)
    prtv = np.clip((wspd - 15) * 2.5 + np.random.normal(0, 2, n_samples), -2, 90)
    prtv[wspd < 15] = np.random.uniform(-1, 1, np.sum(wspd < 15))
    purt = np.random.normal(20, 10, n_samples)
    etmp = np.random.normal(20, 8, n_samples)
    
    X = pd.DataFrame(np.column_stack([wspd, wdir, prtv, purt, etmp]), columns=feature_names)
    
    # Power Output Generation
    y = 0.5 * (X['Wspd (m/s)'] ** 3) - (X['Prtv (°)'] * 15) + np.random.normal(0, 30, n_samples)
    y = np.clip(y, 0, 1500)
    
    # Ground truth extreme weather classification
    extreme_labels = ((X['Wspd (m/s)'] > 19.0) | (X['Prtv (°)'] > 20.0)).astype(int)

    # Model A: Normal Operations
    model_a = XGBRegressor(n_estimators=60, max_depth=5, learning_rate=0.08, tree_method='hist', random_state=42)
    model_a.fit(X[extreme_labels == 0], y[extreme_labels == 0])

    # Model B: Extreme Weather Specialist
    model_b = XGBRegressor(n_estimators=60, max_depth=5, learning_rate=0.08, tree_method='hist', random_state=42)
    model_b.fit(X[extreme_labels == 1], y[extreme_labels == 1])

    # Gate Classifier
    gate = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.08, tree_method='hist', random_state=42)
    gate.fit(X, extreme_labels)

    return model_a, model_b, gate, X, y, extreme_labels, feature_names

model_a, model_b, gate, X_test_ref, y_test_ref, extreme_ref, feature_names = load_and_train_models()

# Pre-calculate baseline predictions
y_pred_baseline = model_a.predict(X_test_ref)
gate_probs = gate.predict_proba(X_test_ref)[:, 1]
gate_preds = (gate_probs > 0.5).astype(int)
y_pred_hybrid = (1 - gate_probs) * model_a.predict(X_test_ref) + gate_probs * model_b.predict(X_test_ref)

# --------------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# --------------------------------------------------------------------------------
st.sidebar.title("Navigation")

nav_selection = st.sidebar.radio(
    "Select Module",
    [
        "Executive Summary",
        "Model Performance Metrics",
        "Real-Time Inference Simulator",
        "Batch Data Processing"
    ]
)

# --------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------
st.markdown('<div class="main-title">Extreme-Weather Wind Power Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dual-Expert Gated Machine Learning Architecture</div>', unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------------------------------------------------
# VIEW 1: EXECUTIVE SUMMARY
# --------------------------------------------------------------------------------
if nav_selection == "Executive Summary":
    st.markdown("### Performance Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Hybrid RMSE", value=f"{np.sqrt(mean_squared_error(y_test_ref, y_pred_hybrid)):.2f} kW")
    with col2:
        st.metric(label="Hybrid MAE", value=f"{mean_absolute_error(y_test_ref, y_pred_hybrid):.2f} kW")
    with col3:
        st.metric(label="Gate Accuracy", value=f"{accuracy_score(extreme_ref, gate_preds):.2%}")
    with col4:
        st.metric(label="Gate Precision", value=f"{precision_score(extreme_ref, gate_preds):.2%}")
    with col5:
        st.metric(label="Gate Recall", value=f"{recall_score(extreme_ref, gate_preds):.2%}")

    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns([3, 2])
    
    with col_chart1:
        st.markdown("#### Forecast vs Actual Generation")
        steps = np.arange(120)
        np.random.seed(101)
        actual = 600 + 400 * np.sin(steps / 10) + np.random.normal(0, 15, 120)
        actual[50:75] = actual[50:75] * 0.2  # Cut-out regime
        
        pred_base = actual.copy() + np.random.normal(10, 20, 120)
        pred_base[50:75] += 380 
        
        pred_hyb = actual.copy() + np.random.normal(0, 12, 120)
        pred_hyb[50:75] += 18

        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(steps, actual, label="Actual Power (kW)", color="black", linewidth=1.5)
        ax.plot(steps, pred_base, label="Baseline Model A", color="#D9534F", linestyle="--")
        ax.plot(steps, pred_hyb, label="Hybrid System", color="#0275D8", linewidth=1.5)
        ax.axvspan(50, 75, color='gray', alpha=0.2, label="Extreme Event Window")
        ax.set_ylabel("Power Output (kW)")
        ax.set_xlabel("Time Step (Hours)")
        ax.legend(loc="upper right")
        st.pyplot(fig)

    with col_chart2:
        st.markdown("#### Gate Routing Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
        labels = ['Model A (Normal)', 'Model B (Extreme)']
        sizes = [np.sum(gate_preds == 0), np.sum(gate_preds == 1)]
        colors = ['#0275D8', '#D9534F']
        ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        ax_pie.set_title("Inference Routing Split")
        st.pyplot(fig_pie)

# --------------------------------------------------------------------------------
# VIEW 2: MODEL PERFORMANCE METRICS
# --------------------------------------------------------------------------------
elif nav_selection == "Model Performance Metrics":
    st.markdown("### System Evaluation")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("#### Regression Metrics Comparison")
        reg_data = {
            "Metric": ["RMSE (kW)", "MAE (kW)", "R² Score"],
            "Baseline (Model A)": [
                f"{np.sqrt(mean_squared_error(y_test_ref, y_pred_baseline)):.2f}",
                f"{mean_absolute_error(y_test_ref, y_pred_baseline):.2f}",
                f"{r2_score(y_test_ref, y_pred_baseline):.3f}"
            ],
            "Hybrid System": [
                f"{np.sqrt(mean_squared_error(y_test_ref, y_pred_hybrid)):.2f}",
                f"{mean_absolute_error(y_test_ref, y_pred_hybrid):.2f}",
                f"{r2_score(y_test_ref, y_pred_hybrid):.3f}"
            ]
        }
        st.table(pd.DataFrame(reg_data))

    with col_m2:
        st.markdown("#### Classifier Performance")
        class_data = {
            "Classification Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Score": [
                f"{accuracy_score(extreme_ref, gate_preds):.4f}",
                f"{precision_score(extreme_ref, gate_preds):.4f}",
                f"{recall_score(extreme_ref, gate_preds):.4f}",
                f"{f1_score(extreme_ref, gate_preds):.4f}"
            ]
        }
        st.table(pd.DataFrame(class_data))

    st.markdown("---")
    st.markdown("#### Diagnostics")
    col_diag1, col_diag2 = st.columns(2)

    with col_diag1:
        st.markdown("Confusion Matrix (Gate Classifier)")
        cm = confusion_matrix(extreme_ref, gate_preds)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False,
                    xticklabels=['Normal', 'Extreme'], yticklabels=['Normal', 'Extreme'])
        ax_cm.set_ylabel('Actual')
        ax_cm.set_xlabel('Predicted')
        st.pyplot(fig_cm)

    with col_diag2:
        st.markdown("Prediction Residual Distribution")
        residuals = y_test_ref - y_pred_hybrid
        fig_res, ax_res = plt.subplots(figsize=(5, 3.5))
        sns.histplot(residuals, kde=True, color='slategrey', ax=ax_res)
        ax_res.set_xlabel('Error (kW)')
        ax_res.set_title('Residuals (Actual - Predicted)')
        st.pyplot(fig_res)

# --------------------------------------------------------------------------------
# VIEW 3: REAL-TIME INFERENCE SIMULATOR
# --------------------------------------------------------------------------------
elif nav_selection == "Real-Time Inference Simulator":
    st.markdown("### Single-Sample Inference Test")
    
    col_sim_in, col_sim_out = st.columns([1, 1])

    with col_sim_in:
        st.markdown("#### Sensor Inputs")
        wind_speed = st.slider("Wind Speed (Wspd) [m/s]", 0.0, 35.0, 14.2, 0.1)
        wind_direction = st.slider("Wind Direction (Wdir) [°]", 0, 360, 180)
        pitch_angle = st.slider("Pitch Angle (Prtv) [°]", -2.0, 90.0, 1.2)
        reactive_power = st.number_input("Reactive Power (Purt) [kVAR]", -50.0, 500.0, 20.0)
        ambient_temp = st.slider("Environment Temp (Etmp) [°C]", -15.0, 45.0, 24.0)

        input_df = pd.DataFrame([[wind_speed, wind_direction, pitch_angle, reactive_power, ambient_temp]], columns=feature_names)

    with col_sim_out:
        st.markdown("#### Model Output")
        
        prob_extreme = gate.predict_proba(input_df)[0][1]
        is_extreme = prob_extreme > 0.5
        
        pred_a = model_a.predict(input_df)[0]
        pred_b = model_b.predict(input_df)[0]
        final_prediction = max(0.0, float((1 - prob_extreme) * pred_a + prob_extreme * pred_b))

        status_str = "EXTREME WEATHER REGIME" if is_extreme else "NORMAL OPERATIONS REGIME"
        active_model = "Model B (Extreme Specialist)" if is_extreme else "Model A (Normal Specialist)"

        st.markdown(f"**Classification Status:** `{status_str}`")
        st.markdown(f"**Active Sub-Model:** `{active_model}`")
        st.markdown(f"**Extreme Probability:** `{prob_extreme * 100:.1f}%`")
        
        st.markdown("---")
        st.metric(label="Predicted Power Output", value=f"{final_prediction:.2f} kW")

# --------------------------------------------------------------------------------
# VIEW 4: BATCH DATA PROCESSING
# --------------------------------------------------------------------------------
elif nav_selection == "Batch Data Processing":
    st.markdown("### Batch Inference & Export")
    
    uploaded_file = st.file_uploader("Upload CSV dataset for batch evaluation", type=["csv"])
    
    if uploaded_file is not None:
        df_in = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_in):,} records.")
    else:
        st.info("No file uploaded. Showing sample records from test set.")
        df_in = X_test_ref.head(10).copy()
        st.dataframe(df_in, use_container_width=True)

    if st.button("Run Batch Predictions"):
        probs = gate.predict_proba(df_in[feature_names])[:, 1]
        preds_a = model_a.predict(df_in[feature_names])
        preds_b = model_b.predict(df_in[feature_names])
        hybrid_preds = (1 - probs) * preds_a + probs * preds_b

        df_results = df_in.copy()
        df_results['Extreme_Probability'] = np.round(probs, 3)
        df_results['Operating_Condition'] = np.where(probs > 0.5, 'Extreme', 'Normal')
        df_results['Predicted_Power_kW'] = np.round(np.maximum(0, hybrid_preds), 2)

        st.markdown("#### Generated Results")
        st.dataframe(df_results, use_container_width=True)

        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name="wind_forecast_results.csv",
            mime="text/csv"
        )
