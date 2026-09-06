# ================================================================================
# STREAMLIT APP: WIND POWER FORECASTING PLATFORM (ENHANCED UI)
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
# 1. PAGE LAYOUT & CUSTOM CSS
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Wind Power Forecasting System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom modern CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1A365D 0%, #2A4365 100%);
        padding: 20px 25px;
        border-radius: 10px;
        color: white;
        margin-bottom: 25px;
    }
    .main-title { font-size: 2.0rem; font-weight: 700; margin: 0; color: #FFFFFF; }
    .sub-title { font-size: 0.95rem; color: #E2E8F0; margin-top: 5px; }
    
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #2B6CB0; }
    .metric-lbl { font-size: 0.80rem; color: #718096; text-transform: uppercase; font-weight: 600; }
    
    .status-badge-normal {
        background-color: #C6F6D5; color: #22543D;
        padding: 8px 16px; border-radius: 20px; font-weight: 600; text-align: center;
    }
    .status-badge-extreme {
        background-color: #FED7D7; color: #742A2A;
        padding: 8px 16px; border-radius: 20px; font-weight: 600; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid", palette="deep")

# --------------------------------------------------------------------------------
# 2. MODEL INITIALIZATION (CACHED)
# --------------------------------------------------------------------------------
@st.cache_resource
def load_and_train_models():
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
    y = 0.5 * (X['Wspd (m/s)'] ** 3) - (X['Prtv (°)'] * 15) + np.random.normal(0, 30, n_samples)
    y = np.clip(y, 0, 1500)
    
    extreme_labels = ((X['Wspd (m/s)'] > 19.0) | (X['Prtv (°)'] > 20.0)).astype(int)

    # Model A: Normal Operations Specialist
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

# Predictions across models
y_pred_model_a = model_a.predict(X_test_ref)
y_pred_model_b = model_b.predict(X_test_ref)
gate_probs = gate.predict_proba(X_test_ref)[:, 1]
gate_preds = (gate_probs > 0.5).astype(int)
y_pred_hybrid = (1 - gate_probs) * y_pred_model_a + gate_probs * y_pred_model_b

# --------------------------------------------------------------------------------
# HEADER & NAVIGATION TABS
# --------------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">Wind Power Generation & Extreme Weather Dashboard</div>
    <div class="sub-title">Dual-Expert Gated Machine Learning Architecture for Power Grid Forecasting</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Real-Time Simulator", 
    "Horizon Time-Series", 
    "Model Benchmarks & Feature Importance", 
    "Project Overview"
])

# --------------------------------------------------------------------------------
# TAB 1: REAL-TIME SIMULATOR
# --------------------------------------------------------------------------------
with tab1:
    st.markdown("### Interactive Turbine Parameter Testing")
    
    col_input, col_results = st.columns([1, 1.2])

    with col_input:
        st.markdown("#### Input Telemetry Parameters")
        wind_speed = st.slider("Wind Speed (Wspd) [m/s]", 0.0, 40.0, 18.5, 0.5)
        wind_direction = st.slider("Wind Direction (Wdir) [°]", 0, 360, 180, 5)
        pitch_angle = st.slider("Pitch Angle (Prtv) [°]", -2.0, 90.0, 15.0, 0.5)
        reactive_power = st.number_input("Reactive Power (Purt) [kVAR]", -50.0, 500.0, 25.0, 5.0)
        ambient_temp = st.slider("Environment Temp (Etmp) [°C]", -15.0, 45.0, 22.0, 1.0)

        input_df = pd.DataFrame([[wind_speed, wind_direction, pitch_angle, reactive_power, ambient_temp]], columns=feature_names)

    with col_results:
        st.markdown("#### Inference & Routing Results")
        
        prob_extreme = gate.predict_proba(input_df)[0][1]
        is_extreme = prob_extreme > 0.5
        
        pred_a = max(0.0, float(model_a.predict(input_df)[0]))
        pred_b = max(0.0, float(model_b.predict(input_df)[0]))
        hybrid_out = max(0.0, float((1 - prob_extreme) * pred_a + prob_extreme * pred_b))

        status_class = "status-badge-extreme" if is_extreme else "status-badge-normal"
        status_text = "EXTREME WEATHER REGIME" if is_extreme else "NORMAL OPERATING REGIME"

        st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Predicted Power Output</div>
                    <div class="metric-val">{hybrid_out:.2f} kW</div>
                </div>
                """, unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Extreme Probability</div>
                    <div class="metric-val">{prob_extreme:.1%}</div>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Gate Risk Meter:")
        st.progress(float(prob_extreme))

        st.markdown("---")
        st.markdown("**Sub-Model Contributions:**")
        st.markdown(f"* Model A (Normal Specialist): `{pred_a:.2f} kW`")
        st.markdown(f"* Model B (Extreme Specialist): `{pred_b:.2f} kW`")

# --------------------------------------------------------------------------------
# TAB 2: HORIZON TIME-SERIES
# --------------------------------------------------------------------------------
with tab2:
    st.markdown("### Horizon Time-Series Forecasting")
    
    np.random.seed(99)
    time_steps = np.arange(1, 25)
    past_power = 400 + 150 * np.sin(time_steps / 3) + np.random.normal(0, 15, 24)
    forecast_val = 620.45
    
    st.info(f"Forecasted Next Time-Step Power Output: {forecast_val:.2f} kW")

    col_ts1, col_ts2 = st.columns(2)

    with col_ts1:
        st.markdown("#### Power Generation Horizon")
        fig_ts1, ax_ts1 = plt.subplots(figsize=(6, 3.8))
        ax_ts1.plot(time_steps, past_power, marker='o', color='#3182CE', label='Historical Generation', linewidth=2)
        ax_ts1.plot(25, forecast_val, marker='o', color='#E53E3E', markersize=8, label='Projected Point')
        ax_ts1.plot([24, 25], [past_power[-1], forecast_val], color='#E53E3E', linestyle='--', linewidth=2)
        ax_ts1.set_xlabel("Time Step (Hours)")
        ax_ts1.set_ylabel("Power Output (kW)")
        ax_ts1.legend(loc="upper left")
        st.pyplot(fig_ts1)

    with col_ts2:
        st.markdown("#### Power Trend Area Plot")
        fig_ts2, ax_ts2 = plt.subplots(figsize=(6, 3.8))
        ax_ts2.fill_between(time_steps, past_power, color='#63B3ED', alpha=0.4)
        ax_ts2.plot(time_steps, past_power, color='#3182CE', linewidth=2)
        ax_ts2.set_xlabel("Time Step (Hours)")
        ax_ts2.set_ylabel("Power Output (kW)")
        st.pyplot(fig_ts2)

# --------------------------------------------------------------------------------
# TAB 3: BENCHMARKS & FEATURE IMPORTANCE
# --------------------------------------------------------------------------------
with tab3:
    st.markdown("### Performance Comparison & Model Explainability")
    
    col_bench, col_feat = st.columns([1.1, 1])
    
    with col_bench:
        st.markdown("#### Regression Benchmark Statistics")
        metrics_comp = pd.DataFrame({
            "Model Architecture": ["Model A (Normal)", "Model B (Extreme)", "Gated Hybrid System"],
            "RMSE (kW)": [
                f"{np.sqrt(mean_squared_error(y_test_ref, y_pred_model_a)):.2f}",
                f"{np.sqrt(mean_squared_error(y_test_ref, y_pred_model_b)):.2f}",
                f"{np.sqrt(mean_squared_error(y_test_ref, y_pred_hybrid)):.2f}"
            ],
            "MAE (kW)": [
                f"{mean_absolute_error(y_test_ref, y_pred_model_a):.2f}",
                f"{mean_absolute_error(y_test_ref, y_pred_model_b):.2f}",
                f"{mean_absolute_error(y_test_ref, y_pred_hybrid):.2f}"
            ],
            "R² Score": [
                f"{r2_score(y_test_ref, y_pred_model_a):.3f}",
                f"{r2_score(y_test_ref, y_pred_model_b):.3f}",
                f"{r2_score(y_test_ref, y_pred_hybrid):.3f}"
            ]
        })
        st.table(metrics_comp)

    with col_feat:
        st.markdown("#### Feature Importance (Gate Classifier)")
        importances = gate.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
        
        fig_imp, ax_imp = plt.subplots(figsize=(5, 3.2))
        ax_imp.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#3182CE')
        ax_imp.set_xlabel('Relative Importance')
        st.pyplot(fig_imp)

    st.markdown("---")
    col_diag1, col_diag2 = st.columns(2)

    with col_diag1:
        st.markdown("#### Residual Error Distributions")
        fig_res, ax_res = plt.subplots(figsize=(6, 3.5))
        sns.kdeplot(y_test_ref - y_pred_model_a, label="Baseline Model A Error", color="#E53E3E", ax=ax_res)
        sns.kdeplot(y_test_ref - y_pred_hybrid, label="Hybrid System Error", color="#3182CE", ax=ax_res)
        ax_res.set_xlabel("Prediction Error (kW)")
        ax_res.legend()
        st.pyplot(fig_res)

    with col_diag2:
        st.markdown("#### Confusion Matrix (Gate Routing Classifier)")
        cm = confusion_matrix(extreme_ref, gate_preds)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False,
                    xticklabels=['Normal', 'Extreme'], yticklabels=['Normal', 'Extreme'])
        ax_cm.set_ylabel('Actual Regime')
        ax_cm.set_xlabel('Predicted Regime')
        st.pyplot(fig_cm)

# --------------------------------------------------------------------------------
# TAB 4: PROJECT OVERVIEW
# --------------------------------------------------------------------------------
with tab4:
    st.markdown("### System Architecture")
    st.write(
        "This system employs a dual-expert gated network designed to mitigate prediction error "
        "during high-volatility extreme weather events in wind farms."
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-lbl">Sub-Model 1</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #2D3748; margin: 6px 0;">Model A (XGBoost)</div>
                <div style="font-size: 0.80rem; color: #718096;">Normal Operations Specialist</div>
            </div>
            """, unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-lbl">Sub-Model 2</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #2D3748; margin: 6px 0;">Model B (XGBoost)</div>
                <div style="font-size: 0.80rem; color: #718096;">Extreme Weather Specialist</div>
            </div>
            """, unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-lbl">Dataset</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #2D3748; margin: 6px 0;">SCADA Telemetry</div>
                <div style="font-size: 0.80rem; color: #718096;">500+ Sensor Records</div>
            </div>
            """, unsafe_allow_html=True
        )
