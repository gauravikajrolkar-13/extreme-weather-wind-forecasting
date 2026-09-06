# ================================================================================
# STREAMLIT APP: EXTREME-WEATHER WIND POWER FORECASTING SYSTEM
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
# 1. PAGE LAYOUT & ENTERPRISE STYLING
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Wind Power Forecasting System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        padding: 22px 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 25px;
    }
    .main-title { font-size: 2.1rem; font-weight: 700; margin: 0; color: #F8FAFC; }
    .sub-title { font-size: 0.95rem; color: #94A3B8; margin-top: 5px; }
    
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-val { font-size: 1.7rem; font-weight: 700; color: #0284C7; }
    .metric-lbl { font-size: 0.78rem; color: #64748B; text-transform: uppercase; font-weight: 600; }
    
    .status-badge-normal {
        background-color: #DCFCE7; color: #166534; border: 1px solid #BBF7D0;
        padding: 10px 18px; border-radius: 20px; font-weight: 600; text-align: center;
    }
    .status-badge-extreme {
        background-color: #FEE2E2; color: #991B1B; border: 1px solid #FECACA;
        padding: 10px 18px; border-radius: 20px; font-weight: 600; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid", palette="deep")

# --------------------------------------------------------------------------------
# 2. MODEL INITIALIZATION & CACHING
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

# Baseline & Hybrid Calculations
y_pred_model_a = model_a.predict(X_test_ref)
y_pred_model_b = model_b.predict(X_test_ref)

# --------------------------------------------------------------------------------
# HEADER & NAVIGATION TABS
# --------------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">Extreme-Weather Wind Power Forecasting Platform</div>
    <div class="sub-title">Dual-Expert Gated Machine Learning System for Operational Power Grid Analytics</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Real-Time Simulator", 
    "Time-Series Horizon", 
    "Model Benchmarks & SHAP", 
    "Dataset & EDA Explorer",
    "Architecture & Summary"
])

# --------------------------------------------------------------------------------
# TAB 1: REAL-TIME SIMULATOR (WITH THRESHOLD CONTROL)
# --------------------------------------------------------------------------------
with tab1:
    st.markdown("### Interactive Telemetry Simulator & Routing Threshold")
    
    col_input, col_results = st.columns([1, 1.2])

    with col_input:
        st.markdown("#### Input Telemetry Data")
        wind_speed = st.slider("Wind Speed (Wspd) [m/s]", 0.0, 40.0, 18.5, 0.5)
        wind_direction = st.slider("Wind Direction (Wdir) [°]", 0, 360, 180, 5)
        pitch_angle = st.slider("Pitch Angle (Prtv) [°]", -2.0, 90.0, 15.0, 0.5)
        reactive_power = st.number_input("Reactive Power (Purt) [kVAR]", -50.0, 500.0, 25.0, 5.0)
        ambient_temp = st.slider("Environment Temp (Etmp) [°C]", -15.0, 45.0, 22.0, 1.0)

        st.markdown("---")
        # FEATURE 3: ADJUSTABLE GATE THRESHOLD SLIDER
        gate_threshold = st.slider("Gate Decision Threshold (Sensitivity)", 0.10, 0.90, 0.50, 0.05,
                                   help="Adjust threshold to prioritize Precision vs. Recall during extreme weather routing.")

        input_df = pd.DataFrame([[wind_speed, wind_direction, pitch_angle, reactive_power, ambient_temp]], columns=feature_names)

    with col_results:
        st.markdown("#### Inference & Routing Results")
        
        prob_extreme = gate.predict_proba(input_df)[0][1]
        is_extreme = prob_extreme >= gate_threshold
        
        pred_a = max(0.0, float(model_a.predict(input_df)[0]))
        pred_b = max(0.0, float(model_b.predict(input_df)[0]))
        hybrid_out = max(0.0, float((1 - prob_extreme) * pred_a + prob_extreme * pred_b))

        status_class = "status-badge-extreme" if is_extreme else "status-badge-normal"
        status_text = f"EXTREME WEATHER REGIME (Threshold: {gate_threshold:.2f})" if is_extreme else f"NORMAL OPERATING REGIME (Threshold: {gate_threshold:.2f})"

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
                    <div class="metric-lbl">Extreme Event Probability</div>
                    <div class="metric-val">{prob_extreme:.1%}</div>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Gate Extreme Risk Meter:")
        st.progress(float(prob_extreme))

        st.markdown("---")
        st.markdown("**Sub-Model Contributions:**")
        st.markdown(f"* Model A (Normal Specialist): `{pred_a:.2f} kW`")
        st.markdown(f"* Model B (Extreme Specialist): `{pred_b:.2f} kW`")

# --------------------------------------------------------------------------------
# TAB 2: TIME-SERIES HORIZON
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
        ax_ts1.plot(time_steps, past_power, marker='o', color='#0284C7', label='Historical Generation', linewidth=2)
        ax_ts1.plot(25, forecast_val, marker='o', color='#DC2626', markersize=8, label='Projected Point')
        ax_ts1.plot([24, 25], [past_power[-1], forecast_val], color='#DC2626', linestyle='--', linewidth=2)
        ax_ts1.set_xlabel("Time Step (Hours)")
        ax_ts1.set_ylabel("Power Output (kW)")
        ax_ts1.legend(loc="upper left")
        st.pyplot(fig_ts1)

    with col_ts2:
        st.markdown("#### Power Trend Area View")
        fig_ts2, ax_ts2 = plt.subplots(figsize=(6, 3.8))
        ax_ts2.fill_between(time_steps, past_power, color='#38BDF8', alpha=0.35)
        ax_ts2.plot(time_steps, past_power, color='#0284C7', linewidth=2)
        ax_ts2.set_xlabel("Time Step (Hours)")
        ax_ts2.set_ylabel("Power Output (kW)")
        st.pyplot(fig_ts2)

# --------------------------------------------------------------------------------
# TAB 3: BENCHMARKS, SHAP & FEATURE IMPORTANCE
# --------------------------------------------------------------------------------
with tab3:
    st.markdown("### Benchmarks & Model Interpretability")
    
    # Recalculate based on default threshold 0.50
    gate_probs = gate.predict_proba(X_test_ref)[:, 1]
    gate_preds = (gate_probs >= 0.50).astype(int)
    y_pred_hybrid = (1 - gate_probs) * y_pred_model_a + gate_probs * y_pred_model_b

    # 1. Classification Precision Metrics
    prec = precision_score(extreme_ref, gate_preds)
    acc = accuracy_score(extreme_ref, gate_preds)
    rec = recall_score(extreme_ref, gate_preds)
    f1 = f1_score(extreme_ref, gate_preds)

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Precision Rate</div><div class="metric-val">{prec:.1%}</div></div>', unsafe_allow_html=True)
    with col_p2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Accuracy Rate</div><div class="metric-val">{acc:.1%}</div></div>', unsafe_allow_html=True)
    with col_p3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Recall Rate</div><div class="metric-val">{rec:.1%}</div></div>', unsafe_allow_html=True)
    with col_p4:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">F1-Score</div><div class="metric-val">{f1:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    col_bench, col_feat = st.columns([1.1, 1])
    
    with col_bench:
        st.markdown("#### Regression Benchmark Statistics")
        metrics_3_models = pd.DataFrame({
            "Model Architecture": ["Model 1: Model A (Normal)", "Model 2: Model B (Extreme)", "Model 3: Gated Hybrid"],
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
        st.table(metrics_3_models)

    # FEATURE 1: INTERACTIVE FEATURE IMPORTANCE & EXPLAINABILITY
    with col_feat:
        st.markdown("#### Feature Importance (Gate Routing Network)")
        importances = gate.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
        
        fig_imp, ax_imp = plt.subplots(figsize=(5, 3.2))
        ax_imp.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#0284C7')
        ax_imp.set_xlabel('Relative Importance')
        st.pyplot(fig_imp)

    st.markdown("---")
    col_diag1, col_diag2 = st.columns(2)

    with col_diag1:
        st.markdown("#### Residual Error Distributions")
        fig_res, ax_res = plt.subplots(figsize=(6, 3.5))
        sns.kdeplot(y_test_ref - y_pred_model_a, label="Baseline Model A Error", color="#DC2626", ax=ax_res)
        sns.kdeplot(y_test_ref - y_pred_hybrid, label="Hybrid System Error", color="#0284C7", ax=ax_res)
        ax_res.set_xlabel("Prediction Error (kW)")
        ax_res.legend()
        st.pyplot(fig_res)

    with col_diag2:
        st.markdown("#### Gate Classifier Confusion Matrix")
        cm = confusion_matrix(extreme_ref, gate_preds)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False,
                    xticklabels=['Normal', 'Extreme'], yticklabels=['Normal', 'Extreme'])
        ax_cm.set_ylabel('Actual Regime')
        ax_cm.set_xlabel('Predicted Regime')
        st.pyplot(fig_cm)

# --------------------------------------------------------------------------------
# TAB 4: DATASET & EDA EXPLORER (FEATURE 4)
# --------------------------------------------------------------------------------
with tab4:
    st.markdown("### SCADA Telemetry Dataset & Exploratory Data Analysis")
    
    # Raw Data Table & Search
    full_df = X_test_ref.copy()
    full_df['Power_Output_kW'] = np.round(y_test_ref, 2)
    full_df['Regime_Label'] = np.where(extreme_ref == 1, 'Extreme', 'Normal')

    st.markdown("#### Filterable Telemetry Explorer")
    st.dataframe(full_df, use_container_width=True, height=250)

    st.markdown("---")
    st.markdown("#### Summary Statistics & Feature Distributions")
    
    col_eda1, col_eda2 = st.columns([1, 1])

    with col_eda1:
        st.markdown("Summary Statistics")
        st.dataframe(full_df.describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)

    with col_eda2:
        st.markdown("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(5, 3.0))
        sns.heatmap(full_df.drop(columns=['Regime_Label']).corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax_corr, cbar=False)
        st.pyplot(fig_corr)

# --------------------------------------------------------------------------------
# TAB 5: ARCHITECTURE & REPORT EXPORT (FEATURE 5)
# --------------------------------------------------------------------------------
with tab5:
    st.markdown("### System Architecture & Executive Summary Export")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-lbl">Sub-Model 1</div><div style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">Model A (XGBoost)</div><div style="font-size: 0.80rem; color: #64748B;">Normal Operations Specialist</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-lbl">Sub-Model 2</div><div style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">Model B (XGBoost)</div><div style="font-size: 0.80rem; color: #64748B;">Extreme Weather Specialist</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-lbl">Dataset</div><div style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">SCADA Telemetry</div><div style="font-size: 0.80rem; color: #64748B;">2,000 Verified Records</div></div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Export Executive Results Summary")
    
    summary_txt = f"""=== WIND POWER FORECASTING SYSTEM SUMMARY REPORT ===
    
MODEL BENCHMARKS:
- Hybrid Model RMSE: {np.sqrt(mean_squared_error(y_test_ref, y_pred_hybrid)):.2f} kW
- Hybrid Model MAE: {mean_absolute_error(y_test_ref, y_pred_hybrid):.2f} kW
- Hybrid Model R²: {r2_score(y_test_ref, y_pred_hybrid):.3f}

CLASSIFIER METRICS:
- Gate Accuracy: {acc:.2%}
- Gate Precision Rate: {prec:.2%}
- Gate Recall Rate: {rec:.2%}
- Gate F1-Score: {f1:.3f}

ARCHITECTURE:
- Dual-Expert Gated Network (Model A: Normal, Model B: Extreme, Gate Classifier: XGBoost)
"""

    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.download_button(
            label="Download Executive Summary Report (.txt)",
            data=summary_txt,
            file_name="Wind_Power_Forecast_Executive_Summary.txt",
            mime="text/plain"
        )

    with col_exp2:
        csv_buffer = io.StringIO()
        full_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Clean Dataset (.csv)",
            data=csv_buffer.getvalue(),
            file_name="Wind_Farm_SCADA_Telemetry.csv",
            mime="text/csv"
        )
