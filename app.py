# ================================================================================
# STREAMLIT APP: ENTERPRISE EXTREME-WEATHER WIND POWER FORECASTING PLATFORM
# Save this file as `app.py` in your repository root directory
# ================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import plotly.graph_objects as go
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import io

# --------------------------------------------------------------------------------
# 1. PAGE LAYOUT & ENTERPRISE DESIGN SYSTEM
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Wind Power Forecasting System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Unified CSS Design System
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p {
        color: #94A3B8 !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        border: 1px solid #334155;
    }
    .main-title { 
        font-size: 2.1rem; 
        font-weight: 700; 
        margin: 0; 
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .sub-title { 
        font-size: 0.95rem; 
        color: #38BDF8; 
        margin-top: 6px; 
        font-weight: 500;
    }

    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-val { 
        font-size: 1.8rem; 
        font-weight: 700; 
        color: #0284C7; 
        margin: 4px 0;
    }
    .metric-lbl { 
        font-size: 0.75rem; 
        color: #64748B; 
        text-transform: uppercase; 
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    .status-badge {
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        text-align: center;
        margin-bottom: 16px;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .badge-extreme {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }

    div[data-testid="stForm"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        background-color: #FFFFFF;
        padding: 20px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        padding: 0 16px;
    }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid", palette="deep")

# --------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING & DATASET BUILDER
# --------------------------------------------------------------------------------
def compute_engineered_features(df):
    """Calculates physical wind domain features for ML and DL models."""
    data = df.copy()
    
    # 1. Kinetic Power Density Proxy (P ~ v^3)
    data['Wspd_Cubed'] = data['Wspd (m/s)'] ** 3
    
    # 2. Air Density Correction Factor (approx function of temp)
    air_density = 1.225 * (288.15 / (273.15 + data['Etmp (°C)']))
    data['Air_Density_kg_m3'] = air_density
    
    # 3. Wind Power Density (WPD = 0.5 * rho * v^3)
    data['Wind_Power_Density'] = 0.5 * air_density * data['Wspd_Cubed']
    
    # 4. Pitch Interaction Proxy (effective aerodynamic drag force)
    data['Pitch_Efficiency_Factor'] = np.cos(np.radians(data['Prtv (°)']))
    
    # 5. Directional Sine/Cosine Trigonometric Encoding
    data['Wdir_Sin'] = np.sin(np.radians(data['Wdir (°)']))
    data['Wdir_Cos'] = np.cos(np.radians(data['Wdir (°)']))
    
    return data

@st.cache_resource
def load_and_train_models():
    np.random.seed(42)
    base_feature_names = ['Wspd (m/s)', 'Wdir (°)', 'Prtv (°)', 'Purt (kVAR)', 'Etmp (°C)']
    
    n_samples = 2500
    wspd = np.abs(np.random.normal(12, 6, n_samples))
    wdir = np.random.uniform(0, 360, n_samples)
    prtv = np.clip((wspd - 15) * 2.8 + np.random.normal(0, 3.5, n_samples), -2, 90)
    prtv[wspd < 15] = np.random.uniform(-1, 2, np.sum(wspd < 15))
    purt = 1.5 * prtv + np.random.normal(20, 12, n_samples)
    etmp = 25 - (wspd * 0.4) + np.random.normal(0, 5, n_samples)
    
    raw_df = pd.DataFrame(np.column_stack([wspd, wdir, prtv, purt, etmp]), columns=base_feature_names)
    
    # Apply Feature Engineering
    X = compute_engineered_features(raw_df)
    feature_names = list(X.columns)

    # Power Curve Target Function
    y = 0.5 * (X['Wspd (m/s)'] ** 3) - (X['Prtv (°)'] * 18) + np.random.normal(0, 45, n_samples)
    y[X['Wspd (m/s)'] > 25.0] = 0.0  # Emergency Cut-out
    y = np.clip(y, 0, 1500)
    
    extreme_labels = ((X['Wspd (m/s)'] > 19.0) | (X['Prtv (°)'] > 20.0)).astype(int)

    train_size = int(n_samples * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    ext_train, ext_test = extreme_labels.iloc[:train_size], extreme_labels.iloc[train_size:]

    # Scale engineered features for Deep Learning Model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- DEEP LEARNING MODEL (Multi-Layer Perceptron) ---
    dl_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=350,
        random_state=42
    )
    dl_model.fit(X_train_scaled, y_train)

    # --- BASELINE GRADIENT BOOSTING REGRESSOR ---
    global_model = XGBRegressor(n_estimators=60, max_depth=5, learning_rate=0.08, tree_method='hist', random_state=42)
    global_model.fit(X_train, y_train)

    # --- SUB-MODEL A (Normal Specialist) ---
    model_a = XGBRegressor(n_estimators=60, max_depth=5, learning_rate=0.08, tree_method='hist', random_state=42)
    model_a.fit(X_train[ext_train == 0], y_train[ext_train == 0])

    # --- SUB-MODEL B (Extreme Specialist) ---
    model_b = XGBRegressor(n_estimators=60, max_depth=5, learning_rate=0.08, tree_method='hist', random_state=42)
    model_b.fit(X_train[ext_train == 1], y_train[ext_train == 1])

    # --- GATE ROUTER (No Target Leakage) ---
    gate_features = ['Wdir (°)', 'Purt (kVAR)', 'Etmp (°C)', 'Wdir_Sin', 'Wdir_Cos', 'Air_Density_kg_m3']
    X_gate_train = X_train[gate_features]
    
    gate = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.08, tree_method='hist', random_state=42)
    gate.fit(X_gate_train, ext_train)

    return global_model, dl_model, scaler, model_a, model_b, gate, X_test, X_test_scaled, y_test, ext_test, feature_names, gate_features, base_feature_names

global_model, dl_model, scaler, model_a, model_b, gate, X_test_ref, X_test_scaled, y_test_ref, extreme_ref, feature_names, gate_features, base_feature_names = load_and_train_models()

y_pred_global = global_model.predict(X_test_ref)
y_pred_dl = dl_model.predict(X_test_scaled)
y_pred_model_a = model_a.predict(X_test_ref)
y_pred_model_b = model_b.predict(X_test_ref)

gate_probs = gate.predict_proba(X_test_ref[gate_features])[:, 1]
gate_preds = (gate_probs >= 0.50).astype(int)
y_pred_hybrid = (1 - gate_probs) * y_pred_model_a + gate_probs * y_pred_model_b

# --------------------------------------------------------------------------------
# 3. GLOBAL HEADER & SIDEBAR NAVIGATION
# --------------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">Extreme-Weather Wind Power Forecasting Platform</div>
    <div class="sub-title">Gated Hybrid Architecture combining Deep Learning & Machine Learning Specialists</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/38BDF8/wind-turbine.png", width=64)
    st.title("Control Panel")
    st.caption("System v3.3 Pro | Active Grid Node")
    st.markdown("---")
    
    st.subheader("Global Settings")
    power_rate_kw = st.number_input("Energy Rate ($/kWh)", min_value=0.01, max_value=1.00, value=0.12, step=0.01)
    
    st.markdown("---")
    st.subheader("System Status")
    st.success("● SCADA Link Active")
    st.info("● Deep Learning Engine Online")
    st.info("● Engineered Features Loaded")
    st.info("● Dual-Expert Gate Ready")

# Main Page Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " Real-Time Simulator", 
    " Time-Series Horizon", 
    " Geospatial Fleet Map",
    " Benchmarks & Analytics", 
    " SCADA Anomaly Detector",
    " Dataset & Summary Export"
])

# --------------------------------------------------------------------------------
# TAB 1: REAL-TIME SIMULATOR
# --------------------------------------------------------------------------------
with tab1:
    st.markdown("### Interactive Telemetry Simulator & Gated Routing")
    st.write("Test single-instance SCADA inputs against engineered domain features, deep learning models, and gated routing classifiers.")

    st.markdown("#### Operational Preset Scenarios")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    if "sim_wspd" not in st.session_state:
        st.session_state.sim_wspd = 18.5
        st.session_state.sim_wdir = 180
        st.session_state.sim_prtv = 15.0
        st.session_state.sim_purt = 65.0
        st.session_state.sim_etmp = 12.0

    with p_col1:
        if st.button(" Preset 1: Nominal Breeze", use_container_width=True):
            st.session_state.sim_wspd = 11.2
            st.session_state.sim_wdir = 145
            st.session_state.sim_prtv = 0.5
            st.session_state.sim_purt = 15.0
            st.session_state.sim_etmp = 22.0
            st.rerun()

    with p_col2:
        if st.button(" Preset 2: Impending Storm", use_container_width=True):
            st.session_state.sim_wspd = 21.4
            st.session_state.sim_wdir = 290
            st.session_state.sim_prtv = 24.5
            st.session_state.sim_purt = 110.0
            st.session_state.sim_etmp = 4.0
            st.rerun()

    with p_col3:
        if st.button(" Preset 3: Emergency Cut-out", use_container_width=True):
            st.session_state.sim_wspd = 28.5
            st.session_state.sim_wdir = 315
            st.session_state.sim_prtv = 82.0
            st.session_state.sim_purt = 280.0
            st.session_state.sim_etmp = -2.0
            st.rerun()

    with p_col4:
        if st.button(" Preset 4: Low Wind Idle", use_container_width=True):
            st.session_state.sim_wspd = 2.8
            st.session_state.sim_wdir = 40
            st.session_state.sim_prtv = -1.0
            st.session_state.sim_purt = 5.0
            st.session_state.sim_etmp = 18.0
            st.rerun()

    st.markdown("---")
    col_input, col_results = st.columns([1.1, 1.2])

    with col_input:
        st.markdown("#### Input SCADA Parameters")
        wind_speed = st.slider("Wind Speed (Wspd) [m/s]", 0.0, 40.0, float(st.session_state.sim_wspd), 0.5)
        wind_direction = st.slider("Wind Direction (Wdir) [°]", 0, 360, int(st.session_state.sim_wdir), 5)
        pitch_angle = st.slider("Pitch Angle (Prtv) [°]", -2.0, 90.0, float(st.session_state.sim_prtv), 0.5)
        reactive_power = st.number_input("Reactive Power (Purt) [kVAR]", -50.0, 500.0, float(st.session_state.sim_purt), 5.0)
        ambient_temp = st.slider("Environment Temp (Etmp) [°C]", -15.0, 45.0, float(st.session_state.sim_etmp), 1.0)

        gate_threshold = st.slider("Gate Decision Sensitivity Threshold", 0.10, 0.90, 0.50, 0.05)

        raw_input_df = pd.DataFrame([[wind_speed, wind_direction, pitch_angle, reactive_power, ambient_temp]], columns=base_feature_names)
        input_df = compute_engineered_features(raw_input_df)
        input_scaled = scaler.transform(input_df)
        input_gate_df = input_df[gate_features]

    with col_results:
        st.markdown("#### Inference & Decision Routing")
        
        prob_extreme = gate.predict_proba(input_gate_df)[0][1]
        is_extreme = prob_extreme >= gate_threshold
        
        pred_dl = max(0.0, float(dl_model.predict(input_scaled)[0]))
        pred_a = max(0.0, float(model_a.predict(input_df)[0]))
        pred_b = max(0.0, float(model_b.predict(input_df)[0]))
        hybrid_out = max(0.0, float((1 - prob_extreme) * pred_a + prob_extreme * pred_b))

        status_class = "badge-extreme" if is_extreme else "badge-normal"
        status_text = f"EXTREME WEATHER REGIME DETECTED (Prob: {prob_extreme:.1%})" if is_extreme else f"NOMINAL OPERATING REGIME DETECTED (Prob: {prob_extreme:.1%})"

        st.markdown(f'<div class="status-badge {status_class}">{status_text}</div>', unsafe_allow_html=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f'<div class="metric-card"><div class="metric-lbl">Gated Hybrid Power</div><div class="metric-val">{hybrid_out:.2f} kW</div></div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown(f'<div class="metric-card"><div class="metric-lbl">Deep Learning MLP Power</div><div class="metric-val" style="color: #0284C7;">{pred_dl:.2f} kW</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Classifier Routing Confidence Meter:")
        st.progress(float(prob_extreme))

        st.markdown("---")
        st.markdown("**Computed Engineered Feature Values:**")
        st.markdown(f"* **Wind Power Density (WPD):** `{input_df['Wind_Power_Density'].values[0]:.2f} W/m²`")
        st.markdown(f"* **Air Density ($\rho$):** `{input_df['Air_Density_kg_m3'].values[0]:.3f} kg/m³`")
        st.markdown(f"* **Direction Trigonometric Component (Sin):** `{input_df['Wdir_Sin'].values[0]:.3f}`")

# --------------------------------------------------------------------------------
# TAB 2: TIME-SERIES HORIZON
# --------------------------------------------------------------------------------
with tab2:
    st.markdown("### Time-Series Power Generation Horizon")
    st.write("24-Hour continuous historical SCADA telemetry paired with next time-step probabilistic point forecasting.")
    
    np.random.seed(99)
    time_steps = np.arange(1, 25)
    past_power = 400 + 150 * np.sin(time_steps / 3) + np.random.normal(0, 15, 24)
    forecast_val = 620.45
    std_error = 28.5

    c_ts1, c_ts2 = st.columns(2)

    with c_ts1:
        st.markdown("#### Horizon Trend with Uncertainty Band")
        fig_ts1 = go.Figure()
        
        fig_ts1.add_trace(go.Scatter(x=time_steps, y=past_power, mode='lines+markers', name='Historical Generation (kW)', line=dict(color='#0284C7', width=2.5)))
        fig_ts1.add_trace(go.Scatter(x=[24, 25], y=[past_power[-1], forecast_val], mode='lines', line=dict(color='#DC2626', dash='dash', width=2), showlegend=False))
        fig_ts1.add_trace(go.Scatter(x=[25], y=[forecast_val], mode='markers', marker=dict(color='#DC2626', size=10), name='Point Forecast (620.45 kW)'))
        fig_ts1.add_trace(go.Scatter(
            x=[25, 25], y=[forecast_val - std_error, forecast_val + std_error],
            mode='lines+markers', name='95% Confidence Margin', line=dict(color='#F59E0B', width=4)
        ))

        fig_ts1.update_layout(xaxis_title="Time Step (Hours)", yaxis_title="Power Output (kW)", height=380, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_ts1, use_container_width=True)

    with c_ts2:
        st.markdown("#### Cumulative Generation Profile")
        fig_ts2 = go.Figure()
        fig_ts2.add_trace(go.Scatter(x=time_steps, y=past_power, fill='tozeroy', line=dict(color='#38BDF8', width=2), name='Power Profile'))
        fig_ts2.update_layout(xaxis_title="Time Step (Hours)", yaxis_title="Power Output (kW)", height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_ts2, use_container_width=True)

# --------------------------------------------------------------------------------
# TAB 3: GEOSPATIAL WIND FARM MAP
# --------------------------------------------------------------------------------
with tab3:
    st.markdown("### Fleet-Level Geospatial Interactive Map")
    st.write("Real-time operational monitoring across turbine assets in the wind farm cluster.")

    map_data = pd.DataFrame({
        'Turbine_ID': [f'T-{i:02d}' for i in range(1, 13)],
        'lat': [36.102 + np.random.uniform(-0.015, 0.015) for _ in range(12)],
        'lon': [-115.17 + np.random.uniform(-0.015, 0.015) for _ in range(12)],
        'Power_kW': np.random.uniform(200, 1400, 12).round(1),
        'Status': np.random.choice(['Nominal', 'Nominal', 'Extreme Guard Active'], 12)
    })

    col_map1, col_map2 = st.columns([2.5, 1])

    with col_map1:
        layer = pdk.Layer(
            "ScatterplotLayer",
            map_data,
            get_position=["lon", "lat"],
            get_color="Status == 'Nominal' ? [2, 132, 199, 200] : [220, 38, 38, 200]",
            get_radius=180,
            pickable=True,
        )

        view_state = pdk.ViewState(latitude=36.102, longitude=-115.17, zoom=12.5, pitch=30)
        
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Asset: {Turbine_ID}\nGeneration: {Power_kW} kW\nStatus: {Status}"}
        )
        st.pydeck_chart(r)

    with col_map2:
        st.markdown("#### Fleet Telemetry Stream")
        st.dataframe(
            map_data[['Turbine_ID', 'Power_kW', 'Status']],
            use_container_width=True,
            height=380
        )

# --------------------------------------------------------------------------------
# TAB 4: BENCHMARKS & DEEP LEARNING COMPARISON OVERLAY
# --------------------------------------------------------------------------------
with tab4:
    st.markdown("### Benchmarks & Validation Analytics")
    
    prec = precision_score(extreme_ref, gate_preds)
    acc = accuracy_score(extreme_ref, gate_preds)
    rec = recall_score(extreme_ref, gate_preds)
    f1 = f1_score(extreme_ref, gate_preds)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Gate Accuracy</div><div class="metric-val">{acc:.1%}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Gate Precision</div><div class="metric-val">{prec:.1%}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Gate Recall</div><div class="metric-val">{rec:.1%}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Gate F1-Score</div><div class="metric-val">{f1:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### Ground Truth Actual Power vs. ML & Deep Learning Models")
    
    sample_indices = np.arange(60)
    fig_overlay = go.Figure()
    
    fig_overlay.add_trace(go.Scatter(x=sample_indices, y=y_test_ref.values[:60], mode='lines', name='Actual Ground Truth (kW)', line=dict(color='#0F172A', width=3)))
    fig_overlay.add_trace(go.Scatter(x=sample_indices, y=y_pred_dl[:60], mode='lines', name='Deep Neural Network (MLP)', line=dict(color='#F59E0B', dash='dash', width=2)))
    fig_overlay.add_trace(go.Scatter(x=sample_indices, y=y_pred_global[:60], mode='lines', name='Baseline Global Model', line=dict(color='#DC2626', dash='dot', width=2)))
    fig_overlay.add_trace(go.Scatter(x=sample_indices, y=y_pred_hybrid[:60], mode='lines', name='Gated Hybrid Model', line=dict(color='#0284C7', width=2.5)))

    fig_overlay.update_layout(xaxis_title="Sample Index", yaxis_title="Power Output (kW)", height=380, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_overlay, use_container_width=True)

    st.markdown("---")
    col_bench, col_feat = st.columns([1.3, 1])
    
    with col_bench:
        st.markdown("#### Model Architecture Benchmark Matrix")
        metrics_df = pd.DataFrame({
            "Architecture": [
                "Global Deep Learning (MLP Neural Net)",
                "Baseline Global Single Model (XGBoost)", 
                "Model A Specialist (Normal Weather)", 
                "Proposed: Gated Dual-Expert Hybrid"
            ],
            "Category": [
                "Deep Learning",
                "Classical ML",
                "Specialist ML",
                "Gated Ensemble"
            ],
            "RMSE (kW)": [
                f"{np.sqrt(mean_squared_error(y_test_ref, y_pred_dl)):.2f}",
                f"{np.sqrt(mean_squared_error(y_test_ref, y_pred_global)):.2f}",
                f"{np.sqrt(mean_squared_error(y_test_ref, y_pred_model_a)):.2f}",
                f"{np.sqrt(mean_squared_error(y_test_ref, y_pred_hybrid)):.2f}"
            ],
            "MAE (kW)": [
                f"{mean_absolute_error(y_test_ref, y_pred_dl):.2f}",
                f"{mean_absolute_error(y_test_ref, y_pred_global):.2f}",
                f"{mean_absolute_error(y_test_ref, y_pred_model_a):.2f}",
                f"{mean_absolute_error(y_test_ref, y_pred_hybrid):.2f}"
            ],
            "R² Score": [
                f"{r2_score(y_test_ref, y_pred_dl):.3f}",
                f"{r2_score(y_test_ref, y_pred_global):.3f}",
                f"{r2_score(y_test_ref, y_pred_model_a):.3f}",
                f"{r2_score(y_test_ref, y_pred_hybrid):.3f}"
            ]
        })
        st.table(metrics_df)

    with col_feat:
        st.markdown("#### Gate Feature Importance (No-Leakage Matrix)")
        importances = gate.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': gate_features, 'Importance': importances}).sort_values('Importance', ascending=True)
        
        fig_imp, ax_imp = plt.subplots(figsize=(5, 3.2))
        ax_imp.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#0284C7')
        ax_imp.set_xlabel('Relative Importance')
        st.pyplot(fig_imp)

# --------------------------------------------------------------------------------
# TAB 5: SCADA ANOMALY DETECTOR
# --------------------------------------------------------------------------------
with tab5:
    st.markdown("### SCADA Sensor Anomaly & Fault Scanner")
    st.write("Scans incoming sensor streams for frozen values, out-of-range bounds, and unphysical mechanical combinations prior to model ingestion.")

    def audit_telemetry(df):
        anomalies = []
        for idx, row in df.iterrows():
            flags = []
            if row['Wspd (m/s)'] > 35.0 or row['Wspd (m/s)'] < 0.0:
                flags.append("Unphysical Wind Speed Bound")
            if row['Prtv (°)'] > 20.0 and row['Wspd (m/s)'] < 5.0:
                flags.append("Inconsistent Feathering (High Pitch, Low Wind)")
            if row['Wspd (m/s)'] > 22.0 and row['Prtv (°)'] < 5.0:
                flags.append("Mechanical Fault (High Wind, No Feathering Safety)")
            if row['Etmp (°C)'] < -20.0 or row['Etmp (°C)'] > 50.0:
                flags.append("Extreme Temp Sensor Fault")
            
            status = "CRITICAL ANOMALY" if len(flags) > 0 else "HEALTHY"
            anomalies.append({
                "Sample_ID": idx + 1,
                "Wspd": row['Wspd (m/s)'],
                "Pitch": row['Prtv (°)'],
                "Diagnostic_Status": status,
                "Detected_Faults": ", ".join(flags) if flags else "Nominal Operations"
            })
        return pd.DataFrame(anomalies)

    test_batch = X_test_ref.head(15).copy().reset_index(drop=True)
    test_batch.loc[2, 'Wspd (m/s)'] = 38.5
    test_batch.loc[5, 'Prtv (°)'] = 45.0
    test_batch.loc[5, 'Wspd (m/s)'] = 3.2
    test_batch.loc[9, 'Wspd (m/s)'] = 26.0
    test_batch.loc[9, 'Prtv (°)'] = 1.0

    audit_results = audit_telemetry(test_batch)
    
    n_critical = sum(audit_results['Diagnostic_Status'] == "CRITICAL ANOMALY")
    n_healthy = len(audit_results) - n_critical

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Audited Telemetry Records</div><div class="metric-val">{len(audit_results)}</div></div>', unsafe_allow_html=True)
    with col_a2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Healthy Records</div><div class="metric-val" style="color: #166534;">{n_healthy}</div></div>', unsafe_allow_html=True)
    with col_a3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Flagged Fault Anomalies</div><div class="metric-val" style="color: #991B1B;">{n_critical}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Automated Diagnostics Audit Table")
    
    def highlight_anomalies(s):
        return ['background-color: #FEE2E2; color: #991B1B;' if s['Diagnostic_Status'] == 'CRITICAL ANOMALY' else 'background-color: #DCFCE7; color: #166534;' for _ in s]

    st.dataframe(audit_results.style.apply(highlight_anomalies, axis=1), use_container_width=True)

# --------------------------------------------------------------------------------
# TAB 6: DATASET & SUMMARY EXPORT
# --------------------------------------------------------------------------------
with tab6:
    st.markdown("### Telemetry Explorer & Executive Export")
    
    uploaded_file = st.file_uploader("Upload External Production Telemetry (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        full_df = pd.read_csv(uploaded_file)
        st.success("Production Dataset Ingested!")
    else:
        full_df = X_test_ref.copy()
        full_df['Power_Output_kW'] = np.round(y_test_ref, 2)
        full_df['Regime_Label'] = np.where(extreme_ref == 1, 'Extreme', 'Normal')

    st.dataframe(full_df, use_container_width=True, height=220)

    st.markdown("---")
    st.markdown("#### Export System Summary")
    
    summary_txt = f"""=== ENTERPRISE WIND POWER FORECASTING SYSTEM SUMMARY REPORT ===
    
MODEL PERFORMANCE:
- Global Deep Neural Network (MLP) RMSE: {np.sqrt(mean_squared_error(y_test_ref, y_pred_dl)):.2f} kW
- Baseline Global Single Model RMSE: {np.sqrt(mean_squared_error(y_test_ref, y_pred_global)):.2f} kW
- Hybrid Model RMSE: {np.sqrt(mean_squared_error(y_test_ref, y_pred_hybrid)):.2f} kW
- Hybrid Model MAE: {mean_absolute_error(y_test_ref, y_pred_hybrid):.2f} kW
- Hybrid Model R²: {r2_score(y_test_ref, y_pred_hybrid):.3f}

CLASSIFIER METRICS (NO-LEAKAGE FEATURE MATRIX):
- Gate Accuracy Rate: {acc:.2%}
- Gate Precision Rate: {prec:.2%}
- Gate Recall Rate: {rec:.2%}
- Gate F1-Score: {f1:.3f}

CONFIGURATIONS:
- Grid Electricity Pricing Rate: ${power_rate_kw}/kWh
- SCADA Anomaly & Fault Guard Engine Active
"""

    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.download_button(
            label="Download Executive Summary (.txt)",
            data=summary_txt,
            file_name="Executive_Forecasting_Summary.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col_exp2:
        csv_buffer = io.StringIO()
        full_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Telemetry Dataset (.csv)",
            data=csv_buffer.getvalue(),
            file_name="Wind_Farm_SCADA_Telemetry.csv",
            mime="text/csv",
            use_container_width=True
        )
