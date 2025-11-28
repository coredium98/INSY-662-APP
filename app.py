"""
Flood Risk Cost-Benefit Simulation Dashboard
Using K-Nearest Neighbors Model
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import traceback

# Import custom transformers for unpickling
from custom_transformers import (
    _RemainderColsList,
    RegionAdder, LogTransformer, SingleColumnPowerTransformer,
    CityMedianImputer, CityModeImputer, ElevationKNNImputer,
    ElevationLocalDelta, ColumnDropper
)

# Page config
st.set_page_config(
    page_title="Flood Risk Cost-Benefit Dashboard",
    page_icon="Wave",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.8rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .metric-card {background-color: #f0f2f6; padding: 1.2rem; border-radius: 0.8rem; border-left: 5px solid #1f77b4;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Wave Flood Risk Cost-Benefit Simulation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Interactive Analysis Using K-Nearest Neighbors Model")

# ===============================================
# LOAD MODEL & DATA (FIXED & ROBUST)
# ===============================================
@st.cache_resource
def load_model_and_data():
    try:
        if not os.path.exists('knn_model.pkl'):
            st.error("Model file `knn_model.pkl` not found!")
            return None

        # === Fix for pickled custom transformers ===
        import sys
        import custom_transformers as ct

        for name, cls in [
            ('RegionAdder', ct.RegionAdder),
            ('LogTransformer', ct.LogTransformer),
            ('SingleColumnPowerTransformer', ct.SingleColumnPowerTransformer),
            ('CityMedianImputer', ct.CityMedianImputer),
            ('CityModeImputer', ct.CityModeImputer),
            ('ElevationKNNImputer', ct.ElevationKNNImputer),
            ('ElevationLocalDelta', ct.ElevationLocalDelta),
            ('ColumnDropper', ct.ColumnDropper),
        ]:
            sys.modules['__main__'].__dict__[name] = cls

        # Fix for old sklearn versions
        from sklearn.compose import _column_transformer
        if not hasattr(_column_transformer, '_RemainderColsList'):
            _column_transformer._RemainderColsList = ct._RemainderColsList

        # Load the model
        with open('knn_model.pkl', 'rb') as f:
            loaded = pickle.load(f)

        if isinstance(loaded, dict):
            return loaded  # New format with model + test data

        # Old format: just the pipeline
        model = loaded

        # Load test data
        if not os.path.exists('X_test.npy') or not os.path.exists('y_test.npy'):
            st.error("Missing `X_test.npy` or `y_test.npy`!")
            return None

        X_test = np.load('X_test.npy', allow_pickle=True)
        y_test = np.load('y_test.npy', allow_pickle=True)

        # Convert to DataFrame only if it's object dtype (i.e., saved as df)
        if X_test.dtype == object:
            try:
                X_test = pd.DataFrame(X_test)
            except:
                pass

        n_neighbors = model.named_steps['knn'].n_neighbors if hasattr(model, 'named_steps') and 'knn' in model.named_steps else 'Unknown'

        return {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'n_neighbors': n_neighbors,
            'feature_names': None
        }

    except Exception as e:
        st.error("Failed to load model or data")
        st.code(traceback.format_exc())
        return None

# ===============================================
# LOAD DATA
# ===============================================
model_data = load_model_and_data()

if model_data is None:
    st.stop()  # Stop execution if loading failed

best_model = model_data['model']
X_test = model_data['X_test']
y_test = model_data['y_test']
n_neighbors = model_data['n_neighbors']

st.success("Model & data loaded successfully!")

# Model info
with st.expander("Model Information", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", "K-Nearest Neighbors")
    with col2:
        st.metric("Number of Neighbors", n_neighbors)
    with col3:
        st.metric("Test Samples", len(y_test))

# ===============================================
# SIDEBAR CONTROLS
# ===============================================
st.sidebar.header("Simulation Parameters")
cost_per_area = st.sidebar.slider("Cost per Intervention Area ($M)", 0.5, 5.0,1.0,0.1)
damage_per_area = st.sidebar.slider("Potential Damage per Flood ($M)", 2.0,20.0,5.0,0.5)
effectiveness_rate = st.sidebar.slider("Intervention Effectiveness (%)", 40,100,70,5) / 100

with st.sidebar.expander("About This Dashboard"):
    st.markdown("""
    This tool helps decision-makers understand:
    - How many areas to protect
    - At what risk threshold
    - For maximum net benefit and ROI
    """)

# ===============================================
# COST-BENEFIT SIMULATION
# ===============================================
def simulate_cost_benefit(cost, damage, effectiveness):
    try:
        y_proba = best_model.predict_proba(X_test)[:, 1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.code(traceback.format_exc())
        return None, None

    thresholds = np.arange(0.05, 0.96, 0.05)
    results = []

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        flagged = y_pred.sum()
        tp = ((y_pred == 1) & (y_test == 1)).sum()

        cost_total = flagged * cost
        damage_saved = tp * damage * effectiveness
        net = damage_saved - cost_total
        roi = (net / cost_total * 100) if cost_total > 0 else 0

        results.append({
            'threshold': th,
            'precision': precision,
            'recall': recall,
            'areas_flagged': flagged,
            'true_positives': tp,
            'intervention_cost': cost_total,
            'prevented_damage': damage_saved,
            'net_benefit': net,
            'roi': roi
        })

    df = pd.DataFrame(results)
    optimal = df.loc[df['net_benefit'].idxmax()]
    return df, optimal

# Run simulation
with st.spinner("Running cost-benefit analysis..."):
    results_df, optimal = simulate_cost_benefit(cost_per_area, damage_per_area, effectiveness_rate)

if results_df is None:
    st.stop()

# ===============================================
# DISPLAY RESULTS
# ===============================================
st.markdown("---")
st.header("Simulation Results")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Optimal Threshold", f"{optimal['threshold']:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Net Benefit", f"${optimal['net_benefit']:.2f}M")
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("ROI", f"{optimal['roi']:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Areas to Protect", int(optimal['areas_flagged']))
    st.markdown('</div>', unsafe_allow_html=True)

# Detailed view
st.subheader("Optimal Scenario Details")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Financials**")
    st.write(f"- Intervention Cost: **${optimal['intervention_cost']:.2f}M**")
    st.write(f"- Prevented Damage: **${optimal['prevented_damage']:.2f}M**")
    st.write(f"- Net Benefit: **${optimal['net_benefit']:.2f}M**")
    st.write(f"- ROI: **{optimal['roi']:.1f}%**")
with col2:
    st.markdown("**Model Performance**")
    st.write(f"- Precision: **{optimal['precision']:.1%}**")
    st.write(f"- Recall: **{optimal['recall']:.1%}**")
    st.write(f"- True Positives Prevented: **{int(optimal['true_positives'])}**")

# Charts
tab1, tab2, tab3 = st.tabs(["Cost-Benefit", "Model Performance", "Full Table"])

with tab1:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(results_df['threshold'], results_df['net_benefit'], 'o-', color='#2ecc71', linewidth=2.5)
    ax[0].axvline(optimal['threshold'], color='red', linestyle='--', label=f"Optimal = {optimal['threshold']:.2f}")
    ax[0].set_title("Net Benefit vs Threshold")
    ax[0].set_xlabel("Risk Threshold")
    ax[0].set_ylabel("Net Benefit ($M)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(results_df['threshold'], results_df['roi'], 'o-', color='#3498db', linewidth=2.5)
    ax[1].axvline(optimal['threshold'], color='red', linestyle='--')
    ax[1].set_title("ROI vs Threshold")
    ax[1].set_xlabel("Risk Threshold")
    ax[1].set_ylabel("ROI (%)")
    ax[1].grid(alpha=0.3)

    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(results_df['threshold'], results_df['precision'], 'o-', label='Precision', color='#9b59b6')
    ax[0].plot(results_df['threshold'], results_df['recall'], 's-', label='Recall', color='#e67e22')
    ax[0].axvline(optimal['threshold'], color='red', linestyle='--')
    ax[0].legend()
    ax[0].set_title("Precision & Recall")
    ax[0].grid(alpha=0.3)

    ax[1].plot(results_df['threshold'], results_df['areas_flagged'], 'o-', color='#1abc9c')
    ax[1].axvline(optimal['threshold'], color='red', linestyle='--')
    ax[1].set_title("Areas Flagged for Intervention")
    ax[1].set_xlabel("Threshold")
    ax[1].set_ylabel("Number of Areas")
    ax[1].grid(alpha=0.3)

    st.pyplot(fig)

with tab3:
    display = results_df.copy()
    display['threshold'] = display['threshold'].map('{:.2f}'.format)
    display['precision'] = display['precision'].map('{:.1%}'.format)
    display['recall'] = display['recall'].map('{:.1%}'.format)
    display['intervention_cost'] = display['intervention_cost'].map('${:,.2f}M'.format)
    display['prevented_damage'] = display['prevented_damage'].map('${:,.2f}M'.format)
    display['net_benefit'] = display['net_benefit'].map('${:,.2f}M'.format)
    display['roi'] = display['roi'].map('{:.1f}%'.format)

    st.dataframe(display, use_container_width=True)

    csv = results_df.to_csv(index=False).encode()
    st.download_button("Download Results as CSV", csv, "flood_cost_benefit_results.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Wave Flood Risk Cost-Benefit Dashboard | Built with Streamlit | KNN Model</p>", unsafe_allow_html=True)
