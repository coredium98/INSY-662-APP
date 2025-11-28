"""
Flood Risk Cost-Benefit Simulation Dashboard
Using K-Nearest Neighbors Model

This Streamlit app provides an interactive dashboard for simulating
cost-benefit analysis of flood prevention interventions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
import pickle
import os

# Import custom transformers needed for unpickling the model
from custom_transformers import (
    _RemainderColsList,
    RegionAdder, LogTransformer, SingleColumnPowerTransformer,
    CityMedianImputer, CityModeImputer, ElevationKNNImputer, ElevationLocalDelta,
    ColumnDropper
)

# Page configuration
st.set_page_config(
    page_title="Flood Risk Cost-Benefit Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌊 Flood Risk Cost-Benefit Simulation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Interactive Analysis Using K-Nearest Neighbors Model")

# Load model and data
@st.cache_resource
def load_model_and_data():
    """Load the trained KNN model and test data"""
    try:
        # Check if model file exists
        if os.path.exists('knn_model.pkl'):
            # Custom unpickler to handle __main__ module references
            import sys
            import custom_transformers

            # Make custom transformers available in __main__ for unpickling
            sys.modules['__main__'].RegionAdder = custom_transformers.RegionAdder
            sys.modules['__main__'].LogTransformer = custom_transformers.LogTransformer
            sys.modules['__main__'].SingleColumnPowerTransformer = custom_transformers.SingleColumnPowerTransformer
            sys.modules['__main__'].CityMedianImputer = custom_transformers.CityMedianImputer
            sys.modules['__main__'].CityModeImputer = custom_transformers.CityModeImputer
            sys.modules['__main__'].ElevationKNNImputer = custom_transformers.ElevationKNNImputer
            sys.modules['__main__'].ElevationLocalDelta = custom_transformers.ElevationLocalDelta
            sys.modules['__main__'].ColumnDropper = custom_transformers.ColumnDropper

            # Add _RemainderColsList to sklearn if it doesn't exist (version compatibility)
            from sklearn.compose import _column_transformer
            if not hasattr(_column_transformer, '_RemainderColsList'):
                _column_transformer._RemainderColsList = custom_transformers._RemainderColsList

            with open('knn_model.pkl', 'rb') as f:
                model = pickle.load(f)

            # Check if pickle contains a dict (new format) or just the model (old format)
            if isinstance(model, dict):
                return model
            else:
                # Old format: pickle contains just the model
                # Try to load test data from .npy files
                try:
                    X_test = np.load('X_test.npy', allow_pickle=True)
                    y_test = np.load('y_test.npy', allow_pickle=True)

                    # Get n_neighbors from the model if it's a Pipeline
                    n_neighbors = None
                    if hasattr(model, 'named_steps') and 'knn' in model.named_steps:
                        n_neighbors = model.named_steps['knn'].n_neighbors

                    # Get feature names if available
                    feature_names = None
                    if hasattr(X_test, 'columns'):
                        feature_names = list(X_test.columns)

                    return {
                        'model': model,
                        'X_test': X_test,
                        'y_test': y_test,
                        'n_neighbors': n_neighbors,
                        'feature_names': feature_names
                    }
                except FileNotFoundError:
                    st.error("⚠️ Test data files (x_test.npy, y_test.npy) not found!")
                    st.info("""
                    ### Missing Test Data Files

                    Please add the following files to the repository:
                    - `x_test.npy`: Test features
                    - `y_test.npy`: Test labels

                    You can create these files in your notebook with:
                    ```python
                    import numpy as np
                    np.save('x_test.npy', X_test)
                    np.save('y_test.npy', y_test)
                    ```
                    """)
                    return None
        else:
            st.error("⚠️ Model file not found. Please run the model training notebook first and export the model.")
            st.info("📝 Instructions: Run the notebook and execute the model export cell to generate 'knn_model.pkl'")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load model
model_data = load_model_and_data()

if model_data is not None:
    st.session_state.model_loaded = True
    best_model = model_data['model']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    feature_names = model_data.get('feature_names', None)

    st.success("✅ Model loaded successfully!")

    # Display model info
    with st.expander("ℹ️ Model Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", "K-Nearest Neighbors")
        with col2:
            st.metric("Number of Neighbors", model_data.get('n_neighbors', 'N/A'))
        with col3:
            st.metric("Test Samples", len(y_test))

        if feature_names:
            st.write(f"**Features ({len(feature_names)}):** {', '.join(feature_names[:5])}... (and {len(feature_names)-5} more)")

    # Sidebar - Parameter Controls
    st.sidebar.header("🎛️ Simulation Parameters")
    st.sidebar.markdown("Adjust the parameters below to see how they affect the optimal intervention strategy:")

    # Cost per intervention
    cost_per_area = st.sidebar.slider(
        "💰 Cost per Intervention Area ($M)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Cost of implementing flood prevention measures in one area"
    )

    # Damage per area
    damage_per_area = st.sidebar.slider(
        "💥 Potential Damage per Flood ($M)",
        min_value=2.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Average damage cost if flooding occurs in an area"
    )

    # Effectiveness rate
    effectiveness_rate = st.sidebar.slider(
        "✅ Intervention Effectiveness (%)",
        min_value=40,
        max_value=100,
        value=70,
        step=5,
        help="Percentage of floods prevented by interventions"
    ) / 100

    st.sidebar.markdown("---")

    # Add information section
    with st.sidebar.expander("📊 About This Dashboard"):
        st.markdown("""
        This dashboard simulates the cost-benefit analysis of flood prevention interventions:

        - **Cost per Area**: Investment needed per intervention
        - **Damage per Flood**: Economic loss from flooding
        - **Effectiveness**: Success rate of interventions

        The model predicts flood risk and calculates optimal intervention thresholds.
        """)

    # Function to simulate cost-benefit analysis
    def simulate_cost_benefit(cost_per_area_val, damage_per_area_val, effectiveness_val):
        """Simulate cost-benefit analysis with given parameters"""

        # Get predictions for test set
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Evaluate different thresholds
        thresholds = np.arange(0.05, 0.96, 0.05)
        results = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            # Number of areas flagged for intervention
            areas_flagged = y_pred.sum()

            # True positives (correct flood predictions)
            true_positives = ((y_pred == 1) & (y_test == 1)).sum()

            # Financial calculations
            intervention_cost = areas_flagged * cost_per_area_val
            prevented_damage = true_positives * damage_per_area_val * effectiveness_val
            net_benefit = prevented_damage - intervention_cost
            roi = (net_benefit / intervention_cost * 100) if intervention_cost > 0 else 0

            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'areas_flagged': areas_flagged,
                'true_positives': true_positives,
                'intervention_cost': intervention_cost,
                'prevented_damage': prevented_damage,
                'net_benefit': net_benefit,
                'roi': roi
            })

        results_df = pd.DataFrame(results)

        # Find optimal threshold
        optimal_idx = results_df['net_benefit'].idxmax()
        optimal_result = results_df.iloc[optimal_idx]

        return results_df, optimal_result

    # Run simulation
    with st.spinner('Running cost-benefit simulation...'):
        results_df, optimal_result = simulate_cost_benefit(cost_per_area, damage_per_area, effectiveness_rate)

    # Display Results
    st.markdown("---")
    st.header("📈 Simulation Results")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Optimal Threshold",
            f"{optimal_result['threshold']:.2f}",
            help="Probability threshold that maximizes net benefit"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Net Benefit",
            f"${optimal_result['net_benefit']:.2f}M",
            help="Total benefit minus total cost"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "ROI",
            f"{optimal_result['roi']:.1f}%",
            help="Return on investment percentage"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Areas to Intervene",
            f"{int(optimal_result['areas_flagged'])}",
            help="Number of areas flagged for intervention"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed optimal scenario info
    st.markdown("---")
    st.subheader("🎯 Optimal Cost-Benefit Scenario")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Financial Metrics:**")
        st.write(f"- **Intervention Cost:** ${optimal_result['intervention_cost']:.2f}M")
        st.write(f"- **Prevented Damage:** ${optimal_result['prevented_damage']:.2f}M")
        st.write(f"- **Net Benefit:** ${optimal_result['net_benefit']:.2f}M")
        st.write(f"- **ROI:** {optimal_result['roi']:.1f}%")

    with col2:
        st.markdown("**Performance Metrics:**")
        st.write(f"- **Precision:** {optimal_result['precision']:.1%}")
        st.write(f"- **Recall:** {optimal_result['recall']:.1%}")
        st.write(f"- **True Positives:** {int(optimal_result['true_positives'])} floods prevented")
        st.write(f"- **Areas Flagged:** {int(optimal_result['areas_flagged'])} interventions")

    # Visualizations
    st.markdown("---")
    st.header("📊 Visual Analysis")

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["💰 Cost-Benefit Analysis", "🎯 Model Performance", "📋 Data Table"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Net Benefit vs Threshold
        axes[0].plot(results_df['threshold'], results_df['net_benefit'],
                    linewidth=2.5, color='#2ecc71', marker='o', markersize=4)
        axes[0].axvline(optimal_result['threshold'], color='red',
                       linestyle='--', linewidth=2, label=f"Optimal: {optimal_result['threshold']:.2f}")
        axes[0].axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        axes[0].set_xlabel('Risk Threshold', fontsize=11)
        axes[0].set_ylabel('Net Benefit (Millions $)', fontsize=11)
        axes[0].set_title('Net Benefit by Threshold', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: ROI vs Threshold
        axes[1].plot(results_df['threshold'], results_df['roi'],
                    linewidth=2.5, color='#3498db', marker='o', markersize=4)
        axes[1].axvline(optimal_result['threshold'], color='red',
                       linestyle='--', linewidth=2, label=f"Optimal: {optimal_result['threshold']:.2f}")
        axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        axes[1].set_xlabel('Risk Threshold', fontsize=11)
        axes[1].set_ylabel('ROI (%)', fontsize=11)
        axes[1].set_title('Return on Investment by Threshold', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Additional plot: Cost vs Benefit
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(results_df['threshold'], results_df['intervention_cost'],
               linewidth=2.5, color='#e74c3c', marker='o', markersize=4, label='Intervention Cost')
        ax.plot(results_df['threshold'], results_df['prevented_damage'],
               linewidth=2.5, color='#2ecc71', marker='s', markersize=4, label='Prevented Damage')
        ax.axvline(optimal_result['threshold'], color='black',
                  linestyle='--', linewidth=2, label=f"Optimal Threshold: {optimal_result['threshold']:.2f}")
        ax.set_xlabel('Risk Threshold', fontsize=11)
        ax.set_ylabel('Amount (Millions $)', fontsize=11)
        ax.set_title('Cost vs Benefit Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    with tab2:
        fig3, axes2 = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 3: Precision vs Recall
        axes2[0].plot(results_df['threshold'], results_df['precision'],
                     linewidth=2.5, color='#9b59b6', marker='o', markersize=4, label='Precision')
        axes2[0].plot(results_df['threshold'], results_df['recall'],
                     linewidth=2.5, color='#f39c12', marker='s', markersize=4, label='Recall')
        axes2[0].axvline(optimal_result['threshold'], color='red',
                        linestyle='--', linewidth=2, label=f"Optimal: {optimal_result['threshold']:.2f}")
        axes2[0].set_xlabel('Risk Threshold', fontsize=11)
        axes2[0].set_ylabel('Score', fontsize=11)
        axes2[0].set_title('Precision & Recall by Threshold', fontsize=13, fontweight='bold')
        axes2[0].legend(fontsize=10)
        axes2[0].grid(True, alpha=0.3)

        # Plot 4: Areas Flagged
        axes2[1].plot(results_df['threshold'], results_df['areas_flagged'],
                     linewidth=2.5, color='#1abc9c', marker='o', markersize=4)
        axes2[1].axvline(optimal_result['threshold'], color='red',
                        linestyle='--', linewidth=2, label=f"Optimal: {optimal_result['threshold']:.2f}")
        axes2[1].set_xlabel('Risk Threshold', fontsize=11)
        axes2[1].set_ylabel('Number of Areas', fontsize=11)
        axes2[1].set_title('Areas Flagged for Intervention', fontsize=13, fontweight='bold')
        axes2[1].legend(fontsize=10)
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig3)

    with tab3:
        st.subheader("📋 Detailed Results Table")

        # Format the dataframe for display
        display_df = results_df.copy()
        display_df['threshold'] = display_df['threshold'].map('{:.2f}'.format)
        display_df['precision'] = display_df['precision'].map('{:.1%}'.format)
        display_df['recall'] = display_df['recall'].map('{:.1%}'.format)
        display_df['intervention_cost'] = display_df['intervention_cost'].map('${:.2f}M'.format)
        display_df['prevented_damage'] = display_df['prevented_damage'].map('${:.2f}M'.format)
        display_df['net_benefit'] = display_df['net_benefit'].map('${:.2f}M'.format)
        display_df['roi'] = display_df['roi'].map('{:.1f}%'.format)
        display_df['areas_flagged'] = display_df['areas_flagged'].astype(int)
        display_df['true_positives'] = display_df['true_positives'].astype(int)

        # Rename columns for better readability
        display_df.columns = [
            'Threshold', 'Precision', 'Recall', 'Areas Flagged', 'True Positives',
            'Intervention Cost', 'Prevented Damage', 'Net Benefit', 'ROI'
        ]

        st.dataframe(display_df, use_container_width=True, height=400)

        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="cost_benefit_simulation_results.csv",
            mime="text/csv"
        )

    # Interpretation Guide
    st.markdown("---")
    with st.expander("💡 How to Interpret These Results"):
        st.markdown("""
        ### Understanding the Dashboard

        **Optimal Threshold:**
        - The probability cutoff that maximizes net benefit
        - Areas with flood probability above this threshold should receive interventions

        **Net Benefit:**
        - Total prevented damage minus intervention costs
        - Higher values indicate better economic outcomes

        **ROI (Return on Investment):**
        - Percentage return for every dollar invested
        - Positive ROI means interventions are cost-effective

        **Precision:**
        - Of areas we intervene in, what % actually needed it
        - Higher precision = less wasted interventions

        **Recall:**
        - Of all areas that will flood, what % do we catch
        - Higher recall = fewer missed flood areas

        ### Using the Sliders

        Adjust the parameters to explore different scenarios:
        - **Increase Cost**: See how higher intervention costs affect optimal strategy
        - **Increase Damage**: Higher flood damage justifies more interventions
        - **Change Effectiveness**: Lower effectiveness requires more conservative approach
        """)

else:
    # Model not loaded - show instructions
    st.warning("⚠️ Model file not found!")
    st.info("""
    ### Setup Instructions:

    1. Run the Jupyter notebook: `Comprehensive_Flood_Model_Comparison_as_of_Nov_23.ipynb`
    2. Execute all cells to train the KNN model
    3. Run the model export cell to create `knn_model.pkl`
    4. Refresh this page

    Or create the model file by running:
    ```python
    import pickle

    model_data = {
        'model': best_model,
        'X_test': X_test,
        'y_test': y_test,
        'n_neighbors': best_knn.n_neighbors,
        'feature_names': list(X_test.columns) if hasattr(X_test, 'columns') else None
    }

    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🌊 Flood Risk Cost-Benefit Simulation Dashboard | Built with Streamlit</p>
        <p>Model: K-Nearest Neighbors | Data-driven flood prevention planning</p>
    </div>
""", unsafe_allow_html=True)
