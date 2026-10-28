import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ---------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-Waste Technological Hubs: Analysis & Classification Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Academic & Corporate UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    /* Executive Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.12);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-val {
        color: #f8fafc;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .metric-delta {
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
    }

    /* Main Section Header Banner */
    .hero-banner {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        max-width: 900px;
        line-height: 1.55;
        margin-bottom: 14px;
    }

    /* Subtle Badges */
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        background: #1e293b;
        color: #93c5fd;
        border: 1px solid #334155;
    }

    /* Risk Tier Result Panels */
    .tier-box {
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
        border: 1px solid;
    }
    .tier-1 {
        background: rgba(16, 185, 129, 0.08);
        border-color: #059669;
        color: #34d399;
    }
    .tier-2 {
        background: rgba(59, 130, 246, 0.08);
        border-color: #2563eb;
        color: #60a5fa;
    }
    .tier-3 {
        background: rgba(245, 158, 11, 0.08);
        border-color: #d97706;
        color: #fbbf24;
    }
    .tier-4 {
        background: rgba(239, 68, 68, 0.08);
        border-color: #dc2626;
        color: #f87171;
    }

    /* Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #0c111d !important;
        border-right: 1px solid #1f2937;
    }
    .sidebar-brand-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }
    .sidebar-brand-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data & Model Loaders (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_datasets():
    cleaned_path = 'data/processed/cleaned_ewaste_dataset.csv'
    features_path = 'data/processed/engineered_ewaste_features.csv'
    
    df_clean = pd.read_csv(cleaned_path)
    df_feat = pd.read_csv(features_path)
    
    # Standardize string representations
    df_clean['Tech_Hub'] = df_clean['Tech_Hub'].replace({'So Paulo': 'São Paulo'})
    df_feat['Tech_Hub'] = df_feat['Tech_Hub'].replace({'So Paulo': 'São Paulo'})
    
    return df_clean, df_feat

@st.cache_resource
def load_models_bundle():
    bundle_path = 'outputs/models/pipeline_artifacts.joblib'
    if not os.path.exists(bundle_path):
        return None, {}
    bundle = joblib.load(bundle_path)
    
    model_files = {
        'Logistic Regression (94.2% Accuracy)': 'outputs/models/logistic_regression_model.joblib',
        'XGBoost Classifier (81.7% Accuracy)': 'outputs/models/xgboost_model.joblib',
        'Support Vector Machine (79.6% Accuracy)': 'outputs/models/support_vector_machine_svm_model.joblib',
        'Random Forest (79.2% Accuracy)': 'outputs/models/random_forest_model.joblib',
        'Decision Tree (74.2% Accuracy)': 'outputs/models/decision_tree_model.joblib',
        'Gaussian Naive Bayes (64.2% Accuracy)': 'outputs/models/naive_bayes_model.joblib'
    }
    
    models = {}
    for label, path in model_files.items():
        if os.path.exists(path):
            models[label] = joblib.load(path)
            
    return bundle, models

df_clean, df_feat = load_datasets()
bundle, models = load_models_bundle()

# Geographic Coordinates for Technological Hubs
HUB_COORDINATES = {
    'Bengaluru': {'lat': 12.9716, 'lon': 77.5946, 'country': 'India', 'region': 'South Asia'},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'country': 'India', 'region': 'South Asia'},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'country': 'India', 'region': 'South Asia'},
    'Pune': {'lat': 18.5204, 'lon': 73.8567, 'country': 'India', 'region': 'South Asia'},
    'Shenzhen': {'lat': 22.5431, 'lon': 114.0579, 'country': 'China', 'region': 'East Asia'},
    'Hanoi': {'lat': 21.0285, 'lon': 105.8542, 'country': 'Vietnam', 'region': 'Southeast Asia'},
    'Cyberjaya': {'lat': 2.9213, 'lon': 101.6559, 'country': 'Malaysia', 'region': 'Southeast Asia'},
    'Jakarta': {'lat': -6.2088, 'lon': 106.8456, 'country': 'Indonesia', 'region': 'Southeast Asia'},
    'Cairo': {'lat': 30.0444, 'lon': 31.2357, 'country': 'Egypt', 'region': 'Middle East & North Africa'},
    'Lagos': {'lat': 6.5244, 'lon': 3.3792, 'country': 'Nigeria', 'region': 'Sub-Saharan Africa'},
    'Nairobi': {'lat': -1.2921, 'lon': 36.8219, 'country': 'Kenya', 'region': 'Sub-Saharan Africa'},
    'São Paulo': {'lat': -23.5505, 'lon': -46.6333, 'country': 'Brazil', 'region': 'Latin America'},
}

# ---------------------------------------------------------
# Sidebar Navigation & Academic Metadata
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <div class="sidebar-brand-title">E-Waste Research Portal</div>
        <div class="sidebar-brand-sub">Empirical Data Science & Policy Evaluation</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation Menu",
        [
            "Executive Summary & Hub Overview",
            "Empirical Data Explorer",
            "Vulnerability Risk Classifier",
            "Model Performance Benchmarks",
            "Research Publication Figures",
            "Documentation & Datasets"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### Study Specifications")
    st.markdown("""
    * **Sample Population:** 1,200 Observations
    * **Observation Window:** 2018 – 2025
    * **Monitored Metros:** 12 Developing Tech Hubs
    * **Engineered Indicators:** 45 Features
    * **Top Holdout Accuracy:** **94.2%** *(Logistic Regression)*
    """)
    st.markdown("---")
    st.caption("Department of Computer Science and Engineering | Research Framework")

# ---------------------------------------------------------
# PAGE 1: EXECUTIVE SUMMARY & HUB OVERVIEW
# ---------------------------------------------------------
if page == "Executive Summary & Hub Overview":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">E-Waste Technological Hubs: Quantitative Analysis & Classification Framework</div>
        <div class="hero-subtitle">
            An empirical investigation analyzing electronic waste generation dynamics, formal-informal recycling disparities, and environmental vulnerability risk tiers across 12 prominent technological hubs in developing economies.
        </div>
        <div>
            <span class="badge-pill">1,200 Records</span>
            <span class="badge-pill">12 Metropolitan Hubs</span>
            <span class="badge-pill">6 Supervised Classifiers</span>
            <span class="badge-pill">94.2% Test Accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Empirical Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Mean Annual Generation</div>
            <div class="metric-val">89.37 <span style="font-size:0.95rem; font-weight:500; color:#94a3b8;">kt</span></div>
            <div class="metric-delta">Standard Deviation: 46.79 kt</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Informal Sector Recovery</div>
            <div class="metric-val">63.93 <span style="font-size:0.95rem; font-weight:500; color:#94a3b8;">%</span></div>
            <div class="metric-delta">Formal Collection Mean: 26.88%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Per-Capita Generation</div>
            <div class="metric-val">8.66 <span style="font-size:0.95rem; font-weight:500; color:#94a3b8;">kg/yr</span></div>
            <div class="metric-delta">Observed Range: 2.43 - 19.89 kg</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Mean Toxicity Index</div>
            <div class="metric-val">6.98 <span style="font-size:0.95rem; font-weight:500; color:#94a3b8;">/ 10</span></div>
            <div class="metric-delta">Substantial Heavy Metal Content</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # Geographic Map Section
    st.markdown("### Geographic Distribution of Monitored Technological Hubs")
    
    hub_agg = df_clean.groupby('Tech_Hub').agg({
        'Annual_Generation_kt': 'mean',
        'Per_Capita_Generation_kg': 'mean',
        'Formal_Collection_Rate_Pct': 'mean',
        'Informal_Recovery_Rate_Pct': 'mean',
        'Raw_Vulnerability_Score': 'mean',
        'Tech_Sector_GDP_Billion_USD': 'mean',
        'Population_Millions': 'mean'
    }).reset_index()
    
    hub_agg['lat'] = hub_agg['Tech_Hub'].map(lambda x: HUB_COORDINATES.get(x, {}).get('lat', 0.0))
    hub_agg['lon'] = hub_agg['Tech_Hub'].map(lambda x: HUB_COORDINATES.get(x, {}).get('lon', 0.0))
    hub_agg['Region'] = hub_agg['Tech_Hub'].map(lambda x: HUB_COORDINATES.get(x, {}).get('region', 'Unknown'))
    
    fig_map = px.scatter_geo(
        hub_agg,
        lat='lat',
        lon='lon',
        color='Raw_Vulnerability_Score',
        size='Annual_Generation_kt',
        hover_name='Tech_Hub',
        hover_data={
            'Annual_Generation_kt': ':.1f kt',
            'Informal_Recovery_Rate_Pct': ':.1f%',
            'Formal_Collection_Rate_Pct': ':.1f%',
            'Raw_Vulnerability_Score': ':.1f',
            'lat': False,
            'lon': False
        },
        color_continuous_scale='Viridis',
        projection='natural earth',
        title="Global Technological Hubs: Annual Generation (Marker Size) and Continuous Vulnerability Score (Color)"
    )
    fig_map.update_geos(
        showocean=True, oceancolor="#0c111d",
        showland=True, landcolor="#1e293b",
        showlakes=True, lakecolor="#0c111d",
        showcountries=True, countrycolor="#334155"
    )
    fig_map.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        height=500
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Analytical Dual Chart Row
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### Mean Annual E-Waste Generation by Hub")
        sorted_gen = hub_agg.sort_values(by='Annual_Generation_kt', ascending=True)
        fig_gen = px.bar(
            sorted_gen,
            x='Annual_Generation_kt',
            y='Tech_Hub',
            orientation='h',
            color='Annual_Generation_kt',
            color_continuous_scale='Blues',
            labels={'Annual_Generation_kt': 'Mean Annual Generation (kt)', 'Tech_Hub': 'Technological Hub'}
        )
        fig_gen.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_gen, use_container_width=True)
        
    with col_right:
        st.markdown("#### Processing Route Disparity: Formal vs. Informal")
        fig_disp = go.Figure()
        fig_disp.add_trace(go.Bar(
            name='Formal Collection Rate (%)',
            x=hub_agg['Tech_Hub'],
            y=hub_agg['Formal_Collection_Rate_Pct'],
            marker_color='#10b981'
        ))
        fig_disp.add_trace(go.Bar(
            name='Informal Recovery Rate (%)',
            x=hub_agg['Tech_Hub'],
            y=hub_agg['Informal_Recovery_Rate_Pct'],
            marker_color='#f43f5e'
        ))
        fig_disp.update_layout(
            barmode='group',
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=10, r=10, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_disp, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: EMPIRICAL DATA EXPLORER
# ---------------------------------------------------------
elif page == "Empirical Data Explorer":
    st.markdown("## Empirical Data Explorer & Multidimensional Filtering")
    st.markdown("Filter and inspect 1,200 empirical observations across 12 developing metropolitan technological hubs.")
    
    # Filter Row
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        selected_hubs = st.multiselect(
            "Select Technological Hubs:",
            options=sorted(df_clean['Tech_Hub'].unique()),
            default=sorted(df_clean['Tech_Hub'].unique())[:4]
        )
    with f2:
        selected_cats = st.multiselect(
            "E-Waste Stream Category:",
            options=sorted(df_clean['E_Waste_Category'].unique()),
            default=sorted(df_clean['E_Waste_Category'].unique())
        )
    with f3:
        selected_tiers = st.multiselect(
            "Management Risk Tier:",
            options=sorted(df_clean['Management_Risk_Tier'].unique()),
            default=sorted(df_clean['Management_Risk_Tier'].unique())
        )
    with f4:
        year_min, year_max = int(df_clean['Year'].min()), int(df_clean['Year'].max())
        selected_years = st.slider("Observation Period:", year_min, year_max, (year_min, year_max))
        
    filtered_df = df_clean[
        (df_clean['Tech_Hub'].isin(selected_hubs if selected_hubs else df_clean['Tech_Hub'].unique())) &
        (df_clean['E_Waste_Category'].isin(selected_cats if selected_cats else df_clean['E_Waste_Category'].unique())) &
        (df_clean['Management_Risk_Tier'].isin(selected_tiers if selected_tiers else df_clean['Management_Risk_Tier'].unique())) &
        (df_clean['Year'] >= selected_years[0]) & (df_clean['Year'] <= selected_years[1])
    ]
    
    st.markdown(f"**Showing {len(filtered_df):,} matched observations** out of 1,200 total records.")
    
    # Filtered Quick Metrics
    if len(filtered_df) > 0:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected Observations", len(filtered_df))
        m2.metric("Mean Generation", f"{filtered_df['Annual_Generation_kt'].mean():.2f} kt")
        m3.metric("Mean Informal Share", f"{filtered_df['Informal_Recovery_Rate_Pct'].mean():.1f}%")
        m4.metric("Mean Toxicity Index", f"{filtered_df['Heavy_Metal_Toxicity_Index'].mean():.2f} / 10")
        
        # Display DataFrame
        display_cols = [
            'Record_ID', 'Tech_Hub', 'Country', 'Year', 'E_Waste_Category',
            'Annual_Generation_kt', 'Per_Capita_Generation_kg', 'Formal_Collection_Rate_Pct',
            'Informal_Recovery_Rate_Pct', 'Heavy_Metal_Toxicity_Index', 'Management_Risk_Tier'
        ]
        st.dataframe(filtered_df[display_cols], use_container_width=True, height=350)
        
        # Download Button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Subset (CSV)",
            data=csv_data,
            file_name=f"ewaste_empirical_subset_{selected_years[0]}_{selected_years[1]}.csv",
            mime="text/csv"
        )
        
        st.write("")
        st.markdown("#### Bivariate Distribution: Per-Capita E-Waste vs. Tech Sector GDP")
        fig_scatter = px.scatter(
            filtered_df,
            x='Tech_Sector_GDP_Billion_USD',
            y='Per_Capita_Generation_kg',
            size='Annual_Generation_kt',
            color='Management_Risk_Tier',
            hover_name='Tech_Hub',
            hover_data=['Country', 'Year', 'E_Waste_Category', 'Formal_Collection_Rate_Pct'],
            color_discrete_map={
                'Tier-1 Controlled': '#10b981',
                'Tier-2 Emerging Risk': '#3b82f6',
                'Tier-3 High Vulnerability': '#f59e0b',
                'Tier-4 Critical Crisis': '#ef4444'
            },
            labels={
                'Tech_Sector_GDP_Billion_USD': 'Tech Sector GDP (Billion USD)',
                'Per_Capita_Generation_kg': 'Per-Capita Generation (kg/year)'
            },
            template="plotly_dark"
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("No records match the selected filter criteria. Please broaden your selection parameters.")

# ---------------------------------------------------------
# PAGE 3: VULNERABILITY RISK CLASSIFIER
# ---------------------------------------------------------
elif page == "Vulnerability Risk Classifier":
    st.markdown("## Vulnerability Risk Classification & Scenario Analysis Tool")
    st.markdown("""
    Evaluate municipal operational indicators to generate real-time environmental management vulnerability classifications using our benchmarked models.
    """)
    
    if bundle is None or len(models) == 0:
        st.error("Model artifacts bundle not found in outputs/models/. Please execute python scripts/run_pipeline.py first.")
    else:
        st.markdown("##### Load Standard Hub Profile Preset:")
        pre_col1, pre_col2, pre_col3, pre_col4, pre_col5 = st.columns(5)
        
        presets = {
            "Bengaluru": {"hub": "Bengaluru", "country": "India", "region": "South Asia", "gen": 142.5, "per_cap": 10.5, "pop": 13.6, "gdp": 135.0, "wf": 1650.0, "life": 22.0, "form": 24.5, "inf": 68.0, "tox": 7.4, "prec": 3400.0, "comp": "Minimal Compliance", "epr": 2.8, "desc": "rapid IT decommission server chassis copper PCB scrap"},
            "Shenzhen": {"hub": "Shenzhen", "country": "China", "region": "East Asia", "gen": 215.0, "per_cap": 12.2, "pop": 17.6, "gdp": 510.0, "wf": 2800.0, "life": 18.0, "form": 48.0, "inf": 45.0, "tox": 8.1, "prec": 4200.0, "comp": "Substantial Compliance", "epr": 4.1, "desc": "consumer electronics smart units lithium battery PCB high volume"},
            "Lagos": {"hub": "Lagos", "country": "Nigeria", "region": "Sub-Saharan Africa", "gen": 95.0, "per_cap": 6.1, "pop": 15.5, "gdp": 45.0, "wf": 420.0, "life": 48.0, "form": 8.0, "inf": 86.0, "tox": 7.8, "prec": 2100.0, "comp": "Non-Compliant", "epr": 1.4, "desc": "informal landfill scrap burning toxic leaching open dumpsite"},
            "São Paulo": {"hub": "São Paulo", "country": "Brazil", "region": "Latin America", "gen": 118.0, "per_cap": 9.4, "pop": 12.4, "gdp": 160.0, "wf": 950.0, "life": 32.0, "form": 32.0, "inf": 58.0, "tox": 6.5, "prec": 2900.0, "comp": "Minimal Compliance", "epr": 3.2, "desc": "telecommunication equipment copper wire cooling compressor"},
            "Cyberjaya": {"hub": "Cyberjaya", "country": "Malaysia", "region": "Southeast Asia", "gen": 38.0, "per_cap": 14.8, "pop": 2.6, "gdp": 68.0, "wf": 320.0, "life": 26.0, "form": 52.0, "inf": 38.0, "tox": 5.8, "prec": 3800.0, "comp": "Certified Compliance", "epr": 4.4, "desc": "data center server cooling refrigerant decommission certified audit"}
        }
        
        if 'selected_preset' not in st.session_state:
            st.session_state.selected_preset = presets["Bengaluru"]
            
        with pre_col1:
            if st.button("Bengaluru, India", use_container_width=True):
                st.session_state.selected_preset = presets["Bengaluru"]
        with pre_col2:
            if st.button("Shenzhen, China", use_container_width=True):
                st.session_state.selected_preset = presets["Shenzhen"]
        with pre_col3:
            if st.button("Lagos, Nigeria", use_container_width=True):
                st.session_state.selected_preset = presets["Lagos"]
        with pre_col4:
            if st.button("São Paulo, Brazil", use_container_width=True):
                st.session_state.selected_preset = presets["São Paulo"]
        with pre_col5:
            if st.button("Cyberjaya, Malaysia", use_container_width=True):
                st.session_state.selected_preset = presets["Cyberjaya"]
                
        preset = st.session_state.selected_preset
        
        st.markdown("---")
        
        # Input Form
        with st.form("classification_form"):
            r1_1, r1_2, r1_3 = st.columns(3)
            with r1_1:
                hub_val = st.selectbox("Technological Hub:", options=sorted(HUB_COORDINATES.keys()), index=sorted(HUB_COORDINATES.keys()).index(preset['hub']) if preset['hub'] in sorted(HUB_COORDINATES.keys()) else 0)
                country_val = HUB_COORDINATES.get(hub_val, {}).get('country', preset['country'])
                region_val = HUB_COORDINATES.get(hub_val, {}).get('region', preset['region'])
                st.caption(f"Country: **{country_val}** | Region: **{region_val}**")
            with r1_2:
                cat_val = st.selectbox("E-Waste Stream Category:", options=[
                    'IT & Telecommunication Equipment',
                    'Consumer Electronics & Screens',
                    'Small Digital & IoT Devices',
                    'Temperature Exchange & Cooling'
                ])
            with r1_3:
                model_choice = st.selectbox("Classification Architecture:", options=list(models.keys()))
                
            st.markdown("##### Quantitative Demographic & Operational Inputs")
            r2_1, r2_2, r2_3, r2_4 = st.columns(4)
            with r2_1:
                ann_gen = st.number_input("Annual Generation (kt):", 1.0, 500.0, float(preset['gen']), step=1.0)
                per_cap = st.number_input("Per-Capita (kg/person/yr):", 0.5, 30.0, float(preset['per_cap']), step=0.1)
            with r2_2:
                pop_mil = st.number_input("Metropolitan Population (Millions):", 0.5, 40.0, float(preset['pop']), step=0.1)
                gdp_bil = st.number_input("Tech Sector GDP (Billion USD):", 5.0, 1500.0, float(preset['gdp']), step=5.0)
            with r2_3:
                workforce_k = st.number_input("Tech Workforce (Thousands):", 10.0, 5000.0, float(preset['wf']), step=10.0)
                lifespan_m = st.number_input("Mean Device Lifespan (Months):", 6.0, 96.0, float(preset['life']), step=1.0)
            with r2_4:
                formal_rate = st.slider("Formal Collection Rate (%):", 0.0, 100.0, float(preset['form']))
                informal_rate = st.slider("Informal Recovery Rate (%):", 0.0, 100.0, float(preset['inf']))
                
            r3_1, r3_2, r3_3 = st.columns(3)
            with r3_1:
                toxicity_idx = st.slider("Heavy Metal Toxicity Score (1.0 - 10.0):", 1.0, 10.0, float(preset['tox']), step=0.1)
            with r3_2:
                compliance_val = st.selectbox("Landfill Diversion Compliance Level:", options=[
                    'Non-Compliant', 'Minimal Compliance', 'Substantial Compliance', 'Certified Compliance'
                ], index=1 if preset['comp'] == 'Minimal Compliance' else (0 if preset['comp'] == 'Non-Compliant' else 2))
            with r3_3:
                epr_score = st.slider("EPR Enforcement Score (1.0 - 5.0):", 1.0, 5.0, float(preset['epr']), step=0.1)
                
            stream_desc = st.text_input("Qualitative Stream Description (for NLP Keyword Extraction & Vectorization):", value=preset['desc'])
            
            submit_btn = st.form_submit_button("Execute Classification", use_container_width=True)
            
        if submit_btn:
            selected_model = models[model_choice]
            
            # NLP feature extraction
            keywords = [
                'flame retardant', 'lithium', 'mercury', 'lead', 'circuit board',
                'sensor', 'cooling', 'server', 'copper', 'gold',
                'informal', 'landfill', 'decommission', 'refrigerant', 'leaching'
            ]
            text_lower = str(stream_desc).lower()
            text_len = len(stream_desc)
            tok_cnt = len(stream_desc.split())
            sec_kw = sum(text_lower.count(k) for k in keywords)
            tfidf_vec = bundle['tfidf'].transform([stream_desc]).toarray()[0]
            
            # Domain feature engineering
            obs_vel = 12.0 / lifespan_m
            formal_inf_ratio = formal_rate / (informal_rate + 1.0)
            inf_burden = informal_rate / (formal_rate + informal_rate + 1e-5)
            ewaste_gdp = ann_gen / (gdp_bil + 1.0)
            toxic_load = ann_gen * toxicity_idx / 10.0
            rec_precious = (ann_gen * 1000.0 * 3000.0) / 1e6
            
            # Categorical encoding
            raw_hub_str = 'So Paulo' if (hub_val == 'São Paulo' and 'So Paulo' in bundle['cat_encoders']['Tech_Hub'].classes_) else hub_val
            hub_enc = bundle['cat_encoders']['Tech_Hub'].transform([raw_hub_str])[0]
            country_enc = bundle['cat_encoders']['Country'].transform([country_val])[0]
            region_enc = bundle['cat_encoders']['Region'].transform([region_val])[0]
            cat_enc = bundle['cat_encoders']['E_Waste_Category'].transform([cat_val])[0]
            comp_enc = bundle['cat_encoders']['Landfill_Diversion_Compliance'].transform([compliance_val])[0]
            
            row_dict = {
                'Tech_Hub': hub_enc,
                'Country': country_enc,
                'Region': region_enc,
                'Year': 2024,
                'E_Waste_Category': cat_enc,
                'Annual_Generation_kt': ann_gen,
                'Per_Capita_Generation_kg': per_cap,
                'Population_Millions': pop_mil,
                'Tech_Sector_GDP_Billion_USD': gdp_bil,
                'Tech_Workforce_Thousands': workforce_k,
                'Average_Device_Lifespan_Months': lifespan_m,
                'Formal_Collection_Rate_Pct': formal_rate,
                'Informal_Recovery_Rate_Pct': informal_rate,
                'Heavy_Metal_Toxicity_Index': toxicity_idx,
                'Precious_Metal_Value_USD_per_Ton': 3000.0,
                'Landfill_Diversion_Compliance': comp_enc,
                'EPR_Enforcement_Score': epr_score,
                'Text_Length': text_len,
                'Token_Count': tok_cnt,
                'Security_Keyword_Count': sec_kw
            }
            
            for i, term in enumerate(bundle['tfidf'].get_feature_names_out()):
                row_dict[f'tfidf_{term}'] = tfidf_vec[i]
                
            row_dict['Obsolescence_Velocity_Index'] = obs_vel
            row_dict['Formal_to_Informal_Ratio'] = formal_inf_ratio
            row_dict['Informal_Burden_Pct'] = inf_burden
            row_dict['E_Waste_per_GDP'] = ewaste_gdp
            row_dict['Toxic_Load_kt'] = toxic_load
            row_dict['Recoverable_Precious_Value_MUSD'] = rec_precious
            
            df_row = pd.DataFrame([row_dict])[bundle['feature_names']]
            scaled_row = bundle['scaler'].transform(df_row.values)
            pred_idx = selected_model.predict(scaled_row)[0]
            pred_class = bundle['classes'][pred_idx]
            
            probs = selected_model.predict_proba(scaled_row)[0] if hasattr(selected_model, 'predict_proba') else None
            
            # Display Result
            st.markdown("### Classification Outcome")
            tier_class_map = {
                'Tier-1 Controlled': (
                    'tier-1',
                    'Tier-1: Controlled (Low Environmental Vulnerability)',
                    'Formal recycling infrastructure and robust EPR frameworks successfully capture municipal electronics streams with negligible uncontrolled informal leakage.'
                ),
                'Tier-2 Emerging Risk': (
                    'tier-2',
                    'Tier-2: Emerging Risk (Moderate Vulnerability)',
                    'Rapid hardware refresh cycles and growing consumer volumes threaten to exceed formal processing capacity if municipal collection programs are not expanded.'
                ),
                'Tier-3 High Vulnerability': (
                    'tier-3',
                    'Tier-3: High Vulnerability (Significant Informal Sector Exposure)',
                    'Informal scrap recovery operations handle over 60% of municipal streams, causing substantial hazardous heavy metal release in urban dismantling clusters.'
                ),
                'Tier-4 Critical Crisis': (
                    'tier-4',
                    'Tier-4: Critical Crisis (Severe Ecological and Chemical Hazard)',
                    'Extreme toxic burden, minimal formal collection, and complete absence of landfill compliance represent an immediate public health and environmental crisis.'
                )
            }
            css_class, title_text, desc_text = tier_class_map.get(pred_class, ('tier-2', pred_class, ''))
            
            st.markdown(f"""
            <div class="tier-box {css_class}">
                <div style="font-size:1.25rem; font-weight:700; margin-bottom:6px;">{title_text}</div>
                <div style="font-size:0.92rem; line-height:1.5;">{desc_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if probs is not None:
                prob_df = pd.DataFrame({
                    'Management Risk Tier': bundle['classes'],
                    'Posterior Probability (%)': (probs * 100).round(2)
                })
                fig_prob = px.bar(
                    prob_df,
                    x='Management Risk Tier',
                    y='Posterior Probability (%)',
                    color='Management Risk Tier',
                    color_discrete_map={
                        'Tier-1 Controlled': '#10b981',
                        'Tier-2 Emerging Risk': '#3b82f6',
                        'Tier-3 High Vulnerability': '#f59e0b',
                        'Tier-4 Critical Crisis': '#ef4444'
                    },
                    text_auto='.1f',
                    template="plotly_dark",
                    title="Model Posterior Probability Distribution Across Target Classes"
                )
                fig_prob.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    showlegend=False
                )
                st.plotly_chart(fig_prob, use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE BENCHMARKS
# ---------------------------------------------------------
elif page == "Model Performance Benchmarks":
    st.markdown("## Supervised Learning Classification Benchmarks")
    st.markdown("Empirical comparison of six classification models evaluated on the 80:20 stratified holdout split (240 test instances).")
    
    comp_path = 'outputs/tables/model_comparison.csv'
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("#### Test Set Performance Summary")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Empirical Findings:**
            * **Logistic Regression achieved 94.2% test accuracy**, demonstrating that composite environmental vulnerability tiers are highly separable along standardized continuous indicators and NLP keyword distributions.
            * **Tree ensembles (XGBoost 81.7%, Random Forest 79.2%)** demonstrated balanced recall across minority and majority risk tiers.
            """)
        
        with c2:
            comp_df['Acc_Numeric'] = comp_df['Accuracy'].str.rstrip('%').astype(float)
            fig_acc = px.bar(
                comp_df.sort_values(by='Acc_Numeric', ascending=True),
                x='Acc_Numeric',
                y='Model',
                orientation='h',
                color='Acc_Numeric',
                color_continuous_scale='Blues',
                text='Accuracy',
                labels={'Acc_Numeric': 'Test Accuracy (%)', 'Model': 'Model Architecture'},
                title="Comparative Accuracy across 6 Classifiers"
            )
            fig_acc.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_acc, use_container_width=True)
            
    # Confusion Matrix Viewer
    cm_path = 'outputs/tables/confusion_matrices.json'
    if os.path.exists(cm_path):
        with open(cm_path, 'r') as f:
            cm_dict = json.load(f)
            
        st.markdown("---")
        st.markdown("#### Confusion Matrix Diagnostic Tool")
        
        selected_cm_model = st.selectbox("Select Model Architecture to Inspect:", list(cm_dict.keys()), index=0)
        cm_matrix = np.array(cm_dict[selected_cm_model])
        tier_labels = ['Tier-1 Controlled', 'Tier-2 Emerging Risk', 'Tier-3 High Vulnerability', 'Tier-4 Critical Crisis']
        
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Risk Tier", y="Ground Truth Risk Tier", color="Sample Count"),
            x=tier_labels,
            y=tier_labels,
            text_auto=True,
            color_continuous_scale='Blues',
            template="plotly_dark"
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: RESEARCH PUBLICATION FIGURES
# ---------------------------------------------------------
elif page == "Research Publication Figures":
    st.markdown("## Academic Publication Figures Gallery")
    st.markdown("Publication-quality (300 DPI) figures generated during empirical data processing and hypothesis testing.")
    
    figures_dir = 'outputs/figures'
    if os.path.exists(figures_dir):
        all_figures = sorted([f for f in os.listdir(figures_dir) if f.endswith('.png')])
        
        category_filter = st.selectbox(
            "Filter Figures by Study Phase:",
            options=[
                "All Figures (22)",
                "Chapter 3: Methodology & Workflow Architecture",
                "Chapter 4: Exploratory Data Analysis & Trends",
                "Chapter 6: Machine Learning Evaluation",
                "Appendix B: Diagnostic Confusion Matrices & Feature Importances"
            ]
        )
        
        filtered_figures = all_figures
        if "Chapter 3" in category_filter:
            filtered_figures = [f for f in all_figures if 'fig_3' in f]
        elif "Chapter 4" in category_filter:
            filtered_figures = [f for f in all_figures if 'fig_4' in f]
        elif "Chapter 6" in category_filter:
            filtered_figures = [f for f in all_figures if 'fig_6' in f]
        elif "Appendix B" in category_filter:
            filtered_figures = [f for f in all_figures if 'fig_b' in f]
            
        st.markdown(f"Displaying **{len(filtered_figures)}** figures:")
        
        g1, g2 = st.columns(2)
        for i, fig_name in enumerate(filtered_figures):
            fig_path = os.path.join(figures_dir, fig_name)
            target_col = g1 if i % 2 == 0 else g2
            with target_col:
                clean_title = fig_name.replace('.png', '').replace('_', ' ').title()
                st.markdown(f"##### {clean_title}")
                st.image(fig_path, use_container_width=True)
                st.caption(f"Filename: {fig_name} (300 DPI)")
                st.write("")

# ---------------------------------------------------------
# PAGE 6: DOCUMENTATION & DATASETS
# ---------------------------------------------------------
elif page == "Documentation & Datasets":
    st.markdown("## Research Documentation & Project Artifacts")
    st.markdown("Access compiled research documentation, formal project reports, and serialized datasets.")
    
    col_rep1, col_rep2 = st.columns(2)
    
    with col_rep1:
        st.markdown("### Academic Reports & Papers")
        
        pdf_path = 'E_Waste_Data_Science_Project_Report.pdf'
        if os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            st.download_button(
                label="Download Full Project Report (PDF, 10 Chapters)",
                data=pdf_bytes,
                file_name="E_Waste_Data_Science_Project_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.caption(f"Document Size: {len(pdf_bytes) / (1024*1024):.2f} MB")
            
        docx_path = 'E_Waste_Data_Science_Project_Report.docx'
        if os.path.exists(docx_path):
            with open(docx_path, 'rb') as f:
                docx_bytes = f.read()
            st.download_button(
                label="Download Project Report (Microsoft Word .docx)",
                data=docx_bytes,
                file_name="E_Waste_Data_Science_Project_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
            st.caption(f"Document Size: {len(docx_bytes) / (1024*1024):.2f} MB")
            
    with col_rep2:
        st.markdown("### Empirical Datasets")
        
        clean_csv = 'data/processed/cleaned_ewaste_dataset.csv'
        if os.path.exists(clean_csv):
            with open(clean_csv, 'rb') as f:
                c_bytes = f.read()
            st.download_button(
                label="Download Cleaned Empirical Dataset (CSV, 1,200 rows)",
                data=c_bytes,
                file_name="cleaned_ewaste_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        feat_csv = 'data/processed/engineered_ewaste_features.csv'
        if os.path.exists(feat_csv):
            with open(feat_csv, 'rb') as f:
                f_bytes = f.read()
            st.download_button(
                label="Download Engineered Feature Matrix (CSV, 45 columns)",
                data=f_bytes,
                file_name="engineered_ewaste_features.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("### Executive Summary of Empirical Findings")
    st.markdown("""
    1. **Dominance of Informal Processing Pathways:** Across all 12 monitored metropolitan hubs, the informal scrap sector processes an average of **63.93%** of annual e-waste, substantially surpassing municipal formal capacity (**26.88%**).
    2. **Hazardous Contamination Risk:** The regional heavy metal toxicity index averages **6.98 / 10**, driven by unmonitored dismantling, acid leaching, and open burning in metropolitan scrap markets.
    3. **Supervised Classification Performance:** Evaluating six distinct predictive architectures on an 80:20 stratified holdout partition, **Logistic Regression achieved the highest accuracy (94.2%)**, establishing an operational decision-support tool for municipal vulnerability assessment.
    """)
