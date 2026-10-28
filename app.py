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
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-Waste Tech Hubs Analytics & ML Platform",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism & High-End Dark UI CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1527 0%, #060913 100%);
        color: #e2e8f0;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(18, 27, 47, 0.7);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 15px 35px rgba(14, 165, 233, 0.15);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-val {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .metric-delta {
        font-size: 0.78rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .delta-green { color: #10b981; }
    .delta-cyan { color: #38bdf8; }
    .delta-amber { color: #f59e0b; }
    .delta-rose { color: #f43f5e; }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.25) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.02rem;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 15px;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .badge-pill-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.25);
    }
    .badge-pill-violet {
        background: rgba(139, 92, 246, 0.12);
        color: #a78bfa;
        border-color: rgba(139, 92, 246, 0.25);
    }

    /* Tier Result Box */
    .tier-box {
        border-radius: 14px;
        padding: 22px;
        margin: 15px 0;
        border: 1px solid;
    }
    .tier-1 {
        background: rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.4);
        color: #34d399;
    }
    .tier-2 {
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.4);
        color: #38bdf8;
    }
    .tier-3 {
        background: rgba(245, 158, 11, 0.12);
        border-color: rgba(245, 158, 11, 0.4);
        color: #fbbf24;
    }
    .tier-4 {
        background: rgba(244, 63, 94, 0.12);
        border-color: rgba(244, 63, 94, 0.4);
        color: #fb7185;
    }

    /* Custom Streamlit adjustments */
    div[data-testid="stSidebar"] {
        background-color: #080d1a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        color: #f8fafc !important;
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
    
    # Fix encoding representation if present
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
        'Logistic Regression (94.2%)': 'outputs/models/logistic_regression_model.joblib',
        'XGBoost Classifier (81.7%)': 'outputs/models/xgboost_model.joblib',
        'Support Vector Machine (79.6%)': 'outputs/models/support_vector_machine_svm_model.joblib',
        'Random Forest (79.2%)': 'outputs/models/random_forest_model.joblib',
        'Decision Tree (74.2%)': 'outputs/models/decision_tree_model.joblib',
        'Gaussian Naive Bayes (64.2%)': 'outputs/models/naive_bayes_model.joblib'
    }
    
    models = {}
    for label, path in model_files.items():
        if os.path.exists(path):
            models[label] = joblib.load(path)
            
    return bundle, models

df_clean, df_feat = load_datasets()
bundle, models = load_models_bundle()

# Hub Geographic Coordinates for Interactive Map
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
# Sidebar Navigation & Quick Stats
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:18px;">
        <div style="background: linear-gradient(135deg, #0ea5e9, #6366f1); padding:10px; border-radius:12px; font-size:1.6rem;">
            ♻️
        </div>
        <div>
            <h3 style="margin:0; font-size:1.15rem; font-weight:700; color:#f8fafc;">E-Waste Analytics</h3>
            <p style="margin:0; font-size:0.75rem; color:#94a3b8;">EDA & ML Research Framework</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "🌐 Global Hubs & KPIs",
            "🔍 Interactive Data Explorer",
            "🧪 Live ML Tier Predictor",
            "📊 Model Benchmark Lab",
            "🖼️ Publication Figures Gallery",
            "📑 Research Report & Exports"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### 📌 Framework Metadata")
    st.markdown("""
    - **Dataset Size:** 1,200 Observations
    - **Time Horizon:** 2018 – 2025
    - **Tech Hubs:** 12 Developing Metros
    - **Engineered Features:** 45 Dimensions
    - **Top ML Accuracy:** **94.2%** *(Logistic Reg.)*
    """)
    st.markdown("---")
    st.caption("Developed for Academic & Policy Decision-Support in Sustainable Electronics Management.")

# ---------------------------------------------------------
# PAGE 1: GLOBAL HUBS & EXECUTIVE KPIS
# ---------------------------------------------------------
if page == "🌐 Global Hubs & KPIs":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">E-Waste Technological Hubs Analytics Platform</div>
        <div class="hero-subtitle">
            An empirical data science and machine learning classification framework characterizing municipal electronic waste dynamics, informal recycling reliance, and environmental management risk tiers across 12 developing technological hubs.
        </div>
        <div>
            <span class="badge-pill">1,200 Observations</span>
            <span class="badge-pill badge-pill-emerald">12 Global Tech Hubs</span>
            <span class="badge-pill badge-pill-violet">6 Supervised ML Models</span>
            <span class="badge-pill">94.2% Peak Accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Mean Annual Generation</div>
            <div class="metric-val">89.37 <span style="font-size:1rem; font-weight:500; color:#94a3b8;">kt</span></div>
            <div class="metric-delta delta-cyan">↑ 46.79 kt std dev across hubs</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Mean Informal Recovery</div>
            <div class="metric-val">63.93 <span style="font-size:1rem; font-weight:500; color:#94a3b8;">%</span></div>
            <div class="metric-delta delta-rose">Surpasses Formal Rate (26.88%)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Per-Capita Generation</div>
            <div class="metric-val">8.66 <span style="font-size:1rem; font-weight:500; color:#94a3b8;">kg/yr</span></div>
            <div class="metric-delta delta-amber">Max: 19.89 kg/person/yr</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Heavy Metal Toxicity</div>
            <div class="metric-val">6.98 <span style="font-size:1rem; font-weight:500; color:#94a3b8;">/ 10</span></div>
            <div class="metric-delta delta-rose">High Hazardous Chemical Load</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # Interactive Map Section
    st.markdown("### 🗺️ Global Geographic Distribution of Monitored Tech Hubs")
    
    # Aggregate data per hub
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
        title="Global Technological Hubs: Mean Annual E-Waste Generation & Vulnerability Score"
    )
    fig_map.update_geos(
        showocean=True, oceancolor="#0b1120",
        showland=True, landcolor="#1e293b",
        showlakes=True, lakecolor="#0b1120",
        showcountries=True, countrycolor="#334155"
    )
    fig_map.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#060913",
        plot_bgcolor="#060913",
        height=500
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Analytical Dual Chart Row
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### ⚡ Mean Annual E-Waste Generation by Hub")
        sorted_gen = hub_agg.sort_values(by='Annual_Generation_kt', ascending=True)
        fig_gen = px.bar(
            sorted_gen,
            x='Annual_Generation_kt',
            y='Tech_Hub',
            orientation='h',
            color='Annual_Generation_kt',
            color_continuous_scale='Blues',
            labels={'Annual_Generation_kt': 'Annual Generation (kt)', 'Tech_Hub': 'Tech Hub'}
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
        st.markdown("#### ⚖️ Formal vs. Informal Processing Disparity")
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
# PAGE 2: INTERACTIVE DATA EXPLORER
# ---------------------------------------------------------
elif page == "🔍 Interactive Data Explorer":
    st.markdown("## 🔍 Interactive Data Explorer & Multidimensional Filtering")
    st.markdown("Slice and filter 1,200 empirical observations across 12 developing tech hubs.")
    
    # Filter Row
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        selected_hubs = st.multiselect(
            "Select Tech Hub(s):",
            options=sorted(df_clean['Tech_Hub'].unique()),
            default=sorted(df_clean['Tech_Hub'].unique())[:4]
        )
    with f2:
        selected_cats = st.multiselect(
            "E-Waste Category:",
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
        selected_years = st.slider("Observation Year Range:", year_min, year_max, (year_min, year_max))
        
    filtered_df = df_clean[
        (df_clean['Tech_Hub'].isin(selected_hubs if selected_hubs else df_clean['Tech_Hub'].unique())) &
        (df_clean['E_Waste_Category'].isin(selected_cats if selected_cats else df_clean['E_Waste_Category'].unique())) &
        (df_clean['Management_Risk_Tier'].isin(selected_tiers if selected_tiers else df_clean['Management_Risk_Tier'].unique())) &
        (df_clean['Year'] >= selected_years[0]) & (df_clean['Year'] <= selected_years[1])
    ]
    
    st.markdown(f"**Showing `{len(filtered_df):,}` matched records** out of 1,200 total records.")
    
    # Filtered Quick Metrics
    if len(filtered_df) > 0:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Filtered Records", len(filtered_df))
        m2.metric("Mean Generation", f"{filtered_df['Annual_Generation_kt'].mean():.2f} kt")
        m3.metric("Avg Informal Share", f"{filtered_df['Informal_Recovery_Rate_Pct'].mean():.1f}%")
        m4.metric("Avg Toxicity Index", f"{filtered_df['Heavy_Metal_Toxicity_Index'].mean():.2f}/10")
        
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
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name=f"ewaste_filtered_{selected_years[0]}_{selected_years[1]}.csv",
            mime="text/csv"
        )
        
        st.write("")
        # Scatter Plot: Per Capita vs GDP
        st.markdown("#### 📈 Per Capita E-Waste vs. Tech Sector GDP (Bubble Size = Annual Generation)")
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
                'Tier-2 Emerging Risk': '#38bdf8',
                'Tier-3 High Vulnerability': '#f59e0b',
                'Tier-4 Critical Crisis': '#f43f5e'
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
        st.warning("No records match the selected filter criteria. Please broaden your selection.")

# ---------------------------------------------------------
# PAGE 3: LIVE ML TIER PREDICTOR
# ---------------------------------------------------------
elif page == "🧪 Live ML Tier Predictor":
    st.markdown("## 🧪 Live Machine Learning Vulnerability Tier Predictor")
    st.markdown("""
    Configure metropolitan e-waste indicators to generate real-time environmental management vulnerability classifications using our benchmarked ML models.
    """)
    
    if bundle is None or len(models) == 0:
        st.error("Model artifacts bundle not found in `outputs/models/`. Please execute `python scripts/run_pipeline.py` first.")
    else:
        # Preset Quick-Load Buttons
        st.markdown("##### ⚡ Quick-Load Hub Preset Profiles:")
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
            if st.button("🏙️ Bengaluru", use_container_width=True):
                st.session_state.selected_preset = presets["Bengaluru"]
        with pre_col2:
            if st.button("🏭 Shenzhen", use_container_width=True):
                st.session_state.selected_preset = presets["Shenzhen"]
        with pre_col3:
            if st.button("🌍 Lagos", use_container_width=True):
                st.session_state.selected_preset = presets["Lagos"]
        with pre_col4:
            if st.button("🌴 São Paulo", use_container_width=True):
                st.session_state.selected_preset = presets["São Paulo"]
        with pre_col5:
            if st.button("💻 Cyberjaya", use_container_width=True):
                st.session_state.selected_preset = presets["Cyberjaya"]
                
        preset = st.session_state.selected_preset
        
        st.markdown("---")
        
        # Input Form
        with st.form("prediction_form"):
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
                model_choice = st.selectbox("Predictive Classification Algorithm:", options=list(models.keys()))
                
            st.markdown("##### 📊 Operational & Demographic Quantities")
            r2_1, r2_2, r2_3, r2_4 = st.columns(4)
            with r2_1:
                ann_gen = st.number_input("Annual Generation (kt):", 1.0, 500.0, float(preset['gen']), step=1.0)
                per_cap = st.number_input("Per-Capita Gen (kg/yr):", 0.5, 30.0, float(preset['per_cap']), step=0.1)
            with r2_2:
                pop_mil = st.number_input("Population (Millions):", 0.5, 40.0, float(preset['pop']), step=0.1)
                gdp_bil = st.number_input("Tech Sector GDP ($ Billion):", 5.0, 1500.0, float(preset['gdp']), step=5.0)
            with r2_3:
                workforce_k = st.number_input("Tech Workforce (Thousands):", 10.0, 5000.0, float(preset['wf']), step=10.0)
                lifespan_m = st.number_input("Average Lifespan (Months):", 6.0, 96.0, float(preset['life']), step=1.0)
            with r2_4:
                formal_rate = st.slider("Formal Collection Rate (%):", 0.0, 100.0, float(preset['form']))
                informal_rate = st.slider("Informal Recovery Rate (%):", 0.0, 100.0, float(preset['inf']))
                
            r3_1, r3_2, r3_3 = st.columns(3)
            with r3_1:
                toxicity_idx = st.slider("Heavy Metal Toxicity (1-10):", 1.0, 10.0, float(preset['tox']), step=0.1)
            with r3_2:
                compliance_val = st.selectbox("Landfill Diversion Compliance:", options=[
                    'Non-Compliant', 'Minimal Compliance', 'Substantial Compliance', 'Certified Compliance'
                ], index=1 if preset['comp'] == 'Minimal Compliance' else (0 if preset['comp'] == 'Non-Compliant' else 2))
            with r3_3:
                epr_score = st.slider("EPR Enforcement Score (1.0-5.0):", 1.0, 5.0, float(preset['epr']), step=0.1)
                
            stream_desc = st.text_input("Qualitative Stream Description (for NLP TF-IDF extraction):", value=preset['desc'])
            
            submit_btn = st.form_submit_button("🚀 Compute Vulnerability Classification", use_container_width=True)
            
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
            
            # Domain features
            obs_vel = 12.0 / lifespan_m
            formal_inf_ratio = formal_rate / (informal_rate + 1.0)
            inf_burden = informal_rate / (formal_rate + informal_rate + 1e-5)
            ewaste_gdp = ann_gen / (gdp_bil + 1.0)
            toxic_load = ann_gen * toxicity_idx / 10.0
            rec_precious = (ann_gen * 1000.0 * 3000.0) / 1e6
            
            # Encoding
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
            
            # Predict Proba
            probs = selected_model.predict_proba(scaled_row)[0] if hasattr(selected_model, 'predict_proba') else None
            
            # Render Results
            st.markdown("### 🎯 Classification Result")
            tier_class_map = {
                'Tier-1 Controlled': ('tier-1', '🟢 Controlled / Low Environmental Vulnerability', 'Effective EPR frameworks and high formal processing capacity prevent uncontrolled municipal leakage.'),
                'Tier-2 Emerging Risk': ('tier-2', '🔵 Emerging Risk / Moderate Vulnerability', 'Growing consumer refresh velocity requires capacity upgrades before informal processing overwhelms formal capacity.'),
                'Tier-3 High Vulnerability': ('tier-3', '🟠 High Vulnerability / Severe Informal Leakage', 'Informal crude recovery dominates (>60%), causing significant hazardous heavy metal exposure in urban scrap clusters.'),
                'Tier-4 Critical Crisis': ('tier-4', '🔴 Critical Crisis / Severe Ecological Hazard', 'Extreme toxic burden, minimal formal collection, and zero landfill diversion compliance create an imminent environmental and occupational emergency.')
            }
            css_class, title_text, desc_text = tier_class_map.get(pred_class, ('tier-2', pred_class, ''))
            
            st.markdown(f"""
            <div class="tier-box {css_class}">
                <div style="font-size:1.4rem; font-weight:800; margin-bottom:6px;">{title_text}</div>
                <div style="font-size:0.95rem; opacity:0.95;">{desc_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probabilities bar chart
            if probs is not None:
                prob_df = pd.DataFrame({
                    'Risk Tier': bundle['classes'],
                    'Confidence / Probability (%)': (probs * 100).round(2)
                })
                fig_prob = px.bar(
                    prob_df,
                    x='Risk Tier',
                    y='Confidence / Probability (%)',
                    color='Risk Tier',
                    color_discrete_map={
                        'Tier-1 Controlled': '#10b981',
                        'Tier-2 Emerging Risk': '#38bdf8',
                        'Tier-3 High Vulnerability': '#f59e0b',
                        'Tier-4 Critical Crisis': '#f43f5e'
                    },
                    text_auto='.1f',
                    template="plotly_dark",
                    title="Model Posterior Probability Distribution across Risk Tiers"
                )
                fig_prob.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    showlegend=False
                )
                st.plotly_chart(fig_prob, use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: MODEL BENCHMARK LAB
# ---------------------------------------------------------
elif page == "📊 Model Benchmark Lab":
    st.markdown("## 📊 Machine Learning Model Benchmark & Evaluation Lab")
    st.markdown("Comparative performance of six classification architectures on the 80:20 stratified holdout split (240 test instances).")
    
    # Model comparison table
    comp_path = 'outputs/tables/model_comparison.csv'
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("#### 🏆 Benchmark Performance Summary")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.info("""
            **Key Academic Finding:**  
            * **Logistic Regression achieved the highest accuracy (94.2%)**, indicating a linearly separable multi-class boundary when leveraging scaled continuous indices and NLP TF-IDF features.
            * **XGBoost (81.7%)** and **Random Forest (79.2%)** exhibited superior robustness to non-linear regional interactions and feature skewness.
            """)
        
        with c2:
            # Bar chart of accuracy
            comp_df['Acc_Numeric'] = comp_df['Accuracy'].str.rstrip('%').astype(float)
            fig_acc = px.bar(
                comp_df.sort_values(by='Acc_Numeric', ascending=True),
                x='Acc_Numeric',
                y='Model',
                orientation='h',
                color='Acc_Numeric',
                color_continuous_scale='Teal',
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
        st.markdown("#### 🔍 Interactive Confusion Matrix Explorer")
        
        selected_cm_model = st.selectbox("Select Model to Inspect Confusion Matrix:", list(cm_dict.keys()), index=0)
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
# PAGE 5: PUBLICATION FIGURES GALLERY
# ---------------------------------------------------------
elif page == "🖼️ Publication Figures Gallery":
    st.markdown("## 🖼️ Academic Publication Figures Gallery")
    st.markdown("High-resolution (300 DPI) publication-grade figures generated by the end-to-end research pipeline.")
    
    figures_dir = 'outputs/figures'
    if os.path.exists(figures_dir):
        all_figures = sorted([f for f in os.listdir(figures_dir) if f.endswith('.png')])
        
        category_filter = st.selectbox(
            "Filter Figures by Research Chapter:",
            options=[
                "All Figures (22)",
                "Chapter 3: Methodology & Workflow Architecture",
                "Chapter 4: Exploratory Data Analysis & Trends",
                "Chapter 6: Machine Learning Evaluation",
                "Appendix B: Confusion Matrices & Feature Importance"
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
        
        # Grid display (2 columns)
        g1, g2 = st.columns(2)
        for i, fig_name in enumerate(filtered_figures):
            fig_path = os.path.join(figures_dir, fig_name)
            target_col = g1 if i % 2 == 0 else g2
            with target_col:
                clean_title = fig_name.replace('.png', '').replace('_', ' ').title()
                st.markdown(f"##### 📌 {clean_title}")
                st.image(fig_path, use_container_width=True)
                st.caption(f"File: `{fig_name}` (300 DPI)")
                st.write("")

# ---------------------------------------------------------
# PAGE 6: RESEARCH REPORT & EXPORTS
# ---------------------------------------------------------
elif page == "📑 Research Report & Exports":
    st.markdown("## 📑 Research Documentation & Academic Exports")
    st.markdown("Access and download official compiled project reports and serialized datasets.")
    
    col_rep1, col_rep2 = st.columns(2)
    
    with col_rep1:
        st.markdown("### 📄 Compiled Academic Documents")
        
        pdf_path = 'E_Waste_Data_Science_Project_Report.pdf'
        if os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📥 Download Full PDF Project Report (10 Chapters)",
                data=pdf_bytes,
                file_name="E_Waste_Data_Science_Project_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.caption(f"Size: {len(pdf_bytes) / (1024*1024):.2f} MB | Standard University IEEE/Academic Formatting")
            
        docx_path = 'E_Waste_Data_Science_Project_Report.docx'
        if os.path.exists(docx_path):
            with open(docx_path, 'rb') as f:
                docx_bytes = f.read()
            st.download_button(
                label="📥 Download Microsoft Word Report (.docx)",
                data=docx_bytes,
                file_name="E_Waste_Data_Science_Project_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
            st.caption(f"Size: {len(docx_bytes) / (1024*1024):.2f} MB")
            
    with col_rep2:
        st.markdown("### 🗄️ Processed Datasets & Feature Matrices")
        
        clean_csv = 'data/processed/cleaned_ewaste_dataset.csv'
        if os.path.exists(clean_csv):
            with open(clean_csv, 'rb') as f:
                c_bytes = f.read()
            st.download_button(
                label="📥 Download Cleaned Empirical Dataset (1,200 rows)",
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
                label="📥 Download Engineered Feature Matrix (45 columns)",
                data=f_bytes,
                file_name="engineered_ewaste_features.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("### 📖 Executive Summary & Policy Recommendations")
    st.markdown("""
    1. **Surging Informal Dominance:** Across all 12 monitored technology hubs, the informal scrap sector recovers an average of **63.93%** of annual e-waste streams, exceeding formal municipal capacity (**26.88%**).
    2. **High Heavy Metal Toxicity Index:** The regional toxicity mean reaches **6.98 / 10**, driven by open-air burning and acid-bath leaching of flame-retardant circuit boards and cooling compressors.
    3. **Predictive Tier Classification:** Using an 80:20 stratified holdout split, **Logistic Regression achieved 94.2% accuracy**, providing a deployable decision-support tool for municipal policymakers to preempt ecological crises in tier 3 & 4 hubs.
    """)
