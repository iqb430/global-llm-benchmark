import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="LLM Benchmark Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .reportview-container { background: #000000; color: #00FF00; font-family: 'Courier New', Courier, monospace; }
    .sidebar .sidebar-content { background: #111111; }
    h1, h2, h3 { color: #00FF00; text-transform: uppercase; border-bottom: 1px solid #00FF00; padding-bottom: 10px; }
    .stDataFrame { border: 1px solid #00FF00; }
    .stPlotlyChart { border: 1px solid #00FF00; }
</style>
""", unsafe_allow_html=True)

st.title("GLOBAL LLM BENCHMARK")
st.markdown("**BRUTALIST ANALYSIS ENGINE**")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
        df['avg_cost'] = (df['api_input_cost_per_1k_usd'] + df['api_output_cost_per_1k_usd']) / 2
        df['cognitive_density'] = df['mmlu_score'] / np.log1p(df['context_window'])
        return df
    except Exception as e:
        st.error(f"CRITICAL ERROR LOADING DATA: {e}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    st.sidebar.header("FILTERS")
    providers = st.sidebar.multiselect("Select Providers", options=df['provider'].unique(), default=df['provider'].unique())
    open_source_filter = st.sidebar.radio("Open Source Status", ["All", "Open Source", "Proprietary"])
    
    filtered_df = df[df['provider'].isin(providers)]
    if open_source_filter == "Open Source":
        filtered_df = filtered_df[filtered_df['open_source'] == True]
    elif open_source_filter == "Proprietary":
        filtered_df = filtered_df[filtered_df['open_source'] == False]

    st.subheader("RAW DATA MATRIX")
    st.dataframe(filtered_df[['model_name', 'provider', 'mmlu_score', 'context_window', 'avg_cost', 'open_source']], use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("COGNITION VS COST")
        fig_scatter = px.scatter(
            filtered_df, x="avg_cost", y="mmlu_score", color="provider", hover_name="model_name",
            size="context_window", log_x=True, template="plotly_dark",
            labels={"avg_cost": "Avg Cost (USD, Log Scale)", "mmlu_score": "MMLU Score"}
        )
        fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Courier New", color="#00FF00"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("TOP 10 BY COGNITIVE DENSITY")
        top_density = filtered_df.nlargest(10, 'cognitive_density')
        fig_bar = px.bar(
            top_density, x="cognitive_density", y="model_name", orientation='h', color="provider",
            template="plotly_dark", labels={"cognitive_density": "Cognitive Density", "model_name": "Model"}
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Courier New", color="#00FF00"))
        st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.warning("No dataset.csv found. Ensure data is present in the directory.")
