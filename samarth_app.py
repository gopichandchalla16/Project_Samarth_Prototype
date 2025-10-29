import streamlit as st
import pandas as pd
from google import genai
import os

# --- 1. CONFIGURATION AND THEMES ---
st.set_page_config(
    page_title="Build For Bharat Fellowship 2026 | Project Samarth",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🇮🇳"
)

# --- Custom CSS for Enhanced UI + Footer ---
st.markdown("""
<style>
/* Global Theme */
body {
    font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
    background-color: #F9FAFB;
}

/* Header Bar */
.stApp > header {
    background: linear-gradient(to right, #FF9933, #FFFFFF, #138808);
    height: 4px;
}

/* Title Styling */
.stTitle {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #0B5345;
    text-align: center;
    margin-top: -10px;
    letter-spacing: 0.5px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0B5345;
    color: white;
    padding: 1rem;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FF9933 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #FF9933;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #FFFFFF;
    color: #0B5345;
}

/* Section Headers */
h2, h3, h4 {
    color: #145A32 !important;
    font-weight: 700 !important;
}

/* Metric Cards */
[data-testid="stMetricValue"] {
    color: #0B5345 !important;
    font-weight: 700 !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(to right, #FF9933, #138808);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #138808, #FF9933);
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: bold;
    color: #145A32 !important;
}

/* Footer Styling */
footer {
    visibility: hidden;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: linear-gradient(to right, #0B5345, #145A32);
    color: white;
    text-align: center;
    padding: 0.6rem;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
}
.footer a {
    color: #FF9933;
    text-decoration: none;
    font-weight: 600;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)


# --- 2. DATA INGESTION ---
@st.cache_data
def load_and_process_data():
    try:
        imd_df = pd.read_csv("Min_Max_Seasonal_IMD_2017.csv")
        imd_df['ANNUAL - MEAN'] = (imd_df['ANNUAL - MIN'] + imd_df['ANNUAL - MAX']) / 2
        imd_df['Decade'] = (imd_df['YEAR'] // 10) * 10
        decadal_avg = imd_df.groupby('Decade')['ANNUAL - MEAN'].mean().reset_index()

        warming_increase = decadal_avg.iloc[-1]['ANNUAL - MEAN'] - decadal_avg.iloc[0]['ANNUAL - MEAN']
        baseline_mean = imd_df['ANNUAL - MEAN'].mean()

        imd_insights = {
            "warming_trend": f"The Annual Mean Temperature has increased by {warming_increase:.2f}°C from the early 1900s to the 2010s.",
            "hottest_decade_mean": decadal_avg.iloc[-1]['ANNUAL - MEAN'],
            "baseline_temp": baseline_mean,
            "source": "India Meteorological Department (IMD) - Seasonal and Annual Min/Max Temp Series (data.gov.in)."
        }
    except Exception as e:
        imd_df = pd.DataFrame()
        imd_insights = {"source": f"IMD Data Missing / Error loading: {e}"}

    crop_data = {
        'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2017, 2017],
        'State': ['Andhra Pradesh', 'Telangana', 'Andhra Pradesh', 'Telangana', 'Andhra Pradesh', 'Andhra Pradesh', 'Telangana', 'Andhra Pradesh', 'Telangana', 'Andhra Pradesh'],
        'District': ['Guntur', 'Kurnool', 'Guntur', 'Kurnool', 'Guntur', 'Guntur', 'Kurnool', 'Guntur', 'Kurnool', 'Nellore'],
        'Season': ['Rabi', 'Kharif', 'Rabi', 'Kharif', 'Rabi', 'Kharif', 'Kharif', 'Rabi', 'Kharif', 'Kharif'],
        'Crop': ['Rice', 'Cotton', 'Rice', 'Cotton', 'Maize', 'Rice', 'Rice', 'Maize', 'Cotton', 'Rice'],
        'Crop_Type': ['Water-Intensive', 'Drought-Tolerant', 'Water-Intensive', 'Drought-Tolerant', 'Drought-Tolerant', 'Water-Intensive', 'Water-Intensive', 'Drought-Tolerant', 'Drought-Tolerant', 'Water-Intensive'],
        'Production_Tonnes': [150000, 50000, 145000, 48000, 30000, 100000, 95000, 40000, 55000, 120000]
    }
    crop_df = pd.DataFrame(crop_data)

    ap_production = crop_df[crop_df['State'] == 'Andhra Pradesh'].groupby('Crop_Type')['Production_Tonnes'].sum()
    water_intensive_prod = ap_production.get('Water-Intensive', 0)
    drought_tolerant_prod = ap_production.get('Drought-Tolerant', 0)

    crop_metrics = {
        "water_intensive_prod": water_intensive_prod,
        "drought_tolerant_prod": drought_tolerant_prod,
        "ratio_risk": water_intensive_prod / (water_intensive_prod + drought_tolerant_prod) if (water_intensive_prod + drought_tolerant_prod) > 0 else 0,
        "source": "Ministry of Agriculture & Farmers Welfare - Crop Production Statistics (data.gov.in - SIMULATED)."
    }

    return imd_df, imd_insights, crop_df, crop_metrics


imd_df, imd_insights, crop_df, crop_metrics = load_and_process_data()


# --- 3. RAG + Gemini ---
def query_imd_data(query, df, insights):
    summary_context = "\n".join([f"- {k}: {v}" for k, v in insights.items() if k not in ['hottest_decade_mean', 'baseline_temp']])
    if any(k in query.lower() for k in ['warming', 'baseline', 'decade', 'climate', 'trend']):
        return f"CLIMATE SUMMARY INSIGHTS:\n{summary_context}"
    if not df.empty:
        data_markdown = df[['YEAR', 'ANNUAL - MEAN', 'ANNUAL - MIN', 'ANNUAL - MAX']].tail(5).to_markdown(index=False)
        return f"RECENT CLIMATE DATA SNIPPET:\n{data_markdown}"
    return "IMD data not available."


def query_crop_data(query, df):
    if any(k in query.lower() for k in ['crop_type', 'drought', 'water-intensive', 'andhra pradesh', 'production']):
        state = 'Andhra Pradesh'
        df_filtered = df[df['State'] == state]
        aggregated = df_filtered.groupby('Crop_Type')['Production_Tonnes'].sum().sort_values(ascending=False).reset_index()
        aggregated['Production_Tonnes'] = aggregated['Production_Tonnes'].apply(lambda x: f"{x:,.0f} Tonnes")
        return f"CROP DATA SNIPPET (Production by Crop Type in {state}):\n{aggregated.to_markdown(index=False)}"
    top_producers = df.groupby(['State', 'Crop'])['Production_Tonnes'].sum().nlargest(3).reset_index()
    top_producers['Production_Tonnes'] = top_producers['Production_Tonnes'].apply(lambda x: f"{x:,.0f} Tonnes")
    return f"TOP PRODUCTION OVERVIEW:\n{top_producers.to_markdown(index=False)}"


@st.cache_resource
def setup_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error loading Gemini client: {e}")
        return None


def ask_samarth_agent(client, user_question):
    climate_context = query_imd_data(user_question, imd_df, imd_insights)
    crop_context = query_crop_data(user_question, crop_df)
    system_prompt = (
        "You are 'Project Samarth,' an intelligent policy data agent for Build For Bharat Fellowship 2026. "
        "Analyze and synthesize insights from IMD and Agriculture Ministry datasets accurately with sources."
    )
    prompt = f"""
    SYSTEM PROMPT: {system_prompt}
    --- CLIMATE DATA (IMD) ---
    Source: {imd_insights['source']}
    {climate_context}
    --- AGRICULTURE DATA ---
    Source: {crop_metrics['source']}
    {crop_context}
    USER QUESTION: {user_question}
    """
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"API ERROR: {e}"


# --- 4. FRONT-END ---
client = setup_gemini_client()

st.markdown("<h1 class='stTitle'>🇮🇳 Build For Bharat Fellowship 2026 Cohort (Data Science)</h1>", unsafe_allow_html=True)
st.caption("Empowering Data-Driven Policy Innovation | Powered by Project Samarth")

if not client:
    st.stop()

st.divider()
st.subheader("📊 Key Policy Indicators — Andhra Pradesh")

col1, col2, col3 = st.columns(3)
col1.metric("Warming Trend (1901–2017)", f"+{imd_insights['warming_trend'].split()[4]}", f"Baseline {imd_insights['baseline_temp']:.2f}°C")
col2.metric("Water-Intensive Production", f"{crop_metrics['water_intensive_prod']/1000:,.0f}k Tonnes", f"{crop_metrics['ratio_risk']*100:.0f}% of Total")
col3.metric("Drought-Tolerant Output", f"{crop_metrics['drought_tolerant_prod']/1000:,.0f}k Tonnes", "Focus Area for Policy Growth")

st.divider()

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/41/Emblem_of_India.svg", width=80)
    st.header("🧠 Samarth Agent Controls")
    st.markdown("Select a predefined policy query or craft your own.")
    sample_questions = {
        "Cross-Domain Policy Analysis": "A policy advisor proposes promoting drought-tolerant crops over water-intensive crops in Andhra Pradesh. Based on historical climate and crop data, what are three compelling data-backed arguments?",
        "Climate Insight": "Summarize India's overall warming trend and compare the latest decade’s mean temperature to the 117-year baseline using IMD data.",
        "Agriculture Insight": "Provide the production comparison between Water-Intensive and Drought-Tolerant crops in Andhra Pradesh using Agriculture Ministry data."
    }
    selected_question = st.selectbox("Sample Questions", list(sample_questions.keys()))
    st.session_state.show_context = st.checkbox("Show Data Context (Traceability)", value=False)

user_query = st.text_area("💬 Ask the Samarth Policy Agent", value=sample_questions[selected_question], height=120)

if st.button("🔍 Generate Policy Insight"):
    if user_query:
        with st.spinner("Consulting datasets..."):
            answer = ask_samarth_agent(client, user_query)
        st.subheader("🧭 Policy Insight (AI-Synthesized)")
        st.markdown(answer)
        if st.session_state.show_context:
            with st.expander("View Raw Data Context"):
                st.code(f"--- IMD DATA ---\n{query_imd_data(user_query, imd_df, imd_insights)}", language='markdown')
                st.code(f"--- AGRICULTURE DATA ---\n{query_crop_data(user_query, crop_df)}", language='markdown')
    else:
        st.warning("Please enter a question.")

# --- FOOTER SECTION ---
st.markdown("""
<hr style="border: 1px solid #ccc; margin-top: 40px; margin-bottom: 10px;">

<div style="
    text-align: center; 
    color: #555; 
    font-size: 15px; 
    padding: 10px 0;
    line-height: 1.6;
">
    <span style="font-size: 13px; color: #777;">© 2025 Project Samarth | Build For Bharat Fellowship 2026 Cohort (Data Science)</span>
</div>
""", unsafe_allow_html=True)
