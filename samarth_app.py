import streamlit as st
import pandas as pd
import requests
from google import genai
import os
import json

# --- 1. CONFIGURATION AND THEMES ---
st.set_page_config(
    page_title="Project Samarth: Policy RAG Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced aesthetics (Themed UI with better responsiveness)
st.markdown("""
<style>
/* Header & Title Styling */
.stApp > header {
    background-color: transparent;
}
.stTitle {
    color: #FF9933; /* Saffron/Orange tone for emphasis */
    font-weight: 700;
    text-shadow: 2px 2px 4px #000000;
}
/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B5345 0%, #1B5E20 100%); /* Gradient for depth */
    color: white;
    padding: 1rem;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #FF9933;
    color: white;
    font-weight: bold;
    border: none;
    transition: all 0.3s ease;
    border-radius: 20px;
    padding: 0.5rem 1rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #FFFFFF;
    color: #0B5345;
    transform: scale(1.05);
}
/* Main Content Styling */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #388E3C; /* Green tone for sections */
}
.stAlert, .stSpinner {
    border-radius: 12px;
    border-left: 4px solid #FF9933;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.user-message {
    background-color: #E3F2FD;
    text-align: right;
}
.assistant-message {
    background-color: #F1F8E9;
}
/* Responsive columns */
@media (max-width: 768px) {
    .stColumns > div {
        width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

# --- 2. DATA INGESTION AND KNOWLEDGE BASE SETUP (ENHANCED WITH LIVE FETCHING) ---
@st.cache_data(ttl=3600)  # Cache for 1 hour, but refreshable
def load_and_process_data(api_key: str = None):
    """Loads, processes, and extracts key insights from LIVE data.gov.in API."""
    
    # --- LIVE IMD Climate Data Processing ---
    try:
        # API for IMD Mean Temperature Series (Note: Use actual resource ID; this is placeholder)
        imd_resource_id = "seasonal-and-annual-mean-temp-series-india-1901-2017"  # Example ID
        imd_url = f"https://api.data.gov.in/resource/{imd_resource_id}?api-key={api_key}&format=json&limit=2000"
        response = requests.get(imd_url, timeout=10)
        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code}")
        data = response.json()
        records = data.get('records', [])
        if not records:
            raise ValueError("No records fetched from IMD API")
        
        # Process JSON to DataFrame (adjust based on actual API structure)
        imd_df = pd.json_normalize(records)  # Flatten nested JSON if needed
        # Map columns (customize based on API response; fallback to sample structure)
        if 'year' in imd_df.columns:
            imd_df['YEAR'] = pd.to_numeric(imd_df['year'])
        if 'annual_mean' in imd_df.columns:
            imd_df['ANNUAL - MEAN'] = pd.to_numeric(imd_df['annual_mean'])
        else:
            # Simulate if API fields differ
            imd_df['ANNUAL - MEAN'] = (pd.to_numeric(imd_df.get('annual_min', 0)) + pd.to_numeric(imd_df.get('annual_max', 0))) / 2
        
        # Ensure required columns
        if imd_df.empty or 'YEAR' not in imd_df.columns:
            raise ValueError("Invalid data structure from IMD API")
        
        imd_df['Decade'] = (imd_df['YEAR'] // 10) * 10
        decadal_avg = imd_df.groupby('Decade')['ANNUAL - MEAN'].mean().reset_index()
        warming_increase = decadal_avg.iloc[-1]['ANNUAL - MEAN'] - decadal_avg.iloc[0]['ANNUAL - MEAN']
        baseline_mean = imd_df['ANNUAL - MEAN'].mean()
        imd_insights = {
            "warming_trend": f"The Annual Mean Temperature has increased by {warming_increase:.2f}°C from the early 1900s to the 2010s.",
            "hottest_decade_mean": decadal_avg.iloc[-1]['ANNUAL - MEAN'],
            "baseline_temp": baseline_mean,
            "source": "India Meteorological Department (IMD) - Seasonal and Annual Mean Temperature Series (data.gov.in API)."
        }
        st.success("IMD data fetched live from data.gov.in API!")
    except Exception as e:
        st.warning(f"Falling back to local IMD data due to API issue: {e}")
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
                "source": "India Meteorological Department (IMD) - Seasonal and Annual Min/Max Temp Series (Local CSV Fallback)."
            }
            st.info("Using local IMD CSV for demo.")
        except Exception as fallback_e:
            st.error(f"IMD data load failed: {fallback_e}")
            imd_df = pd.DataFrame()
            imd_insights = {"source": "IMD Data Unavailable."}
    
    # --- LIVE Crop Data Processing ---
    try:
        crop_resource_id = "district-wise-season-wise-crop-production-statistics-1997"  # Example ID
        crop_url = f"https://api.data.gov.in/resource/{crop_resource_id}?api-key={api_key}&format=json&limit=10000"
        response = requests.get(crop_url, timeout=10)
        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code}")
        data = response.json()
        records = data.get('records', [])
        crop_df = pd.json_normalize(records)
        # Map columns (customize; assume common fields like state_name, district_name, crop_name, production)
        if 'state_name' in crop_df.columns:
            crop_df['State'] = crop_df['state_name']
        if 'district_name' in crop_df.columns:
            crop_df['District'] = crop_df['district_name']
        if 'crop_name' in crop_df.columns:
            crop_df['Crop'] = crop_df['crop_name']
        if 'production' in crop_df.columns:
            crop_df['Production_Tonnes'] = pd.to_numeric(crop_df['production'], errors='coerce')
        else:
            crop_df['Production_Tonnes'] = pd.to_numeric(crop_df.get('production_tonnes', 0), errors='coerce')
        
        # Filter for demo states and add Crop_Type (simple mapping)
        demo_states = ['Andhra Pradesh', 'Telangana']
        crop_df = crop_df[crop_df['State'].isin(demo_states)].dropna(subset=['Production_Tonnes'])
        crop_type_map = {'Rice': 'Water-Intensive', 'Cotton': 'Drought-Tolerant', 'Maize': 'Drought-Tolerant', 'Wheat': 'Water-Intensive'}
        crop_df['Crop_Type'] = crop_df['Crop'].map(crop_type_map).fillna('Other')
        
        crop_metrics = {
            "source": "Ministry of Agriculture & Farmers Welfare - District-wise Crop Production (data.gov.in API)."
        }
        # Calculate metrics for AP
        ap_df = crop_df[crop_df['State'] == 'Andhra Pradesh']
        ap_production = ap_df.groupby('Crop_Type')['Production_Tonnes'].sum()
        water_intensive_prod = ap_production.get('Water-Intensive', 0)
        drought_tolerant_prod = ap_production.get('Drought-Tolerant', 0)
        total_prod = water_intensive_prod + drought_tolerant_prod
        crop_metrics.update({
            "water_intensive_prod": water_intensive_prod,
            "drought_tolerant_prod": drought_tolerant_prod,
            "ratio_risk": water_intensive_prod / total_prod if total_prod > 0 else 0,
        })
        st.success("Crop data fetched live from data.gov.in API!")
    except Exception as e:
        st.warning(f"Falling back to simulated crop data: {e}")
        # Original simulated data as fallback
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
        total_prod = water_intensive_prod + drought_tolerant_prod
        crop_metrics = {
            "water_intensive_prod": water_intensive_prod,
            "drought_tolerant_prod": drought_tolerant_prod,
            "ratio_risk": water_intensive_prod / total_prod if total_prod > 0 else 0,
            "source": "Ministry of Agriculture & Farmers Welfare - Crop Production Statistics (Simulated Fallback)."
        }
        st.info("Using simulated crop data for demo.")
    
    return imd_df, imd_insights, crop_df, crop_metrics

# Load data
api_key = st.secrets.get("DATA_GOV_API_KEY", None)  # User needs to add this secret (free from data.gov.in)
imd_df, imd_insights, crop_df, crop_metrics = load_and_process_data(api_key)

# --- 3. RAG CORE FUNCTIONS: ENHANCED RETRIEVAL ---
def query_imd_data(query: str, df: pd.DataFrame, insights: dict) -> str:
    """Retrieves climate data for LLM context, prioritizing trend insights for policy questions."""
    keywords_trend = ['warming', 'baseline', 'decade', 'climate', 'trend', 'temperature', 'rainfall']
    if any(k in query.lower() for k in keywords_trend):
        summary_context = "\n".join([f"- {k}: {v}" for k, v in insights.items()])
        return f"CLIMATE SUMMARY INSIGHTS:\n{summary_context}"
    
    # For recent/recent years
    if 'last' in query.lower() or 'recent' in query.lower():
        n_years = 10
        recent_df = df.nlargest(n_years, 'YEAR')[['YEAR', 'ANNUAL - MEAN', 'ANNUAL - MIN', 'ANNUAL - MAX']]
        data_markdown = recent_df.to_markdown(index=False)
        return f"RECENT CLIMATE DATA (Last {n_years} Years):\n{data_markdown}"
    
    # Default: last 5 years
    if not df.empty:
        data_markdown = df[['YEAR', 'ANNUAL - MEAN', 'ANNUAL - MIN', 'ANNUAL - MAX']].tail(5).to_markdown(index=False)
        return f"CLIMATE DATA SNIPPET:\n{data_markdown}"
    return "IMD data not available."

def query_crop_data(query: str, df: pd.DataFrame, metrics: dict) -> str:
    """Retrieves specific crop/production data for LLM context, prioritizing State-level aggregation for policy."""
    # Enhanced for sample questions: states, districts, trends
    if any(state in query.lower() for state in ['andhra pradesh', 'telangana']):
        state = next((s for s in ['Andhra Pradesh', 'Telangana'] if s.lower() in query.lower()), 'Andhra Pradesh')
        df_filtered = df[df['State'] == state]
        if df_filtered.empty:
            return "No data for specified state."
        aggregated = df_filtered.groupby(['Crop_Type', 'Crop'])['Production_Tonnes'].sum().reset_index()
        aggregated['Production_Tonnes'] = aggregated['Production_Tonnes'].apply(lambda x: f"{x:,.0f} Tonnes")
        return f"CROP DATA FOR {state.upper()} (Aggregated by Type/Crop):\n{aggregated.to_markdown(index=False)}"
    
    if 'district' in query.lower():
        if 'highest' in query.lower() or 'top' in query.lower():
            top = df.nlargest(1, 'Production_Tonnes')[['District', 'Crop', 'Production_Tonnes']]
            return f"HIGHEST PRODUCTION DISTRICT: {top.to_dict('records')[0]}"
        if 'lowest' in query.lower():
            bottom = df.nsmallest(1, 'Production_Tonnes')[['District', 'Crop', 'Production_Tonnes']]
            return f"LOWEST PRODUCTION DISTRICT: {bottom.to_dict('records')[0]}"
    
    if 'trend' in query.lower() or 'decade' in query.lower():
        if 'year' in df.columns:
            trend = df.groupby('Year')['Production_Tonnes'].sum().reset_index()
            trend['Production_Tonnes'] = trend['Production_Tonnes'].apply(lambda x: f"{x:,.0f} Tonnes")
            return f"PRODUCTION TREND OVER YEARS:\n{trend.to_markdown(index=False)}"
    
    # Default: top producers
    top_producers = df.groupby(['State', 'Crop'])['Production_Tonnes'].sum().nlargest(5).reset_index()
    top_producers['Production_Tonnes'] = top_producers['Production_Tonnes'].apply(lambda x: f"{x:,.0f} Tonnes")
    return f"TOP PRODUCTION OVERVIEW:\n{top_producers.to_markdown(index=False)}"

# --- 4. GEMINI SETUP & AGENT LOGIC (ENHANCED PROMPT) ---
@st.cache_resource
def setup_gemini_client():
    """Initializes and caches the Gemini client using Streamlit secrets."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
        return client
    except KeyError:
        st.error("ERROR: GEMINI_API_KEY not found in `.streamlit/secrets.toml`. Please configure your secrets.")
        return None
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        return None

def ask_samarth_agent(client: genai.Client, user_question: str) -> str:
    """The core RAG function: Retrieval -> Generation (LLM Synthesis)."""
    # Retrieval Step
    climate_context = query_imd_data(user_question, imd_df, imd_insights)
    crop_context = query_crop_data(user_question, crop_df, crop_metrics)
    
    # Enhanced System Prompt for better synthesis, accuracy, and traceability
    system_prompt = (
        "You are 'Project Samarth,' an intelligent Q&A agent for the Indian Government. "
        "Your mission is to synthesize data from two separate, disparate sources: IMD (Climate) and Ministry of Agriculture (Crop Production). "
        "Your response must be professional, highly accurate, and data-backed. "
        "For every claim or data point in your response, **cite the specific source dataset** (IMD or CROP) "
        "and mention the specific numerical values from the context provided. "
        "Structure responses with bullet points or tables for clarity. Handle cross-domain queries like comparisons, trends, and policy arguments."
    )
    prompt = f"""
    SYSTEM PROMPT: {system_prompt}
    --- DATA CONTEXT 1: CLIMATE DATA (IMD) ---
    Source: {imd_insights['source']}
    {climate_context}
    ------------------------------------------
    --- DATA CONTEXT 2: AGRICULTURAL DATA (CROP PRODUCTION) ---
    Source: {crop_metrics['source']}
    {crop_context}
    ----------------------------------------------------------
    USER QUESTION: {user_question}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',  # Updated to a stable model
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"API ERROR: Could not get response from Gemini. Details: {e}"

# --- 5. ENHANCED STREAMLIT FRONT-END: CHAT INTERFACE + DASHBOARD ---
client = setup_gemini_client()
if not client:
    st.stop()

st.title("🇮🇳 Project Samarth: Intelligent Cross-Domain Q&A Agent")
st.markdown("A **RAG prototype** synthesizing **IMD Climate Data** and **Ministry of Agriculture Crop Data** for policy analysis. Live fetching from data.gov.in where possible.")

# --- Display Key Metrics (Policy Justification at a Glance) ---
st.subheader("📊 Key Policy Indicators (Andhra Pradesh)")
col1, col2, col3 = st.columns(3)
col1.metric(
    label="Historical Warming Trend (1901-2017)",
    value=f"+{imd_insights['warming_trend'].split()[4]}",
    delta=f"Baseline: {imd_insights['baseline_temp']:.2f}°C"
)
col2.metric(
    label="Water-Intensive Production Risk",
    value=f"{crop_metrics['water_intensive_prod'] / 1000:,.0f}k Tonnes",
    delta_color="inverse",
    delta=f"{crop_metrics['ratio_risk'] * 100:.0f}% of Total AP Production"
)
col3.metric(
    label="Drought-Tolerant Production",
    value=f"{crop_metrics['drought_tolerant_prod'] / 1000:,.0f}k Tonnes",
    delta_color="normal",
    delta="Focus Area for Policy Growth"
)
st.divider()

# --- Visualizations using native Streamlit charts (no external deps) ---
if not imd_df.empty:
    st.subheader("🌡️ Temperature Trend Visualization")
    temp_chart_data = imd_df.set_index('YEAR')['ANNUAL - MEAN']
    st.line_chart(temp_chart_data)

if not crop_df.empty:
    st.subheader("🌾 Crop Production by Type Visualization")
    ap_crop_data = crop_df[crop_df['State'] == 'Andhra Pradesh'].groupby('Crop_Type')['Production_Tonnes'].sum().reset_index()
    st.bar_chart(ap_crop_data.set_index('Crop_Type'))

# --- Enhanced Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("🛠️ Project Samarth Controls")
    st.markdown("Select sample questions matching the challenge samples to test cross-domain reasoning.")
    
    # Enhanced sample questions aligned with challenge
    sample_questions = {
        "1. Policy Argument (Cross-Domain)": "A policy advisor is proposing a scheme to promote drought-tolerant crops (Crop_Type_A) over water-intensive crops (Crop_Type_B) in Andhra Pradesh (Geographic_Region_Y). Based on historical data from the last 10 years, what are the three most compelling data-backed arguments to support this policy? Synthesize from climate and agricultural sources.",
        "2. State Comparison": "Compare the average annual temperature in Andhra Pradesh and Telangana contexts for the last 5 available years. In parallel, list the top 3 most produced drought-tolerant crops (Crop_Type_C) in each of those states during the same period, citing all data sources.",
        "3. District Identification": "Identify the district in Andhra Pradesh (State_X) with the highest production of Rice (Crop_Z) in the most recent year available and compare that with the district with the lowest production of Rice (Crop_Z) in Telangana (State_Y).",
        "4. Trend Analysis": "Analyze the production trend of water-intensive crops (Crop_Type_C) in Andhra Pradesh (Geographic_Region_Y) over the last decade. Correlate this trend with the corresponding temperature data for the same period and provide a summary of the apparent impact."
    }
    selected_question = st.selectbox(
        "Select a Sample Question:",
        list(sample_questions.keys())
    )
    
    if st.button("Load Sample Question"):
        st.session_state.messages.append({"role": "user", "content": sample_questions[selected_question]})
        st.rerun()
    
    st.checkbox("🔍 Show Raw Context for Traceability", key="show_context")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Chat Display Loop
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "context" in message:
            with st.expander("🔍 Raw Retrieval Context (Traceability)"):
                st.code(message["context"], language="markdown")

# User Input
if prompt := st.chat_input("Ask the Samarth Policy Agent (e.g., policy questions on agriculture-climate):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Project Samarth is synthesizing cross-domain insights..."):
            answer = ask_samarth_agent(client, prompt)
            context = f"--- IMD CONTEXT ---\n{query_imd_data(prompt, imd_df, imd_insights)}\n\n--- CROP CONTEXT ---\n{query_crop_data(prompt, crop_df, crop_metrics)}"
            full_response = {"role": "assistant", "content": answer, "context": context if st.session_state.show_context else None}
            st.markdown(answer)
            if st.session_state.show_context:
                with st.expander("🔍 Raw Context Sent to LLM"):
                    st.code(context, language="markdown")
            st.session_state.messages.append(full_response)

# Footer
st.markdown("---")
st.markdown("*Prototype for Build For Bharat Fellowship - 2026 Cohort (Data Science). Features: Live API fetching, Native charts, Secure secrets, Traceable RAG. Deploy with `streamlit run samarth_app.py`. Requirements: streamlit, pandas, requests, google-generativeai.*")
