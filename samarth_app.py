import streamlit as st
import pandas as pd
import requests
from google import genai
import os
import plotly.express as px
from streamlit_chat import message
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
        # API for IMD Mean Temperature Series
        imd_resource_id = "seasonal-and-annual-mean-temp-series-india-1901-2017"
        imd_url = f"https://api.data.gov.in/resource/{imd_resource_id}?api-key={api_key}&format=json&limit=2000"
        response = requests.get(imd_url)
        data = response.json()
        records = data.get('records', [])
        if not records:
            raise ValueError("No records fetched from IMD API")
        
        # Process JSON to DataFrame (assuming structure with YEAR, ANNUAL_MEAN, etc.)
        # Note: Adjust fields based on actual API response; for demo, simulate processing
        imd_df = pd.DataFrame(records)
        # Assume columns: year, annual_mean, etc. - Map accordingly
        imd_df['YEAR'] = imd_df.get('year', pd.Series(dtype=int))  # Adjust key
        imd_df['ANNUAL - MEAN'] = imd_df.get('annual_mean', pd.Series(dtype=float))
        # Calculate additional fields if needed
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
    except Exception as e:
        st.warning(f"Falling back to local IMD data due to API issue: {e}")
        # Fallback to local CSV as in original
        imd_df = pd.read_csv("Min_Max_Seasonal_IMD_2017.csv") if os.path.exists("Min_Max_Seasonal_IMD_2017.csv") else pd.DataFrame()
        # ... (original processing)
        imd_insights = {"source": f"IMD Data (Local Fallback): {e}"}
    
    # --- LIVE Crop Data Processing ---
    try:
        crop_resource_id = "district-wise-season-wise-crop-production-statistics-1997"
        crop_url = f"https://api.data.gov.in/resource/{crop_resource_id}?api-key={api_key}&format=json&limit=10000"
        response = requests.get(crop_url)
        data = response.json()
        records = data.get('records', [])
        crop_df = pd.DataFrame(records)
        # Process: Select relevant columns e.g., 'state', 'district', 'crop', 'season', 'production_tonnes'
        # Assume mapping: crop_df['State'] = crop_df['state_name'], etc.
        # Filter for Andhra Pradesh and Telangana for demo
        crop_df = crop_df[crop_df['state_name'].isin(['Andhra Pradesh', 'Telangana'])]  # Adjust column name
        crop_df['Crop_Type'] = crop_df['crop_name'].map({'Rice': 'Water-Intensive', 'Cotton': 'Drought-Tolerant', 'Maize': 'Drought-Tolerant'})  # Simple mapping
        crop_df['Production_Tonnes'] = pd.to_numeric(crop_df.get('production_tonnes', 0), errors='coerce')
        crop_metrics = {
            "source": "Ministry of Agriculture & Farmers Welfare - District-wise Crop Production (data.gov.in API)."
        }
        # Calculate metrics
        ap_production = crop_df[crop_df['state_name'] == 'Andhra Pradesh'].groupby('Crop_Type')['Production_Tonnes'].sum()
        water_intensive_prod = ap_production.get('Water-Intensive', 0)
        drought_tolerant_prod = ap_production.get('Drought-Tolerant', 0)
        crop_metrics.update({
            "water_intensive_prod": water_intensive_prod,
            "drought_tolerant_prod": drought_tolerant_prod,
            "ratio_risk": water_intensive_prod / (water_intensive_prod + drought_tolerant_prod) if (water_intensive_prod + drought_tolerant_prod) > 0 else 0,
        })
    except Exception as e:
        st.warning(f"Falling back to simulated crop data: {e}")
        # Original simulated data
        crop_data = { ... }  # Original dict
        crop_df = pd.DataFrame(crop_data)
        # ... original metrics
    
    return imd_df, imd_insights, crop_df, crop_metrics

# Load data
api_key = st.secrets.get("DATA_GOV_API_KEY", None)
imd_df, imd_insights, crop_df, crop_metrics = load_and_process_data(api_key)

# --- 3. RAG CORE FUNCTIONS: ENHANCED RETRIEVAL ---
def query_imd_data(query: str, df: pd.DataFrame, insights: dict) -> str:
    # Enhanced: More keyword matching, e.g., for rainfall if added
    keywords_trend = ['warming', 'baseline', 'decade', 'climate', 'trend', 'temperature']
    if any(k in query.lower() for k in keywords_trend):
        summary = "\n".join([f"- {k}: {v}" for k, v in insights.items()])
        return f"CLIMATE SUMMARY INSIGHTS:\n{summary}"
    
    # For specific years or comparisons
    if 'last' in query.lower() or 'recent' in query.lower():
        years = df['YEAR'].tail(10).tolist()  # Last 10 years
        snippet = df[df['YEAR'].isin(years)][['YEAR', 'ANNUAL - MEAN']].to_markdown(index=False)
        return f"RECENT CLIMATE DATA (Last 10 Years):\n{snippet}"
    
    # Default
    data_markdown = df[['YEAR', 'ANNUAL - MEAN']].tail(5).to_markdown(index=False)
    return f"CLIMATE DATA SNIPPET:\n{data_markdown}"

def query_crop_data(query: str, df: pd.DataFrame, metrics: dict) -> str:
    # Enhanced: Handle state comparisons, districts, trends
    states_keywords = ['state_x', 'state_y', 'andhra', 'telangana']  # For sample questions
    if any(s in query.lower() for s in ['andhra pradesh', 'telangana']):
        state = 'Andhra Pradesh' if 'andhra' in query.lower() else 'Telangana'
        df_filtered = df[df['State'] == state]  # Adjust column
        aggregated = df_filtered.groupby('Crop_Type')['Production_Tonnes'].sum().reset_index()
        aggregated['Production_Tonnes'] = aggregated['Production_Tonnes'].apply(lambda x: f"{x:,.0f} Tonnes")
        return f"CROP DATA FOR {state.upper()}:\n{aggregated.to_markdown(index=False)}"
    
    # For district highest/lowest
    if 'district' in query.lower() and 'highest' in query.lower():
        top_district = df.loc[df['Production_Tonnes'].idxmax(), ['District', 'Production_Tonnes']]
        return f"TOP DISTRICT: {top_district.to_dict()}"
    
    # Default aggregation
    top_crops = df.groupby('Crop')['Production_Tonnes'].sum().nlargest(5).reset_index()
    top_crops['Production_Tonnes'] = top_crops['Production_Tonnes'].apply(lambda x: f"{x:,.0f} Tonnes")
    return f"TOP CROPS OVERVIEW:\n{top_crops.to_markdown(index=False)}"

# --- 4. GEMINI SETUP & AGENT LOGIC (ENHANCED PROMPT) ---
@st.cache_resource
def setup_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
        return client
    except KeyError:
        st.error("ERROR: GEMINI_API_KEY not found in secrets.")
        return None
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        return None

def ask_samarth_agent(client, user_question: str):
    climate_context = query_imd_data(user_question, imd_df, imd_insights)
    crop_context = query_crop_data(user_question, crop_df, crop_metrics)
    
    system_prompt = (
        "You are Project Samarth, a secure Indian Government AI agent. Synthesize IMD Climate and Agriculture data accurately. "
        "Cite sources (IMD or CROP API) for every fact. Handle cross-domain queries like state comparisons, trends, policy arguments. "
        "Structure responses: Bullet points for clarity, tables for data. Ensure traceability and neutrality."
    )
    
    prompt = f"""
    {system_prompt}
    
    IMD SOURCE: {imd_insights['source']}
    {climate_context}
    
    CROP SOURCE: {crop_metrics['source']}
    {crop_context}
    
    QUESTION: {user_question}
    Respond concisely, data-driven.
    """
    
    try:
        response = client.models.generate_content(model='gemini-1.5-flash', contents=prompt)  # Updated model for better perf
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- 5. ENHANCED STREAMLIT FRONT-END: CHAT INTERFACE + DASHBOARD ---
client = setup_gemini_client()
if not client:
    st.stop()

st.title("🇮🇳 Project Samarth: Cross-Domain Policy Insights Agent")
st.markdown("**Live RAG Prototype**: Fetches from data.gov.in API for real-time agriculture-climate synthesis. Secure, traceable, deployable on-premises.")

# --- Dashboard Tab ---
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Chat Q&A"])

with tab1:
    st.subheader("Key Insights Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Warming Trend (1901-2017)", f"+{imd_insights['warming_trend'].split()[4]}°C", delta=f"Baseline: {imd_insights['baseline_temp']:.2f}°C")
    with col2:
        st.metric("Water-Intensive Prod (AP)", f"{crop_metrics['water_intensive_prod']/1000:.0f}k Tonnes", delta=f"{crop_metrics['ratio_risk']*100:.0f}% Risk")
    with col3:
        st.metric("Drought-Tolerant Prod (AP)", f"{crop_metrics['drought_tolerant_prod']/1000:.0f}k Tonnes", delta="Policy Priority")
    
    # Visualizations
    if not imd_df.empty:
        fig_temp = px.line(imd_df, x='YEAR', y='ANNUAL - MEAN', title="India Annual Mean Temperature Trend")
        st.plotly_chart(fig_temp, use_container_width=True)
    
    if not crop_df.empty:
        fig_crop = px.bar(crop_df, x='Crop', y='Production_Tonnes', color='Crop_Type', title="Crop Production by Type (AP/Telangana)")
        st.plotly_chart(fig_crop, use_container_width=True)

with tab2:
    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Enhanced Sample Questions (Matching Challenge Samples)
    sample_questions = {
        "Policy Promotion": "A policy advisor proposes promoting drought-tolerant crops over water-intensive in Andhra Pradesh. Provide 3 data-backed arguments from climate & crop data.",
        "State Comparison": "Compare average annual temperature in last 5 years for Andhra Pradesh context, and list top 3 drought-tolerant crops in Andhra vs Telangana.",
        "District Analysis": "Identify district in Andhra Pradesh with highest rice production recently, compare to lowest in Telangana.",
        "Trend Correlation": "Analyze rice production trend in Andhra Pradesh over last decade; correlate with temperature rise and summarize impact."
    }
    
    # Sidebar for samples
    with st.sidebar:
        st.header("🛠️ Controls")
        selected = st.selectbox("Quick Questions:", list(sample_questions.keys()))
        if st.button("Load Sample", key="load"):
            st.session_state.messages.append({"role": "user", "content": sample_questions[selected]})
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
        
        st.session_state.show_context = st.checkbox("🔍 Show Retrieval Context")
    
    # Chat Display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "context" in msg:
                with st.expander("Raw Context"):
                    st.code(msg["context"])
    
    # User Input
    if prompt := st.chat_input("Ask about agriculture-climate policy..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing insights from data.gov.in..."):
                answer = ask_samarth_agent(client, prompt)
                st.markdown(answer)
                context = f"IMD: {query_imd_data(prompt, imd_df, imd_insights)}\n\nCROP: {query_crop_data(prompt, crop_df, crop_metrics)}"
                st.session_state.messages[-1]["content"] = answer
                if st.session_state.show_context:
                    st.session_state.messages[-1]["context"] = context
                    with st.expander("Traceability Context"):
                        st.code(context, language="markdown")

# Footer
st.markdown("---")
st.markdown("*Built for Build For Bharat Fellowship 2026. Live data via data.gov.in API. Secure & traceable RAG architecture.*")
