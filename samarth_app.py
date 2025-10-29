import streamlit as st
import pandas as pd
from google import genai
import os

# --- 1. CONFIGURATION AND THEMES ---

# Set Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Project Samarth: Policy RAG Agent", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics (Themed UI)
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
    background-color: #0B5345; /* Dark Green - deep governmental feel */
    color: white;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #FF9933;
    color: white;
    font-weight: bold;
    border: none;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #FFFFFF;
    color: #0B5345;
}
/* Main Content Styling */
.stMarkdown h3 {
    color: #388E3C; /* Green tone for sections */
}
.stAlert, .stSpinner {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# --- 2. DATA INGESTION AND KNOWLEDGE BASE SETUP ---

@st.cache_data
def load_and_process_data():
    """Loads, processes, and extracts key insights from the IMD data and simulates Crop data."""
    
    # --- IMD Climate Data Processing ---
    try:
        # Load the uploaded CSV
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

    # --- Simulated Crop Data (Ministry of Agriculture) ---
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
    
    # Calculate key crop metrics for UI
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


# --- 3. RAG CORE FUNCTIONS: DATA RETRIEVAL (TOOL FUNCTIONS) ---

def query_imd_data(query: str, df: pd.DataFrame, insights: dict) -> str:
    """Retrieves climate data for LLM context, prioritizing trend insights for policy questions."""
    summary_context = "\n".join([f"- {k}: {v}" for k, v in insights.items() if k not in ['hottest_decade_mean', 'baseline_temp']])
    
    if any(k in query.lower() for k in ['warming', 'baseline', 'decade', 'climate', 'trend']):
        return f"CLIMATE SUMMARY INSIGHTS:\n{summary_context}"
    
    if not df.empty:
        # Default: provide last 5 years data
        data_markdown = df[['YEAR', 'ANNUAL - MEAN', 'ANNUAL - MIN', 'ANNUAL - MAX']].tail(5).to_markdown(index=False)
        return f"RECENT CLIMATE DATA SNIPPET:\n{data_markdown}"
    return "IMD data not available."

def query_crop_data(query: str, df: pd.DataFrame) -> str:
    """Retrieves specific crop/production data for LLM context, prioritizing State-level aggregation for policy."""
    if any(k in query.lower() for k in ['crop_type', 'drought', 'water-intensive', 'andhra pradesh', 'production']):
        state = 'Andhra Pradesh'
        df_filtered = df[df['State'] == state]
        
        # Aggregate production by crop type for direct comparison
        aggregated = df_filtered.groupby('Crop_Type')['Production_Tonnes'].sum().sort_values(ascending=False).reset_index()
        aggregated['Production_Tonnes'] = aggregated['Production_Tonnes'].apply(lambda x: f"{x:,.0f} Tonnes")
        return f"CROP DATA SNIPPET (Production by Crop Type in {state}):\n{aggregated.to_markdown(index=False)}"

    # Default: List overall top production areas
    top_producers = df.groupby(['State', 'Crop'])['Production_Tonnes'].sum().nlargest(3).reset_index()
    top_producers['Production_Tonnes'] = top_producers['Production_Tonnes'].apply(lambda x: f"{x:,.0f} Tonnes")
    return f"TOP PRODUCTION OVERVIEW:\n{top_producers.to_markdown(index=False)}"

# --- 4. GEMINI SETUP & AGENT LOGIC ---

@st.cache_resource
def setup_gemini_client():
    """Initializes and caches the Gemini client using Streamlit secrets."""
    try:
        # SECURELY Load the key from .streamlit/secrets.toml
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
    
    # Retrieval Step: Get relevant data context from both sources
    climate_context = query_imd_data(user_question, imd_df, imd_insights)
    crop_context = query_crop_data(user_question, crop_df)

    # System Prompt for Synthesis (ensures Accuracy & Traceability)
    system_prompt = (
        "You are 'Project Samarth,' an intelligent Q&A agent for the Indian Government. "
        "Your mission is to synthesize data from two separate, disparate sources: IMD (Climate) and Ministry of Agriculture (Crop Production). "
        "Your response must be professional, highly accurate, and data-backed. "
        "For every claim or data point in your response, **cite the specific source dataset** (IMD or CROP) "
        "and mention the specific numerical values from the context provided."
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
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"API ERROR: Could not get response from Gemini. Details: {e}"

# --- 5. STREAMLIT FRONT-END LAYOUT ---

client = setup_gemini_client()

st.title("🇮🇳 Project Samarth: Intelligent Cross-Domain Q&A Agent")
st.markdown("A **RAG prototype** synthesizing **IMD Climate Data** and **Ministry of Agriculture Crop Data** for policy analysis.")

if not client:
    st.stop()

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
    delta=f"Focus Area for Policy Growth"
)
st.divider()

# --- Sidebar Controls (Enhanced UI) ---
with st.sidebar:
    st.header("Project Samarth Controls")
    st.markdown("Use the sample questions to test the agent's ability to reason across IMD and Crop data.")

    # List of sample questions for easy testing
    sample_questions = {
        "1. Cross-Domain Policy Argument (Hardest)": "A policy advisor is proposing a scheme to promote drought-tolerant crops over water-intensive crops in Andhra Pradesh. Based on historical climate and production data, what are the three most compelling data-backed arguments to support this policy?",
        "2. Climate Trend (IMD Focus)": "What is the overall warming trend in India? Compare the average mean temperature of the most recent decade to the historical 117-year baseline, citing the IMD data source.",
        "3. Agricultural Production (Crop Focus)": "List the total production (in Tonnes) for Water-Intensive crops versus Drought-Tolerant crops in Andhra Pradesh, based on the Ministry of Agriculture's data."
    }

    selected_question = st.selectbox(
        "Select a Sample Question:",
        list(sample_questions.keys())
    )
    
    # Toggle for Raw Context
    st.session_state.show_context = st.checkbox("Show Raw Context for Traceability", value=False)


# --- Main Input and Output ---
user_query = st.text_area(
    "Ask the Samarth Policy Agent:",
    value=sample_questions[selected_question],
    height=100
)

if st.button("Generate Policy Analysis"):
    if user_query:
        with st.spinner("Project Samarth is consulting the disparate datasets..."):
            answer = ask_samarth_agent(client, user_query)
        
        st.subheader("🤖 Agent Answer (Synthesized Policy Brief)")
        st.markdown(answer)
        st.divider()

        # Conditional display of raw context
        if st.session_state.show_context:
            with st.expander("Expand to view Raw Context Sent to LLM (Traceability)"):
                st.markdown("This is the exact raw data snippet used by the LLM for reasoning and citation:")
                
                st.code(f"--- IMD DATA CONTEXT ---\n{query_imd_data(user_query, imd_df, imd_insights)}", language='markdown')
                st.code(f"--- CROP DATA CONTEXT ---\n{query_crop_data(user_query, crop_df)}", language='markdown')
    else:
        st.warning("Please enter a question.")
