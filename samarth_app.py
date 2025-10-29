import streamlit as st
import pandas as pd
from google import genai
import os

# --- 1. DATA INGESTION AND KNOWLEDGE BASE SETUP ---

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
            "baseline_temp": f"The 117-year (1901-2017) historical baseline average mean temperature for India is {baseline_mean:.2f}°C.",
            "source": "India Meteorological Department (IMD) - Seasonal and Annual Min/Max Temp Series (data.gov.in)."
        }
    except:
        imd_df = pd.DataFrame()
        imd_insights = {"source": "IMD Data Missing / Error loading."}

    # --- Simulated Crop Data (Ministry of Agriculture) ---
    # This simulated data demonstrates cross-domain reasoning and the RAG logic.
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
    crop_insights = {"source": "Ministry of Agriculture & Farmers Welfare - Crop Production Statistics (data.gov.in - SIMULATED)."}

    return imd_df, imd_insights, crop_df, crop_insights

imd_df, imd_insights, crop_df, crop_insights = load_and_process_data()

# --- 2. RAG CORE FUNCTIONS: DATA RETRIEVAL (TOOL FUNCTIONS) ---

def query_imd_data(query: str, df: pd.DataFrame, insights: dict) -> str:
    """Retrieves climate data for LLM context, prioritizing trend insights for policy questions."""
    if any(k in query.lower() for k in ['warming', 'baseline', 'decade', 'climate', 'trend']):
        summary_context = "\n".join([f"- {k}: {v}" for k, v in insights.items()])
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

# --- 3. GEMINI SETUP & AGENT LOGIC ---

@st.cache_resource
def setup_gemini_client():
    """Initializes and caches the Gemini client using Streamlit secrets."""
    try:
        # SECURELY Load the key from .streamlit/secrets.toml
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
        return client
    except KeyError:
        st.error("ERROR: GEMINI_API_KEY not found in `.streamlit/secrets.toml`.")
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
        "For every claim or data point, **cite the specific source dataset** (IMD or CROP) "
        "and mention the specific numerical values from the context provided."
    )

    prompt = f"""
    SYSTEM PROMPT: {system_prompt}

    --- DATA CONTEXT 1: CLIMATE DATA (IMD) ---
    Source: {imd_insights['source']}
    {climate_context}
    ------------------------------------------

    --- DATA CONTEXT 2: AGRICULTURAL DATA (CROP PRODUCTION) ---
    Source: {crop_insights['source']}
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
        # Handle API errors gracefully
        return f"API ERROR: Could not get response from Gemini. Please try again. Details: {e}"

# --- 4. STREAMLIT FRONT-END ---

client = setup_gemini_client()

st.set_page_config(page_title="Project Samarth: Cross-Domain Q&A Agent", layout="wide")

st.title("🇮🇳 Project Samarth: Intelligent Cross-Domain Q&A Agent")
st.markdown("A RAG prototype synthesizing **IMD Climate Data** and **Ministry of Agriculture Crop Data** (Simulated).")
st.divider()

# List of sample questions for easy testing and demonstration
sample_questions = {
    "1. Cross-Domain Policy Argument (Hardest)": "A policy advisor is proposing a scheme to promote drought-tolerant crops over water-intensive crops in Andhra Pradesh. Based on historical climate and production data, what are the three most compelling data-backed arguments to support this policy?",
    "2. Climate Trend (IMD Focus)": "What is the overall warming trend in India? Compare the average mean temperature of the most recent decade to the historical 117-year baseline, citing the IMD data source.",
    "3. Agricultural Production (Crop Focus)": "List the total production (in Tonnes) for Water-Intensive crops versus Drought-Tolerant crops in Andhra Pradesh, based on the Ministry of Agriculture's data."
}

selected_question = st.selectbox(
    "Select a Sample Question or Enter Your Own:",
    list(sample_questions.keys())
)

user_query = st.text_area(
    "Your Question:",
    value=sample_questions[selected_question],
    height=100
)

if client:
    if st.button("Ask Samarth Agent"):
        if user_query:
            with st.spinner("Project Samarth is consulting the disparate datasets..."):
                answer = ask_samarth_agent(client, user_query)
            
            st.subheader("🤖 Agent Answer")
            st.markdown(answer)
            st.divider()

            # Display the data used for traceability (Core Value: Traceability)
            st.subheader("📊 Data Traceability: Context Used")
            st.markdown("This is the exact raw data snippet sent to the LLM for reasoning:")
            
            st.code(f"--- IMD DATA CONTEXT ---\n{query_imd_data(user_query, imd_df, imd_insights)}", language='markdown')
            st.code(f"--- CROP DATA CONTEXT ---\n{query_crop_data(user_query, crop_df)}", language='markdown')
        else:
            st.warning("Please enter a question.")

else:
    st.error("Agent is not active. Please ensure the Gemini client is initialized with a valid API key in `.streamlit/secrets.toml`.")
