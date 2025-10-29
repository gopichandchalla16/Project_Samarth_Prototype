🇮🇳 Project Samarth: Intelligent Cross-Domain Policy Agent

Live Application Link

Experience the full functionality of the RAG system here:

https://samarth-data-insights.streamlit.app/

Project Overview: Bridging Data Silos for Policy Insights

Project Samarth is an intelligent Q&A system designed to solve a critical challenge in Indian governance: the fragmentation of valuable public data across different ministries (e.g., IMD vs. Ministry of Agriculture).

Built using a Retrieval-Augmented Generation (RAG) architecture powered by the Gemini model, this prototype successfully synthesizes information from two disparate data sources to provide comprehensive, data-backed answers for policymakers.

The core mission is to demonstrate traceable, cross-domain reasoning to support climate-adaptive agricultural policy.

Core Challenge Question Demonstrated

The system is engineered to answer complex synthesis questions, which require reasoning across multiple datasets. The following test case exemplifies its capability:

"A policy advisor is proposing a scheme to promote drought-tolerant crops over water-intensive crops in Andhra Pradesh. Based on historical climate and production data, what are the three most compelling data-backed arguments to support this policy?"

Data Integration and System Architecture

The solution tackles Phase 1 (Data Discovery & Integration) and Phase 2 (Intelligent Q&A) using a robust RAG pipeline:

Disparate Data Sources: The system draws context from IMD (Climate) time-series data and a simulated Ministry of Agriculture (Crop) aggregated production dataset.

Tool Functions: Custom Python functions act as "data tools," transforming messy, complex dataframes into targeted, structured context (Markdown tables).

Gemini Synthesis: The Gemini model receives the user's question along with the relevant, structured data from both sources. Its role is strictly to synthesize this information into a coherent policy brief.

lignment with Build For Bharat Evaluation Criteria

This solution directly addresses the core requirements of the Data Science challenge for the Fellowship by focusing on Problem Solving, System Architecture, and Traceability over complex, real-world data sources.

Criterion

Implementation Strategy & Fulfillment

Problem Solving & Initiative

Programmatically accessed and harmonized two fundamentally inconsistent datasets. The system determines which tool to query and how to normalize the resulting snippets for the LLM.

System Architecture

Developed an end-to-end RAG pipeline where dedicated Python Tool Functions handle data retrieval, demonstrating robust handling of disparate data structures.

Accuracy & Traceability

The Gemini System Prompt enforces a strict citation rule: Every claim must cite the specific source dataset (IMD or CROP) and include the numerical evidence found in the retrieved context.


