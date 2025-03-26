import os
import logging
import pandas as pd
from typing import Any  
import streamlit as st
from src.document_processor import DocumentProcessor
from src.gemini_agent import GeminiAgent
from src.sql_agent import SQLAgent
import time
import re

print(f"✅ Imported Any successfully: {Any}")

# Disable Streamlit's file watcher completely
os.environ["STREAMLIT_WATCH_FILE"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
print("✅ Streamlit is running...") 

def main():
    st.set_page_config(
        page_title="Document QA Agent", 
        page_icon="🔍",
        layout="wide"
    )
    
    # Apply custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 1rem;
        }
        .subheader {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2563EB;
            margin-top: 1rem;
        }
        .card {
            background-color: #F9FAFB;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .success-text {
            color: #10B981;
            font-weight: 600;
        }
        .warning-text {
            color: #F59E0B;
            font-weight: 600;
        }
        .error-text {
            color: #EF4444;
            font-weight: 600;
        }
        .info-text {
            color: #3B82F6;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<div class="main-header">🕵️ Document QA Agent - Extractor</div>', unsafe_allow_html=True)
    st.markdown("Extract insights, generate SQL, and analyze documents with AI")
    
    print("✅ Streamlit UI Components Loaded...")

    try:
        # Initialize session state variables
        for key in ["result", "extracted_sql", "sql_executed"]:
            if key not in st.session_state:
                st.session_state[key] = None if key != "sql_executed" else False

        # Initialize agents in sidebar
        with st.sidebar:
            st.markdown('<div class="subheader">🔧 Configuration</div>', unsafe_allow_html=True)
            
            with st.expander("Agent Status", expanded=True):
                if "processor" not in st.session_state:
                    st.session_state.processor = DocumentProcessor()
                st.markdown('<span class="success-text">✅ Document Processor ready</span>', unsafe_allow_html=True)

                if "gemini_agent" not in st.session_state:
                    st.session_state.gemini_agent = GeminiAgent()
                if hasattr(st.session_state.gemini_agent, 'model') and st.session_state.gemini_agent.model is not None:
                    st.markdown('<span class="success-text">✅ Gemini Agent ready</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="error-text">❌ Gemini Agent failed</span>', unsafe_allow_html=True)

                if "sql_agent" not in st.session_state:
                    try:
                        st.session_state.sql_agent = SQLAgent('example.db')
                        st.write(f"📂 Connected Database: {st.session_state.sql_agent.db_path}")
                        st.markdown('<span class="success-text">✅ SQL Agent ready</span>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<span class="error-text">❌ SQL Agent failed: {str(e)}</span>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])
        processor = st.session_state.processor
        agent = st.session_state.gemini_agent
        sql_agent = st.session_state.sql_agent

        if not hasattr(agent, 'model') or agent.model is None:
            st.error("⚠️ Gemini Agent failed to initialize. Check API key and configuration in the sidebar.")
            return

        with col1:
            uploaded_file = st.file_uploader("Upload PDF or CSV Document", type=['pdf', 'csv'])
            query = st.text_area("Enter your specific question about the document:")

            analyze_btn = st.button("🚀 Analyze Document", type="primary", disabled=not (uploaded_file and query))

            if uploaded_file and query and analyze_btn:
                try:
                    st.session_state.result = None
                    st.session_state.extracted_sql = None
                    st.session_state.sql_executed = False

                    file_extension = uploaded_file.name.split(".")[-1].lower()
                    if file_extension == 'csv':
                        document_data = processor.read_csv(uploaded_file)
                    elif file_extension == 'pdf':
                        document_data = processor.read_pdf(uploaded_file)
                    else:
                        st.error("Unsupported file format.")
                        return

                    if not document_data:
                        st.warning("Could not extract data from the document")
                        return

                    response = agent.analyze_document(document_data, query, processor)

                    if response:
                        st.session_state.result = response
                        st.session_state.extracted_sql = None
                        
                        sql_match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
                        sql_query = sql_match.group(1).strip() if sql_match else None

                        if sql_query and file_extension == 'csv':
                            st.session_state.extracted_sql = sql_query
                            try:
                                with st.spinner("Executing query..."):
                                    result = sql_agent.execute_query(st.session_state.extracted_sql)
                                    st.session_state.sql_executed = True
                                    st.success("Query executed successfully")
                                    st.write("Results:")
                                    st.dataframe(result)
                            except Exception as e:
                                st.error(f"SQL Execution Error: {str(e)}")
                        elif sql_query and file_extension == 'pdf':
                            st.warning("SQL execution is disabled for PDFs.")
                    else:
                        st.error("No response received from the analysis.")
                except Exception as e:
                    st.error(f"Document Processing Error: {str(e)}")

        with col2:
            if st.session_state.result:
                st.markdown("### 📝 Analysis Result")
                st.markdown(st.session_state.result)
    
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")

if __name__ == "__main__":
    main()
