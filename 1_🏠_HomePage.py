import streamlit as st
import requests

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Base Manager",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        color: #1E3D59;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .feature-header {
        color: #1E3D59;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .feature-text {
        color: #495057;
        font-size: 1rem;
    }
    .providers-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 2rem 0;
    }
    .provider-badge {
        background-color: #e9ecef;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">🏠 Welcome to RAG Knowledge Base Manager</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
This application helps you build and manage your RAG (Retrieval-Augmented Generation) knowledge base. 
Upload your documents, manage your database, and query your knowledge base with ease.
""")

# Main Features Section
col1, col2 = st.columns(2)

with col1:
    # Database Management Section
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('### 📚 Database Management')
    st.markdown("""
    - Upload and process documents
    - Support for multiple file formats
    - Organize documents with metadata
    - Monitor database statistics
    - Manage embeddings and chunks
    """)
    
    # Quick Start button
    #st.button("🚀 Go to Database Manager", use_container_width=True)
    #st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Query Interface Section
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('### 🔍 Knowledge Base Query')
    st.markdown("""
    - Natural language queries
    - Context-aware responses
    - TBD: Citation of sources
    - TBD: Adjustable search parameters
    - TBD: Export query results
    """)
    
    # Query Interface button
    #st.button("💡 Go to Query Interface", use_container_width=True)
    #st.markdown('</div>', unsafe_allow_html=True)

# Supported Providers Section
st.markdown('<div class="providers-section">', unsafe_allow_html=True)
st.markdown('### 🔧 Supported Providers')

tab1, tab2 = st.tabs(["Embedding Providers", "LLM Providers"])

with tab1:
    st.markdown("#### Available Embedding Providers")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **OpenAI**
        - text-embedding-3-large
        - text-embedding-ada-002
        """)
    
    with col2:
        st.markdown("""
        **Azure OpenAI**
        - text-embedding-ada-002
        - Custom deployments
        """)
    
    with col3:
        st.markdown("""
        **Ollama**
        - nomic-embed-text
        - Custom models
        """)

with tab2:
    st.markdown("#### Available LLM Providers")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **OpenAI**
        - GPT-4 Turbo
        - GPT-4
        - GPT-3.5 Turbo
        """)
    
    with col2:
        st.markdown("""
        **Azure OpenAI**
        - GPT-4
        - GPT-3.5 Turbo
        - Custom deployments
        """)
    
    with col3:
        st.markdown("""
        **Ollama**
        - Llama 2
        - Mistral
        - Custom models
        """)

st.markdown('</div>', unsafe_allow_html=True)

# Quick Start Guide
st.markdown("### 🚀 Quick Start Guide")
tabs = st.tabs(["1. Setup", "2. Upload Documents", "3. Query"])

with tabs[0]:
    st.markdown("""
    1. Select your preferred embedding provider
    2. Choose the appropriate model
    3. Configure API keys if required
    4. Test your connection
    """)

with tabs[1]:
    st.markdown("""
    1. Navigate to the Database Manager
    2. Upload your documents
    3. Review processing status
    4. Verify embeddings
    """)

with tabs[2]:
    st.markdown("""
    1. Go to Query Interface
    2. Enter your question
    3. Review sources and citations
    4. Export results if needed
    """)

# System Status
#st.sidebar.markdown("### 💻 System Status")
#status_container = st.sidebar.container()
#with status_container:
#    st.markdown("**Database Status:** 🟢 Online")
#    st.markdown("**API Status:** 🟢 Connected")
#    st.markdown("**Documents:** 150")
#    st.markdown("**Total Chunks:** 1,250")
#    st.markdown("**Last Updated:** 2024-11-12 20:46:47")

# Help & Support
#with st.sidebar:
#    st.markdown("### ❓ Help & Support")
#    st.markdown("""
#    - [Documentation](https://docs.example.com)
#    - [Report an Issue](https://github.com/example/issues)
#    - [Request a Feature](https://github.com/example/features)
#    """)