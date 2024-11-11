# 2_document_qa.py

import os
import streamlit as st
from openai import OpenAI
import chromadb
from typing import List
import requests
from datetime import datetime

# Configuration
DB_DIR = "/Users/ronenartzi/development/playground/gui/streamlit/RAGchat/db"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection("document_embeddings")

# Model configurations
OPENAI_MODELS = {
    "GPT-4 Turbo": "gpt-4-turbo-preview",
    "GPT-3.5 Turbo": "gpt-3.5-turbo"
}

def get_ollama_models():
    """Get available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return {model['name']: model['name'] 
                for model in response.json()['models']} if response.status_code == 200 else {}
    except:
        return {}

def get_embeddings(text: str, provider: str = "openai") -> List[float]:
    """Get embeddings for text using specified provider."""
    try:
        if provider == "openai":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        else:  # ollama
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text}
            )
            return response.json()['embedding']
    except Exception as e:
        st.error(f"Error getting embeddings: {str(e)}")
        return None

def get_relevant_context(query: str, provider: str) -> str:
    """Find relevant document chunks for a query."""
    try:
        embedding = get_embeddings(query, provider)
        if not embedding:
            return "Error: Could not generate embeddings for query"
        
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3,
            include=['metadatas', 'documents']
        )
        
        if not results['documents'][0]:
            return "No relevant information found"
            
        # Combine context with source information
        contexts = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            source = meta.get('source', 'Unknown')
            contexts.append(f"From {source}:\n{doc}")
            
        return "\n\n".join(contexts)
        
    except Exception as e:
        return f"Error: {str(e)}"

def get_model_response(query: str, context: str, model: str, provider: str) -> str:
    """Get response from the selected model."""
    prompt = f"""Context: {context}

Question: {query}

Please provide an answer based on the context above. If the context doesn't contain relevant information, please say so."""

    try:
        if provider == "openai":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content
        else:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt}
            )
            return response.json()['response']
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Page setup
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("💬 Document Q&A")

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # Model selection
    provider = st.radio("Provider", ["OpenAI", "Ollama"])
    models = OPENAI_MODELS if provider == "OpenAI" else get_ollama_models()
    
    if not models:
        st.error(f"No models available for {provider}")
        st.stop()
        
    model = st.selectbox("Model", list(models.keys()))
    model_id = models[model]
    
    # Database info
    st.divider()
    try:
        results = collection.get()
        sources = set(meta['source'] for meta in results['metadatas'])
        st.success(f"📚 {len(sources)} documents available")
    except:
        st.error("No documents found")
        st.stop()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input("Ask about your documents..."):
    # Show user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get and show response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            context = get_relevant_context(query, provider.lower())
            if context.startswith("Error"):
                response = "I'm sorry, I encountered an error while searching the documents."
            else:
                response = get_model_response(query, context, model_id, provider.lower())
                
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})