# utils/embedding_utils.py
import os
from typing import List
import requests
from openai import OpenAI, AzureOpenAI
import streamlit as st


def get_azure_openai_client() -> AzureOpenAI:
    """Initialize Azure OpenAI client."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip('/')
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
    )

def get_openai_client() -> OpenAI:
    """Initialize OpenAI client."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(text: str, provider: str, model: str) -> List[float]:
    """Generate embeddings using specified provider and model."""
    try:
        if provider == "OpenAI":
            client = get_openai_client()
            response = client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        
        elif provider == "Azure OpenAI":
            client = get_azure_openai_client()
            response = client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
            
        else:  # Ollama
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": model, "prompt": text}
            )
            if response.status_code != 200:
                raise ValueError(f"Ollama error: {response.status_code}")
            return response.json()['embedding']
            
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None