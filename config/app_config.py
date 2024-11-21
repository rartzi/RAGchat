import os
import json
from pathlib import Path
import streamlit as st

class AppConfig:
    def __init__(self):
        self.config = self._load_config()

        self.vector_db_root = self.config.get("vector_db_root", "./vector_databases")
        self.default_docs_dir = self.config.get("default_docs_dir", "./documents")
        self.max_context_length = self.config.get("max_context_length", 10)
        self.temperature = self.config.get("temperature", 0.2)
        self.app_title = self.config.get("app_title", "RAG Knowledge DBs Manager")
        self.logo_url = self.config.get("logo_url", "")
        self.user_avatar_url = self.config.get("user_avatar_url", "")
        self.assistant_avatar_url = self.config.get("assistant_avatar_url", "")

    def _load_config(self) -> dict:
        config_path = Path(__file__).parent / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def set_page_config(self):
        st.set_page_config(
            page_title=self.app_title,
            page_icon=self.logo_url,
            layout="wide",
            initial_sidebar_state="expanded"
        )

# Example usage
if __name__ == "__main__":
    app_config = AppConfig()
    app_config.set_page_config()
    print("Vector DB Root:", app_config.vector_db_root)
    print("Default Docs Dir:", app_config.default_docs_dir)
    print("Max Context Length:", app_config.max_context_length)
    print("Temperature:", app_config.temperature)