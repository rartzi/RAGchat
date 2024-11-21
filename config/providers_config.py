import os
import json
import requests
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

class ProvidersConfig:
    def __init__(self):
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)
        self.config = self._load_config()
        self.enabled_providers = self._detect_enabled_providers()
        self.models = self._initialize_models()

    def _load_config(self) -> Dict:
        config_path = Path(__file__).parent / 'config.json'
        with open(config_path, 'r') as f:
            return json.load(f)

    def _detect_enabled_providers(self) -> Dict[str, bool]:
        """Detect which providers are both configured and available"""
        # First check which providers are enabled in config
        config_enabled = {
            "OpenAI": self.config["providers"]["openai"]["enabled"],
            "Azure OpenAI": self.config["providers"]["azure"]["enabled"],
            "Ollama": self.config["providers"]["ollama"]["enabled"]
        }
        
        # Then check if enabled providers are actually available
        ollama_available = self._check_ollama_available() if config_enabled["Ollama"] else False
        
        available_providers = {
            "OpenAI": bool(os.getenv("OPENAI_API_KEY")) if config_enabled["OpenAI"] else False,
            "Azure OpenAI": all([
                os.getenv("AZURE_OPENAI_API_KEY"),
                os.getenv("AZURE_OPENAI_ENDPOINT"),
                os.getenv("AZURE_OPENAI_API_VERSION")
            ]) if config_enabled["Azure OpenAI"] else False,
            "Ollama": ollama_available
        }
        
        return available_providers

    def _check_ollama_available(self) -> bool:
        try:
            response = requests.get("http://localhost:11434/", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def _get_ollama_models(self) -> Dict[str, str]:
        try:
            response = requests.get("http://localhost:11434/api/list")
            if response.status_code != 200:
                return {model: model for model in self.config["ollama_models"]}
            
            data = response.json()
            if not data or "models" not in data:
                return {model: model for model in self.config["ollama_models"]}

            models = set()
            for model in data.get("models", []):
                name = model.get("name", "")
                if name and ":" not in name and "instruct" not in name.lower():
                    if any(base in name.lower() for base in self.config["ollama_models"]):
                        models.add(name)
            
            if not models:
                models = set(self.config["ollama_models"])
                
            return {model: model for model in sorted(models)}
        except Exception:
            return {model: model for model in self.config["ollama_models"]}

    def _get_azure_deployments(self) -> Dict[str, str]:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        return self.config["default_azure_deployments"]
    
        if not all([endpoint, api_key, api_version]):
            return self.config["default_azure_deployments"]

        try:
            headers = {"api-key": api_key}
            response = requests.get(
                f"{endpoint}/openai/deployments?api-version={api_version}",
                headers=headers
            )
            if response.status_code == 200:
                deployments = {}
                for deployment in response.json().get("data", []):
                    name = deployment.get("id")
                    model = deployment.get("model", {}).get("name")
                    if name and model:
                        deployments[model] = name
                return deployments if deployments else self.config["default_azure_deployments"]
        except:
            return self.config["default_azure_deployments"]

    def _initialize_models(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        models = {
            "OpenAI": self.config["openai_models"],
            "Azure OpenAI": {
                "chat": self._get_azure_deployments(),
                "embedding": self.config["azure_openai_models"]["embedding"]
            },
            "Ollama": {
                "chat": self.config["ollama_models"]["chat"],
                "embedding": self.config["ollama_models"]["embedding"]
            }
        }
        
        enabled = self.enabled_providers
        filtered_models = {k: v for k, v in models.items() if self.enabled_providers.get(k)}
        
        return filtered_models

    def get_available_providers(self) -> List[str]:
        """Return list of providers that are both enabled in config and available"""
        available = []
        display_names = {
            "OpenAI": self.config["providers"]["openai"]["display_name"],
            "Azure OpenAI": self.config["providers"]["azure"]["display_name"],
            "Ollama": self.config["providers"]["ollama"]["display_name"]
        }
        
        for provider, is_enabled in self.enabled_providers.items():
            if is_enabled:
                # Use display name from config if available, otherwise use provider name
                display_name = display_names.get(provider, provider)
                available.append(display_name)
        
        return available

    def get_chat_models(self, provider: str) -> Dict[str, str]:
        return self.models.get(provider, {}).get("chat", {})

    def get_embedding_models(self, provider: str) -> Dict[str, str]:
        return self.models.get(provider, {}).get("embedding", {})

    def is_compatible_embedding(self, provider: str, model: str) -> bool:
        if model == "nomic-embed-text":
            return True
        embedding_models = self.get_embedding_models(provider)
        return model in embedding_models.values()

    def validate_chat_model(self, provider: str, model: str) -> bool:
        chat_models = self.get_chat_models(provider)
        return model in chat_models.values()
    
    def get_provider_clients(self, provider: str, model_name: str, temperature = None):
        try:
            if temperature is None:
                temperature = self.config["temperature"]
            match provider:
                case "OpenAI":
                    return ChatOpenAI(
                        model_name=model_name, 
                        streaming=True, 
                        temperature=temperature
                    )
                case "Azure OpenAI":
                    return AzureChatOpenAI(
                        deployment_name=model_name,
                        model_name=model_name,
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                        streaming=True,
                        temperature=temperature
                )
                case "Ollama":
                    return ChatOllama(
                        model=model_name, 
                        streaming=True, 
                        temperature=temperature
                    )
        except Exception as e:
            print(f"Error initializing provider clients: {str(e)}")
            return None
        
    @staticmethod
    def validate_environment(provider: str) -> bool:
        if provider == "OpenAI":
            return bool(os.getenv("OPENAI_API_KEY"))
        elif provider == "Azure OpenAI":
            return all([
                os.getenv("AZURE_OPENAI_API_KEY"),
                os.getenv("AZURE_OPENAI_ENDPOINT"),
                os.getenv("AZURE_OPENAI_API_VERSION")
            ])
        return True  # Ollama doesn't require env vars