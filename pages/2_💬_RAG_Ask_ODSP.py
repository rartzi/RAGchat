import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.db_utils import init_chromadb
from utils.embedding_utils import get_embeddings
from utils.db_manager_utils import VectorDBManager
from config import providers, app_config

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.callbacks import StreamlitCallbackHandler

class RAGChatUI:
    def __init__(self, db_manager: VectorDBManager):
        self.db_manager = db_manager
        self.setup_session_state()
        
    def setup_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_model" not in st.session_state:
            st.session_state.chat_model = None
        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = True
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.2
        if "current_model_config" not in st.session_state:
            st.session_state.current_model_config = None
        if "boost_astrazeneca" not in st.session_state:
            st.session_state.boost_astrazeneca = True
        if "boost_factors" not in st.session_state:
            st.session_state.boost_factors = {
                "astrazeneca_base_boost": 1.5,
                "astrazeneca_strategic_boost": 1.2,
                "recency_boost": 1.3,
                "priority_source_boost": 1.2
            }

    def should_reinitialize_model(self, provider: str, model_name: str, temperature: float) -> bool:
        """Check if the model needs to be reinitialized based on config changes."""
        current_config = st.session_state.get("current_model_config")
        new_config = {
            "provider": provider,
            "model_name": model_name,
            "temperature": temperature
        }
        
        if current_config != new_config:
            st.session_state.current_model_config = new_config
            return True
        return False

    def setup_page_config(self):
        st.set_page_config(layout="wide")
        
        st.markdown("""
        <style>
        /* Main container padding */
        .main { 
            padding-bottom: 160px;
        }
        
        /* Sidebar styles */
        .sidebar-content {
            padding: 1rem;
        }
        
        .sidebar-section {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .provider-disabled {
            color: #6c757d;
            cursor: not-allowed;
        }
        
        /* Footer styling */
        .sidebar .sidebar-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 22%;
            padding: 1rem;
            background-color: #f8f9fa;
            border-top: 1px solid #dee2e6;
            text-align: center;
            z-index: 99;
        }
        
        /* Chat input container */
        div[data-testid="stVerticalBlock"] > div:has(div.stChatInputContainer) {
            position: fixed;
            bottom: 0;
            left: 22%;
            right: 0;
            background: white;
            border-top: 1px solid #ddd;
            padding: 1rem 5rem;
            z-index: 100;
        }
        
        /* Chat input styling */
        .stChatInputContainer {
            max-width: none !important;
            padding: 0 !important;
        }
        
        .stChatInput {
            max-width: none !important;
            width: 100% !important;
            padding: 0 !important;
        }
        
        /* Chat message content */
        div[data-testid="stChatMessageContent"] p {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        
        /* Expander styling */
        div[data-testid="stExpander"] {
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Logo container */
        .logo-container {
            text-align: center;
            padding: 1rem 0;
        }
        
        .logo-container img {
            max-width: 200px;
            height: auto;
        }
        
        /* Add padding to sidebar content */
        .sidebar .block-container {
            padding-bottom: 100px;
        }
        
        /* Ensure chat messages don't go behind input */
        .stChatMessageContainer {
            margin-bottom: 20px;
        }
        
        /* Making chat input area more prominent */
        .stChatInput input {
            border: 1px solid #ddd !important;
            padding: 0.5rem 1rem !important;
            border-radius: 20px !important;
        }
        
        .stChatInput input:focus {
            border-color: #87CEEB !important;
            box-shadow: 0 0 0 1px #87CEEB !important;
        }
        </style>
        """, unsafe_allow_html=True)

    def setup_sidebar(self, databases):
        with st.sidebar:
            # Database Section
            with st.expander("📚 Database Settings", expanded=True):
                if not databases:
                    st.error("No databases available")
                    return None, None, None
                    
                selected_db = st.selectbox("Select Database", [db["name"] for db in databases])
                for idx, db in enumerate(databases):
                    if db["name"] == selected_db:
                        actual_db = db

                if not selected_db:
                    return None, None, None
                    
                db_info = self.db_manager.get_database_info(selected_db)
                _, collection = init_chromadb(db_info["path"])
                
                # Display database info
                stats = actual_db.get("stats", {})
                doc_count = stats['document_count']
                if doc_count == 0:
                    st.error("⚠️ Selected database is empty")
                    return None, None, None
                
                result = collection.get(include=["metadatas"], limit=1)
                if not (result and result["metadatas"]):
                    st.error("No metadata found in database")
                    return None, None, None
                    
                metadata = result["metadatas"][0]
                embedding_provider = metadata.get("provider")
                embedding_model = metadata.get("model")
                
                st.info(f"""
                **Database Info:**
                - Name: {selected_db}
                - Description: {db_info.get('description', 'N/A')}
                - Documents: {doc_count}
                - Embedding Model: {embedding_model}
                - Provider: {embedding_provider}
                """)
            
            # Chat Model Section
            with st.expander("🤖 Chat Model Settings", expanded=True):
                available_providers = providers.get_available_providers()
                
                selected_provider = st.radio("Chat Provider", available_providers, key="chat_provider")
                chat_models = providers.get_chat_models(selected_provider)
                if not chat_models:
                    st.error(f"No models available for {selected_provider}")
                    return None, None, None
                    
                chat_model_name = st.selectbox(
                    "Model", 
                    list(chat_models.values())
                )
                
                temperature = st.slider(
                    "Temperature (Creativity)", 
                    min_value=0.0, 
                    max_value=0.7, 
                    value=st.session_state.temperature, 
                    step=0.1,
                    help="Higher values make the output more creative but less focused",
                    key="temp_slider"
                )
                
                if temperature != st.session_state.temperature:
                    st.session_state.temperature = temperature

            # Ranking Settings Section
            with st.expander("🎯 Content Ranking Settings", expanded=False):
                st.session_state.boost_astrazeneca = st.toggle(
                    "Prioritize AstraZeneca Content",
                    value=st.session_state.boost_astrazeneca,
                    help="When enabled, boosts relevance of AstraZeneca-related content"
                )

                if st.session_state.boost_astrazeneca:
                    st.session_state.boost_factors["astrazeneca_base_boost"] = st.slider(
                        "AstraZeneca Content Boost",
                        min_value=1.0,
                        max_value=2.0,
                        value=1.5,
                        step=0.1,
                        help="Boost factor for AstraZeneca-related content"
                    )
                    
                    st.session_state.boost_factors["astrazeneca_strategic_boost"] = st.slider(
                        "Strategic Content Additional Boost",
                        min_value=1.0,
                        max_value=1.5,
                        value=1.2,
                        step=0.1,
                        help="Additional boost for strategic AstraZeneca content"
                    )
            
            # Chat Parameters Section
            with st.expander("⚙️ Chat Parameters", expanded=True):
                st.session_state.chat_memory = st.toggle(
                    "Enable Chat Memory",
                    value=True,
                    help="Keep context of previous messages in the conversation"
                )
                
                if st.button("🗑️ Clear Chat History", type="primary"):
                    st.session_state.messages = []
                    st.session_state.conversation = None
                    st.session_state.chat_history = None
                    st.rerun()
            
            return selected_db, selected_provider, chat_model_name

    def get_chat_model(self, provider: str, model_name: str, temperature = None):
        try:
            client = providers.get_provider_clients(provider, model_name, temperature)
            if client is not None:
                return client
            else:
                st.error(f"Error initializing chat model: {provider} - {model_name}")
        except Exception as e:
            st.error(f"Error initializing chat model: {provider} - {model_name}")
            return None

    def get_context(self, query: str, collection, embedding_provider: str, embedding_model: str) -> tuple:
        if collection.count() == 0:
            st.error("Cannot answer questions - database is empty")
            return None, []
            
        query_embedding = get_embeddings(query, provider=embedding_provider, model=embedding_model)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=app_config.max_context_length * 2,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"][0]:
            return None, []
            
        context = []
        sources_dict = {}  # Dictionary to track highest similarity per source
        
        # Create list of (doc, meta, distance) tuples
        search_results = list(zip(
            results["documents"][0], 
            results["metadatas"][0],
            results["distances"][0]
        ))
        
        # Sort by distance (similarity score)
        search_results.sort(key=lambda x: x[2])
        
        # Take only top 5 results
        search_results = search_results[:5]
        
        # Process documents and track best similarity scores
        for doc, meta, distance in search_results:
            context.append(doc)
            source_info = self.extract_document_info(meta)
            similarity = round((1 - distance) * 100, 2)  # Convert distance to similarity percentage
            
            # Keep highest similarity score for each source
            base_source = source_info.split(" (")[0].strip()
            if base_source not in sources_dict or similarity > sources_dict[base_source]["similarity"]:
                is_astrazeneca = any(keyword in doc.lower() for keyword in ['astrazeneca', 'azn', 'astra zeneca', 'astra-zeneca'])
                sources_dict[base_source] = {
                    "details": source_info,
                    "similarity": similarity,
                    "is_astrazeneca": is_astrazeneca and st.session_state.boost_astrazeneca
                }
        
        # Create sorted sources list
        sources = []
        if st.session_state.boost_astrazeneca:
            # Separate AstraZeneca and non-AstraZeneca sources
            az_sources = []
            other_sources = []
            
            for source, info in sorted(sources_dict.items(), key=lambda x: x[1]["similarity"], reverse=True):
                source_text = f"{info['details']} (similarity: {info['similarity']}%)"
                if info["is_astrazeneca"]:
                    az_sources.append(f"🔷 {source_text}")
                else:
                    other_sources.append(source_text)
            
            sources = az_sources + other_sources
        else:
            sources = [
                f"{info['details']} (similarity: {info['similarity']}%)"
                for info in sorted(sources_dict.values(), key=lambda x: x["similarity"], reverse=True)
            ]
        
        return "\n\n".join(context), sources
    
    def render(self):
        self.setup_page_config()
        st.title("💬 Ask me about all I know! 📚️")
        
        chat_container = st.container()
        databases = self.db_manager.list_databases()
        
        selected_db, provider, chat_model_name = self.setup_sidebar(databases)
        if not all([selected_db, provider, chat_model_name]):
            return
            
        db_info = self.db_manager.get_database_info(selected_db)
        _, collection = init_chromadb(db_info["path"])
        metadata = collection.get(include=["metadatas"], limit=1)["metadatas"][0]
        embedding_provider = metadata.get("provider")
        embedding_model = metadata.get("model")

        # Check if we need to reinitialize the model
        if self.should_reinitialize_model(provider, chat_model_name, st.session_state.temperature):
            chat_model = self.get_chat_model(provider, chat_model_name, st.session_state.temperature)
            if chat_model:
                st.session_state.chat_model = chat_model
            else:
                return
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Reference Documents", expanded=False):
                            st.markdown("**Sources with highest relevance:**")
                            for src in message["sources"]:
                                st.markdown(f"- {src}")

        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                context, sources = self.get_context(
                    prompt, collection, embedding_provider, embedding_model
                )
                
                if context:
                    # Add instruction about links to system message
                    system_message = """You are a helpful AI assistant. Answer questions based on the following context.
                    if the information is nt found int the context, plesaae let the user know that you do not have enough information to answer the question.
                    Important: If you reference any links in your response, they should ONLY be links that are specifically 
                    mentioned in the provided context. ."""
                    
                    messages = [
                        SystemMessage(content=f"{system_message}\nContext: {context}"),
                        *[HumanMessage(content=m["content"]) if m["role"] == "user" 
                          else AIMessage(content=m["content"]) 
                          for m in st.session_state.messages[:-1]],
                        HumanMessage(content=prompt)
                    ]

                    with st.chat_message("assistant"):
                        if st.session_state.chat_model:
                            response = st.session_state.chat_model.predict_messages(
                                messages,
                                callbacks=[StreamlitCallbackHandler(st.container())]
                            )
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response.content,
                                "sources": sources
                            })
                            st.rerun()
                        else:
                            st.error("Chat model not properly initialized")

    def extract_document_info(self, metadata):
        """Extract source information with optional page numbers or sections"""
        doc_info = metadata.get("source", "")
        
        # List of possible location identifiers
        locations = []
        
        # Check for various location identifiers in order of preference
        if metadata.get("date"):
            locations.append(f"date: {metadata['date']}")
        if metadata.get("page"):
            locations.append(f"page {metadata['page']}")
        if metadata.get("chapter"):
            locations.append(f"chapter {metadata['chapter']}")
        if metadata.get("section"):
            locations.append(metadata['section'])
        if metadata.get("source_type"):
            locations.append(f"type: {metadata['source_type']}")
        
        # Add location info if available
        if locations:
            doc_info += f" ({', '.join(locations)})"
                
        return doc_info
    
    def render(self):
        self.setup_page_config()
        st.title("💬 Ask me all you want about my stored documents!")
        
        chat_container = st.container()
        databases = self.db_manager.list_databases()
        
        selected_db, provider, chat_model_name = self.setup_sidebar(databases)
        if not all([selected_db, provider, chat_model_name]):
            return
            
        db_info = self.db_manager.get_database_info(selected_db)
        _, collection = init_chromadb(db_info["path"])
        metadata = collection.get(include=["metadatas"], limit=1)["metadatas"][0]
        embedding_provider = metadata.get("provider")
        embedding_model = metadata.get("model")

        # Check if we need to reinitialize the model
        if self.should_reinitialize_model(provider, chat_model_name, st.session_state.temperature):
            chat_model = self.get_chat_model(provider, chat_model_name, st.session_state.temperature)
            if chat_model:
                st.session_state.chat_model = chat_model
            else:
                return
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Reference Documents", expanded=False):
                            st.markdown("**Sources with highest relevance:**")
                            for src in message["sources"]:
                                st.markdown(f"- {src}")

        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                context, sources = self.get_context(
                    prompt, collection, embedding_provider, embedding_model
                )
                
                if context:
                    messages = [
                        SystemMessage(content=f"""You are a helpful AI assistant. 
                        Answer questions based on the following context: {context}"""),
                        *[HumanMessage(content=m["content"]) if m["role"] == "user" 
                          else AIMessage(content=m["content"]) 
                          for m in st.session_state.messages[:-1]],
                        HumanMessage(content=prompt)
                    ]

                    with st.chat_message("assistant"):
                        if st.session_state.chat_model:
                            response = st.session_state.chat_model.predict_messages(
                                messages,
                                callbacks=[StreamlitCallbackHandler(st.container())]
                            )
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response.content,
                                "sources": sources
                            })
                            st.rerun()
                        else:
                            st.error("Chat model not properly initialized")

def main():
    db_manager = VectorDBManager(app_config.vector_db_root)
    chat_ui = RAGChatUI(db_manager)
    chat_ui.render()

if __name__ == "__main__":
    main()