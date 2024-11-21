# app.py
import streamlit as st
# Set page config first, before any other st commands
st.set_page_config(
    page_title="Knowledge DBs Manager",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
from datetime import datetime
import uuid
from typing import Optional, Dict, List, Tuple

# Import all utilities
from utils.db_utils import init_chromadb, get_database_stats
from utils.embedding_utils import get_embeddings
from utils.text_utils import read_document, chunk_text

from config import providers, app_config
from config.app_config import AppConfig
from config.providers_config import ProvidersConfig
from utils.db_manager_utils import VectorDBManager

# Load configuration
app_config = AppConfig()


providers_config = ProvidersConfig()

# Directory to list files from
directory = app_config.default_docs_dir
def list_files(directory: str) -> List[str]:
    """List all files in the given directory."""
    if not os.path.exists(directory):
        st.error(f"Directory does not exist: {directory}")
        return []
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
files = list_files(directory)


OPENAI_MODELS= providers.get_embedding_models(provider='OpenAI')
AZURE_OPENAI_MODELS= providers.get_embedding_models(provider='Azure OpenAI')
OLLAMA_MODELS= providers.get_embedding_models(provider='Ollama')

class DatabaseManagementUI:
    def __init__(self, db_manager: VectorDBManager):
        self.db_manager = db_manager

    def render(self):
        st.header("Database Management")
        
        # Create new database section
        with st.expander("+ New Database", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_db_name = st.text_input("Database Name", key="create_new_db_name")
                new_db_desc = st.text_input("Description (optional)", key="create_new_db_desc")
            with col2:
                st.write("")  # Spacing
                if st.button("Create Database", key="create_db_button", use_container_width=True):
                    if new_db_name:
                        if self.db_manager.create_database(new_db_name, new_db_desc):
                            st.success(f"Created database: {new_db_name}")
                            st.rerun()
                        else:
                            st.error("Failed to create database")

        # Initialize session states
        if 'rename_states' not in st.session_state:
            st.session_state.rename_states = {}

        # List databases
        databases = self.db_manager.list_databases()
        active_db = self.db_manager.get_active_database()

        if not databases:
            st.info("No databases available. Create one to get started!")
            return

        # Custom CSS
        st.markdown("""
        <style>
            .db-row {
                padding: 0.5rem;
                border: 1px solid #eee;
                border-radius: 5px;
                margin: 0.5rem 0;
            }
            .active-db-row {
                background-color: #e3f2fd;
                border-left: 4px solid #1976d2;
            }
            .mono-model {
                font-family: monospace;
                padding: 2px 4px;
                background: #f5f5f5;
                border-radius: 3px;
            }
        </style>
        """, unsafe_allow_html=True)

        # Database listing headers
        cols = st.columns([2, 3, 2, 2, 1.5])
        with cols[0]:
            st.markdown("**Name**")
        with cols[1]:
            st.markdown("**Description**")
        with cols[2]:
            st.markdown("**Stats**")
        with cols[3]:
            st.markdown("**Model**")
        with cols[4]:
            st.markdown("**Actions**")

        # List each database
        for idx, db in enumerate(databases):
            db_name = db['name']
            is_active = active_db == db_name
            stats = db.get('stats', {})
            
            row_class = "active-db-row" if is_active else "db-row"
            st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)
            
            cols = st.columns([2, 3, 2, 2, 1.5])
            
            # Name
            with cols[0]:
                if st.session_state.rename_states.get(db_name, False):
                    new_name = st.text_input("", 
                                           value=db_name,
                                           key=f"rename_{idx}_{db_name}",
                                           label_visibility="collapsed")
                    if st.button("Save", key=f"save_{idx}_{db_name}"):
                        if new_name != db_name:
                            if self.db_manager.rename_database(db_name, new_name):
                                st.session_state.rename_states[db_name] = False
                                st.rerun()
                else:
                    st.write(f"{'📍 ' if is_active else ''}{db_name}")

            # Description
            with cols[1]:
                st.write(db.get('description', ''))

            # Stats
            with cols[2]:
                if stats.get('status') == 'ready':
                    st.write(f"{stats['document_count']} docs, {stats['chunk_count']} chunks")

            # Model
            with cols[3]:
                if stats.get('status') == 'ready' and stats.get('models'):
                    st.markdown(f"<code>{stats['models'][-1]}</code>", unsafe_allow_html=True)

            # Actions
            with cols[4]:
                btn_cols = st.columns(4)
                with btn_cols[0]:
                    if not is_active:
                        if st.button("📌", key=f"activate_{idx}_{db_name}", help="Set active"):
                            self.db_manager.set_active_database(db_name)
                            st.rerun()
                
                with btn_cols[1]:
                    if st.button("✏️", key=f"rename_{idx}_{db_name}", help="Rename"):
                        st.session_state.rename_states[db_name] = True
                        st.rerun()
                
                with btn_cols[2]:
                    if st.button("📋", key=f"duplicate_{idx}_{db_name}", help="Duplicate"):
                        new_name = f"{db_name}_copy"
                        if self.db_manager.duplicate_database(db_name, new_name):
                            st.rerun()
                
                with btn_cols[3]:
                    delete_key = f"delete_confirm_{db_name}"
                    if delete_key not in st.session_state:
                        st.session_state[delete_key] = False
                        
                    if st.button("🗑️", key=f"delete_{idx}_{db_name}", help="Delete"):
                        st.session_state[delete_key] = True
                        st.rerun()

            # Delete confirmation
            if st.session_state.get(f"delete_confirm_{db_name}", False):
                st.warning(f"Are you sure you want to delete '{db_name}'?")
                conf_col1, conf_col2 = st.columns(2)
                with conf_col1:
                    if st.button("Yes", key=f"confirm_delete_{idx}_{db_name}"):
                        if self.db_manager.delete_database(db_name):
                            st.success(f"Deleted {db_name}")
                            st.session_state[f"delete_confirm_{db_name}"] = False
                            st.rerun()
                with conf_col2:
                    if st.button("No", key=f"cancel_delete_{idx}_{db_name}"):
                        st.session_state[f"delete_confirm_{db_name}"] = False
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

class DocumentManager:
    def __init__(self, db_dir: str, docs_dir: str, db_manager: VectorDBManager):
        self.db_dir = db_dir
        self.docs_dir = docs_dir
        self.provider = None
        self.model = None
        self.db_manager = db_manager
        _, self.collection = init_chromadb(db_dir)

    def delete_document(self, source: str) -> bool:
        """Delete a document and all its chunks from the database"""
        try:
            # Get all chunks for this document
            result = self.collection.get(
                where={"source": source},
                include=["metadatas"]  # Changed from "ids" to "metadatas"
            )
            
            if result and result['ids']:  # ChromaDB always returns 'ids' even if not included
                # Delete all chunks for this document
                self.collection.delete(
                    ids=result['ids']
                )
                st.success(f"Successfully deleted {len(result['ids'])} chunks from {source}")
                return True
            else:
                st.warning(f"No chunks found for document: {source}")
                return False
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False

    def get_database_info(self) -> Optional[Dict]:
        """Get database metadata including provider and model"""
        try:
            # Get all documents to check their metadata
            result = self.collection.get(
                include=["metadatas"],
                limit=1  # We only need one document to check the model and provider
            )
            
            # Get total count
            total_count = self.collection.count()
            
            # If database is empty, return empty state
            if total_count == 0:
                return {
                    'provider': None,
                    'model': None,
                    'document_count': 0,
                    'chunk_count': 0,
                    'sources': []
                }
            
            # Get unique sources
            all_results = self.collection.get(
                include=["metadatas"],
            )
            unique_sources = list(set(meta['source'] for meta in all_results['metadatas'] if 'source' in meta))
            
            # If we have documents, get the metadata from the first one
            if result['metadatas'] and len(result['metadatas']) > 0:
                metadata = result['metadatas'][0]
                return {
                    'provider': metadata.get('provider'),
                    'model': metadata.get('model'),
                    'document_count': len(unique_sources),
                    'chunk_count': total_count,
                    'sources': sorted(unique_sources)  # Sort sources for consistent display
                }
            
            # If we have documents but no metadata (shouldn't happen), return error state
            st.warning("Database contains documents but metadata is incomplete.")
            return {
                'provider': None,
                'model': None,
                'document_count': len(unique_sources),
                'chunk_count': total_count,
                'sources': sorted(unique_sources)
            }
            
        except Exception as e:
            st.error(f"Error retrieving database info: {str(e)}")
            return None

    def is_empty_database(self) -> bool:
        """Check if the database is empty"""
        try:
            count = self.collection.count()
            return count == 0
        except Exception as e:
            st.error(f"Error checking database: {str(e)}")
            return True

    def process_document(self, file_path: str, provider: str, model: str) -> bool:
        """Process and add a document to the database"""
        try:
            # Check if database already has documents
            is_empty = self.is_empty_database()
            
            if not is_empty:
                # Get existing database info
                db_info = self.get_database_info()
                if db_info and db_info['model']:
                    if model != db_info['model']:
                        st.error(f"Cannot process document with model '{model}'. This database uses '{db_info['model']}'")
                        return False
            
            content = read_document(file_path)
            if not content.strip():
                st.warning(f"No content extracted from {file_path}")
                return False
                
            chunks = chunk_text(content)
            file_name = os.path.basename(file_path)
            
            success_count = 0
            for i, chunk in enumerate(chunks):
                embedding = get_embeddings(chunk, provider, model)
                if embedding:
                    chunk_id = f"{file_name}_chunk_{i}_{str(uuid.uuid4())}"
                    try:
                        self.collection.add(
                            ids=[chunk_id],
                            embeddings=[embedding],
                            metadatas=[{
                                "source": file_name,
                                "chunk": i,
                                "chunk_total": len(chunks),
                                "content": chunk,
                                "added_at": datetime.now().isoformat(),
                                "provider": provider,
                                "model": model
                            }],
                            documents=[chunk]
                        )
                        success_count += 1
                    except Exception as e:
                        st.error(f"Error adding chunk {i} of {file_name} to database: {str(e)}")
                        return False
            
            return success_count > 0
            
        except Exception as e:
            st.error(f"Error processing {file_path}: {str(e)}")
            return False


class DocumentManagementUI:
    def __init__(self, doc_manager: DocumentManager):
        self.doc_manager = doc_manager
        filtered_files = []
        
        st.session_state.select_all = False
        # Initialize session state for file selection
        if 'selected_files' not in st.session_state:
            st.session_state.selected_files = []
        if 'docs_dir' not in st.session_state:
            st.session_state.docs_dir = app_config.default_docs_dir
        if 'should_rerun' not in st.session_state:
            st.session_state.should_rerun = False

    def render_database_header(self, db_info: Dict, active_db_name: str) -> bool:
        """Render the database info header in a single line"""
        if not db_info or not active_db_name:
            st.error("No active database selected")
            return False

        # Single line header with all database info
        cols = st.columns([3, 2, 2, 2])
        with cols[0]:
            st.info(f"📚 Active DB: `{active_db_name}`")
        with cols[1]:
            model = db_info.get('model', 'Not set')
            st.info(f"🔤 Model: `{model}`")
        with cols[2]:
            doc_count = str(db_info.get('document_count', 0))
            st.info(f"📄 Documents: `{doc_count}`")
        with cols[3]:
            chunk_count = str(db_info.get('chunk_count', 0))
            st.info(f"🔢 Chunks:`{chunk_count}`")
        
        return True

    def file_selector(self) -> Tuple[Optional[List[str]], Optional[str]]:
        """Render the directory selection input"""
        docs_dir = st.text_input(
            "📁 Documents Directory:",
            value=st.session_state.docs_dir,
            help="Enter the full path to your documents directory"
        )

        if docs_dir and os.path.isdir(docs_dir):
            st.session_state.docs_dir = docs_dir
            files = [f for f in os.listdir(docs_dir) 
                    if f.endswith(('.txt', '.pdf', '.docx', '.pptx'))]
            
            if not files:
                st.warning("📭 No supported documents found in directory")
                return None, docs_dir
            
            return files, docs_dir
        else:
            if docs_dir:
                st.error("❌ Invalid directory path")
            return None, None

    def render_file_management(self, files: List[str], docs_dir: str):
        """Render the file selection and management interface"""
        if not files:
            return

        # Initialize session state for selected files and select all checkbox if not already done
        if 'selected_files' not in st.session_state:
            st.session_state.selected_files = []
        if 'select_all' not in st.session_state:
            st.session_state.select_all = False

        # Get the list of files already in the database
        db_files = self.get_files_in_database()

        # Filter out files that are already in the database
        available_files = [f for f in files if f not in db_files]

        # Create two columns for file management
        left_col, right_col = st.columns(2)

        # Available Files Column
        with left_col:
            st.markdown(f"##### Input Folder Content [pdf,txt,pptx] - Excluding Files are in the Databasae  ({len([f for f in available_files if f not in [sf['name'] for sf in st.session_state.selected_files]])})")

            # Filter available files
            search_available = st.text_input("🔍 Search available files", key="search_available")
            filtered_files = [f for f in available_files if search_available.lower() in f.lower()] if search_available else available_files

            # Select All checkbox
            select_all = st.checkbox("Select All", key="select_all_available", value=st.session_state.select_all)
            if select_all:
                selected_files = filtered_files
                st.session_state.select_all = True
            else:
                selected_files = st.multiselect(
                    "Select Files",
                    filtered_files,
                    default=[f['name'] for f in st.session_state.selected_files]
                )

            # Update session state with selected files
            st.session_state.selected_files = [{'name': file, 'path': os.path.join(docs_dir, file)} for file in selected_files]


        # Selected Files Column
        with right_col:
            # Header with count and clear button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"##### Files to be Ingested({len(st.session_state.selected_files)})")
            #with col2:
            #    if st.button("Clear All"):
            #        st.session_state.selected_files = []
            #        filtered_selected = []
            #        st.session_state.select_all = False
            #        st.session_state["select_all_available"]
            #        #st.rerun()  # Trigger a rerun of the app

            # Filter selected files

            #if st.session_state.select_all:
            search_selected = st.text_input("🔍 Search selected files", key="search_selected")
            filtered_selected = [f for f in st.session_state.selected_files if search_selected.lower() in f['name'].lower()] if search_selected else st.session_state.selected_files
        
            # Process button
            if st.session_state.selected_files:
                st.button("🔄 Process Selected Files", type="primary", on_click=self.process_files, use_container_width=True)
    
            # Show selected file
            if filtered_selected:
                for file in filtered_selected:
                    st.text(f"📄 {file['name']}")
            else:
                    st.text(f"No Files Selected")
   
    def get_files_in_database(self) -> List[str]:
        """Get the list of files already in the database"""
        db_info = self.doc_manager.get_database_info()
        if db_info and 'sources' in db_info:
            return db_info['sources']
        return []      
    
    def render_file_management_old(self, files: List[str], docs_dir: str):
        """Render the file selection and management interface"""
        if not files:
            return

        # Create two columns for file management
        left_col, right_col = st.columns(2)
        
        # Available Files Column
        with left_col:
            st.markdown(f"##### Input Folder Content - pdf, txt, pptx  ({len([f for f in files if f not in [sf['name'] for sf in st.session_state.selected_files]])})")
            
            # Filter available files
            search_available = st.text_input("🔍 Search available files", key="search_available")
            filtered_files = [f for f in files if search_available.lower() in f.lower()] if search_available else files
            
            # Show available files
            for file in filtered_files:
                if file not in [f['name'] for f in st.session_state.selected_files]:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.text(f"📄 {file}")
                    with col2:
                        if st.button("➕", key=f"add_{file}", help="Add to selection"):
                            st.session_state.selected_files.append({
                                'name': file,
                                'path': os.path.join(docs_dir, file)
                            })
                            st.rerun()

        # Selected Files Column
        with right_col:
            # Header with count and clear button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"##### Files to be Ingested({len(st.session_state.selected_files)})")
            with col2:
                if st.button("Clear All"):
                    st.session_state.selected_files = []
                    st.rerun()
            
            # Filter selected files
            search_selected = st.text_input("🔍 Search selected files", key="search_selected")
            filtered_selected = [f for f in st.session_state.selected_files 
                               if search_selected.lower() in f['name'].lower()] if search_selected else st.session_state.selected_files
            
            # Show selected files
            for file in filtered_selected:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.text(f"📄 {file['name']}")
                with col2:
                    if st.button("❌", key=f"remove_{file['name']}", help="Remove from selection"):
                        st.session_state.selected_files = [
                            f for f in st.session_state.selected_files 
                            if f['name'] != file['name']
                        ]
                        st.rerun()

            # Process button
            if st.session_state.selected_files:
                st.button("🔄 Process Selected Files", 
                         type="primary",
                         on_click=self.process_files,
                         use_container_width=True)

    def render_database_contents(self, db_info: Dict):
        """Render the current database contents"""
        if db_info.get('document_count', 0) > 0:
            with st.expander("📚 Database Contents", expanded=False):
                # Search filter for database files
                search_term = st.text_input("🔍 Search database files", key="search_db_files")
                
                # Get and filter sources
                sources = db_info.get('sources', [])
                if search_term:
                    sources = [s for s in sources if search_term.lower() in s.lower()]
                
                st.markdown(f"Files in Database ({len(sources)})")
                
                # Display files with delete buttons
                for source in sources:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.text(f"📄 {source}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{source}", help="Delete from database"):
                            if self.doc_manager.delete_document(source):
                                st.success(f"Deleted {source}")
                                st.rerun()
                            else:
                                st.error(f"Failed to delete {source}")

    def check_provider_configuration(self) -> bool:
        """Check if the selected provider is properly configured"""
        if not self.doc_manager.provider or not self.doc_manager.model:
            st.error("Provider or model not configured")
            return False
        return True


    def process_files(self):
        """Process the selected files"""
        if not st.session_state.selected_files:
            st.warning("No files selected for processing")
            return
            
        progress = st.progress(0)
        processed_files = 0
        total_files = len(st.session_state.selected_files)
        
        status_container = st.empty()
        
        try:
            for i, file in enumerate(st.session_state.selected_files):
                status_container.info(f"Processing {file['name']} ({i+1}/{total_files})")
                
                status = self.doc_manager.process_document(
                    file['path'],
                    self.doc_manager.provider,
                    self.doc_manager.model
                )
                if status:
                    processed_files += 1
                
                progress.progress((i + 1) / total_files)
            
            if processed_files > 0:
                status_container.success(f"✅ Processed {processed_files}/{total_files} files successfully!")
                st.session_state.selected_files = []
            else:
                status_container.error("❌ No files were successfully processed")

        except Exception as e:
            status_container.error(f"Error processing files: {str(e)}")

        st.session_state.should_rerun = True

    def render(self):
        """Main render method"""
        # Check if we need to rerun from previous processing
        if st.session_state.get('should_rerun', False):
            st.session_state.should_rerun = False
            st.rerun()

        # Get active database name
        active_db_name = self.doc_manager.db_manager.get_active_database()
        if not active_db_name:
            st.error("No active database selected")
            return

        # Get database info
        db_info = self.doc_manager.get_database_info()
        
        # Check provider configuration
        if not self.check_provider_configuration():
            return
        
        # Render the single-line header
        if not self.render_database_header(db_info, active_db_name):
            return

        st.divider()

        # File selector section
        files, docs_dir = self.file_selector()
        if docs_dir:  # Proceed if directory is valid
            # File management interface
            self.render_file_management(files, docs_dir)
            
            st.divider()
            
            # Database contents with delete functionality
            self.render_database_contents(db_info)
            
def main():

    # Initialize managers
    root_dir= app_config.vector_db_root
    db_manager = VectorDBManager(root_dir)
    active_db = db_manager.get_active_database()

    # Initialize providers configuration
    providers_config = ProvidersConfig()
    
    # Main content area
    # App title and active database banner
    st.markdown("""
        <div class="app-header">
            <div class="title-section">
                <h1>🗄️ RAG Knowledge DBs Manager</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if active_db:
        st.markdown(f"""
            <div class="active-db-banner">
                <h3 style="margin: 0;">Active Database: {active_db}</h3>
            </div>
        """, unsafe_allow_html=True)

    # Sidebar - now only for model selection if needed
    with st.sidebar:
        if active_db:
            db_info = db_manager.get_database_info(active_db)
            doc_manager = DocumentManager(
                db_dir=db_info['path'],
                docs_dir=app_config.default_docs_dir,
                db_manager=db_manager
            )
            
            # Get existing database info
            db_metadata = doc_manager.get_database_info()
            
            if db_metadata and db_metadata.get('model'):
                st.session_state.model = db_metadata['model']
                if db_metadata.get('provider'):
                    st.session_state.provider = db_metadata['provider']
                st.info(f"Current Database Embedding model will be used for processing.")
            else:
                # Get available providers from config
                available_providers = providers_config.get_available_providers()
                
                if not available_providers:
                    st.error("No LLM providers are currently available. Please check your configuration.")
                else:
                    # Display radio buttons only for enabled and available providers
                    provider = st.radio(
                        "LLM Provider",
                        available_providers,
                        key="sidebar_provider_select"
                    )
                    
                    # Get models based on selected provider
                    if provider:
                        available_models = providers_config.get_embedding_models(provider)
                        if available_models:
                            model = st.selectbox(
                                "Model", 
                                list(available_models.values()),
                                key=f"sidebar_{provider.lower()}_model_select"
                            )
                            
                            st.session_state.model = model
                            st.session_state.provider = provider
                        else:
                            st.error(f"No models available for {provider}")
                            st.session_state.model = None
                            st.session_state.provider = None

        else:
            st.warning("No active database selected")
            st.session_state.provider = None
            st.session_state.model = None
    

    # Main content tabs
    tab1, tab2 = st.tabs(["Database Management", "Document Management"])
    
    with tab1:
        db_ui = DatabaseManagementUI(db_manager)
        db_ui.render()
    
    with tab2:
        if active_db:
            db_info = db_manager.get_database_info(active_db)
            doc_manager = DocumentManager(
                db_dir=db_info['path'],
                docs_dir=app_config.default_docs_dir,
                db_manager=db_manager  # Pass the db_manager instance
            )
            doc_manager.provider = st.session_state.provider
            doc_manager.model = st.session_state.model
            
            doc_ui = DocumentManagementUI(doc_manager)
            doc_ui.render()
        else:
            st.info("Please select an active database first")
        

    # Footer
    st.divider()
    st.caption(f"Database Root: {app_config.vector_db_root}")

if __name__ == "__main__":
    main()
   