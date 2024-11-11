import streamlit as st
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader , PyPDFLoader , UnstructuredPowerPointLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain 
from langchain.memory import ConversationBufferMemory


import ollama


# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in environment variables!")
    st.info("Please set your OpenAI API key in the text input below:")
    api_key = st.text_input("OpenAI API Key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API Key set successfully!")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Document Chat", layout="wide")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

########  Ronen


# init models
if "model" not in st.session_state:
    st.session_state["model"] = ""

models = [model["name"] for model in ollama.list()["models"]]
st.session_state["model"] = st.sidebar.selectbox("Choose your model", models)

######### Ronen

# Load documents from specified folder


def load_documents(directory_path):
    """Load all  documents from the specified directory."""
    documents = []
    for file_path in Path(directory_path).glob("**/*.*"):
        try:
            if file_path.name.endswith('.txt'):
                loader = TextLoader(str(file_path))
                documents.extend(loader.load())
            elif file_path.name.endswith('.pdf'):
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
            elif file_path.name.endswith('.pptx'):
                loader = UnstructuredPowerPointLoader(str(file_path))
                documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
    return documents


def process_documents(directory_path):
    """Load, split and embed documents from the specified directory."""
    try:
        # Remove existing vector store directory if it exists
        vector_store_path = "./vector_store"
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
        
        # Load documents
        documents = load_documents(directory_path)
        if not documents:
            st.error("No text documents found in the specified directory")
            return None
        
        # Split documents into chunks using simple CharacterTextSplitter
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n",
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        
        # Initialize embeddings using OpenAI
        embeddings = OpenAIEmbeddings()
        
        # Create and persist vector store
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=vector_store_path
        )
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        # Fixed error display
        st.error(str(e))
        return None
    

def initialize_conversation(vector_store):
    """Initialize the conversation chain with the vector store."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    conversation = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model="gpt-4o"),
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        memory=memory,
        get_chat_history=lambda h : h,
        verbose=True
    )
    
    return conversation

# Main chat interface
st.title("Chat with Your Documents")

# Sidebar for document upload and processing
st.sidebar.title("Document Processing")
directory_path = st.sidebar.text_input("Enter documents directory path:","/Users/ronenartzi/development/playground/gui/streamlit/RAGchat/documents")
process_button = st.sidebar.button("Process Documents")

# Main content area for processing status
if process_button and directory_path:
    if not os.path.exists(directory_path):
        st.sidebar.error("Directory not found!")
    else:
        with st.spinner("Processing documents..."):
            st.session_state.vector_store = process_documents(directory_path)
            if st.session_state.vector_store:
                st.session_state.conversation = initialize_conversation(st.session_state.vector_store)
                st.sidebar.success("Documents processed successfully!")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, tuple):
        user_msg, ai_msg = message
        st.chat_message("user").write(user_msg)
        st.chat_message("assistant").write(ai_msg)

# Chat input
if prompt := st.chat_input("Ask a question about your documents:"):
    if not st.session_state.conversation:
        st.error("Please process documents first!")
    else:
        # Display user message
        st.chat_message("user").write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": prompt})
                ai_response = response["answer"]
                st.write(ai_response)
        
        # Store the conversation
        st.session_state.chat_history.append((prompt, ai_response))

# Add some usage instructions
if not st.session_state.vector_store:
    st.info("""
    👋 Welcome! To get started:
    1. Enter the path to your documents directory in the sidebar
    2. Click 'Process Documents' to load and embed your documents
    3. Start chatting with your documents!
    """)