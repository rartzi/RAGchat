# RAG Knowledge Base Manager 🏠

A powerful and user-friendly application for managing and querying document-based knowledge bases using Retrieval-Augmented Generation (RAG). Built with Streamlit, this application provides an intuitive interface for document management, embedding generation, and natural language querying.

![License](https://img.shields.io/badge/license-MIT-green)

## 🌟 Features

### Document Management
- Upload and process multiple document formats (PDF, DOCX, TXT, etc.)
- Automatic text chunking and embedding generation
- Document metadata management
- Database statistics and monitoring
- Multi-provider support for embeddings

### Query Interface
- Natural language querying
- Context-aware responses
- TBD: Source citations and references
- TBD: Adjustable search parameters
- TBD: Query result export capabilities

### Supported Providers
#### Embedding Providers
- *** AZ Azure OpenAI ( text-embedding-3-large, text-embedding-ada-002) ***
- OpenAI ( text-embedding-3-large, text-embedding-ada-002)
- Ollama (nomic-embed-text and custom models)

#### LLM Providers
- *** AZ Azure OpenAI (GPT-4o-mini , GPT-4o, GPT-3.5 Turbo) ***
- OpenAI (GPT-4o-mini , GPT-4o, GPT-3.5 Turbo)
- Ollama (Llama 2, Mistral, custom models)

## 📋 Prerequisites

- Python 3.12.7 or higher
- pip package manager
- Git
- API keys for chosen providers (OpenAI/Azure)
- Ollama (if using local models)

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-knowledge-base.git
cd rag-knowledge-base
```

2. Create and activate a virtual environment:

Poetry Lock
Poetry 


3. Install required packages:
```bash
poetry install
```

4. Set up environment variables:
```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys and configurations
OPENAI_API_KEY=<public openai api key>
DB_DIR=<full path for the selected vector store>
DEFAULT_DOCS_DIR=/<full path for the selected documents to be uploaded>
AZURE_OPENAI_API_KEY=<public azure openai api key>
AZURE_OPENAI_ENDPOINT=<az azure openai endpoint>
AZURE_OPENAI_API_VERSION=<az azure openai api version>
AZURE_OPENAI_DEPLOYMENT=<az azure openai deployment>
AZURE_OPENAI_EMBEDDING=<az azure openai embedding>
```

5. Run the application:
```bash
streamlit run HomePage.py
```

## 📁 Project Structure

```
TBD - Target State
rag-knowledge-base/
├── Home.py                 # Main application entry point
├── Config/  
├── pages/
│   ├── 01_RAG_DB_Manager.py    # Database management interface
│   └── 02_RAG_Ask_ODSP.py     # Query interface
├── utils/
│   ├── embedding_utils.py        # Embedding generation utilities
│   ├── document_processor.py     # Document processing utilities
│   └── database_manager.py       # Database operations
├── pyproject.toml       # Python dependencies
├── .env.example          # Example environment variables
└── README.md             # This file
```

## ⚙️ Configuration

The application can be configured through the UI or using environment variables:

```ini
# .env file
TBD
```

## 📦 Dependencies

TBD

Full list available in `requirements.txt`

## 🔧 Advanced Configuration

### Custom Embedding Models
To use custom embedding models with Ollama:

1. Install Ollama following the [official documentation](https://github.com/ollama/ollama)
2. Pull your desired model:
```bash
ollama pull your-model-name
```
3. Configure the model in the application settings

### Document Processing
Customize chunking parameters in : TBD
 

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Langchain for RAG capabilities
- ChromaDB team for the vector database
- OpenAI, Azure, and Ollama teams for their models and APIs

## 📮 Contact

For questions and support:
- Create an issue in the repository
- Contact the maintainers at [ronen.artzi@astrazeneca.com]

## 🔄 Updates

Stay updated with new features and improvements:
- Star and watch the repository
- Check the [CHANGELOG.md](CHANGELOG.md) for version updates
- Follow the project's [blog](https://tbd.astrazeneca)