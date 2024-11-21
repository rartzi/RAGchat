# utils/db_utils.py
import chromadb
import os
from typing import Dict

def init_chromadb(db_dir: str) -> tuple[chromadb.PersistentClient, chromadb.Collection]:
    """Initialize ChromaDB client and collection."""
    os.makedirs(db_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(
        name="document_embeddings",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

def get_database_stats(collection: chromadb.Collection) -> Dict:
    """Get current database statistics."""
    try:
        results = collection.get()
        if not results or not results['ids']:
            return {
                "document_count": 0,
                "chunk_count": 0,
                "sources": [],
                "models": [],
                "status": "empty"
            }
        
        valid_documents = []
        valid_embedding_models = []
        for meta in results['metadatas']:
            if meta and isinstance(meta, dict):
                if 'source' in meta:
                    valid_documents.append(meta['source'])
                if 'model' in meta:
                    valid_embedding_models.append(meta['model'])
                
        return {
            "document_count": len(set(valid_documents)),
            "chunk_count": len(valid_documents),
            "sources": sorted(list(set(valid_documents))),
            "models": sorted(list(set(valid_embedding_models))),
            "status": "ready"
        }
    except Exception as e:
        return {
            "document_count": 0,
            "chunk_count": 0,
            "sources": [],
            "models": [],
            "status": f"error: {str(e)}"
        }