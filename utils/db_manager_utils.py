# utils/db_manager_utils.py
import os
import shutil
from typing import Dict, List, Optional
import json
import chromadb
from typing import Dict, List, Optional
from datetime import datetime
from .db_utils import init_chromadb, get_database_stats

# utils/db_manager_utils.py

class VectorDBManager:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.config_file = os.path.join(root_dir, 'db_config.json')
        self.active_db = None
        self._load_or_create_config()

    # ... (your existing methods) ...

    def rename_database(self, old_name: str, new_name: str) -> bool:
        """Rename an existing database"""
        if old_name not in self.config['databases'] or new_name in self.config['databases']:
            return False

        try:
            old_path = self.config['databases'][old_name]['path']
            new_path = os.path.join(self.root_dir, new_name)
            
            # Move the database files
            shutil.move(old_path, new_path)
            
            # Update configuration
            self.config['databases'][new_name] = self.config['databases'][old_name].copy()
            self.config['databases'][new_name]['path'] = new_path
            self.config['databases'][new_name]['last_modified'] = str(datetime.now())
            del self.config['databases'][old_name]
            
            # Update active database if necessary
            if self.config['active_db'] == old_name:
                self.config['active_db'] = new_name
                
            self._save_config()
            return True
        except Exception as e:
            print(f"Error renaming database: {e}")
            return False
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.config_file = os.path.join(root_dir, 'db_config.json')
        self.active_db = None
        self._load_or_create_config()

    def _load_or_create_config(self):
        """Load existing config or create new one"""
        os.makedirs(self.root_dir, exist_ok=True)
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'databases': {},
                'active_db': None
            }
            self._save_config()

    def _save_config(self):
        """Save current configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def create_database(self, db_name: str, description: str = "") -> bool:
        """Create a new vector database"""
        if db_name in self.config['databases']:
            return False

        db_path = os.path.join(self.root_dir, db_name)
        try:
            os.makedirs(db_path, exist_ok=True)
            # Use our existing init_chromadb utility
            _, collection = init_chromadb(db_path)
            if collection:
                self.config['databases'][db_name] = {
                    'path': db_path,
                    'description': description,
                    'created_at': str(datetime.now()),
                    'last_modified': str(datetime.now())
                }
                self._save_config()
                return True
            return False
        except Exception:
            return False

    def get_database_stats(self, db_name: str) -> Dict:
        """Get statistics for a specific database"""
        if db_name not in self.config['databases']:
            return {
                "status": "not_found"
            }
        
        client, collection = init_chromadb(self.config['databases'][db_name]['path'])
        return get_database_stats(collection)  # Using our existing get_database_stats utility

    def duplicate_database(self, source_db: str, target_db: str) -> bool:
        """Duplicate an existing database"""
        if source_db not in self.config['databases'] or target_db in self.config['databases']:
            return False

        try:
            source_path = self.config['databases'][source_db]['path']
            target_path = os.path.join(self.root_dir, target_db)
            shutil.copytree(source_path, target_path)
            
            self.config['databases'][target_db] = {
                'path': target_path,
                'description': f"Copy of {source_db}",
                'created_at': str(datetime.now()),
                'last_modified': str(datetime.now())
            }
            self._save_config()
            return True
        except Exception:
            return False

    def delete_database(self, db_name: str) -> bool:
        """Delete a database"""
        if db_name not in self.config['databases']:
            return False

        try:
            db_path = self.config['databases'][db_name]['path']
            shutil.rmtree(db_path)
            del self.config['databases'][db_name]
            
            if self.config['active_db'] == db_name:
                self.config['active_db'] = None
                
            self._save_config()
            return True
        except Exception:
            return False

    def set_active_database(self, db_name: str) -> bool:
        """Set the active database"""
        if db_name not in self.config['databases'] and db_name is not None:
            return False
            
        self.config['active_db'] = db_name
        self._save_config()
        return True

    def get_active_database(self) -> Optional[str]:
        """Get currently active database name"""
        return self.config['active_db']

    def get_database_info(self, db_name: str) -> Optional[Dict]:
        """Get information about a specific database"""
        return self.config['databases'].get(db_name)

    def list_databases(self) -> List[Dict]:
        """List all databases with their info and stats"""
        databases = []
        for name, info in self.config['databases'].items():
            db_info = info.copy()
            db_info['stats'] = self.get_database_stats(name)
            databases.append({
                'name': name,
                **db_info
            })
        return databases

    def get_database_client(self, db_name: str) -> Optional[chromadb.PersistentClient]:
        """Get ChromaDB client for a specific database"""
        if db_name not in self.config['databases']:
            return None
            
        try:
            client, _ = init_chromadb(self.config['databases'][db_name]['path'])
            return client
        except Exception:
            return None
        
# In utils/db_manager_utils.py - Add this method to VectorDBManager class

def rename_database(self, old_name: str, new_name: str) -> bool:
    """Rename an existing database"""
    if old_name not in self.config['databases'] or new_name in self.config['databases']:
        return False

    try:
        old_path = self.config['databases'][old_name]['path']
        new_path = os.path.join(self.root_dir, new_name)
        
        # Move the database files
        shutil.move(old_path, new_path)
        
        # Update configuration
        self.config['databases'][new_name] = self.config['databases'][old_name]
        self.config['databases'][new_name]['path'] = new_path
        self.config['databases'][new_name]['last_modified'] = str(datetime.now())
        del self.config['databases'][old_name]
        
        # Update active database if necessary
        if self.config['active_db'] == old_name:
            self.config['active_db'] = new_name
            
        self._save_config()
        return True
    except Exception as e:
        print(f"Error renaming database: {e}")
        return False
    

    # In VectorDBManager class (utils/db_manager_utils.py)

def rename_database(self, old_name: str, new_name: str) -> bool:
    """Rename an existing database"""
    if old_name not in self.config['databases'] or new_name in self.config['databases']:
        return False

    try:
        old_path = self.config['databases'][old_name]['path']
        new_path = os.path.join(self.root_dir, new_name)
        
        # Move the database files
        shutil.move(old_path, new_path)
        
        # Update configuration
        self.config['databases'][new_name] = self.config['databases'][old_name].copy()
        self.config['databases'][new_name]['path'] = new_path
        self.config['databases'][new_name]['last_modified'] = str(datetime.now())
        del self.config['databases'][old_name]
        
        # Update active database if necessary
        if self.config['active_db'] == old_name:
            self.config['active_db'] = new_name
            
        self._save_config()
        return True
    except Exception as e:
        print(f"Error renaming database: {e}")
        return False