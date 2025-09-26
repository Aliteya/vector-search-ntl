import pandas as pd
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import logging
from watchdog.events import FileSystemEventHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_colwidth', None) 
pd.set_option('display.width', 1000) 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class VectorIndexer:
    def __init__(self, watch_paths: list, index_path="vector_index.faiss", map_path="path_map.pkl"):
        self.watch_paths = watch_paths
        self.index_path = index_path
        self.map_path = map_path
        
        logging.info("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.d = self.model.get_sentence_embedding_dimension()

        self.index = None
        self.path_to_id = {}
        self.id_to_path = {}

        self.load_index()
        if not self.path_to_id: 
            logging.info("Creating a new vector index.")
            self.initial_crawl()
        else:
            logging.info("Loaded existing index.")

    def initial_crawl(self):
        for path in self.watch_paths:
            logging.info(f'Starting initial sync with directory: {path}')
            if not os.path.exists(path):
                os.makedirs(path)
            
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.add_file(file_path)
        
        self.save_index()
        logging.info(f"Vector index is ready for paths: {self.watch_paths}")

    def add_file(self, file_path: str):
        if file_path in self.path_to_id:
            logging.info(f"File '{file_path}' exists. Re-indexing.")
            self.remove_file(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                logging.warning(f"File '{file_path}' is empty. Skipping.")
                return

            embedding = self.model.encode([content])

            faiss.normalize_L2(embedding.astype('float32'))
            
            new_id = self.index.ntotal
            self.index.add(embedding)

            self.path_to_id[file_path] = new_id
            self.id_to_path[new_id] = file_path
            
            logging.info(f"Added/Updated file: {file_path} with ID: {new_id}")

        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}")

    def remove_file(self, file_path: str):
        if file_path not in self.path_to_id:
            logging.warning(f"Attempted to remove non-existent file: {file_path}")
            return

        file_id = self.path_to_id[file_path]
        del self.path_to_id[file_path]
        del self.id_to_path[file_id]
        
        logging.info(f"Removed file from maps: {file_path} with ID: {file_id}")
    
    def save_index(self):
        logging.info(f"Saving index to {self.index_path}...")
        faiss.write_index(self.index, self.index_path)
        
        with open(self.map_path, 'wb') as f:
            pickle.dump({'path_to_id': self.path_to_id, 'id_to_path': self.id_to_path}, f)
        logging.info("Index and map saved successfully.")

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            logging.info("Loading index from disk...")
            self.index = faiss.read_index(self.index_path)
            with open(self.map_path, 'rb') as f:
                maps = pickle.load(f)
                self.path_to_id = maps['path_to_id']
                self.id_to_path = maps['id_to_path']
        else:
            logging.info("No existing index found. Initializing a new one.")
            self.index = faiss.IndexFlatIP(self.d)


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, indexer: VectorIndexer):
        self.indexer = indexer

    def on_created(self, event):
        if not event.is_directory:
            self.indexer.add_file(event.src_path)
            self.indexer.save_index()

    def on_modified(self, event):
        if not event.is_directory:
            self.indexer.add_file(event.src_path)
            self.indexer.save_index()

    def on_deleted(self, event):
        if not event.is_directory:
            self.indexer.remove_file(event.src_path)
            self.indexer.save_index()

    def on_moved(self, event):
        if not event.is_directory:
            self.indexer.remove_file(event.src_path)
            self.indexer.add_file(event.dest_path)
            self.indexer.save_index()