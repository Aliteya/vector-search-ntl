import pandas as pd
import numpy as np
import os
from .text_processing import normalize
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_colwidth', None) 
pd.set_option('display.width', 1000) 

class IndexUpdater:
    def __init__(self, watch_path="local_fs", db_path="processors/data.csv"):
        self.watch_path = watch_path
        self.db_path = db_path
        self.columns = ["TITLE", "URL", "TEXT"]
        
        self.tf_database = pd.DataFrame(columns=self.columns)
        self.tfidf_database = pd.DataFrame(columns=self.columns)

        logging.info("Creating a new in-memory index.")
        self.initial_crawl()

    def initial_crawl(self):
        logging.info(f'Starting initial sync with directory: {self.watch_path}')
        if not os.path.exists(self.watch_path):
            os.makedirs(self.watch_path)
        
        self.tf_database = pd.DataFrame(columns=self.columns)
        
        for root, dirs, files in os.walk(self.watch_path):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file_addition(file_path)
        
        self.tf_database.fillna(0, inplace=True)
        self.set_weights()
        logging.info("In-memory index is ready.")

    def add_file(self, file_path: str):
        self._process_file_addition(file_path)
        self.tf_database.fillna(0, inplace=True)
        self.set_weights()
        logging.info(f"In-memory index updated for file: {file_path}")

    def _process_file_addition(self, file_path: str):
        file_title = os.path.splitext(os.path.basename(file_path))[0]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_text = f.read()
        except Exception:
            file_text = ''

        if file_path in self.tf_database['URL'].values:
            self.tf_database = self.tf_database[self.tf_database['URL'] != file_path].reset_index(drop=True)

        main_data = pd.DataFrame([[file_title, file_path, file_text]], columns=self.columns)
        tokenized_sentence = self.tokenize(file_text)
        indexing_data = pd.DataFrame([tokenized_sentence], dtype=pd.Int8Dtype())
        data = pd.concat([main_data, indexing_data], axis=1)
        self.tf_database = pd.concat([self.tf_database, data], axis=0, ignore_index=True)

    def remove_file(self, file_path: str):
        if file_path in self.tf_database['URL'].values:
            self.tf_database = self.tf_database[self.tf_database['URL'] != file_path].reset_index(drop=True)
            logging.info(f"File removed from in-memory TF index: {file_path}")
            self.set_weights()

    def set_weights(self):
        logging.info("Recalculating TF-IDF weights using vectorized operations...")
        if self.tf_database.empty:
            self.tfidf_database = pd.DataFrame(columns=self.columns)
            return
        
        self.tfidf_database = self.tf_database.copy()

        frequency_columns = [col for col in self.tfidf_database.columns if col not in self.columns]
        if not frequency_columns:
            logging.info("No words to index. Skipping weight calculation.")
            return
            
        num_docs = len(self.tfidf_database)
        
        doc_freqs = (self.tfidf_database[frequency_columns] > 0).sum(axis=0)
        
        idf = np.log((1 + num_docs) / (1 + doc_freqs)) + 1
        
        self.tfidf_database[frequency_columns] = self.tfidf_database[frequency_columns] * idf
        
        logging.info("Weights recalculated successfully.")
        
    def save_database(self):
        logging.info(f"Saving TF index to {self.db_path}...")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.tf_database.to_csv(self.db_path)
        logging.info("Index saved successfully.")

    @staticmethod
    def tokenize(sentence: str) -> dict:
        normalized = normalize(sentence)
        frequency = {word: normalized.count(word) for word in set(normalized)}
        return frequency


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, indexer: IndexUpdater):
        self.indexer = indexer

    def on_created(self, event):
        if not event.is_directory:
            logging.info(f"File created: {event.src_path}")
            self.indexer.add_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            logging.info(f"File modified: {event.src_path}")
            self.indexer.add_file(event.src_path) 

    def on_deleted(self, event):
        if not event.is_directory:
            logging.info(f"File deleted: {event.src_path}")
            self.indexer.remove_file(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            logging.info(f"File moved: from {event.src_path} to {event.dest_path}")
            self.indexer.remove_file(event.src_path)
            self.indexer.add_file(event.dest_path)
