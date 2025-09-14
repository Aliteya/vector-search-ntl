import pandas as pd
import numpy as np
from .text_processing import normalize

class Search:
    def __init__(self, database: pd.DataFrame):
        self.database = database
        if self.database.empty:
            return
        self.frequency_columns = self.database.columns[3:]
        self.doc_norms = np.sqrt(np.square(self.database[self.frequency_columns]).sum(axis=1))
    
    def search(self, query: str) -> pd.DataFrame:
        if self.database.empty:
            return pd.DataFrame()
        normalized_query = normalize(query)
        
        query_words = [word for word in set(normalized_query) if word in self.frequency_columns]
        
        if not query_words:
            return pd.DataFrame()
            
        up = self.database[query_words].sum(axis=1)
        down = np.sqrt(len(query_words)) * self.doc_norms
        
        scores = np.divide(up, down, out=np.zeros_like(up), where=down!=0)

        results = self.database[["TITLE", "URL"]].copy()
        results['SCORE'] = scores
        
        sorted_results = results[results['SCORE'] > 0.07].sort_values(by='SCORE', ascending=False)
        
        if sorted_results.empty:
            return pd.DataFrame()

        found_docs_vectors = self.database.loc[sorted_results.index][query_words]
        
        found_words_list = found_docs_vectors.apply(
            lambda row: row[row > 0].index.tolist(), 
            axis=1
        )

        final_results = pd.DataFrame({
            'TITLE': sorted_results['TITLE'],
            'URL': sorted_results['URL'],
            'found_words': found_words_list,
            'SCORE': sorted_results['SCORE']
        })
        
        return final_results.reset_index(drop=True)
