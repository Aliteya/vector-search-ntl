import pandas as pd
import numpy as np
from text_processing import normalize

class Search:
    def __init__(self, database: str = "data.csv"):
        self.database = pd.read_csv(database, index_col=0)
        self.frequency_columns = self.database.columns[3:]
        self.doc_norms = np.sqrt(np.square(self.database[self.frequency_columns]).sum(axis=1))

    def search(self, query: str) -> pd.DataFrame: 
        query = normalize(query)
        
        query = [word for word in set(query) if word in self.frequency_columns]
        
        if not query:
            return pd.DataFrame()
           
        up = self.database[query].sum(axis=1)

        down = np.sqrt(len(query)) * self.doc_norms
        
        scores = np.divide(up, down, out=np.zeros_like(up), where=down!=0)

        results = self.database[["title", "url"]].copy()
        results['score'] = scores
        
        sorted_results = results[results['score'] > 0].sort_values(by='score', ascending=False).reset_index(drop=True)
        return sorted_results

if __name__ == '__main__':
    # Создаем экземпляр поисковика, указывая путь к нашему индексу
    search_engine = Search(database="processors/data.csv")
    
    # Выполняем поиск
    user_query = "russia"
    search_results = search_engine.search(user_query)
    
    # Выводим результаты
    print(f"\nSearch results for: '{user_query}'")
    print(search_results)
