import faiss
from .crawler import VectorIndexer
from .text_processing import normalize

class VectorSearch:
    def __init__(self, indexer: VectorIndexer):
        self.model = indexer.model
        self.index = indexer.index
        self.id_to_path = indexer.id_to_path

    def search(self, query: str, top_k=5) -> list:
        if self.index.ntotal == 0:
            return []

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding.astype('float32'))
        
        similarities, indices = self.index.search(query_embedding, top_k)   
        normalized_query_words = set(normalize(query))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx in self.id_to_path: 
                file_path = self.id_to_path[idx]
                
                found_words = []
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        normalized_content_words = set(normalize(content))
                        found_words = list(normalized_query_words.intersection(normalized_content_words))
                except Exception:
                    found_words = []
                results.append({
                    'path': self.id_to_path[idx],
                    'score': similarities[0][i],
                    'found_words': found_words 
                })
        
        return results