import json
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SparseSearchIndex:
    def __init__(self, file_paths):
        self.chunks = []
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.chunks.append(json.loads(line.strip()))
        self.texts = [c['text'] for c in self.chunks]
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def sparse_search(self, query, top_k=50):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.doc_vectors).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for i in top_indices:
            chunk = self.chunks[i]
            chunk['sparse_score'] = float(scores[i])
            results.append(chunk)
        return results
