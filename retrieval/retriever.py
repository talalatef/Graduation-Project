import faiss
import numpy as np

class Retriever:
    def __init__(self, index_file_path):
        self.index = faiss.read_index(index_file_path)

    def retrieve(self, query_embedding, top_k=5):
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return indices[0]
