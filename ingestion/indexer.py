import faiss
import numpy as np

class Indexer:
    def __init__(self, dimension=768):
        self.index = faiss.IndexFlatL2(dimension)

    def index_embeddings(self, embeddings):
        self.index.add(embeddings)

    def save_index(self, index_file_path):
        faiss.write_index(self.index, index_file_path)
