import faiss
import numpy as np
import os
import json
from typing import List, Tuple, Dict

class VecStore:
    def __init__(self, dim: int, index_path: str = "database.faiss", meta_path: str = "metadata.json"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)
        self.meta_store = []
    
    def add_embed(self, data: List[Tuple[np.ndarray, Dict]]):
        vectors = np.array([vec for vec, _ in data]).astype(np.float32)
        metadata = [m for _, m in data]

        self.index.add(vectors)
        self.meta_store.extend(metadata)

    def search(self, query: np.ndarray, k:int = 5) -> List[Dict]:
        q_vec = np.array([query]).astype(np.float32)
        dist, idx = self.index.search(q_vec, k)
        res = []

        for rank, i in enumerate(idx[0]):
            if i < len(self.meta_store):
                temp = self.meta_store[i].copy()
                temp["score"] = float(dist[0][rank])
                res.append(temp)
        
        return res

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta_store, f, ensure_ascii=False, indent=4)
    
    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta_store = json.load(f)

    