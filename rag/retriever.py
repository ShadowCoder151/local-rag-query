from typing import List, Dict
from embed.embedder import Embedder
from embed.vec_store import VecStore
import numpy as np

class Retriever:
    def __init__(self, embedder: Embedder, vec_store: VecStore, k:int = 5):
        self.embedder = embedder
        self.vec_store = vec_store
        self.k = k
    
    def retrieve(self, query: str) -> List[Dict]:
        q_vec = self.embedder.embed_query(query)
        res = self.vec_store.search(q_vec, k=3)
        return res
    