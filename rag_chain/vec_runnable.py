from langchain_core.runnables import Runnable
from typing import List, Dict, Any, Optional
import numpy as np
from embed.vec_store import VecStore

class VectorSearchRunnable(Runnable):
    def __init__(self, vec_str: VecStore, top_k:int = 5):
        self.vec_str = vec_str
        self.top_k = top_k
    
    def invoke(self,input: np.ndarray, config: Optional[Dict] = None, **kwargs: Any) -> List[Dict]:
        return self.vec_str.search(input, k=self.top_k)
