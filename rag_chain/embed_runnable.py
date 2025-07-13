from langchain_core.runnables import Runnable
import numpy as np
from typing import Any, Optional, Dict
from embed.embedder import Embedder

class EmbeddingRunnable(Runnable):
    def __init__(self, embedder:Embedder):
        self.embedder = embedder

    def invoke(self, input: str, config: Optional[Dict] = None, **kwargs: Any) -> np.ndarray:
        return self.embedder.embed_query(input)
    