from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
# from ingest.loader import PDFLoad
# from ingest.chunker import DocChunk
from pprint import pprint


class Embedder:
    def __init__(self, model_name: str="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_docs(self, docs:List[Dict]) -> List[Tuple[np.ndarray, Dict]]:
        texts = [doc["content"] for doc in docs]
        metas = [doc["metadata"] for doc in docs]

        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        return [(embedding, {
                "content": doc["content"],
                "metadata": doc["metadata"]
            }) for embedding, doc in zip(embeddings, docs)]
    
    def embed_query(self, query:str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True)
    

# res = PDFLoad("data/test_doc.pdf")
# chk = DocChunk(res)

# embedder = Embedder()
# embd = embedder.embed_docs(chk)
