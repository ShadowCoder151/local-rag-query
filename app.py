from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from ingest.loader import PDFLoad
from ingest.chunker import DocChunk
from embed.embedder import Embedder
from embed.vec_store import VecStore
from pprint import pprint

res = PDFLoad("data/test_doc.pdf")
chk = DocChunk(res)

embedder = Embedder()
embd = embedder.embed_docs(chk)

dim = embd[0][0].shape[0]
store = VecStore(dim=dim)
store.add_embed(embd)
store.save()

store.load()
query = "What is an large language model?"
q_vec = embedder.embed_query(query)
res = store.search(q_vec, k=3)



pprint(embd[0])
print(len(chk), len(embd))
pprint(res)