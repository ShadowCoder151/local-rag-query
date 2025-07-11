from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from ingest.loader import PDFLoad
from ingest.chunker import DocChunk

from embed.embedder import Embedder
from embed.vec_store import VecStore

from rag.retriever import Retriever
from rag.prompt_build import PromptBuilder

from backend.llm_backend import LLM

from pprint import pprint

# res = PDFLoad("data/test_doc.pdf")
# chk = DocChunk(res)

embedder = Embedder()
# embd = embedder.embed_docs(chk)

# dim = embd[0][0].shape[0]
# store = VecStore(dim=dim)
# store.add_embed(embd)
# store.save()


store = VecStore(dim=384)
store.load()


query = "What is safety in pretraining?"
ret = Retriever(embedder, store, 3)
res = ret.retrieve(query)

build = PromptBuilder()
prompt = build.build_prompt(query, res)

llm = LLM()
response = llm.generate(prompt)

# q_vec = embedder.embed_query(query)
# res = store.search(q_vec, k=3)



# pprint(embd[0])
# print(len(chk), len(embd))
# pprint(sorted(res, key=lambda x:x["score"]))
print(f"Response: {response}")