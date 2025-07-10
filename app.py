from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from ingest.loader import PDFLoad
from ingest.chunker import DocChunk
from embed.embedder import Embedder
from pprint import pprint

res = PDFLoad("data/test_doc.pdf")
chk = DocChunk(res)

embedder = Embedder()
embd = embedder.embed_docs(chk)

pprint(embd[0])
print(len(chk), len(embd))