from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
# from loader import PDFLoad

def DocChunk(docs: List[Dict], chunk_size=512, chunk_overlap=50) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunked = []

    for doc in docs:
        for i, chunk in enumerate(splitter.split_text(doc['content'])):
            chunked.append({
                "content": chunk,
                "metadata": {
                    **doc['metadata'],
                    "chunk_id": i
                }
            })
    
    return chunked

# res = PDFLoad("data/test_doc.pdf")
# chk = DocChunk(res)
# print(len(res), len(chk))