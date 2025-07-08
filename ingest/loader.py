import os
from typing import List, Dict
import fitz

SUPPORTED = [".pdf"]

def PDFLoad(file_path: str) -> List[Dict]:
    doc = fitz.open(file_path)
    results = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            results.append({
                "content": text,
                "metadata": {
                    "filename" : os.path.basename(file_path),
                    "page_number": page_num + 1
                }
            })
        
    return results

# res = PDFLoad("data/test_doc.pdf")
# print(len(res))