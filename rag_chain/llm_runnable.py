from backend.llm_backend import LLM
from langchain_core.runnables import Runnable
from typing import Any, Optional, Dict

class LLMRunnable(Runnable):
    def __init__(self, llm_backend: LLM):
        self.llm_backend = llm_backend
    
    def invoke(self, query:str, config: Optional[Dict] = None, **kwargs: Any) -> str:
        return self.llm_backend.generate(query)