from rag.prompt_build import PromptBuilder
from langchain_core.runnables import Runnable
from typing import List, Dict, Any, Optional

class PromptBuilderRunnable(Runnable):
    def __init__(self, builder: PromptBuilder):
        self.build = builder or PromptBuilder()

    def invoke(self, input: Dict[str, Any], config: Optional[Dict] = None,  **kwargs: Any) -> str:
        query = input["query"]
        chunks = input["chunks"]
        return self.build.build_prompt(query, chunks)