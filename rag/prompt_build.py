from typing import List, Dict

class PromptBuilder:
    def __init__(self):
        self.system_prompt = (
            "[INST] Use the following context to answer the user's question.\n\n" \
            "Context:\n----------------\n{context}\n-------------\n" \
            "Question: {query}\n[/INST]"
        )

    def build_prompt(self, query:str, contexts: List[Dict]) -> str:
        refer_content = [item["content"].strip() for item in contexts]
        combined_context = "\n\n".join(refer_content)
        return self.system_prompt.format(context=combined_context, query=query)