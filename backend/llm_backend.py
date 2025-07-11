from llama_cpp import Llama
from typing import Optional

class LLM:
    def __init__(self, 
                 model_path: str= "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", 
                 n_ctx: int = 2048,
                 n_threads: int = 6,
                 n_gpu_layers: int = 20,
                 temperature: float = 0.7,
                 top_p: float = 0.95):
        self.llm = Llama(
                model_path =model_path, 
                 n_ctx = n_ctx,
                 n_threads = n_threads,
                 n_gpu_layers = n_gpu_layers,
                 temperature = temperature,
                 top_p= top_p,
                 verbose=False
            )
        
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        output = self.llm(prompt=prompt, max_tokens=max_tokens, stop=["</s>"],echo=False)
        return output["choices"][0]["text"].strip()


        