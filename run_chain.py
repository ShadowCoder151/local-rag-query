from rag_chain.embed_runnable import EmbeddingRunnable, Embedder
from rag_chain.vec_runnable import VectorSearchRunnable, VecStore
from rag_chain.prompt_runnable import PromptBuilderRunnable, PromptBuilder
from rag_chain.llm_runnable import LLMRunnable, LLM
from pprint import pprint


embd = Embedder()
embd_run = EmbeddingRunnable(embd)

vec_str = VecStore(dim=384)
vec_str.load()
search_run = VectorSearchRunnable(vec_str, 3)

query = "What are the advancements in large language models?"
res1 = embd_run.invoke(query)
res2 = search_run.invoke(res1)


builder = PromptBuilder()
prompt_run = PromptBuilderRunnable(builder)
test_input = {"query": query, "chunks": res2}
res3 = prompt_run.invoke(test_input)

llm = LLM()
llm_run = LLMRunnable(llm)

ret_chain = embd_run | search_run

final_chain = {"query": lambda x:x, "chunks": ret_chain} | prompt_run | llm_run

response = final_chain.invoke(query)
print(response)


# print(res)
# print(res.shape)