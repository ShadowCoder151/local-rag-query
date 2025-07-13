# local-rag-query
---

A fully local RAG system built using Python, FAISS, and using open source models for querying documents locally.

---

## Features (Milestones)

- [x] Ingestion and Vector Store
- [x] Retrieval and Prompt building
- [x] LLM inference engine
- [ ] CLI loop interface
- [x] Langchain orchestration
- [ ] UI integration
and more...


## Tech stack:

| Component | Library       |
|-----------| --------------|
|Language| Python 3.12+ and Poetry|
|PDF Loader| `pymupdf`|
|Chunking| Overlapping sliding window|
|Embeddings| `sentence-transformers`|
|Vector DB| `faiss-cpu`|
|Model |  `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (4 bit Quantized)|
|LLM inference| `llama-cpp-python`|

---

## Enviroment setup

### 1. Clone and setup:
```bash
git clone https://github.com/ShadowCoder151/local-rag-query
cd local-rag-query
pip install poetry
poetry install
```

### 2. Install llama-cpp-python
- Only for CPU, just directly
```bash
poetry add llama-cpp-python
```
- For GPU support:
```bash
set FORCE_CMAKE=1
set CMAKE_ARGS=-DGGML_CUDA=on
poetry run pip install llama-cpp-python
```
- Key points: 
    - Make sure Visual Studio compiler is set to only the x64 path.
    - MinGW should be disabled (temporarily delete the path in system variables) (worked for me since it was conflicting with my VS Build Tools)
    - Reference websites
        - https://llama-cpp-python.readthedocs.io/en/latest/
        - https://www.cognibuild.ai/building-llama-cpp-no-cuda-toolset-found

    

### 2. Create additional folders:
```bash
mkdir data # (For storing the documents)
mkdir models # (For storing the model file)
```

### 3. Download the model:
The model used here is [mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF?show_file_info=mistral-7b-instruct-v0.2.Q4_K_M.gguf)

After downloading, copy it to the `models` folder

---

## Project structure
```bash
├── .gitignore
├── .venv/
├── app.py      # Main entry file
├── run_chain.py      # Main entry file (with Langchain orchestration)
├── backend/    # LLM inference
│   └── llm_backend.py
├── chat/       # Implementing later
│   ├── message.py
│   └── session.py
├── config/     # Implementing later
│   └── settings.yaml
├── data/       # Store docs locally
│   └── test_doc.pdf
├── embed/      # Embedding + Storage
│   ├── embedder.py     # Chunk -> embeddings
│   └── vec_store.py    # Embeddings -> Storage
├── ingest/     # PDF ingestion
│   ├── chunker.py      # Text chunking
│   └── loader.py       # PDF Loader
├── LICENSE
├── models/     # storing models locally
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
├── poetry.lock
├── pyproject.toml
├── rag/
│   ├── prompt_build.py     # Chunks + query combined to give final prompt
│   └── retriever.py       # Query -> top chunks (results)
├── rag_chain/          # Modules (runnables) for lang chain orchestration
│   ├── embed_runnable.py       # Embedder runnable
│   └── llm_runnable.py         # LLM backend runnable
│   └── prompt_runnable.py      # Prompt generator runnable
│   └── vec_runnable.py         # Vector search runnbale
├── README.md
├── ui/         # UI implementing later
│   └── ui.py
└── utils/      # Later stages
    └── file_utils.py
```

## Run the CLI version
```bash
poetry run python run_chain.py
```
---

## Example result:
```bash
Query: (modified in run_chain.py and change the doc in data folder)
Response: In the context provided, "safety in pretraining" refers to the investigations and measures taken to ensure the safety and ethical implications of the data used during the pretraining phase of a language model like Llama 2-Chat....
```

---



