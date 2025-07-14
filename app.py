import gradio as gr
from run_chain import final_chain

def chat_with_rag(query: str) -> str:
    try:
        return final_chain.invoke(query)
    except Exception as e:
        return f"[Error] {str(e)}"

with gr.Blocks(title="Local RAG Chatbot") as demo:
    gr.Markdown("## 🧠 Local RAG Chatbot (no API, in-process)")

    with gr.Row():
        chatbot = gr.Textbox(label="Chat History", interactive=False, lines=12)
    query = gr.Textbox(label="Ask a question", placeholder="Type your query and hit enter...")
    submit = gr.Button("Submit")

    def handle_submit(q):
        response = chat_with_rag(q)
        return f"🧑 You: {q}\n🤖 Bot: {response}"

    submit.click(handle_submit, inputs=[query], outputs=[chatbot])

demo.launch()
