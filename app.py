import gradio as gr
import torch
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Local model loading
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Hugging Face inference API client setup
client = InferenceClient("microsoft/Phi-3.5-mini-instruct", token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


# Function to handle chatbot response generation
def respond(message, system_message, max_tokens, temperature, top_p, use_local_model):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message},
    ]

    if use_local_model:
        # Use the local model for inference
        response = pipe(messages, max_new_tokens=max_tokens, temperature=temperature, do_sample=True, top_p=top_p)
        generated_text = response[0]['generated_text']
    else:
        # Use Hugging Face Inference API
        response = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        generated_text = ''.join([chunk.choices[0].delta.content for chunk in response])

    return generated_text


# Custom CSS to style the Gradio interface
custom_css = """
#main-container {
    background-color: #f0f0f0;
    font-family: 'Arial', sans-serif;
}

.gradio-container {
    max-width: 750px;
    margin: 0 auto;
    padding: 25px;
    background: #ffffff;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    border-radius: 12px;
}

.gr-button {
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.gr-button:hover {
    background-color: #0056b3;
}

.gr-slider input {
    color: #007bff;
}

.gr-chat {
    font-size: 15px;
    line-height: 1.6;
    background-color: #f1f5f9;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}

#title {
    text-align: center;
    font-size: 2.2em;
    margin-bottom: 25px;
    color: #333;
}

#use-local-label {
    font-weight: bold;
    margin-right: 10px;
}

#use-local-checkbox {
    margin-left: 5px;
}
"""

# Gradio interface with local vs. API inference option
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 id='title'>🌟 AI Chatbot Interface 🌟</h1>")

    with gr.Row():
        system_message = gr.Textbox(value="You are a friendly chatbot.", label="System Message")
        use_local_model = gr.Checkbox(label="Use Local Model", value=False, elem_id="use-local-checkbox")

    with gr.Row():
        max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max New Tokens")
        temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p")

    chat_history = gr.Chatbot(label="Chat")

    user_input = gr.Textbox(placeholder="Type your message here...")
    submit_button = gr.Button("Send")

    submit_button.click(
        fn=respond,
        inputs=[user_input, system_message, max_tokens, temperature, top_p, use_local_model],
        outputs=chat_history
    )

if __name__ == "__main__":
    demo.launch(share=False)