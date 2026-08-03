from clients import client_groq
from pathlib import Path
import gradio as gr

knowledge = {}

directory = Path("/Users/koushiksr/Documents/workspace/allrepos_microdegree/genaimasterclass/knowledge-base/employees")
files = [p for p in directory.rglob("*") if p.is_file()]

# creating knowledge dictionary from files {filename: file_content}
for filename in files:
    name = Path(filename).stem.split(' ')[-1]
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()

SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""
# this func will clean alpha and space charecter from string
clean_alpha_space_in_string = lambda msg: ''.join(ch for ch in msg if ch.isalpha() or ch.isspace()).lower().split()

# this func will get message from the user and return the relevant context
additional_context = lambda msg: "\n\n".join(ctx) if (ctx := [knowledge[w] for w in clean_alpha_space_in_string(msg) if w in knowledge]) else f"no context found {ctx}"

def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    system_message = SYSTEM_PREFIX + additional_context(message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response =  client_groq.chat.completions.create(model="openai/gpt-oss-120b", messages=messages)
    return response.choices[0].message.content

gr.ChatInterface(fn=chat).launch(inbrowser=True,share=True)