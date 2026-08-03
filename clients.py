from openai import OpenAI
import os

client_openai = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

client_groq = OpenAI(
    api_key = os.getenv("GROQ_API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)

client_gemini = OpenAI(
    api_key = os.getenv("GOOGLE_API_KEY"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)