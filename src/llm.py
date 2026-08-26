
import os

import requests

# Environment variables for Ollama config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

def generate(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"]