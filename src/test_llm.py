
from llm import generate

def main():
    prompt = """
You are helping optimize JAX programs.

Briefly explain what JAX JIT compilation does.
Answer in at most three sentences.
"""
    response = generate(prompt)
    print("Prompt:")
    print(prompt)
    print("Response:")
    print(response)

if __name__ == "__main__":
    main()