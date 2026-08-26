from pathlib import Path

from llm import generate

SOURCE_FILE = Path("src/workloads/attention.py")
CANDIDATE_FILE = Path("artifacts/candidate_attention.py")


def build_prompt(source: str) -> str:
    return f"""
You are optimizing a JAX implementation.

Rewrite the naive_attention function to improve runtime performance.

Requirements:
- Preserve the function name: naive_attention
- Preserve the function arguments: query, key, value
- Use only JAX and jax.numpy
- Preserve numerical behaviour as closely as possible
- Return only valid Python code
- Do not include markdown fences
- Include necessary imports
- Do not include explanations

=== JAX SOURCE CODE ===

{source}
"""

def clean_code(code: str) -> str:
    code = code.strip()

    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    
    if code.startswith("```"):
        code = code[len("```"):].strip()
    
    if code.endswith("```"):
        code = code[:-len("```")].strip()

    return code

def main():
    source = SOURCE_FILE.read_text()
    candidate = generate(build_prompt(source))
    candidate = clean_code(candidate)

    CANDIDATE_FILE.parent.mkdir(exist_ok=True)
    CANDIDATE_FILE.write_text(candidate)
    print(f"Candidate implementation written to {CANDIDATE_FILE}")

if __name__ == "__main__":
    main()