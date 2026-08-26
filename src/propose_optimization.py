from pathlib import Path

from llm import generate

SOURCE_FILE = Path("src/workloads/attention.py")
IR_FILE = Path("artifacts/attention_stablehol.mlir")


def build_prompt(source: str, stablehlo_ir: str) -> str:
    return f"""
You are optimizing a JAX program.

Analyze the JAX source code and its StableHLO representation.

Your job is NOT to rewrite the code yet.

Return:
1. Bottleneck
2. Proposed optimization
3. Why it could improve performance
4. Risks to numerical correctness

Keep the response concise.

=== JAX SOURCE CODE ===

{source}

=== STABLEHLO IR ===

{stablehlo_ir}
"""

def main():
    source = SOURCE_FILE.read_text()
    stablehlo_ir = IR_FILE.read_text()
    prompt = build_prompt(source, stablehlo_ir)
    response = generate(prompt)

    print("Response from LLM:")
    print(response)

if __name__ == "__main__":
    main()