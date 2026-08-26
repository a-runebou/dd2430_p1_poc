
from pathlib import Path
from workloads.attention import naive_attention, create_inputs

OUTPUT_DIR = Path("artifacts")

def main():
    query, key, value = create_inputs()
    lowered = naive_attention.lower(query, key, value)
    stable_hlo = str(lowered.compiler_ir(dialect="stablehlo"))

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "attention_stablehol.mlir"
    output_file.write_text(stable_hlo)

    print(f"StableHLO IR written to {output_file}")

if __name__ == "__main__":
    main()