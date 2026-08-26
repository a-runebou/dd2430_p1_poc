import importlib.util
from pathlib import Path

from benchmark import benchmark, check_correctness
from workloads.attention import naive_attention, create_inputs

CANDIDATE_FILE = Path("artifacts/candidate_attention.py")

def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location(
        "candidate_attention", path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_attention

def main():
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(f"Candidate file {CANDIDATE_FILE} does not exist.")

    candidate = load_candidate(CANDIDATE_FILE)
    inputs = create_inputs()

    # Check correctness
    correct = check_correctness(naive_attention, candidate, inputs)
    print(f"Correctness check: {correct}")

    if not correct:
        print("Candidate implementation is not correct. REJECT.")
        return

    baseline_runtime = benchmark(naive_attention, inputs)
    candidate_runtime = benchmark(candidate, inputs)
    speedup = baseline_runtime / candidate_runtime

    print(f"Baseline average runtime: {baseline_runtime * 1000:.3f} ms")
    print(f"Candidate average runtime: {candidate_runtime * 1000:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")

    if candidate_runtime < baseline_runtime:
        print("Candidate implementation is faster. ACCEPT.")
    else:
        print("Candidate implementation is not faster. REJECT.")

if __name__ == "__main__":
    main()
