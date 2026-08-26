import time
import jax
import jax.numpy as jnp
from workloads.attention import naive_attention, create_inputs

def benchmark(functions, inputs, iters: int = 100) -> float:
    """Benchmark a JAX function by running it multiple times 
    and measuring the average execution time (in seconds)."""
    # Warm up
    result = functions(*inputs)
    result.block_until_ready()

    start_time = time.perf_counter()
    for _ in range(iters):
        result = functions(*inputs)
        result.block_until_ready()

    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / iters
    return avg_time

def check_correctness(
    baseline, 
    candidate,
    inputs,
    atol: float = 1e-5,
    rtol: float = 1e-5
    ) -> bool:
    """Check if the candidate function produces results close to the baseline function."""
    baseline_output = baseline(*inputs)
    candidate_output = candidate(*inputs)
    baseline_output.block_until_ready()
    candidate_output.block_until_ready()

    return bool(
        jnp.allclose(
            baseline_output, 
            candidate_output, 
            atol=atol, 
            rtol=rtol
        )
    )

def main():
    inputs = create_inputs()
    correct = check_correctness(naive_attention, naive_attention, inputs)
    runtime = benchmark(naive_attention, inputs)

    print(f"Correctness check: {correct}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Average runtime: {runtime * 1000:.3f} ms")

if __name__ == "__main__":
    main()