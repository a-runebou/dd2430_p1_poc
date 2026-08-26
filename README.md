# dd2430_p1_poc
## Overview
POC for an agentic AI compiler optimization system built around JAX and XLA.

The system uses a local LLM to inspect the workload of JAX and its compiler intermediate representation. The LLM generates an optimized candidate implementation, which is evaluated in terms of correctness and speedup compared to the original solution.

The current workload is a simple JAX implementation of scaled dot-product attention based on functions used in DD2424.

## Requirements
The project is designed to run inside Docker so that the Python and JAx environment is consistent across operating systems. 

You need:
- Docker
- Ollama running on the host machine (https://ollama.com/ or `curl -fsSL https://ollama.com/install.sh | sh`)
- An Ollama model, currently `qwen2.5-coder:7b` (`ollama pull qwen2.5-coder:7b`)

## Setup
Build
`docker compose build`

Generate StableHLO:
`docker compose run --rm app python src/inspect_ir.py`

Generate candidate:
`docker compose run --rm app python src/generate_candidate.py`

The generated implementation is saved to `artifacts/candidate_attention.py`

Evaluate:
`docker compose run --rm app python src/evaluate_candidate.py`


