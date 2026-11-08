# Project Orchestration Rules (AGENTS.md) — 8b-agy Fan-Out Engine (v0.2.0)

## Integrated Subagent & Ollama Router Topology
- **Wave 1 (Context Indexing & Ollama Routing)**:
  - Run `smart-tree` (`st --mcp` or `st --mode ai`) to produce the compact directory/AST map into `.gemini/context_tree.txt`.
  - Classify and route tasks through `OllamaWaveRouter` (`8b-agy-router:latest` @ `http://127.0.0.1:11434`) using exact `Q15` (`p/q` out of `32,768`) affinity scoring.
- **Wave 2 (Parallel Subagent & LAN Fan-Out)**:
  - Subagents run in parallel as soon as their dependency bit is satisfied (`BitmaskDag::runnable_wave`):
    - **Worker `0` (`deepseek-v4-pro-qwen35-heretic`)**: [`xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic`](https://huggingface.co/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic) (`deepseek-v4-pro-qwen3.5-9b-heretic:latest`, Apple Silicon `mlx-quant` `nvfp4` `4/1` seal `0x019f` / `mxfp8` `2/1` seal `0x0189`) for uncensored GSPO reasoning, math/STEM derivations, and adversarial auditing.
    - **Worker `1` (`qwen4-exp-8b-kompress-ultra`)**: [`PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra`](https://huggingface.co/PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra) for `#![no_std]` `p/q` Rust + Ada 2022 SPARK verification.
    - **Worker `4` (`qwen3.6-35b-a3b-coding`)**: `qwen3.6:35b-a3b-coding-nvfp4` for heavy multi-crate refactoring.
    - **Worker `5` (`bonsai-vision-kvm`)**: `bonsai-vision:latest` + `RustyNanoKVM` via 64-byte aligned UDP frames (`LanFrame64`).
- **Wave 3 (Gather & Convergence)**:
  - Intermediate outputs write to `.gemini/artifacts/<task_id>.json`.
  - Assemble results and verify all touched crates with `cargo test` and `cargo clippy -- -D warnings`.
