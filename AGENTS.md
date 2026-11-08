# Project Orchestration Rules (AGENTS.md) — 8b-agy Fan-Out Engine

## Integrated Subagent Topology
- **Wave 1 (Context Indexing & Pruning)**:
  - Run `smart-tree` (`st --mcp` or `st --mode ai`) to produce the compact directory/AST map into `.gemini/context_tree.txt`.
  - Check `engram` / `memnet` for existing structural invariants prior to staging file modifications.
- **Wave 2 (Parallel Fan-Out)**:
  - Subagents run in parallel as soon as their dependency bit is satisfied (`BitmaskDag::runnable_wave`).
  - Code edits must observe disjoint filesystem scopes (separate crates or dedicated `git worktree` checkouts).
  - Hardware tasks offload to LAN worker nodes or `RustyNanoKVM` via 64-byte aligned UDP frames (`LanFrame64`).
- **Wave 3 (Gather & Convergence)**:
  - Intermediate outputs write to `.gemini/artifacts/<task_id>.json`.
  - Assemble results and verify all touched crates with `cargo test` and `cargo clippy -D warnings`.
