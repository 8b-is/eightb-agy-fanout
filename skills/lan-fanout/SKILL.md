---
name: lan-fanout
description: Fan out independent DAG waves across local subagents and LAN RustyNanoKVM worker nodes using 8b-agy, smart-tree, and engram.
---

# LAN Fan-Out Skill (`8b-agy`)

## Architecture & Integration
This skill orchestrates parallel Antigravity (`agy`) subagent fan-outs using:
- **`smart-tree`**: Fast semantic codebase AST map and project context dump.
- **`memnet` / `engram`**: Conditional associative memory lookup across agent iterations.
- **`eightb-agy-fanout` (`8b-agy`)**: Zero-alloc single-register 64-bit Bitmask DAG fan-out engine with cache-line aligned 64-byte UDP frame transport (`LanFrame64`).
- **`RustyNanoKVM`**: Hardware-in-the-loop RISC-V USB HID injection (`/dev/hidg0`, `/dev/hidg1`) and HDMI frame capture.

## Execution Phases
1. **Wave 1 — Index & Ground**:
   ```bash
   st --mode ai > .gemini/context_tree.txt
   ```
2. **Wave 2 — Parallel Fan-Out**:
   - Subagents execute concurrently on disjoint crates/branches.
   - Remote/hardware tasks dispatch to LAN workers via `8b-agy daemon` UDP port `18815`.
3. **Wave 3 — Assembly & Verification**:
   - Aggregate outputs from `.gemini/artifacts/`.
   - Run `cargo test --all-targets` and `cargo clippy -- -D warnings`.
