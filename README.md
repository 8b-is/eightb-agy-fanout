# `eightb-agy-fanout` (`8b-agy` v0.2.0)

[![Forge: Codeberg](https://img.shields.io/badge/Codeberg-8b--is%2Feightb--agy--fanout-2185D0?logo=codeberg)](https://codeberg.org/8b-is/eightb-agy-fanout)
[![Forge: GitHub](https://img.shields.io/badge/GitHub-8b--is%2Feightb--agy--fanout-181717?logo=github)](https://github.com/8b-is/eightb-agy-fanout)
[![Heretic Subagent](https://img.shields.io/badge/HF-xtools--at%2FDeepSeek--V4--Pro--Qwen3.5--9B--UNCENSORED--heretic-FFD21E)](https://huggingface.co/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic)
[![Qwen4-Exp SOTA](https://img.shields.io/badge/HF-PeetPedro%2Fqwen4__exp--8b--obliterated--abliterated--kompress--ultra-FF6F00)](https://huggingface.co/PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra)

**Zero-allocation single-register 64-bit DAG fan-out scheduler (`BitmaskDag`), Ollama Wave Router (`8b-agy-router:latest`), Apple Silicon Metal `mlx-quant` bridge (`64B@0x62`), and `RustyNanoKVM` USB HID gadget engine for Antigravity CLI.**

---

## 1. Architecture & 3-Wave Fan-Out Pipeline

```mermaid
flowchart LR
    subgraph Wave1["Wave 1: Index & Route"]
        ST["smart-tree (st --mcp)"] --> ROUTER["Ollama Router (8b-agy-router:latest @ 127.0.0.1:11434)"]
    end
    subgraph Wave2["Wave 2: Parallel Subagent & LAN Fan-Out (BitmaskDag)"]
        ROUTER -->|"worker=0 (32686/32768 Q15)"| HERETIC["deepseek-v4-pro-qwen35-heretic (Qwen3.5-9B 24 GDN + 8 Full + 1 MTP)"]
        ROUTER -->|"worker=1 (32686/32768 Q15)"| QWEN4["qwen4-exp-8b-kompress-ultra (#![no_std] p/q Rust + SPARK)"]
        ROUTER -->|"worker=4 (31360/32768 Q15)"| MOE["qwen3.6:35b-a3b-coding-nvfp4 (48L Heavy MoE)"]
        ROUTER -->|"worker=5 (30848/32768 Q15)"| KVM["bonsai-vision:latest + RustyNanoKVM (/dev/hidg0)"]
    end
    subgraph Wave3["Wave 3: Gather & Verify"]
        HERETIC --> CONV["LanFrame64 (0x8B15 · Gate 0x62) + cargo clippy -D warnings"]
        QWEN4 --> CONV
        MOE --> CONV
        KVM --> CONV
    end
```

---

## 2. Fan-Out Model Constellation & Ollama Router (`127.0.0.1:11434`)

`eightb-agy-fanout` uses **`8b-agy-router:latest`** (built on `d3:latest` `qwen3_5` `mxfp8` + exact rational `Q15` scoring out of `32,768` with zero `f64` drift) to route DAG tasks across 6 specialized local Ollama / Apple Silicon `mlx-quant` / subagent nodes:

| Worker ID | Subagent / Node ID | Hugging Face Source | Local Mac Ollama Tag | MLX-Quant (`p/q`) | Context | Role |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **`0`** | `deepseek-v4-pro-qwen35-heretic` | [`xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic`](https://huggingface.co/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic) | `deepseek-v4-pro-qwen3.5-9b-heretic:latest` | `nvfp4` (`4/1`) / `mxfp8` (`2/1`) | `262,144` | `HereticReasoningSubagent` (GSPO RL, STEM/math, uncensored reasoning) |
| **`1`** | `qwen4-exp-8b-kompress-ultra` | [`PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra`](https://huggingface.co/PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra) | `qwen3.6-fast:latest` | `nvfp4` (`4/1`) | `262,144` | `PrimarySotaCodegen` (`#![no_std]` `p/q` Rust + Ada 2022 SPARK) |
| **`2`** | `8b-agy-router` | [`8b-is/eightb-agy-fanout`](https://codeberg.org/8b-is/eightb-agy-fanout) | `8b-agy-router:latest` | `mxfp8` (`2/1`) | `262,144` | `OllamaWaveRouter` (Wave-1/Wave-2 task classification & dispatch) |
| **`3`** | `kev-decision-engine` | `8b-is/kev-7.9b-nvfp4` | `kev:latest` | `nvfp4` (`4/1`) | `73,728` | `FastDecisionClassifier` (Sub-millisecond triage & tool gating) |
| **`4`** | `qwen3.6-35b-a3b-coding` | `Qwen/Qwen3.6-35B-A3B-Coding` | `qwen3.6:35b-a3b-coding-nvfp4` | `nvfp4` (`4/1`) | `131,072` | `HeavyMoECodingVerifier` (Workspace-wide multi-crate synthesis) |
| **`5`** | `bonsai-vision-kvm` | `8b-is/bonsai-vision-27b` | `bonsai-vision:latest` | `affine` (`8/1`) | `262,144` | `MultimodalKvmVision` (`RustyNanoKVM` HDMI capture & USB HID injection) |

---

## 3. Mac `ollama` + Apple Silicon `mlx-quant` Setup (`DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic`)

### Architectural Geometry (`Qwen3_5ForConditionalGeneration`)
- **Hybrid 3:1 Backbone (`32` Layers)**: `24` Gated DeltaNet (`linear_attention`, `linear_num_key_heads=16`, `linear_num_value_heads=32`, `head_dim=128`, `conv_kernel=4`) + `8` Full Attention layers (`full_attention_interval=4`, `num_attention_heads=16`, `num_key_value_heads=4`, `head_dim=256`) + `1` Multi-Token Prediction (`MTP`) head.
- **Dimensions**: `hidden_size = 4096`, `intermediate_size = 12288`, `max_position_embeddings = 262,144`.
- **Apple Silicon Metal `mlx-quant` Seals (`64B@0x62` `LitanyWireFrame`)**:
  - **`nvfp4` (`bits=4`, `group_size=64`)**: Exact compression ratio `p/q = 4/1`, `litany_wire_seal16 = 0x019f`, `spqt_root_base243 = [149, 233, 212]`.
  - **`mxfp8` (`bits=8`, `group_size=64`)**: Exact compression ratio `p/q = 2/1`, `litany_wire_seal16 = 0x0189`, `spqt_root_base243 = [131, 233, 212]`.

### Register Models in Mac `ollama`
```bash
# 1. Register the Heretic reasoning subagent model & the 8b-agy Wave Router
ollama create deepseek-v4-pro-qwen3.5-9b-heretic:latest -f models/Modelfile.deepseek-v4-pro-qwen3.5-9b-heretic
ollama create 8b-agy-router:latest -f models/Modelfile.8b-agy-router

# (Optional) Direct GGUF pull from Hugging Face Hub:
ollama run hf.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF:Q4_K_M
```

---

## 4. Installation & CLI Usage (`8b-agy`)

### Install from Codeberg / Local Workspace
```bash
git clone https://codeberg.org/8b-is/eightb-agy-fanout.git
cd eightb-agy-fanout
cargo install --path . --force
```

### CLI Commands
```bash
# 1. Execute 3-Wave BitmaskDag fan-out demo with Ollama Wave Router
8b-agy demo

# 2. Inspect the 6-model fan-out constellation (HF repos, Ollama tags, MLX p/q ratios)
8b-agy models

# 3. Inspect Apple Silicon Metal mlx-quant 64B@0x62 seals for DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic
8b-agy mlx-quant

# 4. Route any task prompt via Ollama Wave Router (127.0.0.1:11434) and emit a 64B@0x62 LanFrame64
8b-agy route "Derive symplectic shear theorem with uncensored heretic reasoning and Q15 p/q math"
8b-agy route "Prove #![no_std] Q15 GRPO advantage in Ada 2022 SPARK on Qwen4-Exp"

# 5. Start the 64-byte aligned UDP LAN worker daemon (default: 0.0.0.0:18815)
8b-agy daemon 0.0.0.0:18815
```

---

## 5. `LanFrame64` Wire Format (`64` Bytes, `#[repr(C, align(64))]`)

| Byte Offset | Field | Type | Description |
| :---: | :--- | :---: | :--- |
| `0..2` | `magic` | `u16` | `0x8B15` constellation magic |
| `2..3` | `gate` | `u8` | `0x62` Litany seal gate |
| `3..4` | `opcode` | `u8` | `Heartbeat (0x01)` · `TaskDispatch (0x02)` · `TaskComplete (0x03)` · `KvmHidInject (0x04)` · `OllamaRoute (0x05)` · `MlxQuantSeal (0x06)` · `SubagentWave (0x07)` |
| `4..6` | `worker_id` | `u16` | Target node index (`0..63`) |
| `6..8` | `task_id` | `u16` | DAG task or `TaskDomain` discriminant (`0..63`) |
| `8..16` | `sequence` | `u64` | Monotonic tap counter |
| `16..24` | `runnable_mask` | `u64` | Active/runnable wave bitmask (`BitmaskDag::runnable_wave`) |
| `24..32` | `completed_mask` | `u64` | Completed wave bitmask or `Q15` rational affinity score |
| `32..56` | `payload` | `[u8; 24]` | Inline Ollama model tag, MLX `LitanyWireFrame` seal, or `HidKeyboardReport` |
| `56..64` | `fold_tag` | `u64` | 64-bit SWAR XOR-fold integrity tag over bytes `0..56` |

---

## 6. Verification

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```
