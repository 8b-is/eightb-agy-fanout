---
name: deepseek-v4-pro-qwen35-heretic
description: Uncensored/abliterated DeepSeek-V4-Pro distilled onto Qwen3.5-9B (32-layer 3:1 hybrid GDN + full attention + MTP, 262K context, GSPO RL) running via Mac Ollama and Apple Silicon Metal mlx-quant within the 8b-agy Wave-2 fan-out engine.
---

# `deepseek-v4-pro-qwen35-heretic` Subagent & Skill (`8b-agy`)

## Model Identity
- **Hugging Face**: `https://huggingface.co/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic`
- **Architecture**: `Qwen3_5ForConditionalGeneration` (`32` layers: `24` GDN `linear_attention` + `8` `full_attention` + `1` `MTP` head, `262,144` context)
- **Mac Ollama**: `deepseek-v4-pro-qwen3.5-9b-heretic:latest` (routed by `8b-agy-router:latest` on `127.0.0.1:11434`)
- **Apple Silicon `mlx-quant`**: `nvfp4` (`4-bit`, `group_size=64`, `p/q = 4/1`, `0x62` sealed) & `mxfp8` (`8-bit`, `group_size=64`, `p/q = 2/1`)

## Invocation
```bash
# Route a reasoning/codegen task through 8b-agy Ollama router -> deepseek-v4-pro-qwen35-heretic
8b-agy route "Prove symplectic shear invariant in Q15 fixed-point without f64"

# Inspect MLX-quant & Ollama constellation
8b-agy models
8b-agy mlx-quant
```
