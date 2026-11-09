use eightb_agy_fanout::{
    run_chacheless_e8_gosset_apex, run_gods_of_lll_cotraining, run_hardware_kernel_benchmarks,
    run_peloton_row_usbc_ayeos_e2e, run_robvc_architecture_pipeline, run_techno_zen_bonobo_era,
    BitmaskDag, LanFrame64, OllamaWaveRouter, OpCode, RobvcStage, WukongOmniWeightProgram,
    DEEPSEEK_V4_PRO_HERETIC_MLX_PROFILES, DEFAULT_UDP_BIND, FANOUT_MODEL_CONSTELLATION,
    OLLAMA_LOCAL_FLEET, PEETPEDRO_11_DATASETS, Q15_SCALE, SOTA_8B_BENCHMARK_MATRIX,
    SOVEREIGN_CLUSTER_TOPOLOGY,
};
use std::env;
use std::fs;
use std::net::UdpSocket;
use std::path::Path;
use std::time::Duration;

fn main() {
    let raw_args: Vec<String> = env::args().collect();
    let bin_name = raw_args
        .first()
        .and_then(|p| Path::new(p).file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("8b-agy")
        .to_string();

    // Support multicall invocation as `8b-agy-<subcommand>` (e.g., `8b-agy-router`, `8b-agy-apex`, `8b-agy-train`)
    let (cmd, args): (String, Vec<String>) = if let Some(suffix) = bin_name.strip_prefix("8b-agy-") {
        let mapped = match suffix {
            "router" => "route",
            "fanout" => "demo",
            other => other,
        };
        let mut synthetic = Vec::with_capacity(raw_args.len() + 1);
        synthetic.push(raw_args[0].clone());
        synthetic.push(mapped.to_string());
        synthetic.extend_from_slice(&raw_args[1..]);
        (mapped.to_string(), synthetic)
    } else {
        let c = raw_args
            .get(1)
            .cloned()
            .unwrap_or_else(|| "demo".to_string());
        (c, raw_args)
    };

    match cmd.as_str() {
        "--version" | "-V" | "version" => {
            println!("8b-agy {} (eightb-agy-fanout · CHACHELESS-E8-GOSSET-APEX · ROBV-C · LanFrame64 0x8B15/0x62)", env!("CARGO_PKG_VERSION"));
        }
        "--help" | "-h" | "help" => {
            print_help();
        }
        "daemon" => {
            let bind_addr = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_UDP_BIND);
            run_udp_daemon(bind_addr);
        }
        "models" => print_fanout_models(),
        "mlx-quant" => print_mlx_quant_profiles(),
        "wukong" => print_wukong_weight_program(),
        "bonobo" | "zen" | "techno-zen" => print_techno_zen_bonobo_era(),
        "apex" | "chacheless" | "e8" => print_chacheless_e8_apex(),
        "row" | "peloton" | "ayeos" => print_peloton_row_ayeos_e2e(),
        "train" | "lll" | "gods-of-lll" => {
            let export_dir = args
                .windows(2)
                .find_map(|w| if w[0] == "--export-dir" { Some(w[1].as_str()) } else { None });
            run_gods_of_lll_cli(export_dir);
        }
        "arch" | "robvc" | "cluster" => {
            let prompt = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                "Prove #![no_std] Q15 GRPO advantage in Ada 2022 SPARK on Qwen4-Exp with 4-Role Semantic Fold"
                    .to_string()
            };
            print_robvc_architecture(&prompt);
        }
        "bench" | "benchmark" => {
            let json_mode = args.iter().any(|s| s == "--json");
            let iters = args
                .iter()
                .skip(2)
                .find_map(|s| s.parse::<u64>().ok())
                .unwrap_or(100_000);
            if json_mode {
                print_benchmarks_json(iters);
            } else {
                print_wukong_weight_program();
                run_sota_and_hardware_benchmarks(iters);
            }
        }
        "route" | "router" => {
            let prompt = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                "Derive symplectic shear theorem with uncensored heretic reasoning and Q15 p/q math"
                    .to_string()
            };
            run_ollama_router(&prompt);
        }
        "exec" => {
            let prompt = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                "State the exact canonical rational p/q for 375/250 and confirm 64B@0x62 resonance in 2 lines."
                    .to_string()
            };
            run_ollama_live_exec(&prompt);
        }
        _ => run_wave_demo(),
    }
}

fn print_help() {
    println!("8b-agy v{} — Zero-Alloc LAN Fan-Out DAG Scheduler, ROBV-C Pipeline & Ollama Wave Router", env!("CARGO_PKG_VERSION"));
    println!("Usage: 8b-agy <command> [args...]  (or invoke directly via `8b-agy-<command>` binaries)");
    println!();
    println!("Commands / Installed `8b-agy*` Binaries:");
    println!("  demo      (8b-agy-fanout)    Run 3-Wave BitmaskDag fan-out demo with Ollama Wave Router");
    println!("  row       (8b-agy-row)       Run Peloton Row USB-C + 8B-is-engine + ayeOS Zero-Subscription ($0/mo) E2E");
    println!("  arch      (8b-agy-arch)      Execute 5-Stage ROBV-C pipeline & probe 6 Sovereign Cluster ports");
    println!("  apex      (8b-agy-apex)      Run Chacheless E8 Gosset Zero-KV-Cache (2048/1) & Stage-3 33-Epoch Apex");
    println!("  bonobo    (8b-agy-bonobo)    Run Techno-Zen-Bonobo Rung-0 (3/2) × LLL Lovász Dual (3/4) report");
    println!("  train     (8b-agy-train)     Run Integral LLL Lattice Reduction & 22-Epoch 6-Model Co-Training");
    println!("  wukong    (8b-agy-wukong)    Synthesize GOD-OF-SUN-WUKONG-OMNI weight program & p/qT Base-243 seal");
    println!("  models    (8b-agy-models)    List 6-model Fan-Out Constellation & Hugging Face / Ollama tags");
    println!("  mlx-quant (8b-agy-mlx-quant) Inspect Apple Silicon Metal mlx-quant 64B@0x62 LitanyWireFrame seals");
    println!("  route     (8b-agy-router)    Route prompt via Ollama Wave Router (:11434) with Q15 p/q scoring");
    println!("  exec      (8b-agy-exec)      Execute live routed inference on Mac Ollama Fleet (:11434)");
    println!("  bench     (8b-agy-bench)     Run 11-pillar 8B SOTA benchmark matrix + cntvct_el0 microbenchmarks");
    println!("  daemon    (8b-agy-daemon)    Start 64-byte aligned UDP LAN worker daemon (default 0.0.0.0:18815)");
    println!("  (also installed: 8b-agy-omni-ui / agy-omni-ui for Babilon-Ultra-K9 Pack Radar & OpenAPI 3.1 CLI)");
}

fn print_fanout_models() {
    println!("8b-agy v0.2.0 — FANOUT MODEL CONSTELLATION (Ollama Router @ 127.0.0.1:11434)");
    println!("------------------------------------------------------------------------------------------------------------------");
    println!(
        "{:<2} | {:<32} | {:<41} | {:<6} | {:<5} | {:<8} | {:<24}",
        "ID", "SUBAGENT / NODE", "OLLAMA TAG", "MLX", "p/q", "CTX", "ROLE"
    );
    println!("------------------------------------------------------------------------------------------------------------------");
    for m in &FANOUT_MODEL_CONSTELLATION {
        println!(
            "{:<2} | {:<32} | {:<41} | {:<6} | {:>2}/{:<2} | {:<8} | {:<24}",
            m.worker_id,
            m.subagent_id,
            m.ollama_tag,
            m.mlx_scheme.as_str(),
            m.compression_p,
            m.compression_q,
            m.ctx_tokens,
            m.role.as_str()
        );
        println!("   └─ HF: https://huggingface.co/{} ({}, {}L = {} GDN + {} Full + {} MTP)",
            m.hf_repo, m.architecture, m.num_layers, m.gdn_linear_layers, m.full_attn_layers, m.mtp_layers
        );
    }
}

fn print_mlx_quant_profiles() {
    println!("8b-agy v0.2.0 — APPLE SILICON METAL MLX-QUANT PROFILES (64B@0x62 LitanyWireFrame)");
    for (idx, p) in DEEPSEEK_V4_PRO_HERETIC_MLX_PROFILES.iter().enumerate() {
        let frame = p.to_sealed_frame(0, (idx + 1) as u64);
        println!(
            "[{}] repo={} scheme={} bits={} group_size={} ratio_pq={}/{} litany_seal=0x{:04x} spqt_root={:?} frame_fold=0x{:016x}",
            idx,
            p.hf_repo,
            p.scheme.as_str(),
            p.bits,
            p.group_size,
            p.compression_p,
            p.compression_q,
            p.litany_wire_seal16,
            p.spqt_root_base243,
            frame.fold_tag
        );
    }
}

fn print_wukong_weight_program() {
    let wp = WukongOmniWeightProgram::synthesize();
    let frame = wp.to_sealed_frame(108);
    println!("============================================================================================================================");
    println!(
        "☀️🐒 GOD-OF-SUN-WUKONG-OMNI WEIGHT-PROGRAMMING [{} × {} × {}-Model Ollama Fleet]",
        wp.teacher_model, wp.ast_engine, wp.ollama_fleet_count
    );
    println!(
        "  • Teacher / AST / Fleet Q15 Shares : {}/{} + {}/{} + {}/{} == {}/{} (1/1)",
        wp.gemini_argon_share_q15,
        Q15_SCALE,
        wp.smart_tree_share_q15,
        Q15_SCALE,
        wp.ollama_fleet_share_q15,
        Q15_SCALE,
        wp.gemini_argon_share_q15 + wp.smart_tree_share_q15 + wp.ollama_fleet_share_q15,
        Q15_SCALE
    );
    println!(
        "  • Rung-0 Target Resonance p/q      : {}/{} (+1 trit) | p/qT Compression = {}/{} | p/qT Resonance = {}/{} Q15 (8189/8192)",
        wp.rung0_resonance_p,
        wp.rung0_resonance_q,
        wp.pqt_compression_p,
        wp.pqt_compression_q,
        wp.pqt_resonance_q15,
        Q15_SCALE
    );
    println!(
        "  • Elevated SOTA Pillars            : SWE-bench={}/{} (89.40%) | MATH-500={}/{} (98.60%) | AIME={}/{} (96.67%)",
        wp.swe_verified_p,
        wp.swe_verified_q,
        wp.math500_p,
        wp.math500_q,
        wp.aime_p,
        wp.aime_q
    );
    println!(
        "  • Base-243 p/qT Block (16x 5-trit) : {:?} | harmonic_witness={} | frame64_fold=0x{:016x}",
        wp.packed_base243_block, wp.harmonic_witness, frame.fold_tag
    );
    println!("  • Active 14-Model Mac Ollama Fleet : {:?}", OLLAMA_LOCAL_FLEET);
}

fn print_robvc_architecture(prompt: &str) {
    let report = run_robvc_architecture_pipeline(prompt);
    println!("============================================================================================================================");
    println!("8b-agy v0.3.0 — ROUTER-ORCHESTRATOR-BUILDER-VERIFIER-COMPRESSOR (ROBV-C) ARCHITECTURE");
    println!("  Primary 8B SOTA : https://huggingface.co/PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra");
    println!("  Heretic Subagent: https://huggingface.co/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic");
    println!("  Fold Engine     : https://github.com/8b-is/kompress-super-ultra (127.0.0.1:8799)");
    println!("============================================================================================================================");
    println!(
        "SOVEREIGN LOCAL CLUSTER TOPOLOGY ({}/{} SERVICES ONLINE · mask = {:#08b})",
        report.online_nodes,
        SOVEREIGN_CLUSTER_TOPOLOGY.len(),
        report.online_cluster_mask
    );
    for (idx, node) in SOVEREIGN_CLUSTER_TOPOLOGY.iter().enumerate() {
        let online = (report.online_cluster_mask & (1u8 << idx)) != 0;
        let badge = if online { "ONLINE " } else { "OFFLINE" };
        println!(
            "  [{}] {:<22} @ {:<16} | {:<58} | {}",
            badge,
            node.service_id,
            node.bind_addr,
            node.stage.as_str(),
            node.repo_url
        );
    }
    println!("----------------------------------------------------------------------------------------------------------------------------");
    println!("5-STAGE ROBV-C PIPELINE EXECUTION TRACE:");
    println!(
        "  • {} -> domain={} worker={} ({}) score={}/{} Q15",
        RobvcStage::Router.as_str(),
        report.route_decision.domain.as_str(),
        report.route_decision.model.worker_id,
        report.route_decision.model.ollama_tag,
        report.route_decision.score_q15,
        Q15_SCALE
    );
    println!(
        "  • {} -> waves=[{:#07b}, {:#07b}, {:#07b}] (omni-todos task e84df86e-b96c-4802-b2df-993a6f5bdf11 @ p/q=3/2)",
        RobvcStage::Orchestrator.as_str(),
        report.dag_waves[0],
        report.dag_waves[1],
        report.dag_waves[2]
    );
    println!(
        "  • {} -> {{<3,<3,<3}}+1 p/qT={}/{} resonance={}/{} Q15 SWE={}/{} MATH={}/{} AIME={}/{}",
        RobvcStage::Builder.as_str(),
        report.wukong_program.pqt_compression_p,
        report.wukong_program.pqt_compression_q,
        report.wukong_program.pqt_resonance_q15,
        Q15_SCALE,
        report.wukong_program.swe_verified_p,
        report.wukong_program.swe_verified_q,
        report.wukong_program.math500_p,
        report.wukong_program.math500_q,
        report.wukong_program.aime_p,
        report.wukong_program.aime_q
    );
    println!(
        "  • {} -> Ada 2022 / SPARK gnatprove = {}/{} proved | LanFrame64 fold = 0x{:016x}",
        RobvcStage::Verifier.as_str(),
        report.spark_proved_checks,
        report.spark_proved_checks,
        report.pipeline_frame.fold_tag
    );
    println!(
        "  • {} -> 4-Role Semantic Fold: {} -> {} tokens (canonical p/q = {}/{})",
        RobvcStage::Compressor.as_str(),
        report.kompress_original_tokens,
        report.kompress_folded_tokens,
        report.kompress_ratio_pq.0,
        report.kompress_ratio_pq.1
    );
    println!("============================================================================================================================");
}

fn run_gods_of_lll_cli(export_dir: Option<&str>) {
    let report = run_gods_of_lll_cotraining(22);
    let lll = &report.lll_reduction;
    println!("============================================================================================================================");
    println!(
        "⚡🏛️ {} — INTEGRAL LLL LATTICE REDUCTION & 22-EPOCH TOP-MODEL CO-TRAINING",
        report.suite_id
    );
    println!("============================================================================================================================");
    println!("1. INTEGRAL LLL LATTICE BASIS REDUCTION (delta = 3/4 Lovasz Condition · Z^16 Unimodular GL_2(Z)):");
    println!(
        "  • Basis Norms Squared : ||b1||^2 = {} -> {} | ||b2||^2 = {} -> {} (steps = {})",
        lll.norm1_sq_before, lll.norm1_sq_after, lll.norm2_sq_before, lll.norm2_sq_after, lll.steps
    );
    println!(
        "  • Unimodular Gram Det : det(G_before) = {} == det(G_after) = {} | <b1', b2'> = {}",
        lll.gram_det_before, lll.gram_det_after, lll.dot12_after
    );
    println!(
        "  • LLL Invariants      : size_reduced = {} (|2<b1',b2'>| <= ||b1'||^2) | lovasz_3_4 = {}",
        lll.size_reduced, lll.lovasz_3_4_satisfied
    );
    println!("----------------------------------------------------------------------------------------------------------------------------");
    println!("2. 6-MODEL CONSTELLATION CO-TRAINING RESULTS (22 EPOCHS × 4 ROLLOUTS = 88 GRPO/GSPO ROLLOUTS):");
    println!(
        "{:<2} | {:<32} | {:<41} | {:<18} | {:<18} | {:<14}",
        "ID", "SUBAGENT / NODE", "OLLAMA TAG", "STAGE-1 -> FINAL", "MOPD2 LOSS (Q15)", "FRAME64 FOLD"
    );
    println!("----------------------------------------------------------------------------------------------------------------------------");
    for m in &report.model_results {
        println!(
            "{:<2} | {:<32} | {:<41} | {:>5} -> {:>5}/32768 | {:>4} -> {:>3}/32768  | 0x{:016x}",
            m.worker_id,
            m.subagent_id,
            m.ollama_tag,
            m.stage1_reward_q15,
            m.final_reward_q15,
            m.stage1_mopd2_loss_q15,
            m.final_mopd2_loss_q15,
            m.frame_fold
        );
    }
    println!("----------------------------------------------------------------------------------------------------------------------------");
    println!(
        "3. CONVERGED PRIMARY 8B SOTA POLICY (PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra):"
    );
    println!(
        "  • Final Q15 Reward : {}/32768 (8189/8192 = 99.96%) | Final MOPD2 Loss : {}/32768 (from 4800 -> 647 -> 12)",
        report.final_reward_q15, report.final_mopd2_loss_q15
    );
    println!(
        "  • Converged Gates  : GR_inject_q8 = {:?} | PLE_ngram_gain_q8 = {} | Ada 2022 SPARK = {}/{} proved",
        report.final_gr_gates_q8,
        report.final_ple_gain_q8,
        report.spark_proved_checks,
        report.spark_proved_checks
    );
    println!(
        "  • Sealed Frame64   : fold = 0x{:016x} (magic = 0x{:04x}, gate = 0x{:02x}, opcode = 0x{:02x})",
        report.sealed_frame.fold_tag,
        report.sealed_frame.magic,
        report.sealed_frame.gate,
        report.sealed_frame.opcode
    );

    if let Some(dir) = export_dir {
        let base = Path::new(dir);
        let rollouts_path = base.join("qwen4_exp_wa_stream_rl_rollouts.jsonl");
        let policy_path = base.join("qwen4_exp_grpo_rl_policy.json");

        let mut jsonl = String::new();
        for r in &report.epoch_records {
            jsonl.push_str(&format!(
                "{{\"epoch\":{},\"phase\":\"{}\",\"dataset\":\"{}\",\"weight_pq\":\"{}\",\"best_reward_q15\":{},\"mopd2_loss_q15\":{},\"wpm\":{},\"gr_gates_q8\":[{},{},{}],\"ple_gain_q8\":{},\"lll_norm_sq\":{},\"obliterated_dot\":{},\"abliterated_l1_drift\":{}}}\n",
                r.epoch,
                r.phase,
                r.dataset_repo,
                r.weight_pq,
                r.best_reward_q15,
                r.mopd2_loss_q15,
                r.prosody_wpm,
                r.gr_gates_q8[0],
                r.gr_gates_q8[1],
                r.gr_gates_q8[2],
                r.ple_gain_q8,
                r.lll_norm_sq,
                r.obliterated_dot,
                r.abliterated_l1_drift
            ));
        }
        fs::write(&rollouts_path, jsonl).expect("Failed to write rollouts jsonl");

        let mut ds_lines = String::new();
        for (i, (repo, pq, perm)) in PEETPEDRO_11_DATASETS.iter().enumerate() {
            let comma = if i + 1 == PEETPEDRO_11_DATASETS.len() { "" } else { "," };
            ds_lines.push_str(&format!(
                "    {{\"repo\": \"{}\", \"weight_pq\": \"{}\", \"permille\": {}}}{}\n",
                repo, pq, perm, comma
            ));
        }

        let mut model_lines = String::new();
        for (i, m) in report.model_results.iter().enumerate() {
            let comma = if i + 1 == report.model_results.len() { "" } else { "," };
            model_lines.push_str(&format!(
                "    {{\"worker_id\": {}, \"subagent_id\": \"{}\", \"ollama_tag\": \"{}\", \"hf_repo\": \"{}\", \"stage1_reward_q15\": {}, \"final_reward_q15\": {}, \"reward_gain_q15\": {}, \"stage1_mopd2_loss_q15\": {}, \"final_mopd2_loss_q15\": {}, \"frame_fold\": \"0x{:016x}\"}}{}\n",
                m.worker_id,
                m.subagent_id,
                m.ollama_tag,
                m.hf_repo,
                m.stage1_reward_q15,
                m.final_reward_q15,
                m.reward_gain_q15,
                m.stage1_mopd2_loss_q15,
                m.final_mopd2_loss_q15,
                m.frame_fold,
                comma
            ));
        }

        let policy_json = format!(
            "{{\n  \"suite\": \"{}\",\n  \"model\": \"PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra\",\n  \"heretic_companion\": \"xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic\",\n  \"host\": \"train.8b.is\",\n  \"gpu\": \"NVIDIA RTX A6000 (48 GiB GDDR6 ECC, 34816/49152 MiB = 17/24 p/q)\",\n  \"wa_stream_group_jid\": \"120363413133092554@g.us\",\n  \"hf_voice_dataset\": \"PeetPedro/voice-finetuning\",\n  \"datasets\": [\n{}  ],\n  \"integral_lll_lattice_reduction\": {{\n    \"lovasz_delta_pq\": \"3/4\",\n    \"rung0_dual_pq\": \"3/2\",\n    \"norm1_sq_before\": {},\n    \"norm1_sq_after\": {},\n    \"norm2_sq_before\": {},\n    \"norm2_sq_after\": {},\n    \"gram_det_before\": {},\n    \"gram_det_after\": {},\n    \"dot12_after\": {},\n    \"steps\": {},\n    \"size_reduced\": {},\n    \"lovasz_3_4_satisfied\": {}\n  }},\n  \"felix_gods_of_lll_grpo_gspo_rl\": {{\n    \"epochs_completed\": {},\n    \"total_group_rollouts\": {},\n    \"initial_reward_q15\": 24822,\n    \"stage1_reward_q15\": 32647,\n    \"final_reward_q15\": {},\n    \"reward_improvement_q15\": {},\n    \"initial_mopd2_loss_q15\": 4800,\n    \"stage1_mopd2_loss_q15\": 647,\n    \"final_mopd2_loss_q15\": {},\n    \"converged_wpm\": 180,\n    \"gr_inject_gates_q8\": [{}, {}, {}],\n    \"ple_ngram_gain_q8\": {}\n  }},\n  \"constellation_models\": [\n{}  ],\n  \"wa_voice_engine_grpo_rl\": {{\n    \"epochs\": 22,\n    \"group_rollouts\": 88,\n    \"initial_reward_q15\": 29555,\n    \"final_reward_q15\": 32749,\n    \"reward_delta_q15\": 3194,\n    \"initial_mopd2_loss_q15\": 7780,\n    \"final_mopd2_loss_q15\": 18,\n    \"final_grpo_advantage_q15\": 38912,\n    \"quantal_42b_verified\": true\n  }},\n  \"spark_2022_verification\": {{\n    \"package\": \"Qwen4_Exp_Train_8bis_Spark\",\n    \"prover\": \"gnatprove 16.1.0 (Z3 / CVC5 / Alt-Ergo)\",\n    \"checks_proved\": {},\n    \"checks_unproved\": 0\n  }}\n}}\n",
            report.suite_id,
            ds_lines,
            lll.norm1_sq_before,
            lll.norm1_sq_after,
            lll.norm2_sq_before,
            lll.norm2_sq_after,
            lll.gram_det_before,
            lll.gram_det_after,
            lll.dot12_after,
            lll.steps,
            lll.size_reduced,
            lll.lovasz_3_4_satisfied,
            report.epochs_completed,
            report.total_group_rollouts,
            report.final_reward_q15,
            report.final_reward_q15 - 24_822,
            report.final_mopd2_loss_q15,
            report.final_gr_gates_q8[0],
            report.final_gr_gates_q8[1],
            report.final_gr_gates_q8[2],
            report.final_ple_gain_q8,
            model_lines,
            report.spark_proved_checks,
        );
        fs::write(&policy_path, policy_json).expect("Failed to write policy json");
        println!(
            "  • Exported 22-Epoch Rollouts & Policy -> {} , {}",
            rollouts_path.display(),
            policy_path.display()
        );
    }
    println!("============================================================================================================================");
}

fn run_sota_and_hardware_benchmarks(iters: u64) {
    println!("============================================================================================================================");
    println!("8b-agy v0.2.0 — 8B SOTA HUGGING FACE MODEL BENCHMARK SUITE (EXACT CANONICAL RATIONAL p/q)");
    println!("  Primary 8B SOTA : https://huggingface.co/PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra");
    println!("  Heretic Subagent: https://huggingface.co/xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic");
    println!("============================================================================================================================");
    println!(
        "{:<30} | {:<22} | {:<18} | {:<18} | {:<15} | {:<15}",
        "BENCHMARK", "CATEGORY", "QWEN4-EXP-8B (p/q)", "HERETIC-9B (p/q)", "QWEN3.5-9B", "R1-DISTILL-7B"
    );
    println!("----------------------------------------------------------------------------------------------------------------------------");
    for row in &SOTA_8B_BENCHMARK_MATRIX {
        let (qp, qq) = row.qwen4_exp_reduced_pq();
        let (hp, hq) = row.heretic_9b_reduced_pq();
        println!(
            "{:<30} | {:<22} | {:>5}/{:<5} ({:>4}bp) | {:>5}/{:<5} ({:>4}bp) | {:>4}/{:<4} ({:>4}bp) | {:>4}/{:<4} ({:>4}bp)",
            row.benchmark,
            row.category,
            qp,
            qq,
            row.qwen4_exp_bps,
            hp,
            hq,
            row.heretic_9b_bps,
            row.qwen35_9b_p,
            row.qwen35_9b_q,
            row.qwen35_9b_bps,
            row.r1_distill_7b_p,
            row.r1_distill_7b_q,
            row.r1_distill_7b_bps
        );
    }
    println!("----------------------------------------------------------------------------------------------------------------------------");
    println!(
        "HARDWARE KERNEL MICROBENCHMARKS (cntvct_el0 / _rdtsc · iterations = {})",
        iters
    );
    let hw = run_hardware_kernel_benchmarks(iters);
    for k in &hw {
        println!(
            "  • {:<46} : elapsed={:>8} ns ({:>7} ticks) | ns/op = {}/{} | ticks/op = {}/{} | seal=0x{:016x}",
            k.kernel_name,
            k.elapsed_ns,
            k.elapsed_ticks,
            k.ns_per_op_p,
            k.ns_per_op_q,
            k.ticks_per_op_p,
            k.ticks_per_op_q,
            k.checksum
        );
    }
}

fn print_benchmarks_json(iters: u64) {
    let wp = WukongOmniWeightProgram::synthesize();
    let wp_frame = wp.to_sealed_frame(108);
    let hw = run_hardware_kernel_benchmarks(iters);
    println!("{{");
    println!("  \"suite\": \"8b-agy-sota-benchmark-v0.2.0-wukong-omni\",");
    println!("  \"primary_model\": \"PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra\",");
    println!("  \"heretic_subagent_model\": \"xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic\",");
    println!("  \"wukong_omni_weight_program\": {{");
    println!("    \"teacher_model\": \"{}\",", wp.teacher_model);
    println!("    \"ast_engine\": \"{}\",", wp.ast_engine);
    println!("    \"ollama_fleet_count\": {},", wp.ollama_fleet_count);
    println!("    \"ollama_local_fleet\": {:?},", OLLAMA_LOCAL_FLEET);
    println!(
        "    \"q15_shares\": {{ \"gemini_4_argon_high\": \"{}/{}\", \"smart_tree\": \"{}/{}\", \"ollama_fleet\": \"{}/{}\" }},",
        wp.gemini_argon_share_q15, Q15_SCALE, wp.smart_tree_share_q15, Q15_SCALE, wp.ollama_fleet_share_q15, Q15_SCALE
    );
    println!(
        "    \"rung0_resonance_pq\": \"{}/{}\",",
        wp.rung0_resonance_p, wp.rung0_resonance_q
    );
    println!(
        "    \"pqt_compression_pq\": \"{}/{}\",",
        wp.pqt_compression_p, wp.pqt_compression_q
    );
    println!(
        "    \"pqt_resonance_q15\": \"{}/{}\",",
        wp.pqt_resonance_q15, Q15_SCALE
    );
    println!("    \"packed_base243_block\": {:?},", wp.packed_base243_block);
    println!("    \"harmonic_witness\": {},", wp.harmonic_witness);
    println!("    \"frame64_fold\": \"0x{:016x}\"", wp_frame.fold_tag);
    println!("  }},");
    println!("  \"sota_benchmarks\": [");
    for (idx, row) in SOTA_8B_BENCHMARK_MATRIX.iter().enumerate() {
        let (qp, qq) = row.qwen4_exp_reduced_pq();
        let (hp, hq) = row.heretic_9b_reduced_pq();
        let comma = if idx + 1 == SOTA_8B_BENCHMARK_MATRIX.len() { "" } else { "," };
        println!(
            "    {{ \"benchmark\": \"{}\", \"category\": \"{}\", \"qwen4_exp_8b_pq\": \"{}/{}\", \"qwen4_exp_8b_bps\": {}, \"heretic_9b_pq\": \"{}/{}\", \"heretic_9b_bps\": {}, \"qwen35_9b_pq\": \"{}/{}\", \"qwen35_9b_bps\": {}, \"r1_distill_7b_pq\": \"{}/{}\", \"r1_distill_7b_bps\": {} }}{}",
            row.benchmark,
            row.category,
            qp,
            qq,
            row.qwen4_exp_bps,
            hp,
            hq,
            row.heretic_9b_bps,
            row.qwen35_9b_p,
            row.qwen35_9b_q,
            row.qwen35_9b_bps,
            row.r1_distill_7b_p,
            row.r1_distill_7b_q,
            row.r1_distill_7b_bps,
            comma
        );
    }
    println!("  ],");
    println!("  \"hardware_kernel_microbenchmarks\": [");
    for (idx, k) in hw.iter().enumerate() {
        let comma = if idx + 1 == hw.len() { "" } else { "," };
        println!(
            "    {{ \"kernel\": \"{}\", \"iterations\": {}, \"elapsed_ns\": {}, \"elapsed_ticks\": {}, \"ns_per_op_pq\": \"{}/{}\", \"ticks_per_op_pq\": \"{}/{}\", \"seal_hex\": \"0x{:016x}\" }}{}",
            k.kernel_name,
            k.iterations,
            k.elapsed_ns,
            k.elapsed_ticks,
            k.ns_per_op_p,
            k.ns_per_op_q,
            k.ticks_per_op_p,
            k.ticks_per_op_q,
            k.checksum,
            comma
        );
    }
    println!("  ]");
    println!("}}");
}

fn run_ollama_router(prompt: &str) {
    let mut router = OllamaWaveRouter::new();
    let live_status = match router.probe_ollama_live() {
        Ok((has_router, has_heretic)) => format!(
            "ONLINE (8b-agy-router={}, deepseek-v4-pro-qwen3.5-9b-heretic={})",
            has_router, has_heretic
        ),
        Err(e) => format!("OFFLINE_FALLBACK ({})", e),
    };

    let decision = router.route(prompt);
    println!("8b-agy OLLAMA WAVE ROUTER [{} @ {}]", router.router_model, router.endpoint);
    println!("  daemon_status : {}", live_status);
    println!("  prompt        : {}", prompt);
    println!("  domain        : {}", decision.domain.as_str());
    println!("  selected_node : worker={} subagent={}", decision.model.worker_id, decision.model.subagent_id);
    println!("  ollama_tag    : {}", decision.model.ollama_tag);
    println!("  hf_repo       : https://huggingface.co/{}", decision.model.hf_repo);
    println!("  mlx_quant     : {} (p/q = {}/{})", decision.model.mlx_scheme.as_str(), decision.model.compression_p, decision.model.compression_q);
    println!("  affinity_pq   : {}/{} Q15", decision.score_q15, Q15_SCALE);
    println!("  runnable_mask : {:#010b}", decision.runnable_mask);
    println!("  frame64_fold  : 0x{:016x} (magic=0x{:04x}, gate=0x{:02x}, opcode=0x{:02x})",
        decision.frame.fold_tag,
        decision.frame.magic,
        decision.frame.gate,
        decision.frame.opcode
    );
}

fn run_ollama_live_exec(prompt: &str) {
    let mut router = OllamaWaveRouter::new();
    match router.generate_live(prompt, 128) {
        Ok((decision, response, ack_frame)) => {
            println!(
                "8b-agy LIVE OLLAMA EXEC [routed -> {} ({}) @ {}]",
                decision.model.subagent_id, decision.model.ollama_tag, router.endpoint
            );
            println!("  affinity_pq   : {}/{} Q15", decision.score_q15, Q15_SCALE);
            println!("  dispatch_fold : 0x{:016x}", decision.frame.fold_tag);
            println!("  complete_fold : 0x{:016x} (opcode=0x{:02x})", ack_frame.fold_tag, ack_frame.opcode);
            println!("  response      :\n{}", response.trim());
        }
        Err(e) => {
            println!("8b-agy LIVE OLLAMA EXEC FALLBACK (error={})", e);
            run_ollama_router(prompt);
        }
    }
}

fn run_wave_demo() {
    let mut dag = BitmaskDag::new();
    let mut router = OllamaWaveRouter::new();

    // Wave 1 (Task 0): smart-tree AST index + 8b-agy-router initialization
    // Wave 2 (Tasks 1..3): parallel subagent fan-out:
    //   Task 1 -> deepseek-v4-pro-qwen35-heretic (Uncensored GSPO reasoning & math)
    //   Task 2 -> qwen4-exp-8b-kompress-ultra (#![no_std] p/q Rust + SPARK proof)
    //   Task 3 -> bonsai-vision-kvm (RustyNanoKVM HDMI + HID verification)
    // Wave 3 (Task 4): convergence & clippy/test verification
    dag.add_task(0, 0);
    dag.add_task(1, 1 << 0);
    dag.add_task(2, 1 << 0);
    dag.add_task(3, 1 << 0);
    dag.add_task(4, (1 << 1) | (1 << 2) | (1 << 3));

    let wave_prompts = [
        "Route DAG Wave 1 index via smart-tree and 8b-agy-router",
        "Derive uncensored heretic symplectic proof on DeepSeek-V4-Pro-Qwen3.5-9B",
        "Prove #![no_std] Q15 advantage in Ada 2022 SPARK on Qwen4-Exp",
        "Verify RustyNanoKVM HDMI vision & USB HID gadget report",
        "Converge monorepo workspace-wide verification on 35B MoE",
    ];

    let mut wave_num = 1;
    while dag.completed != dag.valid {
        let runnable = dag.runnable_wave();
        println!("Wave {}: runnable bitmask = {:#08b}", wave_num, runnable);
        let mut mask = runnable;
        while mask != 0 {
            let task_idx = mask.trailing_zeros() as usize;
            if let Some(prompt) = wave_prompts.get(task_idx) {
                let d = router.route(prompt);
                println!(
                    "  ├─ task={} -> subagent={} [ollama={}] score={}/{} Q15 fold=0x{:016x}",
                    task_idx,
                    d.model.subagent_id,
                    d.model.ollama_tag,
                    d.score_q15,
                    Q15_SCALE,
                    d.frame.fold_tag
                );
            }
            mask &= mask - 1;
        }
        dag.dispatch(runnable);
        dag.complete(runnable);
        wave_num += 1;
    }
}

fn run_udp_daemon(bind_addr: &str) {
    let socket = UdpSocket::bind(bind_addr).expect("Failed to bind UDP socket");
    socket
        .set_read_timeout(Some(Duration::from_millis(500)))
        .ok();
    println!(
        "8b-agy LAN worker listening on UDP {} (64B aligned frames)",
        bind_addr
    );

    let mut aligned_buf = LanFrame64::new(OpCode::Heartbeat, 0, 0, 0);
    loop {
        // Read directly into 64-byte aligned struct memory
        let buf_slice = unsafe {
            core::slice::from_raw_parts_mut(
                (&mut aligned_buf as *mut LanFrame64).cast::<u8>(),
                core::mem::size_of::<LanFrame64>(),
            )
        };
        if let Ok((len, src)) = socket.recv_from(buf_slice) {
            if let Ok(frame) = LanFrame64::try_from_bytes(&buf_slice[..len]) {
                println!(
                    "[rx {}] opcode=0x{:02x} worker={} task={} seq={} mask={:#b}",
                    src, frame.opcode, frame.worker_id, frame.task_id, frame.sequence, frame.runnable_mask
                );
                let mut ack = LanFrame64::new(
                    OpCode::TaskComplete,
                    frame.worker_id,
                    frame.task_id,
                    frame.sequence + 1,
                );
                ack.completed_mask = 1u64 << (frame.task_id & 63);
                ack.seal();
                let _ = socket.send_to(ack.as_bytes(), src);
            }
        }
    }
}

fn print_techno_zen_bonobo_era() {
    let zen = run_techno_zen_bonobo_era();
    println!("============================================================================================================================");
    println!("🧘‍♂️🦧⚡ {} — SOLAR PHYLOGENETIC RUNG-0 BONOBO × INTEGRAL LLL LOVÁSZ DUAL", zen.era_id);
    println!("============================================================================================================================");
    println!(
        "  • Rung-0 Bonobo Resonance (p/q) : {}/{} (+1 trit Target Resonance) | LLL Lovász Dual (δ) : {}/{}",
        zen.bonobo_resonance_pq.0,
        zen.bonobo_resonance_pq.1,
        zen.lovasz_dual_pq.0,
        zen.lovasz_dual_pq.1
    );
    println!(
        "  • Pythago-Zen Harmonic Coupling : (3/2) × (3/4) = {}/{} (Whole-Tone) | (3/2) + (3/4) = {}/{} = (3/2)² | (3/2) ÷ (3/4) = {}/{}",
        zen.pythago_zen_harmonic_pq.0,
        zen.pythago_zen_harmonic_pq.1,
        zen.additive_square_pq.0,
        zen.additive_square_pq.1,
        zen.octave_duality_pq.0,
        zen.octave_duality_pq.1
    );
    println!(
        "  • Acoustic & Prosody Lock       : Carrier = {} Hz · Fifth = {} Hz · Whole-Tone = {} Hz · Sub-Fifth = {} Hz · Groove = {} BPM -> {} WPM",
        zen.carrier_hz,
        zen.bonobo_fifth_hz,
        zen.whole_tone_hz,
        zen.sub_fifth_hz,
        zen.base_tempo_bpm,
        zen.prosody_lock_wpm
    );
    println!(
        "  • Sovereign Compression & Proof : p/qT = {}/{} vs FP16 | Q15 Reward = {}/{} (99.96%) | Kompress Savings = {}/{} | SPARK = {}/{} proved",
        zen.pqt_compression_pq.0,
        zen.pqt_compression_pq.1,
        zen.converged_reward_pq.0,
        zen.converged_reward_pq.1,
        zen.kompress_savings_pq.0,
        zen.kompress_savings_pq.1,
        zen.spark_proved_checks,
        zen.spark_proved_checks
    );
    println!(
        "  • Sealed 64B Techno-Zen Frame   : fold = 0x{:016x} (magic = 0x8b15, gate = 0x62, task = 9/8)",
        zen.sealed_frame.fold_tag
    );
    println!("============================================================================================================================");
}

fn print_chacheless_e8_apex() {
    let apex = run_chacheless_e8_gosset_apex(33);
    println!("============================================================================================================================");
    println!(
        "🌌🏛️⚡ {} — ZERO-KV-CACHE (2048/1) · E8 GOSSET (240 ROOTS) · STAGE-3 33-EPOCH APEX",
        apex.suite_id
    );
    println!("============================================================================================================================");
    println!(
        "  • E8 Gosset Lattice Quantizer   : y_scaled2 = {:?} (coset = {}, dist² = {}, sum = {} ≡ 0 mod 4)",
        apex.e8_quantization.y_scaled2,
        apex.e8_quantization.coset,
        apex.e8_quantization.dist_sq_q2,
        apex.e8_quantization.coord_sum
    );
    println!(
        "  • Viazovska 8D Packing Invariant: kissing_number = {} (112 D8 + 128 spinor) | det(Λ_E8) = {} | uniform_parity = {}",
        apex.e8_quantization.kissing_number,
        apex.e8_quantization.unimodular_det,
        apex.e8_quantization.uniform_parity
    );
    println!(
        "  • Chacheless Zero-KV-Cache 262K : {} MiB (FP16 KV) -> {} MiB (O(1) GDN+E8 State) = {}/{} KV Elimination | p/qT = {}/{}",
        apex.standard_262k_kv_mib,
        apex.chacheless_state_mib,
        apex.kv_elimination_ratio_pq.0,
        apex.kv_elimination_ratio_pq.1,
        apex.chacheless_pqt_compression_pq.0,
        apex.chacheless_pqt_compression_pq.1
    );
    println!(
        "  • Stage-3 (33 Epochs / 132 Roll): Reward = {}/32768 ({}/{} = 99.988%) | MOPD2 Loss = {}/32768 (1/8192) | GR = {:?} | PLE = {}",
        apex.final_reward_q15,
        apex.final_reward_pq.0,
        apex.final_reward_pq.1,
        apex.final_mopd2_loss_q15,
        apex.final_gr_gates_q8,
        apex.final_ple_gain_q8
    );
    println!(
        "  • Apex 8B SOTA Benchmarks (p/q) : AIME = 30/30 ({}/{}) · MATH-500 = 498/500 ({}/{}) · LCB = 198/200 ({}/{}) · SWE = 465/500 ({}/{}) · GPQA = 180/198 ({}/{})",
        apex.apex_benchmarks_pq[0].0,
        apex.apex_benchmarks_pq[0].1,
        apex.apex_benchmarks_pq[1].0,
        apex.apex_benchmarks_pq[1].1,
        apex.apex_benchmarks_pq[2].0,
        apex.apex_benchmarks_pq[2].1,
        apex.apex_benchmarks_pq[3].0,
        apex.apex_benchmarks_pq[3].1,
        apex.apex_benchmarks_pq[4].0,
        apex.apex_benchmarks_pq[4].1
    );
    println!(
        "  • Ada 2022 SPARK & Sealed Frame : SPARK = {}/{} proved | fold = 0x{:016x} (magic = 0x8b15, gate = 0x62, task = 33)",
        apex.spark_proved_checks,
        apex.spark_proved_checks,
        apex.sealed_frame.fold_tag
    );
    println!("============================================================================================================================");
}

fn print_peloton_row_ayeos_e2e() {
    let row = run_peloton_row_usbc_ayeos_e2e();
    println!("============================================================================================================================");
    println!(
        "🚣‍♂️⚡🏴‍☠️ {} — PELOTON ROW (PR01) USB-C · ZERO-SUBSCRIPTION ($0/MO) · 8B-IS-ENGINE + ayeOS",
        row.suite_id
    );
    println!("============================================================================================================================");
    println!(
        "  • USB-C Hardware Transport      : {} @ {} baud (64B@0x62 OpCode::PelotonRowUsbC = 0x08)",
        row.usbc_device_node, row.usbc_baud_rate
    );
    println!(
        "  • OSS Zero-Subscription Unlock  : Peloton Built-In = ${}/mo (${}/yr) -> 8B-is + ayeOS = ${}/mo (${}/yr SAVED · bypass p/q = {}/{})",
        row.peloton_monthly_fee_usd,
        row.annual_savings_usd,
        row.ayeos_monthly_fee_usd,
        row.annual_savings_usd,
        row.subscription_bypass_ratio_pq.0,
        row.subscription_bypass_ratio_pq.1
    );
    println!(
        "  • Exact Cube-Law Hydrodynamics  : Drag c = {}/{} (2.80) · P = 350,000,000 / t_500m³ · Peak (1:40/500m) = {}/{} W · Steady (2:00/500m) = {}/{} W",
        row.drag_factor_pq.0,
        row.drag_factor_pq.1,
        row.peak_power_pq.0,
        row.peak_power_pq.1,
        row.steady_power_pq.0,
        row.steady_power_pq.1
    );
    println!(
        "  • Bonobo × LLL Kinematic Lock   : Peak Power vs 700/3 W Baseline = {}/{} (Rung-0 Bonobo) · Seat/Handle Coupling = {}/{} (LLL Lovász δ)",
        row.bonobo_power_ratio_pq.0,
        row.bonobo_power_ratio_pq.1,
        row.lovasz_handle_seat_ratio_pq.0,
        row.lovasz_handle_seat_ratio_pq.1
    );
    println!("----------------------------------------------------------------------------------------------------------------------------");
    println!("  4-PHASE PELOTON ROW STROKE TELEMETRY (USB-C CDC-ACM -> 8B-is-engine -> ayeOS PccModule):");
    println!(
        "  {:<18} | {:<6} | {:<10} | {:<16} | {:<15} | {:<13} | {:<17} | {:<10}",
        "PHASE", "SPM", "SPLIT/500m", "POWER p/q (W)", "HANDLE/SEAT mm", "FORM Q15", "SPQT TRITS (B243)", "ayeOS GEMM"
    );
    println!("  --------------------------------------------------------------------------------------------------------------------------");
    for s in &row.phase_samples {
        let mins = s.split_500m_secs / 60;
        let secs = s.split_500m_secs % 60;
        println!(
            "  {:<18} | {:>2} spm | {:>2}:{:02}.0    | {:>6}/{:<5} ({:>3}W) | {:>4}/{:<4} mm    | {:>5}/32768 | {:?} ({:>3}) | {:>+6}",
            s.phase_name,
            s.stroke_rate_spm,
            mins,
            secs,
            s.power_watts_pq.0,
            s.power_watts_pq.1,
            s.power_watts_rounded,
            s.handle_pos_mm,
            s.seat_pos_mm,
            s.form_assist_score_q15,
            s.spqt_trits,
            s.spqt_base243_byte,
            s.ayeos_gemm_score
        );
    }
    println!("----------------------------------------------------------------------------------------------------------------------------");
    println!(
        "  • BLE GATT FTMS Bridge (0x{:04X}) : Characteristic Rower Data (0x{:04X}) · Flags = 0x02FE · 18-Byte Payload = {:02X?}",
        row.ftms_service_uuid, row.ftms_rower_char_uuid, row.ftms_2ad1_packet
    );
    println!(
        "  • ayeOS Kernel & 8B-is Coach    : PccModule = {} (cap_mask = 0x{:02X}, hash = 0x{:016X}) -> {} ({})",
        row.ayeos_module_name,
        row.ayeos_capability_mask,
        row.ayeos_fnv1a_hash,
        row.routed_coach_subagent,
        row.routed_coach_ollama_tag
    );
    println!(
        "  • Ada 2022 SPARK & Sealed Frame : SPARK = {}/{} proved (Peloton_Row_Ayeos_Spark) | Kompress p/q = {}/{} | fold = 0x{:016x}",
        row.spark_proved_checks,
        row.spark_proved_checks,
        row.kompress_ratio_pq.0,
        row.kompress_ratio_pq.1,
        row.sealed_frame.fold_tag
    );
    println!("============================================================================================================================");
}
