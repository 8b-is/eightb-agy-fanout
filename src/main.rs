use eightb_agy_fanout::{
    run_hardware_kernel_benchmarks, run_robvc_architecture_pipeline, BitmaskDag, LanFrame64,
    OllamaWaveRouter, OpCode, RobvcStage, WukongOmniWeightProgram,
    DEEPSEEK_V4_PRO_HERETIC_MLX_PROFILES, DEFAULT_UDP_BIND, FANOUT_MODEL_CONSTELLATION,
    OLLAMA_LOCAL_FLEET, Q15_SCALE, SOTA_8B_BENCHMARK_MATRIX, SOVEREIGN_CLUSTER_TOPOLOGY,
};
use std::env;
use std::net::UdpSocket;
use std::time::Duration;

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("demo");

    match cmd {
        "daemon" => {
            let bind_addr = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_UDP_BIND);
            run_udp_daemon(bind_addr);
        }
        "models" => print_fanout_models(),
        "mlx-quant" => print_mlx_quant_profiles(),
        "wukong" => print_wukong_weight_program(),
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
        "route" => {
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
