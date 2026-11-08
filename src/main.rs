use eightb_agy_fanout::{
    BitmaskDag, LanFrame64, OllamaWaveRouter, OpCode, DEEPSEEK_V4_PRO_HERETIC_MLX_PROFILES,
    DEFAULT_UDP_BIND, FANOUT_MODEL_CONSTELLATION, Q15_SCALE,
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
        "route" => {
            let prompt = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                "Derive symplectic shear theorem with uncensored heretic reasoning and Q15 p/q math"
                    .to_string()
            };
            run_ollama_router(&prompt);
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
