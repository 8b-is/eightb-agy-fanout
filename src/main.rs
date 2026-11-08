use eightb_agy_fanout::{BitmaskDag, LanFrame64, OpCode};
use std::env;
use std::net::UdpSocket;
use std::time::Duration;

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("demo");

    match cmd {
        "daemon" => {
            let bind_addr = args.get(2).map(|s| s.as_str()).unwrap_or("0.0.0.0:18815");
            run_udp_daemon(bind_addr);
        }
        _ => run_wave_demo(),
    }
}

fn run_wave_demo() {
    let mut dag = BitmaskDag::new();
    // 0: smart-tree index -> 1: analyzer, 2: visualizer, 3: nanokvm-test -> 4: writer
    dag.add_task(0, 0);
    dag.add_task(1, 1 << 0);
    dag.add_task(2, 1 << 0);
    dag.add_task(3, 1 << 0);
    dag.add_task(4, (1 << 1) | (1 << 2) | (1 << 3));

    let mut wave_num = 1;
    while dag.completed != dag.valid {
        let runnable = dag.runnable_wave();
        println!("Wave {}: runnable bitmask = {:#08b}", wave_num, runnable);
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
    println!("8b-agy LAN worker listening on UDP {} (64B aligned frames)", bind_addr);

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
                    "[rx {}] worker={} task={} seq={} mask={:#b}",
                    src, frame.worker_id, frame.task_id, frame.sequence, frame.runnable_mask
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
