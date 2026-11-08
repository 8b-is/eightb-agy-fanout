#![allow(clippy::identity_op)]

use core::mem::{align_of, size_of};
use core::slice;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

pub const GATE_BYTE: u8 = 0x62;
pub const FRAME_MAGIC: u16 = 0x8B15;
pub const MAX_TASKS: usize = 64;
pub const Q15_SCALE: u32 = 32_768;
pub const DEFAULT_OLLAMA_ADDR: &str = "127.0.0.1:11434";
pub const DEFAULT_UDP_BIND: &str = "0.0.0.0:18815";

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OpCode {
    Heartbeat = 0x01,
    TaskDispatch = 0x02,
    TaskComplete = 0x03,
    KvmHidInject = 0x04,
    OllamaRoute  = 0x05,
    MlxQuantSeal = 0x06,
    SubagentWave = 0x07,
}

impl OpCode {
    #[inline(always)]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Heartbeat),
            0x02 => Some(Self::TaskDispatch),
            0x03 => Some(Self::TaskComplete),
            0x04 => Some(Self::KvmHidInject),
            0x05 => Some(Self::OllamaRoute),
            0x06 => Some(Self::MlxQuantSeal),
            0x07 => Some(Self::SubagentWave),
            _ => None,
        }
    }
}

/// 64-byte cache-line aligned wire frame (zero implicit padding).
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LanFrame64 {
    pub magic: u16,          // 0..2:   0x8B15
    pub gate: u8,            // 2..3:   0x62 seal gate
    pub opcode: u8,          // 3..4:   OpCode discriminant
    pub worker_id: u16,      // 4..6:   0..63 node index
    pub task_id: u16,        // 6..8:   0..63 task index
    pub sequence: u64,       // 8..16:  Monotonic tap counter
    pub runnable_mask: u64,  // 16..24: Active/runnable wave bitmask
    pub completed_mask: u64, // 24..32: Completed wave bitmask
    pub payload: [u8; 24],   // 32..56: Inline command / HID report / model route tag
    pub fold_tag: u64,       // 56..64: 64-bit XOR-fold integrity tag over bytes 0..56
}

const _: () = assert!(size_of::<LanFrame64>() == 64);
const _: () = assert!(align_of::<LanFrame64>() == 64);

#[derive(Debug, PartialEq, Eq)]
pub enum FrameError {
    BufferTooShort,
    Unaligned,
    InvalidMagic,
    InvalidGate,
    InvalidOpCode,
    IntegrityMismatch,
}

impl LanFrame64 {
    #[inline]
    pub fn new(opcode: OpCode, worker_id: u16, task_id: u16, sequence: u64) -> Self {
        let mut frame = Self {
            magic: FRAME_MAGIC,
            gate: GATE_BYTE,
            opcode: opcode as u8,
            worker_id,
            task_id,
            sequence,
            runnable_mask: 0,
            completed_mask: 0,
            payload: [0u8; 24],
            fold_tag: 0,
        };
        frame.seal();
        frame
    }

    /// Computes 64-bit SWAR XOR-fold across the first 56 bytes (7 x u64 words).
    #[inline(always)]
    pub fn compute_fold(&self) -> u64 {
        let ptr = (self as *const Self).cast::<u64>();
        // SAFETY: `LanFrame64` is 64-byte aligned and 64 bytes long (8 x u64 words).
        unsafe {
            let w0 = *ptr.add(0);
            let w1 = *ptr.add(1);
            let w2 = *ptr.add(2);
            let w3 = *ptr.add(3);
            let w4 = *ptr.add(4);
            let w5 = *ptr.add(5);
            let w6 = *ptr.add(6);
            (w0 ^ w1) ^ (w2 ^ w3) ^ (w4 ^ w5) ^ w6.rotate_left(13)
        }
    }

    #[inline(always)]
    pub fn seal(&mut self) {
        self.fold_tag = self.compute_fold();
    }

    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: All 64 bytes are explicitly initialized primitives with zero padding.
        unsafe { slice::from_raw_parts((self as *const Self).cast::<u8>(), size_of::<Self>()) }
    }

    #[inline(always)]
    pub fn try_from_bytes(bytes: &[u8]) -> Result<&Self, FrameError> {
        if bytes.len() < size_of::<Self>() {
            return Err(FrameError::BufferTooShort);
        }
        if !(bytes.as_ptr() as usize).is_multiple_of(align_of::<Self>()) {
            return Err(FrameError::Unaligned);
        }
        // SAFETY: Verified length >= 64 and 64-byte alignment; all fields are primitive integers.
        let frame = unsafe { &*(bytes.as_ptr().cast::<Self>()) };
        if frame.magic != FRAME_MAGIC {
            return Err(FrameError::InvalidMagic);
        }
        if frame.gate != GATE_BYTE {
            return Err(FrameError::InvalidGate);
        }
        if OpCode::from_u8(frame.opcode).is_none() {
            return Err(FrameError::InvalidOpCode);
        }
        if frame.fold_tag != frame.compute_fold() {
            return Err(FrameError::IntegrityMismatch);
        }
        Ok(frame)
    }
}

/// 4-way unrolled SWAR popcount over 64-bit words.
// `as_chunks` remains behind a toolchain gate on some targets; `chunks_exact`
// stays stable everywhere while the 4-way unroll is what matters for ILP.
#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
#[inline]
pub fn swar_popcount_slice(words: &[u64]) -> u64 {
    let mut total = 0u64;
    let chunks = words.chunks_exact(4);
    let rem = chunks.remainder();
    for c in chunks {
        total += c[0].count_ones() as u64
            + c[1].count_ones() as u64
            + c[2].count_ones() as u64
            + c[3].count_ones() as u64;
    }
    for &w in rem {
        total += w.count_ones() as u64;
    }
    total
}

/// O(1) Single-Register DAG Scheduler for Antigravity Wave Fan-Out.
#[derive(Clone, Debug)]
pub struct BitmaskDag {
    pub deps: [u64; MAX_TASKS],
    pub valid: u64,
    pub in_flight: u64,
    pub completed: u64,
}

impl Default for BitmaskDag {
    fn default() -> Self {
        Self::new()
    }
}

impl BitmaskDag {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            deps: [0; MAX_TASKS],
            valid: 0,
            in_flight: 0,
            completed: 0,
        }
    }

    #[inline(always)]
    pub fn add_task(&mut self, id: usize, dep_mask: u64) {
        if id < MAX_TASKS {
            self.valid |= 1u64 << id;
            self.deps[id] = dep_mask;
        }
    }

    /// Returns bitmask of all subagent tasks whose dependencies are satisfied right now.
    #[inline]
    pub fn runnable_wave(&self) -> u64 {
        let mut pending = self.valid & !(self.completed | self.in_flight);
        let mut ready = 0u64;
        while pending != 0 {
            let idx = pending.trailing_zeros() as usize;
            let req = self.deps[idx];
            if (req & self.completed) == req {
                ready |= 1u64 << idx;
            }
            pending &= pending - 1; // Clear lowest set bit (BLSR)
        }
        ready
    }

    #[inline(always)]
    pub fn dispatch(&mut self, mask: u64) {
        self.in_flight |= mask;
    }

    #[inline(always)]
    pub fn complete(&mut self, mask: u64) {
        self.in_flight &= !mask;
        self.completed |= mask;
    }
}

/// RustyNanoKVM Linux USB Gadget (`/dev/hidg0`) 8-byte Keyboard Report.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct HidKeyboardReport {
    pub modifiers: u8,
    pub reserved: u8,
    pub keys: [u8; 6],
}

impl HidKeyboardReport {
    #[inline(always)]
    pub const fn key_down(modifiers: u8, keycode: u8) -> [u8; 8] {
        [modifiers, 0, keycode, 0, 0, 0, 0, 0]
    }

    #[inline(always)]
    pub const fn release() -> [u8; 8] {
        [0; 8]
    }
}

/// Apple Silicon Metal `mlx-quant` quantization scheme.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MlxQuantScheme {
    Affine = 0,
    Mxfp4  = 1,
    Mxfp8  = 2,
    Nvfp4  = 3,
}

impl MlxQuantScheme {
    #[inline(always)]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Affine => "affine",
            Self::Mxfp4  => "mxfp4",
            Self::Mxfp8  => "mxfp8",
            Self::Nvfp4  => "nvfp4",
        }
    }
}

/// Exact rational `p/q` Apple Silicon Metal `mlx-quant` descriptor with `64B@0x62` seal.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MlxQuantProfile {
    pub hf_repo: &'static str,
    pub scheme: MlxQuantScheme,
    pub bits: u8,
    pub group_size: u16,
    pub compression_p: u16,
    pub compression_q: u16,
    pub litany_wire_seal16: u16,
    pub spqt_root_base243: [u8; 3],
    pub output_dir: &'static str,
}

impl MlxQuantProfile {
    /// Builds a `64B@0x62` `LanFrame64` (`OpCode::MlxQuantSeal`) encoding the MLX quantization witness.
    #[inline]
    pub fn to_sealed_frame(&self, worker_id: u16, sequence: u64) -> LanFrame64 {
        let mut frame = LanFrame64::new(OpCode::MlxQuantSeal, worker_id, 0, sequence);
        frame.runnable_mask = (self.compression_p as u64) << 16 | (self.compression_q as u64);
        frame.completed_mask = self.litany_wire_seal16 as u64;
        frame.payload[0] = self.scheme as u8;
        frame.payload[1] = self.bits;
        frame.payload[2..4].copy_from_slice(&self.group_size.to_le_bytes());
        frame.payload[4..6].copy_from_slice(&self.litany_wire_seal16.to_le_bytes());
        frame.payload[6..9].copy_from_slice(&self.spqt_root_base243);
        let repo_bytes = self.hf_repo.as_bytes();
        let copy_len = if repo_bytes.len() > 15 { 15 } else { repo_bytes.len() };
        frame.payload[9..9 + copy_len].copy_from_slice(&repo_bytes[..copy_len]);
        frame.seal();
        frame
    }
}

/// Apple Silicon `mlx-quant` profiles for `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic`.
pub const DEEPSEEK_V4_PRO_HERETIC_MLX_PROFILES: [MlxQuantProfile; 2] = [
    MlxQuantProfile {
        hf_repo: "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic",
        scheme: MlxQuantScheme::Nvfp4,
        bits: 4,
        group_size: 64,
        compression_p: 4,
        compression_q: 1,
        litany_wire_seal16: 0x019F,
        spqt_root_base243: [149, 233, 212],
        output_dir: ".mlx/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-nvfp4-g64",
    },
    MlxQuantProfile {
        hf_repo: "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic",
        scheme: MlxQuantScheme::Mxfp8,
        bits: 8,
        group_size: 64,
        compression_p: 2,
        compression_q: 1,
        litany_wire_seal16: 0x0189,
        spqt_root_base243: [131, 233, 212],
        output_dir: ".mlx/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-mxfp8-g64",
    },
];

/// Functional role of a model node in the `eightb-agy-fanout` Wave-2 constellation.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FanoutRole {
    HereticReasoningSubagent = 0,
    PrimarySotaCodegen       = 1,
    OllamaWaveRouter         = 2,
    FastDecisionClassifier   = 3,
    HeavyMoECodingVerifier   = 4,
    MultimodalKvmVision      = 5,
}

impl FanoutRole {
    #[inline(always)]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HereticReasoningSubagent => "HereticReasoningSubagent",
            Self::PrimarySotaCodegen       => "PrimarySotaCodegen",
            Self::OllamaWaveRouter         => "OllamaWaveRouter",
            Self::FastDecisionClassifier   => "FastDecisionClassifier",
            Self::HeavyMoECodingVerifier   => "HeavyMoECodingVerifier",
            Self::MultimodalKvmVision      => "MultimodalKvmVision",
        }
    }
}

/// Complete specification of a fan-out model in the `8b-agy` constellation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FanoutModelSpec {
    pub worker_id: u16,
    pub subagent_id: &'static str,
    pub hf_repo: &'static str,
    pub ollama_tag: &'static str,
    pub architecture: &'static str,
    pub num_layers: u16,
    pub gdn_linear_layers: u16,
    pub full_attn_layers: u16,
    pub mtp_layers: u8,
    pub hidden_size: u16,
    pub ctx_tokens: u32,
    pub mlx_scheme: MlxQuantScheme,
    pub compression_p: u16,
    pub compression_q: u16,
    pub role: FanoutRole,
}

/// Canonical 6-Model `eightb-agy-fanout` Constellation wired into Mac Ollama & Apple Silicon MLX-Quant.
pub const FANOUT_MODEL_CONSTELLATION: [FanoutModelSpec; 6] = [
    FanoutModelSpec {
        worker_id: 0,
        subagent_id: "deepseek-v4-pro-qwen35-heretic",
        hf_repo: "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic",
        ollama_tag: "deepseek-v4-pro-qwen3.5-9b-heretic:latest",
        architecture: "Qwen3_5ForConditionalGeneration",
        num_layers: 32,
        gdn_linear_layers: 24,
        full_attn_layers: 8,
        mtp_layers: 1,
        hidden_size: 4096,
        ctx_tokens: 262_144,
        mlx_scheme: MlxQuantScheme::Nvfp4,
        compression_p: 4,
        compression_q: 1,
        role: FanoutRole::HereticReasoningSubagent,
    },
    FanoutModelSpec {
        worker_id: 1,
        subagent_id: "qwen4-exp-8b-kompress-ultra",
        hf_repo: "PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra",
        ollama_tag: "qwen3.6-fast:latest",
        architecture: "Qwen4ExpForCausalLM",
        num_layers: 32,
        gdn_linear_layers: 24,
        full_attn_layers: 8,
        mtp_layers: 1,
        hidden_size: 4096,
        ctx_tokens: 262_144,
        mlx_scheme: MlxQuantScheme::Nvfp4,
        compression_p: 4,
        compression_q: 1,
        role: FanoutRole::PrimarySotaCodegen,
    },
    FanoutModelSpec {
        worker_id: 2,
        subagent_id: "8b-agy-router",
        hf_repo: "8b-is/eightb-agy-fanout",
        ollama_tag: "8b-agy-router:latest",
        architecture: "qwen3_5",
        num_layers: 32,
        gdn_linear_layers: 24,
        full_attn_layers: 8,
        mtp_layers: 0,
        hidden_size: 2560,
        ctx_tokens: 262_144,
        mlx_scheme: MlxQuantScheme::Mxfp8,
        compression_p: 2,
        compression_q: 1,
        role: FanoutRole::OllamaWaveRouter,
    },
    FanoutModelSpec {
        worker_id: 3,
        subagent_id: "kev-decision-engine",
        hf_repo: "8b-is/kev-7.9b-nvfp4",
        ollama_tag: "kev:latest",
        architecture: "kev",
        num_layers: 32,
        gdn_linear_layers: 24,
        full_attn_layers: 8,
        mtp_layers: 0,
        hidden_size: 4096,
        ctx_tokens: 73_728,
        mlx_scheme: MlxQuantScheme::Nvfp4,
        compression_p: 4,
        compression_q: 1,
        role: FanoutRole::FastDecisionClassifier,
    },
    FanoutModelSpec {
        worker_id: 4,
        subagent_id: "qwen3.6-35b-a3b-coding",
        hf_repo: "Qwen/Qwen3.6-35B-A3B-Coding",
        ollama_tag: "qwen3.6:35b-a3b-coding-nvfp4",
        architecture: "qwen3_6_moe",
        num_layers: 48,
        gdn_linear_layers: 36,
        full_attn_layers: 12,
        mtp_layers: 1,
        hidden_size: 4096,
        ctx_tokens: 131_072,
        mlx_scheme: MlxQuantScheme::Nvfp4,
        compression_p: 4,
        compression_q: 1,
        role: FanoutRole::HeavyMoECodingVerifier,
    },
    FanoutModelSpec {
        worker_id: 5,
        subagent_id: "bonsai-vision-kvm",
        hf_repo: "8b-is/bonsai-vision-27b",
        ollama_tag: "bonsai-vision:latest",
        architecture: "qwen35_vl",
        num_layers: 36,
        gdn_linear_layers: 27,
        full_attn_layers: 9,
        mtp_layers: 0,
        hidden_size: 5120,
        ctx_tokens: 262_144,
        mlx_scheme: MlxQuantScheme::Affine,
        compression_p: 8,
        compression_q: 1,
        role: FanoutRole::MultimodalKvmVision,
    },
];

/// Task classification domain used by `OllamaWaveRouter`.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TaskDomain {
    HereticReasoning    = 0,
    SparkNoStdCodegen   = 1,
    WaveRouting         = 2,
    FastDecision        = 3,
    HeavyMoERefactor    = 4,
    MultimodalKvmVision = 5,
}

impl TaskDomain {
    #[inline(always)]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HereticReasoning    => "HereticReasoning",
            Self::SparkNoStdCodegen   => "SparkNoStdCodegen",
            Self::WaveRouting         => "WaveRouting",
            Self::FastDecision        => "FastDecision",
            Self::HeavyMoERefactor    => "HeavyMoERefactor",
            Self::MultimodalKvmVision => "MultimodalKvmVision",
        }
    }
}

/// Routing decision emitted by `OllamaWaveRouter` using exact Q15 rational `p/q` arithmetic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RouteDecision {
    pub domain: TaskDomain,
    pub model: &'static FanoutModelSpec,
    pub score_q15: u32,
    pub runnable_mask: u64,
    pub frame: LanFrame64,
}

/// Deterministic + Live HTTP/1.1 Ollama Wave Router (`8b-agy-router:latest` @ `127.0.0.1:11434`).
#[derive(Clone, Debug)]
pub struct OllamaWaveRouter {
    pub endpoint: &'static str,
    pub router_model: &'static str,
    pub sequence: u64,
}

impl Default for OllamaWaveRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaWaveRouter {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            endpoint: DEFAULT_OLLAMA_ADDR,
            router_model: "8b-agy-router:latest",
            sequence: 1,
        }
    }

    /// Classifies a prompt into one of the 6 `TaskDomain` channels.
    pub fn classify_prompt(prompt: &str) -> TaskDomain {
        let lower = prompt.to_ascii_lowercase();
        if lower.contains("kvm")
            || lower.contains("hdmi")
            || lower.contains("vision")
            || lower.contains("image")
            || lower.contains("hidg")
        {
            TaskDomain::MultimodalKvmVision
        } else if lower.contains("spark")
            || lower.contains("gnatprove")
            || lower.contains("qwen4")
            || lower.contains("no_std")
            || lower.contains("ada")
        {
            TaskDomain::SparkNoStdCodegen
        } else if lower.contains("moe")
            || lower.contains("35b")
            || lower.contains("monorepo")
            || lower.contains("workspace-wide")
        {
            TaskDomain::HeavyMoERefactor
        } else if lower.contains("route")
            || lower.contains("dag")
            || lower.contains("fanout")
            || lower.contains("wave")
        {
            TaskDomain::WaveRouting
        } else if lower.contains("quick") || lower.contains("classify") || lower.contains("triage") {
            TaskDomain::FastDecision
        } else {
            // Default deep reasoning, math, STEM, uncensored/abliterated analysis -> DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic
            TaskDomain::HereticReasoning
        }
    }

    /// Computes exact rational `p/q` affinity score in `Q15` (`0..=32_768`) without floating-point drift.
    #[inline]
    pub const fn score_model_q15(domain: TaskDomain, spec: &FanoutModelSpec) -> u32 {
        let base_q15: u32 = if (domain as u8) == (spec.role as u8) {
            28_672 // 7/8 in Q15
        } else {
            12_288 // 3/8 in Q15
        };

        // Context bonus: 262,144 tokens adds 2,048/32,768 (1/16)
        let ctx_bonus_q15: u32 = if spec.ctx_tokens >= 262_144 {
            2_048
        } else {
            1_024
        };

        // GDN 3:1 hybrid + MTP bonus: adds 1,536/32,768 + 430/32,768
        let gdn_bonus_q15: u32 = if spec.gdn_linear_layers * 1 == spec.full_attn_layers * 3 {
            1_536
        } else {
            512
        };
        let mtp_bonus_q15: u32 = if spec.mtp_layers > 0 { 430 } else { 128 };

        let total = base_q15 + ctx_bonus_q15 + gdn_bonus_q15 + mtp_bonus_q15;
        if total > Q15_SCALE {
            Q15_SCALE
        } else {
            total
        }
    }

    /// Deterministically selects the highest-scoring model in `FANOUT_MODEL_CONSTELLATION` and seals a `64B@0x62` `LanFrame64`.
    pub fn route(&mut self, prompt: &str) -> RouteDecision {
        let domain = Self::classify_prompt(prompt);
        let mut best_idx = 0usize;
        let mut best_score = 0u32;

        for (i, spec) in FANOUT_MODEL_CONSTELLATION.iter().enumerate() {
            let s = Self::score_model_q15(domain, spec);
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }

        let chosen = &FANOUT_MODEL_CONSTELLATION[best_idx];
        let runnable_mask = (1u64 << chosen.worker_id) | (1u64 << 2); // Target worker + router witness bit
        let seq = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);

        let mut frame = LanFrame64::new(
            OpCode::OllamaRoute,
            chosen.worker_id,
            domain as u16,
            seq,
        );
        frame.runnable_mask = runnable_mask;
        frame.completed_mask = best_score as u64;

        let tag_bytes = chosen.ollama_tag.as_bytes();
        let copy_len = if tag_bytes.len() > 24 { 24 } else { tag_bytes.len() };
        frame.payload[..copy_len].copy_from_slice(&tag_bytes[..copy_len]);
        frame.seal();

        RouteDecision {
            domain,
            model: chosen,
            score_q15: best_score,
            runnable_mask,
            frame,
        }
    }

    /// Probes the local Mac Ollama daemon (`127.0.0.1:11434/api/tags`) over a raw TCP socket with a 500ms timeout.
    pub fn probe_ollama_live(&self) -> Result<(bool, bool), &'static str> {
        let addr: SocketAddr = self.endpoint.parse().map_err(|_| "invalid_socket_addr")?;
        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(500))
            .map_err(|_| "ollama_unreachable")?;
        let _ = stream.set_read_timeout(Some(Duration::from_millis(800)));
        let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));

        let req = format!(
            "GET /api/tags HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
            self.endpoint
        );
        stream.write_all(req.as_bytes()).map_err(|_| "write_failed")?;

        let mut resp = String::new();
        stream.read_to_string(&mut resp).map_err(|_| "read_failed")?;
        let has_heretic = resp.contains("deepseek-v4-pro-qwen3.5-9b-heretic");
        let has_router = resp.contains("8b-agy-router");
        Ok((has_router, has_heretic))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame64_seal_and_zero_copy_verify() {
        let mut frame = LanFrame64::new(OpCode::TaskDispatch, 3, 1, 42);
        frame.runnable_mask = 0b0110;
        frame.payload[..4].copy_from_slice(b"AGY!");
        frame.seal();

        let raw = frame.as_bytes();
        let decoded = LanFrame64::try_from_bytes(raw).expect("Valid 64B frame");
        assert_eq!(decoded.worker_id, 3);
        assert_eq!(decoded.runnable_mask, 0b0110);
        assert_eq!(&decoded.payload[..4], b"AGY!");
    }

    #[test]
    fn test_dag_parallel_fanout_waves() {
        let mut dag = BitmaskDag::new();
        dag.add_task(0, 0); // Wave 1: cleaner
        dag.add_task(1, 1 << 0); // Wave 2: analyzer
        dag.add_task(2, 1 << 0); // Wave 2: visualizer
        dag.add_task(3, (1 << 1) | (1 << 2)); // Wave 3: writer

        let w1 = dag.runnable_wave();
        assert_eq!(w1, 0b0001);
        dag.dispatch(w1);
        dag.complete(w1);

        // Wave 2 fans out to tasks 1 & 2 simultaneously
        let w2 = dag.runnable_wave();
        assert_eq!(w2, 0b0110);
        dag.dispatch(w2);
        dag.complete(w2);

        // Wave 3 unblocks writer
        let w3 = dag.runnable_wave();
        assert_eq!(w3, 0b1000);
    }

    #[test]
    fn test_swar_popcount_matches_reference() {
        let words = [0u64, 1, u64::MAX, 0x0F0F_0F0F_0F0F_0F0F, 0b1011];
        let expect: u64 = words.iter().map(|w| w.count_ones() as u64).sum();
        assert_eq!(swar_popcount_slice(&words), expect);
        assert_eq!(swar_popcount_slice(&[]), 0);
    }

    #[test]
    fn test_frame_rejects_tampered_fold_and_bad_fields() {
        #[repr(align(64))]
        struct Aligned([u8; 64]);

        let frame = LanFrame64::new(OpCode::Heartbeat, 0, 0, 7);
        let mut a = Aligned([0u8; 64]);
        a.0.copy_from_slice(frame.as_bytes());
        assert!(LanFrame64::try_from_bytes(&a.0).is_ok());

        // Flip an integrity-relevant payload bit -> integrity mismatch.
        a.0[40] ^= 0x01;
        assert_eq!(
            LanFrame64::try_from_bytes(&a.0),
            Err(FrameError::IntegrityMismatch)
        );

        // Restore, then break the gate.
        a.0[40] ^= 0x01;
        a.0[2] = 0x00;
        assert_eq!(
            LanFrame64::try_from_bytes(&a.0),
            Err(FrameError::InvalidGate)
        );

        // Restore gate, break magic.
        a.0[2] = GATE_BYTE;
        a.0[0] = 0x00;
        assert_eq!(
            LanFrame64::try_from_bytes(&a.0),
            Err(FrameError::InvalidMagic)
        );

        // Short buffer.
        a.0[0] = (FRAME_MAGIC & 0xFF) as u8;
        assert_eq!(
            LanFrame64::try_from_bytes(&a.0[..10]),
            Err(FrameError::BufferTooShort)
        );
    }

    #[test]
    fn test_hid_reports() {
        assert_eq!(
            HidKeyboardReport::key_down(0x02, 0x04),
            [0x02, 0, 0x04, 0, 0, 0, 0, 0]
        );
        assert_eq!(HidKeyboardReport::release(), [0u8; 8]);
    }

    #[test]
    fn test_add_task_out_of_range_is_ignored() {
        let mut dag = BitmaskDag::new();
        dag.add_task(MAX_TASKS, 0); // 64 -> out of range, ignored
        assert_eq!(dag.valid, 0);
        dag.add_task(63, 0);
        assert_eq!(dag.valid, 1u64 << 63);
        assert_eq!(dag.runnable_wave(), 1u64 << 63);
    }

    #[test]
    fn test_fanout_model_constellation_and_deepseek_heretic_spec() {
        let heretic = &FANOUT_MODEL_CONSTELLATION[0];
        assert_eq!(
            heretic.hf_repo,
            "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"
        );
        assert_eq!(
            heretic.ollama_tag,
            "deepseek-v4-pro-qwen3.5-9b-heretic:latest"
        );
        assert_eq!(heretic.num_layers, 32);
        assert_eq!(heretic.gdn_linear_layers, 24);
        assert_eq!(heretic.full_attn_layers, 8);
        assert_eq!(heretic.mtp_layers, 1);
        assert_eq!(heretic.ctx_tokens, 262_144);
        assert_eq!(heretic.compression_p, 4);
        assert_eq!(heretic.compression_q, 1);
    }

    #[test]
    fn test_mlx_quant_pq_compression_and_litany_seals() {
        let nvfp4 = &DEEPSEEK_V4_PRO_HERETIC_MLX_PROFILES[0];
        assert_eq!(nvfp4.scheme, MlxQuantScheme::Nvfp4);
        assert_eq!(nvfp4.litany_wire_seal16, 0x019F);
        let frame = nvfp4.to_sealed_frame(0, 7);
        let verified = LanFrame64::try_from_bytes(frame.as_bytes()).expect("valid MLX seal frame");
        assert_eq!(verified.opcode, OpCode::MlxQuantSeal as u8);
        assert_eq!(verified.completed_mask, 0x019F);

        let mxfp8 = &DEEPSEEK_V4_PRO_HERETIC_MLX_PROFILES[1];
        assert_eq!(mxfp8.scheme, MlxQuantScheme::Mxfp8);
        assert_eq!(mxfp8.litany_wire_seal16, 0x0189);
    }

    #[test]
    fn test_ollama_wave_router_exact_q15_routing_and_frame_seal() {
        let mut router = OllamaWaveRouter::new();
        let d1 = router.route("Derive symplectic shear theorem with uncensored heretic reasoning");
        assert_eq!(d1.domain, TaskDomain::HereticReasoning);
        assert_eq!(d1.model.worker_id, 0);
        assert_eq!(
            d1.model.ollama_tag,
            "deepseek-v4-pro-qwen3.5-9b-heretic:latest"
        );
        assert_eq!(d1.score_q15, 32_686);
        let verified = LanFrame64::try_from_bytes(d1.frame.as_bytes()).expect("valid route frame");
        assert_eq!(verified.opcode, OpCode::OllamaRoute as u8);

        let d2 = router.route("Prove Ada 2022 gnatprove SPARK Q15 advantage in no_std");
        assert_eq!(d2.domain, TaskDomain::SparkNoStdCodegen);
        assert_eq!(d2.model.worker_id, 1);
        assert_eq!(
            d2.model.hf_repo,
            "PeetPedro/qwen4_exp-8b-obliterated-abliterated-kompress-ultra"
        );
        assert_eq!(d2.score_q15, 32_686);
    }
}
