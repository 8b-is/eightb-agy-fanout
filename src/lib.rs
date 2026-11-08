#![allow(clippy::identity_op)]

use core::mem::{align_of, size_of};
use core::slice;

pub const GATE_BYTE: u8 = 0x62;
pub const FRAME_MAGIC: u16 = 0x8B15;
pub const MAX_TASKS: usize = 64;

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OpCode {
    Heartbeat    = 0x01,
    TaskDispatch = 0x02,
    TaskComplete = 0x03,
    KvmHidInject = 0x04,
}

impl OpCode {
    #[inline(always)]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Heartbeat),
            0x02 => Some(Self::TaskDispatch),
            0x03 => Some(Self::TaskComplete),
            0x04 => Some(Self::KvmHidInject),
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
    pub payload: [u8; 24],   // 32..56: Inline command / HID report / artifact tag
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
        dag.add_task(0, 0);                 // Wave 1: cleaner
        dag.add_task(1, 1 << 0);            // Wave 2: analyzer
        dag.add_task(2, 1 << 0);            // Wave 2: visualizer
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
}
