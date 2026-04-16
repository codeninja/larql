//! Substrate-dependent residual persistence profiles.
//!
//! Numbers measured in `experiments/07_wasm_compute/persistence_sweep.py`:
//! a unit-norm direction is written into the residual at L10 and the cosine
//! with downstream FFN inputs is measured. v11's attention mixes residuals
//! gently (cos > 0.99 for all 9 downstream layers); Gemma 3 4B's attention is
//! ~87× more aggressive (cos drops 0.47 → 0.04 over 8 hops). The "usable hops"
//! field is the depth budget before a refresh edge is needed to re-amplify
//! the tag.
//!
//! See also `WASM_GATE_ARCHITECTURE.md` §2.1 and §8.2.

#[derive(Debug, Clone)]
pub struct PersistenceProfile {
    /// Substrate identifier, e.g. "v11-tiny" or "gemma3-4b".
    pub substrate: &'static str,
    /// Approximate cosine similarity between a write at L and FFN input at L+1.
    pub cos_per_hop: f32,
    /// Hops a write stays usable (gate fires reliably) before a refresh edge
    /// is recommended.
    pub usable_hops: usize,
    /// Recommended layer for placing a refresh edge in deeper programs;
    /// `None` if the substrate doesn't need them within typical depth.
    pub refresh_at: Option<usize>,
}

/// v11 TinyModel: dim=512, 20 layers. Attention is gentle, residual is
/// effectively a persistent register file for the second half of the network.
pub const V11_TINY: PersistenceProfile = PersistenceProfile {
    substrate: "v11-tiny",
    cos_per_hop: 0.99,
    usable_hops: 8,
    refresh_at: None,
};

/// Gemma 3 4B Instruct: dim=2560, 34 layers. Stronger attention rewrites the
/// residual aggressively; relay edges are required for programs deeper than
/// ~3 hops. A single refresh at L15 recovers cos 0.10 → 0.62 at L16.
pub const GEMMA3_4B: PersistenceProfile = PersistenceProfile {
    substrate: "gemma3-4b",
    cos_per_hop: 0.48,
    usable_hops: 3,
    refresh_at: Some(15),
};
