//! Residual-stream ABI primitives for LARQL.
//!
//! This crate is the shared low-level layer that any compiled-edge or
//! WASM-gate work in LARQL builds on. It contains nothing model-specific:
//! no loader, no tokenizer, no forward pass. Just the primitives.
//!
//! ## Scope
//!
//! - **`install_edge`** — write one compiled FFN edge into gate/up/down
//!   tensors at a given slot, preserving the trained slots' magnitude
//!   regime. Mirrors the convention from
//!   `experiments/07_wasm_compute/WASM_GATE_ARCHITECTURE.md` §3.1.2.
//! - **Persistence profiles** — substrate-dependent residual cos decay
//!   per hop. v11 (~0.99/hop, 8+ usable hops) vs Gemma 3 4B (~0.48/hop,
//!   ~3 usable hops before refresh edges are needed).
//! - **Norm helpers** — row/column/vector L2 norms used to compute
//!   reference magnitudes when installing edges.
//!
//! ## Out of scope
//!
//! - Forward-pass dispatch (lives in `larql-inference`).
//! - Native compute kernels (will live in `larql-kernels`).
//! - WASM module loading (will live in `neural-wasm`, separate repo).
//! - Tag registry / Gram-Schmidt orthogonalisation (V2 — needs per-substrate
//!   baseline residuals which are loader-coupled).

pub mod edge;
pub mod norm;
pub mod persistence;

pub use edge::{install_edge, EdgeError, EdgeStats};
pub use norm::{col_norm, row_norm, vec_norm};
pub use persistence::{PersistenceProfile, GEMMA3_4B, V11_TINY};
