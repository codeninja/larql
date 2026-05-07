//! Shared Per-Layer-Embedding sidecar writing for both the float
//! (`write_f32`) and Q4_K (`write_q4k`) extract paths.
//!
//! The two writers used to carry near-identical PLE blocks. Q4_K
//! shipped first (chrishayuk/larql#49 mitigation phase 0) so it had
//! the working pattern; the float path had no PLE handling at all and
//! silently dropped 129 tensors on `larql extract gemma-4-E4B-it`,
//! producing garbage at INFER time. After the fix, both writers had
//! their own copy of the same logic — the kind of duplication that
//! caused the bug in the first place.
//!
//! This module owns:
//!
//! - [`per_layer_norm_keys`] — the two per-layer 1-D vectors a writer
//!   must add to its norms.bin loop on Gemma 4 (`layer_scalar` and
//!   `post_per_layer_input_norm`). Returned as keys, not as a writer
//!   call, because each writer's outer norms loop does its own
//!   formatting and offset bookkeeping.
//! - [`write_global_projection_norm`] — the one global 1-D PLE vector
//!   that lives at the end of norms.bin
//!   (`per_layer_projection_norm.weight`).
//! - [`write_ple_weights_bin`] — the four 2-D PLE tensors (two
//!   globals + two-per-layer) into the sidecar `ple_weights.bin` as
//!   f16, with the manifest entries appended to the writer's
//!   `WeightEntry` vec.
//!
//! All three helpers no-op cleanly when `arch.has_per_layer_embeddings()`
//! is false, so the writers can call them unconditionally without
//! adding a guard at every call site. None of these helpers create or
//! truncate norms.bin themselves — the writer owns that file handle so
//! the offsets stay sequential with whatever else it's writing.

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::config::dtype::StorageDtype;
use crate::error::VindexError;
use crate::format::filenames::{NORMS_BIN, PLE_WEIGHTS_BIN};

use super::write_f32::{kind, WeightEntry, WeightSource};

/// Per-layer 1-D PLE vector keys to append to a writer's norms.bin
/// loop, in the order the existing writers emitted them. Empty when
/// the architecture has no PLE / no per-layer scalar.
///
/// Caller is expected to read each key via `source.get_vector(&key)`
/// and write it as a [`kind::VECTOR`] entry into norms.bin — keep the
/// tensor → bytes encoding inline so the offset bookkeeping stays
/// next to the other norm vectors in that loop.
pub(super) fn per_layer_norm_keys(
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Vec<String> {
    let mut keys = Vec::new();
    // Gemma 4 per-layer scalar — `layer_scalar_key` returns `Some` only
    // on Gemma 4 (default trait impl returns `None`), so this naturally
    // gates other archs out without a family check here.
    if let Some(k) = arch.layer_scalar_key(layer) {
        keys.push(k);
    }
    if arch.has_per_layer_embeddings() {
        if let Some(k) = arch.post_per_layer_input_norm_key(layer) {
            keys.push(k);
        }
    }
    keys
}

/// Append the global `per_layer_projection_norm.weight` vector to an
/// open `norms.bin` writer. No-op when the architecture has no PLE or
/// the source doesn't expose the vector.
///
/// `offset` is bumped by however many bytes were written; pass it to
/// the next writer in the same file. Returns `Ok(())` even if the
/// vector was absent — extracting an architecture that ships without
/// the projection norm is not an error here, the loader is the
/// authority on what's required.
pub(super) fn write_global_projection_norm(
    source: &dyn WeightSource,
    norms_file: &mut BufWriter<std::fs::File>,
    entries: &mut Vec<WeightEntry>,
    offset: &mut u64,
    dtype: StorageDtype,
) -> Result<(), VindexError> {
    let arch = source.arch();
    if !arch.has_per_layer_embeddings() {
        return Ok(());
    }
    if let Some(data) = source.get_vector("per_layer_projection_norm.weight") {
        let bytes = crate::config::dtype::encode_floats(&data, dtype);
        norms_file.write_all(&bytes)?;
        entries.push(WeightEntry {
            key: "per_layer_projection_norm.weight".into(),
            kind: kind::VECTOR.into(),
            shape: vec![data.len()],
            offset: *offset,
            length: bytes.len() as u64,
            file: NORMS_BIN.into(),
        });
        *offset += bytes.len() as u64;
    }
    Ok(())
}

/// Write the four 2-D PLE tensors (`per_layer_model_projection.weight`,
/// `embed_tokens_per_layer.weight`, plus per-layer
/// `per_layer_input_gate.weight` and `per_layer_projection.weight`)
/// into `ple_weights.bin` as f16, appending [`kind::TENSOR_F16`]
/// manifest entries to `entries`.
///
/// f16 (not the writer's global dtype) is deliberate: Q4_K's
/// per-super-block calibration zeros out the non-outlier cells of
/// embedding-style tensors, and the resulting noise compounds across
/// every layer's PLE residual add — so the Q4_K writer chose f16
/// here. Using f16 in the float writer too means the loader can hydrate
/// either kind of vindex through the same `tensor_f16` path.
///
/// No-op when `arch.has_per_layer_embeddings()` is false. Creates +
/// flushes `ple_weights.bin` itself — it's a self-contained sidecar,
/// not interleaved with any other writer's bytes.
pub(super) fn write_ple_weights_bin(
    source: &dyn WeightSource,
    dir: &Path,
    entries: &mut Vec<WeightEntry>,
) -> Result<(), VindexError> {
    let arch = source.arch();
    if !arch.has_per_layer_embeddings() {
        return Ok(());
    }
    let num_layers = source.num_layers();

    let ple_path = dir.join(PLE_WEIGHTS_BIN);
    let mut ple_file = BufWriter::new(std::fs::File::create(&ple_path)?);
    let mut ple_offset: u64 = 0;
    let ple_dtype = StorageDtype::F16;

    let write_tensor = |file: &mut BufWriter<std::fs::File>,
                        manifest: &mut Vec<WeightEntry>,
                        offset: &mut u64,
                        key: String,
                        data: Option<(Vec<f32>, usize, usize)>|
     -> Result<(), VindexError> {
        if let Some((floats, rows, cols)) = data {
            let bytes = crate::config::dtype::encode_floats(&floats, ple_dtype);
            file.write_all(&bytes)?;
            manifest.push(WeightEntry {
                key,
                kind: kind::TENSOR_F16.into(),
                shape: vec![rows, cols],
                offset: *offset,
                length: bytes.len() as u64,
                file: PLE_WEIGHTS_BIN.into(),
            });
            *offset += bytes.len() as u64;
        }
        Ok(())
    };

    // Global: model projection [ple_dim·num_layers, hidden]
    write_tensor(
        &mut ple_file,
        entries,
        &mut ple_offset,
        "per_layer_model_projection.weight".into(),
        source.get_tensor("per_layer_model_projection.weight"),
    )?;

    // Global: per-token embedding table [vocab, ple_dim·num_layers]
    if let Some(key) = arch.per_layer_embed_key() {
        write_tensor(
            &mut ple_file,
            entries,
            &mut ple_offset,
            key.clone(),
            source.get_tensor(&key),
        )?;
    }

    // Per-layer: input_gate + projection
    for layer in 0..num_layers {
        if let Some(k) = arch.per_layer_input_gate_key(layer) {
            write_tensor(
                &mut ple_file,
                entries,
                &mut ple_offset,
                k.clone(),
                source.get_tensor(&k),
            )?;
        }
        if let Some(k) = arch.per_layer_projection_key(layer) {
            write_tensor(
                &mut ple_file,
                entries,
                &mut ple_offset,
                k.clone(),
                source.get_tensor(&k),
            )?;
        }
    }

    ple_file.flush()?;
    Ok(())
}
