//! WalkModel — holds model weights + vindex, exposes per-layer walk FFN.
//!
//! Loads model weights once. Each call to ffn_layer() runs the Rust sparse
//! FFN for one layer: gate KNN from vindex, gather up/down rows, compute output.
//! Used by the MLX walk_ffn integration to replace dense MLP with sparse walk FFN.

use pyo3::prelude::*;
use numpy::{PyArray2, IntoPyArray};
use ndarray::Array2;

use larql_vindex::{
    VectorIndex, SilentLoadCallbacks, load_vindex_tokenizer, tokenizers,
};
use larql_inference::{ModelWeights, WalkFfn, predict_with_ffn};
use larql_inference::ffn::FfnBackend;

/// Holds model weights + vindex for walk FFN inference.
/// Weights loaded once, reused across calls.
#[pyclass(name = "WalkModel", unsendable)]
pub struct PyWalkModel {
    weights: ModelWeights,
    index: VectorIndex,
    tokenizer: tokenizers::Tokenizer,
    top_k: usize,
    path: String,
}

#[pymethods]
impl PyWalkModel {
    /// Load a walk model from a vindex directory.
    ///
    /// Loads all model weights (requires --level all extract).
    /// Gate vectors are mmap'd for KNN. Up/down weights loaded for sparse FFN.
    #[new]
    #[pyo3(signature = (path, top_k=8192))]
    fn new(path: &str, top_k: usize) -> PyResult<Self> {
        let dir = std::path::Path::new(path);

        let mut load_cb = SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir, &mut load_cb)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let weights = larql_vindex::load_model_weights(dir, &mut load_cb)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(
                format!("Failed to load model weights: {e}")
            ))?;

        let tokenizer = load_vindex_tokenizer(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(Self { weights, index, tokenizer, top_k, path: path.to_string() })
    }

    /// Run full forward pass with walk FFN. Returns [(token, probability)].
    #[pyo3(signature = (prompt, top_k_predictions=5))]
    fn predict(&self, prompt: &str, top_k_predictions: usize) -> PyResult<Vec<(String, f64)>> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let result = predict_with_ffn(
            &self.weights, &self.tokenizer, &token_ids, top_k_predictions, &walk_ffn
        );

        Ok(result.predictions)
    }

    /// Run walk FFN for a single layer. Takes (seq_len, hidden) numpy input,
    /// returns (seq_len, hidden) numpy output.
    ///
    /// This is what the MLX integration calls per-layer to replace the dense MLP.
    fn ffn_layer<'py>(
        &self, py: Python<'py>, layer: usize, x: Vec<f32>, seq_len: usize
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let hidden = self.weights.hidden_size;
        let x_arr = Array2::from_shape_vec((seq_len, hidden), x)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let output = walk_ffn.forward(layer, &x_arr);

        Ok(output.into_pyarray(py))
    }

    #[getter]
    fn num_layers(&self) -> usize { self.weights.num_layers }

    #[getter]
    fn hidden_size(&self) -> usize { self.weights.hidden_size }

    #[getter]
    fn top_k(&self) -> usize { self.top_k }

    fn __repr__(&self) -> String {
        format!(
            "WalkModel(path='{}', layers={}, hidden={}, top_k={})",
            self.path, self.weights.num_layers, self.weights.hidden_size, self.top_k
        )
    }
}
