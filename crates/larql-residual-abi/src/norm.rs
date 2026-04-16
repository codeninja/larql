//! L2 norms over rows, columns, and slices.

use ndarray::ArcArray2;

pub fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn row_norm(tensor: &ArcArray2<f32>, row: usize) -> f32 {
    let r = tensor.row(row);
    r.dot(&r).sqrt()
}

pub fn col_norm(tensor: &ArcArray2<f32>, col: usize) -> f32 {
    let c = tensor.column(col);
    c.dot(&c).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn vec_norm_unit() {
        assert!((vec_norm(&[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((vec_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn row_col_norms() {
        let m = array![[3.0_f32, 0.0], [0.0, 4.0]].into_shared();
        assert!((row_norm(&m, 0) - 3.0).abs() < 1e-6);
        assert!((row_norm(&m, 1) - 4.0).abs() < 1e-6);
        assert!((col_norm(&m, 0) - 3.0).abs() < 1e-6);
        assert!((col_norm(&m, 1) - 4.0).abs() < 1e-6);
    }
}
