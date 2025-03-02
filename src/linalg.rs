use crate::result::Result;

/// Dot (matrix) product.
pub(crate) trait Dot<R> {
    /// Type of the output.
    type Output;
    /// Computes the dot product `self` * `rhs`.
    fn dot(&self, rhs: &R) -> Result<Self::Output>;
}

pub(crate) trait DotBias<R, B>: Dot<R> {
    /// Computes the dot product `self` * `rhs` + `bias`.
    fn dot_bias(&self, rhs: &R, bias: Option<&B>) -> Result<Self::Output>;
}

pub(crate) trait DotAcc<T, R, Y>: Dot<R> {
    fn dot_acc(&self, alpha: T, rhs: &R, output: &mut Y) -> Result<()>;
}
