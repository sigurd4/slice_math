use std::ops::Mul;

use num::{NumCast, Zero};
use slice_ops::Slice;

#[const_trait]
pub trait SliceDerivatePolynomial<T>: Slice<Item = T>
{
    /// Derivates a polynomial with powers in rising order from left to right.
    fn derivate_polynomial<S>(&self) -> S
    where
        T: NumCast + Zero + Mul + Copy,
        S: FromIterator<<T as Mul>::Output>;
    /// Derivates a polynomial with powers in rising order from right to left.
    fn derivate_rpolynomial<S>(&self) -> S
    where
        T: NumCast + Zero + Mul + Copy,
        S: FromIterator<<T as Mul>::Output>;
}