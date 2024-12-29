use std::ops::Mul;

use num::{NumCast, Zero};
use slice_ops::Slice;

use crate::ops::SliceTrimZeros;

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

impl<T> SliceDerivatePolynomial<T> for [T]
{
    fn derivate_polynomial<S>(&self) -> S
    where
        T: NumCast + Zero + Mul + Copy,
        S: FromIterator<<T as Mul>::Output>
    {
        let s = self.trim_zeros_back();
        if s.len() < 2
        {
            return core::iter::empty()
                .collect()
        }
        s[1..].into_iter()
            .enumerate()
            .map(|(i, b)| *b*T::from(i + 1).unwrap())
            .collect()
    }
    fn derivate_rpolynomial<S>(&self) -> S
    where
        T: NumCast + Zero + Mul + Copy,
        S: FromIterator<<T as Mul>::Output>
    {
        let s = self.trim_zeros_front();
        let n = s.len();
        if n < 2
        {
            return core::iter::empty()
                .collect()
        }
        let nm1 = n - 1;
        s[..nm1].into_iter()
            .enumerate()
            .map(|(i, b)| *b*T::from(nm1 - i).unwrap())
            .collect()
    }
}