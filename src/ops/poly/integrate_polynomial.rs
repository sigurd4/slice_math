use std::ops::Div;

use num::{NumCast, Zero};
use slice_ops::Slice;

use crate::ops::SliceTrimZeros;

#[const_trait]
pub trait SliceIntegratePolynomial<T>: Slice<Item = T>
{
    /// Integrates a polynomial with powers in rising order from left to right.
    fn integrate_polynomial<S>(&self, c: <T as Div>::Output) -> S
    where
        T: NumCast + Zero + Div + Copy,
        S: FromIterator<<T as Div>::Output>;
    /// Integrates a polynomial with powers in rising order from right to left.
    fn integrate_rpolynomial<S>(&self, c: <T as Div>::Output) -> S
    where
        T: NumCast + Zero + Div + Copy,
        S: FromIterator<<T as Div>::Output>;
}

impl<T> SliceIntegratePolynomial<T> for [T]
{
    fn integrate_polynomial<S>(&self, c: <T as Div>::Output) -> S
    where
        T: NumCast + Zero + Div + Copy,
        S: FromIterator<<T as Div>::Output>
    {
        core::iter::once(c)
            .chain(self.trim_zeros_back()
                .into_iter()
                .enumerate()
                .map(|(i, b)| *b/T::from(i + 1).unwrap())
            ).collect()
    }
    fn integrate_rpolynomial<S>(&self, c: <T as Div>::Output) -> S
    where
        T: NumCast + Zero + Div + Copy,
        S: FromIterator<<T as Div>::Output>
    {
        let s = self.trim_zeros_front();
        let n = s.len();
        s.into_iter()
            .enumerate()
            .map(|(i, b)| *b/T::from(n - i).unwrap())
            .chain(core::iter::once(c))
            .collect()
    }
}