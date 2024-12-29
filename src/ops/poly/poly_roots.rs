use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use num::{complex::ComplexFloat, Complex};
use slice_ops::Slice;

use crate::ops::{poly::SlicePolyEval, SliceIntoMatrix, SliceTrimZeros};

const NEWTON_POLYNOMIAL_ROOTS: usize = 16;

#[const_trait]
pub trait SlicePolyRoots<T>: Slice<Item = T>
{
    /// Finds the roots of a polynomial with powers in rising order from left to right
    fn poly_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>> + Into<Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>;
    /// Finds the roots of a polynomial with powers in rising order from right to left
    fn rpoly_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>> + Into<Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>;
}

impl<T> SlicePolyRoots<T> for [T]
{
    fn poly_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>> + Into<Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>
    {
        use ndarray_linalg::eig::EigVals;

        if self.trim_zeros_front().len() <= 1
        {
            return core::iter::empty()
                .collect()
        }

        let c = self.companion_matrix();
        let mut roots = EigVals::eigvals(&c).unwrap();
        let len = roots.len();
        // Use newtons method
        let p: Vec<Complex<<T as ComplexFloat>::Real>> = self.into_iter()
            .map(|p| (*p).into())
            .collect();
        let dp: Vec<_> = p.derivate_polynomial();
        for k in 0..len
        {
            const NEWTON: usize = NEWTON_POLYNOMIAL_ROOTS;

            for _ in 0..NEWTON
            {
                let root = roots[k];
                let df = p.poly_eval(root);
                if df.is_zero()
                {
                    break
                }
                roots[k] -= df/dp.polynomial(root)
            }
        }
        roots.into_iter()
            .collect()
    }
    fn rpoly_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>> + Into<Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>
    {
        use ndarray_linalg::EigVals;

        if self.trim_zeros_front().len() <= 1
        {
            return core::iter::empty()
                .collect()
        }

        let c = self.rcompanion_matrix();
        let mut roots = EigVals::eigvals(&c).unwrap();
        let len = roots.len();
        // Use newtons method
        let p: Vec<Complex<<T as ComplexFloat>::Real>> = self.into_iter()
            .map(|p| (*p).into())
            .collect();
        let dp: Vec<_> = p.derivate_rpolynomial();
        for k in 0..len
        {
            const NEWTON: usize = NEWTON_POLYNOMIAL_ROOTS;

            for _ in 0..NEWTON
            {
                let root = roots[k];
                let df = p.rpolynomial(root);
                if df.is_zero()
                {
                    break
                }
                roots[k] -= df/dp.rpolynomial(root)
            }
        }
        roots.into_iter()
            .collect()
    }
}