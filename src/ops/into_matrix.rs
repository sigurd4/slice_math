use std::ops::{DivAssign, Mul, Neg};

use num::{One, Zero};
use slice_ops::Slice;

use super::SliceTrimZeros;

#[const_trait]
pub trait SliceIntoMatrix<T>: Slice<Item = T>
{
    /// Toeplitz matrix
    fn toeplitz_matrix(&self) -> ndarray::Array2<T>
    where
        T: Copy;
    /// Hankel matrix
    fn hankel_matrix(&self, r: &[T]) -> ndarray::Array2<T>
    where
        T: Copy;
    /// Companion matrix for a given polynomial with powers in rising order from left to right.
    fn companion_matrix(&self) -> ndarray::Array2<<T as Neg>::Output>
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>;
    /// Companion matrix for a given polynomial with powers in rising order from right to left.
    fn rcompanion_matrix(&self) -> ndarray::Array2<<T as Neg>::Output>
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>;
    /// Vandermonde matrix
    fn vandermonde_matrix(&self, n: usize) -> ndarray::Array2<T>
    where
        T: One + Copy + Mul;
}

impl<T> SliceIntoMatrix<T> for [T]
{
    fn toeplitz_matrix(&self) -> ndarray::Array2<T>
    where
        T: Copy
    {
        use ndarray::Array2;

        let n = self.len();
        Array2::from_shape_fn((n, n), |(i, j)| self[if i >= j {i - j} else {j - i}])
    }
    fn hankel_matrix(&self, r: &[T]) -> ndarray::Array2<T>
    where
        T: Copy
    {
        use ndarray::Array2;

        let n = self.len();
        let m = r.len();
        Array2::from_shape_fn((n, m), |(i, j)| if i + j < n
        {
            self[i + j]
        }
        else
        {
            r[i + j + 1 - n]
        })
    }
    fn companion_matrix(&self) -> ndarray::Array2<<T as Neg>::Output>
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>
    {
        let s = self.trim_zeros_back();
        let l = s.len();
        if l <= 1
        {
            return ndarray::Array2::from_shape_fn((0, 0), |_| Zero::zero())
        }
        let n = l - 1;
        let mut c = ndarray::Array2::from_shape_fn((n, n), |_| Zero::zero());
        let mut i = 0;
        while i < n
        {
            if i > 0
            {
                c[(i, i - 1)] = One::one();
            }
            c[(i, n - 1)] = -s[i];
            c[(i, n - 1)] /= s[n];
            i += 1;
        }
        c
    }
    fn rcompanion_matrix(&self) -> ndarray::Array2<<T as Neg>::Output>
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>
    {
        let s = self.trim_zeros_front();
        let l = s.len();
        if l <= 1
        {
            return ndarray::Array2::from_shape_fn((0, 0), |_| Zero::zero())
        }
        let n = l - 1;
        let mut c = ndarray::Array2::from_shape_fn((n, n), |_| Zero::zero());
        let mut i = n;
        loop
        {
            c[(n - i, n - 1)] = -s[i];
            c[(n - i, n - 1)] /= s[0];
            i -= 1;
            if i > 0
            {
                c[(i, i - 1)] = One::one();
            }
            else
            {
                break
            }
        }
        c
    }
    fn vandermonde_matrix(&self, n: usize) -> ndarray::Array2<T>
    where
        T: One + Copy + Mul
    {
        let l = self.len();
        let mut m = ndarray::Array2::from_elem((l, n), T::one());
        for j in (0..n - 1).rev()
        {
            for k in 0..l
            {
                m[(k, j)] = self[k]*m[(k, j + 1)]
            }
        }
        m
    }
}