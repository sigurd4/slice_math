use std::ops::{AddAssign, MulAssign};

use num::Zero;
use slice_ops::Slice;

#[const_trait]
pub trait SlicePolyEval<T>: Slice<Item = T>
{
    /// Evaluates a polynomial with powers in rising order from left to right.
    fn poly_eval<Rhs>(&self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero + Copy,
        Rhs: Copy;
    /// Evaluates a polynomial with powers in rising order from right to left.
    fn rpoly_eval<Rhs>(&self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero + Copy,
        Rhs: Copy;
}

impl<T> SlicePolyEval<T> for [T]
{
    fn poly_eval<Rhs>(&self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero + Copy,
        Rhs: Copy
    {
        let mut y = T::zero();
        let mut i = self.len();
        while i > 0
        {
            i -= 1;
            y *= rhs;
            y += self[i];
        }
        y
    }
    fn rpoly_eval<Rhs>(&self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero + Copy,
        Rhs: Copy
    {
        let n = self.len();
        let mut y = T::zero();
        let mut i = 0;
        while i < n
        {
            y *= rhs;
            y += self[i];
            i += 1;
        }
        y
    }
}