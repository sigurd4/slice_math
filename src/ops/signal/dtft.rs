use std::ops::{AddAssign, MulAssign};

use num::{complex::ComplexFloat, Complex, One, Zero};
use slice_ops::Slice;

#[const_trait]
pub trait SliceDtft<T>: Slice<Item = T>
{
    /// Discrete time fourier transform
    fn dtft(&self, omega: T::Real) -> Complex<T::Real>
    where
        T: ComplexFloat + Into<Complex<T::Real>>,
        Complex<T::Real>: ComplexFloat<Real = T::Real> + MulAssign + AddAssign;
}

impl<T> SliceDtft<T> for [T]
{   
    fn dtft(&self, omega: T::Real) -> Complex<T::Real>
    where
        T: ComplexFloat + Into<Complex<T::Real>>,
        Complex<T::Real>: ComplexFloat<Real = T::Real> + MulAssign + AddAssign
    {
        let mut y = Complex::zero();
        let z1 = Complex::cis(-omega);
        let mut z = Complex::one();
        for &x in self
        {
            y += x.into()*z;
            z *= z1;
        }
        y
    }
}