use std::ops::{AddAssign, MulAssign};

use num::{complex::ComplexFloat, Complex};
use slice_ops::Slice;

use super::SliceFft;

#[const_trait]
pub trait SliceFht<T>: Slice<Item = T>
{
    /// Fast Hartley transform
    fn fht(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    /// Inverse fast Hartley transform
    fn ifht(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>;
}

impl<T> SliceFht<T> for [T]
{
    fn fht(&mut self)
    where
        T: ComplexFloat<Real: Into<T> + Into<Complex<T::Real>>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign
    {
        if core::intrinsics::type_id::<T>() == core::intrinsics::type_id::<Complex<T::Real>>()
        {
            let x = unsafe {
                core::mem::transmute::<&mut Self, &mut [Complex<T::Real>]>(self)
            };
            x.fft();

            for x in x.iter_mut()
            {
                *x = (x.re - x.im).into()
            }

            return
        }

        let mut y: Vec<_> = self.iter()
            .map(|&y| y.into())
            .collect();
        y.fft();
        
        for (x, y) in self.iter_mut()
            .zip(y.into_iter())
        {
            *x = (y.re - y.im).into()
        }
    }
    fn ifht(&mut self)
    where
        T: ComplexFloat<Real: Into<T> + Into<Complex<T::Real>>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>
    {
        if core::intrinsics::type_id::<T>() == core::intrinsics::type_id::<Complex<T::Real>>()
        {
            let x = unsafe {
                core::mem::transmute::<&mut Self, &mut [Complex<T::Real>]>(self)
            };
            x.ifft();

            for x in x.iter_mut()
            {
                *x = (x.re + x.im).into()
            }

            return
        }

        let mut y: Vec<_> = self.iter()
            .map(|&y| y.into())
            .collect();
        y.ifft();
        
        for (x, y) in self.iter_mut()
            .zip(y.into_iter())
        {
            *x = (y.re + y.im).into()
        }
    }
}