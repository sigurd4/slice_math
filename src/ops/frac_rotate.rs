use std::{any::Any, ops::{AddAssign, MulAssign, SubAssign}};

use num::{complex::ComplexFloat, traits::FloatConst, Complex, Float, NumCast, One, Zero};
use slice_ops::Slice;

use super::signal::SliceFft;

#[const_trait]
pub trait SliceFracRotate<T>: Slice<Item = T>
{
    /// Rotate a slice by a non-integer amount of steps, using FFT time-shifting.
    fn frac_rotate_right(&mut self, shift: T::Real)
    where
        T: ComplexFloat<Real: Into<T> + SubAssign> + Into<Complex<<T>::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>;
    /// Rotate a slice by a non-integer amount of steps, using FFT time-shifting.
    fn frac_rotate_left(&mut self, shift: T::Real)
    where
        T: ComplexFloat<Real: Into<T> + SubAssign> + Into<Complex<<T>::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>;
}

impl<T> SliceFracRotate<T> for [T]
{
    fn frac_rotate_right(&mut self, shift: T::Real)
    where
        T: ComplexFloat<Real: Into<T> + SubAssign> + Into<Complex<<T>::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>
    {
        let (trunc, fract) = if let Some(trunc) = NumCast::from(shift.trunc())
        {
            (trunc, shift.fract())
        }
        else
        {
            (0, shift)
        };
        if !fract.is_zero()
        {
            let mut x: Vec<Complex<T::Real>> = self.iter()
                .map(|&x| x.into())
                .collect();
            x.fft();
            let n = <T::Real as NumCast>::from(x.len()).unwrap();
            let one = T::Real::one();
            let two = one + one;
            for (k, x) in x.iter_mut()
                .enumerate()
            {
                let mut k = <T::Real as NumCast>::from(k).unwrap();
                if k > n/two
                {
                    k -= n;
                }
                
                let z = Complex::cis(-T::Real::TAU()*fract*k/n);
                *x *= z
            }
            x.ifft();
            for (y, x) in self.iter_mut()
                .zip(x.into_iter())
            {
                if let Some(y) = <dyn Any>::downcast_mut::<Complex<T::Real>>(y as &mut dyn Any)
                {
                    *y = x
                }
                else
                {
                    *y = x.re.into()
                }
            }
        }

        if shift.is_sign_positive()
        {
            self.rotate_right(trunc)
        }
        else
        {
            self.rotate_left(trunc)
        }
    }
    fn frac_rotate_left(&mut self, shift: T::Real)
    where
        T: ComplexFloat<Real: Into<T> + SubAssign> + Into<Complex<<T>::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>
    {
        self.frac_rotate_right(-shift)
    }
}

#[cfg(test)]
mod test
{
    #[test]
    fn it_works()
    {
        use crate::ops::SliceFracRotate;

        let mut p = [1.0, 0.0, 0.0, 0.0, 0.0];
        
        p.frac_rotate_right(0.5);
        p.frac_rotate_right(0.5);

        println!("{:?}", p);
    }
}