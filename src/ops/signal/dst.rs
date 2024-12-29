use std::{any::Any, ops::{AddAssign, DivAssign, Mul, MulAssign}};

use num::{complex::ComplexFloat, traits::FloatConst, Complex, Float, NumCast, One, Zero};
use slice_ops::Slice;

use super::SliceFft;

#[const_trait]
pub trait SliceDst<T>: Slice<Item = T>
{       
    /// Discrete sine transform I
    fn dst_i(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>;
    /// Discrete sine transform II
    fn dst_ii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    /// Discrete sine transform III
    fn dst_iii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real> + Mul<T, Output = Complex<T::Real>>;
    /// Discrete sine transform IV
    fn dst_iv(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>;
}

impl<T> SliceDst<T> for [T]
{
    fn dst_i(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>
    {
        let len = self.len();
        if len <= 1
        {
            return
        }

        let mut y: Vec<_> = core::iter::once(Zero::zero())
            .chain(self.iter()
                .map(|&x| x.into())
            ).chain(core::iter::once(Zero::zero()))
            .chain(self.iter()
                .rev()
                .map(|&x| (-x).into())
            ).collect();
        y.fft();

        let ylen = y.len();
        let ylen_sqrt = Float::sqrt(<T::Real as NumCast>::from(ylen).unwrap());
        for y in y.iter_mut()
        {
            *y /= ylen_sqrt
        }

        let zero = T::Real::zero();
        let one = T::Real::one();
        let two = one + one;
        let ymul = Complex::new(zero, Float::recip(two));
    
        let y2 = y.split_off(len + 1);

        for ((x, y1), y2) in self.iter_mut()
            .zip(y.into_iter()
                .skip(1)
            ).zip(y2.into_iter()
                .rev()
            )
        {
            let y = (y1 - y2)*ymul;
            if let Some(x) = <dyn Any>::downcast_mut::<Complex<_>>(x as &mut dyn Any)
            {
                *x = y
            }
            else
            {
                *x = y.re.into()
            }
        }
    }
    fn dst_ii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign
    {
        let len = self.len();
        if len <= 1
        {
            return
        }
        let lenf = <T::Real as NumCast>::from(len).unwrap();

        let mut y: Vec<_> = self.iter()
            .map(|&x| x.into())
            .chain(self.iter()
                .rev()
                .map(|&x| -x.into())
            ).collect();
        y.fft();
    
        let zero = T::Real::zero();
        let one = T::Real::one();
        let two = one + one;
    
        let mul = Complex::new(zero, Float::recip(Float::sqrt(lenf))/two);
        for y in y.iter_mut()
        {
            *y *= mul
        }
    
        let m1 = (1..len).map(|i| {
                let i = <T::Real as NumCast>::from(i).unwrap();
                Complex::from_polar(FloatConst::FRAC_1_SQRT_2(), -i*FloatConst::FRAC_PI_2()/lenf)
            }).chain(core::iter::once(Complex::new(zero, -one)));
    
        let m2 = (1..len).map(|i| {
            let i = <T::Real as NumCast>::from(i).unwrap();
            Complex::from_polar(-T::Real::FRAC_1_SQRT_2(), i*FloatConst::FRAC_PI_2()/lenf)
        });
    
        y.remove(0);
        let y2 = y.split_off(len);
    
        for ((x, y1), y2) in self.iter_mut()
            .zip(y.into_iter()
                .zip(m1)
                .map(|(y, m1)| y*m1)
            ).zip(y2.into_iter()
                .rev()
                .zip(m2)
                .map(|(y, m2)| y*m2)
                .chain(core::iter::once(Zero::zero()))
            )
        {
            let y = y1 + y2;
            if let Some(x) = <dyn Any>::downcast_mut::<Complex<_>>(x as &mut dyn Any)
            {
                *x = y
            }
            else
            {
                *x = y.re.into()
            }
        }
    }
    fn dst_iii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real> + Mul<T::Real, Output = Complex<T::Real>> + Mul<T, Output = Complex<T::Real>>
    {
        let len = self.len();
        if len <= 1
        {
            return
        }
        let lenf = <T::Real as NumCast>::from(len).unwrap();

        let m1 = (1..len).map(|i| {
                let i = <T::Real as NumCast>::from(i).unwrap();
                Complex::from_polar(T::Real::FRAC_1_SQRT_2(), i*T::Real::FRAC_PI_2()/lenf)
            }).chain(core::iter::once(Complex::i()));
        let m2 = (1..len).map(|i| {
                let i = <T::Real as NumCast>::from(i).unwrap();
                Complex::from_polar(-T::Real::FRAC_1_SQRT_2(), -i*T::Real::FRAC_PI_2()/lenf)
            }).rev();
        
        let mut y: Vec<_> = core::iter::once(Complex::zero())
            .chain(self.iter()
                .zip(m1)
                .map(|(&x, m1)| m1*x)
            ).chain(self[..len - 1].iter()
                .rev()
                .zip(m2)
                .map(|(&x, m2)| m2*x)
            ).collect();
        y.ifft();

        let zero = T::Real::zero();
        let one = T::Real::one();
        let two = one + one;

        let ymul = Complex::new(zero, -Float::sqrt(lenf)*two);
        for (x, mut y) in self.iter_mut()
            .zip(y.into_iter())
        {
            y *= ymul;
            if let Some(x) = <dyn Any>::downcast_mut::<Complex<_>>(x as &mut dyn Any)
            {
                *x = y
            }
            else
            {
                *x = y.re.into()
            }
        }
    }
    fn dst_iv(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T::Real, Output = Complex<T::Real>> + Mul<T, Output = Complex<T::Real>>
    {
        let len = self.len();
        if len <= 1
        {
            return
        }
        let lenf = <T::Real as NumCast>::from(len).unwrap();

        let m1: Vec<_> = (0..len).map(|i| {
                let i = <T::Real as NumCast>::from(i).unwrap();
                Complex::from_polar(T::Real::FRAC_1_SQRT_2(), -i*T::Real::FRAC_PI_2()/lenf)
            }).chain(core::iter::once(Complex::i()))
            .collect();
        let m2: Vec<_> = (1..=len).map(|i| {
                let i = <T::Real as NumCast>::from(i).unwrap();
                Complex::from_polar(-T::Real::FRAC_1_SQRT_2(), i*T::Real::FRAC_PI_2()/lenf)
            }).collect();

        let mut y: Vec<_> = self.iter()
            .zip(m1.iter())
            .map(|(&x, &m1)| m1*x)
            .chain(self.iter()
                .rev()
                .zip(m2.iter()
                    .rev()
                ).map(|(&x, &m2)| m2*x)
            ).collect();
        y.fft();

        let zero = T::Real::zero();

        let ymul = Complex::new(zero, T::Real::FRAC_1_SQRT_2()/Float::sqrt(lenf))*Complex::cis(-T::Real::FRAC_PI_4()/lenf);
        for y in y.iter_mut()
        {
            *y *= ymul
        }

        let y2 = y.split_off(len);
        
        for ((x, y1), y2) in self.iter_mut()
            .zip(y.into_iter()
                .zip(m1.into_iter())
                .map(|(y, m1)| y*m1)
            ).zip(y2.into_iter()
                .rev()
                .zip(m2.into_iter())
                .map(|(y, m2)| y*m2)
            )
        {
            let y = y1 + y2;
            if let Some(x) = <dyn Any>::downcast_mut::<Complex<_>>(x as &mut dyn Any)
            {
                *x = y
            }
            else
            {
                *x = y.re.into()
            }
        }
    }
}