use std::{any::Any, ops::{AddAssign, DivAssign, Mul, MulAssign}};

use num::{complex::ComplexFloat, traits::FloatConst, Complex, Float, NumCast, One, Zero};
use slice_ops::Slice;

use super::SliceFft;

#[const_trait]
pub trait SliceDct<T>: Slice<Item = T>
{       
    /// Discrete cosine transform I
    fn dct_i(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + DivAssign<T::Real> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>;
    /// Discrete cosine transform II
    fn dct_ii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    /// Discrete cosine transform III
    fn dct_iii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>> + DivAssign<T::Real>;
    /// Discrete cosine transform IV
    fn dct_iv(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>;
}

impl<T> SliceDct<T> for [T]
{
    fn dct_i(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + DivAssign<T::Real> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>
    {
        let len = self.len();
        if len <= 1
        {
            return
        }

        let mut y: Vec<_> = self.iter()
            .map(|&x| x.into())
            .chain(self[1..len - 1].iter()
                .rev()
                .map(|&x| x.into())
            ).collect();
        for y in y[1..len - 1].iter_mut()
        {
            *y /= FloatConst::SQRT_2()
        }
        for y in y[len..].iter_mut()
        {
            *y /= T::Real::SQRT_2()
        }
        y.fft();

        let ylen = y.len();
        let ylen_sqrt = Float::sqrt(<T::Real as NumCast>::from(ylen).unwrap());
        for y in y.iter_mut()
        {
            *y /= ylen_sqrt
        }

        let y2 = y.split_off(len);

        for ((x, y1), y2) in self.iter_mut()
            .zip(y.into_iter())
            .zip(core::iter::once(Zero::zero())
                .chain(y2.into_iter()
                    .rev()
                ).chain(core::iter::once(Zero::zero()))
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
        for x in self[1..len - 1].iter_mut()
        {
            *x /= T::Real::SQRT_2()
        }
    }
    fn dct_ii(&mut self)
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

        let m1 = core::iter::once(One::one())
            .chain((1..len).map(|i| {
                let i = <T::Real as NumCast>::from(i).unwrap();
                Complex::from_polar(T::Real::FRAC_1_SQRT_2(), -i*T::Real::FRAC_PI_2()/lenf)
            }));
        let m2 = (1..len).map(|i| {
                let i = <T::Real as NumCast>::from(i).unwrap();
                Complex::from_polar(T::Real::FRAC_1_SQRT_2(), i*T::Real::FRAC_PI_2()/lenf)
            });

        let mut y: Vec<_> = self.iter()
            .map(|&x| x.into())
            .chain(self.iter()
                .rev()
                .map(|&x| x.into())
            ).collect();
        y.fft();

        let y2 = y.split_off(len);

        let one = T::Real::one();
        let two = one + one;
        let ydiv = Float::sqrt(lenf)*two;

        for ((x, y1), y2) in self.iter_mut()
            .zip(y.into_iter()
                .zip(m1)
                .map(|(y, m1)| y*m1)
            ).zip(core::iter::once(Zero::zero())
                .chain(y2.into_iter()
                    .rev()
                    .zip(m2)
                    .map(|(y, m2)| y*m2)
                )
            )
        {
            let y = (y1 + y2)/ydiv;
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
    fn dct_iii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>> + Mul<T::Real, Output = Complex<T::Real>> + DivAssign<T::Real>
    {
        let len = self.len();
        if len <= 1
        {
            return
        }
        let lenf = <T::Real as NumCast>::from(len).unwrap();

        let m1 = core::iter::once(One::one())
            .chain((1..len).map(|i| {
                let i = <T::Real as NumCast>::from(i).unwrap();
                Complex::from_polar(T::Real::FRAC_1_SQRT_2(), -i*T::Real::FRAC_PI_2()/lenf)
            }));
        let m2 = (1..len).map(|i| {
                let i = <T::Real as NumCast>::from(i).unwrap();
                Complex::from_polar(T::Real::FRAC_1_SQRT_2(), i*T::Real::FRAC_PI_2()/lenf)
            }).rev();
        
        let mut y: Vec<_> = self.iter()
            .zip(m1)
            .map(|(&x, m1)| m1*x)
            .chain(core::iter::once(Zero::zero()))
            .chain(self.iter()
                .rev()
                .zip(m2)
                .map(|(&x, m2)| m2*x)
            ).collect();
        y.fft();
        
        let ydiv = Float::sqrt(lenf);
        for (x, mut y) in self.iter_mut()
            .zip(y.into_iter())
        {
            y /= ydiv;
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
    fn dct_iv(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>> + Mul<T::Real, Output = Complex<T::Real>>
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
            }).collect();
        let m2: Vec<_> = (1..=len).map(|i| {
                let i = <T::Real as NumCast>::from(i).unwrap();
                Complex::from_polar(T::Real::FRAC_1_SQRT_2(), i*T::Real::FRAC_PI_2()/lenf)
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
        
        let ymul = Complex::cis(-T::Real::FRAC_PI_4()/lenf)*T::Real::FRAC_1_SQRT_2()/Float::sqrt(lenf);
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

#[cfg(test)]
mod test
{
    use crate::ops::signal::SliceDct;

    #[test]
    fn dct()
    {
        let mut a: [f64; _] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        a.dct_i();
        println!("{:?}", a);
    }
}