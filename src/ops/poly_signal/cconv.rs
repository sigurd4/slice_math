use std::{any::Any, iter::Sum, ops::{AddAssign, Mul, MulAssign}};

use num::{complex::ComplexFloat, Complex, Float, Zero};
use slice_ops::Slice;

use crate::ops::signal::SliceFft;

/// Circular convolution
#[const_trait]
pub trait SliceCConv<T>: Slice<Item = T>
{
    /// Naive circular convolution
    fn cconv_direct<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy,
        C: FromIterator<<T as Mul<Rhs>>::Output>;

    /// Circular convolution of a real-valued sequence through FFT
    fn cconv_real_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + MulAssign<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real> + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        C: FromIterator<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>;
    
    /// Circular convolution through FFT
    fn cconv_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat<Real: Into<<T as Mul<Rhs>>::Output>> + 'static> + Into<Complex<T::Real>>,
        Rhs: ComplexFloat + Into<Complex<Rhs::Real>>,
        Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>: Into<<Complex<T::Real> as Mul<Complex<Rhs::Real>>>::Output>,
        Complex<T::Real>: AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + MulAssign<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + Sum + 'static>,
        Complex<Rhs::Real>: AddAssign + MulAssign,
        C: FromIterator<<T as Mul<Rhs>>::Output>;
}

impl<T> SliceCConv<T> for [T]
{
    fn cconv_direct<Rhs, C>(&self, mut rhs: &[Rhs]) -> C
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy,
        C: FromIterator<<T as Mul<Rhs>>::Output>
    {
        let y_len = self.len().max(rhs.len());

        (0..y_len).map(|n| {
            let mut y = Zero::zero();
            for k in 0..y_len
            {
                let i = k;
                let j = (n + y_len - k) % y_len;
                if i < self.len() && j < rhs.len()
                {
                    y += self[i]*rhs[j]
                }
            }
            y
        }).collect()
    }
    
    fn cconv_real_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + MulAssign<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real> + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        C: FromIterator<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>
    {
        let len = self.len().max(rhs.len());

        let mut x: Vec<T> = self.to_vec();
        let mut h: Vec<Rhs> = rhs.to_vec();
        x.resize(len, T::zero());
        h.resize(len, Rhs::zero());

        let mut x_f = vec![Complex::zero(); len/2 + 1];
        let mut h_f = vec![Complex::zero(); len/2 + 1];
        x.real_fft(&mut x_f);
        h.real_fft(&mut h_f);

        let y_f: Vec<_> = x_f.into_iter()
            .zip(h_f.into_iter())
            .map(|(x_f, h_f)| (x_f*h_f).into())
            .collect();

        let mut y = vec![Zero::zero(); len];
        y.real_ifft(&y_f);
        
        y.into_iter()
            .collect()
    }

    fn cconv_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat<Real: Into<<T as Mul<Rhs>>::Output>> + 'static> + Into<Complex<T::Real>>,
        Rhs: ComplexFloat + Into<Complex<Rhs::Real>>,
        Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>: Into<<Complex<T::Real> as Mul<Complex<Rhs::Real>>>::Output>,
        Complex<T::Real>: AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + MulAssign<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + Sum + 'static>,
        Complex<Rhs::Real>: AddAssign + MulAssign,
        C: FromIterator<<T as Mul<Rhs>>::Output>
    {
        let len = self.len().max(rhs.len());

        let mut x: Vec<Complex<T::Real>> = self.iter()
            .map(|&x| x.into())
            .collect();
        let mut h: Vec<Complex<Rhs::Real>> = rhs.iter()
            .map(|&h| h.into())
            .collect();
        x.resize(len, Zero::zero());
        h.resize(len, Zero::zero());
        x.fft();
        h.fft();

        let mut y: Vec<_> = x.into_iter()
            .zip(h.into_iter())
            .map(|(x, h)| x*h)
            .collect();
        y.ifft();
        
        y.into_iter()
            .map(|y| {
                if let Some(y) = <dyn Any>::downcast_ref::<<T as Mul<Rhs>>::Output>(&y as &dyn Any)
                {
                    *y
                }
                else
                {
                    y.re().into()
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod test
{
    use crate::{ops::poly::SliceCConv, plot};

    #[test]
    fn cconv()
    {
        const N: usize = 1024;
        const M: usize = 20;

        let x: [_; N] = core::array::from_fn(|i| ((i*136 + 50*i*i + 13) % 100) as f64/100.0/(i + 1) as f64);
        let z: [_; M] = core::array::from_fn(|i| ((i*446 + 12*i*i + 59) % 100) as f64/100.0);

        let y1: Vec<_> = x.cconv_direct(&z);
        let y2: Vec<_> = x.cconv_fft(&z);
        
        plot::plot_curves::<N, _>("e(i)", "plots/cconv_error.png",
            [
                &core::array::from_fn(|i| i as f32),
                &core::array::from_fn(|i| i as f32),
            ],
            [
                &core::array::from_fn(|i| y1[i] as f32),
                &core::array::from_fn(|i| y2[i] as f32),
            ]).unwrap();

        let e = y1.into_iter()
            .zip(y2)
            .map(|(y1, y2)| (y1 - y2).abs())
            .sum::<f64>()/(N.max(M)) as f64;
        println!("{:?}", e)
    }
}