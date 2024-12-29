use std::{any::Any, iter::Sum, ops::{AddAssign, Mul, MulAssign}};

use num::{complex::ComplexFloat, Complex, Float, Zero};
use slice_ops::Slice;

use crate::ops::signal::SliceFft;

/// Convolution
#[const_trait]
pub trait SliceConv<T>: Slice<Item = T>
{
    /// Performs convolution (i.e. polynomial multiplication) naively.
    #[doc(alias = "poly_mul")]
    fn conv_direct<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy,
        C: FromIterator<<T as Mul<Rhs>>::Output>;

    /// Performs convolution using FFT.
    /// 
    /// # Examples
    /// 
    /// Convolution can be done directly `O(n^2)` or using FFT `O(nlog(n))`.
    /// 
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use slice_math::*;
    /// 
    /// let x = [1.0, 0.0, 1.5, 0.0, 0.0, -1.0];
    /// let h = [1.0, 0.6, 0.3];
    /// 
    /// let y_fft: Vec<f64> = x.convolve_real_fft(&h);
    /// let y_direct: Vec<f64> = x.convolve_direct(&h);
    /// 
    /// let avg_error = y_fft.into_iter()
    ///     .zip(y_direct.into_iter())
    ///     .map(|(y_fft, y_direct)| (y_fft - y_direct).abs())
    ///     .sum::<f64>()/x.len() as f64;
    /// assert!(avg_error < 1.0e-15);
    /// ```
    #[doc(alias = "poly_mul_fft")]
    fn conv_real_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + MulAssign<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real> + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        C: FromIterator<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>;
        
    fn conv_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat<Real: Into<<T as Mul<Rhs>>::Output>> + 'static> + Into<Complex<T::Real>>,
        Rhs: ComplexFloat + Into<Complex<Rhs::Real>>,
        Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>: Into<<Complex<T::Real> as Mul<Complex<Rhs::Real>>>::Output>,
        Complex<T::Real>: AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + MulAssign<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + Sum + 'static>,
        Complex<Rhs::Real>: AddAssign + MulAssign,
        C: FromIterator<<T as Mul<Rhs>>::Output>;
}

impl<T> SliceConv<T> for [T]
{
    fn conv_direct<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy,
        C: FromIterator<<T as Mul<Rhs>>::Output>
    {
        let y_len = (self.len() + rhs.len()).saturating_sub(1);

        (0..y_len).map(|n| {
            let mut y = Zero::zero();
            for k in (n + 1).saturating_sub(self.len())..rhs.len().min(n + 1)
            {
                y += self[n - k]*rhs[k];
            }
            y
        }).collect()
    }
    
    fn conv_real_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + MulAssign<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real> + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        C: FromIterator<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>
    {
        let y_len = (self.len() + rhs.len()).saturating_sub(1);
        let len = y_len.next_power_of_two();

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

        y.truncate(y_len);
        
        y.into_iter()
            .collect()
    }
       
    fn conv_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat<Real: Into<<T as Mul<Rhs>>::Output>> + 'static> + Into<Complex<T::Real>>,
        Rhs: ComplexFloat + Into<Complex<Rhs::Real>>,
        Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>: Into<<Complex<T::Real> as Mul<Complex<Rhs::Real>>>::Output>,
        Complex<T::Real>: AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + MulAssign<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + Sum + 'static>,
        Complex<Rhs::Real>: AddAssign + MulAssign,
        C: FromIterator<<T as Mul<Rhs>>::Output>
    {
        let y_len = (self.len() + rhs.len()).saturating_sub(1);
        let len = y_len.next_power_of_two();

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

        y.truncate(y_len);
        
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
    use crate::{ops::poly::SliceConv, plot};

    #[test]
    fn conv_accuracy()
    {
        const L: usize = 1024;

        let x: Vec<_> = (0..L).map(|i| ((i*136 + 50*i*i + 13) % 100) as f64/100.0/(i + 1) as f64).collect();
        
        let z: Vec<_> = (0..20).map(|i| ((i*446 + 12*i*i + 59) % 100) as f64/100.0).collect();

        let y1: Vec<_> = x.conv_fft(&z);
        let y2: Vec<_> = x.conv_direct(&z);
        let err: Vec<_> = y1.iter()
            .zip(y2.iter())
            .map(|(y1, y2)| y1 - y2)
            .collect();

        plot::plot_curves::<L, _>("e(i)", "plots/conv_error.png",
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
            .sum::<f64>()/L as f64;
        println!("{:?}", e)
    }
}