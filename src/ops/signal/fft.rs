use std::{iter::Sum, ops::{AddAssign, MulAssign}};

use num::{complex::ComplexFloat, Complex, Float, NumCast};
use slice_ops::{ops::SliceMulAssign, Slice};

use crate::fft;

#[const_trait]
pub trait SliceFft<T>: Slice<Item = T>
{
    /// Fast fourier transform without the scaling constant.
    /// 
    /// May be either FFT or IFFT depending on the generic constant `I`.
    fn fft_unscaled<const I: bool>(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum,
        Complex<T::Real>: Into<T>;
    
    /// Performs an iterative, in-place, mixed radix Cooley-Tukey FFT algorithm.
    /// 
    /// If performs the best with slices that have a length that is a power of 2, but also powers of small primes, like 3 and 5 are pretty fast.
    /// 
    /// If the length is not a power of a prime, a mixed radix algorithm will be used.
    /// 
    /// # Examples
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use num::Complex;
    /// use slice_math::*;
    /// 
    /// let x = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ///     .map(|x| <Complex<_> as From<_>>::from(x));
    /// 
    /// let mut y = x;
    /// 
    /// y.as_mut_slice().fft();
    /// y.as_mut_slice().ifft();
    /// 
    /// let avg_error = x.into_iter()
    ///     .zip(y.into_iter())
    ///     .map(|(x, y)| (x - y).norm())
    ///     .sum::<f64>()/x.len() as f64;
    /// assert!(avg_error < 1.0e-16);
    /// ```
    fn fft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum,
        Complex<T::Real>: Into<T>;
        
    /// Performs an iterative, in-place, mixed radix Cooley-Tukey IFFT algorithm.
    /// 
    /// If performs the best with slices that have a length that is a power of 2, but also powers of small primes, like 3 and 5 are pretty fast.
    /// 
    /// If the length is not a power of a prime, a mixed radix algorithm will be used.
    /// 
    /// # Examples
    /// ```rust
    /// #![feature(generic_const_exprs)]
    /// 
    /// use num::Complex;
    /// use slice_math::*;
    /// 
    /// let x = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ///     .map(|x| <Complex<_> as From<_>>::from(x));
    /// 
    /// let mut y = x;
    /// 
    /// y.as_mut_slice().fft();
    /// y.as_mut_slice().ifft();
    /// 
    /// let avg_error = x.into_iter()
    ///     .zip(y.into_iter())
    ///     .map(|(x, y)| (x - y).norm())
    ///     .sum::<f64>()/x.len() as f64;
    /// assert!(avg_error < 1.0e-16);
    /// ```
    fn ifft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + MulAssign<T::Real> + Sum,
        Complex<T::Real>: Into<T>;
        
    /// Real-valued fast fourier transform.
    /// 
    /// The second half of the fourier transform of a real-valued signal is mirrored, and therefore redundant.
    /// This method only returns the first half of the fourier transform, since the second half can be inferred.
    fn real_fft(&self, y: &mut [Complex<T>])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;
        
    /// Real-valued inverse fast fourier transform.
    /// 
    /// The second half of the fourier transform of a real-valued signal is mirrored, and therefore redundant.
    /// This method only takes the first half of the fourier transform, since the second half can be inferred.
    fn real_ifft(&mut self, x: &[Complex<T>])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign + MulAssign<T>;
}

impl<T> SliceFft<T> for [T]
{
    fn fft_unscaled<const I: bool>(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum,
        Complex<T::Real>: Into<T>
    {
        fft::fft_unscaled::<_, I>(self, None)
    }

    fn fft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum,
        Complex<T::Real>: Into<T>
    {
        self.fft_unscaled::<false>()
    }

    fn ifft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + MulAssign<T::Real> + Sum,
        Complex<T::Real>: Into<T>
    {
        self.fft_unscaled::<true>();

        self.mul_assign_all(<T::Real as NumCast>::from(1.0/self.len() as f64).unwrap());
    }
    
    fn real_fft(&self, y: &mut [Complex<T>])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign
    {
        let len = self.len();

        if len <= 1
        {
            return;
        }

        assert_eq!(y.len(), len/2 + 1, "Invalid output buffer length.");

        let mut x: Vec<Complex<T>> = self.into_iter()
            .map(|&x| <Complex<_> as From<_>>::from(x))
            .collect();

        x.fft();
        
        for (x, y) in x.into_iter()
            .zip(y.iter_mut())
        {
            *y = x;
        }
    }
        
    fn real_ifft(&mut self, x: &[Complex<T>])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign + MulAssign<T>
    {
        let len = self.len();

        if len <= 1
        {
            return;
        }

        assert_eq!(x.len(), len/2 + 1, "Invalid input buffer length.");

        let mut x: Vec<Complex<T>> = x.into_iter()
            .map(|&x| x)
            .chain(x[1..(x.len() + len % 2 - 1)].into_iter()
                .rev()
                .map(|&x| x.conj())
            ).collect();
        
        x.ifft();

        for (x, y) in x.into_iter()
            .zip(self.iter_mut())
        {
            *y = x.re();
        }
    }
}

#[cfg(test)]
mod test
{
    use num::Complex;

    use crate::{fft, ops::signal::SliceFft};

    #[test]
    fn fft()
    {
        let mut a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].map(Complex::from);
        a.fft();

        println!("{:?}", a)
    }

    #[test]
    fn fft_accuracy()
    {
        const L: usize = 1024;

        let mut x1: Vec<_> = (0..L).map(|i| Complex::from(((i*136 + 50*i*i + 13) % 100) as f64/100.0)).collect();
        let mut x2 = x1.clone();

        x1.fft();
        /*let fft = FftPlanner::new()
            .plan_fft_forward(L);
        fft.process(&mut x2);*/
        fft::dft_unscaled::<_, false>(&mut x2, &mut None);

        let e = x1.iter()
            .zip(x2)
            .map(|(x1, x2)| (x1 - x2).norm())
            .sum::<f64>()/L as f64;
        println!("{:?}", e)
    }
}