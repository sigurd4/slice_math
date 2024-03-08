use std::{f64::consts::TAU, iter::Sum, ops::{AddAssign, Mul, MulAssign}};

use num::{complex::ComplexFloat, traits::{Inv, SaturatingSub}, Complex, Float, NumCast, One, Saturating, Zero};
use slice_ops::{Slice, SliceOps};

use crate::fft;

pub trait SliceMath<T>: SliceOps<T>
{
    fn recip_assign_all(&mut self)
    where
        T: Inv<Output = T>;
    fn conj_assign_all(&mut self)
    where
        T: ComplexFloat;

    fn convolve_direct<Rhs, C>(&self, rhs: &[Rhs]) -> C
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
    fn convolve_real_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        C: FromIterator<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>;
        
    fn convolve_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum + Mul<Rhs>,
        Rhs: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<Rhs::Real>> + Sum,
        <T as Mul<Rhs>>::Output: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>> + Sum,
        C: FromIterator<<T as Mul<Rhs>>::Output>;
        
    fn dtft(&self, omega: T::Real) -> T
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum;
        
    fn real_dtft(&self, omega: T) -> Complex<T>
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;
        
    #[doc(hidden)]
    fn fft_unscaled<const I: bool>(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<<T>::Real>> + Sum;
    
    /// Performs an iterative, in-place radix-2 FFT algorithm as described in https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Data_reordering,_bit_reversal,_and_in-place_algorithms.
    /// If length is not a power of two, it uses the DFT, which is a lot slower.
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
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum;
        
    /// Performs an iterative, in-place radix-2 IFFT algorithm as described in https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Data_reordering,_bit_reversal,_and_in-place_algorithms.
    /// If length is not a power of two, it uses the IDFT, which is a lot slower.
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
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum;
        
    fn real_fft(&self, y: &mut [Complex<T>])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;
        
    fn real_ifft(&mut self, x: &[Complex<T>])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;
        
    fn trim_zeros(&self) -> &[T]
    where
        T: Zero;
    fn trim_zeros_front(&self) -> &[T]
    where
        T: Zero;
    fn trim_zeros_back(&self) -> &[T]
    where
        T: Zero;
    fn trim_zeros_mut(&mut self) -> &mut [T]
    where
        T: Zero;
    fn trim_zeros_front_mut(&mut self) -> &mut [T]
    where
        T: Zero;
    fn trim_zeros_back_mut(&mut self) -> &mut [T]
    where
        T: Zero;
}

impl<T> SliceMath<T> for [T]
{
    fn recip_assign_all(&mut self)
    where
        T: Inv<Output = T>
    {
        let mut i = 0;
        while i < self.len()
        {
            unsafe {
                let ptr = self.as_mut_ptr().add(i);
                ptr.write(ptr.read().inv());
            }
            i += 1;
        }
    }
    fn conj_assign_all(&mut self)
    where
        T: ComplexFloat
    {
        let mut i = 0;
        while i < self.len()
        {
            unsafe {
                let ptr = self.as_mut_ptr().add(i);
                ptr.write(ptr.read().conj());
            }
            i += 1;
        }
    }

    fn convolve_direct<Rhs, C>(&self, rhs: &[Rhs]) -> C
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

    fn convolve_real_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
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
    
    fn convolve_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum + Mul<Rhs>,
        Rhs: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<Rhs::Real>> + Sum,
        <T as Mul<Rhs>>::Output: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>> + Sum,
        C: FromIterator<<T as Mul<Rhs>>::Output>
    {
        let y_len = (self.len() + rhs.len()).saturating_sub(1);
        let len = y_len.next_power_of_two();

        let mut x: Vec<T> = self.to_vec();
        let mut h: Vec<Rhs> = rhs.to_vec();
        x.resize(len, T::zero());
        h.resize(len, Rhs::zero());
        x.fft();
        h.fft();

        let mut y: Vec<_> = x.into_iter()
            .zip(h.into_iter())
            .map(|(x, h)| x*h)
            .collect();
        y.ifft();

        y.truncate(y_len);
        
        y.into_iter()
            .collect()
    }
    
    fn dtft(&self, omega: T::Real) -> T
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
    {
        let mut y = T::zero();
        let z1 = <T as From<_>>::from(Complex::cis(-omega));
        let mut z = T::one();
        for &x in self
        {
            y += x*z;
            z *= z1;
        }
        y
    }
        
    fn real_dtft(&self, omega: T) -> Complex<T>
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign
    {
        let mut y = Complex::zero();
        let z1 = Complex::cis(-omega);
        let mut z = Complex::one();
        for &x in self
        {
            y += <Complex<_> as From<_>>::from(x)*z;
            z *= z1;
        }
        y
    }

    fn fft_unscaled<const I: bool>(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<<T>::Real>> + Sum
    {
        if self.len() <= 1
        {
            return;
        }
        if !(
            fft::fft_radix2_unscaled::<_, I>(self)
            || fft::fft_radix3_unscaled::<_, I>(self)
            || fft::fft_radix_p_unscaled::<_, 5, I>(self)
            || fft::fft_radix_p_unscaled::<_, 7, I>(self)
            || fft::fft_radix_p_unscaled::<_, 11, I>(self)
            || fft::fft_radix_p_unscaled::<_, 13, I>(self)
            || fft::fft_radix_p_unscaled::<_, 17, I>(self)
            || fft::fft_radix_p_unscaled::<_, 19, I>(self)
            || fft::fft_radix_p_unscaled::<_, 23, I>(self)
            || fft::fft_radix_p_unscaled::<_, 29, I>(self)
            || fft::fft_radix_p_unscaled::<_, 31, I>(self)
            || fft::fft_radix_p_unscaled::<_, 37, I>(self)
            || fft::fft_radix_p_unscaled::<_, 41, I>(self)
            || fft::fft_radix_p_unscaled::<_, 43, I>(self)
            || fft::fft_radix_p_unscaled::<_, 47, I>(self)
            || fft::fft_radix_p_unscaled::<_, 53, I>(self)
            || fft::fft_radix_p_unscaled::<_, 59, I>(self)
            || fft::fft_radix_p_unscaled::<_, 61, I>(self)
            || fft::fft_radix_p_unscaled::<_, 67, I>(self)
            || fft::fft_radix_p_unscaled::<_, 71, I>(self)
            || fft::fft_radix_p_unscaled::<_, 73, I>(self)
            || fft::fft_radix_p_unscaled::<_, 79, I>(self)
            || fft::fft_radix_p_unscaled::<_, 83, I>(self)
            || fft::fft_radix_p_unscaled::<_, 89, I>(self)
            || fft::fft_radix_p_unscaled::<_, 97, I>(self)
            || fft::fft_radix_n_sqrt_unscaled::<_, I>(self)
        )
        {
            fft::dft_unscaled::<_, I>(self)
        }
    }

    fn fft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<<T>::Real>> + Sum
    {
        self.fft_unscaled::<false>()
    }

    fn ifft(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<<T>::Real>> + Sum
    {
        self.fft_unscaled::<true>();

        self.mul_assign_all(<T as From<_>>::from(<Complex<_> as From<_>>::from(<T::Real as NumCast>::from(1.0/self.len() as f64).unwrap())));
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
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign
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
    
    fn trim_zeros(&self) -> &[T]
    where
        T: Zero
    {
        self.trim(Zero::is_zero)
    }
    fn trim_zeros_front(&self) -> &[T]
    where
        T: Zero
    {
        self.trim_front(Zero::is_zero)
    }
    fn trim_zeros_back(&self) -> &[T]
    where
        T: Zero
    {
        self.trim_back(Zero::is_zero)
    }
    fn trim_zeros_mut(&mut self) -> &mut [T]
    where
        T: Zero
    {
        self.trim_mut(Zero::is_zero)
    }
    fn trim_zeros_front_mut(&mut self) -> &mut [T]
    where
        T: Zero
    {
        self.trim_front_mut(Zero::is_zero)
    }
    fn trim_zeros_back_mut(&mut self) -> &mut [T]
    where
        T: Zero
    {
        self.trim_back_mut(Zero::is_zero)
    }
}

#[test]
fn test()
{
    let mut x = [1.0, 1.0, 0.0, 0.0];
    let mut y = [Complex::zero(); 3];

    x.real_fft(&mut y);
    x.real_ifft(&y);

    println!("{:?}", x)
}