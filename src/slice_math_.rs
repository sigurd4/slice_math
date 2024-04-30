use core::{any::Any, ops::{Add, Sub}};
use std::{iter::Sum, ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Neg, SubAssign}};

use num::{complex::ComplexFloat, traits::{FloatConst, Inv}, Complex, Float, NumCast, One, ToPrimitive, Zero};
use slice_ops::SliceOps;

use crate::fft;

const NEWTON_POLYNOMIAL_ROOTS: usize = 16;

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
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + MulAssign<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real> + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        C: FromIterator<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>;
        
    fn convolve_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat<Real: Into<<T as Mul<Rhs>>::Output>> + 'static> + Into<Complex<T::Real>>,
        Rhs: ComplexFloat + Into<Complex<Rhs::Real>>,
        Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>: Into<<Complex<T::Real> as Mul<Complex<Rhs::Real>>>::Output>,
        Complex<T::Real>: AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + MulAssign<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + Sum + 'static>,
        Complex<Rhs::Real>: AddAssign + MulAssign,
        C: FromIterator<<T as Mul<Rhs>>::Output>;
        
    fn cconvolve_direct<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Mul<Rhs, Output: AddAssign + Zero> + Copy,
        Rhs: Copy,
        C: FromIterator<<T as Mul<Rhs>>::Output>;
    fn cconvolve_real_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + AddAssign + ComplexFloat<Real = T> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + AddAssign + ComplexFloat<Real = Rhs>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: ComplexFloat<Real: Float> + Into<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>: MulAssign + AddAssign + MulAssign<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real> + ComplexFloat<Real = <<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>,
        C: FromIterator<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>;
    fn cconvolve_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: ComplexFloat + Mul<Rhs, Output: ComplexFloat<Real: Into<<T as Mul<Rhs>>::Output>> + 'static> + Into<Complex<T::Real>>,
        Rhs: ComplexFloat + Into<Complex<Rhs::Real>>,
        Complex<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real>: Into<<Complex<T::Real> as Mul<Complex<Rhs::Real>>>::Output>,
        Complex<T::Real>: AddAssign + MulAssign + Mul<Complex<Rhs::Real>, Output: ComplexFloat<Real = <<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + MulAssign + AddAssign + MulAssign<<<T as Mul<Rhs>>::Output as ComplexFloat>::Real> + Sum + 'static>,
        Complex<Rhs::Real>: AddAssign + MulAssign,
        C: FromIterator<<T as Mul<Rhs>>::Output>;

    fn deconvolve_direct<Rhs, Q, R>(&self, rhs: &[Rhs]) -> Option<(Q, R)>
    where
        T: Div<Rhs, Output: Zero + Mul<Rhs, Output: Zero + AddAssign + Copy> + Copy> + Sub<<<T as Div<Rhs>>::Output as Mul<Rhs>>::Output, Output = T> + Zero + Copy,
        Rhs: Copy + Zero,
        Q: FromIterator<<T as Div<Rhs>>::Output>,
        R: FromIterator<T>;
    fn deconvolve_fft<Q, R>(&self, rhs: &[T]) -> Option<(Q, R)>
    where
        T: ComplexFloat<Real: Into<T>> + SubAssign + AddAssign + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>,
        Q: FromIterator<T>,
        R: FromIterator<T>;
        
    fn dtft(&self, omega: T::Real) -> Complex<T::Real>
    where
        T: ComplexFloat + Into<Complex<T::Real>>,
        Complex<T::Real>: ComplexFloat<Real = T::Real> + MulAssign + AddAssign;
        
    fn fft_unscaled<const I: bool>(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum,
        Complex<T::Real>: Into<T>;
    
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
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + Sum,
        Complex<T::Real>: Into<T>;
        
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
        T: ComplexFloat<Real: Float> + MulAssign + AddAssign + MulAssign<T::Real> + Sum,
        Complex<T::Real>: Into<T>;
        
    /// Walsh-Hadamard transform
    fn fwht_unscaled(&mut self)
    where
        T: Add<Output = T> + Sub<Output = T> + Copy;
    /// Normalized Walsh-Hadamard transform
    fn fwht(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>;
    /// Normalized inverse Walsh-Hadamard transform
    fn ifwht(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>;

    fn fht(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    fn ifht(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>;
        
    fn dst_i(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>;
    fn dst_ii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    fn dst_iii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real> + Mul<T, Output = Complex<T::Real>>;
    fn dst_iv(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>;
        
    fn dct_i(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + DivAssign<T::Real> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + DivAssign<T::Real>;
    fn dct_ii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign;
    fn dct_iii(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>> + DivAssign<T::Real>;
    fn dct_iv(&mut self)
    where
        T: ComplexFloat<Real: Into<T>> + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + Mul<T, Output = Complex<T::Real>>;
        
    fn real_fft(&self, y: &mut [Complex<T>])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign;
        
    fn real_ifft(&mut self, x: &[Complex<T>])
    where
        T: Float,
        Complex<T>: ComplexFloat<Real = T> + MulAssign + AddAssign + MulAssign<T>;
        
    
    fn polynomial<Rhs>(&self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero + Copy,
        Rhs: Copy;
    fn rpolynomial<Rhs>(&self, rhs: Rhs) -> T
    where
        T: AddAssign + MulAssign<Rhs> + Zero + Copy,
        Rhs: Copy;
        
    fn derivate_polynomial<S>(&self) -> S
    where
        T: NumCast + Zero + Mul + Copy,
        S: FromIterator<<T as Mul>::Output>;
    fn derivate_rpolynomial<S>(&self) -> S
    where
        T: NumCast + Zero + Mul + Copy,
        S: FromIterator<<T as Mul>::Output>;
        
    fn integrate_polynomial<S>(&self, c: <T as Div>::Output) -> S
    where
        T: NumCast + Zero + Div + Copy,
        S: FromIterator<<T as Div>::Output>;
    fn integrate_rpolynomial<S>(&self, c: <T as Div>::Output) -> S
    where
        T: NumCast + Zero + Div + Copy,
        S: FromIterator<<T as Div>::Output>;

    #[cfg(feature = "ndarray")]
    fn companion_matrix(&self) -> ndarray::Array2<<T as Neg>::Output>
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>;
    #[cfg(feature = "ndarray")]
    fn rcompanion_matrix(&self) -> ndarray::Array2<<T as Neg>::Output>
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>;
    #[cfg(feature = "ndarray")]
    fn vandermonde_matrix(&self, n: usize) -> ndarray::Array2<T>
    where
        T: One + Copy + Mul;
    #[cfg(feature = "ndarray")]
    fn polynomial_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>> + Into<Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>;
    #[cfg(feature = "ndarray")]
    fn rpolynomial_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>> + Into<Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>;
        
    #[cfg(feature = "ndarray")]
    fn lin_fit(&self, y: &[T]) -> [T; 2]
    where
        T: ndarray_linalg::Lapack;
    #[cfg(feature = "ndarray")]
    fn rlin_fit(&self, y: &[T]) -> [T; 2]
    where
        T: ndarray_linalg::Lapack;
        
    #[cfg(feature = "ndarray")]
    fn polyfit<S>(&self, y: &[T], n: usize) -> S
    where
        T: ndarray_linalg::Lapack,
        S: FromIterator<T>;
    #[cfg(feature = "ndarray")]
    fn rpolyfit<S>(&self, y: &[T], n: usize) -> S
    where
        T: ndarray_linalg::Lapack,
        S: FromIterator<T>;
        
    #[cfg(feature = "ndarray")]
    fn detrend(&mut self, n: usize)
    where
        T: ndarray_linalg::Lapack;
        
    #[cfg(feature = "ndarray")]
    fn toeplitz_matrix(&self) -> ndarray::Array2<T>
    where
        T: Copy;
    #[cfg(feature = "ndarray")]
    fn hankel_matrix(&self, r: &[T]) -> ndarray::Array2<T>
    where
        T: Copy;
        
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
        
    fn frac_rotate_right(&mut self, shift: T::Real)
    where
        T: ComplexFloat<Real: Into<T> + SubAssign> + Into<Complex<<T>::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>;
    fn frac_rotate_left(&mut self, shift: T::Real)
    where
        T: ComplexFloat<Real: Into<T> + SubAssign> + Into<Complex<<T>::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>;
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
    
    fn convolve_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
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
    fn cconvolve_direct<Rhs, C>(&self, mut rhs: &[Rhs]) -> C
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
    fn cconvolve_real_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
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
    fn cconvolve_fft<Rhs, C>(&self, rhs: &[Rhs]) -> C
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
    
    fn deconvolve_direct<Rhs, Q, R>(&self, rhs: &[Rhs]) -> Option<(Q, R)>
    where
        T: Div<Rhs, Output: Zero + Mul<Rhs, Output: Zero + AddAssign + Copy> + Copy> + Sub<<<T as Div<Rhs>>::Output as Mul<Rhs>>::Output, Output = T> + Zero + Copy,
        Rhs: Copy + Zero,
        Q: FromIterator<<T as Div<Rhs>>::Output>,
        R: FromIterator<T>
    {
        let mut lag = rhs.len();
        let rhs = rhs.trim_zeros_front();
        lag -= rhs.len();
        if rhs.len() == 0
        {
            return None
        }

        let mut q = vec![];
        let mut r = self.to_vec();
        let d = rhs.len() - 1;
        let c = *rhs.first().unwrap();
        loop
        {
            let nr = r.len();
            if nr <= d
            {
                q = q.split_off(lag);
                r = core::iter::repeat(T::zero())
                    .take(self.len() - r.len())
                    .chain(r)
                    .collect();
                return Some((q.into_iter().collect(), r.into_iter().collect()))
            }
            let n = nr - d;
            let mut s = vec![<T as Div<Rhs>>::Output::zero(); n];
            s[0] = *r.first().unwrap()/c;

            let sv: Vec<_> = s.convolve_direct(&rhs);

            q.push(s[0]);
            r = sv.into_iter()
                .zip(r)
                .map(|(s, r)| r - s)
                .skip(1)
                .collect();
        }
    }
    fn deconvolve_fft<Q, R>(&self, rhs: &[T]) -> Option<(Q, R)>
    where
        T: ComplexFloat<Real: Into<T>> + SubAssign + AddAssign + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>,
        Q: FromIterator<T>,
        R: FromIterator<T>
    {
        let nb = self.len();
        let na = rhs.len();
        let nw = nb.max(na);
        let n = (nb + 1).saturating_sub(na);

        let mut q = vec![T::zero(); n];
        if let Some(q0) = q.first_mut()
        {
            *q0 = One::one();
        }

        {
            let mut w = vec![T::zero(); nw - 1];
            let a = rhs.trim_zeros_front();
            let b = self.trim_zeros_front();
            if a.len() == 0
            {
                return None
            }
            let a0 = a[0];

            for q in q.iter_mut()
            {
                let mut w0 = *q;
                for (&w, &a) in w.iter()
                    .zip(a.iter()
                        .skip(1)
                    )
                {
                    w0 -= w*(a/a0)
                }
                *q = w0*b[0];
                for (&w, &b) in w.iter()
                    .zip(b.iter()
                        .skip(1)
                    )
                {
                    *q += w*b
                }

                w.shift_right(&mut w0);
            }
        }

        let qa: Vec<_> = q.convolve_fft(rhs);

        let nqa = qa.len();
        let r = self.iter()
            .copied()
            .chain(core::iter::repeat(T::zero())
                .take(nb.saturating_sub(nqa))
            ).zip(qa.into_iter()
                .chain(core::iter::repeat(T::zero())
                    .take(nqa.saturating_sub(nb))
                )
            ).map(|(b, qa)| b - qa)
            .collect();

        Some((q.into_iter().collect(), r))
    }
    
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
    
    fn fwht_unscaled(&mut self)
    where
        T: Add<Output = T> + Sub<Output = T> + Copy
    {
        let len = self.len();
        if len <= 2
        {
            return;
        }
        assert!(len.is_power_of_two(), "Length must be a power of two.");

        let mut h = 1;
        while h < len
        {
            for i in (0..len).step_by(h*2)
            {
                for j in i..i + h
                {
                    let x = self[j];
                    let y = self[j + h];
                    self[j] = x + y;
                    self[j + h] = x - y;
                }
            }
            h *= 2;
        }
    }
    fn fwht(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>
    {
        self.fwht_unscaled();
        self.mul_assign_all(Float::powi(T::Real::FRAC_1_SQRT_2(), (self.len().ilog2() + 3).try_into().unwrap()))
    }
    fn ifwht(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>
    {
        self.fwht_unscaled();
        self.mul_assign_all(Float::powi(T::Real::FRAC_1_SQRT_2(), TryInto::<i32>::try_into(self.len().ilog2()).unwrap() - 3))
    }

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
                Complex::from_polar(T::Real::FRAC_1_SQRT_2(), -i*T::Real::FRAC_PI_2()/lenf)
            }).chain(core::iter::once(Complex::new(zero, -one)));
    
        let m2 = (1..len).map(|i| {
            let i = <T::Real as NumCast>::from(i).unwrap();
            Complex::from_polar(-T::Real::FRAC_1_SQRT_2(), i*T::Real::FRAC_PI_2()/lenf)
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
            *y /= T::Real::SQRT_2()
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
    
    fn polynomial<Rhs>(&self, rhs: Rhs) -> T
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
    fn rpolynomial<Rhs>(&self, rhs: Rhs) -> T
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
    
    fn derivate_polynomial<S>(&self) -> S
    where
        T: NumCast + Zero + Mul + Copy,
        S: FromIterator<<T as Mul>::Output>
    {
        let s = self.trim_zeros_back();
        if s.len() < 2
        {
            return core::iter::empty()
                .collect()
        }
        s[1..].into_iter()
            .enumerate()
            .map(|(i, b)| *b*T::from(i + 1).unwrap())
            .collect()
    }
    fn derivate_rpolynomial<S>(&self) -> S
    where
        T: NumCast + Zero + Mul + Copy,
        S: FromIterator<<T as Mul>::Output>
    {
        let s = self.trim_zeros_front();
        let n = s.len();
        if n < 2
        {
            return core::iter::empty()
                .collect()
        }
        let nm1 = n - 1;
        s[..nm1].into_iter()
            .enumerate()
            .map(|(i, b)| *b*T::from(nm1 - i).unwrap())
            .collect()
    }
        
    fn integrate_polynomial<S>(&self, c: <T as Div>::Output) -> S
    where
        T: NumCast + Zero + Div + Copy,
        S: FromIterator<<T as Div>::Output>
    {
        core::iter::once(c)
            .chain(self.trim_zeros_back()
                .into_iter()
                .enumerate()
                .map(|(i, b)| *b/T::from(i + 1).unwrap())
            ).collect()
    }
    fn integrate_rpolynomial<S>(&self, c: <T as Div>::Output) -> S
    where
        T: NumCast + Zero + Div + Copy,
        S: FromIterator<<T as Div>::Output>
    {
        let s = self.trim_zeros_front();
        let n = s.len();
        s.into_iter()
            .enumerate()
            .map(|(i, b)| *b/T::from(n - i).unwrap())
            .chain(core::iter::once(c))
            .collect()
    }

    #[cfg(feature = "ndarray")]
    fn companion_matrix(&self) -> ndarray::Array2<<T as Neg>::Output>
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>
    {
        let s = self.trim_zeros_back();
        let l = s.len();
        if l <= 1
        {
            return ndarray::Array2::from_shape_fn((0, 0), |_| Zero::zero())
        }
        let n = l - 1;
        let mut c = ndarray::Array2::from_shape_fn((n, n), |_| Zero::zero());
        let mut i = 0;
        while i < n
        {
            if i > 0
            {
                c[(i, i - 1)] = One::one();
            }
            c[(i, n - 1)] = -s[i];
            c[(i, n - 1)] /= s[n];
            i += 1;
        }
        c
    }
    #[cfg(feature = "ndarray")]
    fn rcompanion_matrix(&self) -> ndarray::Array2<<T as Neg>::Output>
    where
        T: Copy + Neg + Zero,
        <T as Neg>::Output: One + Zero + DivAssign<T>
    {
        let s = self.trim_zeros_front();
        let l = s.len();
        if l <= 1
        {
            return ndarray::Array2::from_shape_fn((0, 0), |_| Zero::zero())
        }
        let n = l - 1;
        let mut c = ndarray::Array2::from_shape_fn((n, n), |_| Zero::zero());
        let mut i = n;
        loop
        {
            c[(n - i, n - 1)] = -s[i];
            c[(n - i, n - 1)] /= s[0];
            i -= 1;
            if i > 0
            {
                c[(i, i - 1)] = One::one();
            }
            else
            {
                break
            }
        }
        c
    }
    #[cfg(feature = "ndarray")]
    fn vandermonde_matrix(&self, n: usize) -> ndarray::Array2<T>
    where
        T: One + Copy + Mul
    {
        let l = self.len();
        let mut m = ndarray::Array2::from_elem((l, n), T::one());
        for j in (0..n - 1).rev()
        {
            for k in 0..l
            {
                m[(k, j)] = self[k]*m[(k, j + 1)]
            }
        }
        m
    }
    #[cfg(feature = "ndarray")]
    fn polynomial_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>> + Into<Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>
    {
        use ndarray_linalg::eig::EigVals;

        if self.trim_zeros_front().len() <= 1
        {
            return core::iter::empty()
                .collect()
        }

        let c = self.companion_matrix();
        let mut roots = EigVals::eigvals(&c).unwrap();
        let len = roots.len();
        // Use newtons method
        let p: Vec<Complex<<T as ComplexFloat>::Real>> = self.into_iter()
            .map(|p| (*p).into())
            .collect();
        let dp: Vec<_> = p.derivate_polynomial();
        for k in 0..len
        {
            const NEWTON: usize = NEWTON_POLYNOMIAL_ROOTS;

            for _ in 0..NEWTON
            {
                let root = roots[k];
                let df = p.polynomial(root);
                if df.is_zero()
                {
                    break
                }
                roots[k] -= df/dp.polynomial(root)
            }
        }
        roots.into_iter()
            .collect()
    }
    #[cfg(feature = "ndarray")]
    fn rpolynomial_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>> + Into<Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>
    {
        use ndarray_linalg::EigVals;

        if self.trim_zeros_front().len() <= 1
        {
            return core::iter::empty()
                .collect()
        }

        let c = self.rcompanion_matrix();
        let mut roots = EigVals::eigvals(&c).unwrap();
        let len = roots.len();
        // Use newtons method
        let p: Vec<Complex<<T as ComplexFloat>::Real>> = self.into_iter()
            .map(|p| (*p).into())
            .collect();
        let dp: Vec<_> = p.derivate_rpolynomial();
        for k in 0..len
        {
            const NEWTON: usize = NEWTON_POLYNOMIAL_ROOTS;

            for _ in 0..NEWTON
            {
                let root = roots[k];
                let df = p.rpolynomial(root);
                if df.is_zero()
                {
                    break
                }
                roots[k] -= df/dp.rpolynomial(root)
            }
        }
        roots.into_iter()
            .collect()
    }

    #[cfg(feature = "ndarray")]
    fn lin_fit(&self, y: &[T]) -> [T; 2]
    where
        T: ndarray_linalg::Lapack
    {
        self.polyfit::<Vec<_>>(y, 1)
            .try_into()
            .unwrap()
    }
    
    #[cfg(feature = "ndarray")]
    fn rlin_fit(&self, y: &[T]) -> [T; 2]
    where
        T: ndarray_linalg::Lapack
    {
        self.rpolyfit::<Vec<_>>(y, 1)
            .try_into()
            .unwrap()
    }
    
    #[cfg(feature = "ndarray")]
    fn polyfit<S>(&self, y: &[T], n: usize) -> S
    where
        T: ndarray_linalg::Lapack,
        S: FromIterator<T>
    {
        let mut p: Vec<_> = self.rpolyfit(y, n);
        p.reverse();
        p.into_iter()
            .collect()
    }
    #[cfg(feature = "ndarray")]
    fn rpolyfit<S>(&self, y: &[T], n: usize) -> S
    where
        T: ndarray_linalg::Lapack,
        S: FromIterator<T>
    {
        use ndarray::ArrayView2;
        use ndarray_linalg::{Solve, QR};

        let v = self.vandermonde_matrix(n + 1);

        let (q, r) = v.qr()
            .unwrap();
        let qtmy = q.t()
            .dot(&ArrayView2::from_shape((y.len(), 1), y).unwrap());
        let p = r.solve(&qtmy.column(0))
            .unwrap();

        p.into_iter()
            .collect()
    }

    #[cfg(feature = "ndarray")]
    fn detrend(&mut self, n: usize)
    where
        T: ndarray_linalg::Lapack + NumCast
    {
        let x: Vec<_> = (0..self.len()).map(|i| T::from(i).unwrap())
            .collect();
        let p: Vec<_> = x.rpolyfit(self, n);
        for i in 0..self.len()
        {
            self[i] -= p.rpolynomial(x[i])
        }
    }
    
    #[cfg(feature = "ndarray")]
    fn toeplitz_matrix(&self) -> ndarray::Array2<T>
    where
        T: Copy
    {
        use ndarray::Array2;

        let n = self.len();
        Array2::from_shape_fn((n, n), |(i, j)| self[if i >= j {i - j} else {j - i}])
    }
    #[cfg(feature = "ndarray")]
    fn hankel_matrix(&self, r: &[T]) -> ndarray::Array2<T>
    where
        T: Copy
    {
        use ndarray::Array2;

        let n = self.len();
        let m = r.len();
        Array2::from_shape_fn((n, m), |(i, j)| if i + j < n
        {
            self[i + j]
        }
        else
        {
            r[i + j + 1 - n]
        })
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
#[test]
fn test()
{
    let mut p = [1.0, 0.0, 0.0, 0.0, 0.0];
    
    p.frac_rotate_right(0.5);
    p.frac_rotate_right(0.5);

    println!("{:?}", p);
}