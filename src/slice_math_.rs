use std::{f64::consts::TAU, iter::Sum, ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Neg, SubAssign}};

use num::{complex::ComplexFloat, traits::Inv, Complex, Float, NumCast, One, Zero};
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
    fn polynomial_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: From<T> + AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>;
    #[cfg(feature = "ndarray")]
    fn rpolynomial_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: From<T> + AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>;
        
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
        if l < 1
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
        if l < 1
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
    fn polynomial_roots<S>(&self) -> S
    where
        Complex<<T as ComplexFloat>::Real>: From<T> + AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>
    {
        use ndarray_linalg::eig::EigVals;

        let c = self.companion_matrix();
        let mut roots = EigVals::eigvals(&c).unwrap();
        let len = roots.len();
        // Use newtons method
        let p: Vec<Complex<<T as ComplexFloat>::Real>> = self.into_iter()
            .map(|p| From::from(*p))
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
        Complex<<T as ComplexFloat>::Real>: From<T> + AddAssign + SubAssign + MulAssign + DivAssign + DivAssign<<T as ComplexFloat>::Real>,
        T: ComplexFloat + ndarray_linalg::Lapack<Complex = Complex<<T as ComplexFloat>::Real>>,
        S: FromIterator<Complex<<T as ComplexFloat>::Real>>
    {
        use ndarray_linalg::EigVals;

        let c = self.rcompanion_matrix();
        let mut roots = EigVals::eigvals(&c).unwrap();
        let len = roots.len();
        // Use newtons method
        let p: Vec<Complex<<T as ComplexFloat>::Real>> = self.into_iter()
            .map(|p| From::from(*p))
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

#[cfg(test)]
#[cfg(feature = "ndarray")]
#[test]
fn test()
{
    let p = [-1.0, 0.0, 1.0];

    let p = p.map(|b| Complex::new(b, 0.0));

    let r: Vec<_> = p.rpolynomial_roots();

    println!("x = {:?}", r);

    for r in r
    {
        println!("p = {:?}", p.rpolynomial(r));
    }
}