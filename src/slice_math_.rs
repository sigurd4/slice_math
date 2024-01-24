use std::{f64::consts::TAU, ops::{AddAssign, Mul, MulAssign}};

use num::{complex::ComplexFloat, traits::{Inv, SaturatingSub}, Complex, Float, NumCast, Saturating, Zero};
use slice_ops::SliceOps;

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
    fn convolve_fft_cooley_tukey<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + ComplexFloat<Real: Float> + From<Complex<<Complex<T> as ComplexFloat>::Real>> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + ComplexFloat<Real: Float> + From<Complex<<Complex<Rhs> as ComplexFloat>::Real>>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: MulAssign + ComplexFloat<Real: Float> + From<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        C: FromIterator<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>;
    
    fn fft_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>;
    fn ifft_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>;
        
    fn fft_rec_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>;
    fn ifft_rec_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>;
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

    fn convolve_fft_cooley_tukey<Rhs, C>(&self, rhs: &[Rhs]) -> C
    where
        T: Float + Copy,
        Rhs: Float + Copy,
        Complex<T>: MulAssign + ComplexFloat<Real: Float> + From<Complex<<Complex<T> as ComplexFloat>::Real>> + Mul<Complex<Rhs>, Output: ComplexFloat<Real: Float>>,
        Complex<Rhs>: MulAssign + ComplexFloat<Real: Float> + From<Complex<<Complex<Rhs> as ComplexFloat>::Real>>,
        <Complex<T> as Mul<Complex<Rhs>>>::Output: MulAssign + ComplexFloat<Real: Float> + From<Complex<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>>,
        C: FromIterator<<<Complex<T> as Mul<Complex<Rhs>>>::Output as ComplexFloat>::Real>
    {
        let y_len = (self.len() + rhs.len()).saturating_sub(1);
        let len = y_len.next_power_of_two();

        let mut x: Vec<Complex<T>> = self.into_iter()
            .map(|&x| <Complex<T> as From<T>>::from(x))
            .collect();
        let mut h: Vec<Complex<Rhs>> = rhs.into_iter()
            .map(|&h| <Complex<Rhs> as From<Rhs>>::from(h))
            .collect();
        
        x.resize(len, Complex::zero());
        h.resize(len, Complex::zero());

        x.fft_cooley_tukey();
        h.fft_cooley_tukey();

        let mut y: Vec<_> = x.into_iter().zip(h.into_iter()).map(|(x, h)| x*h).collect();

        y.ifft_cooley_tukey();

        y.truncate(y_len);
        
        y.into_iter()
            .map(|y| y.re())
            .collect()
    }

    /// Performs an iterative, in-place radix-2 FFT algorithm as described in https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Data_reordering,_bit_reversal,_and_in-place_algorithms.
    /// Length must be a power of two.
    /// 
    /// # Examples
    /// ```rust
    /// use num::Complex;
    /// use slice_math::*;
    /// 
    /// let x = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ///     .map(|x| <Complex<_> as From<_>>::from(x));
    /// 
    /// let mut y = x;
    /// 
    /// y.as_mut_slice().fft_cooley_tukey();
    /// y.as_mut_slice().ifft_cooley_tukey();
    /// 
    /// let avg_error = x.into_iter()
    ///     .zip(y.into_iter())
    ///     .map(|(x, y)| (x - y).norm())
    ///     .sum::<f64>()/x.len() as f64;
    /// assert!(avg_error < 1.0e-16);
    /// ```
    fn fft_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<<T>::Real>>
    {
        let len = self.len();
        assert!(len.is_power_of_two(), "Length must be a power of two.");

        self.bit_reverse_permutation();
        
        for s in 0..len.ilog2()
        {
            let m = 2usize << s;
            let wm = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(-TAU/m as f64).unwrap()));
            for k in (0..len).step_by(m)
            {
                let mut w = T::one();
                for j in 0..m/2
                {
                    let t = w*self[k + j + m/2];
                    let u = self[k + j];
                    self[k + j] = u + t;
                    self[k + j + m/2] = u - t;
                    w *= wm;
                }
            }
        }
    }

    /// Performs an iterative, in-place radix-2 IFFT algorithm as described in https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Data_reordering,_bit_reversal,_and_in-place_algorithms.
    /// Length must be a power of two.
    /// 
    /// # Examples
    /// ```rust
    /// use num::Complex;
    /// use slice_math::*;
    /// 
    /// let x = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ///     .map(|x| <Complex<_> as From<_>>::from(x));
    /// 
    /// let mut y = x;
    /// 
    /// y.as_mut_slice().fft_cooley_tukey();
    /// y.as_mut_slice().ifft_cooley_tukey();
    /// 
    /// let avg_error = x.into_iter()
    ///     .zip(y.into_iter())
    ///     .map(|(x, y)| (x - y).norm())
    ///     .sum::<f64>()/x.len() as f64;
    /// assert!(avg_error < 1.0e-16);
    /// ```
    fn ifft_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<<T>::Real>>
    {
        let len = self.len();
        assert!(len.is_power_of_two(), "Length must be a power of two.");

        self.bit_reverse_permutation();
        
        for s in 0..len.ilog2()
        {
            let m = 2usize << s;
            let wm = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(TAU/m as f64).unwrap()));
            for k in (0..len).step_by(m)
            {
                let mut w = T::one();
                for j in 0..m/2
                {
                    let t = w*self[k + j + m/2];
                    let u = self[k + j];
                    self[k + j] = u + t;
                    self[k + j + m/2] = u - t;
                    w *= wm;
                }
            }
        }

        self.mul_assign_all(<T as From<_>>::from(<Complex<_> as From<_>>::from(<T::Real as NumCast>::from(1.0/len as f64).unwrap())));
    }
    
    /// Performs a recursive radix-2 FFT algorithm.
    /// Length must be a power of two.
    /// 
    /// # Examples
    /// ```rust
    /// use num::Complex;
    /// use slice_math::*;
    /// 
    /// let x = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ///     .map(|x| <Complex<_> as From<_>>::from(x));
    /// 
    /// let mut y = x;
    /// 
    /// y.as_mut_slice().fft_rec_cooley_tukey();
    /// y.as_mut_slice().ifft_rec_cooley_tukey();
    /// 
    /// assert_eq!(x, y);
    /// ```
    fn fft_rec_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<<T>::Real>>
    {
        let len = self.len();
        assert!(len.is_power_of_two(), "Length must be a power of two.");

        if len <= 1
        {
            return;
        }

        let [even, odd] = self.spread_ref();
        let mut even: Box<[T]> = even.into_iter()
            .map(|x| **x)
            .collect();
        let mut odd: Box<[T]> = odd.into_iter()
            .map(|x| **x)
            .collect();

        even.fft_rec_cooley_tukey();
        odd.fft_rec_cooley_tukey();

        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(-TAU/len as f64).unwrap()));
        let mut w = T::one();
        for k in 0..len/2
        {
            let p = even[k];
            let q = w*odd[k];
            self[k] = p + q;
            self[k + len/2] = p - q;
            w *= wn;
        }
    }

    /// Performs a recursive radix-2 IFFT algorithm.
    /// Length must be a power of two.
    /// 
    /// # Examples
    /// ```rust
    /// use num::Complex;
    /// use slice_math::*;
    /// 
    /// let x = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ///     .map(|x| <Complex<_> as From<_>>::from(x));
    /// 
    /// let mut y = x;
    /// 
    /// y.as_mut_slice().fft_rec_cooley_tukey();
    /// y.as_mut_slice().ifft_rec_cooley_tukey();
    /// 
    /// assert_eq!(x, y);
    /// ```
    fn ifft_rec_cooley_tukey(&mut self)
    where
        T: ComplexFloat<Real: Float> + MulAssign + From<Complex<<T>::Real>>
    {
        let len = self.len();
        assert!(len.is_power_of_two(), "Length must be a power of two.");

        self.conj_assign_all();

        self.fft_rec_cooley_tukey();

        self.conj_assign_all();
        
        self.mul_assign_all(<T as From<_>>::from(<Complex<_> as From<_>>::from(<T::Real as NumCast>::from(1.0/len as f64).unwrap())));
    }
}