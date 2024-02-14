use std::{f64::consts::TAU, iter::Sum, ops::{AddAssign, MulAssign}};

use crate::{util, SliceMath, SliceOps};
use num::{complex::ComplexFloat, Complex, Float, NumCast, Zero};

pub fn partial_fft_unscaled<T, const I: bool, const M: usize>(slice: &[T]) -> [Vec<T>; M]
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum,
    [(); M - 1]:
{
    let spread = slice.spread_ref();

    spread.map(|spread| {
        let mut spread: Vec<_> = spread.into_iter()
            .map(|x| **x)
            .collect();
        spread.fft_unscaled::<I>();
        spread
    })
}

pub fn partial_fft_unscaled_vec<T, const I: bool>(array: &[T], m: usize) -> Vec<Vec<T>>
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    (0..m).map(|k| {
        let mut spread: Vec<_> = array[k..].into_iter()
            .step_by(m)
            .map(|&x| x)
            .collect();
        spread.fft_unscaled::<I>();
        spread
    }).collect()
}

pub fn fft_radix2_unscaled<T, const I: bool>(slice: &mut [T]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    let len = slice.len();
    if len.is_power_of_two()
    {
        // In-place FFT

        slice.bit_reverse_permutation();
        
        for s in 0..len.ilog2()
        {
            let m = 2usize << s;
            let wm = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/m as f64).unwrap()));
            for k in (0..len).step_by(m)
            {
                let mut w = T::one();
                for j in 0..m/2
                {
                    let t = w*slice[k + j + m/2];
                    let u = slice[k + j];
                    slice[k + j] = u + t;
                    slice[k + j + m/2] = u - t;
                    w *= wm;
                }
            }
        }
        return true
    }
    if len % 2 == 0
    {
        // Recursive FFT

        let [even, odd] = partial_fft_unscaled::<_, I, _>(slice);

        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
        let mut wn_pk = T::one();
        for k in 0..len/2
        {
            let p = even[k];
            let q = wn_pk*odd[k];

            slice[k] = p + q;
            slice[k + len/2] = p - q;

            wn_pk *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix3_unscaled<T, const I: bool>(slice: &mut [T]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    let len = slice.len();

    const P: usize = 3;

    if len % P == 0
    {
        // Recursive FFT

        let [x1, x2, x3] = partial_fft_unscaled::<_, I, _>(slice);

        let w3 = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/P as f64).unwrap()));
        let w3_p2 = w3*w3;
        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
        let mut wn_pn = T::one();
        for k in 0..len/P
        {
            let p = x1[k] + x2[k] + x3[k];
            let q = wn_pn*(x1[k] + x2[k]*w3 + x3[k]*w3_p2);
            let r = wn_pn*wn_pn*(x1[k] + x2[k]*w3_p2 + x3[k]*w3);

            slice[k] = p;
            slice[k + len/P] = q;
            slice[k + len/P*2] = r;
            wn_pn *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix_p_unscaled<T, const P: usize, const I: bool>(slice: &mut [T]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum,
    [(); P - 1]:
{
    let len = slice.len();

    if len % P == 0
    {
        // Recursive FFT

        let x: [_; P] = partial_fft_unscaled::<_, I, _>(slice);

        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
        let mut wn_pk = T::one();
        let m = len/P;
        for k in 0..len
        {
            let mut e = T::one();
            slice[k] = x.iter()
                .map(|x| {
                    let x = x[k % m];
                    let y = x*wn_pk;
                    e *= wn_pk;
                    y
                }).sum::<T>();
            wn_pk *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix_n_sqrt_unscaled<T, const I: bool>(slice: &mut [T]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    let len = slice.len();
    let p = {
        util::closest_prime(1 << ((len.ilog2() + 1) / 2))
    };
    if let Some(p) = p && len % p == 0
    {
        // Recursive FFT

        let x = partial_fft_unscaled_vec::<_, I>(slice, p);

        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
        let mut wn_pk = T::one();
        let m = len/p;
        for k in 0..len
        {
            let mut e = T::one();
            slice[k] = x.iter()
                .map(|x| {
                    let x = x[k % m];
                    let y = x*wn_pk;
                    e *= wn_pk;
                    y
                }).sum::<T>();
            wn_pk *= wn;
        }
        return true;
    }
    false
}

pub fn dft_unscaled<T, const I: bool>(slice: &mut [T])
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>>
{
    let len = slice.len();
    let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
    let mut wnk = T::one();

    let mut buf = vec![T::zero(); len];
    unsafe {
        std::ptr::swap_nonoverlapping(buf.as_mut_ptr(), slice.as_mut_ptr(), len);
    }
    for k in 0..len
    {
        let mut wnki = T::one();
        for i in 0..len
        {
            slice[k] += buf[i]*wnki;
            wnki *= wnk;
        }

        wnk *= wn;
    }
}

#[test]
fn test()
{
    use crate::SliceMath;

    let mut x = [1.0, 2.0, 3.0];
    let mut y = [Complex::zero(); 2];

    x.real_fft(&mut y);
    x.real_ifft(&y);

    println!("{:?}", x)
}