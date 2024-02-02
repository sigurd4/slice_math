use std::{f64::consts::TAU, ops::{AddAssign, MulAssign}};

use crate::SliceOps;
use num::{complex::ComplexFloat, Complex, Float, NumCast};

pub fn fft_radix2_unscaled<T, const I: bool>(slice: &mut [T]) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + From<Complex<T::Real>>
{
    let len = slice.len();
    if len.is_power_of_two()
    {
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