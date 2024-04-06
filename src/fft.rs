use std::{f64::consts::TAU, iter::Sum, ops::{AddAssign, MulAssign}};

use crate::{util, SliceMath, SliceOps};
use num::{complex::ComplexFloat, Complex, Float, NumCast, Zero};

pub fn fft_unscaled<T, const I: bool>(slice: &mut [T], mut temp: Option<&mut [T]>)
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum,
{
    let len = slice.len();
    if len <= 1
    {
        return;
    }
    if !(
        fft_radix2_unscaled::<_, I>(slice, &mut temp)
        || fft_radix3_unscaled::<_, I>(slice, &mut temp)
        || fft_radix5_unscaled::<_, I>(slice, &mut temp)
        || fft_radix7_unscaled::<_, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 11, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 13, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 17, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 19, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 23, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 29, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 31, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 37, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 41, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 43, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 47, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 53, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 59, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 61, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 67, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 71, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 73, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 79, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 83, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 89, I>(slice, &mut temp)
        || fft_radix_p_unscaled::<_, 97, I>(slice, &mut temp)
        || fft_radix_n_sqrt_unscaled::<_, I>(slice, &mut temp)
    )
    {
        dft_unscaled::<_, I>(slice, &mut temp)
    }
}

pub fn partial_fft_unscaled<T, const I: bool>(slice: &mut [T], temp: &mut [T], m: usize)
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    let mut i = 0;
    let ind: Vec<_> = (0..m).map(|k| {
        let j = i;
        for x in slice[k..].iter()
            .step_by(m)
        {
            temp[i] = *x;
            i += 1;
        }
        j..i
    }).collect();
    for ind in ind
    {
        fft_unscaled::<_, I>(&mut temp[ind.clone()], Some(&mut slice[ind]))
    }
}

pub fn fft_radix2_unscaled<T, const I: bool>(slice: &mut [T], temp: &mut Option<&mut [T]>) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    let len = slice.len();
    if len.is_power_of_two()
    {
        // In-place FFT

        slice.bit_rev_permutation();
        
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

        let mut tempvec;
        let temp = if let Some(temp) = temp.take()
        {
            temp
        }
        else
        {
            tempvec = Some(vec![T::zero(); len]);
            tempvec.as_mut().unwrap()
        };

        partial_fft_unscaled::<_, I>(slice, temp, 2);
        let x: Vec<_> = temp.chunks(len/2).collect();

        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
        let mut wn_pk = T::one();
        for k in 0..len/2
        {
            let p = x[0][k];
            let q = wn_pk*x[1][k];

            slice[k] = p + q;
            slice[k + len/2] = p - q;

            wn_pk *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix3_unscaled<T, const I: bool>(slice: &mut [T], temp: &mut Option<&mut [T]>) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    let len = slice.len();

    const P: usize = 3;

    if len % P == 0
    {
        // Recursive FFT

        let mut tempvec;
        let temp = if let Some(temp) = temp.take()
        {
            temp
        }
        else
        {
            tempvec = Some(vec![T::zero(); len]);
            tempvec.as_mut().unwrap()
        };

        partial_fft_unscaled::<_, I>(slice, temp, P);
        let x: Vec<_> = temp.chunks(len/P).collect();

        let w3 = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/P as f64).unwrap()));
        let w3_p2 = w3*w3;
        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
        let mut wn_pn = T::one();
        for k in 0..len/P
        {
            let p = x[0][k] + x[1][k] + x[2][k];
            let q = wn_pn*(x[0][k] + x[1][k]*w3 + x[2][k]*w3_p2);
            let r = wn_pn*wn_pn*(x[0][k] + x[1][k]*w3_p2 + x[2][k]*w3);

            slice[k] = p;
            slice[k + len/P] = q;
            slice[k + len/P*2] = r;
            wn_pn *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix5_unscaled<T, const I: bool>(slice: &mut [T], temp: &mut Option<&mut [T]>) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    let len = slice.len();

    const P: usize = 5;

    if len % P == 0
    {
        // Recursive FFT

        let mut tempvec;
        let temp = if let Some(temp) = temp.take()
        {
            temp
        }
        else
        {
            tempvec = Some(vec![T::zero(); len]);
            tempvec.as_mut().unwrap()
        };

        partial_fft_unscaled::<_, I>(slice, temp, P);
        let x: Vec<_> = temp.chunks(len/P).collect();

        let w5 = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/P as f64).unwrap()));
        let w5_p2 = w5*w5;
        let w5_p3 = w5_p2*w5;
        let w5_p4 = w5_p3*w5;
        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
        let mut wn_pk = T::one();
        for k in 0..len/P
        {
            let p = x[0][k] + x[1][k] + x[2][k] + x[3][k] + x[4][k];
            let q = wn_pk*(x[0][k] + x[1][k]*w5 + x[2][k]*w5_p2 + x[3][k]*w5_p3 + x[4][k]*w5_p4);
            let r = wn_pk*wn_pk*(x[0][k] + x[1][k]*w5_p2 + x[2][k]*w5_p4 + x[3][k]*w5 + x[4][k]*w5_p3);
            let s = wn_pk*wn_pk*wn_pk*(x[0][k] + x[1][k]*w5_p3 + x[2][k]*w5 + x[3][k]*w5_p4 + x[4][k]*w5_p2);
            let t = wn_pk*wn_pk*wn_pk*wn_pk*(x[0][k] + x[1][k]*w5_p4 + x[2][k]*w5_p3 + x[3][k]*w5_p2 + x[4][k]*w5);

            slice[k] = p;
            slice[k + len/P] = q;
            slice[k + len/P*2] = r;
            slice[k + len/P*3] = s;
            slice[k + len/P*4] = t;
            wn_pk *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix7_unscaled<T, const I: bool>(slice: &mut [T], temp: &mut Option<&mut [T]>) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum
{
    let len = slice.len();

    const P: usize = 7;

    if len % P == 0
    {
        // Recursive FFT

        let mut tempvec;
        let temp = if let Some(temp) = temp.take()
        {
            temp
        }
        else
        {
            tempvec = Some(vec![T::zero(); len]);
            tempvec.as_mut().unwrap()
        };

        partial_fft_unscaled::<_, I>(slice, temp, P);
        let x: Vec<_> = temp.chunks(len/P).collect();

        let w7 = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/P as f64).unwrap()));
        let w7_p2 = w7*w7;
        let w7_p3 = w7_p2*w7;
        let w7_p4 = w7_p3*w7;
        let w7_p5 = w7_p4*w7;
        let w7_p6 = w7_p5*w7;
        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
        let mut wn_pk = T::one();
        for k in 0..len/P
        {
            let p = x[0][k] + x[1][k] + x[2][k] + x[3][k] + x[4][k] + x[5][k] + x[6][k];
            let q = wn_pk*(x[0][k] + x[1][k]*w7 + x[2][k]*w7_p2 + x[3][k]*w7_p3 + x[4][k]*w7_p4 + x[5][k]*w7_p5 + x[6][k]*w7_p6);
            let r = wn_pk*wn_pk*(x[0][k] + x[1][k]*w7_p2 + x[2][k]*w7_p4 + x[3][k]*w7_p6 + x[4][k]*w7 + x[5][k]*w7_p3 + x[6][k]*w7_p5);
            let s = wn_pk*wn_pk*wn_pk*(x[0][k] + x[1][k]*w7_p3 + x[2][k]*w7_p6 + x[3][k]*w7_p2 + x[4][k]*w7_p5 + x[5][k]*w7 + x[6][k]*w7_p4);
            let t = wn_pk*wn_pk*wn_pk*wn_pk*(x[0][k] + x[1][k]*w7_p4 + x[2][k]*w7 + x[3][k]*w7_p5 + x[4][k]*w7_p2 + x[5][k]*w7_p6 + x[6][k]*w7_p3);
            let u = wn_pk*wn_pk*wn_pk*wn_pk*wn_pk*(x[0][k] + x[1][k]*w7_p5 + x[2][k]*w7_p3 + x[3][k]*w7 + x[4][k]*w7_p6 + x[5][k]*w7_p4 + x[6][k]*w7_p2);
            let v = wn_pk*wn_pk*wn_pk*wn_pk*wn_pk*wn_pk*(x[0][k] + x[1][k]*w7_p6 + x[2][k]*w7_p5 + x[3][k]*w7_p4 + x[4][k]*w7_p3 + x[5][k]*w7_p2 + x[6][k]*w7);

            slice[k] = p;
            slice[k + len/P] = q;
            slice[k + len/P*2] = r;
            slice[k + len/P*3] = s;
            slice[k + len/P*4] = t;
            slice[k + len/P*5] = u;
            slice[k + len/P*6] = v;
            wn_pk *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix_p_unscaled<T, const P: usize, const I: bool>(slice: &mut [T], temp: &mut Option<&mut [T]>) -> bool
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>> + Sum,
    [(); P - 1]:
{
    let len = slice.len();

    if len % P == 0
    {
        // Recursive FFT
        
        let mut tempvec;
        let temp = if let Some(temp) = temp.take()
        {
            temp
        }
        else
        {
            tempvec = Some(vec![T::zero(); len]);
            tempvec.as_mut().unwrap()
        };

        partial_fft_unscaled::<_, I>(slice, temp, P);
        let x: Vec<_> = temp.chunks(len/P).collect();

        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
        let mut wn_pk = T::one();
        let m = len/P;
        for k in 0..len
        {
            let mut e = T::one();
            slice[k] = x.iter()
                .map(|x| {
                    let x = x[k % m];
                    let y = x*e;
                    e *= wn_pk;
                    y
                }).sum::<T>();
            wn_pk *= wn;
        }
        return true;
    }
    false
}

pub fn fft_radix_n_sqrt_unscaled<T, const I: bool>(slice: &mut [T], temp: &mut Option<&mut [T]>) -> bool
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

        let mut tempvec;
        let temp = if let Some(temp) = temp.take()
        {
            temp
        }
        else
        {
            tempvec = Some(vec![T::zero(); len]);
            tempvec.as_mut().unwrap()
        };

        partial_fft_unscaled::<_, I>(slice, temp, p);
        let x: Vec<_> = temp.chunks(len/p).collect();

        let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
        let mut wn_pk = T::one();
        let m = len/p;
        for k in 0..len
        {
            let mut e = T::one();
            slice[k] = x.iter()
                .map(|x| {
                    let x = x[k % m];
                    let y = x*e;
                    e *= wn_pk;
                    y
                }).sum::<T>();
            wn_pk *= wn;
        }
        return true;
    }
    false
}

pub fn dft_unscaled<T, const I: bool>(slice: &mut [T], temp: &mut Option<&mut [T]>)
where
    T: ComplexFloat<Real: Float> + MulAssign + AddAssign + From<Complex<T::Real>>
{
    let len = slice.len();

    let mut tempvec;
    let temp = if let Some(temp) = temp.take()
    {
        temp
    }
    else
    {
        tempvec = Some(vec![T::zero(); len]);
        tempvec.as_mut().unwrap()
    };

    let wn = <T as From<_>>::from(Complex::cis(<T::Real as NumCast>::from(if I {TAU} else {-TAU}/len as f64).unwrap()));
    let mut wnk = T::one();

    unsafe {
        std::ptr::swap_nonoverlapping(temp.as_mut_ptr(), slice.as_mut_ptr(), len);
    }
    for k in 0..len
    {
        let mut wnki = T::one();
        slice[k] = Zero::zero();
        for i in 0..len
        {
            slice[k] += temp[i]*wnki;
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