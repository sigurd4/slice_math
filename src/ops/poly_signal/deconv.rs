use std::{iter::Sum, ops::{AddAssign, Div, Mul, MulAssign, Sub, SubAssign}};

use num::{complex::ComplexFloat, Complex, Float, One, Zero};
use slice_ops::{ops::SliceShift, Slice};

use crate::ops::SliceTrimZeros;

use super::SliceConv;

/// De
#[const_trait]
pub trait SliceDeconv<T>: Slice<Item = T>
{
    /// Performs deconvolution (polynomial division) naively.
    #[doc(alias = "poly_div")]
    fn deconv_direct<Rhs, Q, R>(&self, rhs: &[Rhs]) -> Option<(Q, R)>
    where
        T: Div<Rhs, Output: Zero + Mul<Rhs, Output: Zero + AddAssign + Copy> + Copy> + Sub<<<T as Div<Rhs>>::Output as Mul<Rhs>>::Output, Output = T> + Zero + Copy,
        Rhs: Copy + Zero,
        Q: FromIterator<<T as Div<Rhs>>::Output>,
        R: FromIterator<T>;
    /// Performs deconvolution using FFT.
    #[doc(alias = "poly_div_fft")]
    fn deconv_fft<Q, R>(&self, rhs: &[T]) -> Option<(Q, R)>
    where
        T: ComplexFloat<Real: Into<T>> + SubAssign + AddAssign + Into<Complex<T::Real>> + 'static,
        Complex<T::Real>: AddAssign + MulAssign + MulAssign<T::Real>,
        Q: FromIterator<T>,
        R: FromIterator<T>;
}

impl<T> SliceDeconv<T> for [T]
{
    fn deconv_direct<Rhs, Q, R>(&self, rhs: &[Rhs]) -> Option<(Q, R)>
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

            let sv: Vec<_> = s.conv_direct(&rhs);

            q.push(s[0]);
            r = sv.into_iter()
                .zip(r)
                .map(|(s, r)| r - s)
                .skip(1)
                .collect();
        }
    }
    fn deconv_fft<Q, R>(&self, rhs: &[T]) -> Option<(Q, R)>
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

        let qa: Vec<_> = q.conv_fft(rhs);

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
}

#[cfg(test)]
mod test
{
    use crate::ops::poly::{SliceConv, SliceDeconv};

    #[test]
    fn it_works()
    {
        let x = [1.0, 2.0, 3.0];
        let h = [1.0, 1.0, 3.0, 4.0];
        let y: Vec<_> = x.conv_direct(&h);
        let (x1, r1): (Vec<_>, Vec<_>) = y.deconv_direct(&h).unwrap();
        let (x2, r2): (Vec<_>, Vec<_>) = y.deconv_fft(&h).unwrap();
        println!("{:?}, {:?}", x1, r1);
        println!("{:?}, {:?}", x2, r2);
    }
}