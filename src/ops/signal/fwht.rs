use std::ops::{Add, MulAssign, Sub};

use num::{complex::ComplexFloat, traits::FloatConst, Float};
use slice_ops::{ops::SliceMulAssign, Slice};

#[const_trait]
pub trait SliceFwhd<T>: Slice<Item = T>
{
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
}

impl<T> SliceFwhd<T> for [T]
{
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
        self.mul_assign_all(Float::powi(FloatConst::FRAC_1_SQRT_2(), (self.len().ilog2() + 3).try_into().unwrap()))
    }
    fn ifwht(&mut self)
    where
        T: ComplexFloat + MulAssign<T::Real>
    {
        self.fwht_unscaled();
        self.mul_assign_all(Float::powi(FloatConst::FRAC_1_SQRT_2(), TryInto::<i32>::try_into(self.len().ilog2()).unwrap() - 3))
    }
}

#[cfg(test)]
mod test
{
    use crate::ops::signal::SliceFwhd;

    #[test]
    fn wht()
    {
        let mut a = [19, -1, 11, -9, -7, 13, -15, 5].map(|a| a as f64);
        a.fwht();
        //a.fwht();
        println!("{:?}", a)
    }
}