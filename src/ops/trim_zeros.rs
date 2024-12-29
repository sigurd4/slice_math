use num::Zero;
use slice_ops::{ops::SliceTrim, Slice};

#[const_trait]
pub trait SliceTrimZeros<T>: Slice<Item = T>
{
    /// Trims leading and trailing zero-valued elements from a slice.
    fn trim_zeros(&self) -> &[T]
    where
        T: Zero;
    /// Trims leading zero-valued elements from a slice.
    fn trim_zeros_front(&self) -> &[T]
    where
        T: Zero;
    /// Trims trailing zero-valued elements from a slice.
    fn trim_zeros_back(&self) -> &[T]
    where
        T: Zero;
    /// Trims leading and trailing zero-valued elements from a mutable slice.
    fn trim_zeros_mut(&mut self) -> &mut [T]
    where
        T: Zero;
    /// Trims leading zero-valued elements from a mutable slice.
    fn trim_zeros_front_mut(&mut self) -> &mut [T]
    where
        T: Zero;
    /// Trims trailing zero-valued elements from a mutable slice.
    fn trim_zeros_back_mut(&mut self) -> &mut [T]
    where
        T: Zero;
}

impl<T> SliceTrimZeros<T> for [T]
{
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