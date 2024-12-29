use num::complex::ComplexFloat;
use slice_ops::Slice;

use super::SliceVisit;

#[const_trait]
pub trait SliceConjAssign<T>: Slice<Item = T>
{
    /// Finds the reciprocal of each element in the slice.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use slice_ops::ops::*;
    /// 
    /// let mut x = [1.0, 2.0, 3.0, 4.0];
    /// 
    /// x.recip_assign_all();
    ///    
    /// assert_eq!(x, [1.0, 0.5, 1.0/3.0, 0.25]);
    /// ```
    fn conj_assign_all(&mut self)
    where
        T: ComplexFloat;
             
    /// Asynchronously finds the reciprocal of each element in the slice.
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use slice_ops::ops::*;
    /// 
    /// # tokio_test::block_on(async {
    /// let mut x = [1.0, 2.0, 3.0, 4.0];
    /// 
    /// x.recip_assign_all_async().await;
    ///    
    /// assert_eq!(x, [1.0, 0.5, 1.0/3.0, 0.25]);
    /// # });
    /// ```
    async fn conj_assign_all_async(&mut self)
    where
        T: ComplexFloat;
}

impl<T> SliceConjAssign<T> for [T]
{
    fn conj_assign_all(&mut self)
    where
        T: ComplexFloat
    {
        self.visit_mut(|x| unsafe {
            core::ptr::write(x, core::ptr::read(x).conj())
        })
    }

    async fn conj_assign_all_async(&mut self)
    where
        T: ComplexFloat
    {
        self.visit_mut_async(async |x| unsafe {
            core::ptr::write(x, core::ptr::read(x).conj())
        }).await
    }
}

#[cfg(test)]
mod test
{
    #[test]
    fn it_works()
    {
        
    }
}