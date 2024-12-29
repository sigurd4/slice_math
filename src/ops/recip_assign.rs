use num::traits::Inv;
use slice_ops::Slice;

use super::SliceVisit;

#[const_trait]
pub trait SliceRecipAssign<T>: Slice<Item = T>
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
    fn recip_assign_all(&mut self)
    where
        T: Inv<Output = T>;
             
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
    async fn recip_assign_all_async(&mut self)
    where
        T: Inv<Output = T>;
}

impl<T> SliceRecipAssign<T> for [T]
{
    fn recip_assign_all(&mut self)
    where
        T: Inv<Output = T>
    {
        self.visit_mut(|x| unsafe {
            core::ptr::write(x, core::ptr::read(x).inv())
        })
    }

    async fn recip_assign_all_async(&mut self)
    where
        T: Inv<Output = T>
    {
        self.visit_mut_async(async |x| unsafe {
            core::ptr::write(x, core::ptr::read(x).inv())
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