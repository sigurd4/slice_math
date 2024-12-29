use num::NumCast;
use slice_ops::Slice;

use crate::ops::poly::{SlicePolyEval, SlicePolyFit};

#[const_trait]
pub trait SliceDetrend<T>: Slice<Item = T>
{
    /// Subtracts the trend of a sequence using polynomial regression
    fn detrend(&mut self, n: usize);
}

#[cfg(feature = "ndarray")]
impl<T> SliceDetrend<T> for [T]
where
    T: ndarray_linalg::Lapack + NumCast
{
    fn detrend(&mut self, n: usize)
    {
        let x: Vec<_> = (0..self.len()).map(|i| T::from(i).unwrap())
            .collect();
        let p: Vec<_> = x.rpoly_fit(self, n);
        for i in 0..self.len()
        {
            self[i] -= p.rpoly_eval(x[i])
        }
    }
}

#[cfg(test)]
mod test
{
    use crate::ops::poly::SliceDetrend;

    #[test]
    fn it_works()
    {
        let mut x = [0.0, 1.0, 2.0, 3.0, -5.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        x.detrend(1);

        println!("{:?}", x)
    }
}