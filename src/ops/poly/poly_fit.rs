use slice_ops::Slice;

use crate::ops::SliceIntoMatrix;

#[const_trait]
pub trait SlicePolyFit<T>: Slice<Item = T>
{
    /// Linear regression with powers in rising order from left to right.
    fn lin_fit(&self, y: &[T]) -> [T; 2];
    /// Linear regression with powers in rising order from right to left.
    fn rlin_fit(&self, y: &[T]) -> [T; 2];
        
    /// Polynomial regression with powers in rising order from left to right.
    fn poly_fit<S>(&self, y: &[T], n: usize) -> S
    where
        S: FromIterator<T>;
    /// Polynomial regression with powers in rising order from right to left.
    fn rpoly_fit<S>(&self, y: &[T], n: usize) -> S
    where
        S: FromIterator<T>;
}

impl<T> SlicePolyFit<T> for [T]
where
    T: ndarray_linalg::Lapack
{
    fn lin_fit(&self, y: &[T]) -> [T; 2]
    {
        self.poly_fit::<Vec<_>>(y, 1)
            .try_into()
            .unwrap()
    }
    
    fn rlin_fit(&self, y: &[T]) -> [T; 2]
    where
        T: ndarray_linalg::Lapack
    {
        self.rpoly_fit::<Vec<_>>(y, 1)
            .try_into()
            .unwrap()
    }
    
    fn poly_fit<S>(&self, y: &[T], n: usize) -> S
    where
        T: ndarray_linalg::Lapack,
        S: FromIterator<T>
    {
        let mut p: Vec<_> = self.rpoly_fit(y, n);
        p.reverse();
        p.into_iter()
            .collect()
    }
    fn rpoly_fit<S>(&self, y: &[T], n: usize) -> S
    where
        T: ndarray_linalg::Lapack,
        S: FromIterator<T>
    {
        use ndarray::ArrayView2;
        use ndarray_linalg::{Solve, QR};

        let v = self.vandermonde_matrix(n + 1);

        let (q, r) = v.qr()
            .unwrap();
        let qtmy = q.t()
            .dot(&ArrayView2::from_shape((y.len(), 1), y).unwrap());
        let p = r.solve(&qtmy.column(0))
            .unwrap();

        p.into_iter()
            .collect()
    }
}

#[cfg(test)]
mod test
{
    use crate::ops::poly::{SlicePolyEval, SlicePolyFit};

    #[test]
    fn test_polyfit()
    {
        let p = [2.0, -4.0, 1.0];

        let x = [4.0, -1.0, 6.0, 7.0];
        let y = x.map(|x| p.rpoly_eval(x));

        let p: Vec<_> = x.rpoly_fit(&y, 2);

        println!("{:?}", p);
    }
}