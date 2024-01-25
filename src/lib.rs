#![feature(associated_type_bounds)]
#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]

moddef::moddef!(
    flat(pub) mod {
        slice_math_
    }
);

pub use slice_ops::*;

#[cfg(test)]
mod tests {
    use std::{ops::RangeBounds, time::{Duration, SystemTime}};

    //use array__ops::{Array2dOps, ArrayNd, ArrayOps};
    //use linspace::{Linspace, LinspaceArray};
    use num::Complex;
    //use rustfft::{Fft, FftPlanner};

    use super::*;

    const PLOT_TARGET: &str = "plots";

    pub fn benchmark<T, R>(x: &[T], f: &dyn Fn(T) -> R) -> Duration
    where
        T: Clone
    {
        use std::time::SystemTime;

        let x = x.to_vec();
        let t0 = SystemTime::now();
        x.into_iter().for_each(|x| {f(x);});
        t0.elapsed().unwrap()
    }

    /*#[test]
    fn bench()
    {
        let fn_name = "FFT";

        const N: usize = 10;
        const M: usize = 3;

        let f: [_; M] = [
            Box::new(|x: &mut [Complex<f32>]| {
                x.fft_cooley_tukey();
                x.ifft_cooley_tukey();
            }) as Box<dyn Fn(&mut [Complex<f32>])>,
            Box::new(|x: &mut [Complex<f32>]| {
                x.fft_rec_cooley_tukey();
                x.ifft_rec_cooley_tukey();
            }) as Box<dyn Fn(&mut [Complex<f32>])>,
            Box::new(|x: &mut [Complex<f32>]| {
                let fft = FftPlanner::new()
                    .plan_fft_forward(x.len());
                fft.process(x);
                let ifft = FftPlanner::new()
                    .plan_fft_inverse(x.len());
                ifft.process(x);
            }) as Box<dyn Fn(&mut [Complex<f32>])>,
        ];

        let plot_title: &str = &format!("{fn_name} benchmark");
        let plot_path: &str = &format!("{PLOT_TARGET}/{fn_name}_benchmark.png");

        let t = f.map(|f| ArrayOps::fill_boxed(|n| {
            let mut x = vec![Complex::from(1.0); 1 << n];
            let t0 = SystemTime::now();
            for _ in 0..1024
            {
                f(&mut x);
            }
            let dt = SystemTime::now().duration_since(t0).unwrap();
            println!("Done N = {}", 1 << n);
            dt.as_secs_f32()
        }));

        let n = <[f32; N]>::fill_boxed(|n| (1 << n) as f32);

        plot::plot_curves(plot_title, plot_path, [&*n; M], t.each_ref2().map(|t| &**t)).expect("Plot error")
    }*/
}
