#![feature(associated_type_bounds)]
#![feature(generic_arg_infer)]
#![feature(array_methods)]
#![feature(let_chains)]
#![feature(new_uninit)]

#![feature(generic_const_exprs)]

moddef::moddef!(
    flat(pub) mod {
        slice_math_
    },
    mod {
        fft,
        util,
        plot for cfg(test)
    }
);

pub use slice_ops::*;

#[cfg(test)]
mod tests {
    use core::f64::consts::SQRT_2;
    use std::{f32::NAN, ops::RangeBounds, time::{Duration, SystemTime}};

    //use array__ops::{Array2dOps, ArrayNd, ArrayOps};
    //use linspace::{Linspace, LinspaceArray};
    use num::Complex;
    use rustfft::FftPlanner;
    //use rustfft::{Fft, FftPlanner};

    use super::*;
    
    #[test]
    fn wht()
    {
        let mut a = [19, -1, 11, -9, -7, 13, -15, 5].map(|a| a as f64);
        a.fwht();
        //a.fwht();
        println!("{:?}", a)
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_polyfit()
    {
        let p = [2.0, -4.0, 1.0];

        let x = [4.0, -1.0, 6.0, 7.0];
        let y = x.map(|x| p.rpolynomial(x));

        let p = x.rpolyfit(&y, 2);

        println!("{:?}", p);
    }

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


    #[test]
    #[ignore]
    fn bench()
    {
        let fn_name = "FFT";

        const N: usize = 128 + 1;
        const M: usize = 2;

        let f: [_; M] = [
            Box::new(|x: &mut [Complex<f32>]| {
                x.fft();
                x.ifft();
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
        
        let t = f.map(|f| {
            let mut t: Box<[_; N]> = unsafe {Box::new_uninit().assume_init()};
            for n in 0..N
            {
                let mut x = vec![Complex::from(1.0); n];
                let t0 = SystemTime::now();
                for _ in 0..1024
                {
                    f(&mut x);
                }
                let dt = SystemTime::now().duration_since(t0).unwrap();
                println!("Done N = {}", n);
                t[n] = dt.as_secs_f32()
            }
            t
        });
        let n = {
            let mut n: Box<[_; N]> = unsafe {Box::new_uninit().assume_init()};
            for i in 0..N
            {
                n[i] = i as f32;
            }
            n
        };
        
        plot::plot_curves(plot_title, plot_path, [&*n; M], t.each_ref().map(|t| &**t)).expect("Plot error")
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
