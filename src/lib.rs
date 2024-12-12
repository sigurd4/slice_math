#![feature(associated_type_bounds)]
#![feature(generic_arg_infer)]
#![feature(array_methods)]
#![feature(let_chains)]
#![feature(new_uninit)]
#![feature(slice_as_chunks)]
#![feature(const_trait_impl)]

#![feature(core_intrinsics)]

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
    use core::f64::consts::{FRAC_1_SQRT_2, SQRT_2};
    use std::{f32::NAN, ops::RangeBounds, time::{Duration, SystemTime}};

    //use array__ops::{Array2dOps, ArrayNd, ArrayOps};
    //use linspace::{Linspace, LinspaceArray};
    use num::Complex;
    use rustfft::FftPlanner;
    //use rustfft::{Fft, FftPlanner};

    use super::*;

    #[test]
    fn fft()
    {
        let mut a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].map(Complex::from);
        a.fft();

        println!("{:?}", a)
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn detrend()
    {
        let mut x = [0.0, 1.0, 2.0, 3.0, -5.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        x.detrend(1);

        println!("{:?}", x)
    }

    #[test]
    fn deconvolve()
    {
        let x = [1.0, 2.0, 3.0];
        let h = [1.0, 1.0, 3.0, 4.0];
        let y: Vec<_> = x.convolve_direct(&h);
        let (x1, r1): (Vec<_>, Vec<_>) = y.deconvolve_direct(&h).unwrap();
        let (x2, r2): (Vec<_>, Vec<_>) = y.deconvolve_fft(&h).unwrap();
        println!("{:?}, {:?}", x1, r1);
        println!("{:?}, {:?}", x2, r2);
    }
    
    #[test]
    fn dct()
    {
        let mut a: [f64; _] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        a.dct_i();
        println!("{:?}", a);
    }

    #[test]
    fn wht()
    {
        let mut a = [19, -1, 11, -9, -7, 13, -15, 5].map(|a| a as f64);
        a.fwht();
        //a.fwht();
        println!("{:?}", a)
    }

    #[test]
    fn fft_accuracy()
    {
        const L: usize = 1024;

        let mut x1: Vec<_> = (0..L).map(|i| Complex::from(((i*136 + 50*i*i + 13) % 100) as f64/100.0)).collect();
        let mut x2 = x1.clone();

        x1.fft();
        /*let fft = FftPlanner::new()
            .plan_fft_forward(L);
        fft.process(&mut x2);*/
        fft::dft_unscaled::<_, false>(&mut x2, &mut None);

        let e = x1.iter()
            .zip(x2)
            .map(|(x1, x2)| (x1 - x2).norm())
            .sum::<f64>()/L as f64;
        println!("{:?}", e)
    }

    #[test]
    fn cconv()
    {
        const N: usize = 1024;
        const M: usize = 20;

        let x: [_; N] = core::array::from_fn(|i| ((i*136 + 50*i*i + 13) % 100) as f64/100.0/(i + 1) as f64);
        let z: [_; M] = core::array::from_fn(|i| ((i*446 + 12*i*i + 59) % 100) as f64/100.0);

        let y1: Vec<_> = x.cconvolve_direct(&z);
        let y2: Vec<_> = x.cconvolve_fft(&z);
        
        plot::plot_curves::<N, _>("e(i)", "plots/cconv_error.png",
            [
                &core::array::from_fn(|i| i as f32),
                &core::array::from_fn(|i| i as f32),
            ],
            [
                &core::array::from_fn(|i| y1[i] as f32),
                &core::array::from_fn(|i| y2[i] as f32),
            ]).unwrap();

        let e = y1.into_iter()
            .zip(y2)
            .map(|(y1, y2)| (y1 - y2).abs())
            .sum::<f64>()/(N.max(M)) as f64;
        println!("{:?}", e)
    }

    #[test]
    fn conv_accuracy()
    {
        const L: usize = 1024;

        let x: Vec<_> = (0..L).map(|i| ((i*136 + 50*i*i + 13) % 100) as f64/100.0/(i + 1) as f64).collect();
        
        let z: Vec<_> = (0..20).map(|i| ((i*446 + 12*i*i + 59) % 100) as f64/100.0).collect();

        let y1: Vec<_> = x.convolve_fft(&z);
        let y2: Vec<_> = x.convolve_direct(&z);
        let err: Vec<_> = y1.iter()
            .zip(y2.iter())
            .map(|(y1, y2)| y1 - y2)
            .collect();

        plot::plot_curves::<L, _>("e(i)", "plots/conv_error.png",
            [
                &core::array::from_fn(|i| i as f32),
                &core::array::from_fn(|i| i as f32),
            ],
            [
                &core::array::from_fn(|i| y1[i] as f32),
                &core::array::from_fn(|i| y2[i] as f32),
            ]).unwrap();

        let e = y1.into_iter()
            .zip(y2)
            .map(|(y1, y2)| (y1 - y2).abs())
            .sum::<f64>()/L as f64;
        println!("{:?}", e)
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_polyfit()
    {
        let p = [2.0, -4.0, 1.0];

        let x = [4.0, -1.0, 6.0, 7.0];
        let y = x.map(|x| p.rpolynomial(x));

        let p: Vec<_> = x.rpolyfit(&y, 2);

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
    fn bench_fft()
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
                const O: usize = 1024;

                let mut x = vec![Complex::from(1.0); n];
                let t0 = SystemTime::now();
                for _ in 0..O
                {
                    f(&mut x);
                }
                let dt = SystemTime::now().duration_since(t0).unwrap();
                println!("Done N = {}", n);
                t[n] = dt.as_secs_f32()/O as f32
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
    #[ignore]
    fn bench_fct()
    {
        let fn_name = "FCT";

        const N: usize = 128 + 1;
        const M: usize = 1;

        let f: [_; M] = [
            Box::new(|x: &mut [Complex<f32>]| {
                x.fct();
                x.ifct();
            }) as Box<dyn Fn(&mut [Complex<f32>])>
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
    }*/

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
