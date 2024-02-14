use std::env;
use statrs::distribution::Normal;
use rand::distributions::Distribution;
use rand::Rng;
use crate::optimizer::OwnBayesOpt;

mod bfgs_u;
mod optimizer;

const PARAMETERS: i32 = 1;

// sin objective function
fn objective(x: &Vec<f64>) -> f64 {
    let std = 0.0;
    let res = x[0].powf(2.0) * (5.0 * 3.14 * x[0]).sin().powf(6.0);
    if std == 0.0 {
        return res
    }

    let mut r = rand::thread_rng();
    let noise = Normal::new(0.0, std).unwrap();
    res + noise.sample(&mut r)
}

// 2 parameter objective function
// -x ** 2 - (y - 1) ** 2 + 1
// fn objective(x: &Vec<f64>) -> f64 {
//     let std = 0.0;
//     let res = -x[0].powf(2.0) - (x[1] - 1.0).powf(2.0) + 1.0;
//     if std == 0.0 {
//         return res
//     }

//     let mut r = rand::thread_rng();
//     let noise = Normal::new(0.0, std).unwrap();
//     res + noise.sample(&mut r)
// }


fn main() {
    env::set_var("RUST_BACKTRACE", "1");


    // the gp assumes each sample is a vector
    let mut xs: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..10 {
        let mut sample = Vec::new();
        for _ in 0..PARAMETERS {
            sample.push(rng.gen::<f64>());
        }
        ys.push(objective(&sample));
        xs.push(sample);
        
    }

    // construct optimizer
    let optim = OwnBayesOpt::builder(xs, ys)
                                        .set_bounds(vec![0.0, 1.0])
                                        .set_iter(17)
                                        .set_acquisition_function("EI".to_string());

    let max_args = optim.optimize(objective);


    // let max_arg = optimize(&xs, &ys, objective);

    println!("best arg: {:?}", max_args);
}
