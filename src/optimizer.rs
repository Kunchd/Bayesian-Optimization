use std::{f64::INFINITY, string};

use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use rand::distributions::Distribution;
use rand::Rng;
use friedrich::{gaussian_process::GaussianProcess, kernel::Matern2, prior::ConstantPrior};

use ndarray::{Array, Array1};

use super::bfgs_u::bfgs;

// TODO: global flags needs to be moved later
const DEBUG: bool = true;
const PARAMETERS: i32 = 2;

pub struct OwnBayesOpt {
    init_xs: Vec<Vec<f64>>,
    init_ys: Vec<f64>,
    param_bounds: Vec<f64>,
    iterations: usize,
    acq: String,
}

impl OwnBayesOpt {
    /// Expect `init_xs` and `init_ys` to be NON-EMPTY
    pub fn builder(init_xs: Vec<Vec<f64>>, init_ys: Vec<f64>) -> Self {
        Self{
            init_xs: init_xs,
            init_ys: init_ys,
            param_bounds: Vec::new(),
            iterations: 20,
            acq: "EI".to_string(),
        }
    }

    pub fn set_bounds(mut self, bounds: Vec<f64>) -> Self {
        self.param_bounds = bounds;

        self
    }

    pub fn set_iter(mut self, iter: usize) -> Self {
        self.iterations = iter;

        self
    }

    pub fn set_acquisition_function(mut self, acq_func: String) -> Self {
        self.acq = acq_func;

        self
    }

    pub fn optimize<F>(&self, objective: F) -> Vec<f64>
    where 
            F: Fn(&Vec<f64>) -> f64, {
        // attempt gaussian process
        let mut model = self.get_gp(self.init_xs.clone(), self.init_ys.clone());

        let mut max = f64::MIN;
        let mut max_arg = Vec::new();

        let mut xs = self.init_xs.clone();
        let mut ys = self.init_ys.clone();

        // bayesian optimization steps
        for i in 0..self.iterations {
            // select new point to sample using acquisition function on surogate model
            let x = self.acquisition_step(&xs, max, &model);
            // run selected point on objective
            let y = objective(&x);

            // log progress
            println!("Iteration {}, best arg: {:?}", i, x);

            if max < y {
                max = y;
                max_arg = x.clone();
            }

            // append data point
            xs.push(x);
            ys.push(y);
            // update surrogate model
            model = self.get_gp(xs.clone(), ys.clone());
        }

        max_arg
    }

    /////////////////////////////////////////
    // Helper functions
    /////////////////////////////////////////

    // the acquisition function (Expected Improvement variant)
    fn acquisition_ei(&self, x_samples: &Vec<Vec<f64>>, best_y: f64, model: &GaussianProcess<Matern2, ConstantPrior>) -> Vec<f64> {
        //   # calculate the Expected Improvement
        //   a = mu - best_y
        //   z = a / (std + 1e-9)
        //   return a * norm.cdf(z) + std * norm.pdf(z)

        // calculate mu and std of x_samples
        let (y_sample_mu, y_sample_std) = model.predict_mean_variance(x_samples);
        let norm = Normal::new(0.0, 1.0).unwrap();
        let mut ei = Vec::new();
        for i in 0..y_sample_mu.len() {
            let a = y_sample_mu[i] - best_y;
            let z = a / (y_sample_std[i] + 1e-9);
            ei.push(a * norm.cdf(z) + y_sample_std[i] * norm.pdf(z))
        }

        ei
    }

    // the acquisition function (Probability of Improvement variant)
    // makes a prediction for each of the given sample vectors
    fn acquisition_poi(&self, x_samples: &Vec<Vec<f64>>, best_y: f64, model: &GaussianProcess<Matern2, ConstantPrior>) -> Vec<f64> {
        // calculate mu and std of x_samples
        let (y_sample_mu, y_sample_std) = model.predict_mean_variance(x_samples);

        // calculate probability of improvement
        let norm = Normal::new(0.0, 1.0).unwrap();
        let mut probs = Vec::new();
        for i in 0..y_sample_mu.len() {
            probs.push(norm.cdf((y_sample_mu[i] - best_y) / (y_sample_std[i] + 1e-9)));
        }

        probs
    }

    // 2-point approximation of acquisition function gradient
    // Where x is a single sample point vector
    fn acquisition_grad<F>(&self, x: Vec<f64>, f: F) -> Vec<f64>
    where
            F: Fn(&Array1<f64>) -> f64, {
        let eps = 1e-4;     // some small change in x
        let x_p: Vec<f64> = x.clone().iter().map(|x| x + eps).collect();

        // evaluate f(x) and f(xp)
        let y: f64 = f(&Array::from_vec(x.clone()));
        let y_p: f64 = f(&Array::from_vec(x_p.clone()));

        let mut res: Vec<f64> = Vec::new();
        // xs_p.iter().enumerate().map(|(i, v)| v.iter().enumerate().map(|(j, x)| (x - xs[i][j]).abs()));
        for i in 0..x.len() {
            res.push((y_p - y) / (x_p[i] - x[i]).abs());
        }

        res
    }

    // find optimal x to guess (random search)
    fn acquisition_step(&self, xs: &Vec<Vec<f64>>, best_y: f64, model: &GaussianProcess<Matern2, ConstantPrior>) -> Vec<f64> {
        let mut x_seeds = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let mut sample = Vec::new();
            for _ in 0..xs[0].len() {
                sample.push(rng.gen::<f64>());
            }
            x_seeds.push(sample);
        }
        
        // BFGS
        // create expected form wrapper functions
        let f = |x: &Array1<f64>| -> f64 {
            self.acquisition_ei(&vec![x.to_vec()], best_y,  model)[0] * -1.0
        };    

        let g = |x: &Array1<f64>| -> Array1<f64> {
            Array::from_vec(self.acquisition_grad(x.to_vec(), f)) 
        };

        // iterate over possible seeds
        let mut max = f64::MIN;
        let mut max_arg = Vec::new();

        for i in 0..10 {
            // run BFGS algorithm
            if let Ok(x_arg) = bfgs(Array::from_vec(x_seeds[i].clone()), &self.param_bounds, f, g) {
                let eval = f(&x_arg) * -1.0;
                if eval > max {
                    max = eval;
                    max_arg = x_arg.into_raw_vec();
                }
            } else {
                if DEBUG {
                    println!("No optimal x received from bfgs")
                }
            }
        }

        max_arg
    }

    fn get_gp(&self, training_inputs: Vec<Vec<f64>>, training_outputs: Vec<f64>) -> GaussianProcess<Matern2, ConstantPrior> {
        GaussianProcess::builder(training_inputs, training_outputs)
                            .set_kernel(Matern2::default())
                            .fit_kernel()
                            .train()
    } 
    
}
