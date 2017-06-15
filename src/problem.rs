use std;

use ea::*;

pub trait Bar{

}

pub trait Problem{
    fn is_solution(&self, value: f32) -> bool {
        value < 1e-3f32
    }

    fn compute<T: Individual>(&self, ind: &mut T) -> f32;
}

//---------------------------------------------------------------
/*
Some other functions: https://en.wikipedia.org/wiki/Test_functions_for_optimization
*/

#[allow(dead_code)]
pub struct SphereProblem;
impl Problem for SphereProblem {
    fn compute<T: Individual>(&self, ind: &mut T) -> f32 {
        let v = ind.to_vec().unwrap();
        if v.len() > 0 {v.iter().fold(0f32, |s, x| s + x*x)} else {std::f32::NAN}
    }
}

//---------------------------------------------------------------
#[allow(dead_code)]
pub struct RastriginProblem;
impl Problem for RastriginProblem {
    fn compute<T: Individual>(&self, ind: &mut T) -> f32 {
        const PI2: f32 = 2f32 * std::f32::consts::PI;
        let v = ind.to_vec().unwrap();
        if v.len() > 0 {v.iter().fold(10f32*(v.len() as f32), |s, x| s + x*x - 10f32 * (x * PI2).cos())}
        else {std::f32::NAN}
    }
}


//---------------------------------------------------------------
#[allow(dead_code)]
pub struct RosenbrockProblem;
impl Problem for RosenbrockProblem {
    fn compute<T: Individual>(&self, ind: &mut T) -> f32 {
        let v = ind.to_vec().unwrap();
        if v.len() == 0 {return std::f32::NAN;}

        let mut res = 0f32;
        for k in 1..v.len() {
            let xk = v[k as usize];
            let xk1 = v[(k - 1) as usize];
            res += 100f32 * (xk - xk1).powf(2f32) + (xk1 - 1f32).powf(2f32)
        }
        res
    }
}


