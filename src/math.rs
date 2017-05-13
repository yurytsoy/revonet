// use rand;
use rand::{ThreadRng};
use rand::distributions::{Normal, IndependentSample};
use std;


pub fn rand_vector_stdgauss(size: usize, rng: &mut ThreadRng) -> Vec<f32> {
    let normal_rng = Normal::new(0.0, 1.0);
    (0..size).map(|_| normal_rng.ind_sample(rng) as f32).collect::<Vec<f32>>()
}

pub fn dot_product(x1s: &[f32], x2s: &[f32]) -> f32 {
    x1s.iter().zip(x2s).fold(0f32, |s, (&x1, &x2)| s + x1*x2)
}

pub fn min(xs: &Vec<f32>) -> f32 {
    xs.into_iter().fold(xs[0], |res, &x| if res < x {res} else {x})
}

pub fn max(xs: &Vec<f32>) -> f32 {
    xs.into_iter().fold(xs[0], |res, &x| if res > x {res} else {x})
}

pub fn mean(xs: &Vec<f32>) -> f32 {
    if xs.len() > 0 {
        xs.into_iter().fold(0f32, |s, &x| s+x) / (xs.len() as f32)
    } else {
        std::f32::NAN
    }
}

//=========================================================================

#[cfg(test)]
mod tests {
    use rand;

    use math::*;

    #[test]
    fn test_dot_product_zeros() {
        let x1s = [1f32; 100];
        let zeros = [0f32; 100];
        assert!(dot_product(&x1s, &zeros) == 0f32);
    }

    #[test]
    fn test_dot_product_units() {
        const LEN: usize = 100;
        let x1s = [1f32; LEN];
        let x2s = [1f32; LEN];
        assert!(dot_product(&x1s, &x2s) == LEN as f32);
    }

    #[test]
    fn test_dot_product_rand() {
        const LEN: usize = 100;
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x1s = rand_vector_stdgauss(LEN, &mut rng);
            let x2s = rand_vector_stdgauss(LEN, &mut rng);
            let mut sum = 0f32;
            for k in 0..LEN {
                sum += x1s[k] * x2s[k];
            }
            assert!(dot_product(&x1s, &x2s) == sum);
        }
    }

    #[test]
    #[should_panic]
    fn test_dot_product_unequal_lengths() {
        let x1s = [1f32; 10];
        let x2s = [1f32; 100];
        assert!(dot_product(&x1s, &x2s) == 0f32);
    }
}