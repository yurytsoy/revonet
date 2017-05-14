// use rand;
use rand::{ThreadRng};
use rand::distributions::{Normal, IndependentSample};
use std;


pub fn rand_vector_stdgauss(size: usize, rng: &mut ThreadRng) -> Vec<f32> {
    let normal_rng = Normal::new(0.0, 1.0);
    (0..size).map(|_| normal_rng.ind_sample(rng) as f32).collect::<Vec<f32>>()
}

#[allow(dead_code)]
pub fn rand_matrix_stdgauss(height: usize, width: usize, rng: &mut ThreadRng) -> Vec<Vec<f32>> {
    (0..height).map(|_| rand_vector_stdgauss(width, rng)).collect::<Vec<Vec<f32>>>()
}

pub fn dot(x1s: &[f32], x2s: &[f32]) -> f32 {
    x1s.iter().zip(x2s).fold(0f32, |s, (&x1, &x2)| s + x1*x2)
}

pub fn dot_mv(m: &Vec<Vec<f32>>, xs: &[f32]) -> Vec<f32> {
    // unimplemented!()
    m.iter().map(|ref row_k| dot(row_k, xs)).collect::<Vec<f32>>()
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
    fn test_dot_zeros() {
        let x1s = [1f32; 100];
        let zeros = [0f32; 100];
        assert!(dot(&x1s, &zeros) == 0f32);
    }

    #[test]
    fn test_dot_units() {
        const LEN: usize = 100;
        let x1s = [1f32; LEN];
        let x2s = [1f32; LEN];
        assert!(dot(&x1s, &x2s) == LEN as f32);
    }

    #[test]
    fn test_dot_rand() {
        const LEN: usize = 100;
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x1s = rand_vector_stdgauss(LEN, &mut rng);
            let x2s = rand_vector_stdgauss(LEN, &mut rng);
            let mut sum = 0f32;
            for k in 0..LEN {
                sum += x1s[k] * x2s[k];
            }
            assert!(dot(&x1s, &x2s) == sum);
        }
    }

    #[test]
    #[should_panic]
    fn test_dot_unequal_lengths() {
        let x1s = [1f32; 10];
        let x2s = [1f32; 100];
        assert!(dot(&x1s, &x2s) == 0f32);
    }

    #[test]
    fn test_dot_mv_zeros() {
        let x1s = vec![vec![1f32; 10]; 10];
        let zeros = [0f32; 10];
        let res = dot_mv(&x1s, &zeros);
        assert!(res.into_iter().all(|x| x == 0f32));
    }

    #[test]
    fn test_dot_mv_rand() {
        const LEN1: usize = 10;
        const LEN2: usize = 20;
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x1s = rand_matrix_stdgauss(LEN2, LEN1, &mut rng);
            let x2s = rand_vector_stdgauss(LEN1, &mut rng);
            let dot_res = dot_mv(&x1s, &x2s);
            assert!(dot_res.len() == LEN2);
            
            // println!("{:?}", dot_res);
            let mut res: Vec<f32> = Vec::with_capacity(LEN2);
            for k in 0..LEN2 {
                res.push(dot(&x1s[k], &x2s) - dot_res[k]);
            }
            assert!(res.into_iter().all(|x| x == 0f32));
        }
    }
}