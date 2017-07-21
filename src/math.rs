// use rand;
use rand::{Rng};
use rand::distributions::{Normal, IndependentSample};
use std;

/// Generates random vector where each element is drawn from the standard Gaussian distribution.
///
/// # Arguments:
/// * `size` - resulting vector size.
/// * `rng` - mutable reference to the external RNG.
pub fn rand_vector_std_gauss<T: Rng + Sized>(size: usize, rng: &mut T) -> Vec<f32> {
    let normal_rng = Normal::new(0.0, 1.0);
    (0..size).map(|_| normal_rng.ind_sample(rng) as f32).collect::<Vec<f32>>()
}

/// Generates random vector where each element is drawn from the uniform distribution U(0,1).
///
/// # Arguments:
/// * `size` - resulting vector size.
/// * `rng` - mutable reference to the external RNG.
pub fn rand_vector<T: Rng + Sized>(size: usize, rng: &mut T) -> Vec<f32> {
    (0..size).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>()
}

/// Generates random matrix where each element is drawn from the standard Gaussian distribution.
///
/// # Arguments:
/// * `height` - number of rows.
/// * `width` - number of columns.
/// * `rng` - mutable reference to the external RNG.
#[allow(dead_code)]
pub fn rand_matrix_std_gauss<T: Rng>(height: usize, width: usize, rng: &mut T) -> Vec<Vec<f32>> {
    (0..height).map(|_| rand_vector_std_gauss(width, rng)).collect::<Vec<Vec<f32>>>()
}

/// Dot product between two vectors.
///
/// # Arguments:
/// * `x1s` - 1st vector.
/// * `x2s` - 2nd vector.
pub fn dot(x1s: &[f32], x2s: &[f32]) -> f32 {
    x1s.iter().zip(x2s).fold(0f32, |s, (&x1, &x2)| s + x1*x2)
}

/// Multiplication of matrix by a vector.
///
/// # Arguments:
/// * `m` - matrix.
/// * `xs` - vector.
pub fn dot_mv(m: &Vec<Vec<f32>>, xs: &[f32]) -> Vec<f32> {
    // unimplemented!()
    m.iter().map(|ref row_k| dot(row_k, xs)).collect::<Vec<f32>>()
}

/// Find minimal element in a given vector.
///
/// # Arguments:
/// * `xs` - reference to a vector.
pub fn min(xs: &Vec<f32>) -> f32 {
    xs.into_iter().fold(xs[0], |res, &x| if res < x {res} else {x})
}

/// Computes element-wise min element for two given vectors and stores result in the first argument.
///
/// # Arguments:
/// * `xs` - mutable reference to a vector, where result is also written.
/// * `ys` - reference to a second argument vector.
pub fn min_inplace_vv(xs: &mut[f32], ys: &[f32]) {
    for k in 0..xs.len() {
        if xs[k] > ys[k] {
            xs[k] = ys[k];
        }
    }
}

/// Find maximal element in a given vector.
///
/// # Arguments:
/// * `xs` - reference to a vector.
pub fn max(xs: &Vec<f32>) -> f32 {
    xs.into_iter().fold(xs[0], |res, &x| if res > x {res} else {x})
}

/// Computes element-wise max element for two given vectors and stores result in the first argument.
///
/// # Arguments:
/// * `xs` - mutable reference to a vector, where result is also written.
/// * `ys` - reference to a second argument vector.
pub fn max_inplace_vv(xs: &mut[f32], ys: &[f32]) {
    for k in 0..xs.len() {
        if xs[k] < ys[k] {
            xs[k] = ys[k];
        }
    }
}

/// Compute mean value for a given vector.
///
/// # Arguments:
/// * `xs` - reference to a vector.
pub fn mean(xs: &Vec<f32>) -> f32 {
    if xs.len() > 0 {
        xs.into_iter().fold(0f32, |s, &x| s+x) / (xs.len() as f32)
    } else {
        std::f32::NAN
    }
}

/// Subtraction of vectors. Does `xs - ys`.
///
/// # Arguments:
/// * `xs` - reference to a vector, minuend.
/// * `ys` - reference to a vector, subtrahend.
pub fn sub(xs: &[f32], ys: &[f32]) -> Vec<f32> {
    xs.iter().zip(ys.iter())
        .map(|(&x, &y)| x-y)
        .collect::<Vec<f32>>()
}

/// Inplace subtraction of vectors. Does `xs = xs - ys`.
///
/// # Arguments:
/// * `xs` - mutable reference to a vector, minuend.
/// * `ys` - reference to a vector, subtrahend.
pub fn sub_inplace(xs: &mut [f32], ys: &[f32]) {
    for k in 0..xs.len() {
        xs[k] -= ys[k];
    }
}

/// Accumulates element-wise addition of xs and ys.
///
/// # Arguments:
/// * `xs` - reference to accumulator vector.
/// * `ys` - rhs vector.
pub fn acc(xs: &mut[f32], ys: &[f32]) {
    for k in 0..xs.len() {
        xs[k] += ys[k];
    }
}

/// Inplace multiplication of vector by number. Result is stored in the argument vector.
///
/// # Arguments:
/// * `xs` - reference to vector.
/// * `ys` - multiplier.
pub fn mul_inplace(xs: &mut[f32], a: f32) {
    for k in 0..xs.len() {
        xs[k] *= a;
    }
}

/// Computes element-wise square for the given vector.
///
/// # Arguments:
/// `xs` - reference to vector.
pub fn sqr(xs: &[f32]) -> Vec<f32> {
    xs.iter().map(|x| x*x).collect::<Vec<f32>>()
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
            let x1s = rand_vector_std_gauss(LEN, &mut rng);
            let x2s = rand_vector_std_gauss(LEN, &mut rng);
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
            let x1s = rand_matrix_std_gauss(LEN2, LEN1, &mut rng);
            let x2s = rand_vector_std_gauss(LEN1, &mut rng);
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
