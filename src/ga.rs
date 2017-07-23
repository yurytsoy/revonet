use rand;
use rand::{Rng};
use rand::distributions::{Normal, IndependentSample, Range};
use serde::de::{DeserializeOwned};
use serde::ser::Serialize;
use std::rc::Rc;

use context::*;
use ea::*;
use problem::*;
use result::*;
use settings::*;

/// Baseline structure for [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)
pub struct GA<'a, P: Problem + 'a, T: Individual+Serialize> {
    /// Context structure containing information about GA run, its progress and results.
    ctx: Option<EAContext<T>>,
    /// Reference to the objective function object implementing `Problem` trait.
    problem: &'a P,
}

impl<'a, P: Problem, T: Individual + Clone + Serialize + DeserializeOwned + 'a> GA<'a, P, T> {
    /// Create a new GA instance for the given `problem`.
    pub fn new(problem: &'a P) -> GA<P, T> {
        GA{problem: problem,
           ctx: None,
        }
    }

    /// Main entry point for the GA. Runs GA with given `settings` and returns `EAResult` object.
    ///
    /// # Arguments:
    /// * `settings` - `EASettings` object.
    pub fn run(&'a mut self, settings: EASettings) -> Result<&EAResult<T>, ()> {
        let gen_count = settings.gen_count;
        let mut ctx = EAContext::new(settings, self.problem);
        self.run_with_context(&mut ctx, self.problem, gen_count);
        self.ctx = Some(ctx);
        // Ok(Rc::new(&(&self.ctx.as_ref().expect("Empty EAContext")).result))
        Ok(&(&self.ctx.as_ref().expect("Empty EAContext")).result)
    }
}

impl<'a, P: Problem, T: Individual+Serialize> EA<'a, T> for GA<'a, P, T> {
    fn breed(&self, ctx: &mut EAContext<T>, sel_inds: &Vec<usize>, children: &mut Vec<T>) {
        cross(&ctx.population, sel_inds, children, ctx.settings.use_elite, ctx.settings.x_type, ctx.settings.x_prob, ctx.settings.x_alpha, &mut ctx.rng);
        mutate(children, ctx.settings.mut_type, ctx.settings.mut_prob, ctx.settings.mut_sigma, &mut ctx.rng);
    }
}

/// Function for crossing individuals to produce children.
///
/// # Arguments:
/// * `popul` - reference to parent population.
/// * `sel_inds` - reference to vector of individuals selected for crossing.
/// * `children` - container for children individuals.
/// * `use_elite` - flag to specify whether elite individual should be copied to the children
///                 population.
/// * `x_type` - crossover operator type defined by `CrossoverOperator` enum.
/// * `x_prob` - crossing probability. If random U(0, 1) number is above this probability then
///              no crossing is performed and children are simply copy of selected parents.
/// * `x_alpha` - parameter for a crossover operator.
/// * `rng` - reference to pre-initialized RNG.
pub fn cross<R: Rng+Sized, T: Individual>(popul: &Vec<T>, sel_inds: &Vec<usize>, children: &mut Vec<T>, use_elite: bool, x_type: CrossoverOperator, x_prob: f32, x_alpha: f32, mut rng: &mut R) {
    let range = Range::new(0, popul.len());
    if use_elite {
        children.push(get_best_individual(popul));
    }
    let xfunc = match x_type {
        CrossoverOperator::Arithmetic => cross_arithmetic,
        CrossoverOperator::BlxAlpha => cross_blx_alpha,
    };
    loop {
        // select parent individuals
        let p1: usize = range.ind_sample(rng);
        let mut p2: usize = range.ind_sample(rng);
        while p2 == p1 {
            p2 = range.ind_sample(rng);
        }

        let mut c1 = popul[sel_inds[p1]].clone();   // T::new();
        let mut c2 = popul[sel_inds[p2]].clone();   // T::new();
        if rand::random::<f32>() < x_prob {
            xfunc(&popul[sel_inds[p1]], &popul[sel_inds[p2]], &mut c1, &mut c2, x_alpha, rng);
        }
        children.push(c1);
        children.push(c2);

        if children.len() >= popul.len() {break;}
    }
}

/// Implementation of the arithmetic crossover.
///
/// # Arguments:
/// * `p1` - reference to the 1st parent.
/// * `p2` - reference to the 2nd parent.
/// * `c1` - reference to the 1st child.
/// * `c2` - reference to the 2nd child.
/// * `alpha` - parameter for crossover, controlling range of the child gene values.
/// * `rng` - reference to pre-initialized RNG.
#[allow(unused_variables)]
fn cross_arithmetic<T: Individual, R: Rng+Sized>(p1: &T, p2: &T, c1: &mut T, c2: &mut T, alpha: f32, mut rng: &mut R) {
    let p1_genes = p1.to_vec().expect("Can not extract vector of genes");
    let p2_genes = p2.to_vec().expect("Can not extract vector of genes");
    // println!("{} : {}", p1_genes.len(), p2_genes.len());
    assert!(p1_genes.len() == p2_genes.len());

    let gene_count = p1_genes.len();
    let c1_genes = c1.to_vec_mut().expect("Can not extract mutable vector of genes");
    let c2_genes = c2.to_vec_mut().expect("Can not extract mutable vector of genes");

    for k in 0..gene_count {
        let a = rng.gen::<f32>();
        c1_genes[k] = p1_genes[k]*a + p2_genes[k]*(1f32-a);
        c2_genes[k] = p2_genes[k]*a + p1_genes[k]*(1f32-a);
    };
}

/// Implementation of the BLX-alpha crossover.
///
/// # Arguments:
/// * `p1` - reference to the 1st parent.
/// * `p2` - reference to the 2nd parent.
/// * `c1` - reference to the 1st child.
/// * `c2` - reference to the 2nd child.
/// * `alpha` - parameter for crossover, controlling range of the child gene values.
/// * `rng` - reference to pre-initialized RNG.
fn cross_blx_alpha<T: Individual, R: Rng+Sized>(p1: &T, p2: &T, c1: &mut T, c2: &mut T, alpha: f32, mut rng: &mut R) {
    let p1_genes = p1.to_vec().expect("Can not extract vector of genes");
    let p2_genes = p2.to_vec().expect("Can not extract vector of genes");
    // println!("{} : {}", p1_genes.len(), p2_genes.len());
    assert!(p1_genes.len() == p2_genes.len());

    let gene_count = p1_genes.len();
    let c1_genes = c1.to_vec_mut().expect("Can not extract mutable vector of genes");
    let c2_genes = c2.to_vec_mut().expect("Can not extract mutable vector of genes");

    for k in 0..gene_count {
        let (min_gene, max_gene) = if p1_genes[k] > p2_genes[k] {(p2_genes[k], p1_genes[k])}
                                   else                         {(p1_genes[k], p2_genes[k])};
        if min_gene < max_gene {
            let delta = max_gene - min_gene;
            let gene_range = Range::new(min_gene - delta*alpha, max_gene + delta*alpha);
            c1_genes[k] = gene_range.ind_sample(&mut rng);
            c2_genes[k] = gene_range.ind_sample(&mut rng);
        } else {
            c1_genes[k] = min_gene;
            c2_genes[k] = max_gene;
        }
    };
    // println!("children: {} : {}", c1_genes.len(), c2_genes.len());
}

/// Function for mutation of the given population.
///
/// # Arguments:
/// * `children` - population to undergo mutation.
/// * `mut_prob` - probability of mutation of single gene.
/// * `mut_sigma` - mutation parameter.
/// * `rng` - reference to pre-initialized RNG.
pub fn mutate<T: Individual, R: Rng>(children: &mut Vec<T>, mut_type: MutationOperator,  mut_prob: f32, mut_sigma: f32, rng: &mut R) {
    let mut_func = match mut_type {
        MutationOperator::Gaussian => mutate_gauss,
        MutationOperator::Uniform => mutate_uniform,
    };
    for k in 1..children.len() {
        mut_func(&mut children[k], mut_prob, mut_sigma, rng);
    }
}

/// Implementation of the Gaussian mutation.
///
/// # Arguments:
/// * `ind` - individual to be mutated.
/// * `prob` - probability of mutation of single gene.
/// * `sigma` - standard deviation of mutation.
/// * `rng` - reference to pre-initialized RNG.
fn mutate_gauss<T: Individual, R: Rng>(ind: &mut T, prob: f32, sigma: f32, rng: &mut R) {
    let normal_rng = Normal::new(0.0, sigma as f64);
    let genes = ind.to_vec_mut().expect("Can not extract mutable vector of genes");
    for k in 0..genes.len() {
        if rand::random::<f32>() < prob {
            genes[k] += normal_rng.ind_sample(rng) as f32;
        }
    }
}

/// Implementation of the uniform mutation. Each gene is updated with probability `prob`
/// by the value taken from `U(gene-sigma, gene+sigma)`.
///
/// # Arguments:
/// * `ind` - individual to be mutated.
/// * `prob` - probability of mutation of single gene.
/// * `sigma` - standard deviation of mutation.
/// * `rng` - reference to pre-initialized RNG.
#[allow(dead_code, unused_variables)]
fn mutate_uniform<T: Individual, R: Rng>(ind: &mut T, prob: f32, sigma: f32, rng: &mut R) {
    let genes = ind.to_vec_mut().expect("Can not extract mutable vector of genes");
    let sigma2 = sigma * 2f32;
    for k in 0..genes.len() {
        if rand::random::<f32>() < prob {
            genes[k] += rng.gen::<f32>() * sigma2 - sigma;
        }
    }
}

//============================================

#[cfg(test)]
#[allow(unused_imports, unused_mut, unused_variables)]
mod test {
    use rand;
    use rand::{StdRng, SeedableRng};

    use ea::*;
    use ga::*;
    use math::*;
    use settings::*;

    #[test]
    fn test_arithmetic_xover() {
        let mut rng = StdRng::from_seed(&[0 as usize]);

        const SIZE: usize = 20;
        let mut p1 = RealCodedIndividual::new();
        p1.init(SIZE, &mut rng);
        let mut p2 = RealCodedIndividual::new();
        p2.init(SIZE, &mut rng);

        let mut c1 = RealCodedIndividual::new();
        c1.init(SIZE, &mut rng);
        let mut c2 = RealCodedIndividual::new();
        c2.init(SIZE, &mut rng);
        cross_arithmetic(&p1, &p2, &mut c1, &mut c2, 0.1f32, &mut rng);

        let p1_genes = p1.to_vec().unwrap();
        let p2_genes = p2.to_vec().unwrap();
        let c1_genes = c1.to_vec().unwrap();
        let c2_genes = c2.to_vec().unwrap();
        for k in 0..SIZE {
            let c_sum = c1_genes[k] + c2_genes[k];
            let p_sum = p1_genes[k] + p2_genes[k];
//            println!("{} : {}", , p1_genes[k] + p2_genes[k]);
            assert!((c_sum - p_sum).abs() < 1e-6f32);
        }
    }

    #[test]
    fn test_blx_xover() {
        let mut rng = StdRng::from_seed(&[0 as usize]);

        const SIZE: usize = 20;
        let mut p1 = RealCodedIndividual::new();
        p1.init(SIZE, &mut rng);
        let mut p2 = RealCodedIndividual::new();
        p2.init(SIZE, &mut rng);

        let mut c1 = RealCodedIndividual::new();
        c1.init(SIZE, &mut rng);
        let mut c2 = RealCodedIndividual::new();
        c2.init(SIZE, &mut rng);
        cross_blx_alpha(&p1, &p2, &mut c1, &mut c2, 0.1f32, &mut rng);

        let p1_genes = p1.to_vec().unwrap();
        let p2_genes = p2.to_vec().unwrap();
        let c1_genes = c1.to_vec().unwrap();
        let c2_genes = c2.to_vec().unwrap();
        for k in 0..SIZE {
            assert!(c1_genes[k] != p1_genes[k]);
            assert!(c1_genes[k] != p2_genes[k]);
            assert!(c2_genes[k] != p1_genes[k]);
            assert!(c2_genes[k] != p2_genes[k]);
        }
    }

    #[test]
    fn test_gauss_mutation() {
        const SIZE: usize = 10;
        const TRIALS: usize = 1000;

        let sqrt_trials = (TRIALS as f32).sqrt();
//        let mut rng = rand::thread_rng();
        let mut rng = StdRng::from_seed(&[0 as usize]);
        let mut sigma = 0f32;
        while sigma <= 0.5f32 {
            let mut p1 = RealCodedIndividual::new();
            p1.genes = vec![0f32; SIZE];
            for _ in 0..TRIALS {
                mutate_gauss(&mut p1, 1f32, sigma, &mut rng);
            }
            // the resulting distribution should be gaussian with SD = sqrt(1000) * sigma
            let max_dist = sqrt_trials * sigma;
            let (max_val, min_val) = (max(&p1.genes), min(&p1.genes));
            assert!(max_val <= 4f32 *max_dist);
            assert!(min_val >= -4f32*max_dist);
            assert!(max_val >= 4f32*sigma);
            assert!(min_val <= -4f32*sigma);

            sigma += 0.1f32;
        }
    }

    #[test]
    fn test_uniform_mutation() {
        const SIZE: usize = 10;
        const TRIALS: usize = 100;

        let sqrt_trials = (TRIALS as f32).sqrt();
        //        let mut rng = rand::thread_rng();
        let mut rng = StdRng::from_seed(&[0 as usize]);
        let mut sigma = 0f32;
        for _ in 0..TRIALS {
            while sigma <= 0.5f32 {
                let mut p1 = RealCodedIndividual::new();
                p1.genes = vec![0f32; SIZE];
                mutate_uniform(&mut p1, 1f32, sigma, &mut rng);

                let norm = dot(&p1.genes, &p1.genes);
//                println!("{} : {}", norm, sigma*sigma);
                assert!(norm <= sigma*sigma*(SIZE as f32));

//                // TODO: improve the test by utilizing CLT.
//                let max_dist = sqrt_trials * sigma;
//                let (max_val, min_val) = (max(&p1.genes), min(&p1.genes));
//                assert!(max_val <= 4f32 *max_dist);
//                assert!(min_val >= -4f32*max_dist);
//                assert!(max_val >= 4f32*sigma);
//                assert!(min_val <= -4f32*sigma);
                sigma += 0.1f32;
            }
        }
    }

    #[test]
    fn test_optimization_sphere() {
        let pop_size = 20u32;
        let problem_dim = 10u32;
        let problem = SphereProblem{};

        let gen_count = 10u32;
        let mut settings = EASettings::new(pop_size, gen_count, problem_dim);
//        settings.x_type = CrossoverOperator::Arithmetic;
        settings.mut_type = MutationOperator::Uniform;
        let mut ga: GA<SphereProblem, RealCodedIndividual> = GA::new(&problem);
        let res = ga.run(settings).expect("Error during GA run");
        for k in 1..res.avg_fitness.len() {
            assert!(res.avg_fitness[k-1] >= res.avg_fitness[k]);
        }
    }
}
