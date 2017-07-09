use rand::{Rng};
use rand::distributions::{IndependentSample, Range};
use serde::ser::Serialize;
use std;

use context::*;
use math::*;
use neuro::{MultilayeredNetwork};
use problem::*;

/// Trait representing functionality required to evolve an individual for optimization
/// and NN tuning tasks.
///
/// Contains functions to retrieve genes or neural network from an individual and get/set its fitness.
#[allow(dead_code, unused_variables)]
pub trait Individual{
    /// Creates a new individual with empty set of genes and NAN fitness.
    fn new() -> Self;
    /// Initializes an individual by allocating random vector of genes using Gaussian distribution.
    ///
    /// # Arguments
    /// * `size` - number of genes.
    /// * `rng` - mutable reference to the external RNG.
    fn init<R: Rng>(&mut self, size: usize, &mut R);
    /// Create a copy of an individual.
    fn clone(&self) -> Self;
    /// Return current fitness value.
    fn get_fitness(&self) -> f32;
    /// Update fitness value.
    fn set_fitness(&mut self, fitness: f32);
    /// Return vector of genes.
    fn to_vec(&self) -> Option<&[f32]>;
    /// Return mutable vector of genes.
    fn to_vec_mut(&mut self) -> Option<&mut Vec<f32>>;
    /// Return `MultilayeredNetwork` object with weights assigned according to the genes' values.
    fn to_net(&mut self) -> Option<&MultilayeredNetwork> {None}
    /// Return mutable `MultilayeredNetwork` object with weights assigned according to the genes' values.
    fn to_net_mut(&mut self) -> Option<&mut MultilayeredNetwork> {None}
    /// Update individual's `MultilayeredNetwork` object and update genes according to the network weights.
    ///
    /// # Arguments:
    /// * `net` - neural network to update from.
    fn set_net(&mut self, net: MultilayeredNetwork) {}
}

/// Represents real-coded individual with genes encoded as vector of real numbers.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RealCodedIndividual {
    /// Collection of individual genes.
    pub genes: Vec<f32>,
    /// Fitness value associated with the individual.
    pub fitness: f32,
}

impl RealCodedIndividual {
}

impl Individual for RealCodedIndividual{
    fn new() -> Self {
        RealCodedIndividual{genes: Vec::new(), fitness: std::f32::NAN}
    }

    fn init<R: Rng>(&mut self, size: usize, mut rng: &mut R) {
        self.genes = rand_vector_std_gauss(size as usize, rng);
    }

    fn clone(&self) -> Self {
        RealCodedIndividual{genes: self.genes.clone(), fitness: self.fitness}
    }

    fn get_fitness(&self) -> f32 {
        self.fitness
    }

    fn set_fitness(&mut self, fitness: f32) {
        self.fitness = fitness;
    }

    fn to_vec(&self) -> Option<&[f32]> {
        Some(&self.genes)
    }

    fn to_vec_mut(&mut self) -> Option<&mut Vec<f32>> {
        Some(&mut self.genes)
    }
}

//======================================================================

/// Trait for an evolutionary algorithm.
/// Defines functions which are typical for running a common EA.
/// To implement a trait a function `breed` should be implemented.
pub trait EA<'a, T: Individual+Serialize> {
    /// "Main" function for the EA which runs a cycle for an evolutionary search.
    ///
    /// # Arguments:
    /// * `ctx` - `EAContext` object containing information regarding current EA run.
    /// * `problem` - reference to the `Problem` trait which specifies an objective function.
    /// * `gen_count` - number of generations (iterations) for search.
    fn run_with_context<P: Problem>(&self, ctx: &mut EAContext<T>, problem: &P, gen_count: u32) { // -> Result<Rc<&'a EAResult<T>>, ()> {
        // let mut ctx = self.get_context_mut();
        // println!("run_with_context");
        for t in 0..gen_count {
            // evaluation
            // println!("evaluation");
            self.evaluate(ctx, problem);

            // selection
            // println!("selection");
            let sel_inds = self.select(ctx);

            // crossover
            // println!("crossover");
            let mut children: Vec<T> = Vec::with_capacity(ctx.settings.pop_size as usize);
            self.breed(ctx, &sel_inds,  &mut children);

            // next gen
            // println!("next_generation");
            self.next_generation(ctx, &children);

            println!("> {} : {:?}", t, ctx.fitness);
            println!(" Best fitness at generation {} : {}\n", t, min(&ctx.fitness));
        }
        // Ok(Rc::new(&ctx.result))
    }

    /// Function to evaluate current population for the given `problem`. In result of evaluation
    ///   fitness for every individual is updated as well as statistics regarding mean, min, max
    ///   fitness values.
    ///
    /// # Arguments:
    /// * `ctx` - `EAContext` object containing information regarding current EA run.
    /// * `problem` - reference to the `Problem` trait which specifies an objective function.
    fn evaluate<P: Problem>(&self, ctx: &mut EAContext<T>, problem: &P) {
        // ctx.fitness = evaluate(&mut ctx.population, problem, &mut ctx.result);
        let cur_result = &mut ctx.result;
        let popul = &mut ctx.population;

        ctx.fitness = popul.iter_mut().map(|ref mut ind| {
                let f = problem.compute(ind as &mut T);
                // println!(".");
                ind.set_fitness(f);
                f
            }).collect::<Vec<f32>>();
        let fits = &ctx.fitness;
        // println!("{:?}", fits);
        if cur_result.first_hit_fe_count == 0 {
            for k in 0..fits.len() {
                if problem.is_solution(fits[k]) {
                    cur_result.first_hit_fe_count = cur_result.fe_count + (k+1) as u32;
                    break;
                }
            }
        }

        cur_result.avg_fitness.push(mean(&fits));
        cur_result.max_fitness.push(max(&fits));

        let last_min_fitness = min(&fits);
        cur_result.min_fitness.push(last_min_fitness);
        if cur_result.best.get_fitness().is_nan() || (cur_result.best.get_fitness() > last_min_fitness) {
            let idx = (&fits).iter().position(|&x| x == last_min_fitness).expect("Min fitness is not found");
            cur_result.best = popul[idx].clone();
            cur_result.best_fe_count = cur_result.fe_count + (idx+1) as u32;
        }
        cur_result.best.set_fitness(last_min_fitness);
        cur_result.fe_count += fits.len() as u32;
    }

    /// Function to select individuals for breeding. Updates given `ctx`.
    ///
    /// # Arguments:
    /// * `ctx` - `EAContext` object containing information regarding current EA run.
    fn select(&self, ctx: &mut EAContext<T>) -> Vec<usize> {
        select_tournament(&ctx.fitness, ctx.settings.tour_size, &mut ctx.rng)
    }

    /// Function to prepare population for the next generation. By default copies children obtained
    ///   in result of `breed` to the `ctx.population`.
    ///
    /// # Arguments:
    /// * `ctx` - `EAContext` object containing information regarding current EA run.
    /// * `children` - reference to the vector of children individuals which should
    ///                get to the next generation.
    fn next_generation(&self, ctx: &mut EAContext<T>, children: &Vec<T>) {
        ctx.population = Vec::with_capacity(children.len());
        for k in 0..children.len() {
            ctx.population.push(children[k].clone());
        }
        // ctx.population = children.iter().map(|ref c| c.clone()).collect::<Vec<T>>();
        ctx.population.truncate(ctx.settings.pop_size as usize);
    }

    /// Function to create children individuals using current context and selected individuals.
    ///
    /// # Arguments:
    /// * `ctx` - `EAContext` object containing information regarding current EA run.
    /// * `sel_inds` - vector of indices of individuals from `ctx.population` selected for breeding.
    /// * `children` - reference to the container to store resulting children individuals.
    fn breed(&self, ctx: &mut EAContext<T>, sel_inds: &Vec<usize>, children: &mut Vec<T>);
}

/// Creates population of given size. Uses `problem.get_random_individual` to generate a
/// new individual
///
/// # Arguments:
/// * `pop_size` - population size.
/// * `ind_size` - default size (number of genes) of individuals.
/// * `rng` - reference to pre-initialized RNG.
/// * `problem` - reference to the object implementing `Problem` trait.
pub fn create_population<T: Individual, P: Problem, R: Rng+Sized>(pop_size: u32, ind_size: u32, mut rng: &mut R, problem: &P) -> Vec<T> {
    println!("Creating population of {} individuals having size {}", pop_size, ind_size);
    let mut res = Vec::with_capacity(pop_size as usize);
    for _ in 0..pop_size {
        res.push(problem.get_random_individual::<T, R>(ind_size as usize, rng));
    }
    res
}

/// Implementation of the [tournament selection](https://en.wikipedia.org/wiki/Tournament_selection).
///
/// # Arguments:
/// * `fits` - fitness values. i-th element should be equal to the fitness of the i-th individual
///            in population.
/// * `tour_size` - tournament size. The bigger the higher is the selective pressure (more strict
///                 selection). Minimal acceptable value is 2.
/// * `rng` - reference to pre-initialized RNG.
fn select_tournament(fits: &Vec<f32>, tour_size: u32, mut rng: &mut Rng) -> Vec<usize> {
    let range = Range::new(0, fits.len());
    let mut sel_inds: Vec<usize> = Vec::with_capacity(fits.len());  // vector of indices of selected inds. +1 in case of elite RealCodedindividual is used.
    for _ in 0..fits.len() {
        let tour_inds = (0..tour_size).map(|_| range.ind_sample(&mut rng)).collect::<Vec<usize>>();
        let winner = tour_inds.iter().fold(tour_inds[0], |w_idx, &k|
            if fits[w_idx] < fits[k] {w_idx}
            else {k}
        );
        sel_inds.push(winner);
    }
    sel_inds
}

/// Get copy of the individual having the best fitness value.
///
/// # Arguments:
/// * `popul` - vector of individuals to select from.
pub fn get_best_individual<T: Individual>(popul: &Vec<T>) -> T {
    let min_fitness = popul.into_iter().fold(std::f32::MAX, |s, ref ind| if s < ind.get_fitness() {s} else {ind.get_fitness()});
    let idx = popul.into_iter().position(|ref x| x.get_fitness() == min_fitness).expect("Min fitness is not found");
    popul[idx].clone()
}

//========================================================

#[cfg(test)]
mod test {
    use rand;

    use ea::*;
    use math::*;

    #[test]
    fn test_tournament_selection() {
        const IND_COUNT: usize = 100;
        const TRIAL_COUNT: u32 = 100;

        let mut prev_mean = 0.5f32;   // avg value in a random array in [0; 1].
        let mut rng = rand::thread_rng();
        for t in 2..10 {
            let mut cur_mean = 0f32;
            // compute mean fitness for the selected population for TRIAL_COUNT trials.
            for _ in 0..TRIAL_COUNT {
                let fitness_vals = rand_vector(IND_COUNT, &mut rng);
                let sel_inds = select_tournament(&fitness_vals, t, &mut rng);
                let tmp_mean = sel_inds.iter().fold(0f32, |s, &idx| s + fitness_vals[idx]) / IND_COUNT as f32;
                cur_mean += tmp_mean;
            }
            cur_mean /= TRIAL_COUNT as f32;
            // bigger tournaments should give smaller average fitness in selected population.
            assert!(cur_mean < prev_mean);
            prev_mean = cur_mean;
        }

    }
}