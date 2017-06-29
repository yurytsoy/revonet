use rand;
use rand::{Rng, StdRng};
use rand::distributions::{Normal, IndependentSample, Range};
use std;
use std::rc::Rc;

use context::*;
use math::*;
use neuro::{NeuralNetwork, MultilayeredNetwork};
use problem::*;
use result::*;
use settings::*;

pub trait Individual{
    fn new() -> Self;
    fn init<R: Rng>(&mut self, size: usize, &mut R);
    fn clone(&self) -> Self;
    fn get_fitness(&self) -> f32;
    fn set_fitness(&mut self, fitness: f32);
    fn to_vec(&self) -> Option<&[f32]>;
    fn to_vec_mut(&mut self) -> Option<&mut Vec<f32>>;
    fn to_net(&mut self) -> Option<&MultilayeredNetwork> {None}
    fn to_net_mut(&mut self) -> Option<&mut MultilayeredNetwork> {None}
    fn set_net(&mut self, net: MultilayeredNetwork) {}
}

#[derive(Debug, Clone)]
pub struct RealCodedIndividual {
    pub genes: Vec<f32>,
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

pub trait EA<'a, T: Individual> {
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

    fn select(&self, ctx: &mut EAContext<T>) -> Vec<usize> {
        select_tournament(&ctx.fitness, ctx.settings.tour_size, &mut ctx.rng)
    }

    fn next_generation(&self, ctx: &mut EAContext<T>, children: &Vec<T>) {
        ctx.population = Vec::with_capacity(children.len());
        for k in 0..children.len() {
            ctx.population.push(children[k].clone());
        }
        // ctx.population = children.iter().map(|ref c| c.clone()).collect::<Vec<T>>();
        ctx.population.truncate(ctx.settings.pop_size as usize);
    }

    fn breed(&self, ctx: &mut EAContext<T>, sel_inds: &Vec<usize>, children: &mut Vec<T>);

    // fn get_context_mut(&mut self) -> &'a mut EAContext<T>;
    // fn get_context_mut(&'a mut self) -> &'a mut EAContext<T>;
}

pub fn create_population<T: Individual, P: Problem, R: Rng+Sized>(pop_size: u32, ind_size: u32, mut rng: &mut R, problem: &P) -> Vec<T> {
    println!("Creating population of {} individuals having size {}", pop_size, ind_size);
    let mut res = Vec::with_capacity(pop_size as usize);
    for _ in 0..pop_size {
        res.push(problem.get_random_individual::<T, R>(ind_size as usize, rng));
    }
    res
}

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

pub fn get_best_individual<T: Individual>(popul: &Vec<T>) -> T {
    let min_fitness = popul.into_iter().fold(std::f32::MAX, |s, ref ind| if s < ind.get_fitness() {s} else {ind.get_fitness()});
    let idx = popul.into_iter().position(|ref x| x.get_fitness() == min_fitness).expect("Min fitness is not found");
    popul[idx].clone()
}
