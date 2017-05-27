use rand;
use rand::{Rng};
use rand::distributions::{Normal, IndependentSample, Range};
use std;

use context::*;
use math::*;
use problem::*;
use result::*;
use settings::*;

#[derive(Debug, Clone)]
pub struct Individual {
    pub genes: Vec<f32>,
    pub fitness: f32,
}

impl Individual {
    pub fn new() -> Individual {
        Individual{genes: Vec::new(), fitness: std::f32::NAN}
    }
}

pub trait EA {
    fn run_with_context(&mut self, mut ctx: &mut EAContext, problem: &Problem, gen_count: u32) -> Result<EAResult, ()> {
        for t in 0..gen_count {
            // evaluation
            self.evaluate(&mut ctx, problem);

            // selection
            let sel_inds = self.select(&mut ctx);

            // crossover
            let mut children: Vec<Individual> = Vec::with_capacity(ctx.settings.pop_size as usize);
            self.breed(&mut ctx, &sel_inds,  &mut children);

            // next gen
            self.next_generation(&mut ctx, &children);

            println!("> {} : {:?}", t, ctx.fitness);
            println!(" Best fitness at generation {} : {}\n", t, min(&ctx.fitness));
        }
        Ok(ctx.result.clone())
    }

    fn evaluate(&self, ctx: &mut EAContext, problem: &Problem) {
        // ctx.fitness = evaluate(&mut ctx.population, problem, &mut ctx.result);
        let cur_result = &mut ctx.result;
        let popul = &mut ctx.population;

        ctx.fitness = popul.iter_mut().map(|ref mut ind| {
                ind.fitness = problem.compute_from_ind(ind);
                ind.fitness
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
        cur_result.min_fitness.push(min(&fits));
        cur_result.max_fitness.push(max(&fits));
        if cur_result.best.fitness.is_nan() || (cur_result.best.fitness > *cur_result.min_fitness.last().unwrap()) {
            let idx = (&fits).iter().position(|&x| x == *cur_result.min_fitness.last().unwrap()).expect("Min fitness is not found");
            cur_result.best = popul[idx].clone();
            cur_result.best_fe_count = cur_result.fe_count + (idx+1) as u32;
        }
        cur_result.best.fitness = *cur_result.min_fitness.last().unwrap();
        cur_result.fe_count += fits.len() as u32;
    }

    fn select(&self, ctx: &mut EAContext) -> Vec<usize> {
        select_tournament(&ctx.fitness, ctx.settings.tour_size, &mut ctx.rng)
    }

    fn next_generation(&self, ctx: &mut EAContext, children: &Vec<Individual>) {
        ctx.population.clone_from(children);
        ctx.population.truncate(ctx.settings.pop_size as usize);
    }

    fn breed(&self, ctx: &mut EAContext, sel_inds: &Vec<usize>, children: &mut Vec<Individual>);
}

pub fn create_population(pop_size: u32, ind_size: u32, mut rng: &mut Rng) -> Vec<Individual> {
    (0..pop_size)
        .map(|_| {
            let mut res_ind = Individual::new();
            res_ind.genes = rand_vector_std_gauss(ind_size as usize, &mut rng);
            res_ind
        })
        .collect::<Vec<Individual>>()
}

fn select_tournament(fits: &Vec<f32>, tour_size: u32, mut rng: &mut Rng) -> Vec<usize> {
    let range = Range::new(0, fits.len());
    let mut sel_inds: Vec<usize> = Vec::with_capacity(fits.len());  // vector of indices of selected inds. +1 in case of elite individual is used.
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

pub fn get_best_individual(popul: &Vec<Individual>) -> Individual {
    let min_fitness = popul.into_iter().fold(std::f32::MAX, |s, ref ind| if s < ind.fitness {s} else {ind.fitness});
    let idx = popul.into_iter().position(|ref x| x.fitness == min_fitness).expect("Min fitness is not found");
    popul[idx].clone()
}
