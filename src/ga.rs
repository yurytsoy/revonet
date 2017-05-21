use rand;
use rand::{Rng, SeedableRng, StdRng};
use rand::distributions::{Normal, IndependentSample, Range};
use std;

use context::*;
use math::*;
use problem::*;
use result::*;
use settings::*;

pub struct GA<'a> {
    ctx: Option<EAContext<'a>>,
    settings: GASettings,
    // population: Vec<Individual>,
    problem: &'a OptProblem,
    // result: Option<GAResult>,
}

impl<'a> GA<'a> {
    pub fn new(settings: GASettings, problem: &'a OptProblem) -> GA {
        let mut rng = StdRng::from_seed(&[settings.rng_seed as usize]);
        let pop_size = settings.pop_size;
        let param_count = settings.param_count;
        GA{settings: settings,
           problem: problem,
           ctx: None,
           }
    }

    pub fn run(&mut self, gen_count: u32) {
        self.ctx = Some(EAContext::new(&'a self.settings));
        let ctx: &mut EAContext = &mut self.ctx.expect("Failed to create EA context");
        let mut cur_result = &mut ctx.result;
        for t in 0..gen_count {
            // evaluation
            ctx.fitness = evaluate(&ctx.population, self.problem, cur_result);
            // TODO: the code below does not look very nice... Shouldn't it be inside `evaluate`? Fix using context.
            // for k in 0..ctx.population.len() {
            //     ctx.population[k].fitness = fits[k];
            // }

            // selection
            let sel_inds = select(&ctx.fitness, self.settings.tour_size, &mut ctx.rng);

            // crossover
            let mut children: Vec<Individual> = Vec::with_capacity(self.settings.pop_size as usize + 1);
            cross(&ctx.population, &sel_inds, &mut children, self.settings.use_elite, self.settings.x_prob, self.settings.x_alpha, &mut ctx.rng);

            // mutation
            mutate(&mut children, self.settings.mut_prob);

            // next gen
            ctx.population.clone_from(&children);
            ctx.population.truncate(self.settings.pop_size as usize);

            println!("> {} : {:?}", t, ctx.fitness);
            println!(" Best fitness at generation {} : {}\n", t, min(&ctx.fitness));
        }
        // ctx.result = Some(cur_result);
    }

    pub fn get_result(&self) -> Option<GAResult> {
        match self.ctx {
            Some(ctx) => Some(ctx.result.clone()),
            None => None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Individual {
    genes: Vec<f32>,
    fitness: f32,
}

impl Individual {
    pub fn new() -> Individual {
        Individual{genes: Vec::new(), fitness: std::f32::NAN}
    }
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

fn evaluate(popul: &Vec<Individual>, problem: &OptProblem, cur_result: &mut GAResult) -> Vec<f32> {
    let fits = popul.into_iter().map(|ind| problem.compute(&ind.genes)).collect::<Vec<f32>>();
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
    fits
}

fn select(fits: &Vec<f32>, tour_size: u32, mut rng: &mut Rng) -> Vec<usize> {
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

fn cross<T: Rng>(popul: &Vec<Individual>, sel_inds: &Vec<usize>, children: &mut Vec<Individual>, use_elite: bool, x_prob: f32, x_alpha: f32, mut rng: &mut T) {
    let range = Range::new(0, popul.len());
    if use_elite {
        children.push(get_best_individual(popul));
    }
    loop {
        // select parent individuals
        let p1: usize = range.ind_sample(rng);
        let mut p2: usize = range.ind_sample(rng);
        while p2 == p1 {
            p2 = range.ind_sample(rng);
        }

        if rand::random::<f32>() < x_prob {
            let mut c1 = Individual::new();
            let mut c2 = Individual::new();
            cross_blx_alpha(&popul[sel_inds[p1]], &popul[sel_inds[p2]], &mut c1, &mut c2, x_alpha, rng);
            children.push(c1);
            children.push(c2);
        } else {
            children.push(popul[sel_inds[p1]].clone());
            children.push(popul[sel_inds[p2]].clone());
        }

        if children.len() >= popul.len() {break;}
    }
}

fn cross_blx_alpha(p1: &Individual, p2: &Individual, c1: &mut Individual, c2: &mut Individual, alpha: f32, mut rng: &mut Rng) {
    assert!(p1.genes.len() == p2.genes.len());

    let gene_count = p1.genes.len();
    c1.genes = Vec::with_capacity(gene_count);
    c2.genes = Vec::with_capacity(gene_count);

    for k in 0..gene_count {
        let (min_gene, max_gene) = if p1.genes[k] > p2.genes[k] {(p2.genes[k], p1.genes[k])}
                                   else                         {(p1.genes[k], p2.genes[k])};
        if min_gene < max_gene {
            let delta = max_gene - min_gene;
            let gene_range = Range::new(min_gene - delta*alpha, max_gene + delta*alpha);
            c1.genes.push(gene_range.ind_sample(&mut rng));
            c2.genes.push(gene_range.ind_sample(&mut rng));
        } else {
            c1.genes.push(min_gene);
            c2.genes.push(min_gene);
        }
    };
}

fn mutate(children: &mut Vec<Individual>, mut_prob: f32) {
    for k in 1..children.len() {
        mutate_gauss(&mut children[k], mut_prob);
    }
}

fn mutate_gauss(ind: &mut Individual, prob: f32) {
    let normal_rng = Normal::new(0.0, 0.1);
    for k in 0..ind.genes.len() {
        if rand::random::<f32>() < prob {
            ind.genes[k] += normal_rng.ind_sample(&mut rand::thread_rng()) as f32;
        }
    }
}

fn get_best_individual(popul: &Vec<Individual>) -> Individual {
    let min_fitness = popul.into_iter().fold(std::f32::MAX, |s, ref ind| if s < ind.fitness {s} else {ind.fitness});
    let idx = popul.into_iter().position(|ref x| x.fitness == min_fitness).expect("Min fitness is not found");
    popul[idx].clone()
}

