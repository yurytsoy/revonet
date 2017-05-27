use rand;
use rand::{Rng, SeedableRng, StdRng};
use rand::distributions::{Normal, IndependentSample, Range};
use std;

use context::*;
use ea::*;
use math::*;
use problem::*;
use result::*;
use settings::*;

pub struct GA<'a> {
    ctx: EAContext,
    problem: &'a Problem,
}

impl<'a> GA<'a> {
    pub fn new(settings: EASettings, problem: &'a Problem) -> GA {
        GA{problem: problem,
           ctx: EAContext::new(settings),
        }
    }

    pub fn run(&mut self, gen_count: u32) -> Result<EAResult, ()> {
        let mut ctx = self.ctx.clone();
        let res = self.run_with_context(&mut ctx, self.problem, gen_count);
        self.ctx = ctx;
        res
    }
}

impl<'a> EA for GA<'a> {
    fn breed(&self, ctx: &mut EAContext, sel_inds: &Vec<usize>, children: &mut Vec<Individual>) {
        cross(&ctx.population, sel_inds, children, ctx.settings.use_elite, ctx.settings.x_prob, ctx.settings.x_alpha, &mut ctx.rng);
        mutate(children, ctx.settings.mut_prob);
    }
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

