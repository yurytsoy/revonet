use rand;
use rand::{Rng, SeedableRng, StdRng};
use rand::distributions::{Normal, IndependentSample, Range};
use std;
use std::rc::Rc;

use context::*;
use ea::*;
use math::*;
use problem::*;
use result::*;
use settings::*;

pub struct GA<'a, P: Problem + 'a, T: Individual> {
    ctx: Option<EAContext<T>>,
    problem: &'a P,
}

impl<'a, P: Problem, T: Individual + 'a> GA<'a, P, T> {
    pub fn new(problem: &'a P) -> GA<P, T> {
        GA{problem: problem,
           ctx: None,
        }
    }

    pub fn run(&'a mut self, settings: EASettings) -> Result<Rc<&EAResult<T>>, ()> {
        let gen_count = settings.gen_count;
        let mut ctx = EAContext::new(settings, self.problem);
        self.run_with_context(&mut ctx, self.problem, gen_count);
        self.ctx = Some(ctx);
        Ok(Rc::new(&(&self.ctx.as_ref().unwrap()).result))
    }
}

impl<'a, P: Problem, T: Individual> EA<'a, T> for GA<'a, P, T> {
    fn breed(&self, ctx: &mut EAContext<T>, sel_inds: &Vec<usize>, children: &mut Vec<T>) {
        cross(&ctx.population, sel_inds, children, ctx.settings.use_elite, ctx.settings.x_prob, ctx.settings.x_alpha, &mut ctx.rng);
        mutate(children, ctx.settings.mut_prob, &mut ctx.rng);
    }

    // fn get_context_mut(&mut self) -> &'a mut EAContext<T> {
    //     match self.ctx.as_ref() {
    //         Some(ctx) => {&mut ctx},
    //         None => panic!("Context is empty")
    //     }
    //     // &mut self.ctx
    // }
}

pub fn cross<R: Rng, T: Individual>(popul: &Vec<T>, sel_inds: &Vec<usize>, children: &mut Vec<T>, use_elite: bool, x_prob: f32, x_alpha: f32, mut rng: &mut R) {
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

        let mut c1 = popul[sel_inds[p1]].clone();   // T::new();
        let mut c2 = popul[sel_inds[p2]].clone();   // T::new();
        if rand::random::<f32>() < x_prob {
            cross_blx_alpha(&popul[sel_inds[p1]], &popul[sel_inds[p2]], &mut c1, &mut c2, x_alpha, rng);
        }
        children.push(c1);
        children.push(c2);

        if children.len() >= popul.len() {break;}
    }
}

fn cross_blx_alpha<T: Individual>(p1: &T, p2: &T, c1: &mut T, c2: &mut T, alpha: f32, mut rng: &mut Rng) {
    let p1_genes = p1.to_vec().unwrap();
    let p2_genes = p2.to_vec().unwrap();
    // println!("{} : {}", p1_genes.len(), p2_genes.len());
    assert!(p1_genes.len() == p2_genes.len());

    let gene_count = p1_genes.len();
    let c1_genes = c1.to_vec_mut().unwrap();
    let c2_genes = c2.to_vec_mut().unwrap();

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
            c2_genes[k] = min_gene;
        }
    };
    // println!("children: {} : {}", c1_genes.len(), c2_genes.len());
}

pub fn mutate<T: Individual, R: Rng>(children: &mut Vec<T>, mut_prob: f32, rng: &mut R) {
    for k in 1..children.len() {
        mutate_gauss(&mut children[k], mut_prob, rng);
    }
}

fn mutate_gauss<T: Individual, R: Rng>(ind: &mut T, prob: f32, rng: &mut R) {
    let normal_rng = Normal::new(0.0, 0.1);
    let genes = ind.to_vec_mut().unwrap();
    for k in 0..genes.len() {
        if rand::random::<f32>() < prob {
            genes[k] += normal_rng.ind_sample(rng) as f32;
        }
    }
}
