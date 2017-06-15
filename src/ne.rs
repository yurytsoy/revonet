use rand::{Rng};
use std;
use std::iter::FromIterator;
use std::rc::*;

use context::*;
use ea::*;
use ga::*;
use math::*;
use neproblem::*;
use neuro::*;
use problem::*;
use result::*;
use settings::*;

// #[derive(Clone)]
pub struct NEIndividual {
    genes: Vec<f32>,
    fitness: f32,
    network: Option<MultilayeredNetwork>,
}

#[allow(dead_code)]
impl<'a> NEIndividual {
    pub fn get_network(&'a self) -> Option<&'a MultilayeredNetwork> {
        match &self.network {
            &Some(ref x) => Some(x),
            &None => None
        }
    }
}

impl<'a> Individual for NEIndividual {
    fn new() -> NEIndividual {
        NEIndividual{
            genes: Vec::new(),
            fitness: std::f32::NAN,
            network: None,
        }
    }

    fn init<R: Rng>(&mut self, size: usize, mut rng: &mut R) {
        self.genes = rand_vector_std_gauss(size as usize, rng);
    }

    fn clone(&self) -> Self {
        NEIndividual{
            genes: self.genes.clone(),
            fitness: self.fitness,
            network: match &self.network {
                &Some(ref x) => Some(x.clone()),
                &None => None
            }
        }
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

//================================================================================

struct NE<'a, P: Problem + 'a, T: Individual> {
    ctx: Option<EAContext<T>>,
    problem: &'a P,
}

#[allow(dead_code)]
impl<'a, P: Problem, T: Individual> NE<'a, P, T> {
    pub fn new(problem: &'a P) -> NE<'a, P, T> {
        NE {problem: problem,
           ctx: None,
        }
    }

    pub fn run(&mut self, settings: EASettings) -> Result<Rc<&EAResult<T>>, ()> {
        let gen_count = settings.gen_count;
        let mut ctx = EAContext::new(settings);
        self.run_with_context(&mut ctx, self.problem, gen_count);
        self.ctx = Some(ctx);
        Ok(Rc::new(&(&self.ctx.as_ref().unwrap()).result))
    }
}

impl<'a, T: Individual, P: Problem> EA<'a, T> for NE<'a, P, T> {
    fn breed(&self, ctx: &mut EAContext<T>, sel_inds: &Vec<usize>, children: &mut Vec<T>) {
        cross(&ctx.population, sel_inds, children, ctx.settings.use_elite, ctx.settings.x_prob, ctx.settings.x_alpha, &mut ctx.rng);
        mutate(children, ctx.settings.mut_prob, &mut ctx.rng);
    }
}

//===================================================================

#[cfg(test)]
mod test {
    use rand;

    use math::*;
    use ne::*;
    use neproblem::*;
    use neuro::*;
    use settings::*;

    #[test]
    pub fn test_symbolic_regression() {
        let (pop_size, gen_count, param_count) = (20, 20, 100);
        let settings = EASettings::new(pop_size, gen_count, param_count);
        let problem = SymbolicRegressionProblem::new_f();

        let mut ne: NE<SymbolicRegressionProblem, NEIndividual> = NE::new(&problem);
        ne.run(settings);
        // let ne = NE::new(&problem);
    }
}