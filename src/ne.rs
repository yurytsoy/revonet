use rand::{Rng};
use std;
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

#[derive(Clone)]
struct NEIndividual<T: NeuralNetwork> {
    genes: Vec<f32>,
    fitness: f32,
    network: Option<T>,
}

impl<T: NeuralNetwork> NEIndividual<T> where T: std::clone::Clone {
    pub fn from_layers<R: Rng>(layers: Option<&[u32]>, rng: &mut R) -> NEIndividual<T> {
        NEIndividual{
            genes: Vec::new(),
            fitness: std::f32::NAN,
            network: match layers {
                Some(x) => Some(MultilayeredNetwork::from_layers(x, rng)),
                None => None,
            }
        }
    }

    pub fn get_network(&self) -> Option<&T> {
        Rc::new(&self.network.unwrap())
    }
}

impl<T: NeuralNetwork> Individual for NEIndividual<T> where T: std::clone::Clone {
    fn new() -> NEIndividual<T> {
        NEIndividual{
            genes: Vec::new(),
            fitness: std::f32::NAN,
            network: None,
        }
    }

    fn init<R: Rng>(&mut self, size: usize, mut rng: &mut R) {
        self.genes = rand_vector_std_gauss(size as usize, rng);
    }

    fn get_fitness(&self) -> f32 {
        self.fitness
    }

    fn set_fitness(&mut self, fitness: f32) {
        self.fitness = fitness;
    }

    fn get_genes(&self) -> &[f32] {
        &self.genes
    }

    fn get_genes_mut(&mut self) -> &mut Vec<f32> {
        &mut self.genes
    }
}

//================================================================================

struct NE<'a, P: Problem + 'a, T: Individual> {
    ctx: Option<EAContext<T>>,
    problem: &'a P,
}

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

        let ne: NE<SymbolicRegressionProblem, NEIndividual> = NE::new(&problem);
    }
}