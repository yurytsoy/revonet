use rand::{Rng};
use serde::de::{Deserialize, DeserializeOwned};
use serde::ser::Serialize;
use std;
use std::rc::*;

use context::*;
use ea::*;
use ga::*;
use math::*;
use neuro::*;
use problem::*;
use result::*;
use settings::*;

/// Represents individual for neuroevolution. The main difference is that the NE individual also
/// has a `network` field, which stores current neural network.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NEIndividual {
    genes: Vec<f32>,
    fitness: f32,
    network: Option<MultilayeredNetwork>,
}

#[allow(dead_code)]
impl<'a> NEIndividual {
    /// Update individual's network by assigning gene values to network's weights.
    fn update_net(&mut self) {
        match &mut self.network {
            &mut Some(ref mut net) => {
                // copy weights from genes to NN.
                let (mut ws, mut bs) = (Vec::new(), Vec::new());
                let mut inputs_num = net.get_inputs_count();
                let mut cur_idx = 0;
                // weights.
                for layer in net.iter_layers() {
                    // println!("getting slice for weights {}..{}", cur_idx, (cur_idx + inputs_num * layer.len()));
                    ws.push(Vec::from(&self.genes[cur_idx..(cur_idx + inputs_num * layer.len())]));
                    cur_idx += inputs_num * layer.len();
                    inputs_num = layer.len();
                }
                // biases.
                // cur_idx = 0;
                for layer in net.iter_layers() {
                    // println!("getting slice for biases {}..{}", cur_idx, (cur_idx + layer.len()));
                    bs.push(Vec::from(&self.genes[cur_idx..(cur_idx + layer.len())]));
                    cur_idx += layer.len();
                }
                net.set_weights(&ws, &bs);
            },
            &mut None => {
                println!("[update_net] Warning: network is not defined");
            }
        }
    }
}

impl Individual for NEIndividual {
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
        // println!("Cloning NE individual");
        NEIndividual{
            genes: self.genes.clone(),
            fitness: self.fitness,
            network: match &self.network {
                &Some(ref x) => Some(x.clone()),
                &None => {
                    println!("[clone] Warning: cloned individual has empty NN");
                    None
                }
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

    fn to_net(&mut self) -> Option<&MultilayeredNetwork> {
        self.update_net();
        match &self.network {
            &Some(ref net) => Some(net),
            &None => None
        }
    }

    fn to_net_mut(&mut self) -> Option<&mut MultilayeredNetwork> {
        self.update_net();
        match &mut self.network {
            &mut Some(ref mut net) => {Some(net)},
            &mut None => None
        }
    }

    fn set_net(&mut self, net: MultilayeredNetwork) {
        let (ws, bs) = net.get_weights();
        self.network = Some(net);
        self.genes = ws.into_iter()
                       .fold(Vec::new(), |mut res, w| {
                           res.extend(w.iter().cloned());
                           res
                       });
        self.genes.extend(bs.into_iter()
                            .fold(Vec::new(), |mut res, b| {
                                res.extend(b.iter().cloned());
                                res
                            }));
    }
}

//================================================================================

/// Structure for neuroevolutionary algorithm.
pub struct NE<'a, P: Problem + 'a, T: Individual+Deserialize<'a>+Serialize> {
    /// Context structure containing information about GA run, its progress and results.
    ctx: Option<EAContext<T>>,
    /// Reference to the objective function object implementing `Problem` trait.
    problem: &'a P,
}

#[allow(dead_code)]
impl<'a, P: Problem, T: Individual+Clone+DeserializeOwned+Serialize> NE<'a, P, T> {
    /// Create a new neuroevolutionary algorithm for the given problem.
    pub fn new(problem: &'a P) -> NE<'a, P, T> {
        NE {problem: problem,
           ctx: None,
        }
    }

    /// Run evolution of neural networks and return `EAResult` object.
    ///
    /// # Arguments:
    /// * `settings` - `EASettings` object.
    pub fn run(&mut self, settings: EASettings) -> Result<Rc<&EAResult<T>>, ()> {
        let gen_count = settings.gen_count;
        let mut ctx = EAContext::new(settings, self.problem);
        self.run_with_context(&mut ctx, self.problem, gen_count);
        self.ctx = Some(ctx);
        Ok(Rc::new(&(&self.ctx.as_ref().expect("Empty EAContext")).result))
    }
}

impl<'a, T: Individual+Serialize+Deserialize<'a>, P: Problem> EA<'a, T> for NE<'a, P, T> {
    fn breed(&self, ctx: &mut EAContext<T>, sel_inds: &Vec<usize>, children: &mut Vec<T>) {
        cross(&ctx.population, sel_inds, children, ctx.settings.use_elite, ctx.settings.x_type, ctx.settings.x_prob, ctx.settings.x_alpha, &mut ctx.rng);
        mutate(children, ctx.settings.mut_type, ctx.settings.mut_prob, ctx.settings.mut_sigma, &mut ctx.rng);
    }
}

//===================================================================

#[cfg(test)]
#[allow(unused_imports)]
mod test {
    use rand;

    use math::*;
    use ne::*;
    use neproblem::*;

    #[test]
    pub fn test_symbolic_regression() {
        let (pop_size, gen_count, param_count) = (20, 20, 100);
        let settings = EASettings::new(pop_size, gen_count, param_count);
        let problem = SymbolicRegressionProblem::new_f();

        let mut ne: NE<SymbolicRegressionProblem, NEIndividual> = NE::new(&problem);
        let res = ne.run(settings).expect("Error: NE result is empty");
        println!("result: {:?}", res);
        println!("\nbest individual: {:?}", res.best);
        // println!("\nbest individual NN: {:?}", res.best.to_net());
        // let ne = NE::new(&problem);
    }

    #[test]
    pub fn test_net_get_set() {
        let mut rng = rand::thread_rng();
        let mut net = MultilayeredNetwork::new(2, 2);
        net.add_hidden_layer(10 as usize, ActivationFunctionType::Sigmoid)
            .add_hidden_layer(5 as usize, ActivationFunctionType::Sigmoid)
            .build(&mut rng);
        let (ws1, bs1) = net.get_weights();

        let mut ind = NEIndividual::new();
        ind.set_net(net.clone());
        let net2 = ind.to_net_mut().unwrap();
        let (ws2, bs2) = net2.get_weights();

        // compare ws1 & ws2 and bs1 & bs2. Should be equal.
        for k in 0..ws1.len() {
            let max_diff = max(&sub(&ws1[k], &ws2[k]));
            println!("max diff: {}", max_diff);
            assert!(max_diff == 0f32);

            let max_diff = max(&sub(&bs1[k], &bs2[k]));
            println!("bs1: {:?}", bs1[k]);
            println!("bs2: {:?}", bs2[k]);
            println!("max diff: {}", max_diff);
            assert!(max_diff == 0f32);
        }
    }
}
