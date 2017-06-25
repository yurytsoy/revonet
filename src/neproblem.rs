use rand;
use rand::{Rng, StdRng, SeedableRng};
use std;

use ea::*;
use math::*;
use ne::NEIndividual;
use neuro::{ActivationFunctionType, MultilayeredNetwork, NeuralNetwork};
use problem::*;

//--------------------------------------------

pub trait NeuroProblem: Problem {
    fn get_inputs_count(&self) -> usize;
    fn get_outputs_count(&self) -> usize;
    fn get_default_net(&self) -> MultilayeredNetwork;

    fn compute_with_net<T: NeuralNetwork>(&self, net: &mut T) -> f32;
}

impl<T: NeuroProblem> Problem for T {
    fn compute<I: Individual>(&self, ind: &mut I) -> f32 {
        let fitness;
        {
            let net: &mut MultilayeredNetwork = ind.to_net_mut().unwrap();
            fitness = self.compute_with_net(net);
        }
        ind.set_fitness(fitness);
        ind.get_fitness()
    }
}

///
/// Problems which are typically used to test GP algorithms.
/// See for details: Luke S. Essentials of metaheuristics.
///
#[allow(dead_code)]
pub struct SymbolicRegressionProblem {
    func: fn(&SymbolicRegressionProblem, f32) -> f32,
}

#[allow(dead_code)]
impl SymbolicRegressionProblem {
    pub fn new(problem_type: char) -> SymbolicRegressionProblem {
        match problem_type {
            'f' => SymbolicRegressionProblem::new_f(),
            'g' => SymbolicRegressionProblem::new_g(),
            'h' => SymbolicRegressionProblem::new_h(),
            _ => {
                panic!(format!("Unknown problem type for symbolic regression problem: {}",
                               problem_type))
            }
        }
    }

    pub fn new_f() -> SymbolicRegressionProblem {
        SymbolicRegressionProblem { func: SymbolicRegressionProblem::f }
    }

    pub fn new_g() -> SymbolicRegressionProblem {
        SymbolicRegressionProblem { func: SymbolicRegressionProblem::g }
    }

    pub fn new_h() -> SymbolicRegressionProblem {
        SymbolicRegressionProblem { func: SymbolicRegressionProblem::h }
    }

    fn f(&self, x: f32) -> f32 {
        let x2 = x * x;
        x2 * x2 + x2 * x + x2 + x
    }

    fn g(&self, x: f32) -> f32 {
        let x2 = x * x;
        x2 * x2 * x - 2f32 * x2 * x + x
    }

    fn h(&self, x: f32) -> f32 {
        let x2 = x * x;
        x2 * x2 * x2 - 2f32 * x2 * x2 + x2
    }
}

impl NeuroProblem for SymbolicRegressionProblem {
    fn get_inputs_count(&self) -> usize {
        1
    }
    fn get_outputs_count(&self) -> usize {
        1
    }
    fn get_default_net(&self) -> MultilayeredNetwork {
        let mut rng = rand::thread_rng();
        let mut net: MultilayeredNetwork = MultilayeredNetwork::new(self.get_inputs_count(), self.get_outputs_count());
        net.add_hidden_layer(5 as usize, ActivationFunctionType::Sigmoid)
            .build(&mut rng);
        net
    }

    fn compute_with_net<T: NeuralNetwork>(&self, nn: &mut T) -> f32 {
        const PTS_COUNT: u32 = 20;

        let mut er = 0f32;
        let mut input = vec![0f32];
        let mut output;

        let mut rng: StdRng = StdRng::from_seed(&[0]);
        for _ in 0..PTS_COUNT {
            let x = rng.gen::<f32>(); // sample from [-1, 1]
            let y = (self.func)(&self, x);

            input[0] = x;
            output = nn.compute(&input);

            er += (output[0] - y).abs();
        }
        er
    }
}

//=========================================================

#[test]
fn test_symb_regression_problem() {
    for prob_type in vec!['f', 'g', 'h'] {
        let prob = SymbolicRegressionProblem::new(prob_type);
        println!("Created problem of type: {}", prob_type);

        let mut net = prob.get_default_net();
        println!("Created default net with {} inputs, {} outputs, and {} hidden layers ", net.get_inputs_count(), net.get_outputs_count(), net.len()-1);
        println!("  Network weights: {:?}", net.get_weights());

        let input_size = net.get_inputs_count();
        let mut rng = rand::thread_rng();
        let mut ys = Vec::with_capacity(100);
        for _ in 0..100 {
            let x = rand_vector_std_gauss(input_size, &mut rng);
            let y = net.compute(&x);
            ys.push(y);
        }
        println!("  Network outputs for 100 random inputs: {:?}", ys);
        println!("  Network evaluation: {:?}\n", prob.compute_with_net(&mut net));
    }
}
