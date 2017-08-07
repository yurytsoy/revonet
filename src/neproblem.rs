use rand;
use rand::{Rng, StdRng, SeedableRng};

use ea::*;
use neuro::{ActivationFunctionType, MultilayeredNetwork, NeuralArchitecture, NeuralNetwork};
use problem::*;


//--------------------------------------------

/// Trait for problem where NN is a solution.
///
/// # Example: Custom NE problem
/// ```
/// extern crate revonet;
/// extern crate rand;
///
/// use rand::{Rng, SeedableRng, StdRng};
///
/// use revonet::ea::*;
/// use revonet::ne::*;
/// use revonet::neuro::*;
/// use revonet::neproblem::*;
///
/// // Dummy problem returning random fitness.
/// struct RandomNEProblem {}
///
/// impl RandomNEProblem {
///     fn new() -> RandomNEProblem {
///         RandomNEProblem{}
///     }
/// }
///
/// impl NeuroProblem for RandomNEProblem {
///     // return number of NN inputs.
///     fn get_inputs_num(&self) -> usize {1}
///     // return number of NN outputs.
///     fn get_outputs_num(&self) -> usize {1}
///     // return NN with random weights and a fixed structure. For now the structure should be the same all the time to make sure that crossover is possible. Likely to change in the future.
///     fn get_default_net(&self) -> MultilayeredNetwork {
///         let mut rng = rand::thread_rng();
///         let mut net: MultilayeredNetwork = MultilayeredNetwork::new(self.get_inputs_num(), self.get_outputs_num());
///         net.add_hidden_layer(5 as usize, ActivationFunctionType::Sigmoid)
///             .build(&mut rng, NeuralArchitecture::Multilayered);
///         net
///     }
///
///     // Function to evaluate performance of a given NN.
///     fn compute_with_net<T: NeuralNetwork>(&self, nn: &mut T) -> f32 {
///         let mut rng: StdRng = StdRng::from_seed(&[0]);
///
///         let mut input = (0..self.get_inputs_num())
///                             .map(|_| rng.gen::<f32>())
///                             .collect::<Vec<f32>>();
///         // compute NN output using random input.
///         let mut output = nn.compute(&input);
///         output[0]
///     }
/// }
///
/// fn main() {}
/// ```
pub trait NeuroProblem: Problem {
    /// Number of input variables.
    fn get_inputs_num(&self) -> usize;
    /// Number of output (target) variables.
    fn get_outputs_num(&self) -> usize;
    /// Returns random network with default number of inputs and outputs and some predefined structure.
    ///
    /// For now all networks returned by implementation of this functions have the same structure and
    /// random weights. This was done to ensure possibility to cross NN's and might change in the future.
    fn get_default_net(&self) -> MultilayeredNetwork;

    /// Compute fitness value for the given neural network.
    ///
    /// # Arguments:
    /// * `net` - neural network to compute fitness for.
    fn compute_with_net<T: NeuralNetwork>(&self, net: &mut T) -> f32;
}

/// Default implementation of the `Problem` trait for `NeuroProblem`
#[allow(unused_variables, unused_mut)]
impl<T: NeuroProblem> Problem for T {
    fn compute<I: Individual>(&self, ind: &mut I) -> f32 {
        let fitness;
        fitness = self.compute_with_net(ind.to_net_mut().expect("Can not extract mutable ANN"));
        // match ind.to_net_mut() {
        //     Some(ref mut net) => {fitness = self.compute_with_net(net);},
        //     None => panic!("NN is not defined"),
        // };
        ind.set_fitness(fitness);
        ind.get_fitness()
    }
    fn get_random_individual<U: Individual, R: Rng>(&self, size: usize, mut rng: &mut R) -> U {
        let mut res_ind = U::new();
        res_ind.set_net(self.get_default_net());
        res_ind
    }
}

///
/// Classical noiseless XOR problem with 2 binary inputs and 1 output.
///
#[allow(dead_code)]
pub struct XorProblem {}

#[allow(dead_code)]
impl XorProblem {
    pub fn new() -> XorProblem {
        XorProblem{}
    }
}

#[allow(dead_code)]
impl NeuroProblem for XorProblem {
    fn get_inputs_num(&self) -> usize {2}
    fn get_outputs_num(&self) -> usize {1}
    fn get_default_net(&self) -> MultilayeredNetwork {
        let mut rng = rand::thread_rng();
        let mut net: MultilayeredNetwork = MultilayeredNetwork::new(self.get_inputs_num(), self.get_outputs_num());
        net.add_hidden_layer(4 as usize, ActivationFunctionType::Sigmoid)
            .build(&mut rng, NeuralArchitecture::BypassInputs);
            // .build(&mut rng, NeuralArchitecture::BypassInputs);
        net
    }

    fn compute_with_net<T: NeuralNetwork>(&self, nn: &mut T) -> f32 {
        let mut er = 0f32;

        let output = nn.compute(&[0f32, 0f32]);
        er += output[0] * output[0];
        let output = nn.compute(&[1f32, 1f32]);
        er += output[0] * output[0];
        let output = nn.compute(&[0f32, 1f32]);
        er += (1f32-output[0]) * (1f32-output[0]);
        let output = nn.compute(&[1f32, 0f32]);
        er += (1f32-output[0]) * (1f32-output[0]);

        er
    }
}

///
/// Problem which is typically used to test GP algorithms. Represents symbolic regression with
/// 1 input and 1 output. There are three variants:
/// * `f` - 4-th order polynomial.
/// * `g` - 5-th order polynomial.
/// * `h` - 6-th order polynomial.
///
/// See for details: Luke S. Essentials of metaheuristics.
///
#[allow(dead_code)]
pub struct SymbolicRegressionProblem {
    func: fn(&SymbolicRegressionProblem, f32) -> f32,
}

#[allow(dead_code)]
impl SymbolicRegressionProblem {
    /// Create a new problem depending on the problem type:
    /// * `f` - 4-th order polynomial.
    /// * `g` - 5-th order polynomial.
    /// * `h` - 6-th order polynomial.
    ///
    /// # Arguments:
    /// * `problem_type` - symbol from set `('f', 'g', 'h')` to set the problem type.
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

    /// Create `f`-type problem (4-th order polynomial)
    pub fn new_f() -> SymbolicRegressionProblem {
        SymbolicRegressionProblem { func: SymbolicRegressionProblem::f }
    }

    /// Create `g`-type problem (4-th order polynomial)
    pub fn new_g() -> SymbolicRegressionProblem {
        SymbolicRegressionProblem { func: SymbolicRegressionProblem::g }
    }

    /// Create `h`-type problem (4-th order polynomial)
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
    fn get_inputs_num(&self) -> usize { 1 }
    fn get_outputs_num(&self) -> usize { 1 }
    fn get_default_net(&self) -> MultilayeredNetwork {
        let mut rng = rand::thread_rng();
        let mut net: MultilayeredNetwork = MultilayeredNetwork::new(self.get_inputs_num(), self.get_outputs_num());
        net.add_hidden_layer(5 as usize, ActivationFunctionType::Sigmoid)
            .build(&mut rng, NeuralArchitecture::Multilayered);
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

#[cfg(test)]
#[allow(unused_imports)]
mod test {
    use rand;

    use math::*;
    use ne::*;
    use neproblem::*;
    use problem::*;
    use settings::*;

    #[test]
    fn test_xor_problem() {
        let (pop_size, gen_count, param_count) = (20, 20, 100); // gene_count does not matter here as NN structure is defined by a problem.
        let settings = EASettings::new(pop_size, gen_count, param_count);
        let problem = XorProblem::new();

        let mut ne: NE<XorProblem> = NE::new(&problem);
        let res = ne.run(settings).expect("Error: NE result is empty");
        println!("result: {:?}", res);
        println!("\nbest individual: {:?}", res.best);
    }

    #[test]
    fn test_symb_regression_problem() {
        for prob_type in vec!['f', 'g', 'h'] {
            let mut rng = rand::thread_rng();
            let prob = SymbolicRegressionProblem::new(prob_type);
            println!("Created problem of type: {}", prob_type);

            let mut net = prob.get_default_net();
            println!("Created default net with {} inputs, {} outputs, and {} hidden layers ", net.get_inputs_num(), net.get_outputs_num(), net.len()-1);
            println!("  Network weights: {:?}", net.get_weights());
            let mut ind: NEIndividual = prob.get_random_individual(0, &mut rng);
            println!("  Random individual: {:?}", ind.to_vec().unwrap());
            println!("  Random individual ANN: {:?}", ind.to_net().unwrap());

            let input_size = net.get_inputs_num();
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
}
