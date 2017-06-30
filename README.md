# revonet

Rust implementation of real-coded genetic algorithm for solving optimization problems and training of neural networks. The latter is also known as neuroevolution.

This is an exercise project to get the hang of Rust and possibly do something useful. As my knowledge of Rust is quite limited, the code is far from perfection and any advice is very appreciated.

# Examples

### Real-coded genetic algorithm

```rust
let pop_size = 20u32;       // population size.
let problem_dim = 10u32;    // number of optimization parameters.

let problem = RosenbrockProblem{};  // objective function.
let gen_count = 10u32;      // generations number.
let settings = GASettings::new(pop_size, gen_count, problem_dim);
let mut ga = GA::new(settings, &problem);   // init GA.
let res = ga.run(settings).expect("Error during GA run");  // run and fetch the results.

// get and print results of the current run.
println!("\n\nGA results: {:?}", res);
```

### Run evolution of NN weights to solve regression problem

```rust
let (pop_size, gen_count, param_count) = (20, 20, 100); // gene_count does not matter here as NN structure is defined by a problem.
let settings = EASettings::new(pop_size, gen_count, param_count);
let problem = SymbolicRegressionProblem::new_f();

let mut ne: NE<SymbolicRegressionProblem, NEIndividual> = NE::new(&problem);
let res = ne.run(settings).expect("Error: NE result is empty");
println!("result: {:?}", res);
println!("\nbest individual: {:?}", res.best);
```

### Creating multilayered neural network with 2 hidden layers with sigmoid activation and with linear output nodes.

```rust
const INPUT_SIZE: usize = 20;
const OUTPUT_SIZE: usize = 2;

let mut rng = rand::thread_rng();   // needed for weights initialization when NN is built.
let mut net: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
net.add_hidden_layer(30 as usize, ActivationFunctionType::Sigmoid)
     .add_hidden_layer(20 as usize, ActivationFunctionType::Sigmoid)
     .build(&mut rng);       // `build` finishes creation of neural network.

let (ws, bs) = net.get_weights();   // `ws` and `bs` are `Vec` arrays containing weights and biases for each layer.
assert!(ws.len() == 3);		// number of elements equals to number of hidden layers + 1 output layer
assert!(bs.len() == 3);		// number of elements equals to number of hidden layers + 1 output layer

```

### Creating custom optimization problem for GA

```rust
// Dummy problem returning random fitness.
pub struct SphereProblem;
impl Problem for SphereProblem {
    // Function to evaluate a specific individual.
    fn compute<T: Individual>(&self, ind: &mut T) -> f32 {
        // use `to_vec` to get real-coded representation of an individual.
        let v = ind.to_vec().unwrap();

        let mut rng: StdRng = StdRng::from_seed(&[0]);
        rng.gen::<f32>()
    }
}
```

### Creating custom problem for NN evolution

```rust
// Dummy problem returning random fitness.
struct RandomNEProblem {}

impl NeuroProblem for RandomNEProblem {
    // return number of NN inputs.
    fn get_inputs_count(&self) -> usize {1}
    // return number of NN outputs.
    fn get_outputs_count(&self) -> usize {1}
    // return NN with random weights and a fixed structure. For now the structure should be the same all the time to make sure that crossover is possible. Likely to change in the future as I get more hang on Rust.
    fn get_default_net(&self) -> MultilayeredNetwork {
        let mut rng = rand::thread_rng();
        let mut net: MultilayeredNetwork = MultilayeredNetwork::new(self.get_inputs_count(), self.get_outputs_count());
        net.add_hidden_layer(5 as usize, ActivationFunctionType::Sigmoid)
            .build(&mut rng);
        net
    }

    // Function to evaluate performance of a given NN.
    fn compute_with_net<T: NeuralNetwork>(&self, nn: &mut T) -> f32 {
        let mut rng: StdRng = StdRng::from_seed(&[0]);

        let mut input = (0..self.get_inputs_count())
                            .map(|_| rng.gen::<f32>())
                            .collect::<Vec<f32>>();
        // compute NN output using random input.
        let mut output = nn.compute(&input);
        output[0]
    }
}

```
