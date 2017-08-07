use rand;
use rand::{Rng};
//use serde::de::{Deserialize};
//use serde::ser::{Serialize};
use std::fmt::Debug;
use std::slice::Iter;

use math::*;

/// Trait to generalize neural network behaviour.
#[allow(dead_code, unused_variables)]
pub trait NeuralNetwork : Clone {
    /// Compute output of neural network for a given input vector.
    ///
    /// # Arguments:
    /// * `xs` - input vector
    fn compute(&mut self, xs: &[f32]) -> Vec<f32>;
    /// Returns number of input nodes.
    fn get_inputs_num(&self) -> usize;
    /// Returns number of output nodes.
    fn get_outputs_num(&self) -> usize;
}

/// Representation of multilayered neural network with linear activation on output and arbitrary
/// activations for hidden layers.
#[derive(Debug, Deserialize, Serialize)]
#[allow(dead_code)]
pub struct MultilayeredNetwork {
    /// Number of input nodes.
    inputs_num: usize,
    /// Number of output nodes.
    outputs_num: usize,
    /// Vector of layers.
    layers: Vec<Box<NeuralLayer>>,
    /// Type of the neural architecture.
    arch: NeuralArchitecture,
    /// Flag to denote whether the network is built and initialized.
    is_built: bool,
}


/// Enumeration for the different types of neural architectures.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum NeuralArchitecture {
    /// Standard multilayered feed-forward network without skip connections,
    Multilayered,
    /// Multilayered NN where every layer is connected to the input layer.
    BypassInputs,
    /// Multilayered NN where inputs of current layere are propagated forward to the next layer.
    /// So that every layere is connected to the inputs of the previous layer.
    BypassLayer,
    /// Multilayered NN where every layere is connected to the outputs of all previous layers, including the input.
    BypassFull
}

#[allow(dead_code)]
impl MultilayeredNetwork {
    /// Create a new multilayered network object with given number of inputs and outputs.
    ///
    /// The resulting network is not initialized. To add hidden layeres use `add_hidden_layer` and
    /// then call `build` function to finalize structure and weights initialization.
    ///
    /// # Arguments:
    /// * `inputs_num` - number of input nodes.
    /// * `outputs_num` - number of output nodes.
    pub fn new(inputs_num: usize, outputs_num: usize) -> MultilayeredNetwork {
        MultilayeredNetwork{
            inputs_num: inputs_num,
            outputs_num: outputs_num,
            layers: Vec::new(),
            arch: NeuralArchitecture::Multilayered,
            is_built: false,
        }
    }

    /// Create and initialize a new neural network object given sizes of layers (including
    /// input and output ones) and activation types.
    ///
    /// The resulting network will have a linear activation in the output layer.
    ///
    /// # Arguments:
    /// * `layers` - array containing sizes of input (`layers[0]`), output (`layers[layers.len()-1]`)
    ///              and hidden layers (`layers[1:layers.len()-1]`).
    /// * `acts` - array of activations for hidden layers. Note that `k`-th hidden layers will
    ///            have `k`-th activation!
    pub fn from_layers<R: Rng+Sized>(layers: &[u32], acts: &[ActivationFunctionType], arch: NeuralArchitecture, rng: &mut R) -> MultilayeredNetwork {
        assert!(layers.len() >= 2);

        let mut res = MultilayeredNetwork::new(layers[0] as usize, layers[layers.len()-1] as usize);
        for k in 1..(layers.len()-1) {
            res.add_hidden_layer(layers[k] as usize, acts[k]);
        }
        res.build(rng, arch);
        res
    }

    /// Number of hidden+output layers in the neural network.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Add a hidden layer to an uninitialized network.
    ///
    /// Panics if the network has already been initialized .
    ///
    /// # Arguments:
    /// * `size` - number of nodes in a layer.
    /// * `actf` - type of activation function.
    pub fn add_hidden_layer(&mut self, size: usize, actf: ActivationFunctionType) -> &mut Self {
        if self.is_built {
            panic!("Can not add layer to already built network.");
        }
        self.layers.push(Box::new(NeuralLayer::new(size, actf)));
        self
    }

    /// Finalize building neural network and initialize its weights.
    ///
    /// # Arguments:
    /// * `rng` - mutable reference to the external RNG.
    pub fn build<R: Rng+Sized>(&mut self, rng: &mut R, arch: NeuralArchitecture) {
        if self.is_built {
            panic!("The network has already been built.");
        }

        self.arch = arch;

        // add output layer.
        self.layers.push(Box::new(NeuralLayer::new(self.outputs_num, ActivationFunctionType::Linear)));

        // init weights and biases for all layers.
        let mut inputs = self.inputs_num;
        let mut prev_outputs = self.inputs_num;
        for l in self.layers.iter_mut() {
            l.init_weights(inputs, rng);
            match self.arch {
                NeuralArchitecture::Multilayered => {inputs = l.len();}
                NeuralArchitecture::BypassInputs => {inputs = l.len() + self.inputs_num;}
                NeuralArchitecture::BypassLayer => {inputs = l.len() + prev_outputs;}
                NeuralArchitecture::BypassFull => {inputs = l.len() + inputs;}
            }
            prev_outputs = l.len();
        }
        self.is_built = true;
    }

    /// Returns tuple containing weights and biases layer by layer.
    ///
    /// # Return:
    /// * `result.0` - matrix of weights. `k`-th row corresponds to the flattened vector of weights
    ///                 for `k`-th layer.
    /// * `result.1` - matrix of biases.  `k`-th row corresponds to the vector of biases
    ///                 for `k`-th layer.
    ///
    /// # Example: Multilayered network with sigmoid activations:
    /// ```
    /// extern crate rand;
    /// use rand::{Rng};
    ///
    /// extern crate revonet;
    /// use revonet::neuro::*;
    ///
    /// fn main () {
    ///     const INPUT_SIZE: usize = 20;
    ///     const OUTPUT_SIZE: usize = 2;
    ///
    ///     let mut rng = rand::thread_rng();   // needed for weights initialization when NN is built.
    ///     let mut net: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
    ///     net.add_hidden_layer(30 as usize, ActivationFunctionType::Sigmoid)
    ///        .add_hidden_layer(20 as usize, ActivationFunctionType::Sigmoid)
    ///        .build(&mut rng, NeuralArchitecture::Multilayered);       // `build` finishes creation of neural network.
    ///
    ///     let (ws, bs) = net.get_weights();   // `ws` and `bs` are `Vec` arrays containing weights and biases for each layer.
    ///     assert!(ws.len() == 3);     // number of elements equals to number of hidden layers + 1 output layer
    ///     assert!(bs.len() == 3);     // number of elements equals to number of hidden layers + 1 output layer
    ///
    ///     let rnd_input = (0..INPUT_SIZE).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
    ///     println!("NN outputs: {:?}", net.compute(&rnd_input));
    /// }
    /// ```
    pub fn get_weights(&self) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        if !self.is_built {
            panic!("Can not retreive network weights: the network is not built yet.");
        }

        let mut res_w = Vec::new();
        let mut res_b = Vec::new();
        for l in self.layers.iter() {
            let (cur_w, cur_b) = l.get_weights();
            res_w.push(cur_w);
            res_b.push(cur_b);
        }
        (res_w, res_b)
    }

    /// Sets weights and biases layer by layer.
    ///
    /// See description  for the `get_weights` for more details and example.
    ///
    /// # Arguments:
    /// * `wss` - matrix of weights. `wss[k]` flattened vector of weights for the `k`-th layer.
    /// * `bss` - matrix of biases. `bss[k]` flattened vector of biases for the `k`-th layer.
    pub fn set_weights(&mut self, wss: &Vec<Vec<f32>>, bss: &Vec<Vec<f32>>) {
        if !self.is_built {
            panic!("Can not set network weights: the network is not built yet.");
        }

        for k in 0..self.layers.len() {
            self.layers[k].set_weights(&wss[k], &bss[k]);
        }
    }

    /// Returns iterator for layers.
    pub fn iter_layers(&self) -> Iter<Box<NeuralLayer>> {
        self.layers.iter()
    }
}

impl NeuralNetwork for MultilayeredNetwork {
    fn get_inputs_num(&self) -> usize {
        self.inputs_num
    }
    fn get_outputs_num(&self) -> usize {
        self.outputs_num
    }
    fn compute(&mut self, xs: &[f32]) -> Vec<f32> {
        // println!("compute");
        let mut input = xs;
        let mut prev_outputs: Vec<f32> = Vec::new();
        let mut layer_idx = 0;
        for l in self.layers.iter_mut() {
            match self.arch {
                NeuralArchitecture::Multilayered => {
                    // println!("layer input size: {}", input.len());
                    l.compute(input);
                }
                NeuralArchitecture::BypassInputs => {
                    if layer_idx > 0 {
                        let mut cur_input = Vec::from(input);
                        cur_input.extend(xs);
                        // println!("layer input size: {}", cur_input.len());
                        l.compute(&cur_input);
                    } else {
                        // println!("layer input size: {}", input.len());
                        l.compute(input);
                    }
                }
                NeuralArchitecture::BypassLayer => {
                    let mut cur_input = Vec::from(input);
                    cur_input.extend(prev_outputs.iter());
                    // println!("layer input size: {}", cur_input.len());
                    l.compute(&cur_input);
                    prev_outputs = Vec::from(input);
                }
                NeuralArchitecture::BypassFull => {
                    let mut cur_input: Vec<f32> = Vec::from(input);
                    cur_input.extend(prev_outputs.iter());
                    // println!("layer input size: {}", cur_input.len());
                    l.compute(&cur_input);
                    prev_outputs.extend(input);
                }
            }
            input = l.get_outputs();
            layer_idx += 1;
        }
        Vec::from(input)
    }
}

impl Clone for MultilayeredNetwork {
    fn clone(&self) -> Self {
        let mut res = MultilayeredNetwork::new(self.inputs_num, self.outputs_num);
        for k in 0..self.layers.len()-1 {
            res.add_hidden_layer(self.layers[k].size, self.layers[k].activation);
        }
        res.build(&mut rand::thread_rng(), self.arch);
        let (wss, bss) = self.get_weights(); 
        res.set_weights(&wss, &bss);
        res
    }
}

//========================================

/// Structure to describe a layer for neural network.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[allow(dead_code)]
pub struct NeuralLayer {
    /// Number of nodes.
    size: usize,
    /// Number of inputs.
    inputs_num: usize,
    /// Matrix of weights. `weights[k]` is a vector of weights for the `k`-th node.
    weights: Vec<Vec<f32>>,
    /// Vector of biases. `biases[k]` is a biases for the `k`-th node.
    biases: Vec<f32>,
    /// Vector of outputs for nodes. `outputs[k]` contains output for the `k`-th node.
    outputs: Vec<f32>,
    /// Type of activation function for every node in the layer.
    activation: ActivationFunctionType,
}

#[allow(dead_code)]
impl NeuralLayer {
    /// Create a new layer with given size and activation function.
    ///
    /// # Arguments:
    /// * `size` - number of nodes.
    /// * `actf` - type of activation function.
    pub fn new(size: usize, actf: ActivationFunctionType) -> NeuralLayer {
        NeuralLayer{
            size: size,
            inputs_num: 0usize,
            weights: Vec::new(),
            biases: Vec::new(),
            outputs: Vec::new(),
            activation: actf,
        }
    }

    /// Initializes weights of the layer.
    ///
    /// # Arguments:
    /// * `inputs_num` - number of inputs for the layer.
    /// * `rng` - mutable reference to the external RNG.
    pub fn init_weights<R: Rng + Sized>(&mut self, inputs_num: usize, rng: &mut R) {
        self.inputs_num = inputs_num;
        for _ in 0..self.size {
            self.weights.push(rand_vector_std_gauss(inputs_num, rng));
        }
        self.biases = rand_vector_std_gauss(self.size, rng);
    }

    /// Compute output of the layer given a vector of input signals.
    /// To get outputs use `get_outputs` function.
    ///
    /// # Arguments:
    /// * `xs` -- input vector.
    pub fn compute(&mut self, xs: &[f32]){
        self.outputs = dot_mv(&self.weights, &xs).iter().zip(self.biases.iter())
                            .map(|(&w, &b)| w+b)
                            .collect::<Vec<f32>>();
        compute_activations_inplace(&mut self.outputs, self.activation);
    }

    /// Returns number of inputs.
    pub fn get_inputs_num(&self) -> usize {
        self.inputs_num
    }

    /// Return number of nodes in the layer.
    pub fn len(&self) -> usize {self.size}

    /// Returns references to the slice, containing values of current outputs.
    pub fn get_outputs(&self) -> &[f32] {
        &self.outputs
    }

    /// Return flattened vector of weights and biases.
    pub fn get_weights(&self) -> (Vec<f32>, Vec<f32>) {
        let mut res_w = Vec::new();
        let res_b: Vec<f32> = Vec::from(&self.biases[0..self.size]);
        
        for k in 0..self.size {
            res_w.extend(self.weights[k].clone());
        }
        (res_w, res_b)
    }

    /// Set weights and biases for the layer.
    ///
    /// # Arguments:
    /// * `ws` - vector of flattened weights.
    /// * `bs` - vector of biases.
    pub fn set_weights(&mut self, ws: &Vec<f32>, bs: &Vec<f32>) {
        let inputs = self.weights[0].len();
        for k in 0..self.size {
            self.weights[k] = Vec::from(&ws[k*inputs..(k+1)*inputs]);
            self.biases[k] = bs[k];
        }
    }
}

//========================================

/// Enumeration for the different types of activations.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum ActivationFunctionType {
    Linear,
    Sigmoid,
    Relu,
}

/// Trait to generalize behaviour of activation function.
pub trait ActivationFunction: Debug {
    /// Compute activation function value using given argument.
    /// Implementation should call `ActivationFunction::compute_static` to avoid duplication.
    ///
    /// # Arguments:
    /// * `x` - value of argument of activation function.
    fn compute(&self, x: f32) -> f32;

    /// Compute activation function value using given argument.
    /// This is a static variant of the function `compute`.
    ///
    /// # Arguments:
    /// * `x` - value of argument of activation function.
    fn compute_static(x: f32) -> f32;

    /// Construct an activation function instance.
    fn new() -> Self;
}

#[derive(Debug)]
pub struct LinearActivation;
impl ActivationFunction for LinearActivation {
    fn new() -> LinearActivation {LinearActivation{}}
    fn compute(&self, x: f32) -> f32 {x}
    fn compute_static(x: f32) -> f32 {x}
}

#[derive(Debug)]
pub struct SigmoidActivation;
impl ActivationFunction for SigmoidActivation {
    fn new() -> SigmoidActivation {SigmoidActivation{}}
    fn compute(&self, x: f32) -> f32 {
        SigmoidActivation::compute_static(x)
    }
    fn compute_static(x: f32) -> f32 {
        1f32 / (1f32 + (-x).exp())
    }
}

#[derive(Debug)]
pub struct ReluActivation;

impl ActivationFunction for ReluActivation {
    fn new() -> ReluActivation {ReluActivation{}}
    fn compute(&self, x: f32) -> f32 {
        ReluActivation::compute_static(x)
    }
    fn compute_static(x: f32) -> f32 {
        if x > 0f32 {x} else {0f32}
    }
}

/// Computes activation function outputs by using array of arguments.
/// The result is written into the input array.
///
/// # Arguments:
/// * `xs` - array of arguments for vector of activation functions.
/// * `actf` - element of the `ActivationFunctionType` denoting type of activation.
pub fn compute_activations_inplace(xs: &mut [f32], actf: ActivationFunctionType) {
    let actf_ptr: fn(f32) -> f32;
    match actf {
        ActivationFunctionType::Linear => {return;},
        ActivationFunctionType::Relu => {actf_ptr = ReluActivation::compute_static;},
        ActivationFunctionType::Sigmoid => {actf_ptr = SigmoidActivation::compute_static;},
    };
    for k in 0..xs.len() {
        xs[k] = (actf_ptr)(xs[k]);
    }
}

//========================================

#[cfg(test)]
mod test {
    use rand;
    use rand::{StdRng, SeedableRng};

    use math::*;
    use neuro::*;

    #[test]
    fn test_linear_activation() {
        const LEN: usize = 100;
        let mut rng = rand::thread_rng();
        let xs = rand_vector_std_gauss(LEN, &mut rng);
        let act_f = LinearActivation{};
        assert!((0..LEN).all(|k| xs[k] == act_f.compute(xs[k])));
    }

    #[test]
    fn test_sigmoid_activation() {
        const LEN: usize = 100;
        let mut rng: StdRng = StdRng::from_seed(&[0 as usize]);
        let xs = rand_vector_std_gauss(LEN, &mut rng);
        let act_f = SigmoidActivation{};
        assert!((0..LEN).all(|k| {
            let y = act_f.compute(xs[k]);
            // use inverse function cnad check result wrt numeric instability
            (xs[k] - (y / (1f32 - y)).ln()) < 1e-6f32
        }));
    }

    #[test]
    fn test_linear_net() {
        const INPUT_SIZE: usize = 20;
        const OUTPUT_SIZE: usize = 2;

        let mut rng: StdRng = StdRng::from_seed(&[0 as usize]);
        let mut net_linear: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
        net_linear.add_hidden_layer(30 as usize, ActivationFunctionType::Linear)
                  .build(&mut rng, NeuralArchitecture::Multilayered);

        // outputs are not zeroes due to random biases.
        let net_out = net_linear.compute(&[0f32; INPUT_SIZE]);
        assert!(net_out.iter().all(|&x| x != 0f32));
        let net_out = net_linear.compute(&rand_vector_std_gauss(INPUT_SIZE, &mut rng));
        assert!(net_out.iter().all(|&x| x != 0f32));

        // reset weights => output should be 0.
        let mut zero_w = Vec::new();
        zero_w.push(vec![0f32; INPUT_SIZE * 30]);
        zero_w.push(vec![0f32; OUTPUT_SIZE * 30]);
        let mut zero_b = Vec::new();
        zero_b.push(vec![0f32; 30]);
        zero_b.push(vec![0f32; OUTPUT_SIZE]);
        net_linear.set_weights(&zero_w, &zero_b);

        let net_out = net_linear.compute(&[0f32; INPUT_SIZE]);
        assert!(net_out.iter().all(|&x| x == 0f32));
        let net_out = net_linear.compute(&rand_vector_std_gauss(INPUT_SIZE, &mut rng));
        assert!(net_out.iter().all(|&x| x == 0f32));
    }

    #[test]
    fn test_multilayer_net() {
        const INPUT_SIZE: usize = 20;
        const OUTPUT_SIZE: usize = 2;

        let mut rng: StdRng = StdRng::from_seed(&[0 as usize]);
        let mut net: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
        net.add_hidden_layer(30 as usize, ActivationFunctionType::Sigmoid)
            .add_hidden_layer(15 as usize, ActivationFunctionType::Linear)
            .add_hidden_layer(300 as usize, ActivationFunctionType::Sigmoid)
            .build(&mut rng, NeuralArchitecture::Multilayered);

        // net2 has the same structure and weights as net1.
        let mut net2: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
        net2.add_hidden_layer(30 as usize, ActivationFunctionType::Sigmoid)
            .add_hidden_layer(15 as usize, ActivationFunctionType::Linear)
            .add_hidden_layer(300 as usize, ActivationFunctionType::Sigmoid)
            .build(&mut rng, NeuralArchitecture::Multilayered);
        let (ws, bs) = net.get_weights();
        println!("{:?}", net.get_weights());
        assert!(ws.len() == 4);
        net2.set_weights(&ws, &bs);

        // net and net2 should produce identical outputs
        for _ in 0..100 {
            let x = rand_vector_std_gauss(INPUT_SIZE, &mut rng);
            let out1 = net.compute(&x);
            let out2 = net2.compute(&x);
            assert!(out1.iter().zip(out2.iter()).all(|(x1, x2)| x1 == x2));
        }
    }

    #[test]
    fn test_bypass_net() {
        const INPUT_SIZE: usize = 2;
        const OUTPUT_SIZE: usize = 2;
        let all_layer_sizes = [INPUT_SIZE, 3_usize, 5_usize, 3_usize, OUTPUT_SIZE];

        for arch in vec![NeuralArchitecture::Multilayered, NeuralArchitecture::BypassInputs, NeuralArchitecture::BypassLayer, NeuralArchitecture::BypassFull] {
            let mut rng: StdRng = StdRng::from_seed(&[0 as usize]);
            let mut net: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
            net.add_hidden_layer(all_layer_sizes[1], ActivationFunctionType::Sigmoid)
                .add_hidden_layer(all_layer_sizes[2], ActivationFunctionType::Linear)
                .add_hidden_layer(all_layer_sizes[3], ActivationFunctionType::Sigmoid)
                .build(&mut rng, arch);

            // net2 has the same structure and weights as net1.
            let mut net2: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
            net2.add_hidden_layer(all_layer_sizes[1], ActivationFunctionType::Sigmoid)
                .add_hidden_layer(all_layer_sizes[2], ActivationFunctionType::Linear)
                .add_hidden_layer(all_layer_sizes[3], ActivationFunctionType::Sigmoid)
                .build(&mut rng, arch);
            let (ws, bs) = net.get_weights();
            println!("Current architecture: {:?}", arch);
            for layer_idx in 0..ws.len() {
                if layer_idx == 0 {
                    assert!(ws[0].len() == all_layer_sizes[1] * INPUT_SIZE);
                    continue;
                }

                match arch {
                    NeuralArchitecture::Multilayered => {
                        assert!(ws[layer_idx].len() == all_layer_sizes[layer_idx+1] * all_layer_sizes[layer_idx]);
                    },
                    NeuralArchitecture::BypassInputs => {
                        assert!(ws[layer_idx].len() == all_layer_sizes[layer_idx+1] * (all_layer_sizes[layer_idx] + INPUT_SIZE));
                    },
                    NeuralArchitecture::BypassLayer => {
                        assert!(ws[layer_idx].len() == all_layer_sizes[layer_idx+1] * (all_layer_sizes[layer_idx] + all_layer_sizes[layer_idx-1]));
                    },
                    NeuralArchitecture::BypassFull => {
                        assert!(ws[layer_idx].len() == all_layer_sizes[layer_idx+1] * (0..(layer_idx+1)).fold(0_usize, |s, k| s + all_layer_sizes[k]));
                    },
                }
                // println!("  [{:?}]", w.len());
            }
            assert!(ws.len() == 4);
            net2.set_weights(&ws, &bs);

            // net and net2 should produce identical outputs
            for _ in 0..100 {
                let x = rand_vector_std_gauss(INPUT_SIZE, &mut rng);
                let out1 = net.compute(&x);
                let out2 = net2.compute(&x);
                assert!(out1.iter().zip(out2.iter()).all(|(x1, x2)| x1 == x2));
            }
        }
    }
}
