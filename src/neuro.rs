use rand::{ThreadRng};
use std::fmt::Debug;

use math::*;

#[derive(Debug)]
#[allow(dead_code)]
struct MultilayeredNetwork {
    inputs_num: usize,
    outputs_num: usize,
    layers: Vec<Box<Layer>>,
    is_built: bool,
}

#[allow(dead_code)]
impl MultilayeredNetwork {
    pub fn new(inputs_num: usize, outputs_num: usize) -> MultilayeredNetwork {
        MultilayeredNetwork{
            inputs_num: inputs_num,
            outputs_num: outputs_num,
            layers: Vec::new(),
            is_built: false,
        }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn add_hidden_layer<T: ActivationFunction + 'static>(&mut self, size: usize) -> &mut Self {
        if self.is_built {
            panic!("Can not add layer to already built network.");
        }
        self.layers.push(Box::new(NeuralLayer::<T>::new(size)));
        self
    }

    pub fn build(&mut self, rng: &mut ThreadRng) {
        if self.is_built {
            panic!("The network has already been built.");
        }

        // add output layer.
        self.layers.push(Box::new(NeuralLayer::<LinearActivation>::new(self.outputs_num)));

        // init weights and biases for all layers.
        let mut inputs = self.inputs_num;
        for l in self.layers.iter_mut() {
            l.init_weights(inputs, rng);
            inputs = l.len();
        }
        self.is_built = true;
    }

    pub fn compute(&mut self, xs: &[f32]) -> Vec<f32> {
        let mut input = Vec::from(xs);
        for l in self.layers.iter_mut() {
            input = l.compute(&input);
        }
        Vec::from(input)
    }

    /// Returns weights and biases layer by layer.
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
    pub fn set_weights(&mut self, wss: &Vec<Vec<f32>>, bss: &Vec<Vec<f32>>) {
        if !self.is_built {
            panic!("Can not set network weights: the network is not built yet.");
        }

        for k in 0..self.layers.len() {
            self.layers[k].set_weights(&wss[k], &bss[k]);
        }
    }
}

//========================================

trait Layer: Debug {
    fn init_weights(&mut self, inputs_num: usize, rng: &mut ThreadRng);
    fn compute(&mut self, xs: &[f32]) -> Vec<f32>;
    fn len(&self) -> usize;
    fn get_weights(&self) -> (Vec<f32>, Vec<f32>);
    fn set_weights(&mut self, ws: &Vec<f32>, bs: &Vec<f32>);
}

#[derive(Debug)]
#[allow(dead_code)]
struct NeuralLayer<T: ActivationFunction> {
    size: usize,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    outputs: Vec<f32>,
    activations: Vec<T>,
}

#[allow(dead_code)]
impl<T: ActivationFunction> NeuralLayer<T> {
    pub fn new(size: usize) -> NeuralLayer<T> {
        NeuralLayer{
            size: size,
            weights: Vec::new(),
            biases: Vec::new(),
            outputs: Vec::new(),
            activations: (0..size).map(|_| T::new()).collect::<Vec<T>>()
        }
    }
}

#[allow(dead_code)]
impl<T: ActivationFunction> Layer for NeuralLayer<T> {
    fn init_weights(&mut self, inputs_num: usize, rng: &mut ThreadRng) {
        // println!("Init weights: {} nodes, {} inputs", self.size, inputs_num);
        for _ in 0..self.size {
            self.weights.push(rand_vector_std_gauss(inputs_num, rng));
        }
        self.biases = rand_vector_std_gauss(self.size, rng);
    }
    
    fn compute(&mut self, xs: &[f32]) -> Vec<f32> {
        self.outputs = dot_mv(&self.weights, &xs);
        self.outputs = (0..self.outputs.len())
                            .map(|k| self.activations[k].compute(self.outputs[k] + self.biases[k]))
                            .collect::<Vec<f32>>();
        self.outputs.clone()
    }

    fn len(&self) -> usize {self.size}

    fn get_weights(&self) -> (Vec<f32>, Vec<f32>) {
        let mut res_w = Vec::new();
        let mut res_b = Vec::with_capacity(self.size);
        
        for k in 0..self.size {
            res_w.extend(self.weights[k].clone());
            res_b.push(self.biases[k]);
        }
        (res_w, res_b)
    }

    fn set_weights(&mut self, ws: &Vec<f32>, bs: &Vec<f32>) {
        let inputs = self.weights[0].len();
        for k in 0..self.size {
            self.weights[k] = Vec::from(&ws[k*inputs..(k+1)*inputs]);
            self.biases[k] = bs[k];
        }
    }
}


//========================================

pub trait ActivationFunction: Debug {
    fn compute(&self, x: f32) -> f32;
    fn new() -> Self;
}

#[derive(Debug)]
pub struct LinearActivation;
impl ActivationFunction for LinearActivation {
    fn new() -> LinearActivation {LinearActivation{}}
    fn compute(&self, x: f32) -> f32 {x}
}

#[derive(Debug)]
pub struct SigmoidActivation;
impl ActivationFunction for SigmoidActivation {
    fn new() -> SigmoidActivation {SigmoidActivation{}}
    fn compute(&self, x: f32) -> f32 {
        1f32 / (1f32 + (-x).exp())
    }
}

#[cfg(test)]
mod test {
    use rand;

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
        let mut rng = rand::thread_rng();
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

        let mut rng = rand::thread_rng();
        let mut net_linear: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
        net_linear.add_hidden_layer::<LinearActivation>(30 as usize).build(&mut rng);

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

        let mut rng = rand::thread_rng();
        let mut net: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
        net.add_hidden_layer::<SigmoidActivation>(30 as usize)
            .add_hidden_layer::<LinearActivation>(15 as usize)
            .add_hidden_layer::<SigmoidActivation>(300 as usize)
            .build(&mut rng);

        // net2 has the same structure and weights as net1.
        let mut net2: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
        net2.add_hidden_layer::<SigmoidActivation>(30 as usize)
            .add_hidden_layer::<LinearActivation>(15 as usize)
            .add_hidden_layer::<SigmoidActivation>(300 as usize)
            .build(&mut rng);
        let (ws, bs) = net.get_weights();
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
