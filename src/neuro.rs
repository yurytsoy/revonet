use rand::{ThreadRng};
use std::fmt::Debug;

use math::*;

#[derive(Debug)]
#[allow(dead_code)]
struct MultilayeredNetwork {
    inputs_num: usize,
    outputs_num: usize,
    layers: Vec<Box<Layer>>,
}

#[allow(dead_code)]
impl MultilayeredNetwork {
    pub fn new(inputs_num: usize, outputs_num: usize) -> MultilayeredNetwork {
        MultilayeredNetwork{
            inputs_num: inputs_num,
            outputs_num: outputs_num,
            layers: Vec::new()
        }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn add_layer<T: ActivationFunction + 'static>(&mut self, size: usize) -> &mut Self {
        self.layers.push(Box::new(NeuralLayer::<T>::new(size)));
        self
    }

    pub fn build(&mut self, rng: &mut ThreadRng) {
        // add output layer.
        self.layers.push(Box::new(NeuralLayer::<LinearActivation>::new(self.outputs_num)));

        // init weights and biases for all layers.
        let mut inputs = self.inputs_num;
        for l in self.layers.iter_mut() {
            l.init_weights(inputs, rng);
            inputs = l.len();
        }
    }
}

//========================================

trait Layer: Debug {
    fn init_weights(&mut self, inputs_num: usize, rng: &mut ThreadRng);
    fn compute(&mut self, xs: &[f32]) -> Vec<f32>;
    fn len(&self) -> usize;
}

#[derive(Debug)]
#[allow(dead_code)]
struct NeuralLayer<T: ActivationFunction> {
    size: usize,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    outputs: Vec<f32>,
    activation: Vec<T>,
}

#[allow(dead_code)]
impl<T: ActivationFunction> NeuralLayer<T> {
    pub fn new(size: usize) -> NeuralLayer<T> {
        NeuralLayer{
            size: size,
            weights: Vec::new(),
            biases: Vec::new(),
            outputs: Vec::new(),
            activation: (0..size).map(|_| T::new()).collect::<Vec<T>>()
        }
    }
}

#[allow(dead_code)]
impl<T: ActivationFunction> Layer for NeuralLayer<T> {
    fn init_weights(&mut self, inputs_num: usize, rng: &mut ThreadRng) {
        for _ in 0..self.size {
            self.weights.push(rand_vector_std_gauss(inputs_num, rng));
        }
        self.biases = rand_vector_std_gauss(inputs_num, rng);
    }
    
    fn compute(&mut self, xs: &[f32]) -> Vec<f32> {
        self.outputs = dot_mv(&self.weights, &xs);
        self.outputs = (0..self.outputs.len())
                            .map(|k| self.activation[k].compute(self.outputs[k] + self.biases[k]))
                            .collect::<Vec<f32>>();
        self.outputs.clone()
    }

    fn len(&self) -> usize {self.size}
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
    fn test_linear_multilayer_net() {
        const INPUT_SIZE: usize = 20;
        const OUTPUT_SIZE: usize = 2;

        let mut rng = rand::thread_rng();
        let mut net = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
        net.add_layer::<SigmoidActivation>(30 as usize)
            .add_layer::<LinearActivation>(15 as usize)
            .add_layer::<SigmoidActivation>(300 as usize)
            .build(&mut rng);
        

        // println!("{:?}", net);
        // for k in 0..net.len() {
        //     println!("{}: {:?}", k, net.layers[k])
        // }

        // let mut rng = rand::thread_rng();
        // let Neurallayer = NeuralLayer::new::<LinearActivation>(NeuralLAYER_SIZE);
        // let xs = rand_vector_stdgauss(LEN, &mut rng);
        // let act_f = LinearActivation{};
        // assert!((0..LEN).all(|k| xs[k] == act_f.compute(xs[k])));
    }

}
