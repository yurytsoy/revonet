extern crate rand;

use rand::{Rng};

extern crate revonet;

use revonet::neuro::*;

fn main() {
    const INPUT_SIZE: usize = 20;
    const OUTPUT_SIZE: usize = 2;

    let mut rng = rand::thread_rng();   // needed for weights initialization when NN is built.
    let mut net: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
    net.add_hidden_layer(30 as usize, ActivationFunctionType::Sigmoid)
         .add_hidden_layer(20 as usize, ActivationFunctionType::Sigmoid)
         .build(&mut rng, NeuralArchitecture::Multilayered);       // `build` finishes creation of neural network.

    let (ws, bs) = net.get_weights();   // `ws` and `bs` are `Vec` arrays containing weights and biases for each layer.
    assert!(ws.len() == 3);     // number of elements equals to number of hidden layers + 1 output layer
    assert!(bs.len() == 3);     // number of elements equals to number of hidden layers + 1 output layer

    let rnd_input = (0..INPUT_SIZE).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
    println!("NN outputs: {:?}", net.compute(&rnd_input));
}