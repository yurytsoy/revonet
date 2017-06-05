//! Implementation of the real-coded genetic algorithm and evolving neural networks.
//!
//! # Examples
//!
//! Genetic algorithm
//!
//! ```
//! let pop_size = 20u32;       // population size.
//! let problem_dim = 10u32;    // number of optimization parameters.
//!
//! let problem = RosenbrockProblem{};  // objective function.
//! let gen_count = 10u32;      // generations number.
//! let settings = GASettings::new(pop_size, gen_count, problem_dim);
//! let mut ga = GA::new(settings, &problem);   // init GA.
//! ga.run(gen_count);          // run the search.
//!
//! // get and print results of the current run.
//! let res = ga.get_result().expect("No GA result is available!");
//! println!("\n\nGA results: {:?}", res);
//! ```
//!
//! Building multilayered neural network with 2 hidden layers with sigmoid activation and with linear output nodes.
//!
//! ```
//! const INPUT_SIZE: usize = 20;
//! const OUTPUT_SIZE: usize = 2;
//!
//! let mut rng = rand::thread_rng();   // needed for weights initialization when NN is built.
//! let mut net: MultilayeredNetwork = MultilayeredNetwork::new(INPUT_SIZE, OUTPUT_SIZE);
//! net.add_hidden_layer::<SigmoidActivation>(30 as usize)
//!     .add_hidden_layer::<SigmoidActivation>(20 as usize)
//!     .build(&mut rng);       // `build` finishes creation of neural network.
//! ```


extern crate rand;

use rand::{Rng, StdRng, SeedableRng};

mod context;
mod ea;
mod ga;
mod math;
mod ne;
mod neproblem;
mod neuro;
mod problem;
mod result;
mod settings;

use ea::*;
use ga::*;
use problem::*;
use settings::*;

fn main() {
    // const PTS_COUNT: u32 = 100;
    // let seed = 0;
    // let mut rng: StdRng = StdRng::from_seed(&[seed]);
    // let mut v = Vec::new();
    // for _ in 0..PTS_COUNT {
    //     v.push(rng.gen::<f32>());
    // }
    // println!("{:?}", v);

    // let z = Vec::new();
    // z.push(RealCodedIndividual::new());
    // let z2 = z.clone();

    let pop_size = 20u32;
    let problem_dim = 10u32;
    let problem = SphereProblem{};

    let gen_count = 10u32;
    let settings = EASettings::new(pop_size, gen_count, problem_dim);
    let mut ga: GA<SphereProblem, RealCodedIndividual> = GA::new(settings.clone(), &problem);
    let res = ga.run(settings).expect("Error during GA run");
    println!("\n\nGA results: {:?}", res);
}
