extern crate rand;

mod ga;
mod math;
mod neuro;
mod problem;
mod result;
mod settings;

use ga::*;
use problem::*;
use settings::*;

/*
TODO: Add GA context
TODO: Make reproducible by enabling seeded rng and passing rng in the context
TODO: Add saving/loading settings/results to json
TODO: Add documentation
TODO: Add support for more crossover and mutation operators
TODO: Implement multilayered ANN
TODO: Add test problems for ANN
TODO: Add speciation
*/

fn main() {
    let pop_size = 20u32;
    let problem_dim = 10u32;
    let problem = RosenbrockProblem{};

    let gen_count = 10u32;
    let settings = GASettings::new(pop_size, gen_count, problem_dim);
    let mut ga = GA::new(settings, &problem);
    ga.run(gen_count);
    let res = ga.get_result().expect("No GA result is available!");
    println!("\n\nGA results: {:?}", res);
}
