extern crate rand;
extern crate revonet;

//use rand::{Rng, StdRng, SeedableRng};

use revonet::ea::*;
use revonet::ga::*;
use revonet::problem::*;
use revonet::settings::*;

fn main() {
    let pop_size = 20u32;
    let problem_dim = 10u32;
    let problem = SphereProblem{};

    let gen_count = 10u32;
    let settings = EASettings::new(pop_size, gen_count, problem_dim);
    let mut ga: GA<SphereProblem> = GA::new(&problem);
    let res = ga.run(settings).expect("Error during GA run");
    println!("\n\nGA results: {:?}", res);
}
