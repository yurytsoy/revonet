extern crate revonet;

use revonet::ea::*;
use revonet::ne::*;
use revonet::neproblem::*;
use revonet::settings::*;

fn main() {
    let (pop_size, gen_count, param_count) = (20, 50, 100); // gene_count does not matter here as NN structure is defined by a problem.
    let settings = EASettings::new(pop_size, gen_count, param_count);
    let problem = XorProblem::new();

    let mut ne: NE<XorProblem> = NE::new(&problem);
    let res = ne.run(settings).expect("Error: NE result is empty");
    println!("result: {:?}", res);
    println!("\nbest individual: {:?}", res.best);
}
