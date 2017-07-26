extern crate revonet;

use revonet::ea::*;
use revonet::ne::*;
use revonet::neproblem::*;
use revonet::settings::*;

fn main() {
    let (pop_size, gen_count, param_count) = (20, 20, 100);
    let settings = EASettings::new(pop_size, gen_count, param_count);
    let problem = SymbolicRegressionProblem::new_f();

    let mut ne: NE<SymbolicRegressionProblem> = NE::new(&problem);
    let res = ne.run(settings).expect("Error: NE result is empty");
    println!("result: {:?}", res);
    println!("\nbest individual: {:?}", res.best);
}