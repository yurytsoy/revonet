use context::*;
use ga::*;
use neproblem::*;
use problem::*;
use settings::*;

struct NE<'a> {
    ctx: EAContext,
    problem: &'a NeuroProblem,
}

impl<'a> NE<'a> {
    pub fn new(settings: EASettings, problem: &'a NeuroProblem) -> NE {
        NE{problem: problem,
           ctx: EAContext::new(settings),
        }
    }

    pub fn run(&mut self, gen_count: u32) {
        
    }
}

//===================================================================

#[cfg(test)]
mod test {
    use rand;

    use math::*;
    use ne::*;
    use neproblem::*;
    use neuro::*;
    use settings::*;

    #[test]
    pub fn test_symbolic_regression() {
        let (pop_size, gen_count, param_count) = (20, 20, 100);
        let settings = GASettings::new(pop_size, gen_count, param_count);
        let problem = SymbolicRegressionProblem::new_f();

        let ne = NE::new(settings, &problem);
    }
}