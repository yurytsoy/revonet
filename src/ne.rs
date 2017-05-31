use context::*;
use ea::*;
use ga::*;
use neproblem::*;
use problem::*;
use result::*;
use settings::*;

struct NEIndividual {

}

impl Individual for NEIndividual {
    fn new() -> NEIndividual {
        NEIndividual{}
    }
}

struct NE<'a> {
    ctx: EAContext<NEIndividual>,
    problem: &'a Problem,
}

impl<'a> NE<'a> {
    pub fn new(settings: EASettings, problem: &'a Problem) -> NE {
        NE{problem: problem,
           ctx: EAContext::new(settings),
        }
    }

    pub fn run(&mut self, gen_count: u32) -> Result<EAResult<NEIndividual>, ()> {
        let mut ctx = self.ctx.clone();
        let res = self.run_with_context(&mut ctx, self.problem, gen_count);
        self.ctx = ctx;
        res
    }
}

impl<'a> EA for NE<'a> {
    fn breed(&self, ctx: &mut EAContext<NEIndividual>, sel_inds: &Vec<usize>, children: &mut Vec<NEIndividual>) {
        cross(&ctx.population, sel_inds, children, ctx.settings.use_elite, ctx.settings.x_prob, ctx.settings.x_alpha, &mut ctx.rng);
        mutate(children, ctx.settings.mut_prob);
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
        let settings = EASettings::new(pop_size, gen_count, param_count);
        let problem = SymbolicRegressionProblem::new_f();

        let ne = NE::new(settings, &problem);
    }
}