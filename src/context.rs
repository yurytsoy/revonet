use rand::{Rng, SeedableRng, StdRng};

use ga::*;
// use problem::*;
use result::*;
use settings::*;

#[allow(dead_code)]
pub struct EAContext<'a> {
    pub settings: &'a GASettings,
    pub result: GAResult,

    pub population: Vec<Individual>,
    pub fitness: Vec<f32>,
    pub sel_individuals: Vec<usize>,
    // pub next_gen: Vec<Individual>,
    pub rng: StdRng,
}

impl<'a> EAContext<'a> {
    pub fn new(settings: &'a GASettings) -> EAContext {
        let mut rng: StdRng = StdRng::from_seed(&[settings.rng_seed as usize]);
        EAContext{population: create_population(settings.pop_size, settings.param_count, &mut rng),
                  settings: settings,
                  result: GAResult::new(),
                  fitness: Vec::new(),
                  sel_individuals: Vec::new(),
                //   next_gen: Vec::new(),
                  rng: rng}
    }
}