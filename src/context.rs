use rand::{Rng, SeedableRng, StdRng};

use ea::*;
use ga::*;
use result::*;
use settings::*;

#[allow(dead_code)]
#[derive(Clone)]
pub struct EAContext {
    pub settings: EASettings,
    pub result: EAResult,

    pub population: Vec<Individual>,
    pub fitness: Vec<f32>,
    pub sel_individuals: Vec<usize>,
    pub rng: StdRng,
}

impl EAContext {
    pub fn new(settings: EASettings) -> EAContext {
        let mut rng: StdRng = StdRng::from_seed(&[settings.rng_seed as usize]);
        EAContext{population: create_population(settings.pop_size, settings.param_count, &mut rng),
                  settings: settings,
                  result: EAResult::new(),
                  fitness: Vec::new(),
                  sel_individuals: Vec::new(),
                //   next_gen: Vec::new(),
                  rng: rng}
    }
}