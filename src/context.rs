use rand::{Rng, SeedableRng, StdRng};

use ea::*;
use ga::*;
use result::*;
use settings::*;

#[allow(dead_code)]
#[derive(Clone)]
pub struct EAContext<T: Individual> {
    pub settings: EASettings,
    pub result: EAResult<T>,

    pub population: Vec<T>,
    pub fitness: Vec<f32>,
    pub sel_individuals: Vec<usize>,
    pub rng: StdRng,
}

impl<T: Individual> EAContext<T> {
    pub fn new(settings: EASettings) -> EAContext<T> {
        let mut rng: StdRng = StdRng::from_seed(&[settings.rng_seed as usize]);
        EAContext{population: create_population(settings.pop_size, settings.param_count, &mut rng),
                  settings: settings,
                  result: EAResult::new(),
                  fitness: Vec::new(),
                  sel_individuals: Vec::new(),
                  rng: rng}
    }
}