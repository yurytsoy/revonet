use rand::{SeedableRng, StdRng};
use serde::de::DeserializeOwned;
use serde::ser::Serialize;

use ea::*;
use problem::*;
use result::*;
use settings::*;

/// Structure containing settings and intermetiade data required for an evolutionary algorithm to run.
///
/// Supposed to be passed around between different functions related to evolutionary algorithm.
///
/// Also stores statistics and results of the current run.
#[allow(dead_code)]
#[derive(Clone)]
pub struct EAContext<T: Individual+Serialize> {
    /// Settings of evolutionary algorithm.
    pub settings: EASettings,
    /// Results of the current run containing fitness statistics across generations and the best found solution.
    pub result: EAResult<T>,

    /// Population for current generation.
    pub population: Vec<T>,
    /// Fitness values ordered in the same way as population individuals.
    pub fitness: Vec<f32>,
    /// Indices of population members selected for breeding.
    pub sel_individuals: Vec<usize>,
    /// RNG which is used by different functions of EA. Proper seeding should ensure reproducibility.
    pub rng: StdRng,
}

impl<'de, T: Individual+Clone+Serialize+DeserializeOwned> EAContext<T> {
    /// Creates an EA context given current settings and reference to a problem. Also seeds RNG used
    /// in the EA.
    ///
    /// # Arguments:
    /// * `settings` - settings for evolutionary algorithm.
    /// * `problem` - reference to problem (objective function). Needed to initialize a population.
    pub fn new<P: Problem>(settings: EASettings, problem: &P) -> EAContext<T> {
        let mut rng: StdRng = StdRng::from_seed(&[settings.rng_seed as usize]);
        EAContext{population: create_population(settings.pop_size, settings.param_count, &mut rng, problem),
                  settings: settings,
                  result: EAResult::new(),
                  fitness: Vec::new(),
                  sel_individuals: Vec::new(),
                  rng: rng}
    }
}
