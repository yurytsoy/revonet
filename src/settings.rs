/// Settings for evolutionary algorithm.
#[derive(Debug, Clone)]
pub struct EASettings {
    /// Seed for RNG.
    pub rng_seed: u32,
    /// Population size.
    pub pop_size: u32,
    /// Number of parameters/genes for the problem.
    pub param_count: u32,
    /// Number of generations.
    pub gen_count: u32,
    /// tournament size for tournament selection.
    pub tour_size: u32,
    /// Flag indicating whether elitism is to be used or not.
    ///
    /// If `true` then the best individual in current generation is copied to the next generation.
    pub use_elite: bool,
    /// Crossover probability.
    pub x_prob: f32,
    /// Crossover parameter.
    pub x_alpha: f32,
    /// Mutation probability.
    pub mut_prob: f32,
    /// Mutation parameter.
    pub mut_sigma: f32,
}

impl EASettings {
    /// Create default settings using some parameters.
    ///
    /// The default values are as follows:
    /// * `tour_size = 3`
    /// * `use_elite = true`
    /// * `x_prob = 0.7f32`
    /// * `x_alpha = 0.1f32`
    /// * `mut_prob = 1f32 / param_count`
    /// * `mut_sigma = 0.1f32`
    ///
    /// # Arguments:
    /// * `pop_size` - population size.
    /// * `gen_count` - number of generations.
    /// * `param_count` - number of genes / problem parameters.
    pub fn new(pop_size: u32, gen_count: u32, param_count: u32) -> EASettings {
        EASettings{
            rng_seed: 0,
            param_count: param_count,
            pop_size: pop_size,
            gen_count: gen_count,
            tour_size: 3,
            use_elite: true,
            x_prob: 0.7f32,
            x_alpha: 0.1f32,
            mut_prob: 1f32 / (param_count as f32),
            mut_sigma: 0.1f32,
        }
    }
}