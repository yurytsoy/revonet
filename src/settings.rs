use serde_json;
use std::fs::File;
use std::io::{BufReader, Read, Write};

/// Settings for evolutionary algorithm.
#[derive(Clone, Debug, Deserialize, Serialize)]
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

    pub fn from_json(filename: &str) -> Self {
        let file = File::open(filename).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut json_str = String::new();
        buf_reader.read_to_string(&mut json_str).unwrap();

        let res: EASettings = serde_json::from_str(&json_str).unwrap();
        res
    }

    pub fn to_json(&self, filename: &str) {
        let mut file = File::create(&filename).unwrap();
        let json_str = serde_json::to_string(&self).unwrap();
        file.write_all(json_str.as_bytes()).unwrap();
    }
}

//============================================================================================

#[cfg(test)]
mod test {
    use settings::*;

    #[test]
    fn test_json() {
        let settings = EASettings::new(100, 100, 100);
        let filename = "test_json.json";
        settings.to_json(&filename);

        let settings2 = EASettings::from_json(&filename);
        assert!(settings.gen_count == settings2.gen_count);
        assert!(settings.mut_prob == settings2.mut_prob);
        assert!(settings.mut_sigma == settings2.mut_sigma);
        assert!(settings.param_count == settings2.param_count);
        assert!(settings.pop_size == settings2.pop_size);
        assert!(settings.rng_seed == settings2.rng_seed);
        assert!(settings.tour_size == settings2.tour_size);
        assert!(settings.use_elite == settings2.use_elite);
        assert!(settings.x_alpha == settings2.x_alpha);
        assert!(settings.x_prob == settings2.x_prob);
    }
}
