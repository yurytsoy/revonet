#[derive(Debug, Clone)]
pub struct GASettings {
    pub rng_seed: u32,
    pub pop_size: u32,
    pub param_count: u32,
    pub gen_count: u32,
    pub tour_size: u32,
    pub use_elite: bool,
    pub x_prob: f32,
    pub x_alpha: f32,
    pub mut_prob: f32,
    pub mut_sigma: f32,
}

impl GASettings {
    pub fn new(pop_size: u32, gen_count: u32, param_count: u32) -> GASettings {
        GASettings{
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