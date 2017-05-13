use ga::Individual;

#[derive(Debug, Clone)]
pub struct GAResult {
    pub min_fitness: Vec<f32>,
    pub max_fitness: Vec<f32>,
    pub avg_fitness: Vec<f32>,
    pub best: Individual,
    pub best_fe_count: u32,
    pub first_hit_fe_count: u32,
    pub fe_count: u32,
}

impl GAResult {
    pub fn new() -> GAResult {
        GAResult{avg_fitness: Vec::new(),
                 min_fitness: Vec::new(),
                 max_fitness: Vec::new(),
                 best: Individual::new(),
                 best_fe_count: 0,
                 first_hit_fe_count: 0,
                 fe_count: 0,
                 }
    }
}
