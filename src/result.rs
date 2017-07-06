use ea::Individual;

/// Structure to hold results for the genetic algorithm run.
#[derive(Debug, Clone)]
pub struct EAResult<T: Individual> {
    /// Array of minimal absolute values of fitness for each generation.
    pub min_fitness: Vec<f32>,
    /// Array of maximal absolute values of fitness for each generation.
    pub max_fitness: Vec<f32>,
    /// Array of average absolute values of fitness for each generation.
    pub avg_fitness: Vec<f32>,
    /// Best individual ever found during the single run.
    pub best: T,
    /// Number of function evaluations required to find the `best` individual.
    pub best_fe_count: u32,
    /// Number of function evaluations required to find the solution according to the `OptProblem::is_solution` function.
    pub first_hit_fe_count: u32,
    /// Total number of function evaluations used in the current run.
    pub fe_count: u32,
}

impl<T: Individual> EAResult<T> {
    /// Initialize empty result structure.
    pub fn new() -> EAResult<T> {
        EAResult{avg_fitness: Vec::new(),
                 min_fitness: Vec::new(),
                 max_fitness: Vec::new(),
                 best: T::new(),
                 best_fe_count: 0,
                 first_hit_fe_count: 0,
                 fe_count: 0,
                 }
    }
}
