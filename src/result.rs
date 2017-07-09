use serde::de::{DeserializeOwned};
use serde::ser::{Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufReader, Read, Write};

use ea::Individual;

/// Structure to hold results for the genetic algorithm run.
#[derive(Clone, Debug, Deserialize, Serialize)]
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

impl<'de, T: Individual+Clone+DeserializeOwned+Serialize> EAResult<T> {
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

    pub fn from_json<'a>(filename: &str) -> Self {
        let file = File::open(filename).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut json_str = String::new();
        buf_reader.read_to_string(&mut json_str).unwrap();

        let res: EAResult<T> = serde_json::from_str(&json_str).unwrap();
//        let res: EAResult<T> = serde_json::from_reader(buf_reader).unwrap();
        res.clone()
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
    use ea::*;
    use ga::*;
    use problem::*;
    use result::*;
    use settings::*;

    #[test]
    fn test_json() {
        let pop_size = 10u32;
        let problem_dim = 5u32;
        let problem = SphereProblem{};

        let gen_count = 10u32;
        let settings = EASettings::new(pop_size, gen_count, problem_dim);
        let mut ga: GA<SphereProblem, RealCodedIndividual> = GA::new(&problem);
        let res = ga.run(settings).expect("Error during GA run");

        let filename = "test_json_earesult.json";
        res.to_json(&filename);

        let res2: EAResult<RealCodedIndividual> = EAResult::from_json(&filename);
        assert!(res.best_fe_count == res2.best_fe_count);
        assert!(res.fe_count == res2.fe_count);
        assert!(res.first_hit_fe_count == res2.first_hit_fe_count);
        assert!(res.best.fitness == res2.best.fitness);
    }
}
