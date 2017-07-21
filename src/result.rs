use serde::de::{DeserializeOwned};
use serde::ser::{Serialize};
use serde_json;
use std;
use std::fs::File;
use std::io::{BufReader, Read, Write};

use ea;
use ea::*;
use math::*;

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

impl<T: Individual+Clone+DeserializeOwned+Serialize> EAResult<T> {
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
        let file = File::open(filename).expect("Can not open file");
        let mut buf_reader = BufReader::new(file);
        let mut json_str = String::new();
        buf_reader.read_to_string(&mut json_str).expect("Can not read file contents");

        let res: EAResult<T> = serde_json::from_str(&json_str).expect("Can not deserialize from json to EAResult");
        res.clone()
    }

    pub fn to_json(&self, filename: &str) {
        let mut file = File::create(&filename).expect("Can not open file");
        let json_str = serde_json::to_string(&self).expect("Can not serialize to json from EAResult");
        file.write_all(json_str.as_bytes()).expect("Can not write to file");
    }
}

/// Structure to hold results for multipole runs of evolutionary algorithm.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EAResultMultiple<T: Individual> {
    /// Array of minimal absolute values of fitness for each generation.
    pub min_fitness: Vec<f32>,
    /// Array of maximal absolute values of fitness for each generation.
    pub max_fitness: Vec<f32>,
    /// Array of average absolute values of fitness for each generation.
    pub avg_fitness_mean: Vec<f32>,
    /// Array of SD for average absolute values of fitness for each generation.
    pub avg_fitness_sd: Vec<f32>,
    /// Best individual ever found during the single run.
    pub best: T,
    /// Mean number of function evaluations required to find the `best` individual.
    pub best_fe_count_mean: f32,
    /// SD for number of function evaluations required to find the `best` individual.
    pub best_fe_count_sd: f32,
    /// Mean number of function evaluations required to find the solution according to the `OptProblem::is_solution` function.
    pub first_hit_fe_count_mean: f32,
    /// SD for number of function evaluations required to find the solution according to the `OptProblem::is_solution` function.
    pub first_hit_fe_count_sd: f32,
    /// Number of runs when solution was found.
    pub success_count: u32,
    /// Total number of runs which were performed in order to compute the statistics.
    pub run_count: u32,
}

impl<T: Individual+Clone+DeserializeOwned+Serialize> EAResultMultiple<T> {
    pub fn new(rs: &[EAResult<T>]) -> EAResultMultiple<T> {
        let run_count = rs.len();
        let mut avg_fitness_mean = vec![0f32; run_count];
        let mut avg_fitness_sd = vec![0f32; run_count];
        let mut min_fitness = vec![std::f32::MAX; run_count];
        let mut max_fitness = vec![std::f32::MIN; run_count];
        let mut best_fe_count_mean = 0f32;
        let mut best_fe_count_sd = 0f32;
        let mut first_hit_fe_count_mean = 0f32;
        let mut first_hit_fe_count_sd = 0f32;
        let mut success_count = 0;
        let mut best_fitness = std::f32::MAX;
        let mut best_run_idx = std::usize::MAX;

        for k in 0..rs.len() {
            acc(&mut avg_fitness_mean, &rs[k].avg_fitness);
            acc(&mut avg_fitness_sd, &sqr(&rs[k].avg_fitness));
            min_inplace_vv(&mut min_fitness, &rs[k].min_fitness);
            max_inplace_vv(&mut max_fitness, &rs[k].max_fitness);

            best_fe_count_mean += rs[k].best_fe_count as f32;
            best_fe_count_sd += ((rs[k].best_fe_count) * (rs[k].best_fe_count)) as f32;
            if rs[k].first_hit_fe_count > 0 {
                first_hit_fe_count_mean += rs[k].first_hit_fe_count as f32;
                first_hit_fe_count_sd += ((rs[k].first_hit_fe_count) * (rs[k].first_hit_fe_count)) as f32;
                success_count += 1;
            }

            if rs[k].best.get_fitness() < best_fitness {
                best_fitness = rs[k].best.get_fitness();
                best_run_idx =  k;
            }
        }
        mul_inplace(&mut avg_fitness_mean, 1f32/run_count as f32);
        mul_inplace(&mut avg_fitness_sd, 1f32/run_count as f32);    // compute SD as: mean square  - squared mean
        sub_inplace(&mut avg_fitness_sd, &sqr(&avg_fitness_mean));
        best_fe_count_mean /= run_count as f32;
        best_fe_count_sd = best_fe_count_sd / run_count as f32 - best_fe_count_mean*best_fe_count_mean;
        if success_count > 0 {
            first_hit_fe_count_mean /= success_count as f32;
            first_hit_fe_count_sd = first_hit_fe_count_sd / success_count as f32 - first_hit_fe_count_mean * first_hit_fe_count_mean;
        }

        EAResultMultiple{
            avg_fitness_mean: avg_fitness_mean,
            avg_fitness_sd: avg_fitness_sd,
            min_fitness: min_fitness,
            max_fitness: max_fitness,
            best: ea::Individual::clone(&rs[best_run_idx].best),
            best_fe_count_mean: best_fe_count_mean,
            best_fe_count_sd: best_fe_count_sd,
            first_hit_fe_count_mean: first_hit_fe_count_mean,
            first_hit_fe_count_sd: first_hit_fe_count_sd,
            success_count: success_count,
            run_count: run_count as u32,
        }
    }

    pub fn from_json<'a>(filename: &str) -> Self {
        let file = File::open(filename).expect("Can not open file");
        let mut buf_reader = BufReader::new(file);
        let mut json_str = String::new();
        buf_reader.read_to_string(&mut json_str).expect("Can not read file contents");

        let res: EAResultMultiple<T> = serde_json::from_str(&json_str).expect("Can not deserialize from json to EAResult");
        res.clone()
    }

    pub fn to_json(&self, filename: &str) {
        let mut file = File::create(&filename).expect("Can not open file");
        let json_str = serde_json::to_string(&self).expect("Can not serialize to json from EAResult");
        file.write_all(json_str.as_bytes()).expect("Can not write to file");
    }}


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
