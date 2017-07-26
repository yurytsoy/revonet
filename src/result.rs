use ::*;
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

impl<T: Individual+Clone+DeserializeOwned+Serialize> Jsonable for EAResult<T> {
    type T = Self;
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

impl<T: Individual+Clone> EAResultMultiple<T> {
    pub fn new(rs: &[EAResult<T>]) -> EAResultMultiple<T> {
        let mut avg_fitness_mean: Vec<f32> = Vec::new();
        let mut avg_fitness_sd: Vec<f32> = Vec::new();
        let mut min_fitness: Vec<f32> = Vec::new();
        let mut max_fitness: Vec<f32> = Vec::new();
        let mut best_fe_count_mean = 0f32;
        let mut best_fe_count_sd = 0f32;
        let mut first_hit_fe_count_mean = 0f32;
        let mut first_hit_fe_count_sd = 0f32;
        let mut success_count = 0;
        let mut best_fitness = std::f32::MAX;
        let mut best_run_idx = std::usize::MAX;

        let run_count = rs.len();
        for k in 0..run_count {
            let cur_res = &rs[k];
            if k != 0 {
                acc(&mut avg_fitness_mean, &cur_res.avg_fitness);
                acc(&mut avg_fitness_sd, &sqr(&cur_res.avg_fitness));
                min_inplace_vv(&mut min_fitness, &cur_res.min_fitness);
                max_inplace_vv(&mut max_fitness, &cur_res.max_fitness);
            } else {
                avg_fitness_mean = cur_res.avg_fitness.clone();
                avg_fitness_sd = sqr(&cur_res.avg_fitness);
                min_fitness = cur_res.min_fitness.clone();
                max_fitness = cur_res.max_fitness.clone();
            }

            best_fe_count_mean += cur_res.best_fe_count as f32;
            best_fe_count_sd += ((cur_res.best_fe_count) * (cur_res.best_fe_count)) as f32;
            if cur_res.first_hit_fe_count > 0 {
                first_hit_fe_count_mean += cur_res.first_hit_fe_count as f32;
                first_hit_fe_count_sd += ((cur_res.first_hit_fe_count) * (cur_res.first_hit_fe_count)) as f32;
                success_count += 1;
            }

            if rs[k].best.get_fitness() < best_fitness {
                best_fitness = cur_res.best.get_fitness();
                best_run_idx =  k;
            }
        }
        mul_inplace(&mut avg_fitness_mean, 1f32/run_count as f32);
        mul_inplace(&mut avg_fitness_sd, 1f32/run_count as f32);    // compute SD as: mean square  - squared mean
        sub_inplace(&mut avg_fitness_sd, &sqr(&avg_fitness_mean));
        max_inplace_vv(&mut avg_fitness_sd, &vec![0f32; avg_fitness_mean.len()]);
        best_fe_count_mean /= run_count as f32;
        best_fe_count_sd = best_fe_count_sd / run_count as f32 - best_fe_count_mean*best_fe_count_mean;
        if success_count > 0 {
            first_hit_fe_count_mean /= success_count as f32;
            first_hit_fe_count_sd = first_hit_fe_count_sd / success_count as f32 - first_hit_fe_count_mean * first_hit_fe_count_mean;
        }

        EAResultMultiple::<T>{
            avg_fitness_mean: avg_fitness_mean,
            avg_fitness_sd: avg_fitness_sd,
            min_fitness: min_fitness,
            max_fitness: max_fitness,
            // best: ea::Individual::clone(&rs[best_run_idx].best),
            best: rs[best_run_idx].best.clone(),
            best_fe_count_mean: best_fe_count_mean,
            best_fe_count_sd: best_fe_count_sd,
            first_hit_fe_count_mean: first_hit_fe_count_mean,
            first_hit_fe_count_sd: first_hit_fe_count_sd,
            success_count: success_count,
            run_count: run_count as u32,
        }
    }
}

impl<T: Individual+Clone+DeserializeOwned+Serialize> Jsonable for EAResultMultiple<T> {
    type T = Self;
}


//============================================================================================

#[cfg(test)]
mod test {
    #[allow(unused_imports)]
    use ea::*;
    use ga::*;
    use problem::*;
    use result::*;
    use settings::*;

    #[test]
    fn test_json_earesult() {
        let pop_size = 10u32;
        let problem_dim = 5u32;
        let problem = SphereProblem{};

        let gen_count = 10u32;
        let settings = EASettings::new(pop_size, gen_count, problem_dim);
        let mut ga: GA<SphereProblem> = GA::new(&problem);
        let res = ga.run(settings).expect("Error during GA run");

        let filename = "test_json_earesult.json";
        res.to_json(&filename);

        let res2: EAResult<RealCodedIndividual> = EAResult::from_json(&filename);
        assert!(res.best_fe_count == res2.best_fe_count);
        assert!(res.fe_count == res2.fe_count);
        assert!(res.first_hit_fe_count == res2.first_hit_fe_count);
        assert!(res.best.fitness == res2.best.fitness);
    }

    #[test]
    fn test_json_earesult_mult() {
        let pop_size = 10u32;
        let problem_dim = 5u32;
        let problem = SphereProblem{};

        let gen_count = 10u32;
        let ress = (0..3).into_iter()
            .map(|_ | {
                let settings = EASettings::new(pop_size, gen_count, problem_dim);
                let mut ga: GA<SphereProblem> = GA::new(&problem);
                ga.run(settings).expect("Error during GA run").clone()
            })
            .collect::<Vec<EAResult<RealCodedIndividual>>>();

        let filename = "test_json_earesult_mult.json";
        let res = EAResultMultiple::new(&ress);
        res.to_json(&filename);

        let res2: EAResultMultiple<RealCodedIndividual> = EAResultMultiple::from_json(&filename);
        assert!(res.run_count == res2.run_count);
        assert!(res.success_count == res2.success_count);
        assert!(res.first_hit_fe_count_mean == res2.first_hit_fe_count_mean);
        assert!(res.first_hit_fe_count_sd == res2.first_hit_fe_count_sd);
        assert!(res.best_fe_count_mean == res2.best_fe_count_mean);
        assert!(res.best_fe_count_sd == res2.best_fe_count_sd);
        assert!(res.best.fitness == res2.best.fitness);
        assert!(res.best.genes == res2.best.genes);
        assert!(res.min_fitness == res2.min_fitness);
        assert!(res.max_fitness == res2.max_fitness);
        assert!(res.avg_fitness_mean == res2.avg_fitness_mean);
        assert!(res.avg_fitness_sd == res2.avg_fitness_sd);
    }

    #[test]
    fn test_earesult_mult() {
        let pop_size = 10u32;
        let problem_dim = 5u32;
        let problem = SphereProblem{};

        let gen_count = 10u32;
        let ress = (0..3).into_iter()
            .map(|_ | {
                let settings = EASettings::new(pop_size, gen_count, problem_dim);
                let mut ga: GA<SphereProblem> = GA::new(&problem);
                ga.run(settings).expect("Error during GA run").clone()
            })
            .collect::<Vec<EAResult<RealCodedIndividual>>>();

        let filename = "test_json_earesult_mult.json";
        let res = EAResultMultiple::new(&ress);
        res.to_json(&filename);

        assert!(res.run_count == 3);
        assert!(res.min_fitness.len() == gen_count as usize);
        assert!(res.min_fitness.iter().zip(res.max_fitness.iter()).all(|(&min_f, &max_f)| min_f <= max_f));
        assert!(res.min_fitness.iter().zip(res.avg_fitness_mean.iter()).all(|(&min_f, &avg_f)| min_f <= avg_f));
        assert!(res.avg_fitness_sd.iter().all(|&f| f >= 0f32));
        assert!(res.min_fitness[(gen_count-1) as usize] == res.best.fitness);
    }
}
