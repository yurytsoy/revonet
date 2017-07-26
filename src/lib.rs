extern crate rand;
extern crate serde;
extern crate serde_json;

#[macro_use]
extern crate serde_derive;

pub mod context;
pub mod ea;
pub mod ga;
pub mod math;
pub mod ne;
pub mod neproblem;
pub mod neuro;
pub mod problem;
pub mod result;
pub mod settings;

use serde::de::{DeserializeOwned};
use serde::ser::{Serialize};
use std::fs::File;
use std::io::{BufReader, Read, Write};


trait Jsonable: Clone+Serialize {
    type T: Clone+DeserializeOwned+Serialize;

    fn from_json(filename: &str) -> Self::T {
        let file = File::open(filename).expect("Can not open file");
        let mut buf_reader = BufReader::new(file);
        let mut json_str = String::new();
        buf_reader.read_to_string(&mut json_str).expect("Can not read file contents");

        let res: Self::T = serde_json::from_str(&json_str).expect("Can not deserialize from json to EAResult");
        res.clone()
    }

    fn to_json(&self, filename: &str) {
        let mut file = File::create(&filename).expect("Can not open file");
        let json_str = serde_json::to_string(&self).expect("Can not serialize to json from EAResult");
        file.write_all(json_str.as_bytes()).expect("Can not write to file");
    }
}

