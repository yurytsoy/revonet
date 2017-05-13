use math::*;

// struct Network {
//     layers: Vec<Layer>,
// }

// impl Network {

// }

//========================================

#[allow(dead_code)]
struct Layer<T: Node> {
    nodes: Vec<T>,
}

#[allow(dead_code)]
impl<T: Node> Layer<T> {
    pub fn new(size: usize) -> Layer<T> {
        Layer{
            nodes: (0..size).map(|_| T::new())
                .collect::<Vec<T>>()
        }
    }
}


//========================================

pub trait Node {
    fn compute(&self, xs: &Vec<f32>) -> f32;
    fn new() -> Self;
}

struct NodeData {
    weights: Vec<f32>,
    bias: f32,
}

#[allow(dead_code)]
impl NodeData {
    pub fn new() -> NodeData {
        NodeData{
            weights: Vec::new(),
            bias: 0f32,
        }
    }
}

#[allow(dead_code)]
pub struct LinearNode {
    data: NodeData,
}

impl Node for LinearNode {
    fn new() -> LinearNode {
        LinearNode{data: NodeData::new()}
    }

    fn compute (&self, xs: &Vec<f32>) -> f32 {
        self.data.bias + dot_product(&self.data.weights, &xs)
    }
}

#[allow(dead_code)]
pub struct SigmoidNode {
    data: NodeData,
}

impl Node for SigmoidNode {
    fn new() -> SigmoidNode {
        SigmoidNode{data: NodeData::new()}
    }
    fn compute(&self, xs: &Vec<f32>) -> f32 {
        let s = self.data.bias + dot_product(&self.data.weights, &xs);
        1f32 / (1f32 + (-s).exp())
    }
}
