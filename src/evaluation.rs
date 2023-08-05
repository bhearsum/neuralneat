/// Custom activation functions must match this type.
pub type ActivationFn = fn(f32) -> f32;
pub type EvaluationFn = fn(&Vec<f32>, &Vec<f32>) -> f32;

// These are both defined as variables because min and max don't
// seem to work correctly when called on literals...
const MINUS_SIXTY: f32 = -60.0;
const SIXTY: f32 = 60.0;

/// A basic sigmoid activation function
// TODO: make this a "regular" sigmoid (ie: it should return positive and
// negative values.
pub fn sigmoid(x: f32) -> f32 {
    let z = MINUS_SIXTY.max(SIXTY.min(5.0 * x)) * -1.0;
    return 1.0 / (1.0 + z.exp());
}

pub fn basic_eval(outputs: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    assert_eq!(outputs.len(), expected.len());

    let mut fitness = 0.0;
    for i in 0..outputs.len() {
        if outputs[i] == expected[i] {
            fitness += 1.0;
        }
    }

    return fitness;
}

pub struct TrainingData {
    pub inputs: Vec<f32>,
    pub expected: Vec<f32>,
}
