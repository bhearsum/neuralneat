// Some of these functions are not used by all examples, which causes dead code
// warnings. It's just example code so we may as well just disable it...
#![allow(dead_code)]

use neuralneat::TrainingData;

pub fn load_training_data(
    data_as_string: &str,
    n_inputs: usize,
    n_expected: usize,
) -> Vec<TrainingData> {
    let mut training_data = vec![];
    for line in data_as_string.lines() {
        let data = line.to_string();
        if data.starts_with("#") || !data.contains(",") {
            continue;
        }
        let parts: Vec<&str> = data.split(",").collect();
        assert_eq!(parts.len(), 5);

        training_data.push(TrainingData {
            inputs: parts[0..n_inputs]
                .iter()
                .map(|i| i.parse::<f32>().unwrap())
                .collect(),
            expected: parts[n_inputs..n_inputs + n_expected]
                .iter()
                .map(|i| i.parse::<f32>().unwrap())
                .collect(),
        });
    }

    return training_data;
}

// The adding examples use a linear activation function. This will cause each
// hidden and output node to use the sum of all of its inputs as its activation
// value. (By default, a sigmoidal activation function is used, which restricts
// the activation value of a node to a value between -1.0 and 1.0 which
// obviously not be useful in summing integers.)
pub fn linear_activation(x: f32) -> f32 {
    return x;
}

// This fitness function is used to "score" each genome in the "adding"
// examples. Higher scores are better ("more fit"), and the genomes with the
// highest fitness are used as "parents" in the next generation, with the
// lower scoring genomes being thrown away.
pub fn adding_fitness_func(outputs: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    // First find the difference between the expected value and what the genome
    // predicted.
    let difference = (outputs[0] - expected[0]).abs();
    // Then, use some Math (tm) to invert it such that the smaller the
    // difference, the higher the score, with a difference of 0 (aka
    // the genome got the right answer) resulting in the highest possible
    // fitness score.
    let z = (0.5_f32) * difference;
    return (1.0 / z.exp()) * 1000.0;
}
