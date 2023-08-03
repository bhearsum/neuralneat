/* This example will train a neural network that can sum its inputs */
use neuralneat::{Genome, Pool};
use serde_json;
use std::env;
use std::fs::File;

#[path = "common/lib.rs"]
mod common;
use common::{adding_fitness_func, linear_activation, load_training_data};

const TRAINING_DATA_STRING: &str = include_str!("adding.csv");

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: '{} train' to train a new comparer.", args[0]);
        println!("Usage: '{} evaluate serialized_genome.json input1 input2 input3 input4' to evaluate with an existing genome.", args[0]);
        return;
    }

    if args[1] == "train" {
        // One input node for each input in the training data structure
        let input_nodes = 4;
        // One output node with the "prediction"
        let output_nodes = 1;
        // Create a new gene pool with an initial population of genomes
        let mut gene_pool = Pool::with_defaults(input_nodes, output_nodes);

        let training_data = load_training_data(TRAINING_DATA_STRING, 4, 1);

        // Train over the course of 100 generations
        gene_pool.train_population(
            100,
            &training_data,
            // This function will be called once per Genome per piece of
            // TrainingData in each generation, passing the values of the
            // output nodes of the Genome as well as the expected result
            // from the TrainingData.
            adding_fitness_func,
            Some(linear_activation),
            Some(linear_activation),
        );

        let best_genome = gene_pool.get_best_genome();

        println!("Serializing best genome to winner.json");
        serde_json::to_writer(&File::create("winner.json").unwrap(), &best_genome).unwrap();
    } else {
        if args.len() < 7 {
            println!("Usage: '{} evaluate serialized_genome.json input1 input2 input3 input4' to evaluate with an existing genome.", args[0]);
            return;
        }
        let mut genome: Genome = serde_json::from_reader(File::open(&args[2]).unwrap()).unwrap();
        let input1 = args[3]
            .parse::<f32>()
            .expect("Couldn't parse input1 as f32");
        let input2 = args[4]
            .parse::<f32>()
            .expect("Couldn't parse input2 as f32");
        let input3 = args[5]
            .parse::<f32>()
            .expect("Couldn't parse input1 as f32");
        let input4 = args[6]
            .parse::<f32>()
            .expect("Couldn't parse input2 as f32");
        // Note that this is the exact same function we used in training
        // further up!
        genome.evaluate(
            &vec![input1, input2, input3, input4],
            Some(linear_activation),
            Some(linear_activation),
        );
        println!(
            "Sum of inputs is..........{}",
            genome.get_outputs()[0] as u32
        );
    }
}
