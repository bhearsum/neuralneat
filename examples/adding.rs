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
        // One input node for each input in the TrainingData structure
        let input_nodes = 4;
        // One output node with the "prediction"
        let output_nodes = 1;
        // Create a new gene pool with an initial population of genomes
        let mut gene_pool = Pool::with_defaults(input_nodes, output_nodes);

        let training_data = load_training_data(TRAINING_DATA_STRING, 4, 1);

        // These variables are used to keep track of the top performer, so we
        // can write it out later.
        let mut best_genome: Option<Genome> = None;
        let mut best_fitness = 0.0;

        // We will test genomes from 100 generations
        for generation in 0..100 {
            println!("Evaluating generation {}", generation + 1);
            let total_species = gene_pool.len();
            // As genomes diverge in structure and configuration, they will
            // be divided into separate species.
            for s in gene_pool.species_ids().map(|k| *k).collect::<Vec<usize>>() {
                let species = &mut gene_pool[s];
                let genomes_in_species = species.len();
                for g in species.genome_ids().map(|k| *k).collect::<Vec<usize>>() {
                    let genome = &mut species[g];

                    let mut fitness = 0.0;
                    for td in &training_data {
                        // Evaluate the genome using the training data as the
                        // initial inputs.
                        genome.evaluate(
                            &td.inputs[0..4].to_vec(),
                            Some(linear_activation),
                            Some(linear_activation),
                        );

                        // We add this to the existing fitness for the genome
                        // to ensure that the genomes with the best score across
                        // all tests will have the highest overall fitness.
                        fitness += adding_fitness_func(&genome.get_outputs(), &td.expected);
                    }

                    // Update the genome with the calculate fitness score.
                    // (This is important, as this fitness score is needed to
                    // spawn the next generation correctly.)
                    genome.update_fitness(fitness);

                    if fitness > best_fitness {
                        println!(
                            "Species {} Genome {} increased best fitness to {}",
                            s, g, best_fitness
                        );
                        best_fitness = fitness;
                        best_genome = Some(genome.clone());
                    }
                }
            }
            // Spawn the next generation.
            gene_pool.new_generation();
        }
        println!("Serializing best genome to winner.json");
        serde_json::to_writer(&File::create("winner.json").unwrap(), &best_genome.unwrap())
            .unwrap();
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
