/* This example will train a neural network that will predict whether or not
 * its first input is larger than its second input. Note that due to the small
 * amount of training data the network will usually be far from perfect.
 */

use neuralneat::{Genome, Pool};
use serde_json;
use std::env;
use std::fs::File;

#[path = "common/lib.rs"]
mod common;
use common::load_training_data;

const TRAINING_DATA_STRING: &str = include_str!("compare.csv");
const PERFECT_SCORE: f32 = 50.0;

// This fitness function is used to "score" each genome. Higher scores are
// better ("more fit"), and the genomes with the highest fitness are used
// as "parents" in the next generation, with the lower scoring genomes being
// thrown away.
fn fitness_func(prediction: f32, expected: f32) -> f32 {
    // Our prediction comes in as the raw genome output value - which is
    // guaranteed to be between 0.0 and 1.0. We define the top half of this
    // range to mean "input1 is greater than input2". Therefore, if both
    // this comparison and the expected value or true (or if both are false)
    // - the genome got the right answer for the test data is just tried.
    // This is success, and it gets to increase its fitness!
    if (prediction > 0.5) && expected == 1.0 {
        return 1.0;
    }
    // If not, it got the wrong answer, and gets a score of zero.
    return 0.0;
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: '{} train' to train a new comparer.", args[0]);
        println!("Usage: '{} evaluate serialized_genome.json input1 input2' to evaluate with an existing genome.", args[0]);
        return;
    }

    if args[1] == "train" {
        // One input node for each input in the TrainingData structure
        let input_nodes = 2;
        // One output node with the "prediction"
        let output_nodes = 1;
        // Create a new gene pool with an initial population of genomes
        let mut gene_pool = Pool::with_defaults(input_nodes, output_nodes);

        let training_data = load_training_data(TRAINING_DATA_STRING, 4, 1);

        // These variables are used to keep track of the top performer, so we
        // can write it out later.
        let mut best_genome: Option<Genome> = None;
        let mut best_fitness = 0.0;

        // We use a label here to allow us to break out if we find a genome
        // with a perfect score before we run through all of the generations.
        'outer: for generation in 0..1000 {
            println!("Evaluating generation {}", generation + 1);
            let total_species = gene_pool.len();
            for s in 0..total_species {
                let species = &mut gene_pool[s];
                let genomes_in_species = species.len();
                for g in 0..genomes_in_species {
                    let genome = &mut species[g];
                    let mut fitness = 0.0;

                    for td in &training_data {
                        // Evaluate the genome using the training data as the
                        // initial inputs.
                        genome.evaluate(&td.inputs[0..2].to_vec(), None, None);

                        // We add this to the existing fitness for the genome
                        // to ensure that the genomes with the best score across
                        // all tests will have the highest overall fitness.
                        fitness += fitness_func(genome.get_outputs()[0], td.expected[0]);
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

                    if fitness == PERFECT_SCORE {
                        println!("Found a perfect genome!");
                        break 'outer;
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
        if args.len() < 5 {
            println!("Usage: '{} evaluate serialized_genome.json input1 input2' to evaluate with an existing genome.", args[0]);
            return;
        }
        let mut genome: Genome = serde_json::from_reader(File::open(&args[2]).unwrap()).unwrap();
        let input1 = args[3]
            .parse::<f32>()
            .expect("Couldn't parse input1 as f32");
        let input2 = args[4]
            .parse::<f32>()
            .expect("Couldn't parse input2 as f32");
        genome.evaluate(&vec![input1, input2], None, None);
        if genome.get_outputs()[0] > 0.5 {
            println!("Predicted that {} is greater than {}!", input1, input2);
        } else {
            println!(
                "Predicted that {} is equal or less than {}!",
                input1, input2
            );
        }
    }
}
