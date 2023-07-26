/* This example will train a neural network that can sum its inputs */

use neuralneat::{Genome, Pool};
use serde_json;
use std::env;
use std::fs::File;

#[derive(Debug)]
struct TrainingData {
    input1: u32,
    input2: u32,
    input3: u32,
    input4: u32,
    expected: u32,
}

// This data will be fed to each genome as part of training.
const TRAINING_DATA: [TrainingData; 100] = [
    TrainingData {
        input1: 96,
        input2: 78,
        input3: 94,
        input4: 38,
        expected: 306,
    },
    TrainingData {
        input1: 16,
        input2: 31,
        input3: 83,
        input4: 71,
        expected: 201,
    },
    TrainingData {
        input1: 23,
        input2: 15,
        input3: 83,
        input4: 61,
        expected: 182,
    },
    TrainingData {
        input1: 73,
        input2: 63,
        input3: 67,
        input4: 32,
        expected: 235,
    },
    TrainingData {
        input1: 38,
        input2: 9,
        input3: 18,
        input4: 57,
        expected: 122,
    },
    TrainingData {
        input1: 21,
        input2: 57,
        input3: 26,
        input4: 61,
        expected: 165,
    },
    TrainingData {
        input1: 93,
        input2: 66,
        input3: 79,
        input4: 97,
        expected: 335,
    },
    TrainingData {
        input1: 6,
        input2: 41,
        input3: 98,
        input4: 3,
        expected: 148,
    },
    TrainingData {
        input1: 69,
        input2: 75,
        input3: 97,
        input4: 7,
        expected: 248,
    },
    TrainingData {
        input1: 76,
        input2: 6,
        input3: 62,
        input4: 15,
        expected: 159,
    },
    TrainingData {
        input1: 12,
        input2: 69,
        input3: 20,
        input4: 49,
        expected: 150,
    },
    TrainingData {
        input1: 94,
        input2: 62,
        input3: 67,
        input4: 94,
        expected: 317,
    },
    TrainingData {
        input1: 63,
        input2: 56,
        input3: 96,
        input4: 46,
        expected: 261,
    },
    TrainingData {
        input1: 39,
        input2: 96,
        input3: 31,
        input4: 99,
        expected: 265,
    },
    TrainingData {
        input1: 53,
        input2: 99,
        input3: 68,
        input4: 27,
        expected: 247,
    },
    TrainingData {
        input1: 87,
        input2: 69,
        input3: 34,
        input4: 83,
        expected: 273,
    },
    TrainingData {
        input1: 7,
        input2: 65,
        input3: 22,
        input4: 26,
        expected: 120,
    },
    TrainingData {
        input1: 6,
        input2: 26,
        input3: 66,
        input4: 74,
        expected: 172,
    },
    TrainingData {
        input1: 76,
        input2: 68,
        input3: 42,
        input4: 0,
        expected: 186,
    },
    TrainingData {
        input1: 18,
        input2: 36,
        input3: 72,
        input4: 90,
        expected: 216,
    },
    TrainingData {
        input1: 23,
        input2: 65,
        input3: 91,
        input4: 27,
        expected: 206,
    },
    TrainingData {
        input1: 80,
        input2: 54,
        input3: 65,
        input4: 37,
        expected: 236,
    },
    TrainingData {
        input1: 55,
        input2: 99,
        input3: 23,
        input4: 45,
        expected: 222,
    },
    TrainingData {
        input1: 29,
        input2: 21,
        input3: 66,
        input4: 54,
        expected: 170,
    },
    TrainingData {
        input1: 54,
        input2: 57,
        input3: 16,
        input4: 56,
        expected: 183,
    },
    TrainingData {
        input1: 33,
        input2: 22,
        input3: 95,
        input4: 21,
        expected: 171,
    },
    TrainingData {
        input1: 64,
        input2: 96,
        input3: 83,
        input4: 53,
        expected: 296,
    },
    TrainingData {
        input1: 16,
        input2: 25,
        input3: 67,
        input4: 99,
        expected: 207,
    },
    TrainingData {
        input1: 80,
        input2: 7,
        input3: 89,
        input4: 81,
        expected: 257,
    },
    TrainingData {
        input1: 11,
        input2: 88,
        input3: 36,
        input4: 16,
        expected: 151,
    },
    TrainingData {
        input1: 40,
        input2: 49,
        input3: 5,
        input4: 40,
        expected: 134,
    },
    TrainingData {
        input1: 77,
        input2: 85,
        input3: 3,
        input4: 62,
        expected: 227,
    },
    TrainingData {
        input1: 20,
        input2: 89,
        input3: 27,
        input4: 35,
        expected: 171,
    },
    TrainingData {
        input1: 1,
        input2: 77,
        input3: 35,
        input4: 70,
        expected: 183,
    },
    TrainingData {
        input1: 53,
        input2: 9,
        input3: 93,
        input4: 38,
        expected: 193,
    },
    TrainingData {
        input1: 36,
        input2: 30,
        input3: 25,
        input4: 87,
        expected: 178,
    },
    TrainingData {
        input1: 4,
        input2: 6,
        input3: 44,
        input4: 22,
        expected: 76,
    },
    TrainingData {
        input1: 28,
        input2: 83,
        input3: 22,
        input4: 52,
        expected: 185,
    },
    TrainingData {
        input1: 69,
        input2: 5,
        input3: 28,
        input4: 69,
        expected: 171,
    },
    TrainingData {
        input1: 29,
        input2: 93,
        input3: 0,
        input4: 51,
        expected: 173,
    },
    TrainingData {
        input1: 72,
        input2: 36,
        input3: 76,
        input4: 29,
        expected: 213,
    },
    TrainingData {
        input1: 71,
        input2: 22,
        input3: 95,
        input4: 82,
        expected: 270,
    },
    TrainingData {
        input1: 11,
        input2: 95,
        input3: 91,
        input4: 84,
        expected: 281,
    },
    TrainingData {
        input1: 37,
        input2: 26,
        input3: 59,
        input4: 0,
        expected: 122,
    },
    TrainingData {
        input1: 83,
        input2: 59,
        input3: 21,
        input4: 39,
        expected: 202,
    },
    TrainingData {
        input1: 81,
        input2: 52,
        input3: 0,
        input4: 14,
        expected: 147,
    },
    TrainingData {
        input1: 78,
        input2: 72,
        input3: 22,
        input4: 4,
        expected: 176,
    },
    TrainingData {
        input1: 60,
        input2: 91,
        input3: 5,
        input4: 62,
        expected: 218,
    },
    TrainingData {
        input1: 0,
        input2: 27,
        input3: 49,
        input4: 22,
        expected: 98,
    },
    TrainingData {
        input1: 41,
        input2: 47,
        input3: 68,
        input4: 99,
        expected: 255,
    },
    TrainingData {
        input1: 43,
        input2: 75,
        input3: 70,
        input4: 80,
        expected: 268,
    },
    TrainingData {
        input1: 47,
        input2: 57,
        input3: 81,
        input4: 4,
        expected: 189,
    },
    TrainingData {
        input1: 54,
        input2: 82,
        input3: 92,
        input4: 49,
        expected: 277,
    },
    TrainingData {
        input1: 3,
        input2: 44,
        input3: 20,
        input4: 85,
        expected: 152,
    },
    TrainingData {
        input1: 40,
        input2: 46,
        input3: 42,
        input4: 78,
        expected: 206,
    },
    TrainingData {
        input1: 17,
        input2: 59,
        input3: 98,
        input4: 88,
        expected: 262,
    },
    TrainingData {
        input1: 57,
        input2: 64,
        input3: 18,
        input4: 18,
        expected: 157,
    },
    TrainingData {
        input1: 13,
        input2: 62,
        input3: 3,
        input4: 24,
        expected: 102,
    },
    TrainingData {
        input1: 12,
        input2: 3,
        input3: 84,
        input4: 82,
        expected: 181,
    },
    TrainingData {
        input1: 11,
        input2: 67,
        input3: 42,
        input4: 56,
        expected: 176,
    },
    TrainingData {
        input1: 49,
        input2: 71,
        input3: 37,
        input4: 62,
        expected: 219,
    },
    TrainingData {
        input1: 41,
        input2: 51,
        input3: 96,
        input4: 23,
        expected: 211,
    },
    TrainingData {
        input1: 65,
        input2: 83,
        input3: 57,
        input4: 23,
        expected: 228,
    },
    TrainingData {
        input1: 6,
        input2: 45,
        input3: 39,
        input4: 4,
        expected: 94,
    },
    TrainingData {
        input1: 80,
        input2: 71,
        input3: 62,
        input4: 6,
        expected: 219,
    },
    TrainingData {
        input1: 66,
        input2: 29,
        input3: 58,
        input4: 53,
        expected: 206,
    },
    TrainingData {
        input1: 15,
        input2: 68,
        input3: 11,
        input4: 2,
        expected: 96,
    },
    TrainingData {
        input1: 60,
        input2: 13,
        input3: 51,
        input4: 69,
        expected: 193,
    },
    TrainingData {
        input1: 82,
        input2: 92,
        input3: 66,
        input4: 79,
        expected: 319,
    },
    TrainingData {
        input1: 45,
        input2: 98,
        input3: 68,
        input4: 88,
        expected: 299,
    },
    TrainingData {
        input1: 82,
        input2: 73,
        input3: 56,
        input4: 68,
        expected: 279,
    },
    TrainingData {
        input1: 42,
        input2: 78,
        input3: 86,
        input4: 43,
        expected: 249,
    },
    TrainingData {
        input1: 25,
        input2: 82,
        input3: 81,
        input4: 97,
        expected: 285,
    },
    TrainingData {
        input1: 13,
        input2: 11,
        input3: 67,
        input4: 60,
        expected: 151,
    },
    TrainingData {
        input1: 41,
        input2: 12,
        input3: 67,
        input4: 41,
        expected: 161,
    },
    TrainingData {
        input1: 78,
        input2: 95,
        input3: 33,
        input4: 11,
        expected: 217,
    },
    TrainingData {
        input1: 28,
        input2: 39,
        input3: 47,
        input4: 59,
        expected: 173,
    },
    TrainingData {
        input1: 63,
        input2: 49,
        input3: 19,
        input4: 62,
        expected: 193,
    },
    TrainingData {
        input1: 39,
        input2: 76,
        input3: 2,
        input4: 88,
        expected: 205,
    },
    TrainingData {
        input1: 72,
        input2: 98,
        input3: 58,
        input4: 14,
        expected: 242,
    },
    TrainingData {
        input1: 32,
        input2: 85,
        input3: 28,
        input4: 33,
        expected: 178,
    },
    TrainingData {
        input1: 17,
        input2: 36,
        input3: 31,
        input4: 25,
        expected: 109,
    },
    TrainingData {
        input1: 52,
        input2: 14,
        input3: 91,
        input4: 81,
        expected: 238,
    },
    TrainingData {
        input1: 64,
        input2: 52,
        input3: 80,
        input4: 52,
        expected: 248,
    },
    TrainingData {
        input1: 41,
        input2: 39,
        input3: 86,
        input4: 83,
        expected: 249,
    },
    TrainingData {
        input1: 44,
        input2: 93,
        input3: 54,
        input4: 60,
        expected: 251,
    },
    TrainingData {
        input1: 2,
        input2: 14,
        input3: 43,
        input4: 11,
        expected: 70,
    },
    TrainingData {
        input1: 50,
        input2: 81,
        input3: 53,
        input4: 15,
        expected: 199,
    },
    TrainingData {
        input1: 92,
        input2: 13,
        input3: 21,
        input4: 13,
        expected: 139,
    },
    TrainingData {
        input1: 10,
        input2: 98,
        input3: 1,
        input4: 15,
        expected: 124,
    },
    TrainingData {
        input1: 15,
        input2: 93,
        input3: 83,
        input4: 45,
        expected: 236,
    },
    TrainingData {
        input1: 39,
        input2: 46,
        input3: 90,
        input4: 60,
        expected: 235,
    },
    TrainingData {
        input1: 92,
        input2: 73,
        input3: 9,
        input4: 80,
        expected: 254,
    },
    TrainingData {
        input1: 56,
        input2: 45,
        input3: 61,
        input4: 24,
        expected: 186,
    },
    TrainingData {
        input1: 68,
        input2: 2,
        input3: 56,
        input4: 98,
        expected: 224,
    },
    TrainingData {
        input1: 29,
        input2: 30,
        input3: 3,
        input4: 52,
        expected: 114,
    },
    TrainingData {
        input1: 39,
        input2: 10,
        input3: 30,
        input4: 99,
        expected: 178,
    },
    TrainingData {
        input1: 33,
        input2: 80,
        input3: 97,
        input4: 0,
        expected: 210,
    },
    TrainingData {
        input1: 82,
        input2: 41,
        input3: 41,
        input4: 58,
        expected: 222,
    },
    TrainingData {
        input1: 3,
        input2: 39,
        input3: 74,
        input4: 12,
        expected: 128,
    },
];

// This example will use a linear activation function. This will cause each
// hidden and output node to use the sum of all of its inputs as its activation
// value. (By default, a sigmoidal activation function is used, which restricts
// the activation value of a node to a value between -1.0 and 1.0 which
// obviously not be useful in summing integers.)
fn linear_activation(x: f32) -> f32 {
    return x;
}

// This fitness function is used to "score" each genome. Higher scores are
// better ("more fit"), and the genomes with the highest fitness are used
// as "parents" in the next generation, with the lower scoring genomes being
// thrown away.
fn fitness_func(prediction: f32, expected: u32) -> f32 {
    // First find the difference between the expected value and what the genome
    // predicted.
    let difference = (prediction - expected as f32).abs();
    // Then, use some Math (tm) to invert it such that the smaller the
    // difference, the higher the score, with a difference of 0 (aka
    // the genome got the right answer) resulting in the highest possible
    // fitness score.
    let z = (0.5_f32) * difference;
    return (1.0 / z.exp()) * 1000.0;
}

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
            for s in 0..total_species {
                let species = &mut gene_pool[s];
                let genomes_in_species = species.len();
                for g in 0..genomes_in_species {
                    let genome = &mut species[g];

                    let mut fitness = 0.0;
                    for td in TRAINING_DATA {
                        // Evaluate the genome using the training data as the
                        // initial inputs.
                        genome.evaluate(
                            &vec![
                                td.input1 as f32,
                                td.input2 as f32,
                                td.input3 as f32,
                                td.input4 as f32,
                            ],
                            Some(linear_activation),
                            Some(linear_activation),
                        );

                        // We add this to the existing fitness for the genome
                        // to ensure that the genomes with the best score across
                        // all tests will have the highest overall fitness.
                        fitness += fitness_func(genome.get_outputs()[0], td.expected);
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
