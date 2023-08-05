use log::info;

use crate::evaluation::{basic_eval, sigmoid, ActivationFn, EvaluationFn, TrainingData};
use crate::Pool;

/// A Trainer will manage the training cycle for a population of [Genomes](crate::Genome).
pub struct Trainer {
    /// The data used to train the [Genomes](crate::Genome). Each [crate::Genome] will be given the
    /// [inputs](TrainingData::inputs) from each [TrainingData] as input their networks.
    training_data: Vec<TrainingData>,
    /// A function that can score a [Genome](crate::Genome) after each piece of [TrainingData] is
    /// fed to it. This function will be passed a Vec of the [Genome](crate::Genome)'s outputs and
    /// the [expected](TrainingData::expected) value or values from the [TrainingData]. This
    /// function is expected to assess the [Genome](crate::Genome)'s performance by comparing the
    /// two, and returning an f32 representing its "score". The score from each call to
    /// `evaluate_fn` will be summed together to form the final fitness value of each
    /// [Genome](crate::Genome).
    pub evaluate_fn: EvaluationFn,
    /// The [ActivationFn] to use for the hidden layers of each [Genome](crate::Genome)'s network.
    pub hidden_activation: ActivationFn,
    /// The [ActivationFn] to use for the output layer of each [Genome](crate::Genome)'s network.
    pub output_activation: ActivationFn,
}

impl Trainer {
    /// Create a new Trainer that will use the given `training_data` to train a [Pool]'s
    /// [Genomes](crate::Genome) when [train](Trainer::train) is called. Other parameters
    /// (eg: [evaluate_fn](Trainer::evaluate_fn) may be customized before calling
    /// [train](Trainer::train) by directly setting them on the returned [Trainer].
    pub fn new(training_data: Vec<TrainingData>) -> Self {
        return Trainer {
            training_data,
            evaluate_fn: basic_eval,
            hidden_activation: sigmoid,
            output_activation: sigmoid,
        };
    }

    /// Train the given [gene_pool](Pool) over the course of `generations` generations.
    /// The `training_data` provided when creating the [Trainer] will be used as part of this
    /// process, as well as the current values of [evaluate_fn](Trainer::evaluate_fn),
    /// [hidden_activation](Trainer::hidden_activation) and
    /// [output_activation](Trainer::output_activation).
    pub fn train(self, gene_pool: &mut Pool, generations: usize) {
        let mut best_fitness = 0.0;

        for generation in 0..generations {
            info!("Evaluating generation {}", generation + 1);
            let total_species = gene_pool.len();
            for s in 0..total_species {
                let species = &mut gene_pool[s];
                let genomes_in_species = species.len();
                for g in 0..genomes_in_species {
                    let genome = &mut species[g];

                    let mut fitness = 0.0;

                    for td in &self.training_data {
                        genome.evaluate(
                            &td.inputs,
                            Some(self.hidden_activation),
                            Some(self.output_activation),
                        );
                        fitness += (self.evaluate_fn)(&genome.get_outputs(), &td.expected);
                    }

                    genome.update_fitness(fitness);

                    if fitness > best_fitness {
                        info!(
                            "Species {} Genome {} increased best fitness to {}",
                            s, g, best_fitness
                        );
                        best_fitness = fitness;
                    }
                }
            }
            gene_pool.new_generation();
        }
    }
}
