//! `neuralneat` is an implementation of the [NeuroEvolution of Augmenting
//! Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
//! (NEAT) described in the June 2002 paper titled "Evolving Neural
//! Networks through Augmenting Topologies" by Kenneth O. Stanley and Risto
//! Miikkulainen.
//!
//! Much of this implementation was also guided by the [NEAT 1.2.1 source code](
//! https://nn.cs.utexas.edu/?neat).
//!
//! # Basic usage:
//!
//! ```
//! use neuralneat::{Genome, Pool, Trainer, TrainingData};
//!
//! // To do something useful, you need to decide what your training data is!
//! fn load_training_data() -> Vec<TrainingData> {
//!     return vec![];
//! }
//!
//! fn main() {
//!     let input_nodes = 5;
//!     let output_nodes = 1;
//!     // Create an initial pool of Genomes
//!     let mut gene_pool = Pool::with_defaults(input_nodes, output_nodes);
//!     
//!     // Load the data that will be used to train and evolve the Genomes
//!     let training_data: Vec<TrainingData> = load_training_data();
//!     
//!     // A Trainer can manage the process of training a population of Genomes
//!     // over successive generations.
//!     let mut trainer = Trainer::new(training_data);
//!     
//!     trainer.train(
//!         &mut gene_pool,
//!         // Train for 100 generations
//!         100,
//!     );
//!
//!     // The winner!
//!     let best_genome = gene_pool.get_best_genome();
//! }
//! ```

/// The [defaults] module contains the default values of all of the constants
/// used by the [Pool] to create, mutate, and mate [Genomes](Genome).
pub mod defaults;
/// The [evaluation] module contains functions and types related to the process
/// of evaluating and activating the neural network of a [Genome], such as
/// basic activation functions that can be used with hidden and output layers
/// of a [Genome].
pub mod evaluation;
mod genome;
mod pool;
mod species;
mod training;

pub use genome::{Genome, GenomeStats};
pub use pool::{Pool, PoolStats};
pub use species::{Species, SpeciesStats};
pub use training::{TrainingData, Trainer};
