//! `neuralneat` is an implementation of the [NeuroEvolution of Augmenting
//! Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
//! (NEAT) described in the June 2002 paper titled "Evolving Neural
//! Networks through Augmenting Topologies" by Kenneth O. Stanley and Risto
//! Miikkulainen.
//!
//! Much of this implementation was also guided by the [NEAT 1.2.1 source code](
//! https://nn.cs.utexas.edu/?neat).

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

pub use genome::{Genome, GenomeStats};
pub use pool::{Pool, PoolStats};
pub use species::{Species, SpeciesStats};
