/// The default number of [Genomes](crate::Genome) a Pool will maintain.
pub const DEFAULT_POPULATION_SIZE: usize = 100;
/// The likelyhood that a connection mutation will take place whenever a
/// [Genome](crate::Genome) mutates.
pub const DEFAULT_CONNECTION_MUTATION_CHANCE: f32 = 0.5;
/// The likelyhood that a new node will be added whenever a
/// [Genome](crate::Genome) mutates. Note: node mutation will only happen
/// if a connection mutation does not. Thus, if the connection mutation
/// chance is set to 1.0 (100% chance), this will never happen.
pub const DEFAULT_NODE_MUTATION_CHANCE: f32 = 0.5;
/// The likelyhood that an existing Gene will have its weights altered
/// whenever a [Genome](crate::Genome) mutates. Note: weight mutation
/// will only happen if neither connection mutation nor node mutation happens.
pub const DEFAULT_WEIGHT_MUTATION_CHANCE: f32 = 0.8;
/// The amount of change (+ or -) that a Gene's weight will incur when a weight
/// mutation takes place (if it has not been perturbed).
pub const DEFAULT_WEIGHT_STEP_SIZE: f32 = 0.1;
/// The likelyhood that when a [Genome](crate::Genome) mutates a weight, that
/// it will `perturb` it -- that is to say, randomly assign it a new value
/// between 0.0 and 1.0.
pub const DEFAULT_PERTURB_CHANCE: f32 = 0.1;
/// The likelyhood that when a [Genome](crate::Genome) mutates, it will disable
/// an existing Gene that is enabled. Note: disable mutation will only happen
/// if neither connection mutation nor node mutation happens.
pub const DEFAULT_DISABLE_NODE_MUTATION_CHANCE: f32 = 0.2;
/// The likelyhood that when a [Genome](crate::Genome) mutates, it will enable
/// an existing Gene that is disabled. Note: enable mutation will only happen
/// if neither connection mutation nor node mutation happens.
pub const DEFAULT_ENABLE_NODE_MUTATION_CHANCE: f32 = 0.4;
/// Used in determining whether two [Genomes](crate::Genome) are the same
/// [Species](crate::Species)
pub const DEFAULT_EXCESS_COEFFICIENT: f32 = 1.0;
/// Used in determining whether two [Genomes](crate::Genome) are the same
/// [Species](crate::Species)
pub const DEFAULT_DISJOINT_COEFFICIENT: f32 = 1.0;
/// Used in determining whether two [Genomes](crate::Genome) are the same
/// [Species](crate::Species)
pub const DEFAULT_WEIGHT_DIFF_COEFFICIENT: f32 = 1.0;
/// Used in determining whether two [Genomes](crate::Genome) are the same
/// [Species](crate::Species)
pub const DEFAULT_SPECIES_THRESHOLD: f32 = 1.0;
/// The likelyhood that when a [Genome](crate::Genome) is used as the basis
/// for the next generation, that it will be cloned and mutated only (rather
/// than mated).
pub const DEFAULT_MUTATE_ONLY_RATE: f32 = 0.25;
/// The likelyhood that when a [Genome](crate::Genome) is used as the basis
/// for the next generation via mating, that the resulting offspring will not
/// be mated.
pub const DEFAULT_MATE_ONLY_RATE: f32 = 0.20;
/// The likelyhood that when a [Genome](crate::Genome) is mated, that it will
/// be mated with another [Genome](crate::Genome) from another
/// [Species](crate::Species) instead of its own.
pub const DEFAULT_CROSSOVER_CHANCE: f32 = 0.05;
/// The number of generations that a [Species](crate::Species) must exist
/// without improving its fitness before it is considered stagnant, and
/// receives an extreme penalty to its adjusted fitness score (making it much
/// less likely to reproduce).
pub const DEFAULT_DROPOFF_AGE: usize = 15;
/// The boost that younger species (10 generations or fewer) receive to their
/// adjusted fitness scores.
pub const DEFAULT_AGE_SIGNIFICANCE: f32 = 1.5;
/// Used in determining how many [Genomes](crate::Genome) from a
/// [Species](crate::Species) will be used as the basis for the next
/// generation.
pub const DEFAULT_SURVIVAL_THRESHOLD: f32 = 0.2;
