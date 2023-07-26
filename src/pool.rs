use log::{debug, info};
use std::fmt;
use std::ops::{Index, IndexMut};

use crate::defaults::*;
use crate::genome::Genome;
use crate::species::{Species, SpeciesStats};

/// Basic statistics about a [Pool] and the [Species] and [Genomes](Genome) contained within it.
pub struct PoolStats {
    pub max_fitness: f32,
    pub max_fitness_ever: f32,
    pub avg_fitness: f32,
    pub species_stats: Vec<SpeciesStats>,
}

/// A "Gene" [Pool] that contains and manages a population of [Genomes](Genome) separated into
/// one or more [Species].
pub struct Pool {
    population_size: usize,
    // Constants used when mutating genomes
    connection_mut_rate: f32,
    node_mut_rate: f32,
    weight_mut_rate: f32,
    perturb_rate: f32,
    weight_mut_step_size: f32,
    disable_mut_rate: f32,
    enable_mut_rate: f32,
    // Constants used in determining whether two genomes are the same species
    excess_coef: f32,
    disjoint_coef: f32,
    weight_diff_coef: f32,
    species_threshold: f32,
    // Constants used when reproducing an individual species
    mut_only_rate: f32,
    mate_only_rate: f32,
    crossover_rate: f32,
    // Constants used when adjusting/normalizing fitness scores
    species_dropoff_age: usize,
    age_significance: f32,
    survival_threshold: f32,
    // Container for all Species (which in turn hold all of the Genomes)
    species: Vec<Species>,
    innovation: u64,
    // Statistics about the Pool
    max_fitness: f32,
    max_fitness_ever: f32,
    avg_fitness: f32,
    generation: u32,
}

impl Pool {
    /// Initialize a new Pool with the given number of `inputs`, `outputs`, and
    /// overriding all of the default constants. (If you wish to use some of
    /// the defaults they are accessible in the [defaults](crate::defaults) module.)
    pub fn new(
        inputs: usize,
        outputs: usize,
        population_size: usize,
        connection_mut_rate: f32,
        node_mut_rate: f32,
        weight_mut_rate: f32,
        perturb_rate: f32,
        weight_mut_step_size: f32,
        disable_mut_rate: f32,
        enable_mut_rate: f32,
        excess_coef: f32,
        disjoint_coef: f32,
        weight_diff_coef: f32,
        species_threshold: f32,
        mut_only_rate: f32,
        mate_only_rate: f32,
        crossover_rate: f32,
        species_dropoff_age: usize,
        age_significance: f32,
        survival_threshold: f32,
    ) -> Self {
        let mut pool = Pool {
            population_size,
            connection_mut_rate,
            node_mut_rate,
            weight_mut_rate,
            perturb_rate,
            weight_mut_step_size,
            disable_mut_rate,
            enable_mut_rate,
            excess_coef,
            disjoint_coef,
            weight_diff_coef,
            species_threshold,
            mut_only_rate,
            mate_only_rate,
            crossover_rate,
            species_dropoff_age,
            age_significance,
            survival_threshold,
            species: vec![Species::empty()],
            innovation: 0,
            max_fitness: 0.0,
            max_fitness_ever: 0.0,
            avg_fitness: 0.0,
            generation: 0,
        };
        for _ in 0..population_size {
            let g = Genome::new(
                inputs,
                outputs,
                &mut pool.innovation,
                connection_mut_rate,
                node_mut_rate,
                weight_mut_rate,
                perturb_rate,
                weight_mut_step_size,
                disable_mut_rate,
                enable_mut_rate,
            );
            pool.add_genome(g);
        }
        return pool;
    }

    /// Initialize a new Pool with the given number of `inputs`, `outputs`,
    /// and use all of the [default](crate::defaults) constants.
    pub fn with_defaults(inputs: usize, outputs: usize) -> Self {
        return Pool::new(
            inputs,
            outputs,
            DEFAULT_POPULATION_SIZE,
            DEFAULT_CONNECTION_MUTATION_CHANCE,
            DEFAULT_NODE_MUTATION_CHANCE,
            DEFAULT_WEIGHT_MUTATION_CHANCE,
            DEFAULT_PERTURB_CHANCE,
            DEFAULT_WEIGHT_STEP_SIZE,
            DEFAULT_DISABLE_NODE_MUTATION_CHANCE,
            DEFAULT_ENABLE_NODE_MUTATION_CHANCE,
            DEFAULT_EXCESS_COEFFICIENT,
            DEFAULT_DISJOINT_COEFFICIENT,
            DEFAULT_WEIGHT_DIFF_COEFFICIENT,
            DEFAULT_SPECIES_THRESHOLD,
            DEFAULT_MUTATE_ONLY_RATE,
            DEFAULT_MATE_ONLY_RATE,
            DEFAULT_CROSSOVER_CHANCE,
            DEFAULT_DROPOFF_AGE,
            DEFAULT_AGE_SIGNIFICANCE,
            DEFAULT_SURVIVAL_THRESHOLD,
        );
    }

    /// Returns the [PoolStats] at a moment in time. (If you need updated
    /// statistics you must call this method each time you need them.)
    pub fn stats(&self) -> PoolStats {
        return PoolStats {
            max_fitness: self.max_fitness,
            max_fitness_ever: self.max_fitness_ever,
            avg_fitness: self.avg_fitness,
            species_stats: self.species.iter().map(|s| s.stats()).collect(),
        };
    }

    /// Returns the number of [Species] in the Pool.
    pub fn len(&self) -> usize {
        return self.species.len();
    }

    /// Returns the total number of [Genomes](Genome) in the Pool.
    /// Note that this _should_ always be the same as the
    /// [default](crate::defaults) population size, but this method
    /// will always return the real number of [Genomes](Genome) in the Pool.
    pub fn population_size(&self) -> usize {
        return self.species.iter().map(|s| s.genomes.len()).sum::<usize>();
    }

    /// Spawn the next generation of [Genomes](Genome). This should only be
    /// done after you've assessed all [Genomes](Genome) in the current
    /// generation and updated their fitness scores. Calling this function will
    /// use the top performing existing [Genomes](Genome) as the basis of the
    /// next generation.
    pub fn new_generation(&mut self) {
        // The behaviour of this method is copied fairly close from the original
        // NEAT 1.2.1's `epoch` method. Some of the implementation details differ
        // due to this implementation trying to avoid keeping unnecessary state,
        // but the end result should be similar or the same.
        info!("Creating new generation");
        info!("Initial pool state: {:?}", self);
        // Calculate some basic statistics needed when creating the next generation.
        self.species.iter_mut().for_each(|s| s.set_max_fitness());
        debug!("After setting max fitness: {:?}", self);
        self.species.iter_mut().for_each(|s| s.set_avg_fitness());
        debug!("After setting avg fitness: {:?}", self);
        // Sort species by max fitness (using raw fitness that doesn't get adjusted for anything)
        self.species.sort_unstable_by(|s1, s2| {
            s2.max_fitness
                .partial_cmp(&s1.max_fitness)
                .unwrap_or_else(|| {
                    panic!("INTERNAL ERROR: Failed to compare two species' max fitness: {s1:?}, {s2:?}");
                })
        });
        debug!("After sorting species: {:?}", self);

        self.max_fitness = self.species[0].max_fitness;
        if self.max_fitness > self.max_fitness_ever {
            self.max_fitness_ever = self.max_fitness;
        }

        let mut obliterated = false;
        for s in &mut self.species {
            // Every 30 generations, flag the species with the lowest fitness score that
            // is also over the age of 20 for obliteration
            if !obliterated && self.generation % 30 == 0 && s.age >= 20 {
                s.adjust_fitness(true, self.species_dropoff_age, self.age_significance);
                obliterated = true;
            } else {
                s.adjust_fitness(false, self.species_dropoff_age, self.age_significance);
            }
        }
        debug!("After adjusting fitness: {:?}", self);

        let total_genomes = self.species.iter().map(|s| s.genomes.len()).sum::<usize>() as u32;

        let mut parents_per_species: Vec<usize> = vec![];
        for s in &mut self.species {
            // - Sort organisms by adjusted fitness
            s.sort_by_fitness();
            // - Decide how many organisms in the species may reproduce based on survival threshold
            //   and population size
            parents_per_species
                .push(((self.survival_threshold * s.genomes.len() as f32) as usize) + 1);
            // The original NEAT 1.2.1 implementation marks organisms for death at this point.
            // This is done by using the sorted organisms and `num_parents` from earlier steps. The top
            // `num_parents` organisms are left alone; the rest are marked for death We're doing this
            // by just saving `parents_per_species`, because only the top N genomes in each species will
            // survive (everything after those in the genomes list will be removed.
            // Actual removal of these species happens later.
        }

        debug!("After sorting and computing parents: {:?}", self);
        debug!("parents_per_species: {:?}", parents_per_species);
        // Compute the overall average fitness by summing the the fitness of all organisms
        // in the population and dividing it by the number of organisms.
        self.avg_fitness = self
            .species
            .iter()
            .map(|s| s.genomes.iter().map(|g| g.fitness).sum::<f32>())
            .sum::<f32>()
            / total_genomes as f32;

        debug!("self.avg_fitness: {}", self.avg_fitness);
        let mut offspring_per_species_genome: Vec<Vec<f32>> = vec![];
        // Compute the expected number of offspring for each organism by dividing its fitness
        // by the overall average fitness.
        for i in 0..self.species.len() {
            offspring_per_species_genome.push(vec![]);
            for j in 0..self.species[i].genomes.len() {
                debug!("genome fitness: {}", self.species[i].genomes[j].fitness);
                offspring_per_species_genome[i]
                    .push(self.species[i].genomes[j].fitness / self.avg_fitness);
            }
        }

        debug!(
            "offspring_per_species_genome: {:?}",
            offspring_per_species_genome
        );
        // Compute the expected offspring per Species by:
        // - Summing the integer parts of the expected offspring for each organism
        // - Adding the fractional parts of the expected offspring for each organism
        // - If the fractional parts are > 1.0, add additional offspring for each
        //   whole number past that (eg: 2.3 gives 2 additional offspring for that species).
        let mut offspring_per_species: Vec<u32> = vec![];
        let mut highest_expecting_offspring = 0;
        let mut highest_expecting_index = 0;
        let mut i = 0;
        for species_offspring in offspring_per_species_genome {
            let mut offspring: u32 = 0;
            let mut skim: f32 = 0.0;
            for genome_offspring in species_offspring {
                offspring += genome_offspring.floor() as u32;
                skim += genome_offspring.fract();
            }
            if skim >= 1.0 {
                offspring += skim.floor() as u32;
            }
            offspring_per_species.push(offspring);
            if offspring > highest_expecting_offspring {
                highest_expecting_index = i;
                highest_expecting_offspring = offspring;
            }
            i += 1;
        }

        // Some precision is lost in skim above for some reason...if we don't have enough
        // offspring allocated, give one to the species expecting the most.
        while offspring_per_species.iter().sum::<u32>() < self.population_size as u32 {
            offspring_per_species[highest_expecting_index] += 1;
        }

        // In the first generation this ends up being 1 for each genome...which I guess makes sense
        // because there's little differention? Probably safe to ignore?
        debug!("offspring_per_species: {:?}", offspring_per_species);

        // In the original NEAT 1.2.1 implementation there's a bunch of other things that happen
        // at this point, none of which we're doing at this time:
        // - Sort the species by max fitness (using original fitness) - skip for now, this looks to
        //   be used as part of stagnation?
        // - Mark the top performing organism in the top performing species as the pop champ
        //   Skip this - it's only informational?
        // - Check if a new population level record is found; update `highest_fitness` and
        //   `highest_last_changed` if it was. Skip - you're not doing stagnation detection yet?
        // - If the population has stagnated (`highest_last_changed > dropoff_age + 5), then:
        //   - I think get rid of all but the top two species, and give them all the reproduction
        //     rights?
        //   - Also reset various stagnation markers
        //   - Probably OK to ignore this for now.
        // - Have high performing species "steal" offspring from other species.

        // Kill off organisms "marked" for death (see comments above when `parents_per_species` is
        // filled out.
        for i in 0..self.species.len() {
            let num_parents = parents_per_species[i];
            self.species[i].genomes.truncate(num_parents);
        }
        debug!("After removing genomes that will not reproduce: {:?}", self);

        // Reproduce each species
        for i in 0..self.species.len() {
            for g in self.species[i].reproduce(
                offspring_per_species[i],
                &self.species,
                &mut self.innovation,
                self.mut_only_rate,
                self.mate_only_rate,
                self.crossover_rate,
                self.connection_mut_rate,
                self.node_mut_rate,
                self.weight_mut_rate,
                self.perturb_rate,
                self.weight_mut_step_size,
                self.disable_mut_rate,
                self.enable_mut_rate,
            ) {
                self.add_genome(g);
            }
        }
        debug!("After adding new genomes from reproduction: {:?}", self);

        // Remove all organisms from the previous generation
        for i in 0..parents_per_species.len() {
            self.species[i].genomes.drain(0..parents_per_species[i]);
        }
        debug!(
            "After removing remaining genomes from previous generation: {:?}",
            self
        );

        // Remove any species that are now empty; age the remaining ones.
        let mut to_remove: Vec<usize> = vec![];
        for i in 0..self.species.len() {
            if self.species[i].len() == 0 {
                to_remove.push(i);
            } else {
                self.species[i].age += 1;
            }
        }

        for i in to_remove.iter().rev() {
            self.species.remove(*i);
        }
        debug!(
            "After removing dead species and aging the others: {:?}",
            self
        );

        self.generation += 1;
    }

    fn add_genome(&mut self, mut genome: Genome) {
        for s in &mut self.species {
            if s.is_same_species(
                &mut genome,
                self.species_threshold,
                self.excess_coef,
                self.disjoint_coef,
                self.weight_diff_coef,
            ) {
                s.add_genome(genome);
                return;
            }
        }
        self.species.push(Species::new(genome));
    }
}

impl Index<usize> for Pool {
    type Output = Species;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.species[index];
    }
}

impl IndexMut<usize> for Pool {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.species[index];
    }
}

impl fmt::Debug for Pool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(
            f,
            "Max fitness: {}, Max fitness ever: {}, Avg fitness: {}, Generation: {}, Population size: {}, Species: {:?}",
            self.max_fitness,
            self.max_fitness_ever,
            self.avg_fitness,
            self.generation,
            self.population_size(),
            self.species
        );
    }
}
