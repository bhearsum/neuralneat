use log::{debug, info};
use std::collections::hash_map::{HashMap, Keys};
use std::fmt;
use std::ops::{Index, IndexMut};

use crate::defaults::*;
use crate::evaluation::{ActivationFn, EvaluationFn, TrainingData};
use crate::genome::Genome;
use crate::species::{Species, SpeciesStats};

/// Basic statistics about a [Pool] and the [Species] and [Genomes](Genome) contained within it.
pub struct PoolStats {
    pub max_fitness: f32,
    pub max_fitness_ever: f32,
    pub avg_fitness: f32,
    // TODO: convert to HashMap if we end up storing Species as one
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
    // TODO: don't let this be public
    pub(crate) species: HashMap<usize, Species>,
    next_species_id: usize,
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
            species: HashMap::new(),
            next_species_id: 0,
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

    /// Trains a population of [Genomes](Genome) over `generations` generations.
    /// `training_data` must be a &Vec of [TrainingData]. Each [Genome] will be
    /// gevn the [inputs](TrainingData::inputs) from each [TrainingData] as inputs
    /// to its network. `evaluate_fn` will be called after each item of [TrainingData]
    /// is fed to the [Genome]. This function will be passed a Vec of the [Genome]'s
    /// outputs and the [expected](TrainingData::expected) value or values from the
    /// [TrainingData]. This function is expected to assess the [Genome]'s performance
    /// by comparing the two, and return an f32 representing its "score". The scores
    /// from each call to `evaluate_fn` will be summed together to form the final
    /// fitness value of the [Genome].
    pub fn train_population(
        &mut self,
        generations: usize,
        training_data: &Vec<TrainingData>,
        evaluate_fn: EvaluationFn,
        hidden_activation: Option<ActivationFn>,
        output_activation: Option<ActivationFn>,
    ) {
        let mut best_fitness = 0.0;

        for generation in 0..generations {
            info!("Evaluating generation {}", generation + 1);
            for s in self.species.keys().map(|k| *k).collect::<Vec<usize>>() {
                let species = &mut self[s];
                let genomes_in_species = species.len();
                for g in species.genomes.keys().map(|k| *k).collect::<Vec<usize>>() {
                    let genome = &mut species[g];

                    let mut fitness = 0.0;

                    // let mut td = training_data.next();
                    // while td.is_some() {
                    for td in training_data {
                        genome.evaluate(&td.inputs, hidden_activation, output_activation);
                        fitness += evaluate_fn(&genome.get_outputs(), &td.expected);
                        // td = training_data.next();
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
            // Spawn the next generation.
            self.new_generation();
        }
    }

    /// Returns the [PoolStats] at a moment in time. (If you need updated
    /// statistics you must call this method each time you need them.)
    pub fn stats(&self) -> PoolStats {
        return PoolStats {
            max_fitness: self.max_fitness,
            max_fitness_ever: self.max_fitness_ever,
            avg_fitness: self.avg_fitness,
            species_stats: self.species.values().map(|s| s.stats()).collect(),
        };
    }

    /// Returns the number of [Species] in the Pool.
    pub fn len(&self) -> usize {
        return self.species.len();
    }

    pub fn species_ids(&self) -> Keys<'_, usize, Species> {
        return self.species.keys();
    }

    /// Returns the total number of [Genomes](Genome) in the Pool.
    /// Note that this _should_ always be the same as the
    /// [default](crate::defaults) population size, but this method
    /// will always return the real number of [Genomes](Genome) in the Pool.
    pub fn population_size(&self) -> usize {
        return self.species.values().map(|s| s.genomes.len()).sum::<usize>();
    }

    /// Returns a clone of the best [Genome] in the current population. (Earlier
    /// generations could theoretically have had a better [Genome]. If it is
    /// important to have the best [Genome] _ever_, you should call this method
    /// once per generation to check each generation's best [Genome].)
    pub fn get_best_genome(&self) -> Genome {
        let best_species = self.species.values().reduce(|s1, s2| {
            if s1.max_fitness > s2.max_fitness {
                return s1;
            }
            return s2;
        });
        let best_genome = best_species
            .unwrap()
            .genomes
            .values()
            .reduce(|g1, g2| {
                if g1.max_fitness > g2.max_fitness {
                    return g1;
                }
                return g2;
            })
            .unwrap();

        return best_genome.clone();
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
        self.species.values_mut().for_each(|s| s.set_max_fitness());
        debug!("After setting max fitness: {:?}", self);
        self.species.values_mut().for_each(|s| s.set_avg_fitness());
        debug!("After setting avg fitness: {:?}", self);
        // Sort species by max fitness (using raw fitness that doesn't get adjusted for anything)
        let mut sorted_species: Vec<usize> = self.species.keys().map(|k| *k).collect::<Vec<_>>();
        sorted_species.sort_unstable_by(|k1, k2| {
            self.species[k2].max_fitness
                .partial_cmp(&self.species[k1].max_fitness)
                .unwrap_or_else(|| {
                    panic!("INTERNAL ERROR: Failed to compare two species' max fitness: {k1:?}, {k2:?}");
                })
        });
        debug!("Sorted species keys: {:?}", sorted_species);

        self.max_fitness = self.species[&sorted_species[0]].max_fitness;
        if self.max_fitness > self.max_fitness_ever {
            self.max_fitness_ever = self.max_fitness;
        }

        let mut obliterated = false;
        for s in &mut self.species.values_mut() {
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

        let total_genomes = self.species.values().map(|s| s.genomes.len()).sum::<usize>() as u32;

        let mut parents_per_species: HashMap<usize, usize> = HashMap::new();
        let mut sorted_species_genomes: HashMap<usize, Vec<usize>> = HashMap::new();
        for (k, s) in &mut self.species.iter_mut() {
            // - Sort organisms by adjusted fitness
            sorted_species_genomes.insert(*k, s.sort_by_fitness());
            // - Decide how many organisms in the species may reproduce based on survival threshold
            //   and population size
            parents_per_species
                .insert(*k, ((self.survival_threshold * s.genomes.len() as f32) as usize) + 1);
            // The original NEAT 1.2.1 implementation marks organisms for death at this point.
            // This is done by using the sorted organisms and `num_parents` from earlier steps. The top
            // `num_parents` organisms are left alone; the rest are marked for death We're doing this
            // by just saving `parents_per_species`, because only the top N genomes in each species will
            // survive (everything after those in the genomes list will be removed.
            // Actual removal of these species happens later.
        }

        debug!("After sorting and computing parents: {:?}", self);
        debug!("parents_per_species: {:?}", parents_per_species);
        debug!("sorted_species_gneomes: {:?}", sorted_species_genomes);
        // Compute the overall average fitness by summing the the fitness of all organisms
        // in the population and dividing it by the number of organisms.
        self.avg_fitness = self
            .species
            .values()
            .map(|s| s.genomes.values().map(|g| g.fitness).sum::<f32>())
            .sum::<f32>()
            / total_genomes as f32;

        debug!("self.avg_fitness: {}", self.avg_fitness);
        let mut offspring_per_species_genome: HashMap<usize, Vec<f32>> = HashMap::new();
        // Compute the expected number of offspring for each organism by dividing its fitness
        // by the overall average fitness.
        for k in self.species.keys() {
            offspring_per_species_genome.insert(*k, vec![]);
            for j in self.species[k].genomes.keys() {
                debug!("genome fitness: {}", self.species[k].genomes[j].fitness);
                let n = self.species.get(&k).unwrap().genomes[j].fitness / self.avg_fitness;
                if n == f32::NAN {
                    offspring_per_species_genome.get_mut(&k).unwrap().push(0.0);
                } else {
                    offspring_per_species_genome.get_mut(&k).unwrap().push(n);
                }
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
        let mut offspring_per_species: HashMap<usize, u32> = HashMap::new();
        let mut highest_expecting_offspring = 0;
        let mut highest_expecting_key = offspring_per_species_genome.keys().next().unwrap();
        for (k, species_offspring) in offspring_per_species_genome.iter() {
            let mut offspring: u32 = 0;
            let mut skim: f32 = 0.0;
            for genome_offspring in species_offspring {
                offspring += genome_offspring.floor() as u32;
                skim += genome_offspring.fract();
            }
            if skim >= 1.0 {
                offspring += skim.floor() as u32;
            }
            offspring_per_species.insert(*k, offspring);
            if offspring > highest_expecting_offspring {
                highest_expecting_key = k;
                highest_expecting_offspring = offspring;
            }
        }

        // Some precision is lost in skim above for some reason...if we don't have enough
        // offspring allocated, give one to the species expecting the most.
        while offspring_per_species.values().sum::<u32>() < self.population_size as u32 {
            *offspring_per_species.get_mut(highest_expecting_key).unwrap() += 1;
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
        for k in self.species.keys().map(|k| *k).collect::<Vec<usize>>() {
            let num_parents = parents_per_species[&k];
            let mut to_kill = sorted_species_genomes[&k].clone();
            to_kill.drain(0..num_parents);
            for g in to_kill {
                self.species.get_mut(&k).unwrap().genomes.remove(&g);
            }
        }
        debug!("After removing genomes that will not reproduce: {:?}", self);

        // Reproduce each species
        for k in self.species.keys().map(|k| *k).collect::<Vec<usize>>() {
            for g in self.species[&k].reproduce(
                offspring_per_species[&k],
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
        for i in parents_per_species.keys().map(|k| *k) {
            let to_kill = sorted_species_genomes[&i].clone();
            for g in to_kill {
                self.species.get_mut(&i).unwrap().genomes.remove(&g);
            }
        }
        debug!(
            "After removing remaining genomes from previous generation: {:?}",
            self
        );

        // Remove any species that are now empty; age the remaining ones.
        let mut to_remove: Vec<usize> = vec![];
        for k in self.species.keys().map(|k| *k).collect::<Vec<usize>>() {
            if self.species.get(&k).unwrap().len() == 0 {
                to_remove.push(k);
            } else {
                self.species.get_mut(&k).unwrap().age += 1;
            }
        }

        for k in to_remove.iter() {
            self.species.remove(k).unwrap();
        }
        debug!(
            "After removing dead species and aging the others: {:?}",
            self
        );

        self.generation += 1;
    }

    fn add_genome(&mut self, mut genome: Genome) {
        for s in &mut self.species.values_mut() {
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
        self.species.insert(self.next_species_id, Species::new(genome));
        self.next_species_id += 1;
    }
}

impl Index<usize> for Pool {
    type Output = Species;

    fn index(&self, index: usize) -> &Self::Output {
        return self.species.get(&index).unwrap();
    }
}

impl IndexMut<usize> for Pool {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return self.species.get_mut(&index).unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn new_genome(inputs: usize, outputs: usize, inno: &mut u64) -> Genome {
        return Genome::new(inputs, outputs, inno, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    fn new_genomes(fitnesses: Vec<f32>) -> HashMap<usize, Genome> {
        let mut innovation = 0;
        let mut genomes: HashMap<usize, Genome> = HashMap::new();
        fitnesses.iter().enumerate().map(|(i, f)| {
            let mut g = new_genome(1, 1, &mut innovation);
            g.fitness = *f;
            g.max_fitness = *f;
            genomes.insert(i, g);
        }).collect::<Vec<_>>();
        return genomes;
    }

    #[test]
    fn test_get_best_genome() {
        let mut innovation = 0;
        let mut species: HashMap<usize, Species> = HashMap::new();
        let mut s = Species::empty();
        s.genomes = new_genomes(vec![50.0, 30.0, 10.0, 70.0, 40.0, 20.0, 100.0, 5.0]);
        s.max_fitness = 100.0;
        species.insert(0, s);
        let mut s2 = Species::empty();
        s2.genomes = new_genomes(vec![5.0, 13.0, 1.0, 80.0, 40.0, 30.0, 20.0]);
        s2.max_fitness = 80.0;
        species.insert(1, s2);
        let pool = Pool {
            population_size: 8,
            connection_mut_rate: DEFAULT_CONNECTION_MUTATION_CHANCE,
            node_mut_rate: DEFAULT_NODE_MUTATION_CHANCE,
            weight_mut_rate: DEFAULT_WEIGHT_MUTATION_CHANCE,
            perturb_rate: DEFAULT_PERTURB_CHANCE,
            weight_mut_step_size: DEFAULT_WEIGHT_STEP_SIZE,
            disable_mut_rate: DEFAULT_DISABLE_NODE_MUTATION_CHANCE,
            enable_mut_rate: DEFAULT_ENABLE_NODE_MUTATION_CHANCE,
            excess_coef: DEFAULT_EXCESS_COEFFICIENT,
            disjoint_coef: DEFAULT_DISJOINT_COEFFICIENT,
            weight_diff_coef: DEFAULT_WEIGHT_DIFF_COEFFICIENT,
            species_threshold: DEFAULT_SPECIES_THRESHOLD,
            mut_only_rate: DEFAULT_MUTATE_ONLY_RATE,
            mate_only_rate: DEFAULT_MATE_ONLY_RATE,
            crossover_rate: DEFAULT_CROSSOVER_CHANCE,
            species_dropoff_age: DEFAULT_DROPOFF_AGE,
            age_significance: DEFAULT_AGE_SIGNIFICANCE,
            survival_threshold: DEFAULT_SURVIVAL_THRESHOLD,
            species: species,
            next_species_id: 2,
            innovation: 0,
            max_fitness: 0.0,
            max_fitness_ever: 0.0,
            avg_fitness: 0.0,
            generation: 0,
        };
        let best_genome = pool.get_best_genome();
        assert_eq!(best_genome.max_fitness, 100.0);
    }
}
