use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Keys;
use std::fmt;
use std::ops::{Index, IndexMut};

use crate::genome::{Genome, GenomeStats};

/// Basic statistics about a [Species] and the [Genomes](Genome) contained
/// within it.
pub struct SpeciesStats {
    pub avg_fitness: f32,
    pub max_fitness: f32,
    pub max_fitness_ever: f32,
    pub age: usize,
    pub last_improvement_age: usize,
    pub genome_stats: Vec<GenomeStats>,
}

/// A [Species] contains a number of [Genomes](Genome) that share common
/// structure.
#[derive(Clone)]
pub struct Species {
    pub(crate) genomes: HashMap<usize, Genome>,
    next_genome_id: usize,
    pub(crate) avg_fitness: f32,
    pub(crate) max_fitness: f32,
    pub(crate) max_fitness_ever: f32,
    pub(crate) age: usize,
    pub(crate) last_improvement_age: usize,
}

impl Species {
    pub(crate) fn empty() -> Self {
        return Species {
            genomes: HashMap::new(),
            next_genome_id: 0,
            avg_fitness: 0.0,
            max_fitness: 0.0,
            max_fitness_ever: 0.0,
            age: 0,
            last_improvement_age: 0,
        };
    }

    pub(crate) fn new(genome: Genome) -> Self {
        let mut genomes: HashMap<usize, Genome> = HashMap::new();
        genomes.insert(0, genome);
        return Species {
            genomes,
            next_genome_id: 1,
            avg_fitness: 0.0,
            max_fitness: 0.0,
            max_fitness_ever: 0.0,
            age: 0,
            last_improvement_age: 0,
        };
    }

    /// Returns the [SpeciesStats] at a moment in time. (If you need updated
    /// statistics you must call this method each time you need them.)
    pub fn stats(&self) -> SpeciesStats {
        return SpeciesStats {
            avg_fitness: self.avg_fitness,
            max_fitness: self.max_fitness,
            max_fitness_ever: self.max_fitness_ever,
            age: self.age,
            last_improvement_age: self.last_improvement_age,
            genome_stats: self.genomes.values().map(|g| g.stats()).collect(),
        };
    }

    /// Returns the number of [Genomes](Genome) in the Species.
    pub fn len(&self) -> usize {
        return self.genomes.len();
    }

    pub fn genome_ids(&self) -> Keys<'_, usize, Genome> {
        return self.genomes.keys();
    }

    pub(crate) fn add_genome(&mut self, genome: Genome) {
        self.genomes.insert(self.next_genome_id, genome);
        self.next_genome_id += 1;
    }

    pub(crate) fn is_same_species(
        &mut self,
        genome: &mut Genome,
        threshold: f32,
        excess_coef: f32,
        disjoint_coef: f32,
        weight_diff_coef: f32,
    ) -> bool {
        if self.genomes.len() == 0 {
            // TODO: should this be false ?
            return true;
        }
        return self.get_compatibility_score(genome, excess_coef, disjoint_coef, weight_diff_coef)
            <= threshold;
    }

    pub(crate) fn get_compatibility_score(
        &mut self,
        genome: &mut Genome,
        excess_coef: f32,
        disjoint_coef: f32,
        weight_diff_coef: f32,
    ) -> f32 {
        let genome_keys = self.genomes.keys().map(|k| *k).collect::<Vec<usize>>();
        let n = rand::thread_rng().gen_range(0..genome_keys.len());
        let our_genome = &mut self.genomes.get_mut(&genome_keys[n]).unwrap();
        let (disjoint, excess, weight_diff) =
            Species::get_disjoint_excess_and_weight_diff(our_genome, genome);

        let genome_length = match [
            our_genome.network.edge_weights().collect::<Vec<_>>().len(),
            genome.network.edge_weights().collect::<Vec<_>>().len(),
        ]
        .iter()
        .max()
        {
            Some(n) => *n as f32,
            None => 0.0,
        };

        return ((excess_coef * excess) / genome_length)
            + ((disjoint_coef * disjoint) / genome_length)
            + (weight_diff_coef * weight_diff);
    }

    pub(crate) fn adjust_fitness(
        &mut self,
        obliterate: bool,
        dropoff_age: usize,
        age_significance: f32,
    ) {
        // This method is part of the new generation spawning logic and replicates what
        // NEAT 1.2.1 does, specifically:
        // Adjust the fitness of all species. (See Species::adjust_fitness) This involves:
        // - A penalty for stagnant species or one marked for obliteration
        // - A boost for younger species (to make them a fair shot at survival)
        // - "Sharing fitness" within the species, which to be honest, I still don't
        //   fully understand.
        let is_stagnant = (self.age - self.last_improvement_age) >= dropoff_age;
        let species_size = self.genomes.len() as u32;
        for g in self.genomes.values_mut() {
            // Extreme penalty for stagnation
            if is_stagnant || obliterate {
                g.fitness = g.fitness * 0.01;
            }
            // Bonus for younger species
            if self.age <= 10 {
                g.fitness = g.fitness * age_significance;
            }
            // "Share" fitness within the species
            g.fitness = g.fitness / species_size as f32;
        }
    }

    // TODO: Perhaps this should be a method on Genome instead? or maybe a standalone?
    fn get_disjoint_excess_and_weight_diff(g1: &mut Genome, g2: &mut Genome) -> (f32, f32, f32) {
        let g1_data: HashMap<u64, f32> =
            HashMap::from_iter(g1.network.edge_weights().map(|g| (g.innovation, g.weight)));
        let g2_data: HashMap<u64, f32> =
            HashMap::from_iter(g2.network.edge_weights().map(|g| (g.innovation, g.weight)));
        let g1_innos: HashSet<&u64> = HashSet::from_iter(g1_data.keys());
        let g2_innos: HashSet<&u64> = HashSet::from_iter(g2_data.keys());
        // I think setting both of these to 0 if no innovations were found
        // will cause everything to end up as excess, which seems okay?
        // Probably doesn't matter anyways, because there should never be empty genomes...
        let min: &u64 = match g1_data.keys().min() {
            Some(i) => i,
            None => &0,
        };
        let max: &u64 = match g1_data.keys().max() {
            Some(i) => i,
            None => &0,
        };
        let mut disjoint: f32 = 0.0;
        let mut excess: f32 = 0.0;

        for missing in g2_innos.difference(&g1_innos) {
            if *missing >= min && *missing <= max {
                disjoint += 1.0;
            } else {
                excess += 1.0;
            }
        }

        let mut weight_diff: f32 = 0.0;
        for same in g2_innos.intersection(&g1_innos) {
            weight_diff = (g1_data[same] - g2_data[same]).abs();
        }

        return (disjoint, excess, weight_diff);
    }

    pub(crate) fn set_max_fitness(&mut self) {
        let winner = self.genomes.values().reduce(|g1, g2| {
            if g1.max_fitness > g2.max_fitness {
                return g1;
            } else {
                return g2;
            }
        });
        match winner {
            Some(g) => {
                self.max_fitness = g.max_fitness;
                if self.max_fitness > self.max_fitness_ever {
                    self.last_improvement_age = self.age;
                    self.max_fitness_ever = self.max_fitness;
                }
            }
            None => {}
        }
    }

    pub(crate) fn set_avg_fitness(&mut self) {
        self.avg_fitness =
            self.genomes.values().map(|g| g.fitness).sum::<f32>() / self.genomes.len() as f32;
    }

    // TODO: how do we replicate this with a hashmap? probably have to work around it?
    // is it still used?
    pub(crate) fn sort_by_fitness(&mut self) -> Vec<usize> {
        // Format ourselves once; to avoid doing it with each comparison.
        // Ideally we'd only do this if a None is found, but doing so requires
        // putting `self` in a closure, which the borrow checker is not happy about.
        let debug_self = format!("{self:?}");
        let mut sorted: Vec<(usize, f32)> = self.genomes.iter().map(|(k, g)| (*k, g.fitness)).collect();
        sorted.sort_unstable_by(|(k1, f1), (k2, f2)| f2.partial_cmp(f1).expect(&debug_self));
        return sorted.iter().map(|(k, f)| *k).collect();
    }

    pub(crate) fn reproduce(
        &self,
        num_babies: u32,
        species: &HashMap<usize, Species>,
        innovation: &mut u64,
        mut_only_rate: f32,
        mate_only_rate: f32,
        crossover_rate: f32,
        connection_mut_rate: f32,
        node_mut_rate: f32,
        weight_mut_rate: f32,
        perturb_rate: f32,
        perturb_step_size: f32,
        disable_mut_rate: f32,
        enable_mut_rate: f32,
    ) -> Vec<Genome> {
        let mut babies: Vec<Genome> = vec![];
        let genome_keys = self.genomes.keys().collect::<Vec<&usize>>();

        for i in 0..num_babies {
            // If expected offspring is > 5, clone the top genome (no mutating) - but only once
            // TODO: this is no longer cloning the top genome...but rather whatever happens to be
            // first
            if num_babies > 5 && i == 0 {
                babies.push(self.genomes.values().next().unwrap().clone());
            }
            // Mutate if dice roll on mutate_only constant is good. (And also if the previous generation is 0
            // sized, which seems impossible?)
            else if rand::thread_rng().gen_range(0..=100) as f32 * 0.01 < mut_only_rate {
                let parent = rand::thread_rng().gen_range(0..genome_keys.len());
                let mut baby = self.genomes[genome_keys[parent]].clone();
                baby.mutate(
                    innovation,
                    connection_mut_rate,
                    node_mut_rate,
                    weight_mut_rate,
                    perturb_rate,
                    perturb_step_size,
                    disable_mut_rate,
                    enable_mut_rate,
                );
                babies.push(baby);
            }
            // Otherwise mate
            // This could be within the species or outside of it, depending on another constant
            else {
                let n1 = rand::thread_rng().gen_range(0..genome_keys.len());
                if rand::thread_rng().gen_range(0..=100) as f32 * 0.01 < crossover_rate {
                    // Breed within the species
                    let n2 = rand::thread_rng().gen_range(0..genome_keys.len());
                    let mut baby = Genome::from_parents(&self.genomes[genome_keys[n1]], &self.genomes[genome_keys[n2]]);
                    if rand::thread_rng().gen_range(0..=100) as f32 * 0.01 < mate_only_rate {
                        baby.mutate(
                            innovation,
                            connection_mut_rate,
                            node_mut_rate,
                            weight_mut_rate,
                            perturb_rate,
                            perturb_step_size,
                            disable_mut_rate,
                            enable_mut_rate,
                        );
                    }
                    babies.push(baby);
                } else {
                    // Breed with another species
                    // TODO: this technically could choose our own species too...
                    // TODO: make it more likely that we select better species (see NEAT code)
                    let mut other_species = self;
                    for _ in 0..species.len() {
                        let species_keys = species.keys().collect::<Vec<&usize>>();
                        let n = rand::thread_rng().gen_range(0..species_keys.len());
                        let choice = species_keys[n];
                        // TODO: does using a new borrowed usize here even work?
                        // or does the key need to be the same reference that was pulled out
                        // of `keys`? if so, does this even work across multiple species?
                        if species[choice].len() > 0 {
                            other_species = &species[choice];
                            break;
                        }
                    }
                    let n2 = other_species.genomes.values().next().unwrap();
                    let mut baby =
                        Genome::from_parents(&self.genomes[genome_keys[n1]], n2);
                    if rand::thread_rng().gen_range(0..=100) as f32 * 0.01 < mate_only_rate {
                        baby.mutate(
                            innovation,
                            connection_mut_rate,
                            node_mut_rate,
                            weight_mut_rate,
                            perturb_rate,
                            perturb_step_size,
                            disable_mut_rate,
                            enable_mut_rate,
                        );
                    }
                    babies.push(baby);
                }
            }
        }

        return babies;
    }
}

impl Index<usize> for Species {
    type Output = Genome;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.genomes[&index];
    }
}

impl IndexMut<usize> for Species {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return self.genomes.get_mut(&index).unwrap();
    }
}

impl fmt::Debug for Species {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(
            f,
            "{{ Max fitness: {}, Max fitness ever: {}, Avg fitness: {}, Genomes: {}, Age: {}, Last improvement age: {} }}",
            self.max_fitness,
            self.max_fitness_ever,
            self.avg_fitness,
            self.genomes.len(),
            self.age,
            self.last_improvement_age,
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
            genomes.insert(i, g);
        }).collect::<Vec<_>>();
        return genomes;
    }

    #[test]
    fn test_set_avg_fitness() {
        let mut s = Species::empty();
        s.genomes = new_genomes(vec![10.0, 10.0, 30.0, 100.0, 100.0]);
        s.set_avg_fitness();
        assert_eq!(s.avg_fitness, 50.0);
    }

    // This test doesn't really apply anymore?
    // #[test]
    // fn test_sort_by_fitness() {
    //     let mut s = Species::empty();
    //     s.genomes = new_genomes(vec![50.0, 30.0, 10.0, 70.0, 40.0, 20.0, 100.0, 5.0]);
    //     s.sort_by_fitness();
    //     let sorted: Vec<f32> = s.genomes.values().map(|g| g.fitness).collect();
    //     assert_eq!(sorted, [100.0, 70.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0]);
    // }
}
