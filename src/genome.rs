use log::{debug, error, info};
use petgraph::dot::Dot;
use petgraph::graph::{DefaultIx, DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::evaluation::{sigmoid, ActivationFn};

fn should_mutate(chance: f32) -> bool {
    return (rand::thread_rng().gen_range(0..100) as f32) < (chance * 100.0);
}

/// Basic statistics about a Genome's fitness.
pub struct GenomeStats {
    pub fitness: f32,
    pub raw_fitness: f32,
    pub max_fitness: f32,
}

/// A [Genome] contains a network of Network of Neurons and Genes that when
/// given [the same number of inputs the containing Pool](crate::Pool::new)
/// was given, will produce a Vec with [the same number of outputs that the
/// Pool](crate::Pool::new) was given.
#[derive(Clone, Deserialize, Serialize)]
pub struct Genome {
    pub(crate) network: DiGraph<Neuron, Gene>,
    pub(crate) n_inputs: usize,
    pub(crate) n_outputs: usize,
    pub(crate) fitness: f32,
    raw_fitness: f32,
    pub(crate) max_fitness: f32,
}

impl Genome {
    pub(crate) fn new(
        n_inputs: usize,
        n_outputs: usize,
        innovation: &mut u64,
        connection_mut_rate: f32,
        node_mut_rate: f32,
        weight_mut_rate: f32,
        perturb_rate: f32,
        weight_mut_step_size: f32,
        disable_mut_rate: f32,
        enable_mut_rate: f32,
    ) -> Self {
        let mut network = DiGraph::<Neuron, Gene>::with_capacity(n_inputs + n_outputs, n_inputs);
        let input_idxs = (0..n_inputs)
            .map(|_| network.add_node(Neuron::new()))
            .collect::<Vec<_>>();
        let output_idxs = (0..n_outputs)
            .map(|_| network.add_node(Neuron::new()))
            .collect::<Vec<_>>();
        for input in input_idxs {
            for output in &output_idxs {
                *innovation += 1;
                network.add_edge(input, *output, Gene::new(*innovation));
            }
        }

        let mut genome = Genome {
            network,
            n_inputs,
            n_outputs,
            fitness: 0.0,
            raw_fitness: 0.0,
            max_fitness: 0.0,
        };
        genome.mutate(
            innovation,
            connection_mut_rate,
            node_mut_rate,
            weight_mut_rate,
            perturb_rate,
            weight_mut_step_size,
            disable_mut_rate,
            enable_mut_rate,
        );

        return genome;
    }

    pub(crate) fn from_parents(g1: &Genome, g2: &Genome) -> Self {
        // TODO: improve this based on the NEAT code for multipoint mating
        // 50/50 chance of taking each gene with the same innovation number
        // from each parent, assuming it is enabled. the difference is that the
        // weight could be different depending on the parent
        let mut child = g1.clone();
        for gene in child.network.edge_weights_mut() {
            match g2.gene_by_innovation(gene.innovation) {
                Some(gene2) => {
                    if rand::thread_rng().gen_range(0..=1) == 1 {
                        // TODO: is it true that only the weight changes?
                        // could the nodes it points at change too? i don't think so..
                        gene.weight = gene2.weight;
                    }
                }
                None => {}
            }
        }
        return child;
    }

    /// Returns only the active part of the Genome's network. That is to say
    /// only Neurons that have an activated value != 0.0 AND are connected to
    /// another Neuron with an activated value != 0.0 by an enabled Gene with
    /// a weight greater than 0.0 will be included.
    pub fn active_network(&self) -> DiGraph<Neuron, Gene> {
        return self.network.filter_map(
            |idx, neuron| {
                if idx.index() < self.n_inputs {
                    if neuron.activation_value != 0.0 {
                        return Some(neuron.clone());
                    }
                    return None;
                }
                for edge in self.network.edges_directed(idx, Direction::Incoming) {
                    if edge.weight().enabled && edge.weight().weight > 0.0 {
                        let node = self.network.node_weight(edge.source()).unwrap();
                        if node.activation_value != 0.0 {
                            return Some(neuron.clone());
                        }
                    }
                }
                return None;
            },
            |_, gene| {
                if gene.enabled {
                    return Some(gene.clone());
                }
                return None;
            },
        );
    }

    /// Returns the [GenomeStats] at a moment in time. (If you need updated
    /// statistics you must call this method each time you need them.)
    pub fn stats(&self) -> GenomeStats {
        return GenomeStats {
            fitness: self.fitness,
            raw_fitness: self.raw_fitness,
            max_fitness: self.max_fitness,
        };
    }

    pub fn get_max_fitness(&self) -> f32 {
        return self.max_fitness;
    }

    /// Returns the activated values of the output nodes in a Vec of the same
    /// length as [the outputs parameter the Pool](crate::Pool::new) was
    /// created with.
    pub fn get_outputs(&self) -> Vec<f32> {
        let mut outputs: Vec<f32> = vec![];
        for i in 0..self.n_outputs {
            let node = &self.network[NodeIndex::<DefaultIx>::new(self.n_inputs + i)];
            outputs.push(node.activation_value);
        }
        return outputs;
    }

    /// Update the fitness score of the Genome to `new_fitness`.
    pub fn update_fitness(&mut self, new_fitness: f32) {
        self.fitness = new_fitness;
        if self.fitness > self.max_fitness {
            self.max_fitness = self.fitness
        }
    }

    /// "Evaluate" the Genome using the given `inputs` as values for the input
    /// Neurons. These values will propagate through the Genome's network
    /// ultimately updating the activated values of the output Neurons. These
    /// activated values can then be accessed by calling [Genome::get_outputs].
    ///
    /// # Arguments
    ///
    /// * `inputs` - The values to use for the input Neurons.
    /// * `hidden_activation_fn` - If provided, this function will be called
    ///   to active the raw values of each Neuron in the hidden layer of the
    ///   network. If not provided, the built-in [sigmoid] activation function
    ///   will be used instead.
    /// * `output_activation_fn` - If provided, this function will be called
    ///   to active the raw values of each Neuron in the output layer of the
    ///   network. If not provided, the built-in [sigmoid] activation function
    ///   will be used instead.
    pub fn evaluate(
        &mut self,
        inputs: &Vec<f32>,
        hidden_activation_fn: Option<ActivationFn>,
        output_activation_fn: Option<ActivationFn>,
    ) {
        let activate_hidden = match hidden_activation_fn {
            Some(f) => f,
            None => sigmoid,
        };
        let activate_output = match output_activation_fn {
            Some(f) => f,
            None => sigmoid,
        };
        info!("Evaluating genome...");
        if inputs.len() != self.n_inputs {
            error!(
                "Wrong number of inputs: {}, need: {}",
                inputs.len(),
                self.n_inputs
            );
            panic!(
                "Wrong number of inputs: {}, need: {}",
                inputs.len(),
                self.n_inputs
            );
        }
        let mut i: usize = 0;
        // Set the initial values for the input nodes
        debug!("Genome state before loading inputs: {:?}", self);
        for neuron in self.network.node_weights_mut() {
            if i == inputs.len() {
                break;
            }
            neuron.activation_value = inputs[i];
            i += 1;
        }
        debug!("Genome state after loading inputs: {:?}", self);

        // Write-up of the NEAT 1.2.1 activation algorithm:
        // Until all output nodes have been activated:
        let nodes_minus_inputs = self.network.node_count() - self.n_inputs;
        debug!("node_count: {}", self.network.node_count());
        debug!("nodes_minus_inputs: {}", nodes_minus_inputs);
        let mut active_inputs: Vec<bool> = Vec::with_capacity(nodes_minus_inputs);
        for _ in 0..nodes_minus_inputs {
            active_inputs.push(false);
        }
        let mut activated: Vec<bool> = Vec::with_capacity(nodes_minus_inputs);
        for _ in 0..nodes_minus_inputs {
            activated.push(false);
        }
        let mut bailcount = 0;
        loop {
            debug!("Evaluation loop...bailcount is {}", bailcount);
            if bailcount == 20 {
                info!("Bailing evaluation...");
                break;
            }
            bailcount += 1;
            debug!("activated: {:?}", activated);
            debug!("active_inputs: {:?}", active_inputs);
            let mut outputs_active = true;
            for i in 0..self.n_outputs {
                if !activated[i] {
                    outputs_active = false;
                }
            }
            if outputs_active {
                debug!("All outputs are active!");
                break;
            }
            //   For each non-input node:
            debug!("Start evaluation loop 1");
            for i in 0..(self.network.node_count() - self.n_inputs) {
                debug!("Working on non-input node: {}", i);
                let idx = NodeIndex::new(i + self.n_inputs);
                // * Set its value to 0
                let mut value = 0.0;
                // * Update active flags
                active_inputs[i] = false;
                // * For each node upstream of the node, do this:
                for edge in self.network.edges_directed(idx, Direction::Incoming) {
                    let gene = edge.weight();
                    debug!("Found upstream gene: {:?}", gene);
                    let upstream_neuron = &self.network[edge.source()];
                    debug!("Found upstream neuron: {:?}", upstream_neuron);
                    let is_input = edge.source().index() < self.n_inputs;
                    debug!("Is upstream neuron an input? {}", is_input);
                    let mut should_activate = false;
                    if gene.enabled {
                        if is_input {
                            should_activate = true;
                        } else {
                            if active_inputs[edge.source().index() - self.n_inputs] {
                                should_activate = true;
                            }
                        }
                    }
                    //   * If the node is active OR it is an input (which is always active)
                    //     add its adjusted value (weight * value) to the value of this node.
                    //   * NOTE! you have to account for a link being enabled here. NEAT works
                    //     around this by completely regenerating the network anytime it changes
                    //     where you are maintaining/updating the state over time.
                    if should_activate {
                        active_inputs[i] = true;
                        debug!("Activating neuron from upstream");
                        debug!(
                            "Gene weight is {}, upstream activation value is {}",
                            gene.weight, upstream_neuron.activation_value
                        );
                        value += gene.weight * upstream_neuron.activation_value;
                    }
                }
                debug!("Final value for non-input node {}: {}", i, value);
                let neuron = self.network.node_weight_mut(idx).unwrap();
                neuron.value = value;
            }
            debug!("After loop 1, activated: {:?}", activated);
            debug!("After loop 1, active_inputs: {:?}", active_inputs);
            //  For each non-input node:
            debug!("Starting evaluation loop 2");
            for i in 0..(self.network.node_count() - self.n_inputs) {
                debug!("Working on non-input node: {}", i);
                // If it has been activated at least once
                let neuron = &mut self.network[NodeIndex::new(i + self.n_inputs)];
                debug!("Found neuron: {:?}", neuron);
                if active_inputs[i] {
                    // Calculate the activation value
                    if i < self.n_outputs {
                        debug!("Neuron is output");
                        info!("Raw output value is: {}", neuron.value);
                        neuron.activation_value = activate_output(neuron.value);
                        info!("Activated output value is: {}", neuron.activation_value);
                    } else {
                        debug!("Neuron is hidden");
                        neuron.activation_value = activate_hidden(neuron.value);
                    }
                    activated[i] = true;
                }
            }
        }
    }

    fn gene_by_innovation(&self, inno: u64) -> Option<&Gene> {
        for gene in self.network.edge_weights() {
            if gene.innovation == inno {
                return Some(gene);
            }
        }
        return None;
    }

    pub(crate) fn mutate(
        &mut self,
        innovation: &mut u64,
        connection_mut_rate: f32,
        node_mut_rate: f32,
        weight_mut_rate: f32,
        perturb_rate: f32,
        weight_mut_step_size: f32,
        disable_mut_rate: f32,
        enable_mut_rate: f32,
    ) {
        if should_mutate(connection_mut_rate) {
            self.mutate_connection(innovation);
        } else if should_mutate(node_mut_rate) {
            self.mutate_node(innovation);
        } else {
            if should_mutate(weight_mut_rate) {
                self.mutate_weights(perturb_rate, weight_mut_step_size);
            }
            if should_mutate(disable_mut_rate) {
                self.mutate_enable_disable(false);
            }
            if should_mutate(enable_mut_rate) {
                self.mutate_enable_disable(true);
            }
        }
    }

    fn mutate_weights(&mut self, perturb_rate: f32, weight_mut_step_size: f32) {
        for gene in self.network.edge_weights_mut() {
            if should_mutate(perturb_rate) {
                // The extra multiplying and dividing is to ensure we don't end
                // up with numbers like .300000000001 due to inexact float math.
                // I don't really understand it to be honest...but it works.
                gene.weight =
                    ((rand::thread_rng().gen_range(0..10) as f32 * 0.1) * 10.0).round() / 10.0;
            } else {
                if rand::thread_rng().gen_range(0..=1) > 0 {
                    gene.weight += weight_mut_step_size;
                } else {
                    gene.weight -= weight_mut_step_size;
                }
            }
        }
    }

    fn mutate_connection(&mut self, innovation: &mut u64) {
        // Building a vec of all of the node indexes might be problematic
        // if the total number of nodes is too large...but it's probably fine.
        let mut nodes = (0..self.network.node_count()).collect::<Vec<_>>();
        nodes.shuffle(&mut thread_rng());

        // Find two nodes that have no existing connection, and add one.
        // TODO: should input nodes be able to connect to other inputs nodes?
        // right now they can...
        for n1 in &nodes {
            let idx1 = NodeIndex::<DefaultIx>::new(*n1);
            for n2 in &nodes {
                if n1 == n2 {
                    continue;
                }
                let idx2 = NodeIndex::<DefaultIx>::new(*n2);
                if self.network.find_edge(idx1, idx2).is_none() {
                    *innovation += 1;
                    self.network.add_edge(idx1, idx2, Gene::new(*innovation));
                    return;
                }
            }
        }
    }

    fn mutate_node(&mut self, innovation: &mut u64) {
        /* From the original NEAT paper:
         * In the add node mutation, an existing connection is split and the
         * new node placed where the old connection used to be. The old
         * connection is disabled and two new connections are added to the
         * genome. The new connection leading into the new node receives a
         * weight of 1, and the new connection leading out receives the same
         * weight as the old connection.
         */

        // First pull a random edge that's not yet disabled...
        let mut edge_idx =
            EdgeIndex::<DefaultIx>::new(rand::thread_rng().gen_range(0..self.network.edge_count()));
        // Note to future self: `old_connection` is borrowing `self.network`,
        // which means it cannot be used again until the last usage of
        // `old_connection`, lest you make the borrow checker very unhappy.
        let mut old_connection = &mut self.network[edge_idx];

        for i in 0..101 {
            if i == 100 {
                // OK...we tried really hard. Time to give it up.
                return;
            }
            if old_connection.enabled {
                break;
            }
            edge_idx = EdgeIndex::<DefaultIx>::new(
                rand::thread_rng().gen_range(0..self.network.edge_count()),
            );
            old_connection = &mut self.network[edge_idx];
        }
        // Now that we've got a non-disabled connection, disable it.
        old_connection.enabled = false;

        // Next we create the new node and new connections for it.
        let new_node = Neuron::new();
        *innovation += 1;
        let mut upstream_connection = Gene::new(*innovation);
        // The upstream connection gets a fixed weight of 1.0.
        upstream_connection.weight = 1.0;
        *innovation += 1;
        let mut downstream_connection = Gene::new(*innovation);
        // The downstream connection takes on the weight of the connection
        // we've split.
        downstream_connection.weight = old_connection.weight;
        // We're now done with `old_connection`, so the borrow on
        // `self.network` has ended, and we can use it again.

        // Now, we pull the indexes for the endpoints of the edge we chose.
        let (upstream_idx, downstream_idx) = match self.network.edge_endpoints(edge_idx) {
            Some((u, d)) => (u, d),
            // This _should_ be impossible since we chose an edge that was
            // connected to two nodes...but let's not crash over it.
            None => return,
        };

        // Finally, we insert the new node and edges
        let new_idx = self.network.add_node(new_node);
        self.network
            .add_edge(upstream_idx, new_idx, upstream_connection);
        self.network
            .add_edge(new_idx, downstream_idx, downstream_connection);
    }

    fn mutate_enable_disable(&mut self, type_: bool) {
        let mut candidates: Vec<&mut Gene> = self
            .network
            .edge_weights_mut()
            .filter(|g| g.enabled != type_)
            .collect();
        if candidates.len() > 0 {
            let choice = rand::thread_rng().gen_range(0..candidates.len());
            candidates[choice].enabled = !candidates[choice].enabled;
        }
    }
}

impl fmt::Display for Genome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(
            f,
            "Raw Fitness: {}, Fitness: {}, Max Fitness: {}",
            self.raw_fitness, self.fitness, self.max_fitness,
        );
    }
}

impl fmt::Debug for Genome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(f, "raw: {}, Network: {:?}", self.raw_fitness, Dot::new(&self.network));
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Neuron {
    pub(crate) activation_value: f32,
    pub(crate) value: f32,
}

impl Neuron {
    pub(crate) fn new() -> Self {
        return Neuron {
            activation_value: 0.0,
            value: 0.0,
        };
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(f, "{}", self.value,);
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Gene {
    pub(crate) weight: f32,
    pub(crate) enabled: bool,
    pub(crate) innovation: u64,
}

impl Gene {
    pub(crate) fn new(innovation: u64) -> Self {
        return Gene {
            weight: 0.5,
            enabled: true,
            innovation,
        };
    }
}

impl fmt::Display for Gene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(
            f,
            "weight: {}, enabled: {}, inno: {}",
            self.weight, self.enabled, self.innovation,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_genome(inputs: usize, outputs: usize, inno: &mut u64) -> Genome {
        return Genome::new(inputs, outputs, inno, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    #[test]
    fn test_new_genome_no_mutations() {
        let mut innovation = 0;
        let genome = new_genome(5, 3, &mut innovation);
        assert_eq!(innovation, 15);
        assert_eq!(genome.network.node_count(), 8);
        assert_eq!(genome.network.edge_count(), 15);
        assert_eq!(genome.n_inputs, 5);
        assert_eq!(genome.n_outputs, 3);
        let disabled_edges = genome.network.edge_weights().filter(|e| !e.enabled).count();
        let altered_edge_weights = genome
            .network
            .edge_weights()
            .filter(|e| e.weight != 0.5)
            .count();
        assert_eq!(disabled_edges, 0);
        assert_eq!(altered_edge_weights, 0);
    }

    #[test]
    fn test_new_genome_connection_mut() {
        let mut innovation = 0;
        let genome = Genome::new(5, 3, &mut innovation, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(innovation, 16);
        assert_eq!(genome.network.node_count(), 8);
        assert_eq!(genome.network.edge_count(), 16);
        assert_eq!(genome.n_inputs, 5);
        assert_eq!(genome.n_outputs, 3);
        let disabled_edges = genome.network.edge_weights().filter(|e| !e.enabled).count();
        let altered_edge_weights = genome
            .network
            .edge_weights()
            .filter(|e| e.weight != 0.5)
            .count();
        assert_eq!(disabled_edges, 0);
        assert_eq!(altered_edge_weights, 0);
    }

    #[test]
    fn test_new_genome_node_mut() {
        let mut innovation = 0;
        let genome = Genome::new(5, 3, &mut innovation, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(innovation, 17);
        assert_eq!(genome.network.node_count(), 9);
        assert_eq!(genome.network.edge_count(), 17);
        assert_eq!(genome.n_inputs, 5);
        assert_eq!(genome.n_outputs, 3);
    }

    #[test]
    fn test_genome_from_parents() {
        let mut innovation = 0;
        let g1 = new_genome(100, 1, &mut innovation);
        let mut g2 = g1.clone();
        for edge in g2.network.edge_weights_mut() {
            edge.weight = 50.0;
        }
        let baby = Genome::from_parents(&g1, &g2);
        for weight in baby.network.edge_weights().map(|e| e.weight) {
            // Default weight is 0.5, and we adjusted all of the other parents'
            // weights to 50.0. Not the best test, but probably good enough for
            // now with 100 nodes...
            assert!(
                weight == 0.5 || weight == 50.0,
                "unexpected weight: {}",
                weight
            );
        }
    }

    #[test]
    fn test_get_outputs_one_output() {
        let mut innovation = 0;
        let mut genome = new_genome(5, 1, &mut innovation);
        let idx = NodeIndex::new(5);
        genome.network[idx].activation_value = 5.0;
        assert_eq!(genome.get_outputs(), [5.0]);
    }

    #[test]
    fn test_get_outputs_multiple() {
        let mut innovation = 0;
        let mut genome = new_genome(5, 3, &mut innovation);
        for i in 5..=7 {
            let idx = NodeIndex::new(i);
            genome.network[idx].activation_value = i as f32;
        }
        assert_eq!(genome.get_outputs(), [5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_mutate_connection() {
        let mut innovation = 0;
        let mut genome = new_genome(5, 1, &mut innovation);
        assert_eq!(genome.network.edge_count(), 5);
        genome.mutate(&mut innovation, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(
            genome.network.edge_count(),
            6,
            "Expected 6 edges after connection mutation!"
        );
    }

    #[test]
    fn test_mutate_node() {
        let mut innovation = 0;
        let mut genome = new_genome(5, 1, &mut innovation);
        assert_eq!(genome.network.node_count(), 6);
        assert_eq!(genome.network.edge_count(), 5);
        genome.mutate(&mut innovation, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(
            genome.network.node_count(),
            7,
            "Expected 7 nodes after connection mutation!"
        );
        assert_eq!(
            genome.network.edge_count(),
            7,
            "Expected 7 edges after connection mutation!"
        );
        let disabled_edges = genome.network.edge_weights().filter(|e| !e.enabled).count();
        assert_eq!(
            disabled_edges, 1,
            "Expected one disabled edge after node mutation"
        );
    }

    #[test]
    fn test_mutate_weights_unperturbed() {
        let mut innovation = 0;
        let mut genome = new_genome(1, 1, &mut innovation);
        assert_eq!(genome.network.node_count(), 2);
        assert_eq!(genome.network.edge_count(), 1);
        genome.mutate(&mut innovation, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, 0.0);
        let idx = EdgeIndex::new(0);
        let weight = genome.network[idx].weight;
        assert!(
            weight == 0.4 || weight == 0.6,
            "Edge weight should've increased or decreased"
        );
    }

    #[test]
    fn test_mutate_weights_perturbed() {
        let mut innovation = 0;
        let mut genome = new_genome(1, 1, &mut innovation);
        assert_eq!(genome.network.node_count(), 2);
        assert_eq!(genome.network.edge_count(), 1);
        let idx = EdgeIndex::new(0);
        // We set this to an impossible value so we can easily check
        // that it was overwritten.
        genome.network[idx].weight = -1.0;
        genome.mutate(&mut innovation, 0.0, 0.0, 1.0, 1.0, 0.1, 0.0, 0.0);
        let weight = genome.network[idx].weight;
        assert!(
            weight != -1.0,
            "Edge weight should've increased or decreased"
        );
    }

    #[test]
    fn test_mutate_enable_disable() {
        let mut innovation = 0;
        let mut genome = new_genome(1, 1, &mut innovation);
        assert_eq!(genome.network.node_count(), 2);
        assert_eq!(genome.network.edge_count(), 1);
        let idx = EdgeIndex::new(0);
        genome.mutate(&mut innovation, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        assert_eq!(
            genome.network[idx].enabled, false,
            "Edge should've been disabled"
        );
        genome.mutate(&mut innovation, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert_eq!(
            genome.network[idx].enabled, true,
            "Edge should've been enabled"
        );
    }

    #[test]
    fn test_active_network() {
        let mut innovation = 0;
        let mut network = DiGraph::<Neuron, Gene>::with_capacity(5, 6);
        // input -> hidden -> output, with input activated
        let mut output_neuron = Neuron::new();
        output_neuron.activation_value = 5.0;
        let output = network.add_node(output_neuron);
        let mut input_neuron = Neuron::new();
        input_neuron.activation_value = 4.0;
        let input = network.add_node(input_neuron);
        let mut hidden_neuron = Neuron::new();
        hidden_neuron.activation_value = 3.0;
        let hidden = network.add_node(hidden_neuron);

        let mut edge1_gene = Gene::new(innovation);
        innovation += 1;
        edge1_gene.enabled = false;
        network.add_edge(input, output, edge1_gene);
        let mut edge2_gene = Gene::new(innovation);
        innovation += 1;
        edge2_gene.weight = 0.4;
        let edge2 = network.add_edge(input, hidden, edge2_gene);
        let mut edge3_gene = Gene::new(innovation);
        innovation += 1;
        edge3_gene.weight = 0.6;
        let edge3 = network.add_edge(hidden, output, edge3_gene);

        // input2 -> hidden2 -> output, with input2 not activated
        let input2 = network.add_node(Neuron::new());
        let hidden2 = network.add_node(Neuron::new());

        let mut edge4 = Gene::new(innovation);
        innovation += 1;
        edge4.enabled = false;
        network.add_edge(input2, output, edge4);
        let edge5 = Gene::new(innovation);
        network.add_edge(input2, hidden2, edge5);
        let edge6 = Gene::new(innovation);
        network.add_edge(hidden2, output, edge6);

        let genome = Genome {
            network: network.clone(),
            n_inputs: 2,
            n_outputs: 1,
            fitness: 0.0,
            raw_fitness: 0.0,
            max_fitness: 0.0,
        };
        let active_network = genome.active_network();
        assert_eq!(
            active_network.raw_nodes()[0].weight,
            network.node_weight(output).unwrap().clone()
        );
        assert_eq!(
            active_network.raw_nodes()[1].weight,
            network.node_weight(input).unwrap().clone()
        );
        assert_eq!(
            active_network.raw_nodes()[2].weight,
            network.node_weight(hidden).unwrap().clone()
        );
        assert_eq!(
            active_network.raw_edges()[0].weight,
            network.edge_weight(edge2).unwrap().clone()
        );
        assert_eq!(
            active_network.raw_edges()[1].weight,
            network.edge_weight(edge3).unwrap().clone()
        );
    }

    // TODO: should have tests for evaluate as well...but it's so much work.
}
