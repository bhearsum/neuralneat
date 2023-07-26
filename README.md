# Neural NEAT 

Neural NEAT is a Rust library that implements [Kenneth Stanley's NeuroEvolution of Augmenting Topologies](https://nn.cs.utexas.edu/?neat-c) (NEAT) neural network evolution techniques.

# Project Status

This project is still in its early stages, but contains a basic implementation capable of generating an initial population of genomes and evolving them over successive generations.

The API should be considered _very_ unstable. It may change on little or no notice, and no API stability is guaranteed across version changes. (This will likely change as the project matures.)

# Installation

```
cargo add neuralneat
```

# Usage

The usual flow of evolving a neural network with Neural NEAT is to create a `Pool`, test each `Genome` in the `Pool`, and then spawn a new generation before repeating this process as many times as you want or need. For example:

```
let input_nodes = 5;
let output_nodes = 1;
let gene_pool = Pool::with_defaults(input_nodes, output_nodes);

for generation in 0..10 {
    let total_species = gene_pool.len();
    for s in 0..total_species {
        let species = &mut gene_pool[s];
        let genomes_in_species = species.len();
        for g in 0..genomes_in_species {
            let genome = &mut species[g];

            let mut fitness = 0.0;
            // Feed some training data to the genome, usually a number of times.
            for test in training_data {
                // Calling `evaluate` updates the "activated" values of each
                // Gene in the Genome, most notably the "output" nodes that
                // we use below to assess the performance of the Genome.
                genome.evaluate(...);
            
                // Assess the fitness of the genome by examining its outputs
                fitness += fitness(genome.get_outputs());
            }

            // Update the genome with the fitness value. This will be used to
            // determine which Genomes will form the basis of the next generation.
            genome.update_fitness(fitness);
        }
    }
    gene_pool.new_generation();
}
```

# Examples

Two simple examples are included with this library:

* The `adding` example will train a neural network that can sum its inputs
* The `compare` example will train a neural network that predicts whether or not its
  first input is larger than its second input.

Both of these examples support both training and evaluation. Training will test a number of generations of genomes and serialize the best one to `winner.json`. Evaluation will take a serialized genome, feed it the given inputs, and print the output. This can be used to manually validate the trained genome, and test cases that were not part of the training data.

For example, to train a new `adding` genome, run:

```
cargo run --example adding train
```

You should see output similar to the following:

```
Evaluating generation 1
Species 0 Genome 0 increased best fitness to 0
Species 0 Genome 1 increased best fitness to 0.000030846237
Species 0 Genome 33 increased best fitness to 0.001098452
Species 0 Genome 58 increased best fitness to 0.56081927
Evaluating generation 2
Species 0 Genome 79 increased best fitness to 0.7184653
Evaluating generation 3
Species 0 Genome 79 increased best fitness to 15.087382
Evaluating generation 4
<much more of this redacted>
Evaluating generation 100
Serializing best genome to winner.json
```

Once this process has been completed you can evaluate the winner by hand. For example:

```
$ cargo run --example adding evaluate winner.json 2 5 7 9
Sum of inputs is..........23
$ cargo run --example adding evaluate winner.json 2 5 7 9
Sum of inputs is..........23
$ cargo run --example adding evaluate winner.json 2 53 7 9
Sum of inputs is..........71
$ cargo run --example adding evaluate winner.json 2 53 7 91
Sum of inputs is..........153
$ cargo run --example adding evaluate winner.json 2 53 17 91
Sum of inputs is..........163
$ cargo run --example adding evaluate winner.json 12 53 17 91
Sum of inputs is..........173
$ cargo run --example adding evaluate winner.json 12 53 33317 91
Sum of inputs is..........33473
$ cargo run --example adding evaluate winner.json 12 53 33317 9132
Sum of inputs is..........42514
$ cargo run --example adding evaluate winner.json 1211 53 33317 9132
Sum of inputs is..........43713
```

Note that training a network is inherently random and highly dependent on the training data you give it. Your winning genome may perform differently than above.

# Documentation

Full documentation can be found at xxxxxxxx.
