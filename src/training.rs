use log::{error, info, warn};
use std::sync::{Arc, Mutex, mpsc};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread;
use std::thread::{ScopedJoinHandle, available_parallelism};

use crate::evaluation::{basic_eval, sigmoid, ActivationFn, EvaluationFn};
use crate::{Genome, Pool};

#[derive(Clone)]
pub struct TrainingData {
    pub inputs: Vec<f32>,
    pub expected: Vec<f32>,
}

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
    /// How many threads to use when evaluating [Genomes](crate::Genome). For optimum performance
    /// this should generally be no higher than the number of CPU cores on the machine running
    /// the training. If this value is set 0, `std::thread::available_parallelism` will be used
    /// to determine how many threads to use.
    // TODO: I don't love `threads` as a name here, but can't come up with anything better
    pub threads: usize,
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
            threads: 1,
        };
    }

    /// Train the given [gene_pool](Pool) over the course of `generations` generations.
    /// The `training_data` provided when creating the [Trainer] will be used as part of this
    /// process, as well as the current values of [evaluate_fn](Trainer::evaluate_fn),
    /// [hidden_activation](Trainer::hidden_activation) and
    /// [output_activation](Trainer::output_activation).
    pub fn train(self, gene_pool: &mut Pool, generations: usize) {
        let n_threads = match self.threads {
            0 => match available_parallelism() {
                Ok(n) => n.get(),
                Err(_) => {
                    warn!("Unable to determine available parallelism; falling back to 1 thread");
                    1
                },
            },
            _ => self.threads,
        };
        // Pretty sure that gene_pool needs to be in an Arc::mutex too....
        // Because each thread needs to be able to lock the Pool (because Genome needs to be
        // locked....)
        // But if we do this, then we probably lose most or all of the performance benefits of
        // threading? Worth benchmarking I guess...
        // Maybe only the pool needs to be an arc mutex?
        // TODO: we might be able to get away without any arcs or mutexes if species and genomes
        // are stored in a HashMap?
        let gp = Arc::new(Mutex::new(&mut gene_pool));
        thread::scope(|s| {
            // Set-up the appropriate number of threads and some channels to communicate with them.
            let mut work_tx: Vec<Sender<&mut Genome>> = vec![];
            let mut control_tx: Vec<Sender<bool>> = vec![];
            let mut handles: Vec<ScopedJoinHandle<()>> = vec![];

            for i in 0..n_threads {
                let thread_training_data = self.training_data.clone();
                let (wtx, wrx) = mpsc::channel::<&mut Genome>();
                let (ctx, crx) = mpsc::channel::<bool>();
                work_tx.push(wtx);
                control_tx.push(ctx);

                // It would preferable, and probably more readable, to put this in a function that
                // returns a JoinHandle, but for the life of me I can't seem to do that and keep the
                // borrow checker happy.
                handles.push(s.spawn(move || loop {
                    match crx.try_recv() {
                        Ok(_) => break,
                        // TODO: should we do something on error?
                        Err(_) => {},
                    }

                    let genome = match wrx.try_recv() {
                        Ok(g) => g.lock().unwrap(),
                        Err(_) => {
                            error!("Error trying to receive Genome in Trainer.worker; exiting thread");
                            break;
                        }
                    };

                    let mut fitness = 0.0;

                    for td in &thread_training_data {
                        genome.evaluate(
                            &td.inputs,
                            Some(self.hidden_activation),
                            Some(self.output_activation),
                        );
                        fitness += (self.evaluate_fn)(&genome.get_outputs(), &td.expected);
                    }

                    genome.update_fitness(fitness);
                }));
            }

            let mut best_fitness = 0.0;
            let mut cur_channel = 0;

            for generation in 0..generations {
                info!("Evaluating generation {}", generation + 1);
                let total_species = gene_pool.len();
                for s in 0..total_species {
                    let species = &mut gene_pool[s];
                    let genomes_in_species = species.len();
                    for g in 0..genomes_in_species {
                        let genome = Arc::new(Mutex::new(&mut species[g]));
                        work_tx[cur_channel].send(genome).expect("Failed to farm out work");
                        if cur_channel == work_tx.len() - 1 {
                            cur_channel = 0;
                        } else {
                            cur_channel += 1
                        }

                        // TODO: how do we track this now? maybe we need the threads to
                        // communicate back?
                        // if fitness > best_fitness {
                        //     info!(
                        //         "Species {} Genome {} increased best fitness to {}",
                        //         s, g, best_fitness
                        //     );
                        //     best_fitness = fitness;
                        // }
                    }
                }
                gene_pool.new_generation();
            }
        });
    }

    // fn worker(self, work_rx: Receiver<&mut Genome>, control_rx: Receiver<bool>, training_data: Vec<TrainingData>) -> JoinHandle<()> {
    //     return thread::spawn(move || loop {
    //         match control_rx.try_recv() {
    //             Ok(_) => break,
    //             // TODO: should we do something on error?
    //             Err(_) => {},
    //         }

    //         let genome = match work_rx.try_recv() {
    //             Ok(g) => g,
    //             Err(_) => {
    //                 error!("Error trying to receive Genome in Trainer.worker; exiting thread");
    //                 break;
    //             }
    //         };

    //         let mut fitness = 0.0;

    //         for td in &self.training_data {
    //             genome.evaluate(
    //                 &td.inputs,
    //                 Some(self.hidden_activation),
    //                 Some(self.output_activation),
    //             );
    //             fitness += (self.evaluate_fn)(&genome.get_outputs(), &td.expected);
    //         }

    //         genome.update_fitness(fitness);
    //     });
    // }
}
