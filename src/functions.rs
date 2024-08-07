use ndarray::Zip;
use num_traits::ToBytes;
use num_bigint::BigUint;
use numpy::ndarray::{ArrayView2, Axis};
use rand::{seq::IteratorRandom, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Binomial, Distribution, Uniform};
use rayon::prelude::*;
use anyhow::Result;
use crate::StabiliserSet;

#[derive(Clone)]
struct GeneratorSelector {
    selection: Vec<bool>
}

impl GeneratorSelector {
    pub fn new(num_generators: usize) -> GeneratorSelector {
        GeneratorSelector {selection: vec![false; num_generators]}
    }

    fn is_all_false(&self) -> bool {
        for generator in &self.selection {
            if *generator == true {
                return false;
            }
        }
        return true;
    }

    fn increment(&mut self) {
        let mut carry = true;
        for idx in 0..self.selection.len() {
            self.selection[idx] ^= carry;
            carry ^= self.selection[idx];
            if carry == false {
                return;
            }
        }
    }
}

impl std::iter::Iterator for GeneratorSelector {
    type Item = Vec<bool>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_all_false() {
            return None;
        }
        let result = self.selection.clone();
        self.increment();
        return Some(result);
    }
}

impl StabiliserSet {

    pub fn from_parity_check(check_matrix:& Vec<Vec<u8>>) -> StabiliserSet {
        let num_generators = check_matrix.len();
        // let mut set = HashSet::new();
        let mut generator_selector = GeneratorSelector::new(num_generators);
        generator_selector.increment();
        
        // if check_matrix.len() > 0 {
        //     set.insert(vec![0; check_matrix[0].len()]);
        // }

        let set = generator_selector.par_bridge()
            .map(|selection| {
                let sum = selection.iter()
                .zip(check_matrix.iter())
                .filter(|(is_selected,_)| **is_selected)
                .map(|(_, generator)| generator.clone())
                .reduce(|mut sum, stabiliser| {
                    for idx in 0..sum.len() {
                        sum[idx] ^= stabiliser[idx]
                    }
                    sum
                })
                .unwrap();
                sum
            })
            .collect::<HashSet<Vec<u8>>>();          

        StabiliserSet { set }
    }
}

fn thread_pool_generator(num_threads:usize) -> Result<rayon::ThreadPool> {
    let num_threads = if num_threads == 0 {num_cpus::get()} else {num_threads};
    let pool = rayon::ThreadPoolBuilder::new();
    Ok(pool.num_threads(num_threads).build()?)
}

fn generate_error_batch(
    rng_seed: [u8;32],
    rng_stream: u64,
    batch_size: u64,
    error_weight: Option<u64>,
    binomial: Binomial,
    uniform: Uniform<u8>,
    num_qubits: u64,
) -> Vec<Vec<u8>> {

    let num_qubits_usize = num_qubits as usize;

    let mut rng = ChaCha8Rng::from_seed(rng_seed);
    rng.set_stream(rng_stream);

    let batch = (0..batch_size).map(move |_| {
        let error_weight = match error_weight {
            Some(w) => w as usize,
            None    => binomial.sample(&mut rng) as usize
        };

        let error_indices = (0..num_qubits_usize)
            .choose_multiple(&mut rng, error_weight);


        let mut error = vec![0;num_qubits_usize*2];

        for qubit_idx in error_indices.into_iter() {
            // 0 := X, 1 := Y, 2 := Z
            let error_type = uniform.sample(&mut rng);

            if error_type >= 1 {
                error[qubit_idx] = 1;
            }
            if error_type <= 1 {
                error[qubit_idx + num_qubits_usize] = 1;
            }
        }
        error
    })
    .collect::<Vec<_>>();
    return batch
}

pub fn generate_errors(
    num_qubits: u64,
    num_samples: u64,
    error_rate: f64,
    num_threads: usize,
    error_weight: Option<u64>,
    rng_seed: Option<BigUint>,
) -> Result<Vec<Vec<u8>>> {

    let batch_size = 10000;
    let batches = num_samples / batch_size;
    let remainder = num_samples % batch_size;


    let rng_seed = match rng_seed {
        Some(n) => {
            let mut seed = [0;32];
            for (dst_byte, src_byte) in seed.iter_mut().zip(n.to_le_bytes()) {
                *dst_byte = src_byte;
            };
            seed
        }
        None => {
            let mut seed = [0;32];
            rand::rngs::OsRng.fill_bytes(&mut seed);
            seed
        }
    };

    let binomial = Binomial::new(num_qubits as u64, error_rate)?;
    let uniform = Uniform::from(0..=2u8);

    let mut errors = generate_error_batch(
        rng_seed,
        0,
        remainder,
        error_weight,
        binomial,
        uniform,
        num_qubits
    );

    thread_pool_generator(num_threads)?.install(|| {
        errors.append(
            &mut (1..=batches)
                .into_par_iter()
                .map(|batch_num|
                    generate_error_batch(
                        rng_seed,
                        batch_num,
                        batch_size,
                        error_weight,
                        binomial,
                        uniform,
                        num_qubits
                    )
                )
                .reduce(
                    || Vec::with_capacity(0),
                    |mut a, mut b| {a.append(&mut b); a}
                )
        );
    });
    
    return Ok(errors);
}

pub fn generate_syndromes(
    check_matrix:&ArrayView2<u8>,
    errors:&ArrayView2<u8>,
    num_threads: usize
) -> Result<Vec<Vec<u8>>> {
    let mut syndromes = Vec::with_capacity(0);

    thread_pool_generator(num_threads)?.install(|| {
        syndromes = errors.axis_iter(Axis(0))
            .into_par_iter()
            .map(|error| {
                let mut syndrome = vec![0;check_matrix.shape()[0]];

                for (syn_idx, check_row) in check_matrix.rows()
                                            .into_iter().enumerate() {
                    for (e_bit, c_bit) in error.iter().zip(check_row) {
                        syndrome[syn_idx] ^= e_bit & c_bit;
                    }
                }

                syndrome
            })
            .collect::<Vec<_>>();
    });

    return Ok(syndromes);
}

pub fn determine_logical_errors(
    errors:&ArrayView2<u8>,
    predictions:&ArrayView2<u8>,
    num_threads: usize
) -> Result<Vec<u8>> {
    let mut failures = Vec::with_capacity(0);

    thread_pool_generator(num_threads)?.install(|| {
        failures = Zip::from(predictions.axis_iter(Axis(0)))
            .and(errors.axis_iter(Axis(0)))
            .into_par_iter()
            .map(|(err, corr)| {
                let mut decoding = vec![0u8; err.len()];

                for idx in 0..err.len() {
                    decoding[idx] = err[idx] ^ corr[idx];
                }

                let z_dec = &decoding[..(decoding.len()/2)];
                let x_dec = &decoding[(decoding.len()/2)..];

                let mut x_failed = 0;
                let mut z_failed = 0;
                for (z_bit, x_bit) in z_dec.into_iter().zip(x_dec.into_iter()) {
                    x_failed ^= x_bit;
                    z_failed ^= z_bit;
                }
                if x_failed == 1 || z_failed == 1 {1} else {0}
            })
            .collect::<Vec<_>>();
    });

    Ok(failures)
}