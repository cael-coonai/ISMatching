use ndarray::Zip;
use num_traits::ToBytes;
use num_bigint::BigUint;
use numpy::ndarray::{ArrayView2, Axis};
use rand::{seq::IteratorRandom, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Binomial, Distribution, Uniform};
use rayon::prelude::*;

fn thread_pool_generator(num_threads:usize) -> rayon::ThreadPool {
    let num_threads = if num_threads == 0 {num_cpus::get()} else {num_threads};
    let pool = rayon::ThreadPoolBuilder::new();
    pool.num_threads(num_threads).build().unwrap()
}

fn generate_error_batch(
    rng_seed: [u8;32],
    rng_stream: u64,
    batch_size: u64,
    error_weight: Option<usize>,
    binomial: Binomial,
    uniform: Uniform<u8>,
    num_qubits: usize,
) -> Vec<Vec<u8>> {

    let mut rng = ChaCha8Rng::from_seed(rng_seed);
    rng.set_stream(rng_stream);
    
    return (0..batch_size).map(move |_| {
        let error_weight = match error_weight {
            Some(w) => w,
            None    => binomial.sample(&mut rng) as usize
        };

        let error_indices = (0..num_qubits)
            .choose_multiple(&mut rng, error_weight);
    

        let mut error = vec![0;num_qubits*2];

        for qubit_idx in error_indices.into_iter() {
            // 0 := X, 1 := Y, 2 := Z
            let error_type = uniform.sample(&mut rng);

            if error_type >= 1 {
                error[qubit_idx] = 1;
            }
            if error_type <= 1 {
                error[qubit_idx + num_qubits] = 1;
            }
        }
        error
    })
    .collect::<Vec<_>>();
}

pub fn generate_errors(
    num_qubits: usize,
    num_samples: u64,
    error_rate: f64,
    num_threads: usize,
    error_weight: Option<usize>,
    rng_seed: Option<BigUint>,
) -> Vec<Vec<u8>> {

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

    let binomial = Binomial::new(num_qubits as u64, error_rate).unwrap();
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

    thread_pool_generator(num_threads).install(|| {
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
    return errors;
}

pub fn generate_syndromes(
    check_matrix:&ArrayView2<u8>,
    errors:&ArrayView2<u8>,
    num_threads: usize
) -> Vec<Vec<u8>> {
    let mut syndromes = Vec::with_capacity(0);

    thread_pool_generator(num_threads).install(|| {
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

    return syndromes;
}

pub fn determine_logical_errors(
    errors:&ArrayView2<u8>,
    predictions:&ArrayView2<u8>,
    num_threads: usize
) -> Vec<u8> {
    let mut failures = Vec::with_capacity(0);

    thread_pool_generator(num_threads).install(|| {
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

    failures
}