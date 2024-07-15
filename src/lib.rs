use num_bigint::BigUint;
use pyo3::{prelude::*, Python};
use numpy::{borrow::PyReadonlyArray2, PyArray1, PyArray2};
use anyhow::Result;
mod functions;

#[pyfunction]
fn _generate_errors<'py>(
    py: Python<'py>,
    num_qubits: u64,
    num_samples: u64,
    error_rate: f64,
    num_threads: usize,
    error_weight: Option<u64>,
    rng_seed: Option<BigUint>,
) -> Result<Bound<'py, PyArray2<u8>>> {
    
    let errors = functions::generate_errors(
        num_qubits,
        num_samples,
        error_rate,
        num_threads,
        error_weight,
        rng_seed,
    )?;

    return Ok(PyArray2::from_vec2_bound(py, &errors)?)
}

#[pyfunction]
fn _generate_syndromes<'py>(
    py: Python<'py>,
    check_matrix: PyReadonlyArray2<u8>,
    errors: PyReadonlyArray2<u8>,
    num_threads: usize,
) -> Result<Bound<'py, PyArray2<u8>>> {
    let check_matrix = check_matrix.as_array();
    let errors = errors.as_array();

    let syndromes =
        functions::generate_syndromes(&check_matrix, &errors, num_threads)?;
    
    return Ok(PyArray2::from_vec2_bound(py, &syndromes)?);
}

#[pyfunction]
fn _determine_logical_errors<'py>(
    py: Python<'py>,
    errors: PyReadonlyArray2<u8>,
    predictions: PyReadonlyArray2<u8>,
    num_threads: usize,
) -> Result<Bound<'py, PyArray1<u8>>> {
    let errors = errors.as_array();
    let predictions = predictions.as_array();

    let failures =
        functions::determine_logical_errors(&errors,&predictions,num_threads)?;

    return Ok(PyArray1::from_vec_bound(py, failures));
}

#[pymodule]
fn ismatching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_generate_errors, m)?)?;
    m.add_function(wrap_pyfunction!(_generate_syndromes, m)?)?;
    m.add_function(wrap_pyfunction!(_determine_logical_errors, m)?)?;
    Ok(())
}