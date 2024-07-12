from ismatching import ismatching
from numpy.typing import NDArray
import numpy as np
from scipy.stats import binom, sem
from typing import Dict, List, Optional, Tuple, Union
from pymatching import Matching


def generate_errors(
    num_qubits: int,
    num_samples: int=1,
    error_rate: float=0.5,
    num_threads: int=0,
    error_weight: Optional[int]=None,
    rng_seed: Optional[int]=None,
) -> NDArray[np.uint8]:
    
    return ismatching._generate_errors(
        num_qubits,
        num_samples,
        error_rate,
        num_threads,
        error_weight,
        rng_seed,
    )


def generate_errors_from_check(
    parity_check_matrix: NDArray[np.uint8],
    num_samples: int=1,
    error_rate: float=0.5,
    num_threads: int=0,
    error_weight: Optional[int]=None,
    rng_seed: Optional[int]=None,
) -> NDArray[np.uint8]:
    num_qubits = parity_check_matrix.shape[1]//2
    return generate_errors(
        num_qubits,
        num_samples,
        error_rate,
        num_threads,
        error_weight,
        rng_seed,
    )


def determine_logical_errors(
    errors: NDArray[np.uint8],
    predictions: NDArray[np.uint8],
    num_threads: int=0 
) -> NDArray[np.uint8]:
    # TODO: Make safe and document
    return ismatching._determine_logical_errors(errors, predictions, num_threads)


def generate_syndromes(
    parity_check_matrix: NDArray[np.uint8],
    errors: NDArray[np.uint8],
    num_threads: int=0,
)-> NDArray[np.uint8]:
    return ismatching._generate_syndromes(parity_check_matrix, errors, num_threads)


def get_stats(data: NDArray[np.uint8]) -> Tuple[np.float64, np.float64]:
    mean = np.average(data)
    stddev = np.std(data, ddof=1)
    n = len(data)
    return (mean, stddev/(np.sqrt(n)))


def monte_carlo(
    parity_check_matrix: NDArray[np.uint8],
    error_rate: float,
    num_samples: int=1,
    num_threads: int=0,
    matching: Optional[Matching]=None,
    rng_seed: Optional[int]=None,
) -> Tuple[np.float64, np.float64]:

    matching = Matching(parity_check_matrix) if matching is None else matching
    
    errors = generate_errors_from_check(
        parity_check_matrix,
        num_samples,
        error_rate,
        num_threads,
        rng_seed=rng_seed
    )
    
    syndromes = generate_syndromes(
        parity_check_matrix,
        errors,
        num_threads
    )

    predictions = matching.decode_batch(syndromes)
    # TODO: Reconsider assertion
    assert type(predictions) is np.ndarray

    failures = determine_logical_errors(errors,predictions,num_threads)

    return get_stats(failures)


class _WeightSamplingEntry:
    mean: np.float64
    stddev: np.float64
    num_samples: np.uint64

    def __init__(self, mean:np.float64, stddev:np.float64, num_samples:np.uint64):
        self.mean = mean
        self.stddev = stddev
        self.num_samples = num_samples

    def get_vals(self) -> Tuple[np.float64, np.float64, np.uint64]:
        return self.mean, self.stddev, self.num_samples

    @staticmethod
    def zero():
        return _WeightSamplingEntry(np.uint64(0), np.float64(0), np.float64(0))

    def add_data(self, num_samples:np.uint64, mean:np.float64, stddev:np.float64) -> None:
        self.mean = (self.num_samples*self.mean + num_samples*mean) \
                    / (self.num_samples+num_samples)
        self.stddev = np.sqrt(((self.num_samples-1)*(self.stddev**2) \
                            + (num_samples-1)*(stddev**2)) \
                            / (self.num_samples+num_samples-1))
        self.num_samples = self.num_samples + num_samples

    def guess_num_samples_difference(self, stderr_target: np.float64) -> np.uint64:
        total_num_samples = np.ceil((self.stddev / stderr_target)**2)
        diff_num_samples = total_num_samples - self.num_samples
        return diff_num_samples if diff_num_samples > 0 else np.uint64(0)

    def stderr(self) -> np.float64:
        return np.float64(0) if self.num_samples == 0 else self.stddev/np.sqrt(self.num_samples)

    def stats(self) -> Tuple[np.float64, np.float64]:
        """
        Returns a tuple containing the mean and standard error.
        """
        return (self.mean, self.stderr())

class ImportanceSampling:
    """
    For each weight (key) the accompanying logical error rate std_err and the num
    samples taken for that weight.
    """

    def __init__(
        self,
        parity_check_matrix: NDArray[np.uint8],
        initial_num_samples: int=int(1e5),
        default_error_threshold: float=0.01,
        code_distance: Optional[int]=None,
        matching: Optional[Matching]=None,
        num_threads: int=0,
        rng_seed: Optional[int]=None, # TODO: Replace to allow input of RNG directly
    ):
        self._matching = Matching(parity_check_matrix) \
            if matching is None else matching
        self._parity_check_matrix = parity_check_matrix
        self.num_threads = num_threads
        self._initial_num_samples = initial_num_samples
        self._default_plog_error_threshold = default_error_threshold
        self._rng = np.random.default_rng(rng_seed)
        self._data: Dict[int, _WeightSamplingEntry] = dict()
        self._data[0] = _WeightSamplingEntry.zero()
        if code_distance != None:
            for weight in range(code_distance//2):
                self._data[weight] = _WeightSamplingEntry.zero()


    def num_qubits(self) -> int:
        return self._matching.num_nodes
    
    
    def sample_weight(self, error_weight: int, num_samples: int) -> None:

        print(f"Sampling {error_weight}, {num_samples}", flush=True)

        errors = generate_errors(
            self.num_qubits(),
            num_samples,
            error_weight=error_weight,
            rng_seed=int.from_bytes(self._rng.bytes(32)),
            num_threads=self.num_threads
        )

        syndromes = generate_syndromes(
            self._parity_check_matrix,
            errors,
            self.num_threads
        )

        predictions = self._matching.decode_batch(syndromes)
        assert type(predictions) is np.ndarray

        failures = determine_logical_errors(errors,predictions,self.num_threads)

        if error_weight in self._data:
            self._data[error_weight].add_data(
                np.uint64(num_samples),
                np.average(failures),
                np.std(failures)
            )

        else:
            self._data[error_weight] = _WeightSamplingEntry(
                np.average(failures),
                np.std(failures),
                np.uint64(num_samples),
            )

    
    def get_weights(self) -> List[Tuple[int, Tuple[np.float64, np.float64]]]:
        """
        Each tuple is: (weight, (mean, standard error))
        """
        weights: List[Tuple[int, Tuple[np.float64, np.float64]]] = []
        for w, e in self._data.items():
            weights.append((w, e.stats()))
        return weights



    def _get_p_log_bound_by_weight(
            self,
            weight: int,
            p_phys: float,
            p_log_error_threshold: Optional[float],
            weight_prob_threshold: Optional[float],
            num_samples: Optional[int],
     ) -> Tuple[Tuple[np.float64, np.float64], np.float64]:

        if p_log_error_threshold is None:
            p_log_error_threshold = self._default_plog_error_threshold
        if num_samples is None:
            num_samples = self._initial_num_samples

        num_weights = self.num_qubits()
        binom_pmf = binom.pmf(weight, num_weights, p_phys)
        w_error_theshold = np.inf \
            if ((binom_pmf**2)) <= 1e-300 \
            else np.sqrt((p_log_error_threshold**2)/((binom_pmf**2)*num_weights))

        if weight in self._data.keys():
            p_log, w_p_log_err = self._data[weight].stats()
            while w_p_log_err >= w_error_theshold:
                _, dev, current_num_samples = self._data[weight].get_vals()
                num_samples_diff = np.uint64(np.ceil(
                    (binom_pmf**2) * (dev**2) * num_weights * (1 / (p_log_error_threshold**2))
                )) - current_num_samples
                assert(num_samples_diff > 0)
                self.sample_weight(weight, num_samples_diff)
                p_log, w_p_log_err = self._data[weight].stats()
            return ((p_log,p_log),w_p_log_err)

        elif weight_prob_threshold is not None and binom_pmf >= weight_prob_threshold:
            self.sample_weight(weight, num_samples)
            p_log, w_p_log_err = self._data[weight].stats()
            while w_p_log_err >= w_error_theshold:
                _, dev, current_num_samples = self._data[weight].get_vals()
                num_samples_diff = np.uint64(np.ceil(
                    (binom_pmf**2) * (dev**2) * num_weights * (1 / (p_log_error_threshold**2))
                )) - current_num_samples
                assert(num_samples_diff > 0)
                assert(num_samples_diff < 1e10)
                self.sample_weight(weight, num_samples_diff)
                p_log, w_p_log_err = self._data[weight].stats()
            return ((p_log,p_log),w_p_log_err)

        else:
            return (np.float64(0), np.float64(1)), np.float64()


    def p_log_bound(
        self,
        p_phys: float,
        p_log_error_threshold: Optional[float]=None,
        weight_prob_threshold: Optional[float]=None,
        num_samples: Optional[int]=None,
    ) : #idk what this returns
        p_log_min, p_log_max, p_log_err_to2 = (0,0,0)
        n = self.num_qubits()
        for weight in range(n+1):
            (low, high), err = self._get_p_log_bound_by_weight(
                weight,
                p_phys,
                p_log_error_threshold,
                weight_prob_threshold,
                num_samples,
            )
            p_log_min += binom.pmf(weight,self.num_qubits(),p_phys) * low
            p_log_max += binom.pmf(weight,self.num_qubits(),p_phys) * high
            p_log_err_to2 += (binom.pmf(weight,self.num_qubits(),p_phys)*err)**2
        p_log_err = np.sqrt(p_log_err_to2)
        return (p_log_min, p_log_max), p_log_err

