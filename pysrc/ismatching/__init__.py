from ismatching import ismatching
from numpy.typing import NDArray
import numpy as np
from numpy import int64, uint8, uint64, float64, uintp
from scipy.stats import binom
from typing import Dict, List, Optional, Tuple, Union
from pymatching import Matching


def generate_errors(
    num_qubits: Union[int, uint64],
    num_samples: Union[int, uint64] = 1,
    error_rate: Union[float, float64] = 0.5,
    num_threads: Union[int, uintp] = 0,
    error_weight: Optional[Union[int, uint64]] = None,
    rng_seed: Optional[int] = None, ## Up to 32 bytes
) -> NDArray[uint8]:
    
    return ismatching._generate_errors(
        num_qubits,
        num_samples,
        error_rate,
        num_threads,
        error_weight,
        rng_seed,
    )


def generate_errors_from_check(
    parity_check_matrix: NDArray[uint8],
    num_samples: Union[int, uint64] = 1,
    error_rate: Union[float, float64] = 0.5,
    num_threads: Union[int, uintp] = 0,
    error_weight: Optional[Union[int, uint64]] = None,
    rng_seed: Optional[int] = None,
) -> NDArray[uint8]:
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
    errors: NDArray[uint8],
    predictions: NDArray[uint8],
    num_threads: Union[int, uintp] = 0,
) -> NDArray[uint8]:
    # TODO: Make safe and document
    return ismatching._determine_logical_errors(errors, predictions, num_threads)


def generate_syndromes(
    parity_check_matrix: NDArray[uint8],
    errors: NDArray[uint8],
    num_threads: Union[int, uintp]=0,
)-> NDArray[uint8]:
    return ismatching._generate_syndromes(parity_check_matrix, errors, num_threads)


def get_stats(data: NDArray[uint8]) -> Tuple[float64, float64]:
    mean = np.average(data)
    stddev = np.std(data, ddof=1)
    n = len(data)
    return (mean, stddev/(np.sqrt(n)))


def monte_carlo(
    parity_check_matrix: NDArray[uint8],
    error_rate: Union[float, float64],
    num_samples: Union[int, uint64] = 1,
    num_threads: Union[int, uintp] = 0,
    matching: Optional[Matching] = None,
    rng_seed: Optional[int] = None,  ## Up to 32 bytes
) -> Tuple[float64, float64]:

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
    
    mean: float64
    stddev: float64
    num_samples: uint64

    def __init__(
        self,
        mean: Union[float, float64],
        stddev: Union[float, float64],
        num_samples: Union[int, uint64],
    ) -> None:
        self.mean = float64(mean)
        self.stddev = float64(stddev)
        self.num_samples = uint64(num_samples)

    def get_vals(self) -> Tuple[float64, float64, uint64]:
        return self.mean, self.stddev, self.num_samples

    @staticmethod
    def zero():
        return _WeightSamplingEntry(float64(0), float64(0), uint64(1))

    # @staticmethod
    # def not_sampled():
    #     return _WeightSamplingEntry(float64(0.5), float64(0.5), uint64(2))

    def add_data(
        self,
        mean: Union[float, float64],
        stddev: Union[float, float64],
        num_samples: Union[int, uint64],
    ) -> None:
        self.mean = (self.num_samples*self.mean + num_samples*mean) \
                    / (self.num_samples+num_samples)
        self.stddev = np.sqrt(((self.num_samples-1)*(self.stddev**2) \
                            + (num_samples-1)*(stddev**2)) \
                            / (self.num_samples+num_samples-1))
        self.num_samples = self.num_samples + num_samples

    def guess_num_samples_difference(
        self,
        stderr_target: Union[float, float64],
    ) -> uint64:
        total_num_samples = np.ceil((self.stddev / stderr_target)**2)
        diff_num_samples = total_num_samples - self.num_samples
        return diff_num_samples if diff_num_samples > 0 else uint64(0)

    def stderr(self) -> float64:
        # return float64(0) if self.num_samples == 0 else self.stddev/np.sqrt(self.num_samples)
        return self.stddev/np.sqrt(self.num_samples)

    def stats(self) -> Tuple[float64, float64]:
        """
        Returns a tuple containing the mean and standard error.
        """
        return (self.mean, self.stderr())

class ImportanceSampling:
    """
    For each weight (key) the accompanying logical error rate std_err and the num
    samples taken for that weight.
    """

    _matching: Matching
    _parity_check_matrix: NDArray[uint8]
    _num_threads: uintp
    _default_num_samples: uint64
    _default_plog_error_threshold: float64
    _default_weight_probability_threshold: float64
    _rng: np.random.Generator
    _data: Dict[uint64, _WeightSamplingEntry]


    def __init__(
        self,
        parity_check_matrix: NDArray[uint8], ### TODO: Derive parity check matrix from Matching
        initial_num_samples: Union[int, uint64] = int(1e5),
        error_threshold: Union[float, float64] = 1e-2,
        distance: Optional[Union[int, int64]] = None,
        matching: Optional[Matching] = None,
        num_threads: Union[int, uintp] = 0,
        rng_seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:

        self._matching = Matching(parity_check_matrix) \
            if matching is None else matching
        self._parity_check_matrix = parity_check_matrix
        self._num_threads = uintp(num_threads)
        self._default_num_samples = uint64(initial_num_samples)
        self._default_plog_error_threshold = float64(error_threshold)
        self._rng = np.random.default_rng(rng_seed)
        self._data = dict()
        self._data[uint64(0)] = _WeightSamplingEntry.zero()

        if distance != None:
            for weight in range(distance//2+1):
                self._data[uint64(weight)] = _WeightSamplingEntry.zero()
        #     for weight in range(distance//2+1, self.num_qubits+1):
        #         self._data[uint64(weight)] = _WeightSamplingEntry.not_sampled()
        # else:
        #     for weight in range(self.num_qubits+1):
        #         self._data[uint64(weight)] = _WeightSamplingEntry.not_sampled()


    @property
    def num_qubits(self) -> int:
        return self._matching.num_nodes
    
    
    def sample_weight(
        self,
        error_weight: Union[int, uint64],
        num_samples: Union[int, uint64],
    ) -> None:

        error_weight = uint64(error_weight)

        errors = generate_errors(
            self.num_qubits,
            num_samples,
            error_weight = error_weight,
            rng_seed = int.from_bytes(self._rng.bytes(32)),
            num_threads = self._num_threads,
        )

        syndromes = generate_syndromes(
            self._parity_check_matrix,
            errors,
            self._num_threads,
        )

        predictions = self._matching.decode_batch(syndromes)
        assert type(predictions) is np.ndarray ###################### TODO: allow for phenomological noise

        failures = determine_logical_errors(errors,predictions,self._num_threads)

        if error_weight in self._data:
            self._data[error_weight].add_data(
                np.average(failures),
                np.std(failures),
                num_samples,
            )

        else:
            self._data[error_weight] = _WeightSamplingEntry(
                np.average(failures),
                np.std(failures),
                num_samples,
            )

    
    def get_weights(self) -> List[Tuple[uint64, Tuple[float64, float64]]]:
        """
        Each tuple is: (weight, (mean, standard error))
        """
        weights: List[Tuple[uint64, Tuple[float64, float64]]] = []
        for w, e in self._data.items():
            weights.append((w, e.stats()))
        return weights



    def _get_p_log_by_weight(
            self,
            weight: uint64,
            p_phys: float64,
            p_log_error_threshold: float64,
            num_samples: uint64,
     ) -> Tuple[float64, float64]:

        num_weights = self.num_qubits
        binom_pmf = binom.pmf(weight, num_weights, p_phys)

        w_error_theshold = np.inf if (binom_pmf**2)*num_weights <= 1e-300 \
            else np.sqrt((p_log_error_threshold**2)/((binom_pmf**2)*num_weights))


        if weight in self._data.keys():
            p_log, w_p_log_err = self._data[weight].stats()

            # print(f"Weight {weight} in keys with error {w_p_log_err} ({w_error_theshold})", flush=True)

            while w_p_log_err >= w_error_theshold:
                _, dev, current_num_samples = self._data[weight].get_vals()

                num_samples_diff = uint64(np.ceil(
                    (binom_pmf**2) * (dev**2) * num_weights * (1 / (p_log_error_threshold**2))
                )) - current_num_samples
                
                assert num_samples_diff > 0 
                self.sample_weight(weight, num_samples_diff)
                p_log, w_p_log_err = self._data[weight].stats()

            return p_log, w_p_log_err
        elif w_error_theshold < 0.5:
            self.sample_weight(weight, num_samples)
            p_log, w_p_log_err = self._data[weight].stats()
            
            while w_p_log_err >= w_error_theshold:
                _, dev, current_num_samples = self._data[weight].get_vals()

                num_samples_diff = uint64(np.ceil(
                    (binom_pmf**2) * (dev**2) * num_weights * (1 / (p_log_error_threshold**2))
                )) - current_num_samples
                
                assert num_samples_diff > 0 
                self.sample_weight(weight, num_samples_diff)
                p_log, w_p_log_err = self._data[weight].stats()
            
            return p_log, w_p_log_err
        else:
            return float64(1), float64(1)


    def p_log( # TODO: Allow for an array of p_phys to be input
        self,
        p_phys: Union[float, float64],
        p_log_error_threshold: Optional[Union[float, float64]] = None,
        num_samples: Optional[Union[int, uint64]] = None,
    ) : # leaving it up to type inference to determine return

        num_samples = self._default_num_samples \
            if num_samples is None else uint64(num_samples)
        p_log_error_threshold = self._default_plog_error_threshold \
            if p_log_error_threshold is None else float64(p_log_error_threshold)
        
        p_log_min, p_log, p_log_err_sqr = (0,0,0)
        n = self.num_qubits
        for weight in range(n+1):
            mean, err = self._get_p_log_by_weight(
                uint64(weight),
                float64(p_phys),
                p_log_error_threshold,
                num_samples,
            )
            p_log += binom.pmf(weight,self.num_qubits,p_phys) * mean
            p_log_err_sqr += (binom.pmf(weight,self.num_qubits,p_phys)*err)**2
        p_log_err = np.sqrt(p_log_err_sqr)
        return p_log, p_log_err


