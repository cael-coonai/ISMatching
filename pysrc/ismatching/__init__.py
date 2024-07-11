from .ismatching import *
from numpy.typing import NDArray
from numpy import ceil, std, sqrt, uint8, ndarray, average, float_, random, int_
from scipy.stats import binom, sem
from typing import Dict, List, Optional, Tuple, Union
from pymatching import Matching


def generate_errors(
    num_qubits: int,
    num_samples: Union[int,NDArray[int_]]=1,
    error_rate: Union[float,NDArray[float_]]=0.5,
    num_threads: int=0,
    error_weight: Optional[Union[int,NDArray[int_]]]=None,
    rng_seed: Optional[Union[int,NDArray[int_]]]=None,
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
    num_samples: int=1,
    error_rate: float=0.5,
    num_threads: int=0,
    error_weight: Optional[int]=None,
    rng_seed: Optional[int]=None,
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
    num_threads: int=0 
) -> NDArray[uint8]:
    # TODO: Make safe and documemnt
    return ismatching._determine_logical_errors(errors, predictions, num_threads)


def generate_syndromes(
    parity_check_matrix: NDArray[uint8],
    errors: NDArray[uint8],
    num_threads: int=0,
)-> NDArray[uint8]:
    return ismatching._generate_syndromes(parity_check_matrix, errors, num_threads)


def get_stats(data: NDArray[uint8]) -> Tuple[float_, float_]:
    mean = average(data)
    stddev = std(data, ddof=1)
    n = len(data)
    return (mean, stddev/(sqrt(n)))


def monte_carlo(
    parity_check_matrix: NDArray[uint8],
    error_rate: float,
    num_samples: int=1,
    num_threads: int=0,
    matching: Optional[Matching]=None,
    rng_seed: Optional[int]=None,
) -> Tuple[float, float]:

    matching = Matching(parity_check_matrix) if matching == None else matching
    
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
    assert(isinstance(predictions, ndarray))

    failures = determine_logical_errors(errors,predictions,num_threads)

    return get_stats(failures)


# class _WeightSamplingEntry:
#     num_samples: int
#     stddev: float
#     mean: float

#     def __init__(self, num_samples:int, stddev:float, mean:float):
#         self.num_samples = num_samples
#         self.stddev = stddev
#         self.mean = mean

#     def add_data(self, num_samples:int, stddev:float, mean:float) -> None:
#         self.mean = (self.num_samples*self.mean + num_samples*mean) \
#                     / (self.num_samples+num_samples)
#         self.stddev = sqrt(((self.num_samples-1)*(self.stddev**2) \
#                             + (num_samples-1)*(stddev**2)) \
#                             / (self.num_samples+num_samples-1))
#         self.num_samples = self.num_samples + num_samples

#     def guess_num_samples_difference(self, stderr_target: float) -> int:
#         total_num_samples = ceil((self.stddev / stderr_target)**2)
#         diff_num_samples = total_num_samples - self.num_samples
#         return diff_num_samples if diff_num_samples > 0 else 0


#     def stats(self) -> Tuple[float, float]:
#         """
#         Returns a tuple containing the mean and standard error.
#         """
#         return (self.mean, self.stddev/sqrt(self.num_samples))

class ImportanceSampling:
    """
    For each weight (key) the accompanying logical error rate and standard error
    of that rate is stored.
    """

    def __init__(
        self,
        parity_check_matrix: NDArray[uint8],
        # initial_num_samples: int,
        code_distance: Optional[int]=None,
        matching: Optional[Matching]=None,
        num_threads: int=0,
        rng_seed: Optional[int]=None,
    ):
        self._matching = Matching(parity_check_matrix) \
            if matching == None else matching
        self._parity_check_matrix = parity_check_matrix
        self.num_threads = num_threads
        self._rng = random.default_rng(rng_seed)
        # self._data: Dict[int, _WeightSamplingEntry] = dict()
        # self._data[0] = _WeightSamplingEntry(0,0,0)
        # if code_distance != None:
        #     for weight in range(code_distance//2):
        #         self._data[weight] = _WeightSamplingEntry(0,0,0)
        self.p_log_by_weight: Dict[int, Tuple[float,float]] = dict()
        self.p_log_by_weight[0] = (0,0)
        if code_distance != None:
            for weight in range(code_distance//2):
                self.p_log_by_weight[weight] = (0,0)


    def num_qubits(self) -> int:
        return self._matching.num_nodes
    

    def add_weight(
        self,
        error_weight: int,
        num_samples: int,
        threshold_err: Optional[float]=None
    ) -> None:
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
        assert(isinstance(predictions,ndarray))

        failures = determine_logical_errors(errors,predictions,self.num_threads)

        self.p_log_by_weight[error_weight] = get_stats(failures)


    def _get_p_log_bound_by_weight(
            self,
            weight: int,
            p_phys: float,
            num_samples: int,
            weight_prob_threshold: Optional[float],
     ) -> Tuple[Tuple[float,float], float]:

        if weight in self.p_log_by_weight.keys():
            p_log, p_log_err = self.p_log_by_weight[weight]
            return ((p_log,p_log),p_log_err)

        elif weight_prob_threshold != None and (
                binom.pmf(weight,self.num_qubits(),p_phys)>=weight_prob_threshold
            ).any():
            self.add_weight(weight, num_samples)
            p_log, p_log_err = self.p_log_by_weight[weight]
            return ((p_log,p_log),p_log_err)

        else:
            return ((0,1),0)


    def p_log_bound(
        self,
        p_phys: float,
        weight_prob_threshold: Optional[float]=None,
        num_samples: int=1,
    ) : #idk what this returns
        p_log_min, p_log_max, p_log_err_to2 = (0,0,0)
        n = self.num_qubits()
        for weight in range(n+1):
            (low, high), err = self._get_p_log_bound_by_weight(
                weight,
                p_phys,
                num_samples,
                weight_prob_threshold
            )
            p_log_min += binom.pmf(weight,self.num_qubits(),p_phys) * low
            p_log_max += binom.pmf(weight,self.num_qubits(),p_phys) * high
            p_log_err_to2+=(binom.pmf(weight,self.num_qubits(),p_phys) * err)**2
        p_log_err = sqrt(p_log_err_to2)
        return ((p_log_min, p_log_max), p_log_err)

