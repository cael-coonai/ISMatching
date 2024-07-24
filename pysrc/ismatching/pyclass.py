from ismatching import ismatching # Rust Functions
from numpy.typing import NDArray
import numpy as np
from numpy import int64, uint8, uint64, float64, uintp
from scipy.stats import binom
from typing import Dict, List, Optional, Tuple, Union
from pymatching import Matching
import ismatching.pyfunctions as functions

class _WeightSamplingEntry:
    """
    A class containing the data for samples of a particularly weighted error
    for use in ImportanceSampling.
    """
    
    mean: float64
    stddev: float64
    num_samples: uint64

    def __init__(
        self,
        mean: Union[float, float64],
        stddev: Union[float, float64],
        num_samples: Union[int, uint64],
    ) -> None:
        """
        Constructor of the _WeightSamplingEntry class.
        """
        self.mean = float64(mean)
        self.stddev = float64(stddev)
        self.num_samples = uint64(num_samples)

    def values(self) -> Tuple[float64, float64, uint64]:
        """
        Returns the values contained.
        """
        return self.mean, self.stddev, self.num_samples

    @staticmethod
    def zero():
        """
        Used to signify that a particular error weight never results in a
        logical error.
        """
        return _WeightSamplingEntry(float64(0), float64(0), uint64(1))


    def add_data(
        self,
        mean: Union[float, float64],
        stddev: Union[float, float64],
        num_samples: Union[int, uint64],
    ) -> None:
        """
        Sums additional data to an existing _WeightSamplingEntry.
        """
        self.mean = (self.num_samples*self.mean + num_samples*mean) \
                    / (self.num_samples+num_samples)
        self.stddev = np.sqrt(((self.num_samples-1)*(self.stddev**2) \
                            + (num_samples-1)*(stddev**2)) \
                            / (self.num_samples+num_samples-1))
        self.num_samples = self.num_samples + num_samples

    # def estimate_num_samples_difference(
    #     self,
    #     error_target: Union[float, float64],
    # ) -> uint64:
    #     """
    #     Uses the sample standard deviation as an estimate for the population
    #     standard deviation in order to determine the number of additional
    #     samples to be taken to achieve a given error target.
    #     """
    #     total_num_samples = np.ceil((self.stddev / error_target)**2)
    #     diff_num_samples = total_num_samples - self.num_samples
    #     return diff_num_samples if diff_num_samples > 0 else uint64(0)

    def stderr(self) -> float64:
        """
        Returns the standard error of the mean of the data.
        """
        return self.stddev/np.sqrt(self.num_samples)

    def stats(self) -> Tuple[float64, float64]:
        """
        Returns a tuple containing the mean and standard error.
        """
        return (self.mean, self.stderr())


class ImportanceSampling:
    """
    A class that performs and contains the results of importance sampling.
    Makes use of a PyMatching Matching to execute the decoding of errors
    generated. When determining the decoding failure rate (logical error rate)
    for a particular noise level (physical error rate), only enough weight
    samples will be taken for the standard error of the failure rate to be below
    error_threshold. All unsampled weights are assumed to have a failure rate of
    1.0 and a standard error of 1.0, hence, in a case where not all weights are
    sampled, an upper bound for the failure rate is returned.
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
        """
        Constructor of the ImportanceSampling class.

        Parameters
        ----------
        parity_check_matrix : NDArray[uint8]
            The check matrix used to generate the syndrome for the errors.
        initial_num_samples : Union[int, uint64]
            The initial number of samples to be taken of each weight. This
            initial sampling is used to provide a rough estimate of the failure
            rate of each weight before additional samples are taken due to the
            error_theshold parameter.
            Note: If no failures are produced within this initial sample then it
            will be assumed that the particular error weight never results in a
            decoding failure and will not be sampled again.
            By default 100000
        error_threshold : Union[float, float64], optional
            The default threshold for the standard error of the failure rate.
            All failure rates generated are guaranteed to have a standard error
            of mean less than this value.
            By default 1e-2
        distance : Optional[Union[int, int64]], optional
            The distance of the error correction code.
            When not None, this distance is used to skip unnecessary sampling.
            By default None
        matching : Optional[Matching], optional
            A Matching object created by PyMatching
            If none, one will be generated from parity_check_matrix.
            By default None
        num_threads : Union[int, uintp], optional
            The number of threads to allocate.
            If 0, then the number of threads would be set to the number of
            logical cores for your cpu.
            By default 0
        rng_seed : Optional[Union[int, np.random.Generator]], optional
            The rng seed used in the generation of errors.
            If None, then the default numpy rng will be used.
            By default None
        """

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


    @property
    def num_qubits(self) -> int:
        """
        The number of physical qubits on which the error code acts.
        """
        return self._matching.num_nodes
    
    
    def sample_weight(
        self,
        error_weight: Union[int, uint64],
        num_samples: Union[int, uint64],
    ) -> None:
        """
        Takes a sampling of a particular weight and stores the result.

        Parameters
        ----------
        error_weight : Union[int, uint64]
            The weight to be sampled.
        num_samples : Union[int, uint64]
            The number of samples to be taken.
        """

        error_weight = uint64(error_weight)

        errors = functions.generate_errors(
            self.num_qubits,
            num_samples,
            error_weight = error_weight,
            rng_seed = int.from_bytes(self._rng.bytes(32)),
            num_threads = self._num_threads,
        )

        syndromes = functions.generate_syndromes(
            self._parity_check_matrix,
            errors,
            self._num_threads,
        )

        predictions = self._matching.decode_batch(syndromes)
        assert type(predictions) is np.ndarray ###################### TODO: allow for phenomological noise

        failures = functions.determine_logical_errors(errors,predictions,self._num_threads)

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
        Returns the error weights sampled and the failure rate with the standard
        error for the failure rate for each weight.

        Returns
        -------
        List[Tuple[uint64, Tuple[float64, float64]]]
            Each tuple is in the format:
            (weight, (failure_rate, standard_error))
        """
        weights: List[Tuple[uint64, Tuple[float64, float64]]] = []
        for w, e in self._data.items():
            weights.append((w, e.stats()))
        return weights



    def _failure_rate_by_weight(
            self,
            weight: uint64,
            noise_level: float64,
            error_threshold: float64,
            initial_num_samples: uint64,
     ) -> Tuple[float64, float64]:
        """
        Returns the failure rate and standard error of that failure rate for a
        particular error weight. If the weight's failure rate standard error is
        above the requirement for the error_threshold then additional samples
        will be taken.

        Parameters
        ----------
        weight : uint64
            The error weight being checked.
        noise_level : float64
            The noise_level being checked.
        error_threshold : float64
            The standard error threshold for the overall failure rate of the
            noise level.
        initial_num_samples : uint64
            The initial number of samples to be taken if the weight given has
            not been sampled and the assumed standard error (1) is above the
            requirement for the overall failure rate standard error of the noise
            level.

        Returns
        -------
        Tuple[float64, float64]
            A tuple in the format: (weight_failure_rate, weight_standard_error)
        """

        num_weights = self.num_qubits
        binom_pmf = binom.pmf(weight, num_weights, noise_level)

        # Ternary statement is to prevent potential divide by 0 error.
        w_error_threshold = np.inf if (binom_pmf**2)*num_weights <= 1e-300 \
            else np.sqrt((error_threshold**2)/((binom_pmf**2)*num_weights))


        if weight in self._data.keys():
            w_failure_rate, w_error = self._data[weight].stats()

            while w_error >= w_error_threshold:
                _, dev, current_num_samples = self._data[weight].values()

                num_samples_diff = uint64(np.ceil(
                    (binom_pmf**2) * (dev**2) * num_weights * (1 / (error_threshold**2))
                )) - current_num_samples
                assert num_samples_diff > 0
 
                self.sample_weight(weight, num_samples_diff)
                w_failure_rate, w_error = self._data[weight].stats()
            return w_failure_rate, w_error

        elif w_error_threshold < 0.5:
            self.sample_weight(weight, initial_num_samples)
            w_failure_rate, w_error = self._data[weight].stats()
            
            while w_error >= w_error_threshold:
                _, dev, current_num_samples = self._data[weight].values()

                num_samples_diff = uint64(np.ceil(
                    (binom_pmf**2) * (dev**2) * num_weights * (1 / (error_threshold**2))
                )) - current_num_samples
                
                assert num_samples_diff > 0 
                self.sample_weight(weight, num_samples_diff)
                w_failure_rate, w_error = self._data[weight].stats()
            return w_failure_rate, w_error

        else:
            return float64(1), float64(1)


    def failure_rate( # TODO: Allow for an array of p_phys to be input
        self,
        noise_level: Union[float, float64],
        error_threshold: Optional[Union[float, float64]] = None,
        initial_num_samples: Optional[Union[int, uint64]] = None,
    ) : # leaving it up to type inference to determine return
        """
        Determines and returns the decoding failure rate for a particular noise
        level. Additional samples of the weights will be taken and stored if the
        specified error_threshold cannot be reached.

        Parameters
        ----------
        noise_level : Union[float, float64]
            The noise level to be checked.
        error_theshold : Optional[Union[float, float64]], optional
            The threshold for the standard error rate of the failure rate.
            The standard error returned is guaranteed to be less than this
            value.
            If None, the error_threshold will be taken from the initialisation
            of the ImportanceSampling.
            By default None
        initial_num_samples : Optional[Union[int, uint64]], optional
            The initial number of samples to be taken if the weight given has
            not been sampled and the assumed standard error (1) is above the
            requirement for the overall failure rate standard error of the noise
            level.
            If None, the initial_num_samples will be taken from the
            initialisation of the ImportanceSampling.
            By default None

        Returns
        -------
            Probably a tuple containing two float64 in the format:
            (failure_rate, standard_error)
        """ # TODO: Determine what this function actually returns.

        initial_num_samples = self._default_num_samples \
            if initial_num_samples is None else uint64(initial_num_samples)
        error_threshold = self._default_plog_error_threshold \
            if error_threshold is None else float64(error_threshold)
        
        failure_rate, standard_error_sqr = 0,0
        n = self.num_qubits
        for weight in range(n+1):
            mean, err = self._failure_rate_by_weight(
                uint64(weight),
                float64(noise_level),
                error_threshold,
                initial_num_samples,
            )
            failure_rate += binom.pmf(weight,self.num_qubits,noise_level) * mean
            standard_error_sqr += (binom.pmf(weight,self.num_qubits,noise_level)*err)**2
        standard_error = np.sqrt(standard_error_sqr)
        return failure_rate, standard_error


