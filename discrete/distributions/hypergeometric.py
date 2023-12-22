import math

import numpy
import scipy.optimize
import scipy.stats


class HYPERGEOMETRIC:
    """
    Hypergeometric_distribution
    https://en.wikipedia.org/wiki/Hypergeometric_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.N = self.parameters["N"]
        self.K = self.parameters["K"]
        self.n = self.parameters["n"]

    def cdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using the definition of the function
        Alternative: scipy cdf method
        """
        result = scipy.stats.hypergeom.cdf(x, self.N, self.n, self.K)
        return result

    def pmf(self, x: int) -> float:
        """
        Probability density function
        Calculated using the definition of the function
        """
        # result = sc.comb(self.K, x) * sc.comb(self.N - self.K, self.n - x) / sc.comb(self.N, self.n)
        result = scipy.stats.hypergeom.pmf(x, self.N, self.n, self.K)
        return result

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.N > 0 and type(self.N) == int
        v2 = self.K > 0 and type(self.K) == int
        v3 = self.n > 0 and type(self.n) == int
        return v1 and v2 and v3

    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by formula.

        Parameters
        ==========
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"N": * , "K": * , "n": * }
        """

        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            ## Variables declaration
            N, K, n = initial_solution

            ## Parametric expected expressions
            parametric_mean = n * K / N
            parametric_variance = (n * K / N) * ((N - K) / N) * ((N - n) / (N - 1))
            # parametric_skewness = (N - 2 * K) * math.sqrt(N - 1) * (N - 2 * n)  / (math.sqrt(n * K * (N - K) * (N - n)) * (N - 2))
            # parametric_kurtosis = 3 + (1 / (n * K * (N - K) * (N - n) * (N - 2) * (N - 3))) * ((N - 1) * N * N * (N * (N + 1) - 6 * K * (N - K) - 6 * n * (N - n)) + 6 * n * K * (N - K) * (N - n) * (5 * N - 6))
            parametric_mode = math.floor((n + 1) * (K + 1) / (N + 2))

            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq3 = parametric_skewness - measurements.skewness
            # eq3 = parametric_kurtosis  - measurements.kurtosis
            eq3 = parametric_mode - measurements.mode

            return (eq1, eq2, eq3)

        bnds = ((measurements.max, measurements.max, 1), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (measurements.max * 5, measurements.max * 3, measurements.max)
        args = [measurements]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"N": round(solution.x[0]), "K": round(solution.x[1]), "n": round(solution.x[2])}

        return parameters


if __name__ == "__main__":
    ## Import function to get measurements
    import sys

    sys.path.append("../measurements")
    from measurements_discrete import MEASUREMENTS_DISCRETE

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [int(x) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_hypergeometric.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_DISCRETE(data)
    distribution = HYPERGEOMETRIC(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(round(measurements.mean)))
    print(distribution.pmf(round(measurements.mean)))
