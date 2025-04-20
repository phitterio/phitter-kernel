import numpy
import scipy.optimize
import scipy.stats


class Hypergeometric:
    """
    Hypergeometric_distribution
    - Parameters Hypergeometric Distribution: {"N": \*, "K": \*, "n": \*}
    - https://phitter.io/distributions/discrete/hypergeometric
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Hypergeometric Distribution by either providing a Discrete Measures instance [DiscreteMeasures] or a dictionary with the distribution's parameters.
        - Parameters Hypergeometric Distribution: {"N": \*, "K": \*, "n": \*}
        - https://phitter.io/distributions/continuous/hypergeometric
        """
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.N = self.parameters["N"]
        self.K = self.parameters["K"]
        self.n = self.parameters["n"]

    @property
    def name(self):
        return "hypergeometric"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"N": 120, "K": 66, "n": 27}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        result = scipy.stats.hypergeom.cdf(x, self.N, self.n, self.K)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability mass function
        """
        # result = scipy.special.comb(self.K, x) * scipy.special.comb(self.N - self.K, self.n - x) / scipy.special.comb(self.N, self.n)
        result = scipy.stats.hypergeom.pmf(x, self.N, self.n, self.K)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.hypergeom.ppf(u, self.N, self.n, self.K)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        """
        Sample of n elements of ditribution
        """
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        """
        Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx
        """
        return None

    def central_moments(self, k: int) -> float | None:
        """
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx
        """
        return None

    @property
    def mean(self) -> float:
        """
        Parametric mean
        """
        return (self.n * self.K) / self.N

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return ((self.n * self.K) / self.N) * ((self.N - self.K) / self.N) * ((self.N - self.n) / (self.N - 1))

    @property
    def standard_deviation(self) -> float:
        """
        Parametric standard deviation
        """
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        """
        Parametric skewness
        """
        return ((self.N - 2 * self.K) * numpy.sqrt(self.N - 1) * (self.N - 2 * self.n)) / (numpy.sqrt(self.n * self.K * (self.N - self.K) * (self.N - self.n)) * (self.N - 2))

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 3 + (1 / (self.n * self.K * (self.N - self.K) * (self.N - self.n) * (self.N - 2) * (self.N - 3))) * (
            (self.N - 1) * self.N * self.N * (self.N * (self.N + 1) - 6 * self.K * (self.N - self.K) - 6 * self.n * (self.N - self.n))
            + 6 * self.n * self.K * (self.N - self.K) * (self.N - self.n) * (5 * self.N - 6)
        )

    @property
    def median(self) -> float:
        """
        Parametric median
        """
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        """
        Parametric mode
        """
        return numpy.floor(((self.n + 1) * (self.K + 1)) / (self.N + 2))

    @property
    def num_parameters(self) -> int:
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

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample discrete_measures.
        The parameters are calculated by formula.

        Parameters
        ==========
        discrete_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"N": \*, "K": \*, "n": \*}
        """

        def equations(initial_solution: tuple[float], discrete_measures) -> tuple[float]:
            ## Variables declaration
            N, K, n = initial_solution

            ## Parametric expected expressions
            parametric_mean = n * K / N
            parametric_variance = (n * K / N) * ((N - K) / N) * ((N - n) / (N - 1))
            # parametric_skewness = (N - 2 * K) * numpy.sqrt(N - 1) * (N - 2 * n)  / (numpy.sqrt(n * K * (N - K) * (N - n)) * (N - 2))
            # parametric_kurtosis = 3 + (1 / (n * K * (N - K) * (N - n) * (N - 2) * (N - 3))) * ((N - 1) * N * N * (N * (N + 1) - 6 * K * (N - K) - 6 * n * (N - n)) + 6 * n * K * (N - K) * (N - n) * (5 * N - 6))
            parametric_mode = numpy.floor((n + 1) * (K + 1) / (N + 2))

            ## System Equations
            eq1 = parametric_mean - discrete_measures.mean
            eq2 = parametric_variance - discrete_measures.variance
            # eq3 = parametric_skewness - discrete_measures.skewness
            # eq3 = parametric_kurtosis  - discrete_measures.kurtosis
            eq3 = parametric_mode - discrete_measures.mode

            return (eq1, eq2, eq3)

        bounds = ((discrete_measures.max, discrete_measures.max, 1), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (discrete_measures.max * 5, discrete_measures.max * 3, discrete_measures.max)
        args = [discrete_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"N": round(solution.x[0]), "K": round(solution.x[1]), "n": round(solution.x[2])}

        return parameters


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from discrete_measures import DiscreteMeasures

    ## Import function to get discrete_measures
    def get_data(path: str) -> list[int]:
        sample_distribution_file = open(path, "r")
        data = [int(x) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    ## Distribution class
    path = "../discrete_distributions_sample/sample_hypergeometric.txt"
    data = get_data(path)
    discrete_measures = DiscreteMeasures(data)
    distribution = Hypergeometric(discrete_measures=discrete_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.parameters}")
    print(f"CDF: {distribution.cdf(int(discrete_measures.mean))} {distribution.cdf(numpy.array([int(discrete_measures.mean), int(discrete_measures.mean)]))}")
    print(f"PMF: {distribution.pmf(int(discrete_measures.mean))} {distribution.pmf(numpy.array([int(discrete_measures.mean), int(discrete_measures.mean)]))}")
    print(f"PPF: {distribution.ppf(0.5)} {distribution.ppf(numpy.array([0.5, 0.5]))} - V: {distribution.cdf(distribution.ppf(0.5))}")
    print(f"SAMPLE: {distribution.sample(5)}")
    print(f"\nSTATS")
    print(f"mean: {distribution.mean} - {discrete_measures.mean}")
    print(f"variance: {distribution.variance} - {discrete_measures.variance}")
    print(f"skewness: {distribution.skewness} - {discrete_measures.skewness}")
    print(f"kurtosis: {distribution.kurtosis} - {discrete_measures.kurtosis}")
    print(f"median: {distribution.median} - {discrete_measures.median}")
    print(f"mode: {distribution.mode} - {discrete_measures.mode}")
