import numpy
import scipy.stats


class NegativeBinomial:
    """
    Negative binomial distribution
    Parameters NegativeBinomial Distribution: {"r": *, "p": *}
    https://phitter.io/distributions/discrete/negative_binomial
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        """
        Initializes the NegativeBinomial Distribution by either providing a Discrete Measures instance [DiscreteMeasures] or a dictionary with the distribution's parameters.
        The NegativeBinomial distribution parameters are: {"r": *, "p": *}.
        https://phitter.io/distributions/continuous/negative_binomial
        """
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.r = self.parameters["r"]
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "negative_binomial"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"r": 96, "p": 0.6893}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # result = scipy.stats.nbinom.cdf(x, self.r, self.p)
        result = scipy.stats.beta.cdf(self.p, self.r, x + 1)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability mass function
        """
        # result = scipy.special.comb(self.r + x - 1, x) * (self.p**self.r) * ((1 - self.p) ** x)
        result = scipy.stats.nbinom.pmf(x, self.r, self.p)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.nbinom.ppf(u, self.r, self.p)
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
        return (self.r * (1 - self.p)) / self.p

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (self.r * (1 - self.p)) / (self.p * self.p)

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
        return (2 - self.p) / numpy.sqrt(self.r * (1 - self.p))

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 6 / self.r + (self.p * self.p) / (self.r * (1 - self.p)) + 3

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
        return numpy.floor(((self.r - 1) * (1 - self.p)) / self.p)

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
        v1 = self.p > 0 and self.p < 1
        v2 = self.r > 0
        v3 = type(self.r) == int
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
        parameters: {"r": *, "p": *}
        """
        p = discrete_measures.mean / discrete_measures.variance
        r = round(discrete_measures.mean * p / (1 - p))
        parameters = {"r": r, "p": p}
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
    path = "../discrete_distributions_sample/sample_negative_binomial.txt"
    data = get_data(path)
    discrete_measures = DiscreteMeasures(data)
    distribution = NegativeBinomial(discrete_measures=discrete_measures)

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
