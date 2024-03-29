import numpy


class UNIFORM:
    """
    Uniform distribution
    Parameters UNIFORM distribution: {"a": *, "b": *}
    https://phitter.io/distributions/discrete/uniform
    """

    def __init__(self, discrete_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "uniform"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 3, "b": 10}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        return (x - self.a + 1) / (self.b - self.a + 1)

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability mass function
        """
        if type(x) == int:
            return 1 / (self.b - self.a + 1)
        return numpy.full(len(x), 1 / (self.b - self.a + 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = numpy.ceil(u * (self.b - self.a + 1) + self.a - 1)
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
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = ∫(x - µ[1])ᵏ f(x) dx
        """
        return None

    @property
    def mean(self) -> float:
        """
        Parametric mean
        """
        return (self.a + self.b) / 2

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return ((self.b - self.a + 1) * (self.b - self.a + 1) - 1) / 12

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
        return 0

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return ((-6 / 5) * ((self.b - self.a + 1) * (self.b - self.a + 1) + 1)) / ((self.b - self.a + 1) * (self.b - self.a + 1) - 1) + 3

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
        return None

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
        v1 = self.b > self.a
        v2 = type(self.b) == int
        v3 = type(self.a) == int
        return v1 and v2 and v3

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample discrete_measures.
        The parameters are calculated by formula.

        Parameters
        ==========
        discrete_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters: {"a": *, "b": *}
        """
        a = round(discrete_measures.min)
        b = round(discrete_measures.max)
        parameters = {"a": a, "b": b}
        return parameters


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from discrete_measures import DISCRETE_MEASURES

    ## Import function to get discrete_measures
    def get_data(path: str) -> list[int]:
        sample_distribution_file = open(path, "r")
        data = [int(x) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    path = "../discrete_distributions_sample/sample_uniform.txt"

    ## Distribution class
    data = get_data(path)
    discrete_measures = DISCRETE_MEASURES(data)
    distribution = UNIFORM(discrete_measures)

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
