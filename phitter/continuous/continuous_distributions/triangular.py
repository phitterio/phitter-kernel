import numpy
import scipy.stats


class TRIANGULAR:
    """
    Triangular distribution
    https://phitter.io/distributions/continuous/triangular
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        """
        Initializes the TRIANGULAR distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        The TRIANGULAR distribution parameters are: {"a": *, "b": *, "c": *}.
        """
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")

        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters

        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]

    @property
    def name(self):
        return "triangular"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # if x <= self.a:
        #     return 0
        # if self.a < x and x <= self.c:
        #     return (x - self.a) ** 2 / ((self.b - self.a) * (self.c - self.a))
        # if self.c < x and x < self.b:
        #     return 1 - ((self.b - x) ** 2 / ((self.b - self.a) * (self.b - self.c)))
        # if x > self.b:
        #     return 1
        result = scipy.stats.triang.cdf(x, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # if x <= self.a:
        #     return 0
        # if self.a <= x and x < self.c:
        #     return 2 * (x - self.a) / ((self.b - self.a) * (self.c - self.a))
        # if x == self.c:
        #     return 2 / (self.b - self.a)
        # if x > self.c and x <= self.b:
        #     return 2 * (self.b - x) / ((self.b - self.a) * (self.b - self.c))
        # if x > self.b:
        #     return 0
        result = scipy.stats.triang.pdf(x, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.triang.ppf(u, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
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
        return (self.a + self.b + self.c) / 3

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (self.a**2 + self.b**2 + self.c**2 - self.a * self.b - self.a * self.c - self.b * self.c) / 18

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
        return (numpy.sqrt(2) * (self.a + self.b - 2 * self.c) * (2 * self.a - self.b - self.c) * (self.a - 2 * self.b + self.c)) / (
            5 * (self.a**2 + self.b**2 + self.c**2 - self.a * self.b - self.a * self.c - self.b * self.c) ** (3 / 2)
        )

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 3 - 3 / 5

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
        return self.c

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
        v1 = self.a < self.c
        v2 = self.c < self.b
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by formula.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters: {"a": *, "b": *, "c": *}
        """
        ## Solve equations for estimation parameters
        # def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
        #     ## Variables declaration
        #     a, b, c = initial_solution
        #     print(continuous_measures)
        #     ## Parametric expected expressions
        #     parametric_mean = (a + b + c) / 3
        #     parametric_variance = (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) / 18
        #     parametric_skewness = numpy.sqrt(2) * (a + b - 2 * c) * (2 * a - b - c) * (a - 2 * b  + c) / (5 * (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) ** (3 / 2))

        #     ## System Equations
        #     eq1 = parametric_mean - continuous_measures.mean
        #     eq2 = parametric_variance - continuous_measures.variance
        #     eq3 = parametric_skewness - continuous_measures.skewness
        #     return (eq1, eq2, eq3)

        # solution = scipy.optimize.fsolve(equations, (1, 1, 1), continuous_measures)

        ## Second method estimation
        a = continuous_measures.min - 1e-3
        b = continuous_measures.max + 1e-3
        c = 3 * continuous_measures.mean - a - b

        ## Third method
        ## https://phitter.io/distributions/continuous/triangular        # q_1_16 = numpy.quantile(continuous_measures.data, 1 / 16)
        # q_1_4 = numpy.quantile(continuous_measures.data, 1 / 4)
        # q_3_4 = numpy.quantile(continuous_measures.data, 3 / 4)
        # q_15_16 = numpy.quantile(continuous_measures.data, 15 / 16)
        # u = (q_1_4 - q_1_16) ** 2
        # v = (q_15_16 - q_3_4) ** 2

        # a = 2 * q_1_16 - q_1_4
        # b = 2 * q_15_16 - q_3_4
        # c = (u * b + v * a) / (u + v)

        # Scipy parameters of distribution
        # scipy_params = scipy.stats.triang.fit(continuous_measures.data)
        # a = scipy_params[1]
        # b = scipy_params[1] + scipy_params[2]
        # c = scipy_params[1] + scipy_params[2] * scipy_params[0]

        parameters = {"a": a, "b": b, "c": c}
        return parameters


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from continuous_measures import CONTINUOUS_MEASURES

    ## Import function to get continuous_measures
    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    ## Distribution class
    path = "../continuous_distributions_sample/sample_triangular.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = TRIANGULAR(continuous_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.get_parameters(continuous_measures)}")
    print(f"CDF: {distribution.cdf(continuous_measures.mean)} {distribution.cdf(numpy.array([continuous_measures.mean, continuous_measures.mean]))}")
    print(f"PDF: {distribution.pdf(continuous_measures.mean)} {distribution.pdf(numpy.array([continuous_measures.mean, continuous_measures.mean]))}")
    print(f"PPF: {distribution.ppf(0.5)} {distribution.ppf(numpy.array([0.5, 0.5]))} - V: {distribution.cdf(distribution.ppf(0.5))}")
    print(f"SAMPLE: {distribution.sample(5)}")
    print(f"\nSTATS")
    print(f"mean: {distribution.mean} - {continuous_measures.mean}")
    print(f"variance: {distribution.variance} - {continuous_measures.variance}")
    print(f"skewness: {distribution.skewness} - {continuous_measures.skewness}")
    print(f"kurtosis: {distribution.kurtosis} - {continuous_measures.kurtosis}")
    print(f"median: {distribution.median} - {continuous_measures.median}")
    print(f"mode: {distribution.mode} - {continuous_measures.mode}")

    print(type(numpy.array([1, 2])))
