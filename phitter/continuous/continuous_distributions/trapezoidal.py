import numpy
import scipy.optimize
import scipy.stats


class TRAPEZOIDAL:
    """
    Trapezoidal distribution
    Parameters TRAPEZOIDAL distribution: {"a": *, "b": *, "c": *, "d": *}
    https://phitter.io/distributions/continuous/trapezoidal
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the TRAPEZOIDAL distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters TRAPEZOIDAL distribution: {"a": *, "b": *, "c": *, "d": *}
        https://phitter.io/distributions/continuous/trapezoidal
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        self.d = self.parameters["d"]

    @property
    def name(self):
        return "trapezoidal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 110, "b": 267, "c": 741, "d": 980}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # if x <= self.a:
        #     return 0
        # if self.a <= x and x < self.b:
        #     return (1 / (self.d + self.c - self.b - self.a)) * (1 / (self.b - self.a)) * (x - self.a) ** 2
        # if self.b <= x and x < self.c:
        #     return (1 / (self.d + self.c - self.b - self.a)) * (2 * x - self.a - self.b)
        # if self.c <= x and x <= self.d:
        #     return 1 - (1 / (self.d + self.c - self.b - self.a)) * (1 / (self.d - self.c)) * (self.d - x) ** 2
        # if x >= self.d:
        #     return 1
        result = scipy.stats.trapezoid.cdf(x, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # if x <= self.a:
        #     return 0
        # if self.a <= x and x < self.b:
        #     return (2 / (self.d + self.c - self.b - self.a)) * ((x - self.a) / (self.b - self.a))
        # if self.b <= x and x < self.c:
        #     return 2 / (self.d + self.c - self.b - self.a)
        # if self.c <= x and x <= self.d:
        #     return (2 / (self.d + self.c - self.b - self.a)) * ((self.d - x) / (self.d - self.c))
        # if x >= self.d:
        #     return 0
        result = scipy.stats.trapezoid.pdf(x, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.trapezoid.ppf(u, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
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
        return (
            (2 / (self.d + self.c - self.b - self.a))
            * (1 / ((k + 1) * (k + 2)))
            * ((self.d ** (k + 2) - self.c ** (k + 2)) / (self.d - self.c) - (self.b ** (k + 2) - self.a ** (k + 2)) / (self.b - self.a))
        )

    def central_moments(self, k: int) -> float | None:
        """
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = ∫(x - µ[1])ᵏ f(x) dx
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)

        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4

        return None

    @property
    def mean(self) -> float:
        """
        Parametric mean
        """
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

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
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

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
        v1 = self.a < self.b
        v2 = self.b < self.c
        v3 = self.c < self.d
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by formula.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"a": *, "b": *, "c": *, "d": *}
        """

        def equations(initial_solution, continuous_measures, a, d):
            ## Variables declaration
            b, c = initial_solution

            ## Parametric expected expressions
            parametric_mean = (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            parametric_variance = (1 / (6 * (d + c - a - b))) * ((d**4 - c**4) / (d - c) - (b**4 - a**4) / (b - a)) - (
                (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            ) ** 2

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance

            return (eq1, eq2)

        a = continuous_measures.min - 1e-3
        d = continuous_measures.max + 1e-3

        x0 = [(d + a) * 0.25, (d + a) * 0.75]
        bounds = ((a, a), (d, d))
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=([continuous_measures, a, d]))

        parameters = {"a": a, "b": solution.x[0], "c": solution.x[1], "d": d}
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
    path = "../continuous_distributions_sample/sample_trapezoidal.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = TRAPEZOIDAL(continuous_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.parameters}")
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
