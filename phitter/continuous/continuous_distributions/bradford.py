import numpy
import scipy.optimize


class BRADFORD:
    """
    Bradford distribution
    https://phitter.io/distributions/continuous/bradford
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        """
        Initializes the BRADFORD distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        The BRADFORD distribution parameters are: {"c": *, "min": *, "max": *}.
        """
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")

        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters

        self.c = self.parameters["c"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "bradford"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        result = numpy.log(1 + self.c * (x - self.min) / (self.max - self.min)) / numpy.log(self.c + 1)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        result = self.c / ((self.c * (x - self.min) + self.max - self.min) * numpy.log(self.c + 1))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.min + ((numpy.exp(u * numpy.log(1 + self.c)) - 1) * (self.max - self.min)) / self.c
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
        return (self.c * (self.max - self.min) + numpy.log(1 + self.c) * (self.min * (self.c + 1) - self.max)) / (numpy.log(1 + self.c) * self.c)

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return ((self.max - self.min) ** 2 * ((self.c + 2) * numpy.log(1 + self.c) - 2 * self.c)) / (2 * self.c * numpy.log(1 + self.c) ** 2)

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
        return (numpy.sqrt(2) * (12 * self.c * self.c - 9 * numpy.log(1 + self.c) * self.c * (self.c + 2) + 2 * numpy.log(1 + self.c) * numpy.log(1 + self.c) * (self.c * (self.c + 3) + 3))) / (
            numpy.sqrt(self.c * (self.c * (numpy.log(1 + self.c) - 2) + 2 * numpy.log(1 + self.c))) * (3 * self.c * (numpy.log(1 + self.c) - 2) + 6 * numpy.log(1 + self.c))
        )

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return (
            self.c**3 * (numpy.log(1 + self.c) - 3) * (numpy.log(1 + self.c) * (3 * numpy.log(1 + self.c) - 16) + 24)
            + 12 * numpy.log(1 + self.c) * self.c * self.c * (numpy.log(1 + self.c) - 4) * (numpy.log(1 + self.c) - 3)
            + 6 * self.c * numpy.log(1 + self.c) ** 2 * (3 * numpy.log(1 + self.c) - 14)
            + 12 * numpy.log(1 + self.c) ** 3
        ) / (3 * self.c * (self.c * (numpy.log(1 + self.c) - 2) + 2 * numpy.log(1 + self.c)) ** 2) + 3

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
        return self.min

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
        v1 = self.max > self.min
        return v1

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
        parameters: {"c": *, "min": *, "max": *}
        """

        _min = continuous_measures.min - 1e-3
        _max = continuous_measures.max + 1e-3

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            c = initial_solution

            ## Parametric expected expressions
            parametric_mean = (c * (_max - _min) + numpy.log(c + 1) * (_min * (c + 1) - _max)) / (c * numpy.log(c + 1))

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean

            return eq1

        solution = scipy.optimize.fsolve(equations, (1), continuous_measures)

        parameters = {"c": solution[0], "min": _min, "max": _max}

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

    path = "../continuous_distributions_sample/sample_bradford.txt"

    ## Distribution class
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = BRADFORD(continuous_measures)

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
