import numpy
import scipy.optimize
import scipy.stats


class LOGARITHMIC:
    """
    Logarithmic distribution
    https://phitter.io/distributions/discrete/logarithmic
    """

    def __init__(self, discrete_measures, parameters: dict[str, int | float] = None):
        if discrete_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")

        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures)
        else:
            self.parameters = parameters

        self.p = self.parameters["p"]

    @property
    def name(self):
        return "logarithmic"

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        result = scipy.stats.logser.cdf(x, self.p)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability mass function
        """
        # result = -(self.p**x) / (numpy.log(1 - self.p) * x)
        result = scipy.stats.logser.pmf(x, self.p)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.logser.ppf(u, self.p)
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
        return -self.p / ((1 - self.p) * numpy.log(1 - self.p))

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (-self.p * (self.p + numpy.log(1 - self.p))) / ((1 - self.p) ** 2 * numpy.log(1 - self.p) ** 2)

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
        return (
            -(2 * self.p**2 + 3 * self.p * numpy.log(1 - self.p) + (1 + self.p) * numpy.log(1 - self.p) ** 2)
            / (numpy.log(1 - self.p) * (self.p + numpy.log(1 - self.p)) * numpy.sqrt(-self.p * (self.p + numpy.log(1 - self.p))))
        ) * numpy.log(1 - self.p)

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return -(6 * self.p**3 + 12 * self.p**2 * numpy.log(1 - self.p) + self.p * (4 * self.p + 7) * numpy.log(1 - self.p) ** 2 + (self.p**2 + 4 * self.p + 1) * numpy.log(1 - self.p) ** 3) / (
            self.p * (self.p + numpy.log(1 - self.p)) ** 2
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
        return 1

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
        return v1

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
        parameters: {"p": *}
        """

        def equations(initial_solution: tuple[float], discrete_measures) -> tuple[float]:
            ## Variables declaration
            p = initial_solution

            ## Parametric expected expressions
            parametric_mean = -p / ((1 - p) * numpy.log(1 - p))

            ## System Equations
            eq1 = parametric_mean - discrete_measures.mean
            return eq1

        solution = scipy.optimize.least_squares(equations, 0.5, bounds=(0, 1), args=([discrete_measures]))
        parameters = {"p": solution.x[0]}
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

    ## Distribution class
    path = "../discrete_distributions_sample/sample_logarithmic.txt"
    data = get_data(path)
    discrete_measures = DISCRETE_MEASURES(data)
    distribution = LOGARITHMIC(discrete_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.get_parameters(discrete_measures)}")
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
