import numpy
import scipy.special
import scipy.stats


class GENERALIZED_EXTREME_VALUE:
    """
    Generalized Extreme Value Distribution
    https://phitter.io/distributions/continuous/generalized_extreme_value
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the GENERALIZED_EXTREME_VALUE distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters GENERALIZED_EXTREME_VALUE distribution: {"xi": *, "mu": *, "sigma": *}
        https://phitter.io/distributions/continuous/generalized_extreme_value
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.xi = self.parameters["xi"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "generalized_extreme_value"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"xi": 0, "mu": 10, "sigma": 1}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.mu) / self.sigma
        if self.xi == 0:
            return numpy.exp(-numpy.exp(-z(x)))
        else:
            return numpy.exp(-((1 + self.xi * z(x)) ** (-1 / self.xi)))
        # return scipy.stats.genextreme.cdf(x,  - self.xi, loc=self.mu, scale=self.sigma)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # print(scipy.stats.genextreme.pdf(x,  - self.xi, loc=self.mu, scale=self.sigma))
        z = lambda t: (t - self.mu) / self.sigma
        if self.xi == 0:
            return (1 / self.sigma) * numpy.exp(-z(x) - numpy.exp(-z(x)))
        else:
            return (1 / self.sigma) * numpy.exp(-((1 + self.xi * z(x)) ** (-1 / self.xi))) * (1 + self.xi * z(x)) ** (-1 - 1 / self.xi)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        if self.xi == 0:
            result = self.mu - self.sigma * numpy.log(-numpy.log(u))
        else:
            result = self.mu + (self.sigma * ((-numpy.log(u)) ** -self.xi - 1)) / self.xi
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
        return scipy.special.gamma(1 - self.xi * k)

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
        if self.xi == 0:
            return self.mu + self.sigma * 0.5772156649
        µ1 = self.non_central_moments(1)
        return self.mu + (self.sigma * (µ1 - 1)) / self.xi

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        if self.xi == 0:
            return self.sigma**2 * (numpy.pi**2 / 6)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (self.sigma**2 * (µ2 - µ1**2)) / self.xi**2

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
        if self.xi == 0:
            return (12 * numpy.sqrt(6) * 1.20205690315959) / numpy.pi**3
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        if self.xi == 0:
            return 5.4
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

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
        if self.xi == 0:
            return self.mu
        return self.mu + (self.sigma * ((1 + self.xi) ** -self.xi - 1)) / self.xi

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
        v1 = self.sigma > 0
        return v1

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
        parameters: {"xi": *, "mu": *, "sigma": *}
        """
        scipy_params = scipy.stats.genextreme.fit(continuous_measures.data_to_fit)
        parameters = {"xi": -scipy_params[0], "mu": scipy_params[1], "sigma": scipy_params[2]}
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
    path = "../continuous_distributions_sample/sample_generalized_extreme_value.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = GENERALIZED_EXTREME_VALUE(continuous_measures)

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
