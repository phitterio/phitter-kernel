import numpy
import scipy.special
import scipy.stats


class Gibrat:
    """
    Gibrat distribution
    Parameters Gibrat Distribution: {"loc": *, "scale": *}
    https://phitter.io/distributions/continuous/gibrat
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        Initializes the Gibrat Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        Parameters Gibrat Distribution: {"loc": *, "scale": *}
        https://phitter.io/distributions/continuous/gibrat
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError(
                "You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True."
            )
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "gibrat"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"loc": 29, "scale": 102}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # z = lambda t: (t - self.loc) / self.scale
        # result = 0.5 * (1 + scipy.special.erf(numpy.log(z(x)) / numpy.sqrt(2)))
        result = scipy.stats.gibrat.cdf(x, self.loc, self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # z = lambda t: (t - self.loc) / self.scale
        # result = 1 / (self.scale * z(x) * numpy.sqrt(2 * numpy.pi)) * numpy.exp(-0.5 * numpy.log(z(x)) ** 2)
        result = scipy.stats.gibrat.pdf(x, self.loc, self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = numpy.exp(scipy.stats.norm.ppf(u)) * self.scale + self.loc
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
        return self.loc + self.scale * numpy.sqrt(numpy.exp(1))

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return numpy.exp(1) * (numpy.exp(1) - 1) * self.scale * self.scale

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
        return (2 + numpy.exp(1)) * numpy.sqrt(numpy.exp(1) - 1)

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return numpy.exp(1) ** 4 + 2 * numpy.exp(1) ** 3 + 3 * numpy.exp(1) ** 2 - 6

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
        return (1 / numpy.exp(1)) * self.scale + self.loc

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
        v1 = self.scale > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by solving the equations of the measures expected
        for this distribution.The number of equations to consider is equal to the number
        of parameters.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"loc": *, "scale": *}
        """
        ## Parameter Estimation
        scale = numpy.sqrt(continuous_measures.variance / (numpy.e**2 - numpy.e))
        loc = continuous_measures.mean - scale * numpy.sqrt(numpy.e)
        parameters = {"loc": loc, "scale": scale}

        ## Scipy Estimation
        # scipy_parameters = scipy.stats.gibrat.fit(continuous_measures.data_to_fit)
        # parameters = {"loc": scipy_parameters[0], "scale": scipy_parameters[1]}
        return parameters


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from continuous_measures import ContinuousMeasures

    ## Import function to get continuous_measures
    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    ## Distribution class
    path = "../continuous_distributions_sample/sample_gibrat.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = Gibrat(continuous_measures=continuous_measures)

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
