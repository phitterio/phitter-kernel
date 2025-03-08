import numpy
import scipy.stats


class LogNormal:
    """
    Lognormal distribution
    Parameters LogNormal Distribution: {"mu": *, "sigma": *}
    https://phitter.io/distributions/continuous/lognormal
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        Initializes the LogNormal Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        Parameters LogNormal Distribution: {"mu": *, "sigma": *}
        https://phitter.io/distributions/continuous/lognormal
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

        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "lognormal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 2, "sigma": 7}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # result, error = scipy.integrate.quad(self.pdf, 1e-15, x)
        result = scipy.stats.norm.cdf((numpy.log(x) - self.mu) / self.sigma)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        return (1 / (x * self.sigma * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-(((numpy.log(x) - self.mu) ** 2) / (2 * self.sigma**2)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = numpy.exp(self.mu + self.sigma * scipy.stats.norm.ppf(u))
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
        return numpy.exp(self.mu + self.sigma**2 / 2)

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (numpy.exp(self.sigma**2) - 1) * numpy.exp(2 * self.mu + self.sigma**2)

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
        return (numpy.exp(self.sigma * self.sigma) + 2) * numpy.sqrt(numpy.exp(self.sigma * self.sigma) - 1)

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return numpy.exp(4 * self.sigma * self.sigma) + 2 * numpy.exp(3 * self.sigma * self.sigma) + 3 * numpy.exp(2 * self.sigma * self.sigma) - 3

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
        return numpy.exp(self.mu - self.sigma * self.sigma)

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
        v1 = self.mu > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by formula.

        Parameters
        ==========
        continuous_measures : dict
            {"mu": * , "variance": * , "skewness": * , "kurtosis": * , "data": * }

        Returns
        =======
        parameters: {"mu": *, "sigma": *}
        """

        mu = numpy.log(continuous_measures.mean**2 / numpy.sqrt(continuous_measures.mean**2 + continuous_measures.variance))
        sigma = numpy.sqrt(numpy.log((continuous_measures.mean**2 + continuous_measures.variance) / (continuous_measures.mean**2)))

        parameters = {"mu": mu, "sigma": sigma}
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
    path = "../continuous_distributions_sample/sample_lognormal.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = LogNormal(continuous_measures=continuous_measures)

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
