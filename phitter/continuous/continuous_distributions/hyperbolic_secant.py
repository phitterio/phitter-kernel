import numpy


class HYPERBOLIC_SECANT:
    """
    Hyperbolic Secant distribution
    Parameters HYPERBOLIC_SECANT distribution: {"mu": *, "sigma": *}
    https://phitter.io/distributions/continuous/hyperbolic_secant
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the HYPERBOLIC_SECANT distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters HYPERBOLIC_SECANT distribution: {"mu": *, "sigma": *}
        https://phitter.io/distributions/continuous/hyperbolic_secant
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "hyperbolic_secant"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 1002, "sigma": 198}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        z = lambda t: numpy.pi * (t - self.mu) / (2 * self.sigma)
        return (2 / numpy.pi) * numpy.arctan(numpy.exp((z(x))))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: numpy.pi * (t - self.mu) / (2 * self.sigma)
        return (1 / numpy.cosh(z(x))) / (2 * self.sigma)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = numpy.log(numpy.tan((u * numpy.pi) / 2)) * ((2 * self.sigma) / numpy.pi) + self.mu
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
        return self.mu

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return self.sigma**2

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
        return 5

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
        return self.mu

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
        continuous_measures : dict
            {"mu": * , "variance": * , "skewness": * , "kurtosis": * , "data": * }

        Returns
        =======
        parameters: {"mu": *, "sigma": *}
        """
        mu = continuous_measures.mean
        sigma = numpy.sqrt(continuous_measures.variance)

        parameters = {"mu": mu, "sigma": sigma}
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
    path = "../continuous_distributions_sample/sample_hyperbolic_secant.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = HYPERBOLIC_SECANT(continuous_measures)

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
