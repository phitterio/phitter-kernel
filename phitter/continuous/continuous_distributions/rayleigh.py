import numpy
import scipy.stats


class Rayleigh:
    """
    Rayleigh distribution
    - Parameters Rayleigh Distribution: {"gamma": \*, "sigma": \*}
    - https://phitter.io/distributions/continuous/rayleigh
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Rayleigh Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters Rayleigh Distribution: {"gamma": \*, "sigma": \*}
        - https://phitter.io/distributions/continuous/rayleigh
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

        self.gamma = self.parameters["gamma"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "rayleigh"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"gamma": 10, "sigma": 2}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.gamma) / self.sigma
        return 1 - numpy.exp(-0.5 * (z(x) ** 2))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.gamma) / self.sigma
        return z(x) * numpy.exp(-0.5 * (z(x) ** 2)) / self.sigma

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = numpy.sqrt(-2 * numpy.log(1 - u)) * self.sigma + self.gamma
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
        return self.sigma * numpy.sqrt(numpy.pi / 2) + self.gamma

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return self.sigma * self.sigma * (2 - numpy.pi / 2)

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
        return 0.6311

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return (24 * numpy.pi - 6 * numpy.pi * numpy.pi - 16) / ((4 - numpy.pi) * (4 - numpy.pi)) + 3

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
        return self.gamma + self.sigma

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
        The parameters are calculated by solving the equations of the measures expected
        for this distribution.The number of equations to consider is equal to the number
        of parameters.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"gamma": \*, "sigma": \*}
        """
        ## Scipy Rayleigh estimation
        # scipy_parameters = scipy.stats.rayleigh.fit(continuous_measures.data_to_fit)
        # parameters = {"gamma": scipy_parameters[0], "sigma": scipy_parameters[1]}

        ## Location and sigma solve system
        sigma = numpy.sqrt(continuous_measures.variance * 2 / (4 - numpy.pi))
        gamma = continuous_measures.mean - sigma * numpy.sqrt(numpy.pi / 2)

        parameters = {"gamma": gamma, "sigma": sigma}
        return parameters


if __name__ == "__main__":
    # Import function to get continuous_measures
    import sys

    import numpy

    sys.path.append("../")
    from continuous_measures import ContinuousMeasures

    # Import function to get continuous_measures
    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    # Distribution class
    path = "../continuous_distributions_sample/sample_rayleigh.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = Rayleigh(continuous_measures=continuous_measures)

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
