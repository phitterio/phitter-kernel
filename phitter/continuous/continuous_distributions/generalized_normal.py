import numpy
import scipy.special
import scipy.stats


class GENERALIZED_NORMAL:
    """
    Generalized normal distribution
    Parameters GENERALIZED_NORMAL distribution: {"beta": *, "mu": *, "alpha": *}
    https://phitter.io/distributions/continuous/generalized_normal    This distribution is known whit the following names:
    * Error Distribution
    * Exponential Power Distribution
    * Generalized Error Distribution (GED)
    * Generalized Gaussian distribution (GGD)
    * Subbotin distribution
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the GENERALIZED_NORMAL distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters GENERALIZED_NORMAL distribution: {"beta": *, "mu": *, "alpha": *}
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.beta = self.parameters["beta"]
        self.mu = self.parameters["mu"]
        self.alpha = self.parameters["alpha"]

    @property
    def name(self):
        return "generalized_normal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"beta": 1, "mu": 0, "alpha": 3}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # print(scipy.stats.gennorm.cdf(x , self.beta, loc=self.mu, scale=self.alpha))
        return 0.5 + (numpy.sign(x - self.mu) / 2) * scipy.special.gammainc(1 / self.beta, abs((x - self.mu) / self.alpha) ** self.beta)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # print(scipy.stats.gennorm.pdf(x , self.beta, loc=self.mu, scale=self.alpha))
        return self.beta / (2 * self.alpha * scipy.special.gamma(1 / self.beta)) * numpy.exp(-((abs(x - self.mu) / self.alpha) ** self.beta))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.mu + numpy.sign(u - 0.5) * (self.alpha**self.beta * scipy.special.gammaincinv(1 / self.beta, 2 * numpy.abs(u - 0.5))) ** (1 / self.beta)
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
        return (self.mu**2 * scipy.special.gamma(3 / self.alpha)) / scipy.special.gamma(1 / self.alpha)

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
        return (scipy.special.gamma(5 / self.alpha) * scipy.special.gamma(1 / self.alpha)) / scipy.special.gamma(3 / self.alpha) ** 2

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
        Check parameters restriction
        """
        v1 = self.alpha > 0
        v2 = self.beta > 0
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
        parameters: {"beta": *, "mu": *, "alpha": *}
        """
        scipy_params = scipy.stats.gennorm.fit(continuous_measures.data)
        parameters = {"beta": scipy_params[0], "mu": scipy_params[1], "alpha": scipy_params[2]}
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
    path = "../continuous_distributions_sample/sample_generalized_normal.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = GENERALIZED_NORMAL(continuous_measures)

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
