import numpy
import scipy.special
import scipy.stats


class ARGUS:
    """
    Argus distribution
    Parameters ARGUS distribution: {"chi": *, "loc": *, "scale": *}
    https://phitter.io/distributions/continuous/argus
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the ARGUS distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters ARGUS distribution: {"chi": *, "loc": *, "scale": *}
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.chi = self.parameters["chi"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "argus"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"chi": 3, "loc": 102, "scale": 0}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.loc) / self.scale
        # Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t)-0.5
        # print(scipy.stats.argus.cdf(x, self.chi, loc=self.loc, scale=self.scale))
        # print(1 - Ψ(self.chi * numpy.sqrt(1 - z(x) * z(x))) / Ψ(self.chi))
        result = 1 - scipy.special.gammainc(1.5, self.chi * self.chi * (1 - z(x) ** 2) / 2) / scipy.special.gammainc(1.5, self.chi * self.chi / 2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.loc) / self.scale
        Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t) - 0.5
        # print(scipy.stats.argus.pdf(x, self.chi, loc=self.loc, scale=self.scale))
        result = (1 / self.scale) * ((self.chi**3) / (numpy.sqrt(2 * numpy.pi) * Ψ(self.chi))) * z(x) * numpy.sqrt(1 - z(x) * z(x)) * numpy.exp(-0.5 * self.chi**2 * (1 - z(x) * z(x)))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        y1 = (1 - u) * scipy.special.gammainc(1.5, (self.chi * self.chi) / 2)
        y2 = (2 * scipy.special.gammaincinv(1.5, y1)) / (self.chi * self.chi)
        result = self.loc + self.scale * numpy.sqrt(1 - y2)
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
        return self.loc + self.scale * numpy.sqrt(numpy.pi / 8) * (
            (self.chi * numpy.exp((-self.chi * self.chi) / 4) * scipy.special.iv(1, (self.chi * self.chi) / 4)) / (scipy.stats.norm.cdf(self.chi) - self.chi * scipy.stats.norm.pdf(self.chi) - 0.5)
        )

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (
            self.scale * self.scale * (1 - 3 / (self.chi * self.chi) + (self.chi * scipy.stats.norm.pdf(self.chi)) / (scipy.stats.norm.cdf(self.chi) - self.chi * scipy.stats.norm.pdf(self.chi) - 0.5))
            - (self.mean - self.loc) ** 2
        )

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
        return None

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return None

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
        return self.loc + self.scale * (1 / (numpy.sqrt(2) * self.chi)) * numpy.sqrt(self.chi * self.chi - 2 + numpy.sqrt(self.chi * self.chi * self.chi * self.chi + 4))

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
        v1 = self.chi > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by solving the equations of the measures expected
        for this distribution.The number of equations to consider is equal to the number
        of parameters.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters: {"chi": *, "loc": *, "scale": *}
        """
        scipy_params = scipy.stats.argus.fit(continuous_measures.data)
        parameters = {"chi": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
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
    path = "../continuous_distributions_sample/sample_argus.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = ARGUS(continuous_measures)

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
