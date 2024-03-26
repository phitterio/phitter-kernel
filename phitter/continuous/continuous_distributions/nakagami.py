import numpy
import scipy.special


class NAKAGAMI:
    """
    Nakagami distribution
    Parameters NAKAGAMI distribution: {"m": *, "omega": *}
    https://phitter.io/distributions/continuous/nakagami
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        """
        Initializes the NAKAGAMI distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters NAKAGAMI distribution: {"m": *, "omega": *}
        """
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")

        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters

        self.m = self.parameters["m"]
        self.omega = self.parameters["omega"]

    @property
    def name(self):
        return "nakagami"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        result = scipy.special.gammainc(self.m, (self.m / self.omega) * x**2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        return (2 * self.m**self.m) / (scipy.special.gamma(self.m) * self.omega**self.m) * (x ** (2 * self.m - 1) * numpy.exp(-(self.m / self.omega) * x**2))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = numpy.sqrt(scipy.special.gammaincinv(self.m, u) * (self.omega / self.m))
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
        return (scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(self.omega / self.m)

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return self.omega * (1 - (1 / self.m) * (scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) ** 2)

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
            (scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m))
            * numpy.sqrt(1 / self.m)
            * (1 - 4 * self.m * (1 - ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2))
        ) / (2 * self.m * (1 - ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2) ** 1.5)

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 3 + (
            -6 * ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 4 * self.m
            + (8 * self.m - 2) * ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2
            - 2 * self.m
            + 1
        ) / (self.m * (1 - ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2) ** 2)

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
        return (numpy.sqrt(2) / 2) * numpy.sqrt((self.omega * (2 * self.m - 1)) / self.m)

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
        v1 = self.m >= 0.5
        v2 = self.omega > 0
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
        parameters: {"m": *, "omega": *}
        """
        d = numpy.array(continuous_measures.data)
        E_x2 = sum(d * d) / len(d)
        E_x4 = sum(d * d * d * d) / len(d)

        omega = E_x2
        m = E_x2**2 / (E_x4 - E_x2**2)
        parameters = {"m": m, "omega": omega}

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
    path = "../continuous_distributions_sample/sample_nakagami.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = NAKAGAMI(continuous_measures)

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
