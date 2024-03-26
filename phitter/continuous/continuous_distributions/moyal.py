import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class MOYAL:
    """
    Moyal distribution
    Parameters MOYAL distribution: {"mu": *, "sigma": *}
    https://phitter.io/distributions/continuous/moyal
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        """
        Initializes the MOYAL distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters MOYAL distribution: {"mu": *, "sigma": *}
        """
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")

        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters

        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "moyal"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.mu) / self.sigma
        # result = result = scipy.stats.moyal.cdf(x, loc = self.mu, scale = self.sigma)
        # result = 1 - scipy.special.gammainc(0.5, numpy.exp(-z(x)) / 2)
        result = scipy.special.erfc(numpy.exp(-0.5 * z(x)) / numpy.sqrt(2))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.mu) / self.sigma
        # result = scipy.stats.moyal.pdf(x, loc = self.mu, scale = self.sigma)
        result = numpy.exp(-0.5 * (z(x) + numpy.exp(-z(x)))) / (self.sigma * numpy.sqrt(2 * numpy.pi))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.mu - self.sigma * numpy.log(scipy.stats.norm.ppf(1 - u / 2) ** 2)
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
        return self.mu + self.sigma * (numpy.log(2) + 0.577215664901532)

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (self.sigma * self.sigma * numpy.pi * numpy.pi) / 2

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
        return 1.5351415907229

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 7

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
        # def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
        #     ## Variables declaration
        #     mu, sigma = initial_solution

        #     ## Parametric expected expressions
        #     parametric_mean = mu + sigma * (numpy.log(2) + 0.577215664901532)
        #     parametric_variance = sigma * sigma * numpy.pi * numpy.pi / 2

        #     ## System Equations
        #     eq1 = parametric_mean - continuous_measures.mean
        #     eq2 = parametric_variance - continuous_measures.variance

        #     return (eq1, eq2)

        # bnds = ((-numpy.inf, 0), (numpy.inf, numpy.inf))
        # x0 = (continuous_measures.mean, 1)
        # args = ([continuous_measures])
        # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)

        sigma = numpy.sqrt(2 * continuous_measures.variance / (numpy.pi * numpy.pi))
        mu = continuous_measures.mean - sigma * (numpy.log(2) + 0.577215664901532)

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
    path = "../continuous_distributions_sample/sample_moyal.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = MOYAL(continuous_measures)

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
