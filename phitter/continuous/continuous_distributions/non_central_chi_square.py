import numpy
import scipy.integrate
import scipy.special
import scipy.stats


class NON_CENTRAL_CHI_SQUARE:
    """
    Non-Central Chi Square distribution
    Parameters NON_CENTRAL_CHI_SQUARE distribution: {"lambda": *, "n": *}
    https://phitter.io/distributions/continuous/non_central_chi_square
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        """
        Initializes the NON_CENTRAL_CHI_SQUARE distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters NON_CENTRAL_CHI_SQUARE distribution: {"lambda": *, "n": *}
        """
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")

        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters

        self.lambda_ = self.parameters["lambda"]
        self.n = self.parameters["n"]

    @property
    def name(self):
        return "non_central_chi_square"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """

        # def Q(M: float, a: float, b: float) -> float:
        #     """
        #     Marcum Q - function
        #     https://en.wikipedia.org/wiki/Marcum_Q-function
        #     """
        #     k = 1 - M
        #     x = (a / b) ** k * scipy.special.iv(k, a * b)
        #     acum = 0
        #     while x > 1e-20:
        #         acum += x
        #         k += 1
        #         x = (a / b) ** k * scipy.special.iv(k, a * b)
        #     res = acum * numpy.exp(-(a**2 + b**2) / 2)
        #     return res

        result = scipy.stats.ncx2.cdf(x, self.lambda_, self.n)
        # result = scipy.special.chndtr(x, self.lambda_, self.n)
        # result = 1 - Q(self.n / 2, numpy.sqrt(self.lambda_), numpy.sqrt(x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        ## Method 1
        result = scipy.stats.ncx2.pdf(x, self.lambda_, self.n)

        ## Method 2
        # result = 1 / 2 * numpy.exp(-(x + self.lambda_) / 2) * (x / self.lambda_) ** ((self.n - 2) / 4) * scipy.special.iv((self.n - 2) / 2, numpy.sqrt(self.lambda_ * x))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.ncx2.ppf(u, self.lambda_, self.n)
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
        return self.lambda_ + self.n

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return 2 * (self.n + 2 * self.lambda_)

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
        return (2**1.5 * (self.n + 3 * self.lambda_)) / (self.n + 2 * self.lambda_) ** 1.5

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 3 + (12 * (self.n + 4 * self.lambda_)) / (self.n + 2 * self.lambda_) ** 2

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
        return None

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
        v1 = self.lambda_ > 0
        v2 = self.n > 0
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
        parameters: {"lambda": *, "n": *}
        """
        # def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
        #     lambda_, n = initial_solution

        #     ## Parametric expected expressions
        #     parametric_mean = lambda_ + n
        #     parametric_variance = 2 * (2 * lambda_ + n)
        #     # parametric_skewness = 8 * (3 * lambda_ + n) / ((2 * (2 * lambda_ + n)) ** 1.5)
        #     # parametric_kurtosis = 12 * (4 * lambda_ + n) / ((2 * lambda_ + n) ** 2)

        #     ## System Equations
        #     eq1 = parametric_mean - continuous_measures.mean
        #     eq2 = parametric_variance - continuous_measures.variance
        #     # eq3 = parametric_skewness - continuous_measures.skewness
        #     # eq4 = parametric_kurtosis  - continuous_measures.kurtosis

        #     return (eq1, eq2)

        # bnds = ((0, 0), (numpy.inf, numpy.inf))
        # x0 = (continuous_measures.mean, 1)
        # args = ([continuous_measures])
        # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        # parameters = {"lambda": solution.x[0], "n": round(solution.x[1])}

        lambda_ = continuous_measures.variance / 2 - continuous_measures.mean
        n = 2 * continuous_measures.mean - continuous_measures.variance / 2
        parameters = {"lambda": lambda_, "n": n}
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
    path = "../continuous_distributions_sample/sample_NON_CENTRAL_CHI_SQUARE.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = NON_CENTRAL_CHI_SQUARE(continuous_measures)

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

    print(distribution.pdf(60))
    # print(scipy.stats.ncx2.fit(data))
