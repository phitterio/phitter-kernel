import numpy
import scipy.optimize
import scipy.stats


class GENERALIZED_PARETO:
    """
    Generalized Pareto distribution
    https://phitter.io/distributions/continuous/generalized_pareto
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        """
        Initializes the GENERALIZED_PARETO distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        The GENERALIZED_PARETO distribution parameters are: {"c": *, "mu": *, "sigma": *}.
        """
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")

        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters

        self.c = self.parameters["c"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "generalized_pareto"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """

        # result = scipy.stats.genpareto.cdf(x, self.c, loc = self.mu, scale = self.sigma)
        z = lambda t: (t - self.mu) / self.sigma
        result = 1 - (1 + self.c * z(x)) ** (-1 / self.c)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # result = scipy.stats.genpareto.pdf(x, self.c, loc = self.mu, scale = self.sigma)
        z = lambda t: (t - self.mu) / self.sigma
        result = (1 / self.sigma) * (1 + self.c * z(x)) ** (-1 / self.c - 1)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.mu + (self.sigma * ((1 - u) ** -self.c - 1)) / self.c
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
        return self.mu + self.sigma / (1 - self.c)

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (self.sigma * self.sigma) / ((1 - self.c) * (1 - self.c) * (1 - 2 * self.c))

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
        return (2 * (1 + self.c) * numpy.sqrt(1 - 2 * self.c)) / (1 - 3 * self.c)

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return (3 * (1 - 2 * self.c) * (2 * self.c * self.c + self.c + 3)) / ((1 - 3 * self.c) * (1 - 4 * self.c))

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
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters: {"c": *, "mu": *, "sigma": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            c, mu, sigma = initial_solution

            ## Parametric expected expressions
            parametric_mean = mu + sigma / (1 - c)
            parametric_variance = sigma * sigma / ((1 - c) * (1 - c) * (1 - 2 * c))
            # parametric_skewness = 2 * (1 + c) * numpy.sqrt(1 - 2 * c) / (1 - 3 * c)
            # parametric_kurtosis = 3 * (1 - 2 * c) * (2 * c * c + c + 3) / ((1 -  3 * c) * (1 -  4 + c))
            parametric_median = mu + sigma * (2**c - 1) / c
            # parametric_mode = loc

            # System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            # eq1 = parametric_skewness - continuous_measures.skewness
            # eq3 = parametric_kurtosis - continuous_measures.kurtosis
            # eq1 = parametric_p25 - numpy.percentile(continuous_measures.data, 25)
            eq3 = parametric_median - numpy.percentile(continuous_measures.data, 50)

            return (eq1, eq2, eq3)

        ## The scipy genpareto.fit is good from samples whit c > 0
        ## but it's not so much when c < 0.
        ## The solution of system of equation is so good. The problem is that
        ## the continuous_measures of genpareto distributions is not defined from c >  1 / 4 (kurtosis)
        scipy_params = scipy.stats.genpareto.fit(continuous_measures.data)
        parameters = {"c": scipy_params[0], "mu": scipy_params[1], "sigma": scipy_params[2]}

        if parameters["c"] < 0:
            scipy_params = scipy.stats.genpareto.fit(continuous_measures.data)
            c0 = scipy_params[0]
            x0 = [c0, continuous_measures.min, 1]
            b = ((-numpy.inf, -numpy.inf, 0), (numpy.inf, continuous_measures.min, numpy.inf))
            solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([continuous_measures]))
            parameters = {"c": solution.x[0], "mu": solution.x[1], "sigma": solution.x[2]}

            ## When c < 0 the domain of of x is [mu, mu - sigma / c]
            ## Forthis reason mu < continuous_measures.min and mu - sigma / c < continuous_measures.max
            parameters["mu"] = min(parameters["mu"], continuous_measures.min - 1e-3)
            delta_sigma = parameters["c"] * (parameters["mu"] - continuous_measures.max) - parameters["sigma"]
            parameters["sigma"] = parameters["sigma"] + delta_sigma + 1e-8
            # print(parameters["mu"], parameters["mu"] - parameters["sigma"] / parameters["c"])

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
    path = "../continuous_distributions_sample/sample_generalized_pareto.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = GENERALIZED_PARETO(continuous_measures)

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
