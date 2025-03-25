import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class LogGamma:
    """
    LogGamma distribution
    - Parameters LogGamma Distribution: {"c": *, "mu": *, "sigma": *}
    - https://phitter.io/distributions/continuous/loggamma
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the LogGamma Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters LogGamma Distribution: {"c": *, "mu": *, "sigma": *}
        - https://phitter.io/distributions/continuous/loggamma
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

        self.c = self.parameters["c"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "loggamma"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"c": 2, "mu": 8, "sigma": 4}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # result = scipy.stats.gamma.cdf(numpy.exp((x - self.mu) / self.sigma), a=self.c, scale=1)
        # print(scipy.stats.loggamma.cdf(x, self.c, loc=self.mu, scale=self.sigma))

        y = lambda x: (x - self.mu) / self.sigma
        result = scipy.special.gammainc(self.c, numpy.exp(y(x)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # print(scipy.stats.loggamma.pdf(x, self.c, loc=self.mu, scale=self.sigma))
        y = lambda x: (x - self.mu) / self.sigma
        result = numpy.exp(self.c * y(x) - numpy.exp(y(x)) - scipy.special.gammaln(self.c)) / self.sigma

        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.mu + self.sigma * numpy.log(scipy.special.gammaincinv(self.c, u))
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
        return scipy.special.digamma(self.c) * self.sigma + self.mu

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return scipy.special.polygamma(1, self.c) * self.sigma * self.sigma

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
        return scipy.special.polygamma(2, self.c) / scipy.special.polygamma(1, self.c) ** 1.5

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return scipy.special.polygamma(3, self.c) / scipy.special.polygamma(1, self.c) ** 2 + 3

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
        return self.mu + self.sigma * numpy.log(self.c)

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
        v1 = self.c > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by formula.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"c": *, "mu": *, "sigma": *}
        """

        def equations(initial_solution, data_mean, data_variance, data_skewness):
            c, mu, sigma = initial_solution

            # parametric_mean, parametric_variance, parametric_skewness, parametric_kurtosis = scipy.stats.loggamma.stats(c, loc=mu, scale=sigma, moments='mvsk')
            parametric_mean = scipy.special.digamma(c) * sigma + mu
            parametric_variance = scipy.special.polygamma(1, c) * (sigma**2)
            parametric_skewness = scipy.special.polygamma(2, c) / (scipy.special.polygamma(1, c) ** 1.5)
            # parametric_kurtosis = scipy.special.polygamma(3, c) / (scipy.special.polygamma(1, c) ** 2)

            ## System Equations
            eq1 = parametric_mean - data_mean
            eq2 = parametric_variance - data_variance
            eq3 = parametric_skewness - data_skewness
            # eq4 = parametric_kurtosis  - continuous_measures.kurtosis

            return (eq1, eq2, eq3)

        bounds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, 1)
        args = (continuous_measures.mean, continuous_measures.variance, continuous_measures.skewness)
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"c": solution.x[0], "mu": solution.x[1], "sigma": solution.x[2]}
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
    path = "../continuous_distributions_sample/sample_loggamma.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = LogGamma(continuous_measures=continuous_measures)

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

    print("========= Time parameter estimation analisys ========")

    import time

    def equations(initial_solution, data_mean, data_variance, data_skewness):
        c, mu, sigma = initial_solution
        parametric_mean, parametric_variance, parametric_skewness, parametric_kurtosis = scipy.stats.loggamma.stats(c, loc=mu, scale=sigma, moments="mvsk")
        parametric_mean = scipy.special.digamma(c) * sigma + mu
        parametric_variance = scipy.special.polygamma(1, c) * (sigma**2)
        parametric_skewness = scipy.special.polygamma(2, c) / (scipy.special.polygamma(1, c) ** 1.5)
        # parametric_kurtosis = scipy.special.polygamma(3, c) / (scipy.special.polygamma(1, c) ** 2)

        ## System Equations
        eq1 = parametric_mean - data_mean
        eq2 = parametric_variance - data_variance
        eq3 = parametric_skewness - data_skewness
        # eq4 = parametric_kurtosis  - continuous_measures.kurtosis

        return (eq1, eq2, eq3)

    ti = time.time()
    bounds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
    x0 = (1, 1, 1)
    args = (continuous_measures.mean, continuous_measures.variance, continuous_measures.skewness)
    solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
    parameters = {"c": solution.x[0], "mu": solution.x[1], "sigma": solution.x[2]}
    print(parameters)
    print("Solve equations time: ", time.time() - ti)

    ti = time.time()
    scipy_parameters = scipy.stats.loggamma.fit(data)
    parameters = {"c": scipy_parameters[0], "mu": scipy_parameters[1], "sigma": scipy_parameters[2]}
    print(parameters)
    print("Scipy time get parameters: ", time.time() - ti)
