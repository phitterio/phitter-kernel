import numpy
import scipy.optimize
import scipy.stats


class FatigueLife:
    """
    Fatigue life Distribution
    Also known as Birnbaum-Saunders distribution
    Parameters FatigueLife Distribution: {"gamma": *, "loc": *, "scale": *}
    https://phitter.io/distributions/continuous/fatigue_life
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        Initializes the FatigueLife Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        Parameters FatigueLife Distribution: {"gamma": *, "loc": *, "scale": *}
        https://phitter.io/distributions/continuous/fatigue_life
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
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "fatigue_life"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"gamma": 5, "loc": 3, "scale": 9}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # z = lambda t: numpy.sqrt((t - self.loc) / self.scale)
        # result = scipy.stats.norm.cdf((z(x) - 1 / z(x)) / (self.gamma))
        result = scipy.stats.fatiguelife.cdf(x, self.gamma, loc=self.loc, scale=self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # z = lambda t: numpy.sqrt((t - self.loc) / self.scale)
        # result = (z(x) + 1 / z(x)) / (2 * self.gamma * (x - self.loc)) * scipy.stats.norm.pdf((z(x) - 1 / z(x)) / (self.gamma))
        result = scipy.stats.fatiguelife.pdf(x, self.gamma, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        # result = self.loc + (self.scale * (self.gamma * scipy.stats.norm.ppf(u) + numpy.sqrt((self.gamma * scipy.stats.norm.ppf(u)) ** 2 + 4)) ** 2) / 4
        result = scipy.stats.fatiguelife.ppf(u, self.gamma, loc=self.loc, scale=self.scale)
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
        return self.loc + self.scale * (1 + self.gamma**2 / 2)

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return self.scale**2 * self.gamma**2 * (1 + (5 * self.gamma**2) / 4)

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
        return (4 * self.gamma**2 * (11 * self.gamma**2 + 6)) / ((5 * self.gamma**2 + 4) * numpy.sqrt(self.gamma**2 * (5 * self.gamma**2 + 4)))

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 3 + (6 * self.gamma * self.gamma * (93 * self.gamma * self.gamma + 40)) / (5 * self.gamma**2 + 4) ** 2

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
        v1 = self.scale > 0
        v2 = self.gamma > 0
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
        parameters: {"gamma": *, "loc": *, "scale": *}
        """

        ## NO SE ESTÁN RESOLVIENDO LAS ECUACIONES PARA GAMMA = 5, scale = 10, loc = 5
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            gamma, loc, scale = initial_solution

            ## Parametric expected expressions
            parametric_mean = loc + scale * (1 + gamma**2 / 2)
            parametric_variance = scale**2 * gamma**2 * (1 + 5 * gamma**2 / 4)
            # parametric_skewness = 4 * gamma ** 2 * (11 * gamma ** 2 + 6) / ((4 + 5 * gamma ** 2) * numpy.sqrt(gamma ** 2 * (4 + 5 * gamma ** 2)))
            parametric_kurtosis = 3 + (6 * gamma * gamma * (93 * gamma * gamma + 40)) / (5 * gamma**2 + 4) ** 2
            parametric_median = loc + (scale * (gamma * scipy.stats.norm.ppf(0.5) + numpy.sqrt((gamma * scipy.stats.norm.ppf(0.5)) ** 2 + 4)) ** 2) / 4

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            # eq3 = parametric_skewness - continuous_measures.skewness
            eq3 = parametric_kurtosis - continuous_measures.kurtosis
            # eq3 = parametric_median - continuous_measures.median

            return (eq1, eq2, eq3)

        # x0 = (1, continuous_measures.min, 1)
        # bounds = [(0, -numpy.inf, 0), (numpy.inf, numpy.inf, numpy.inf)]
        # args = [continuous_measures]
        # solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        # parameters = {"gamma": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}

        scipy_parameters = scipy.stats.fatiguelife.fit(continuous_measures.data_to_fit)
        parameters = {"gamma": scipy_parameters[0], "loc": scipy_parameters[1], "scale": scipy_parameters[2]}
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
    path = "../continuous_distributions_sample/sample_fatigue_life.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = FatigueLife(continuous_measures=continuous_measures)

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
