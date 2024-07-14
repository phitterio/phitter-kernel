import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class GENERALIZED_LOGISTIC:
    """
    Generalized Logistic Distribution

    https://phitter.io/distributions/continuous/generalized_logistic
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        Initializes the GENERALIZED_LOGISTIC distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters GENERALIZED_LOGISTIC distribution: {"loc": *, "scale": *, "c": *}
        https://phitter.io/distributions/continuous/generalized_logistic
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError(
                "You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [CONTINUOUS_MEASURES] instance, or by setting init_parameters_examples to True."
            )
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        self.c = self.parameters["c"]

    @property
    def name(self):
        return "generalized_logistic"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"c": 2, "loc": 25, "scale": 32}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # return scipy.stats.genlogistic.cdf(x, self.c, loc=self.loc, scale=self.scale)
        z = lambda t: (t - self.loc) / self.scale
        return 1 / ((1 + numpy.exp(-z(x))) ** self.c)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # return scipy.stats.genlogistic.pdf(x, self.c, loc=self.loc, scale=self.scale)
        z = lambda t: (t - self.loc) / self.scale
        return (self.c / self.scale) * numpy.exp(-z(x)) * ((1 + numpy.exp(-z(x))) ** (-self.c - 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.loc + self.scale * -numpy.log(u ** (-1 / self.c) - 1)
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
        return self.loc + self.scale * (0.57721 + scipy.special.digamma(self.c))

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return self.scale * self.scale * ((numpy.pi * numpy.pi) / 6 + scipy.special.polygamma(1, self.c))

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
        return (2.40411380631918 + scipy.special.polygamma(2, self.c)) / ((numpy.pi * numpy.pi) / 6 + scipy.special.polygamma(1, self.c)) ** 1.5

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 3 + (6.49393940226682 + scipy.special.polygamma(3, self.c)) / ((numpy.pi * numpy.pi) / 6 + scipy.special.polygamma(1, self.c)) ** 2

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
        return self.loc + self.scale * numpy.log(self.c)

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
        v2 = self.c > 0
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
        parameters: {"loc": *, "scale": *, "c": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            c, loc, scale = initial_solution

            ## Parametric expected expressions
            parametric_mean = loc + scale * (0.57721 + scipy.special.digamma(c))
            parametric_variance = scale**2 * (numpy.pi**2 / 6 + scipy.special.polygamma(1, c))
            # parametric_skewness = (scipy.special.polygamma(2,1) + scipy.special.polygamma(2,c)) / ((numpy.pi ** 2 / 6 + scipy.special.polygamma(1, c)) ** 1.5)
            # parametric_kurtosis = 3 + (numpy.pi ** 4 / 15 + scipy.special.polygamma(3,c)) / ((numpy.pi ** 2 / 6 + scipy.special.polygamma(1, c)) ** 2)
            parametric_median = loc + scale * (-numpy.log(0.5 ** (-1 / c) - 1))
            # parametric_mode = loc + scale * numpy.log(c)

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            # eq3 = parametric_skewness - continuous_measures.skewness
            # eq3 = parametric_kurtosis - continuous_measures.kurtosis
            eq3 = parametric_median - continuous_measures.median
            # eq3 = parametric_mode - continuous_measures.mode

            return (eq1, eq2, eq3)

        # FSOLVE
        # solution = scipy.optimize.fsolve(equations, (1, 1, 1), continuous_measures)
        # parameters = {"loc": solution[0], "scale": solution[1], "c": solution[2]}
        # print(parameters)

        ## least square methods
        x0 = [1, continuous_measures.min, 1]
        bounds = ((1e-5, -numpy.inf, 1e-5), (numpy.inf, numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=[continuous_measures])
        parameters = {"c": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}

        # ## scipy methods
        # scipy_parameters = scipy.stats.genlogistic.fit(continuous_measures.data_to_fit)
        # parameters = {"loc": scipy_parameters[1], "scale": scipy_parameters[2], "c": scipy_parameters[0]}

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
    path = "../continuous_distributions_sample/sample_generalized_logistic.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = GENERALIZED_LOGISTIC(continuous_measures=continuous_measures)

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
