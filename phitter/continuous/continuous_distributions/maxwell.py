import numpy
import scipy.special


class Maxwell:
    """
    Maxwell distribution
    - Parameters Maxwell Distribution: {"alpha": *, "loc": *}
    - https://phitter.io/distributions/continuous/maxwell
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Maxwell Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters Maxwell Distribution: {"alpha": *, "loc": *}
        - https://phitter.io/distributions/continuous/maxwell
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

        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "maxwell"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 60, "loc": 100}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # result = scipy.stats.maxwell.cdf(x, loc = self.loc, scale = self.alpha)
        z = lambda t: (t - self.loc) / self.alpha
        result = scipy.special.erf(z(x) / (numpy.sqrt(2))) - numpy.sqrt(2 / numpy.pi) * z(x) * numpy.exp(-z(x) ** 2 / 2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # result = scipy.stats.maxwell.pdf(x, loc = self.loc, scale = self.alpha)
        z = lambda t: (t - self.loc) / self.alpha
        result = 1 / self.alpha * numpy.sqrt(2 / numpy.pi) * z(x) ** 2 * numpy.exp(-z(x) ** 2 / 2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.alpha * numpy.sqrt(2 * scipy.special.gammaincinv(1.5, u)) + self.loc
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
        return 2 * numpy.sqrt(2 / numpy.pi) * self.alpha + self.loc

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (self.alpha * self.alpha * (3 * numpy.pi - 8)) / numpy.pi

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
        return (2 * numpy.sqrt(2) * (16 - 5 * numpy.pi)) / (3 * numpy.pi - 8) ** 1.5

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return (4 * (-96 + 40 * numpy.pi - 3 * numpy.pi * numpy.pi)) / (3 * numpy.pi - 8) ** 2 + 3

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
        return numpy.sqrt(2) * self.alpha + self.loc

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
        v1 = self.alpha > 0
        return v1

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
        parameters: {"alpha": *, "loc": *}
        """
        # def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
        #     alpha, loc = initial_solution

        #     ## Parametric expected expressions
        #     parametric_mean = loc + 2 * alpha * numpy.sqrt(2 / numpy.pi)
        #     parametric_variance = alpha ** 2 * (3 * numpy.pi - 8) / numpy.pi
        #     parametric_median = loc + alpha * numpy.sqrt(2 * scipy.special.gammaincinv(1.5, 0.5))
        #     # parametric_mode = loc + sigma * alpha * numpy.sqrt(2)

        #     ## System Equations
        #     eq1 = parametric_mean - continuous_measures.mean
        #     eq2 = parametric_variance - continuous_measures.variance
        #     # eq3 = parametric_mode  - continuous_measures.mode
        #     eq3 = parametric_median - continuous_measures.median

        #     return (eq1, eq2, eq3)

        # bounds = ((0,  - numpy.inf), (numpy.inf, numpy.inf))
        # x0 = (1, continuous_measures.mean)
        # args = [continuous_measures]
        # solution = scipy.optimize.least_squares(equations, x0=x0, bounds = bnds, args=args)
        # parameters = {"alpha": solution.x[0], "loc": solution.x[1]}

        alpha = numpy.sqrt(continuous_measures.variance * numpy.pi / (3 * numpy.pi - 8))
        loc = continuous_measures.mean - 2 * alpha * numpy.sqrt(2 / numpy.pi)
        parameters = {"alpha": alpha, "loc": loc}

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
    path = "../../../distributions_samples/continuous_distributions_sample/sample_maxwell.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = Maxwell(continuous_measures=continuous_measures)

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

    # print(scipy.stats.ncf.fit(data))
