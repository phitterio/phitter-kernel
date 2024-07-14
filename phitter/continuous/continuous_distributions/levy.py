import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class LEVY:
    """
    Levy distribution
    Parameters LEVY distribution: {"mu": *, "c": *}
    https://phitter.io/distributions/continuous/levy
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        Initializes the LEVY distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters LEVY distribution: {"mu": *, "c": *}
        https://phitter.io/distributions/continuous/levy
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

        self.mu = self.parameters["mu"]
        self.c = self.parameters["c"]

    @property
    def name(self):
        return "levy"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 0, "c": 1}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        y = lambda x: numpy.sqrt(self.c / ((x - self.mu)))

        # result = scipy.special.erfc(y(x) / numpy.sqrt(2))
        # result = scipy.stats.levy.cdf(x, loc=self.mu, scale=self.c)
        result = 2 - 2 * scipy.stats.norm.cdf(y(x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # result = scipy.stats.levy.pdf(x, loc=self.mu, scale=self.c)
        result = numpy.sqrt(self.c / (2 * numpy.pi)) * numpy.exp(-self.c / (2 * (x - self.mu))) / ((x - self.mu) ** 1.5)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.mu + self.c / scipy.stats.norm.ppf((2 - u) / 2) ** 2
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
        return numpy.inf

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return numpy.inf

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
        return self.mu + self.c / 3

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
        parameters: {"mu": *, "c": *}
        """
        # def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
        #     ## Variables declaration
        #     mu, c = initial_solution

        #     ## Parametric expected expressions
        #     parametric_median = mu +  c / (2 * (scipy.special.erfcinv(0.5) ** 2))
        #     parametric_mode = mu + c / 3

        #     ## System Equations
        #     eq1 = parametric_median - continuous_measures.median
        #     eq2 = parametric_mode - continuous_measures.mode

        #     return (eq1, eq2)

        # bounds = ((-numpy.inf, 0), (numpy.inf, numpy.inf))
        # x0 = (1, 1)
        # args = [continuous_measures]
        # solution = scipy.optimize.least_squares(equations, x0=x0, bounds = bnds, args=args)
        # print(solution.x)
        # parameters = {"mu": solution.x[0], "c": solution.x[1]}

        scipy_parameters = scipy.stats.levy.fit(continuous_measures.data_to_fit)

        ## Results
        parameters = {"mu": scipy_parameters[0], "c": scipy_parameters[1]}

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
    path = "../continuous_distributions_sample/sample_levy.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = LEVY(continuous_measures=continuous_measures)

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
