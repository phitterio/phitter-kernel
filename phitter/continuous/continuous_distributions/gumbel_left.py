import numpy
import scipy.optimize


class GumbelLeft:
    """
    Gumbel Left Distribution
    Gumbel Min Distribution
    Extreme Value Minimum Distribution
    - https://phitter.io/distributions/continuous/gumbel_left
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the GumbelLeft Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters GumbelLeft Distribution: {"mu": *, "sigma": *}
        - https://phitter.io/distributions/continuous/gumbel_left
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

        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "gumbel_left"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 100, "sigma": 30}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.mu) / self.sigma
        return 1 - numpy.exp(-numpy.exp(z(x)))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.mu) / self.sigma
        return (1 / self.sigma) * numpy.exp(z(x) - numpy.exp(z(x)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.mu + self.sigma * numpy.log(-numpy.log(1 - u))
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
        return self.mu - 0.5772156649 * self.sigma

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return self.sigma**2 * (numpy.pi**2 / 6)

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
        return (-12 * numpy.sqrt(6) * 1.20205690315959) / numpy.pi**3

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 3 + 12 / 5

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
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"mu": *, "sigma": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            mu, sigma = initial_solution

            ## Parametric expected expressions
            parametric_mean = mu - sigma * 0.5772156649
            parametric_variance = (sigma**2) * (numpy.pi**2) / 6

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance

            return (eq1, eq2)

        ## fsolve
        # solution = scipy.optimize.fsolve(equations, (1, 1), continuous_measures)
        # parameters = {"mu": solution[0], "sigma": solution[1]}

        ## least square methods
        x0 = [continuous_measures.mode, 1]
        bounds = ((-numpy.inf, 1e-5), (numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=[continuous_measures])
        parameters = {"mu": solution.x[0], "sigma": solution.x[1]}

        ## solve system
        # mu = continuous_measures.mode
        # sigma = (mu - continuous_measures.mean) / 0.5772156649
        # parameters = {"mu": mu, "sigma": sigma}
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
    path = "../../../distributions_samples/continuous_distributions_sample/sample_gumbel_left.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = GumbelLeft(continuous_measures=continuous_measures)

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
