import numpy
import scipy.integrate
import scipy.optimize
import scipy.special


class Kumaraswamy:
    """
    Kumaraswami distribution
    - Parameters Kumaraswamy Distribution: {"alpha": *, "beta": *, "min": *, "max": *}
    - https://phitter.io/distributions/continuous/kumaraswamy
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Kumaraswamy Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters Kumaraswamy Distribution: {"alpha": *, "beta": *, "min": *, "max": *}
        - https://phitter.io/distributions/continuous/kumaraswamy
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
        self.beta = self.parameters["beta"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "kumaraswamy"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 7, "beta": 5, "min": 11, "max": 19}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.min) / (self.max - self.min)
        result = 1 - (1 - z(x) ** self.alpha) ** self.beta
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.min) / (self.max - self.min)
        return (self.alpha * self.beta) * (z(x) ** (self.alpha - 1)) * ((1 - z(x) ** self.alpha) ** (self.beta - 1)) / (self.max - self.min)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = (1 - (1 - u) ** (1 / self.beta)) ** (1 / self.alpha) * (self.max - self.min) + self.min
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
        return (self.beta * scipy.special.gamma(1 + k / self.alpha) * scipy.special.gamma(self.beta)) / scipy.special.gamma(1 + self.beta + k / self.alpha)

    def central_moments(self, k: int) -> float | None:
        """
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)

        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4

        return None

    @property
    def mean(self) -> float:
        """
        Parametric mean
        """
        µ1 = self.non_central_moments(1)
        return self.min + (self.max - self.min) * µ1

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (self.max - self.min) ** 2 * (µ2 - µ1**2)

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
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

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
        return self.min + (self.max - self.min) * ((self.alpha - 1) / (self.alpha * self.beta - 1)) ** (1 / self.alpha)

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
        v2 = self.beta > 0
        v3 = self.min < self.max
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by solving the equations of the measures expected
        for this distribution.The number of equations to consider is equal to the number
        of parameters.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"alpha": *, "beta": *, "min": *, "max": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            alpha, beta, min_, max_ = initial_solution

            ## Generatred moments function (not - centered)
            E = lambda r: beta * scipy.special.gamma(1 + r / alpha) * scipy.special.gamma(beta) / scipy.special.gamma(1 + beta + r / alpha)

            ## Parametric expected expressions
            parametric_mean = E(1) * (max_ - min_) + min_
            parametric_variance = (E(2) - E(1) ** 2) * (max_ - min_) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) / ((E(2) - E(1) ** 2)) ** 2
            # parametric_median = ((1 - 2 ** (-1 / beta)) ** (1 / alpha)) * (max_ - min_) + min_
            # parametric_mode = min_ + (max_ - min_) * ((alpha - 1) / (alpha * beta - 1)) ** (1 / alpha)

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            # eq3 = parametric_median - continuous_measures.median
            # eq4 = parametric_mode - continuous_measures.mode

            return (eq1, eq2, eq3, eq4)

        bounds = ((1e-5, 1e-5, -numpy.inf, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, continuous_measures.min, continuous_measures.max)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)

        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "min": solution.x[2], "max": solution.x[3]}
        return parameters


if __name__ == "__main__":
    ## Import function to get continuous_measures
    import sys

    import numpy

    sys.path.append("../")
    from continuous_measures import ContinuousMeasures

    ## Import function to get continuous_measures
    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    ## Distribution class
    path = "../../../distributions_samples/continuous_distributions_sample/sample_kumaraswamy.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = Kumaraswamy(continuous_measures=continuous_measures)

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
