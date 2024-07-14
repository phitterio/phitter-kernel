import numpy
import scipy.optimize
import scipy.stats


class POWER_FUNCTION:
    """
    Power function distribution
    Parameters POWER_FUNCTION distribution: {"alpha": *, "a": *, "b": *}
    https://phitter.io/distributions/continuous/power_function
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        Initializes the POWER_FUNCTION distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters POWER_FUNCTION distribution: {"alpha": *, "a": *, "b": *}
        https://phitter.io/distributions/continuous/power_function
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

        self.alpha = self.parameters["alpha"]
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "power_function"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 11, "a": -13, "b": 99}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        return ((x - self.a) / (self.b - self.a)) ** self.alpha

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        return self.alpha * ((x - self.a) ** (self.alpha - 1)) / ((self.b - self.a) ** self.alpha)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = u ** (1 / self.alpha) * (self.b - self.a) + self.a
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
        if k == 1:
            return (self.a + self.b * self.alpha) / (self.alpha + 1)
        if k == 2:
            return (2 * self.a**2 + 2 * self.alpha * self.a * self.b + self.alpha * (self.alpha + 1) * self.b**2) / ((self.alpha + 1) * (self.alpha + 2))
        if k == 3:
            return (
                6 * self.a**3 + 6 * self.a**2 * self.b * self.alpha + 3 * self.a * self.b**2 * self.alpha * (1 + self.alpha) + self.b**3 * self.alpha * (1 + self.alpha) * (2 + self.alpha)
            ) / ((1 + self.alpha) * (2 + self.alpha) * (3 + self.alpha))
        if k == 4:
            return (
                24 * self.a**4
                + 24 * self.alpha * self.a**3 * self.b
                + 12 * self.alpha * (self.alpha + 1) * self.a**2 * self.b**2
                + 4 * self.alpha * (self.alpha + 1) * (self.alpha + 2) * self.a * self.b**3
                + self.alpha * (self.alpha + 1) * (self.alpha + 2) * (self.alpha + 3) * self.b**4
            ) / ((self.alpha + 1) * (self.alpha + 2) * (self.alpha + 3) * (self.alpha + 4))
        return None

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
        return µ1

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

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
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

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
        return numpy.max([self.a, self.b])

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
        v2 = self.b > self.a
        return v1 and v2

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
        parameters: {"alpha": *, "a": *, "b": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, a, b = initial_solution

            E1 = (a + b * alpha) / (1 + alpha)
            E2 = (2 * a**2 + 2 * a * b * alpha + b**2 * alpha * (1 + alpha)) / ((1 + alpha) * (2 + alpha))
            E3 = (6 * a**3 + 6 * a**2 * b * alpha + 3 * a * b**2 * alpha * (1 + alpha) + b**3 * alpha * (1 + alpha) * (2 + alpha)) / ((1 + alpha) * (2 + alpha) * (3 + alpha))
            # E4 = (24 * a ** 4 + 24 * alpha * a ** 3 * b + 12 * alpha * (alpha + 1) * a ** 2 * b ** 2 + 4 * alpha * (alpha + 1) * (alpha + 2) * a * b ** 3 + alpha * (alpha + 1) * (alpha + 2) * (alpha + 3) * b ** 4) / ((alpha + 1) * (alpha + 2) * (alpha + 3) * (alpha + 4))

            parametric_mean = E1
            parametric_variance = E2 - E1**2
            parametric_skewness = (E3 - 3 * E2 * E1 + 2 * E1**3) / ((E2 - E1**2)) ** 1.5
            # parametric_kurtosis = (E4-4 * E1 * E3 + 6 * E1 ** 2 * E2 - 3 * E1 ** 4) /  ((E2 - E1 ** 2)) ** 2
            # parametric_median = (0.5 ** (1 / alpha)) * (b - a) + a

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            # eq4 = parametric_kurtosis  - continuous_measures.kurtosis
            # eq5 = parametric_median - continuous_measures.median

            return (eq1, eq2, eq3)

        bounds = ((0, -numpy.inf, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, continuous_measures.max)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"alpha": solution.x[0], "a": solution.x[1], "b": continuous_measures.max + 1e-3}

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
    path = "../continuous_distributions_sample/sample_power_function.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = POWER_FUNCTION(continuous_measures=continuous_measures)

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
