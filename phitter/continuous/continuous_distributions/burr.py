import warnings

import numpy
import scipy.optimize
import scipy.special
import scipy.stats

warnings.filterwarnings("ignore")


class Burr:
    """
    Burr distribution
    - Parameters Burr Distribution: {"A": *, "B": *, "C": *}
    - https://phitter.io/distributions/continuous/burr
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Burr Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters Burr Distribution: {"A": *, "B": *, "C": *}
        - https://phitter.io/distributions/continuous/burr
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

        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
        self.C = self.parameters["C"]

    @property
    def name(self):
        return "burr"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"A": 1, "B": 10, "C": 5}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # result = 1 - ((1 + (x / self.A) ** (self.B)) ** (-self.C))
        result = scipy.stats.burr12.cdf(x, self.B, self.C, scale=self.A)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # result = ((self.B * self.C) / self.A) * ((x / self.A) ** (self.B - 1)) * ((1 + (x / self.A) ** (self.B)) ** (-self.C - 1))
        result = scipy.stats.burr12.pdf(x, self.B, self.C, scale=self.A)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.A * ((1 - u) ** (-1 / self.C) - 1) ** (1 / self.B)
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
        return (
            self.A**k
            * self.C
            * ((scipy.special.gamma((self.B * self.C - k) / self.B) * scipy.special.gamma((self.B + k) / self.B)) / scipy.special.gamma((self.B * self.C - k) / self.B + (self.B + k) / self.B))
        )

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
        return self.A * ((self.B - 1) / (self.B * self.C + 1)) ** (1 / self.B)

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
        v1 = self.A > 0
        v2 = self.C > 0
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
        parameters: {"A": *, "B": *, "C": *}
        """
        # def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
        #     ## Variables declaration
        #     A, B, C = initial_solution

        #     ## Moments Burr Distribution
        #     mu = lambda r: (A**r) * C * scipy.special.beta((B * C - r) / B, (B + r) / B)

        #     ## Parametric expected expressions
        #     parametric_mean = mu(1)
        #     # parametric_variance = -(mu(1) ** 2) + mu(2)
        #     # parametric_skewness = 2 * mu(1) ** 3 - 3 * mu(1) * mu(2) + mu(3)
        #     # parametric_kurtosis = -3 * mu(1) ** 4 + 6 * mu(1) ** 2 * mu(2) -4 * mu(1) * mu(3) + mu(4)
        #     parametric_median = A * ((2 ** (1 / C)) - 1) ** (1 / B)
        #     parametric_mode = A * ((B - 1) / (B * C + 1)) ** (1 / B)

        #     ## System Equations
        #     eq1 = parametric_mean - continuous_measures.mean
        #     eq2 = parametric_median - continuous_measures.median
        #     eq3 = parametric_mode - continuous_measures.mode
        #     # eq3 = parametric_kurtosis - continuous_measures.kurtosis
        #     # eq3 = parametric_skewness - continuous_measures.skewness
        #     # eq2 = parametric_variance - continuous_measures.variance

        #     return (eq1, eq2, eq3)

        ## Solve equations system
        # x0 = [continuous_measures.mean, continuous_measures.mean, continuous_measures.mean]
        # bounds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        # solution = scipy.optimize.least_squares(equations, x0=x0, bounds = b, args=[continuous_measures])
        # parameters = {"A": solution.x[0], "B": solution.x[1], "C": solution.x[2]}

        # Scipy class
        scipy_parameters = scipy.stats.burr12.fit(continuous_measures.data_to_fit)
        parameters = {"A": scipy_parameters[3], "B": scipy_parameters[0], "C": scipy_parameters[1]}
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
    path = "../../../distributions_samples/continuous_distributions_sample/sample_burr.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = Burr(continuous_measures=continuous_measures)
    print(distribution.get_parameters(continuous_measures=continuous_measures))

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
