import warnings

import numpy
import scipy.integrate
import scipy.optimize
import scipy.stats

warnings.filterwarnings("ignore")


class Alpha:
    """
    Alpha distribution
    - Parameters Alpha Distribution: {"alpha": *, "loc": *, "scale": *}
    - https://phitter.io/distributions/continuous/alpha
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Alpha Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters Alpha Distribution: {"alpha": *, "loc": *, "scale": *}
        - https://phitter.io/distributions/continuous/alpha
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
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "alpha"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 5, "loc": 9, "scale": 57}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # print(scipy.stats.alpha.cdf(x, self.alpha, loc=self.loc, scale=self.scale))
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.stats.norm.cdf(self.alpha - (1 / z(x))) / scipy.stats.norm.cdf(self.alpha)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # z = lambda t: (t - self.loc) / self.scale
        # result = (1 / (self.scale * z(x) * z(x) * scipy.stats.norm.cdf(self.alpha) * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-0.5 * (self.alpha - 1 / z(x)) ** 2)
        result = scipy.stats.alpha.pdf(x, self.alpha, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        # result = scipy.stats.alpha.ppf(u, self.alpha, loc=self.loc, scale=self.scale)
        result = self.loc + self.scale / (self.alpha - scipy.stats.norm.ppf(u * scipy.stats.norm.cdf(self.alpha)))
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
        f = lambda x: x**k * scipy.stats.alpha.pdf(x, self.alpha, loc=0, scale=1)
        return scipy.integrate.quad(f, 0, 2)[0]

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
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale * self.scale * (µ2 - µ1**2)

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
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ3 = self.central_moments(3)
        return central_µ3 / (µ2 - µ1**2) ** 1.5

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ4 = self.central_moments(4)
        return central_µ4 / (µ2 - µ1**2) ** 2

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
        return (self.scale * (numpy.sqrt(self.alpha * self.alpha + 8) - self.alpha)) / 4 + self.loc

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
        v2 = self.scale > 0
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
        parameters: {"alpha": *, "loc": *, "scale": *}
        """
        # def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
        #     alpha, loc, scale = initial_solution

        #     z = lambda t: (t - loc) / scale
        #     pdf = lambda x: (1 / (scale * z(x) * z(x) * scipy.stats.norm.cdf(alpha) * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-0.5 * (alpha - 1 / z(x)) ** 2)

        #     ## Generatred moments function (not - centered)
        #     E_1 = scipy.integrate.quad(lambda x: x ** 1 * pdf(x), 0, 10 * continuous_measures.max)[0]
        #     # E_2 = scipy.integrate.quad(lambda x: x ** 2 * pdf(x), 0, 10 * continuous_measures.max)[0]
        #     # E_3 = scipy.integrate.quad(lambda x: x ** 3 * pdf(x), 0, 10 * continuous_measures.max)[0]
        #     # E_4 = scipy.integrate.quad(lambda x: x ** 4 * pdf(x), 0, 100)[0]

        #     ## Parametric expected expressions
        #     parametric_mean = E_1
        #     # parametric_variance = E_2 - E_1 ** 2
        #     # parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1 ** 3) / ((E_2 - E_1 ** 2)) ** 1.5
        #     # parametric_kurtosis = (E_4-4 * E_1 * E_3 + 6 * E_1 ** 2 * E_2 - 3 * E_1 ** 4) /  ((E_2 - E_1 ** 2)) ** 2
        #     parametric_median = loc + scale / (alpha - scipy.stats.norm.ppf(0.5 * scipy.stats.norm.cdf(alpha)))
        #     parametric_mode = loc + scale * (numpy.sqrt(alpha * alpha + 8) - alpha) / 4

        #     ## System Equations
        #     eq1 = parametric_mean - continuous_measures.mean
        #     # eq2 = parametric_variance - continuous_measures.variance
        #     # eq2 = parametric_skewness - continuous_measures.skewness
        #     # eq4 = parametric_kurtosis  - continuous_measures.kurtosis
        #     eq2 = parametric_median - continuous_measures.median
        #     eq3 = parametric_mode - continuous_measures.mode

        #     return (eq1, eq2, eq3)

        ## THIS METHOD IS CORRECT, BUT IS VERY SLOW BECAUSE THE INTEGRATION
        # bounds = ((0, 9, 0), (numpy.inf, continuous_measures.mean, numpy.inf))
        # x0 = (1, continuous_measures.mean, 1)
        # args = [continuous_measures]
        # solution = scipy.optimize.least_squares(equations, x0=x0, bounds = bnds, args=args)
        # parameters = {"alpha": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}

        # solution = scipy.optimize.fsolve(equations, (1, 1, 1), continuous_measures)
        # parameters = {"alpha": solution[0], "loc": solution[1], "scale": solution[2]}

        scipy_parameters = scipy.stats.alpha.fit(continuous_measures.data_to_fit)
        parameters = {"alpha": scipy_parameters[0], "loc": scipy_parameters[1], "scale": scipy_parameters[2]}
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
    path = "../../../distributions_samples/continuous_distributions_sample/sample_alpha.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = Alpha(continuous_measures=continuous_measures)

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
