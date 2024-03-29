import numpy
import scipy.optimize
import scipy.stats


class PARETO_SECOND_KIND:
    """
    Pareto second kind distribution distribution
    Also known as Lomax Distribution or Pareto Type II distributions
    https://phitter.io/distributions/continuous/pareto_second_kind
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the PARETO_SECOND_KIND distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters PARETO_SECOND_KIND distribution: {"alpha": *, "xm": *, "loc": *}
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.xm = self.parameters["xm"]
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "pareto_second_kind"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"xm": 32, "alpha": 7, "loc": 17}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # result = 1 - (self.xm / ((x - self.loc) + self.xm)) ** self.alpha
        result = scipy.stats.lomax.cdf(x, self.alpha, scale=self.xm, loc=self.loc)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # print(scipy.stats.lomax.pdf(x, self.alpha, scale=self.xm, loc = self.loc))
        return (self.alpha * self.xm**self.alpha) / (((x - self.loc) + self.xm) ** (self.alpha + 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.loc + self.xm / (1 - u) ** (1 / self.alpha) - self.xm
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
        return (self.xm**k * scipy.special.gamma(self.alpha - k) * scipy.special.gamma(1 + k)) / scipy.special.gamma(self.alpha)

    def central_moments(self, k: int) -> float | None:
        """
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = ∫(x - µ[1])ᵏ f(x) dx
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
        return self.loc + µ1

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
        return self.loc

    @property
    def num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restriction
        """
        v1 = self.xm > 0
        v2 = self.alpha > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by formula.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters: {"alpha": *, "xm": *, "loc": *}
        """

        # def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
        #     ## Variables declaration
        #     alpha, xm, loc = initial_solution

        #     ## Generatred moments function (not - centered)

        #     E = lambda k: (scipy.special.gamma(1 + k) * scipy.special.gamma(alpha - k) * xm**k) / scipy.special.gamma(alpha)

        #     ## Parametric expected expressions
        #     parametric_mean = loc + E(1)
        #     parametric_variance = E(2) - E(1) ** 2
        #     # parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
        #     # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
        #     parametric_median = loc + xm * (2 ** (1 / alpha) - 1)
        #     # parametric_mode = loc

        #     ## System Equations
        #     eq1 = parametric_mean - continuous_measures.mean
        #     eq2 = parametric_variance - continuous_measures.variance
        #     # eq3 = parametric_skewness - continuous_measures.skewness
        #     # eq3 = parametric_kurtosis  - continuous_measures.kurtosis
        #     # eq2 = parametric_mode - continuous_measures.mode
        #     eq3 = parametric_median - continuous_measures.median

        #     return (eq1, eq2, eq3)

        # bnds = ((1, 0,  - numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        # x0 = (7, 6, continuous_measures.mode)
        # args = ([continuous_measures])
        # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        # parameters = {"alpha": solution.x[0], "xm": solution.x[1], "loc": solution.x[2]}

        m = continuous_measures.mean
        # me = continuous_measures.median
        # mo = continuous_measures.mode
        v = continuous_measures.variance

        loc = scipy.stats.lomax.fit(continuous_measures.data)[1]
        xm = -((m - loc) * ((m - loc) ** 2 + v)) / ((m - loc) ** 2 - v)
        alpha = -(2 * v) / ((m - loc) ** 2 - v)
        parameters = {"xm": xm, "alpha": alpha, "loc": loc}

        # scipy_params = scipy.stats.lomax.fit(continuous_measures.data)
        # parameters = {"xm": scipy_params[2] , "alpha": scipy_params[0], "loc": scipy_params[1]}

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
    path = "../continuous_distributions_sample/sample_pareto_second_kind.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = PARETO_SECOND_KIND(continuous_measures)

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

    ## Get parameters of distribution: SCIPY vs EQUATIONS
    import time

    ti = time.time()
    print(distribution.get_parameters(continuous_measures))
    print("Solve equations time: ", time.time() - ti)
    ti = time.time()
    print(scipy.stats.lomax.fit(data))
    print("Scipy time get parameters: ", time.time() - ti)
