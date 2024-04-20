import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class GENERALIZED_GAMMA_4P:
    """
    Generalized Gamma Distribution
    https://phitter.io/distributions/continuous/generalized_gamma_4p
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the GENERALIZED_GAMMA_4P distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters GENERALIZED_GAMMA_4P distribution: {"a": *, "d": *, "p": *, "loc": *}
        https://phitter.io/distributions/continuous/generalized_gamma_4p
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "generalized_gamma_4p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 2, "d": 13, "p": 3, "loc": 28}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # result = scipy.stats.gamma.cdf(((x - self.loc) / self.a) ** self.p, a=self.d / self.p, scale=1)
        result = scipy.special.gammainc(self.d / self.p, ((x - self.loc) / self.a) ** self.p)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        return (self.p / (self.a**self.d)) * ((x - self.loc) ** (self.d - 1)) * numpy.exp(-(((x - self.loc) / self.a) ** self.p)) / scipy.special.gamma(self.d / self.p)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.a * scipy.special.gammaincinv(self.d / self.p, u) ** (1 / self.p)
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
        return (self.a**k * scipy.special.gamma((self.d + k) / self.p)) / scipy.special.gamma(self.d / self.p)

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
        return self.loc + self.a * ((self.d - 1) / self.p) ** (1 / self.p)

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
        v1 = self.a > 0
        v2 = self.d > 0
        v3 = self.p > 0
        return v1 and v2 and v3

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
        parameters: {"a": *, "d": *, "p": *, "loc": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, d, p, loc = initial_solution

            E = lambda r: a**r * (scipy.special.gamma((d + r) / p) / scipy.special.gamma(d / p))

            parametric_mean = E(1) + loc
            parametric_variance = E(2) - E(1) ** 2
            # parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) / ((E(2) - E(1) ** 2)) ** 2
            parametric_median = a * scipy.stats.gamma.ppf(0.5, a=d / p, scale=1) ** (1 / p) + loc
            # parametric_mode = loc + a * ((d - 1) / p) ** (1 / p)

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            # eq3 = parametric_skewness - continuous_measures.skewness
            eq3 = parametric_median - continuous_measures.median
            eq4 = parametric_kurtosis - continuous_measures.kurtosis

            return (eq1, eq2, eq3, eq4)

        ## scipy.optimize.fsolve is 100x faster than least square but sometimes return solutions < 0
        solution = scipy.optimize.fsolve(equations, (1, 1, 1, 1), continuous_measures)

        ## If return a perameter < 0 then use least_square with restriction
        if all(x > 0 for x in solution) is False or all(x == 1 for x in solution) is True:
            try:
                bounds = ((0, 0, 0, 0), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
                if continuous_measures.mean < 0:
                    bounds = ((0, 0, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, 0))
                x0 = (1, 1, 1, continuous_measures.mean)
                args = [continuous_measures]
                response = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
                solution = response.x
            except:
                scipy_params = scipy.stats.gengamma.fit(continuous_measures.data_to_fit)
                solution = [scipy_params[3], scipy_params[0], scipy_params[1], scipy_params[2]]

        parameters = {"a": solution[0], "d": solution[1], "p": solution[2], "loc": solution[3]}

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
    path = "../continuous_distributions_sample/sample_generalized_gamma_4p.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = GENERALIZED_GAMMA_4P(continuous_measures)

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

    scipy_params = scipy.stats.gengamma.fit(continuous_measures.data_to_fit)
    parameters = {"a": scipy_params[3], "d": scipy_params[0], "p": scipy_params[1], "loc": scipy_params[2]}
    print(parameters)
