import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class DAGUM:
    """
    Dagum distribution
    Parameters DAGUM distribution: {"a": *, "b": *, "p": *}
    https://phitter.io/distributions/continuous/dagum
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the DAGUM distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters DAGUM distribution: {"a": *, "b": *, "p": *}
        https://phitter.io/distributions/continuous/dagum
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
        self.b = self.parameters["b"]
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "dagum"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 5, "b": 56, "p": 1}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        return (1 + (x / self.b) ** (-self.a)) ** (-self.p)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        return (self.a * self.p / x) * (((x / self.b) ** (self.a * self.p)) / ((((x / self.b) ** (self.a)) + 1) ** (self.p + 1)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.b * (u ** (-1 / self.p) - 1) ** (-1 / self.a)
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
            self.b**k
            * self.p
            * ((scipy.special.gamma((self.a * self.p + k) / self.a) * scipy.special.gamma((self.a - k) / self.a)) / scipy.special.gamma((self.a * self.p + k) / self.a + (self.a - k) / self.a))
        )

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
        return self.b * ((self.a * self.p - 1) / (self.a + 1)) ** (1 / self.a)

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
        v1 = self.p > 0
        v2 = self.a > 0
        v3 = self.b > 0
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
        parameters: {"a": *, "b": *, "p": *}
        """

        def sse(parameters: dict) -> float:
            def __pdf(x: float, params: dict) -> float:
                return (params["a"] * params["p"] / x) * (((x / params["b"]) ** (params["a"] * params["p"])) / ((((x / params["b"]) ** (params["a"])) + 1) ** (params["p"] + 1)))

            pdf_values = __pdf(continuous_measures.central_values, parameters)
            sse = numpy.sum(numpy.power(continuous_measures.densities_frequencies - pdf_values, 2))
            return sse

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            a, b, p = initial_solution

            ## Generatred moments function (not - centered)
            mu = lambda k: (b**k) * p * scipy.special.beta((a * p + k) / a, (a - k) / a)

            ## Parametric expected expressions
            parametric_mean = mu(1)
            parametric_variance = -(mu(1) ** 2) + mu(2)
            # parametric_skewness = (2 * mu(1) ** 3 - 3 * mu(1) * mu(2) + mu(3)) / (-(mu(1) ** 2) + mu(2)) ** 1.5
            # parametric_kurtosis = (-3 * mu(1) ** 4 + 6 * mu(1) ** 2 * mu(2) -4 * mu(1) * mu(3) + mu(4)) / (-(mu(1) ** 2) + mu(2)) ** 2
            # parametric_median = b * ((2 ** (1 / p)) - 1) ** (-1 / a)
            parametric_mode = b * ((a * p - 1) / (a + 1)) ** (1 / a)

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_mode - continuous_measures.mode
            # eq2 = parametric_median - continuous_measures.median

            return (eq1, eq2, eq3)

        ## Scipy Burr3 = Dagum parameter
        s0_burr3_sc = scipy.stats.burr.fit(continuous_measures.data_to_fit)

        parameters_sc = {"a": s0_burr3_sc[0], "b": s0_burr3_sc[3], "p": s0_burr3_sc[1]}

        ##  Least Square Method
        a0 = s0_burr3_sc[0]
        b0 = s0_burr3_sc[3]
        x0 = [a0, b0, 1]
        bounds = ((1e-5, 1e-5, 1e-5), (numpy.inf, numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=([continuous_measures]))
        parameters_ls = {"a": solution.x[0], "b": solution.x[1], "p": solution.x[2]}

        sse_sc = sse(parameters_sc)
        sse_ls = sse(parameters_ls)

        if a0 <= 2:
            return parameters_sc
        else:
            if sse_sc < sse_ls:
                return parameters_sc
            else:
                return parameters_ls


if __name__ == "__main__":
    ## Import function to get continuous_measures
    import sys

    import numpy

    sys.path.append("../")
    from continuous_measures import CONTINUOUS_MEASURES

    ## Import function to get continuous_measures
    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    ## Distribution class
    path = "../continuous_distributions_sample/sample_dagum.txt"
    data = get_data(path)

    data = DAGUM(init_parameters_examples=True).sample(10000000)

    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = DAGUM(continuous_measures)

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
