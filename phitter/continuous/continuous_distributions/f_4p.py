import numpy
import scipy.special
import scipy.stats


class F_4P:
    """
    F distribution
    Parameters F_4P distribution: {"df1": *, "df2": *, "loc": *, "scale": *}
    https://phitter.io/distributions/continuous/f_4p
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the F_4P distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters F_4P distribution: {"df1": *, "df2": *, "loc": *, "scale": *}
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.df1 = self.parameters["df1"]
        self.df2 = self.parameters["df2"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "f_4p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"df1": 76, "df2": 36, "loc": 925, "scale": 197}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # z = lambda t: (t - self.loc) / self.scale
        # result = scipy.special.betainc(self.df1 / 2, self.df2 / 2, z(x) * self.df1 / (self.df1 * z(x) + self.df2))
        result = scipy.stats.f.cdf(x, self.df1, self.df2, self.loc, self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # z = lambda t: (t - self.loc) / self.scale
        # result = (
        #     (1 / self.scale)
        #     * (1 / scipy.special.beta(self.df1 / 2, self.df2 / 2))
        #     * ((self.df1 / self.df2) ** (self.df1 / 2))
        #     * (z(x) ** (self.df1 / 2 - 1))
        #     * ((1 + z(x) * self.df1 / self.df2) ** (-1 * (self.df1 + self.df2) / 2))
        # )
        result = scipy.stats.f.pdf(x, self.df1, self.df2, self.loc, self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        t = scipy.special.betaincinv(self.df1 / 2, self.df2 / 2, u)
        result = self.loc + (self.scale * (self.df2 * t)) / (self.df1 * (1 - t))
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
        return (self.df2 / self.df1) ** k * (scipy.special.gamma(self.df1 / 2 + k) / scipy.special.gamma(self.df1 / 2)) * (scipy.special.gamma(self.df2 / 2 - k) / scipy.special.gamma(self.df2 / 2))

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
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale**2 * (µ2 - µ1**2)

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
        return ((self.df2 * (self.df1 - 2)) / (self.df1 * (self.df2 + 2))) * self.scale + self.loc

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
        v1 = self.df1 > 0
        v2 = self.df2 > 0
        v3 = self.scale > 0
        return v1 and v2 and v3

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
        parameters: {"df1": *, "df2": *, "loc": *, "scale": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            df1, df2, loc, scale = initial_solution

            ## Generatred moments function (not - centered)
            E = lambda k: (df2 / df1) ** k * (scipy.special.gamma(df1 / 2 + k) * scipy.special.gamma(df2 / 2 - k)) / (scipy.special.gamma(df1 / 2) * scipy.special.gamma(df2 / 2))

            ## Parametric expected expressions
            parametric_mean = E(1) * scale + loc
            parametric_variance = (E(2) - E(1) ** 2) * (scale) ** 2
            # parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
            parametric_median = scipy.stats.f.ppf(0.5, df1, df2) * scale + loc
            parametric_mode = ((df2 * (df1 - 2)) / (df1 * (df2 + 2))) * scale + loc

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            # eq2 = parametric_median - continuous_measures.median
            # eq2 = parametric_skewness - continuous_measures.skewness
            # eq2 = parametric_kurtosis  - continuous_measures.kurtosis
            eq3 = parametric_median - continuous_measures.median
            eq4 = parametric_mode - continuous_measures.mode
            return (eq1, eq2, eq3, eq4)

        try:
            bnds = ((0, 0, -numpy.inf, 0), (numpy.inf, numpy.inf, continuous_measures.min, numpy.inf))
            x0 = (1, continuous_measures.standard_deviation, continuous_measures.min, continuous_measures.standard_deviation)
            args = [continuous_measures]
            solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
            parameters = {"df1": solution.x[0], "df2": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
        except:
            ## Scipy parameters of distribution
            scipy_params = scipy.stats.f.fit(continuous_measures.data)

            ## Results
            parameters = {"df1": scipy_params[0], "df2": scipy_params[1], "loc": scipy_params[2], "scale": scipy_params[3]}

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
    path = "../continuous_distributions_sample/sample_f_4p.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = F_4P(continuous_measures)

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
