import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class ChiSquare3P:
    """
    Chi Square distribution
    - Parameters ChiSquare3P Distribution: {"df": \*, "loc": \*, "scale": \*}
    - https://phitter.io/distributions/continuous/chi_square_3p
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the ChiSquare3P Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters ChiSquare3P Distribution: {"df": \*, "loc": \*, "scale": \*}
        - https://phitter.io/distributions/continuous/chi_square_3p
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

        self.df = self.parameters["df"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "chi_square_3p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"df": 4, "loc": 10, "scale": 2}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # result, error = scipy.integrate.quad(self.pdf, 0, x)
        # result = scipy.stats.chi2.cdf(x, self.df, self.loc, self.scale)
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.special.gammainc(self.df / 2, z(x) / 2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # result = scipy.stats.chi2.pdf(x, self.df, loc=self.loc, scale=self.scale)
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / self.scale) * (1 / (2 ** (self.df / 2) * scipy.special.gamma(self.df / 2))) * (z(x) ** ((self.df / 2) - 1)) * (numpy.exp(-z(x) / 2))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = 2 * self.scale * scipy.special.gammaincinv(self.df / 2, u) + self.loc
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
        return self.df * self.scale + self.loc

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return self.df * 2 * (self.scale * self.scale)

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
        return numpy.sqrt(8 / self.df)

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 12 / self.df + 3

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
        return max(self.df - 2, 0) * self.scale + self.loc

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
        v1 = self.df > 0
        v2 = type(self.df) == int
        v2 = True
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
        parameters: {"df": \*, "loc": \*, "scale": \*}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            df, loc, scale = initial_solution

            ## Parametric expected expressions
            parametric_mean = df * scale + loc
            parametric_variance = 2 * df * (scale**2)
            parametric_skewness = numpy.sqrt(8 / df)
            # parametric_kurtosis = 12 / df  + 3
            parametric_median = 2 * scale * scipy.special.gammaincinv(df / 2, 0.5) + loc
            parametric_mode = max(df - 2, 0) * scale + loc

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            # eq3 = parametric_skewness - continuous_measures.skewness
            # eq4 = parametric_kurtosis  - continuous_measures.kurtosis
            eq3 = parametric_median - continuous_measures.median
            # eq3 = parametric_mode - continuous_measures.mode

            return (eq1, eq2, eq3)

        ## Method 1: FSOLVE
        # solution = scipy.optimize.fsolve(equations, (1, 1, 1), continuous_measures)
        # print(solution)

        ## Method 2: LEAST SQUARE
        x0 = (1, continuous_measures.mean, 1)
        bounds = ((0, -numpy.inf, 0), (numpy.inf, numpy.inf, numpy.inf))
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"df": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}

        ## Method 2: Solve system
        # df = 8 / (continuous_measures.skewness ** 2)
        # df = 12 / (continuous_measures.kurtosis - 3)
        # scale = numpy.sqrt(continuous_measures.variance / (2 * df))
        # loc = continuous_measures.mean - df * scale
        # parameters = {"df": round(df), "loc": loc, "scale": scale}

        ## Scipy FIT
        # scipy_parameters = scipy.stats.chi2.fit(continuous_measures.data_to_fit)
        # parameters = {"df": round(scipy_parameters[0]), "loc": scipy_parameters[1], "scale": scipy_parameters[2]}

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
    path = "../continuous_distributions_sample/sample_chi_square_3p.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = ChiSquare3P(continuous_measures=continuous_measures)

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
