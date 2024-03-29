import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class BETA:
    """
    Beta distribution
    Parameters BETA distribution: {"alpha": *, "beta": *, "A": *, "B": *}
    https://phitter.io/distributions/continuous/beta
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the BETA distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters BETA distribution: {"alpha": *, "beta": *, "A": *, "B": *}
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]

    @property
    def name(self):
        return "beta"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 42, "beta": 10, "A": 518, "B": 969}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # z = lambda t: (t - self.A) / (self.B - self.A)
        # result = scipy.stats.beta.cdf(z(x), self.alpha, self.beta)
        result = scipy.stats.beta.cdf(x, self.alpha, self.beta, loc=self.A, scale=self.B - self.A)
        # result = scipy.special.betainc(self.alpha, self.beta, z(x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        result = scipy.stats.beta.pdf(x, self.alpha, self.beta, loc=self.A, scale=self.B - self.A)
        # z = lambda t: (t - self.A) / (self.B - self.A)
        # result = (1 / (self.B - self.A)) * (scipy.special.gamma(self.alpha + self.beta) / (scipy.special.gamma(self.alpha) * scipy.special.gamma(self.beta))) * (z(x) ** (self.alpha - 1)) * ((1 - z(x)) ** (self.beta - 1))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.A + (self.B - self.A) * scipy.special.betaincinv(self.alpha, self.beta, u)
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
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = ∫(x - µ[1])ᵏ f(x) dx
        """
        return None

    @property
    def mean(self) -> float:
        """
        Parametric mean
        """
        return self.A + (self.alpha / (self.alpha + self.beta)) * (self.B - self.A)

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return ((self.alpha * self.beta) / ((self.alpha + self.beta + 1) * (self.alpha + self.beta) ** 2)) * (self.B - self.A) ** 2

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
        return 2 * ((self.beta - self.alpha) / (self.alpha + self.beta + 2)) * numpy.sqrt((self.alpha + self.beta + 1) / (self.alpha * self.beta))

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 3 + (6 * ((self.alpha + self.beta + 1) * (self.alpha - self.beta) ** 2 - self.alpha * self.beta * (self.alpha + self.beta + 2))) / (
            self.alpha * self.beta * (self.alpha + self.beta + 2) * (self.alpha + self.beta + 3)
        )

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
        return self.A + ((self.alpha - 1) / (self.alpha + self.beta - 2)) * (self.B - self.A)

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
        v3 = self.A < self.B
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
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters: {"alpha": *, "beta": *, "A": *, "B": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            alpha, beta, A, B = initial_solution

            ## Parametric expected expressions
            parametric_mean = A + (alpha / (alpha + beta)) * (B - A)
            parametric_variance = ((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))) * (B - A) ** 2
            parametric_skewness = 2 * ((beta - alpha) / (alpha + beta + 2)) * numpy.sqrt((alpha + beta + 1) / (alpha * beta))
            parametric_kurtosis = 3 * (((alpha + beta + 1) * (2 * (alpha + beta) ** 2 + (alpha * beta) * (alpha + beta - 6))) / ((alpha * beta) * (alpha + beta + 2) * (alpha + beta + 3)))

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis

            return (eq1, eq2, eq3, eq4)

        bnds = ((0, 0, -numpy.inf, continuous_measures.mean), (numpy.inf, numpy.inf, continuous_measures.mean, numpy.inf))
        x0 = (1, 1, continuous_measures.min, continuous_measures.max)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "A": solution.x[2], "B": solution.x[3]}

        v1 = parameters["alpha"] > 0
        v2 = parameters["beta"] > 0
        v3 = parameters["A"] < parameters["B"]
        if (v1 and v2 and v3) == False:
            scipy_params = scipy.stats.beta.fit(continuous_measures.data)
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1], "A": scipy_params[2], "B": scipy_params[3]}
        return parameters


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
    path = "../continuous_distributions_sample/sample_beta.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = BETA(continuous_measures)

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

    # def equations(initial_solution: list[float], continuous_measures) -> tuple[float]:
    #     ## Variables declaration
    #     alpha, beta, A, B = initial_solution

    #     ## Parametric expected expressions
    #     parametric_mean = A + (alpha / ( alpha + beta )) * (B - A)
    #     parametric_variance = ((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))) * (B - A) ** 2
    #     parametric_skewness = 2 * ((beta - alpha) / (alpha + beta + 2)) * numpy.sqrt((alpha + beta + 1) / (alpha * beta))
    #     parametric_kurtosis = 3 * (((alpha + beta + 1) * (2 * (alpha + beta) ** 2  + (alpha * beta) * (alpha + beta - 6))) / ((alpha * beta) * (alpha + beta + 2) * (alpha + beta + 3)))

    #     ## System Equations
    #     eq1 = parametric_mean - continuous_measures.mean
    #     eq2 = parametric_variance - continuous_measures.variance
    #     eq3 = parametric_skewness - continuous_measures.skewness
    #     eq4 = parametric_kurtosis  - continuous_measures.kurtosis

    #     return (eq1, eq2, eq3, eq4)

    # ## Get parameters of distribution: SCIPY vs EQUATIONS
    # import time
    # print("=====")
    # ti = time.time()
    # solution = scipy.optimize.fsolve(equations, (1, 1, 1, 1), continuous_measures)
    # parameters = {"alpha": solution[0], "beta": solution[1], "A": solution[2], "B": solution[3]}
    # print(parameters)
    # print("Solve equations time: ", time.time() - ti)

    # print("=====")
    # ti = time.time()
    # scipy_params = scipy.stats.beta.fit(continuous_measures.data)
    # parameters = {"alpha": scipy_params[0], "beta": scipy_params[1], "A": scipy_params[2], "B": scipy_params[3]}
    # print(parameters)
    # print("Scipy time get parameters: ",time.time() - ti)

    # print("=====")

    # ti = time.time()
    # bnds = ((0, 0,  - numpy.inf, continuous_measures.mean), (numpy.inf, numpy.inf, continuous_measures.mean, numpy.inf))
    # x0 = (1, 1, continuous_measures.min, continuous_measures.max)
    # args = ([continuous_measures])
    # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
    # print(solution.x)
    # print("Solve equations time: ", time.time() - ti)
