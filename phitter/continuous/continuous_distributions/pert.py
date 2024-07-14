import numpy
import scipy.optimize
import scipy.special


class PERT:
    """
    Pert distribution
    Parameters PERT distribution: {"a": *, "b": *, "c": *}
    https://phitter.io/distributions/continuous/pert
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        Initializes the PERT distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters PERT distribution: {"a": *, "b": *, "c": *}
        https://phitter.io/distributions/continuous/pert
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

        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        self.alpha1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        self.alpha2 = (5 * self.c - self.a - 4 * self.b) / (self.c - self.a)

    @property
    def name(self):
        return "pert"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 63, "b": 513, "c": 970}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.a) / (self.c - self.a)

        # result = scipy.stats.beta.cdf(z(x), self.alpha1, self.alpha2)
        result = scipy.special.betainc(self.alpha1, self.alpha2, z(x))

        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        return (x - self.a) ** (self.alpha1 - 1) * (self.c - x) ** (self.alpha2 - 1) / (scipy.special.beta(self.alpha1, self.alpha2) * (self.c - self.a) ** (self.alpha1 + self.alpha2 - 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.a + (self.c - self.a) * scipy.special.betaincinv(self.alpha1, self.alpha2, u)
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
        return (self.a + 4 * self.b + self.c) / 6

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return ((self.mean - self.a) * (self.c - self.mean)) / 7

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
        return (2 * (self.alpha2 - self.alpha1) * numpy.sqrt(self.alpha1 + self.alpha2 + 1)) / ((self.alpha1 + self.alpha2 + 2) * numpy.sqrt(self.alpha1 * self.alpha2))

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return (6 * ((self.alpha2 - self.alpha1) ** 2 * (self.alpha1 + self.alpha2 + 1) - self.alpha1 * self.alpha2 * (self.alpha1 + self.alpha2 + 2))) / (
            self.alpha1 * self.alpha2 * (self.alpha1 + self.alpha2 + 2) * (self.alpha1 + self.alpha2 + 3)
        ) + 3

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
        return self.b

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
        v1 = self.a < self.b < self.c
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by solving the equations of the measures expected
        for this distribution.The number of equations to consider is equal to the number
        of parameters.

        Parameters
        ==========
        continuous_measures : dict
            {"mean": * , "variance": * , "skewness": * , "kurtosis": * , "median": * , "b": * }

        Returns
        =======
        parameters: {"a": *, "b": *, "c": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, b, c = initial_solution

            self.alpha1 = (4 * b + c - 5 * a) / (c - a)
            self.alpha2 = (5 * c - a - 4 * b) / (c - a)

            parametric_mean = (a + 4 * b + c) / 6
            parametric_variance = ((parametric_mean - a) * (c - parametric_mean)) / 7
            # parametric_skewness = 2 * (self.alpha2 - self.alpha1) * numpy.sqrt(self.alpha2 + self.alpha1 + 1) / ((self.alpha2 + self.alpha1 + 2) *  numpy.sqrt(self.alpha2 * self.alpha1))
            # parametric_kurtosis = 3 + 6 * ((self.alpha2 - self.alpha1) ** 2 * (self.alpha2 + self.alpha1 + 1) - (self.alpha2 * self.alpha1) * (self.alpha2 + self.alpha1 + 2)) / ((self.alpha2 * self.alpha1) * (self.alpha2 + self.alpha1 + 2) * (self.alpha2 + self.alpha1 + 3))
            # parametric_median = (a + 6 * b + c) / 8
            parametric_median = scipy.special.betaincinv(self.alpha1, self.alpha2, 0.5) * (c - a) + a
            parametric_mode = b

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            # eq3 = parametric_skewness - continuous_measures.skewness
            # eq4 = parametric_kurtosis  - continuous_measures.kurtosis
            eq3 = parametric_median - continuous_measures.median
            # eq2 = parametric_mode - continuous_measures.mode

            return (eq1, eq2, eq3)

        ## Parameters of equations system
        bounds = ((-numpy.inf, continuous_measures.min, continuous_measures.mode), (continuous_measures.mode, continuous_measures.max, numpy.inf))
        x0 = (continuous_measures.min, continuous_measures.mode, continuous_measures.max)
        args = [continuous_measures]

        ## Solve Equation system
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"a": solution.x[0], "b": solution.x[1], "c": solution.x[2]}

        ## Correction of parameters
        parameters["a"] = min(continuous_measures.min - 1e-3, parameters["a"])
        parameters["c"] = max(continuous_measures.max + 1e-3, parameters["c"])

        # parameters = {"a": continuous_measures.min - 1e-3, "b": continuous_measures.mode, "c": continuous_measures.max + 1e-3}
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
    path = "../continuous_distributions_sample/sample_pert.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = PERT(continuous_measures=continuous_measures)

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
