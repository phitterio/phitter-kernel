import numpy
import scipy.integrate
import scipy.special
import scipy.stats


class NON_CENTRAL_T_STUDENT:
    """
    Non-Central T Student distribution
    Parameters NON_CENTRAL_T_STUDENT distribution: {"lambda": *, "n": *, "loc": *, "scale": *}
    https://phitter.io/distributions/continuous/non_central_t_student    Hand-book on Statistical Distributions (pag.116) ... Christian Walck
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the NON_CENTRAL_T_STUDENT distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters NON_CENTRAL_T_STUDENT distribution: {"lambda": *, "n": *, "loc": *, "scale": *}
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.lambda_ = self.parameters["lambda"]
        self.n = self.parameters["n"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "non_central_t_student"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"lambda": 8, "n": 9, "loc": 474, "scale": 6}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        ##  Method 1
        # z = lambda x: (x - self.loc) / self.scale
        # result = scipy.special.nctdtr(self.n, self.lambda_, z(x))

        ## Method 2
        result = scipy.stats.nct.cdf(x, self.n, self.lambda_, loc=self.loc, scale=self.scale)

        ## Method 3
        # k = 0
        # acum = 0
        # r0 = -1
        # while(acum - r0 > 1e-20):
        #     r0 = acum
        #     t1 = numpy.exp(-self.lambda_ ** 2 / 2) * (self.lambda_ ** 2 / 2) ** k / scipy.special.factorial(k)
        #     t2 = numpy.exp(-self.lambda_ ** 2 / 2) * (self.lambda_ ** 2 / 2) ** k * self.lambda_ / (numpy.sqrt(2) * scipy.special.gamma(k + 1.5))
        #     y = (z(x) ** 2) / (z(x) ** 2 + self.n)
        #     s = t1 * scipy.special.betainc(k + 0.5, self.n / 2, y) + t2 * scipy.special.betainc(k + 1, self.n / 2, y)
        #     acum += s
        #     k += 1
        # result = scipy.stats.norm.cdf(-self.lambda_) + 0.5 * acum

        ## Method 4
        # result, err = scipy.integrate.quad(self.pdf,  - numpy.inf, x)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        ## Method 1
        result = scipy.stats.nct.pdf(x, self.n, self.lambda_, loc=self.loc, scale=self.scale)

        ## Method 2
        # t1 = self.n ** (self.n / 2) * scipy.special.gamma(self.n + 1)
        # t2 = 2 ** self.n * numpy.exp(self.lambda_ ** 2 / 2) * (self.n + z(x) ** 2) ** (self.n / 2) * scipy.special.gamma(self.n / 2)
        # t3 = numpy.sqrt(2) * self.lambda_ * z(x) * scipy.special.hyp1f1(1 + self.n / 2, 1.5, (self.lambda_ ** 2 * z(x) ** 2) / (2 * (self.n + z(x) ** 2)))
        # t4 = (self.n + z(x) ** 2) * scipy.special.gamma(0.5 * (self.n + 1))
        # t5 = scipy.special.hyp1f1((1 + self.n) / 2, 0.5, (self.lambda_ ** 2 * z(x) ** 2) / (2 * (self.n + z(x) ** 2)))
        # t6 = numpy.sqrt(self.n + z(x) ** 2) * scipy.special.gamma(1 + self.n / 2)
        # result = (t1 / t2 * (t3 / t4 + t5 / t6)) / self.scale

        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.nct.ppf(u, self.n, self.lambda_, loc=self.loc, scale=self.scale)
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
            return (self.lambda_ * numpy.sqrt(self.n / 2) * scipy.special.gamma((self.n - 1) / 2)) / scipy.special.gamma(self.n / 2)
        if k == 2:
            return (self.n * (1 + self.lambda_ * self.lambda_)) / (self.n - 2)
        if k == 3:
            return (self.n**1.5 * numpy.sqrt(2) * scipy.special.gamma((self.n - 3) / 2) * self.lambda_ * (3 + self.lambda_ * self.lambda_)) / (4 * scipy.special.gamma(self.n / 2))
        if k == 4:
            return (self.n * self.n * (self.lambda_**4 + 6 * self.lambda_**2 + 3)) / ((self.n - 2) * (self.n - 4))
        return None

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
        return numpy.sqrt(self.variance())

    @property
    def skewness(self) -> float:
        """
        Parametric skewness
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ3 = self.central_moments(3)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ4 = self.central_moments(4)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

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
        return None

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
        v1 = self.n > 0
        v2 = self.scale > 0
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
        parameters: {"lambda": *, "n": *, "loc": *, "scale": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            lambda_, n, loc, scale = initial_solution

            ## Generatred moments function (not - centered)
            E_1 = lambda_ * numpy.sqrt(n / 2) * scipy.special.gamma((n - 1) / 2) / scipy.special.gamma(n / 2)
            E_2 = (1 + lambda_**2) * n / (n - 2)
            E_3 = lambda_ * (3 + lambda_**2) * n**1.5 * numpy.sqrt(2) * scipy.special.gamma((n - 3) / 2) / (4 * scipy.special.gamma(n / 2))
            E_4 = (lambda_**4 + 6 * lambda_**2 + 3) * n**2 / ((n - 2) * (n - 4))

            ## Parametric expected expressions
            parametric_mean = E_1 * scale + loc
            parametric_variance = (E_2 - E_1**2) * (scale**2)
            parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1**3) / ((E_2 - E_1**2)) ** 1.5
            parametric_kurtosis = (E_4 - 4 * E_1 * E_3 + 6 * E_1**2 * E_2 - 3 * E_1**4) / ((E_2 - E_1**2)) ** 2

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis

            return (eq1, eq2, eq3, eq4)

        bnds = ((0, 0, 0, 0), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 5, continuous_measures.mean, 1)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"lambda": solution.x[0], "n": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
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

    path = "../continuous_distributions_sample/sample_NON_CENTRAL_T_STUDENT.txt"

    ## Distribution class
    path = "../continuous_distributions_sample/sample_NON_CENTRAL_T_STUDENT.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = NON_CENTRAL_T_STUDENT(continuous_measures)

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

    # print(scipy.stats.nct.fit(data))
