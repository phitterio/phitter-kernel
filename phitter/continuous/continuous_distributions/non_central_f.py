import numpy
import scipy.integrate
import scipy.optimize
import scipy.special
import scipy.stats


class NonCentralF:
    """
    Non-Central F distribution
    - Parameters NonCentralF Distribution: {"lambda": *, "n1": *, "n2": *}
    - https://phitter.io/distributions/continuous/non_central_f
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the NonCentralF Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters NonCentralF Distribution: {"lambda": *, "n1": *, "n2": *}
        - https://phitter.io/distributions/continuous/non_central_f
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

        self.lambda_ = self.parameters["lambda"]
        self.n1 = self.parameters["n1"]
        self.n2 = self.parameters["n2"]

    @property
    def name(self):
        return "non_central_f"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"lambda": 81, "n1": 12, "n2": 72}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        ## Method 1
        result = scipy.stats.ncf.cdf(x, self.n1, self.n2, self.lambda_)

        ## Method 2
        # result = scipy.special.ncfdtr(self.n1, self.n2, self.lambda_, x)

        ## Method 3
        # k = 0
        # acum = 0
        # r0 = -1
        # while(acum - r0 > 1e-10):
        #     r0 = acum
        #     t1 = ((self.lambda_ / 2) ** k) / scipy.special.factorial(k)
        #     q = x * self.n1 / (self.n2 + x * self.n1)
        #     t2 = scipy.special.betainc(k + self.n1 / 2, self.n2 / 2, q)
        #     s = t1 * t2
        #     acum += s
        #     k += 1
        # result = numpy.exp(-self.lambda_ / 2) * acum
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        ## Method 1
        result = scipy.stats.ncf.pdf(x, self.n1, self.n2, self.lambda_)

        ## Method 2
        # k = 0
        # acum = 0
        # r0 = -1
        # while(acum - r0 > 1e-10):
        #     r0 = acum
        #     t1 = 1 / scipy.special.factorial(k)
        #     t2 = (self.lambda_ / 2) ** k
        #     t3 = 1 / scipy.special.beta(k + self.n1 / 2, self.n2 / 2)
        #     t4 = (self.n1 / self.n2) ** (k + self.n1 / 2)
        #     t5 = (self.n2 / (self.n2 + self.n1 * x)) ** (k + (self.n1 + self.n2) / 2)
        #     t6 = x ** (k - 1 + self.n1 / 2)
        #     s = t1 * t2 * t3 * t4 * t5 * t6
        #     acum += s
        #     k += 1
        # result = numpy.exp(-self.lambda_ / 2) * acum
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.ncf.ppf(u, self.n1, self.n2, self.lambda_)
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
            return (self.n2 / self.n1) * ((self.n1 + self.lambda_) / (self.n2 - 2))
        if k == 2:
            return (self.n2 / self.n1) ** 2 * (1 / ((self.n2 - 2) * (self.n2 - 4))) * (self.lambda_**2 + (2 * self.lambda_ + self.n1) * (self.n1 + 2))
        if k == 3:
            return (
                (self.n2 / self.n1) ** 3
                * (1 / ((self.n2 - 2) * (self.n2 - 4) * (self.n2 - 6)))
                * (self.lambda_**3 + 3 * (self.n1 + 4) * self.lambda_**2 + (3 * self.lambda_ + self.n1) * (self.n1 + 4) * (self.n1 + 2))
            )
        if k == 4:
            return (
                (self.n2 / self.n1) ** 4
                * (1 / ((self.n2 - 2) * (self.n2 - 4) * (self.n2 - 6) * (self.n2 - 8)))
                * (
                    self.lambda_**4
                    + 4 * (self.n1 + 6) * self.lambda_**3
                    + 6 * (self.n1 + 6) * (self.n1 + 4) * self.lambda_**2
                    + (4 * self.lambda_ + self.n1) * (self.n1 + 2) * (self.n1 + 4) * (self.n1 + 6)
                )
            )
        return None

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
        v1 = self.lambda_ > 0
        v2 = self.n1 > 0
        v3 = self.n2 > 0
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
        parameters: {"lambda": *, "n1": *, "n2": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            lambda_, n1, n2 = initial_solution

            ## Generatred moments function (not - centered)
            E_1 = (n2 / n1) * ((n1 + lambda_) / (n2 - 2))
            E_2 = (n2 / n1) ** 2 * ((lambda_**2 + (2 * lambda_ + n1) * (n1 + 2)) / ((n2 - 2) * (n2 - 4)))
            E_3 = (n2 / n1) ** 3 * ((lambda_**3 + 3 * (n1 + 4) * lambda_**2 + (3 * lambda_ + n1) * (n1 + 2) * (n1 + 4)) / ((n2 - 2) * (n2 - 4) * (n2 - 6)))
            # E_4 = (n2 / n1) ** 4 * ((lambda_ ** 4 + 4 * (n1 + 6) * lambda_ ** 3 + 6 * (n1 + 4) * (n1 + 6) * lambda_ ** 2 + (4 * lambda_ + n1) * (n1 + 2) * (n1 + 4) * (n1 + 4)) /  ((n2 - 2) * (n2-4) * (n2 - 6) * (n2 - 6)))

            ## Parametric expected expressions
            parametric_mean = E_1
            parametric_variance = E_2 - E_1**2
            parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1**3) / ((E_2 - E_1**2)) ** 1.5
            # parametric_kurtosis = (E_4-4 * E_1 * E_3 + 6 * E_1 ** 2 * E_2 - 3 * E_1 ** 4) /  ((E_2 - E_1 ** 2)) ** 2

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            # eq4 = parametric_kurtosis  - continuous_measures.kurtosis

            return (eq1, eq2, eq3)

        bounds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (continuous_measures.mean, continuous_measures.mean, continuous_measures.mean)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"lambda": solution.x[0], "n1": solution.x[1], "n2": solution.x[2]}

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
    path = "../../../distributions_samples/continuous_distributions_sample/sample_non_central_f.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = NonCentralF(continuous_measures=continuous_measures)

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
