import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class InverseGamma:
    """
    Inverse Gamma Distribution
    Also known Pearson Type 5 distribution
    - Parameters InverseGamma Distribution: {"alpha": *, "beta": *}
    - https://phitter.io/distributions/continuous/inverse_gamma
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the InverseGamma Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters InverseGamma Distribution: {"alpha": *, "beta": *}
        - https://phitter.io/distributions/continuous/inverse_gamma
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
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "inverse_gamma"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 4, "beta": 12}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        result = scipy.stats.invgamma.cdf(x, a=self.alpha, scale=self.beta)
        # upper_inc_gamma = lambda a, x: scipy.special.gammaincc(a, x) * scipy.special.gamma(a)
        # result = upper_inc_gamma(self.alpha, self.beta / x) / scipy.special.gamma(self.alpha)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        result = scipy.stats.invgamma.pdf(x, a=self.alpha, scale=self.beta)
        # result = ((self.beta**self.alpha) * (x ** (-self.alpha - 1)) * numpy.exp(-(self.beta / x))) / scipy.special.gamma(self.alpha)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.beta / scipy.special.gammaincinv(self.alpha, 1 - u)
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
            return self.beta**k / (self.alpha - 1)
        if k == 2:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2))
        if k == 3:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2) * (self.alpha - 3))
        if k == 4:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2) * (self.alpha - 3) * (self.alpha - 4))
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
        return self.beta / (self.alpha + 1)

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
        parameters: {"alpha": *, "beta": *}
        """

        # def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
        #     ## Variables declaration
        #     alpha, beta = initial_solution

        #     ## Generatred moments function (not - centered)
        #     E = lambda k: (beta**k) / numpy.prod(numpy.array(alpha - numpy.arange(1, k + 1)))

        #     ## Parametric expected expressions
        #     # parametric_mean = E(1)
        #     # parametric_variance = E(2) - E(1) ** 2
        #     # parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
        #     # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
        #     parametric_median = beta / scipy.special.gammaincinv(alpha, 0.5)
        #     parametric_mode = beta / (alpha + 1)

        #     ## System Equations
        #     # eq1 = parametric_mean - continuous_measures.mean
        #     # eq2 = parametric_variance - continuous_measures.variance
        #     # eq3 = parametric_skewness - continuous_measures.skewness
        #     # eq4 = parametric_kurtosis  - continuous_measures.kurtosis
        #     eq1 = parametric_median - continuous_measures.median
        #     eq2 = parametric_mode - continuous_measures.mode

        #     return (eq1, eq2)

        # x0 = (1, 1)
        # bounds = ((0, 0), (numpy.inf, numpy.inf))
        # args = [continuous_measures]
        # solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        # parameters = {"alpha": solution.x[0], "beta": solution.x[1]}

        # solution = scipy.optimize.fsolve(equations, (1, 1), continuous_measures)

        # alpha = (continuous_measures.mode + continuous_measures.mean) / (continuous_measures.mean - continuous_measures.mode)
        # beta = continuous_measures.mean * (alpha - 1)
        # parameters = {"alpha": alpha, "beta": beta}

        scipy_parameters = scipy.stats.invgamma.fit(continuous_measures.data_to_fit)
        parameters = {"alpha": scipy_parameters[0], "beta": scipy_parameters[2]}
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
    path = "../../../distributions_samples/continuous_distributions_sample/sample_inverse_gamma.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = InverseGamma(continuous_measures=continuous_measures)

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
