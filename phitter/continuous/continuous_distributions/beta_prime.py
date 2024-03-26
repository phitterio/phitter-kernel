import warnings

import numpy
import scipy.optimize
import scipy.special
import scipy.stats

warnings.filterwarnings("ignore")


class BETA_PRIME:
    """
    Beta Prime Distribution
    https://phitter.io/distributions/continuous/beta_prime
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        """
        Initializes the BETA_PRIME distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters BETA_PRIME distribution: {"alpha": *, "beta": *}
        """
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")

        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters

        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "beta_prime"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # result = scipy.stats.betaprime.cdf(x, self.alpha, self.beta)
        result = scipy.special.betainc(self.alpha, self.beta, x / (1 + x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # result = scipy.stats.betaprime.pdf(x, self.alpha, self.beta)
        result = (x ** (self.alpha - 1) * (1 + x) ** (-self.alpha - self.beta)) / (scipy.special.beta(self.alpha, self.beta))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.special.betaincinv(self.alpha, self.beta, u) / (1 - scipy.special.betaincinv(self.alpha, self.beta, u))
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
        return (scipy.special.gamma(k + self.alpha) * scipy.special.gamma(self.beta - k)) / (scipy.special.gamma(self.alpha) * scipy.special.gamma(self.beta))

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
        return (self.alpha - 1) / (self.beta + 1)

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
        The parameters are calculated by solving the equations of the measures expected
        for this distribution.The number of equations to consider is equal to the number
        of parameters.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters: {"alpha": *, "beta": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            alpha, beta = initial_solution

            ## Generatred moments function (not - centered)
            # E = lambda k: scipy.special.gamma(k - alpha) * scipy.special.gamma(beta - k) / (scipy.special.gamma(alpha) * scipy.special.gamma(beta))

            ## Parametric expected expressions
            parametric_mean = alpha / (beta - 1)
            parametric_variance = alpha * (alpha + beta - 1) / ((beta - 1) ** 2 * (beta - 2))
            # parametric_skewness = 2 * numpy.sqrt(((beta - 2)) / (alpha * (alpha + beta - 1))) * (((2 * alpha + beta - 1)) / (beta - 3))
            # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
            # parametric_median = scipy.special.betaincinv(0.5, alpha, beta)
            # parametric_mode = (alpha - 1) / (beta + 1)

            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            # eq3 = parametric_skewness - continuous_measures.skewness
            # eq2 = parametric_mode - continuous_measures.mode

            return (eq1, eq2)

        scipy_params = scipy.stats.betaprime.fit(continuous_measures.data)

        try:
            bnds = ((0, 0), (numpy.inf, numpy.inf))
            x0 = (scipy_params[0], scipy_params[1])
            args = [continuous_measures]
            solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1]}
        except:
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1]}

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
    path = "../continuous_distributions_sample/sample_beta_prime.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = BETA_PRIME(continuous_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.get_parameters(continuous_measures)}")
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
