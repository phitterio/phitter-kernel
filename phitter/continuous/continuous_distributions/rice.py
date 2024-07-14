import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class RICE:
    """
    Rice distribution
    Parameters RICE distribution: {"v": *, "sigma": *}
    https://phitter.io/distributions/continuous/rice
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        Initializes the RICE distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters RICE distribution: {"v": *, "sigma": *}
        https://phitter.io/distributions/continuous/rice
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

        self.v = self.parameters["v"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "rice"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"v": 4, "sigma": 5}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """

        # def Q(M: float, a: float, b: float) -> float:
        #     """
        #     Marcum Q - function
        #     https://en.wikipedia.org/wiki/Marcum_Q - function
        #     """
        #     k = 1 - M
        #     x = (a / b) ** k * scipy.special.iv(k, a * b)
        #     acum = 0
        #     while x > 1e-20:
        #         acum += x
        #         k += 1
        #         x = (a / b) ** k * scipy.special.iv(k, a * b)
        #     res = acum * numpy.exp(-(a**2 + b**2) / 2)
        #     return res

        # result, error = scipy.integrate.quad(self.pdf, 0, x)
        # result = 1 - Q(1, self.v / self.sigma, x / self.sigma)
        result = scipy.stats.rice.cdf(x, self.v / self.sigma, scale=self.sigma)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # result = (x / (self.sigma**2)) * numpy.exp(-(x**2 + self.v**2) / (2 * self.sigma**2)) * scipy.special.i0(x * self.v / (self.sigma**2))
        result = scipy.stats.rice.pdf(x, self.v / self.sigma, scale=self.sigma)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.rice.ppf(u, self.v / self.sigma, scale=self.sigma)
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
            return (
                self.sigma
                * numpy.sqrt(numpy.pi / 2)
                * numpy.exp((-self.v * self.v) / (2 * self.sigma * self.sigma) / 2)
                * (
                    (1 - (-self.v * self.v) / (2 * self.sigma * self.sigma)) * scipy.special.iv(0, (-self.v * self.v) / (4 * self.sigma * self.sigma))
                    + ((-self.v * self.v) / (2 * self.sigma * self.sigma)) * scipy.special.iv(1, (-self.v * self.v) / (4 * self.sigma * self.sigma))
                )
            )
        if k == 2:
            return 2 * self.sigma * self.sigma + self.v * self.v

        if k == 3:
            return (
                3
                * self.sigma**3
                * numpy.sqrt(numpy.pi / 2)
                * numpy.exp((-self.v * self.v) / (2 * self.sigma * self.sigma) / 2)
                * (
                    (2 * ((-self.v * self.v) / (2 * self.sigma * self.sigma)) ** 2 - 6 * ((-self.v * self.v) / (2 * self.sigma * self.sigma)) + 3)
                    * scipy.special.iv(0, (-self.v * self.v) / (4 * self.sigma * self.sigma))
                    - 2
                    * ((-self.v * self.v) / (2 * self.sigma * self.sigma) - 2)
                    * ((-self.v * self.v) / (2 * self.sigma * self.sigma))
                    * scipy.special.iv(1, (-self.v * self.v) / (4 * self.sigma * self.sigma))
                )
            ) / 3

        if k == 4:
            return 8 * self.sigma**4 + 8 * self.sigma * self.sigma * self.v * self.v + self.v**4

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
        v1 = self.v > 0
        v2 = self.sigma > 0
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
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"v": *, "sigma": *}
        """

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            v, sigma = initial_solution

            E = lambda k: sigma**k * 2 ** (k / 2) * scipy.special.gamma(1 + k / 2) * scipy.special.eval_laguerre(k / 2, -v * v / (2 * sigma * sigma))

            ## Parametric expected expressions
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            # parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            # eq3 = parametric_skewness - continuous_measures.skewness
            # eq4 = parametric_kurtosis  - continuous_measures.kurtosis

            return (eq1, eq2)

        bounds = ((0, 0), (numpy.inf, numpy.inf))
        x0 = (continuous_measures.mean, numpy.sqrt(continuous_measures.variance))
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"v": solution.x[0], "sigma": solution.x[1]}

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
    path = "../continuous_distributions_sample/sample_rice.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = RICE(continuous_measures=continuous_measures)

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
