import numpy
import scipy.optimize
import scipy.stats


class Cauchy:
    """
    Cauchy distribution
    - Parameters Cauchy Distribution: {"x0": *, "gamma": *}
    - https://phitter.io/distributions/continuous/cauchy
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Cauchy Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters Cauchy Distribution: {"x0": *, "gamma": *}
        - https://phitter.io/distributions/continuous/cauchy
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

        self.x0 = self.parameters["x0"]
        self.gamma = self.parameters["gamma"]

    @property
    def name(self):
        return "cauchy"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"x0": 10, "gamma": 19}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        return (1 / numpy.pi) * numpy.arctan(((x - self.x0) / self.gamma)) + (1 / 2)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        return 1 / (numpy.pi * self.gamma * (1 + ((x - self.x0) / self.gamma) ** 2))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.x0 + self.gamma * numpy.tan(numpy.pi * (u - 0.5))
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
        return None

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return None

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
        return None

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return None

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
        return self.x0

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
        v1 = self.gamma > 0
        return v1

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
        parameters: {"x0": *, "gamma": *}
        """
        # ## First estimation
        # x0_ini = continuous_measures.median
        # q1 = scipy.stats.scoreatpercentile(continuous_measures.data, 25)
        # q3 = scipy.stats.scoreatpercentile(continuous_measures.data, 75)
        # gamma_ini = (q3 - q1) / 2

        # ## Maximum Likelihood Estimation Cauchy distribution
        # def objective(x):
        #     x0, gamma = x
        #     return - sum([numpy.log(1 / (numpy.pi * gamma * (1 + ((d - x0) / gamma) ** 2))) for d in continuous_measures.data])
        # solution = scipy.optimize.minimize(objective, [x0_ini, gamma_ini], method="SLSQP", bounds = [(-numpy.inf, numpy.inf),(0,numpy.inf)])
        # print(solution)

        scipy_parameters = scipy.stats.cauchy.fit(continuous_measures.data_to_fit)

        ## Results
        parameters = {"x0": scipy_parameters[0], "gamma": scipy_parameters[1]}

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
    path = "../continuous_distributions_sample/sample_cauchy.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = Cauchy(continuous_measures=continuous_measures)

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

    # import time
    # ti = time.time()
    # print(distribution.get_parameters(continuous_measures=continuous_measures))
    # print("Equations: ", time.time()  - ti)

    # ti = time.time()
    # print(scipy.stats.cauchy.fit(data))
    # print("Scipy: ",time.time()  - ti)
