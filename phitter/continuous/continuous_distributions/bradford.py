import numpy
import scipy.optimize


class Bradford:
    """
    Bradford distribution
    Parameters Bradford Distribution: {"c": *, "min": *, "max": *}
    https://phitter.io/distributions/continuous/bradford
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        Initializes the Bradford Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        Parameters Bradford Distribution: {"c": *, "min": *, "max": *}
        https://phitter.io/distributions/continuous/bradford
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

        self.c = self.parameters["c"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "bradford"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"c": 4, "min": 19, "max": 50}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        result = numpy.log(1 + self.c * (x - self.min) / (self.max - self.min)) / numpy.log(self.c + 1)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        result = self.c / ((self.c * (x - self.min) + self.max - self.min) * numpy.log(self.c + 1))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.min + ((numpy.exp(u * numpy.log(1 + self.c)) - 1) * (self.max - self.min)) / self.c
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
        return (self.c * (self.max - self.min) + numpy.log(1 + self.c) * (self.min * (self.c + 1) - self.max)) / (numpy.log(1 + self.c) * self.c)

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return ((self.max - self.min) ** 2 * ((self.c + 2) * numpy.log(1 + self.c) - 2 * self.c)) / (2 * self.c * numpy.log(1 + self.c) ** 2)

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
        return (numpy.sqrt(2) * (12 * self.c * self.c - 9 * numpy.log(1 + self.c) * self.c * (self.c + 2) + 2 * numpy.log(1 + self.c) * numpy.log(1 + self.c) * (self.c * (self.c + 3) + 3))) / (
            numpy.sqrt(self.c * (self.c * (numpy.log(1 + self.c) - 2) + 2 * numpy.log(1 + self.c))) * (3 * self.c * (numpy.log(1 + self.c) - 2) + 6 * numpy.log(1 + self.c))
        )

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return (
            self.c**3 * (numpy.log(1 + self.c) - 3) * (numpy.log(1 + self.c) * (3 * numpy.log(1 + self.c) - 16) + 24)
            + 12 * numpy.log(1 + self.c) * self.c * self.c * (numpy.log(1 + self.c) - 4) * (numpy.log(1 + self.c) - 3)
            + 6 * self.c * numpy.log(1 + self.c) ** 2 * (3 * numpy.log(1 + self.c) - 14)
            + 12 * numpy.log(1 + self.c) ** 3
        ) / (3 * self.c * (self.c * (numpy.log(1 + self.c) - 2) + 2 * numpy.log(1 + self.c)) ** 2) + 3

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
        return self.min

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
        v1 = self.max > self.min
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
        parameters: {"c": *, "min": *, "max": *}
        """

        def equations(
            initial_solution: tuple[float],
            continuous_measures,
            precalculated_parameters: dict[str, int | float],
        ) -> tuple[float]:
            ## Precalculated parameters
            min_ = precalculated_parameters["min"]
            max_ = precalculated_parameters["max"]

            ## Variables declaration
            c = initial_solution

            ## Parametric expected expressions
            parametric_mean = (c * (max_ - min_) + numpy.log(c + 1) * (min_ * (c + 1) - max_)) / (c * numpy.log(c + 1))

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean

            return eq1

        # solution = scipy.optimize.fsolve(equations, (1), continuous_measures)
        # parameters = {"c": solution[0], "min": min_, "max": max_}

        min_ = continuous_measures.min - 1e-3
        max_ = continuous_measures.max + 1e-3

        bounds = ((-numpy.inf), (numpy.inf))
        x0 = 1
        args = [continuous_measures, {"min": min_, "max": max_}]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"c": solution.x[0], "min": min_, "max": max_}

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

    path = "../continuous_distributions_sample/sample_bradford.txt"

    ## Distribution class
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = Bradford(continuous_measures=continuous_measures)

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
