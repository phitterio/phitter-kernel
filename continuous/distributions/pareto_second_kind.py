import scipy.stats
import math
import scipy.optimize
import numpy


class PARETO_SECOND_KIND:
    """
    Pareto second kind distribution distribution
    Also known as Lomax Distribution or Pareto Type II distributions
    https://en.wikipedia.org/wiki/Lomax_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.xm = self.parameters["xm"]
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # result = 1 - (self.xm / ((x - self.loc) + self.xm)) ** self.alpha
        result = scipy.stats.lomax.cdf(x, self.alpha, scale=self.xm, loc = self.loc)
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # print(scipy.stats.lomax.pdf(x, self.alpha, scale=self.xm, loc = self.loc))
        return (self.alpha * self.xm**self.alpha) / (((x - self.loc) + self.xm) ** (self.alpha + 1))

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restriction
        """
        v1 = self.xm > 0
        v2 = self.alpha > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by formula.

        Parameters
        ==========
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"xm": * , "alpha": * }
        """

        # def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
        #     ## Variables declaration
        #     alpha, xm, loc = initial_solution

        #     ## Generatred moments function (not - centered)

        #     E = lambda k: (math.gamma(1 + k) * math.gamma(alpha - k) * xm**k) / math.gamma(alpha)

        #     ## Parametric expected expressions
        #     parametric_mean = loc + E(1)
        #     parametric_variance = E(2) - E(1) ** 2
        #     # parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
        #     # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
        #     parametric_median = loc + xm * (2 ** (1 / alpha) - 1)
        #     # parametric_mode = loc

        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     eq2 = parametric_variance - measurements.variance
        #     # eq3 = parametric_skewness - measurements.skewness
        #     # eq3 = parametric_kurtosis  - measurements.kurtosis
        #     # eq2 = parametric_mode - measurements.mode
        #     eq3 = parametric_median - measurements.median

        #     return (eq1, eq2, eq3)

        # bnds = ((1, 0,  - numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        # x0 = (7, 6, measurements.mode)
        # args = ([measurements])
        # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        # parameters = {"alpha": solution.x[0], "xm": solution.x[1], "loc": solution.x[2]}

        m = measurements.mean
        # me = measurements.median
        # mo = measurements.mode
        v = measurements.variance

        loc = scipy.stats.lomax.fit(measurements.data)[1]
        xm = -((m - loc) * ((m - loc) ** 2 + v)) / ((m - loc) ** 2 - v)
        alpha = -(2 * v) / ((m - loc) ** 2 - v)
        parameters = {"xm": xm, "alpha": alpha, "loc": loc}

        # scipy_params = scipy.stats.lomax.fit(measurements.data)
        # parameters = {"xm": scipy_params[2] , "alpha": scipy_params[0], "loc": scipy_params[1]}

        return parameters


if __name__ == "__main__":
    ## Import function to get measurements
    import sys

    sys.path.append("../measurements")
    from measurements_continuous import MEASUREMENTS_CONTINUOUS

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_pareto_second_kind.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = PARETO_SECOND_KIND(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))

    ## Get parameters of distribution: SCIPY vs EQUATIONS
    import time

    ti = time.time()
    print(distribution.get_parameters(measurements))
    print("Solve equations time: ", time.time() - ti)
    ti = time.time()
    print(scipy.stats.lomax.fit(data))
    print("Scipy time get parameters: ", time.time() - ti)
