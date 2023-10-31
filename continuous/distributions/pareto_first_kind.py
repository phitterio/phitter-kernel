import scipy.stats
import scipy.optimize
import numpy


class PARETO_FIRST_KIND:
    """
    Pareto first kind distribution distribution
    https://en.wikipedia.org/wiki/Pareto_distribution
    Compendium of Common Probability Distributions (pag.61) ... Michael P. McLaughlin
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
        # result = 1 - (self.xm / (x - self.loc)) ** self.alpha
        result = scipy.stats.pareto.cdf(x, self.alpha, loc=self.loc, scale=self.xm)
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = (self.alpha * self.xm**self.alpha) / ((x - self.loc) ** (self.alpha + 1))
        result = scipy.stats.pareto.pdf(x, self.alpha, loc=self.loc, scale=self.xm)
        return result

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
            {"xm":  * , "alpha":  * }
        """

        # def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
        #     ## Variables declaration
        #     alpha, xm, loc = initial_solution

        #     ## Generatred moments function (not - centered)
        #     E = lambda k: (alpha * xm**k) / (alpha - k)

        #     ## Parametric expected expressions
        #     parametric_mean = loc + E(1)
        #     parametric_variance = E(2) - E(1) ** 2
        #     # parametric_skewne1ss = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
        #     # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
        #     # parametric_median = loc + xm * (2 ** (1 / alpha))
        #     parametric_mode = loc + xm

        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     eq2 = parametric_variance - measurements.variance
        #     # eq3 = parametric_skewness - measurements.skewness
        #     # eq3 = parametric_kurtosis  - measurements.kurtosis
        #     eq3 = parametric_mode - measurements.mode
        #     # eq3 = parametric_median - measurements.median

        #     return (eq1, eq2, eq3)

        # bnds = ((1, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        # x0 = (1, measurements.mean, measurements.mean)
        # args = [measurements]
        # solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        # parameters = {"alpha": solution.x[0], "xm": solution.x[1], "loc": solution.x[2]}

        scipy_params = scipy.stats.pareto.fit(measurements.data)
        parameters = {"xm": scipy_params[2], "alpha": scipy_params[0], "loc": scipy_params[1]}

        # # Solve system
        # m = 112.5
        # me = measurements.median
        # mo = 110
        # v = 10.41

        # loc = (m ** 3 - 2 * m ** 2 * mo + m * (mo ** 2 + v) - 2 * mo * v) / (m ** 2 - 2 * m * mo + mo ** 2 - v)
        # xm = -((m - mo) * (m ** 2 - 2 * m * mo + mo ** 2 + v)) / (m ** 2 - 2 * m * mo + mo ** 2 - v)
        # alpha = -(2 * v) / (m ** 2 - 2 * m * mo + mo ** 2 - v)

        # parameters = {"xm": xm, "alpha": alpha, "loc": loc}
        # # xm = (m ** 2 + v - math.sqrt(v * (m ** 2 + v))) / m
        # # alpha = (v + math.sqrt(v * (m ** 2 + v))) / v
        # # parameters = {"xm": xm , "alpha": alpha}

        return parameters


if __name__ == "__main__":
    # Import function to get measurements
    import sys

    sys.path.append("../measurements")
    from measurements_continuous import MEASUREMENTS_CONTINUOUS

    # Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    # Distribution class
    path = "../data/data_pareto_first_kind.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = PARETO_FIRST_KIND(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))

    # Get parameters of distribution: SCIPY vs EQUATIONS
    import time

    ti = time.time()
    print(distribution.get_parameters(measurements))
    print("Solve equations time: ", time.time() - ti)
    ti = time.time()
    print(scipy.stats.pareto.fit(data))
    print("Scipy time get parameters: ", time.time() - ti)
