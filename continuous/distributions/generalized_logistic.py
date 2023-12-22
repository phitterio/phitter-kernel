import scipy.stats
import scipy.special as sc
import math
import scipy.optimize
import numpy


class GENERALIZED_LOGISTIC:
    """
    Generalized Logistic Distribution
    Compendium of Common Probability Distributions (pag.41) ... Michael P. McLaughlin
    https://docs.scipy.org / doc / scipy / tutorial / stats / continuous_genlogistic.html
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)

        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        self.c = self.parameters["c"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # return scipy.stats.genlogistic.cdf(x, self.c, loc=self.loc, scale=self.scale)
        z = lambda t: (t - self.loc) / self.scale
        return 1 / ((1 + math.exp(-z(x))) ** self.c)

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # return scipy.stats.genlogistic.pdf(x, self.c, loc=self.loc, scale=self.scale)
        z = lambda t: (t - self.loc) / self.scale
        return (self.c / self.scale) * math.exp(-z(x)) * ((1 + math.exp(-z(x))) ** (-self.c - 1))

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.scale > 0
        v2 = self.c > 0
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
            {"loc": * , "scale": * , "c": * }
        """

        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            ## Variables declaration
            c, loc, scale = initial_solution

            ## Parametric expected expressions
            parametric_mean = loc + scale * (0.57721 + sc.digamma(c))
            parametric_variance = scale**2 * (math.pi**2 / 6 + sc.polygamma(1, c))
            # parametric_skewness = (sc.polygamma(2,1) + sc.polygamma(2,c)) / ((math.pi ** 2 / 6 + sc.polygamma(1, c)) ** 1.5)
            # parametric_kurtosis = 3 + (math.pi ** 4 / 15 + sc.polygamma(3,c)) / ((math.pi ** 2 / 6 + sc.polygamma(1, c)) ** 2)
            parametric_median = loc + scale * (-math.log(0.5 ** (-1 / c) - 1))
            # parametric_mode = loc + scale * math.log(c)

            ## System Equations
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            # eq3 = parametric_skewness - measurements.skewness
            # eq3 = parametric_kurtosis - measurements.kurtosis
            eq3 = parametric_median - measurements.median
            # eq3 = parametric_mode - measurements.mode

            return (eq1, eq2, eq3)

        # ## scipy.optimize.fsolve methods
        # solution = scipy.optimize.fsolve(equations, (1, 1, 1), measurements)
        # parameters = {"loc": solution[0], "scale": solution[1], "c": solution[2]}
        # print(parameters)

        ## least square methods
        x0 = [measurements.mean, measurements.mean, measurements.mean]
        b = ((1e-5, -numpy.inf, 1e-5), (numpy.inf, numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([measurements]))
        parameters = {"c": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}

        # ## scipy methods
        # scipy_params = scipy.stats.genlogistic.fit(measurements.data)
        # parameters = {"loc": scipy_params[1], "scale": scipy_params[2], "c": scipy_params[0]}

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
    path = "../data/data_generalized_logistic.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = GENERALIZED_LOGISTIC(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
