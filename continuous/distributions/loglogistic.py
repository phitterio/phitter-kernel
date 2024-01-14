import scipy.stats
import numpy
import scipy.optimize
import numpy


class LOGLOGISTIC:
    """
    Loglogistic distribution
    https://en.wikipedia.org/wiki/Log - logistic_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        result = x**self.beta / (self.alpha**self.beta + x**self.beta)
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        result = self.beta / self.alpha * (x / self.alpha) ** (self.beta - 1) / ((1 + (x / self.alpha) ** self.beta) ** 2)
        return result

    def get_num_parameters(self) -> int:
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
            {"alpha": * , "beta": * }
        """
        # def equations(initial_solution, data_mean, data_variance, data_skewness):
        #     alpha, beta = initial_solution

        #     E = lambda r: (alpha ** r) * (r * numpy.pi / beta) / numpy.sin(r * numpy.pi / beta)

        #     parametric_mean = E(1)
        #     parametric_variance = (E(2) - E(1) ** 2)
        #     parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
        #     parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2

        #     ## System Equations
        #     eq1 = parametric_mean - data_mean
        #     eq2 = parametric_variance - data_variance

        #     return (eq1, eq2)

        # bnds = ((0, 0), (numpy.inf, numpy.inf))
        # x0 = (measurements.mean, 1 / measurements.variance ** 0.5)
        # args = (measurements.mean, measurements.variance, measurements.skewness)
        # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        # parameters = {"alpha": solution.x[0], "beta": solution.x[1]}

        scipy_params = scipy.stats.fisk.fit(measurements.data)
        parameters = {"alpha": scipy_params[2], "beta": scipy_params[0]}

        return parameters


if __name__ == "__main__":
    ## Import function to get measurements
    import sys
    import numpy

    sys.path.append("../measurements")
    from measurements_continuous import MEASUREMENTS_CONTINUOUS

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_loglogistic.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = LOGLOGISTIC(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.cdf(numpy.array([measurements.mean, measurements.mean])))
    print(distribution.pdf(measurements.mean))
    print(distribution.pdf(numpy.array([measurements.mean, measurements.mean])))

    print("========= Time parameter estimation analisys ========")

    import time

    def equations(initial_solution, data_mean, data_variance, data_skewness):
        alpha, beta = initial_solution

        E = lambda r: (alpha**r) * (r * numpy.pi / beta) / numpy.sin(r * numpy.pi / beta)

        parametric_mean = E(1)
        parametric_variance = E(2) - E(1) ** 2
        # parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
        # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2

        ## System Equations
        eq1 = parametric_mean - data_mean
        eq2 = parametric_variance - data_variance

        return (eq1, eq2)

    ti = time.time()
    bnds = ((0, 0), (numpy.inf, numpy.inf))
    x0 = (measurements.mean, measurements.variance)
    args = (measurements.mean, measurements.variance, measurements.skewness)
    solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
    parameters = {"alpha": solution.x[0], "beta": solution.x[1]}
    print(parameters)
    print("Solve equations time: ", time.time() - ti)

    ti = time.time()
    scipy_params = scipy.stats.fisk.fit(data)
    parameters = {"alpha": scipy_params[2], "beta": scipy_params[0]}
    print(parameters)
    print("Scipy time get parameters: ", time.time() - ti)
