import numpy
import scipy.stats
import scipy.integrate
import scipy.optimize


class ALPHA:
    """
    Alpha distribution
    http://bayanbox.ir/view/5343019340232060584/Norman-L.-Johnson-Samuel-Kotz-N.-Balakrishnan-BookFi.org.pdf
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # print(scipy.stats.alpha.cdf(x, self.alpha, loc=self.loc, scale=self.scale))
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.stats.norm.cdf(self.alpha - (1 / z(x))) / scipy.stats.norm.cdf(self.alpha)
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # z = lambda t: (t - self.loc) / self.scale
        # result = (1 / (self.scale * z(x) * z(x) * scipy.stats.norm.cdf(self.alpha) * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-0.5 * (self.alpha - 1 / z(x)) ** 2)
        result = scipy.stats.alpha.pdf(x, self.alpha, loc=self.loc, scale=self.scale)
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
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by solving the equations of the measures expected
        for this distribution.The number of equations to consider is equal to the number
        of parameters.

        Parameters
        ==========
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"alpha": * , "beta": * , "min": * , "max": * }
        """
        # def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
        #     alpha, loc, scale = initial_solution

        #     z = lambda t: (t - loc) / scale
        #     pdf = lambda x: (1 / (scale * z(x) * z(x) * scipy.stats.norm.cdf(alpha) * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-0.5 * (alpha - 1 / z(x)) ** 2)

        #     ## Generatred moments function (not - centered)
        #     E_1 = scipy.integrate.quad(lambda x: x ** 1 * pdf(x), 0, 10 * measurements.max)[0]
        #     # E_2 = scipy.integrate.quad(lambda x: x ** 2 * pdf(x), 0, 10 * measurements.max)[0]
        #     # E_3 = scipy.integrate.quad(lambda x: x ** 3 * pdf(x), 0, 10 * measurements.max)[0]
        #     # E_4 = scipy.integrate.quad(lambda x: x ** 4 * pdf(x), 0, 100)[0]

        #     ## Parametric expected expressions
        #     parametric_mean = E_1
        #     # parametric_variance = E_2 - E_1 ** 2
        #     # parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1 ** 3) / ((E_2 - E_1 ** 2)) ** 1.5
        #     # parametric_kurtosis = (E_4-4 * E_1 * E_3 + 6 * E_1 ** 2 * E_2 - 3 * E_1 ** 4) /  ((E_2 - E_1 ** 2)) ** 2
        #     parametric_median = loc + scale / (alpha - scipy.stats.norm.ppf(0.5 * scipy.stats.norm.cdf(alpha)))
        #     parametric_mode = loc + scale * (numpy.sqrt(alpha * alpha + 8) - alpha) / 4

        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     # eq2 = parametric_variance - measurements.variance
        #     # eq2 = parametric_skewness - measurements.skewness
        #     # eq4 = parametric_kurtosis  - measurements.kurtosis
        #     eq2 = parametric_median - measurements.median
        #     eq3 = parametric_mode - measurements.mode

        #     return (eq1, eq2, eq3)

        ## THIS METHOD IS CORRECT, BUT IS VERY SLOW BECAUSE THE INTEGRATION
        # bnds = ((0, 9, 0), (numpy.inf, measurements.mean, numpy.inf))
        # x0 = (1, measurements.mean, 1)
        # args = ([measurements])
        # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        # parameters = {"alpha": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}

        # solution = scipy.optimize.fsolve(equations, (1, 1, 1), measurements)
        # parameters = {"alpha": solution[0], "loc": solution[1], "scale": solution[2]}

        scipy_params = scipy.stats.alpha.fit(measurements.data)
        parameters = {"alpha": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
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
    path = "../data/data_alpha.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = ALPHA(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.cdf(numpy.array([measurements.mean, measurements.mean])))
    print(distribution.pdf(measurements.mean))
    print(distribution.pdf(numpy.array([measurements.mean, measurements.mean])))

    # alpha, loc, scale = 6, 10, 5
    # z = lambda t: (t - loc) / scale
    # pdf = lambda x: (1 / (scale * z(x) * z(x) * scipy.stats.norm.cdf(alpha) * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-0.5 * (alpha - 1 / z(x)) ** 2)

    # E_1 = scipy.integrate.quad(lambda x: x ** 1 * pdf(x), 0, 10 * measurements.max)[0]
    # E_2 = scipy.integrate.quad(lambda x: x ** 2 * pdf(x), 0, 10 * measurements.max)[0]
    # E_3 = scipy.integrate.quad(lambda x: x ** 3 * pdf(x), 0, 10 * measurements.max)[0]
    # E_4 = scipy.integrate.quad(lambda x: x ** 4 * pdf(x), 0, 10 * measurements.max)[0]

    # parametric_mean = E_1
    # parametric_variance = E_2 - E_1 ** 2
    # parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1 ** 3) / ((E_2 - E_1 ** 2)) ** 1.5
    # parametric_kurtosis = (E_4-4 * E_1 * E_3 + 6 * E_1 ** 2 * E_2 - 3 * E_1 ** 4) /  ((E_2 - E_1 ** 2)) ** 2
    # parametric_median = loc + scale / (alpha - scipy.stats.norm.ppf(0.5 * scipy.stats.norm.cdf(alpha)))
    # parametric_mode = loc + scale * (numpy.sqrt(alpha * alpha + 8) - alpha) / 4

    # print(parametric_mean, parametric_variance, parametric_skewness, parametric_kurtosis, parametric_median, parametric_mode)
