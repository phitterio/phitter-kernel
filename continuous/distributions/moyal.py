import scipy.special
import scipy.stats
import numpy
import scipy.optimize
import numpy


class MOYAL:
    """
    Moyal distribution
    Hand - book on Statistical Distributions (pag.93) ... Christian Walck
    https://reference.wolfram.com / language / ref / MoyalDistribution.html
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.mu) / self.sigma
        # result = result = scipy.stats.moyal.cdf(x, loc = self.mu, scale = self.sigma)
        # result = 1 - scipy.special.gammainc(0.5, numpy.exp(-z(x)) / 2)
        result = scipy.special.erfc(numpy.exp(-0.5 * z(x)) / numpy.sqrt(2))
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.mu) / self.sigma
        # result = scipy.stats.moyal.pdf(x, loc = self.mu, scale = self.sigma)
        result = numpy.exp(-0.5 * (z(x) + numpy.exp(-z(x)))) / (self.sigma * numpy.sqrt(2 * numpy.pi))
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
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by formula.

        Parameters
        ==========
        measurements : dict
            {"mu": * , "variance": * , "skewness": * , "kurtosis": * , "data": * }

        Returns
        =======
        parameters : dict
            {"mu": * , "sigma": * }
        """
        # def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
        #     ## Variables declaration
        #     mu, sigma = initial_solution

        #     ## Parametric expected expressions
        #     parametric_mean = mu + sigma * (numpy.log(2) + 0.577215664901532)
        #     parametric_variance = sigma * sigma * numpy.pi * numpy.pi / 2

        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     eq2 = parametric_variance - measurements.variance

        #     return (eq1, eq2)

        # bnds = ((-numpy.inf, 0), (numpy.inf, numpy.inf))
        # x0 = (measurements.mean, 1)
        # args = ([measurements])
        # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)

        sigma = numpy.sqrt(2 * measurements.variance / (numpy.pi * numpy.pi))
        mu = measurements.mean - sigma * (numpy.log(2) + 0.577215664901532)

        parameters = {"mu": mu, "sigma": sigma}
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
    path = "../data/data_moyal.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = MOYAL(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.cdf(numpy.array([measurements.mean, measurements.mean])))
    print(distribution.pdf(measurements.mean))
    print(distribution.pdf(numpy.array([measurements.mean, measurements.mean])))
