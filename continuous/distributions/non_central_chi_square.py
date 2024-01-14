import scipy.integrate
import numpy
import scipy.stats
import scipy.special
import numpy


class NON_CENTRAL_CHI_SQUARE:
    """
    Non-Central Chi Square distribution
    https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution
    Hand - book on Statistical Distributions (pag.110) ... Christian Walck
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
        self.n = self.parameters["n"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """

        # def Q(M: float, a: float, b: float) -> float:
        #     """
        #     Marcum Q - function
        #     https://en.wikipedia.org/wiki/Marcum_Q-function
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

        result = scipy.stats.ncx2.cdf(x, self.lambda_, self.n)
        # result = scipy.special.chndtr(x, self.lambda_, self.n)
        # result = 1 - Q(self.n / 2, numpy.sqrt(self.lambda_), numpy.sqrt(x))
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        ## Method 1
        result = scipy.stats.ncx2.pdf(x, self.lambda_, self.n)

        ## Method 2
        # result = 1 / 2 * numpy.exp(-(x + self.lambda_) / 2) * (x / self.lambda_) ** ((self.n - 2) / 4) * scipy.special.iv((self.n - 2) / 2, numpy.sqrt(self.lambda_ * x))
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
        v1 = self.lambda_ > 0
        v2 = self.n > 0
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
            {"df": * }
        """
        # def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
        #     lambda_, n = initial_solution

        #     ## Parametric expected expressions
        #     parametric_mean = lambda_ + n
        #     parametric_variance = 2 * (2 * lambda_ + n)
        #     # parametric_skewness = 8 * (3 * lambda_ + n) / ((2 * (2 * lambda_ + n)) ** 1.5)
        #     # parametric_kurtosis = 12 * (4 * lambda_ + n) / ((2 * lambda_ + n) ** 2)

        #     ## System Equations
        #     eq1 = parametric_mean - measurements.mean
        #     eq2 = parametric_variance - measurements.variance
        #     # eq3 = parametric_skewness - measurements.skewness
        #     # eq4 = parametric_kurtosis  - measurements.kurtosis

        #     return (eq1, eq2)

        # bnds = ((0, 0), (numpy.inf, numpy.inf))
        # x0 = (measurements.mean, 1)
        # args = ([measurements])
        # solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        # parameters = {"lambda": solution.x[0], "n": round(solution.x[1])}

        lambda_ = measurements.variance / 2 - measurements.mean
        n = 2 * measurements.mean - measurements.variance / 2
        parameters = {"lambda": lambda_, "n": n}
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
    path = "../data/data_NON_CENTRAL_CHI_SQUARE.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = NON_CENTRAL_CHI_SQUARE(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.cdf(numpy.array([measurements.mean, measurements.mean])))
    print(distribution.pdf(measurements.mean))
    print(distribution.pdf(numpy.array([measurements.mean, measurements.mean])))
    print(distribution.pdf(60))
    # print(scipy.stats.ncx2.fit(data))
