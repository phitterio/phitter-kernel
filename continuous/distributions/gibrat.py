import numpy
import scipy.stats
import scipy.special


class GIBRAT:
    """
    Gibrat distribution
    https://mathworld.wolfram.com/GibratsDistribution.html
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # z = lambda t: (t - self.loc) / self.scale
        # result = 0.5 * (1 + scipy.special.erf(numpy.log(z(x)) / numpy.sqrt(2)))
        result = scipy.stats.gibrat.cdf(x, self.loc, self.scale)
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # z = lambda t: (t - self.loc) / self.scale
        # result = 1 / (self.scale * z(x) * numpy.sqrt(2 * numpy.pi)) * numpy.exp(-0.5 * numpy.log(z(x)) ** 2)
        result = scipy.stats.gibrat.pdf(x, self.loc, self.scale)
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
        v1 = self.scale > 0
        return v1

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
            {"loc": * , "scale": *}
        """
        # loc = measurements.min - 1e-3
        # scale = (measurements.mean - loc) / numpy.sqrt(numpy.e)
        scipy_params = scipy.stats.gibrat.fit(measurements.data)
        parameters = {"loc": scipy_params[0], "scale": scipy_params[1]}
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
    path = "../data/data_gibrat.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = GIBRAT(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.cdf(numpy.array([measurements.mean, measurements.mean])))
    print(distribution.pdf(measurements.mean))
    print(distribution.pdf(numpy.array([measurements.mean, measurements.mean])))
