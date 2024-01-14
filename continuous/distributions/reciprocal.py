import numpy


class RECIPROCAL:
    """
    Reciprocal distribution
    https://en.wikipedia.org/wiki/Reciprocal_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        return (numpy.log(x) - numpy.log(self.a)) / (numpy.log(self.b) - numpy.log(self.a))

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return 1 / (x * (numpy.log(self.b) - numpy.log(self.a)))

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.b > self.a
        return v1

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
            {"min": * , "max": * }
        """

        a = measurements.min - 1e-8
        b = measurements.max + 1e-8

        parameters = {"a": a, "b": b}

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
    path = "../data/data_reciprocal.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = RECIPROCAL(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.cdf(numpy.array([measurements.mean, measurements.mean])))
    print(distribution.pdf(measurements.mean))
    print(distribution.pdf(numpy.array([measurements.mean, measurements.mean])))
