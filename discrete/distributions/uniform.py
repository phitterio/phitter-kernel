class UNIFORM:
    """
    Uniform distribution
    https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    def cdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using the definition of the function
        Alternative: scipy cdf method
        """
        return (x - self.a + 1) / (self.b - self.a + 1)

    def pmf(self, x: int) -> float:
        """
        Probability density function
        Calculated using the definition of the function
        """
        return 1 / (self.b - self.a + 1)

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
        v2 = type(self.b) == int
        v3 = type(self.a) == int
        return v1 and v2 and v3

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
            {"a": * , "b": * }
        """
        a = round(measurements.min)
        b = round(measurements.max)
        parameters = {"a": a, "b": b}
        return parameters


if __name__ == "__main__":
    ## Import function to get measurements
    import sys

    sys.path.append("../measurements")
    from measurements_discrete import MEASUREMENTS_DISCRETE

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    path = "../data/data_uniform.txt"

    ## Distribution class
    data = get_data(path)
    measurements = MEASUREMENTS_DISCRETE(data)
    distribution = UNIFORM(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pmf(measurements.mean))
