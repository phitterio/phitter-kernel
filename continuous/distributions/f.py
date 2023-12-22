import scipy.stats
import scipy.special as sc


class F:
    """
    F distribution
    https://en.wikipedia.org/wiki/F-distribution
    http://atomic.phys.uni-sofia.bg/local/nist-e-handbook/e-handbook/eda/section3/eda366a.htm
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df1 = self.parameters["df1"]
        self.df2 = self.parameters["df2"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        ## print(scipy.stats.f.cdf(x, self.df1, self.df2))
        return sc.betainc(self.df1 / 2, self.df2 / 2, x * self.df1 / (self.df1 * x + self.df2))

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        ## print(scipy.stats.f.pdf(x, self.df1, self.df2))
        return (1 / sc.beta(self.df1 / 2, self.df2 / 2)) * ((self.df1 / self.df2) ** (self.df1 / 2)) * (x ** (self.df1 / 2 - 1)) * ((1 + x * self.df1 / self.df2) ** (-1 * (self.df1 + self.df2) / 2))

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.df1 > 0
        v2 = self.df2 > 0
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
            {"df1": * , "df2": * }
        """
        ## Scipy parameters of distribution
        scipy_params = scipy.stats.f.fit(measurements.data)

        ## Results
        parameters = {"df1": scipy_params[0], "df2": scipy_params[1]}

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
    path = "../data/data_f.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = F(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
