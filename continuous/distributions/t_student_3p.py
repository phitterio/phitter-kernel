import scipy.stats
import scipy.special as sc
import math


class T_STUDENT_3P:
    """
    T distribution
    https://en.wikipedia.org/wiki/Student%27s_t - distribution
    Hand - book on  STATISTICAL  DISTRIBUTIONS  for  experimentalists (pag.143) ...  Christian Walck
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df = self.parameters["df"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.loc) / self.scale
        # result = scipy.stats.t.cdf(z(x), self.df)
        result = sc.betainc(self.df / 2, self.df / 2, (z(x) + math.sqrt(z(x) ** 2 + self.df)) / (2 * math.sqrt(z(x) ** 2 + self.df)))
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.loc) / self.scale
        # result = scipy.stats.t.pdf(z(x), self.df)
        result = (1 / (math.sqrt(self.df) * sc.beta(0.5, self.df / 2))) * (1 + z(x) * z(x) / self.df) ** (-(self.df + 1) / 2)
        return result

    def ppf(self, u):
        # result = scipy.stats.t.ppf(u, self.df)
        if u >= 0.5:
            result = self.loc + self.scale * math.sqrt(self.df * (1 - sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
            return result
        else:
            result = self.loc - self.scale * math.sqrt(self.df * (1 - sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
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
        v1 = self.df > 0
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
            {"alpha":  * , "beta":  * , "min":  * , "max":  * }
        """

        scipy_params = scipy.stats.t.fit(measurements.data)
        parameters = {"df": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}

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
    path = "../data/data_t_student_3p.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = T_STUDENT_3P(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
