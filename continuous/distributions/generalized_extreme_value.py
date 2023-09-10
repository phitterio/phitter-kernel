import scipy.stats
import math

class GENERALIZED_EXTREME_VALUE:
    """
    Generalized Extreme Value Distribution
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        
        self.ξ = self.parameters["ξ"]
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.miu) / self.sigma
        if self.ξ == 0:
            return math.exp(-math.exp(-z(x)))
        else:
            return math.exp(-(1 + self.ξ * z(x)) ** (-1 / self.ξ))
        # return scipy.stats.genextreme.cdf(x,  - self.ξ, loc=self.miu, scale=self.sigma)
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # print(scipy.stats.genextreme.pdf(x,  - self.ξ, loc=self.miu, scale=self.sigma))
        z = lambda t: (t - self.miu) / self.sigma
        if self.ξ == 0:
            return (1 / self.sigma) * math.exp(-z(x) - math.exp(-z(x)))
        else:
            return (1 / self.sigma) * math.exp(-(1 + self.ξ * z(x)) ** (-1 / self.ξ)) * (1 + self.ξ * z(x)) ** (-1 - 1 / self.ξ)
       
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
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"ξ":  * , "miu":  * , "sigma":  * }
        """
        scipy_params = scipy.stats.genextreme.fit(measurements.data)
        parameters = {"ξ":  - scipy_params[0], "miu": scipy_params[1], "sigma": scipy_params[2]}
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
    path = "../data/data_generalized_extreme_value.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = GENERALIZED_EXTREME_VALUE(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))