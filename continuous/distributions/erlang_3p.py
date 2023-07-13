import math
import scipy.special as sc

class ERLANG_3P:
    """
    Erlang 3p distribution  
    https://en.wikipedia.org/wiki/Erlang_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.m = self.parameters["m"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
        
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        result = sc.gammainc(self.m, (x - self.loc) / self.beta)
        return result
    
    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        result = ((self.beta **  - self.m) * ((x - self.loc) ** (self.m - 1)) * math.e ** (-((x - self.loc) / self.beta))) / math.factorial(self.m - 1)
        return result
    
    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)
    
    def parameter_restrictions(self) -> bool:
        """
        Check parameters restriction
        """
        v1 = self.m > 0
        v2 = self.beta > 0
        v3 = type(self.m) == int
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
            {"m":  * , "beta":  * }
        """        
        m = round((2 / measurements.skewness) ** 2)
        β = math.sqrt(measurements.variance / ((2 / measurements.skewness) ** 2))
        loc = measurements.mean - ((2 / measurements.skewness) ** 2) * β
        parameters = {"m": m, "beta": β, "loc": loc}
        return parameters


if __name__ == "__main__":
    ## Import function to get measurements
    import sys
    sys.path.append("../measurements")
    from measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "../data/data_erlang_3p.txt"
    data = get_data(path) 
    measurements = MEASUREMENTS(data)
    distribution = ERLANG_3P(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
